import numpy as np
from lietorch import SE3
import os
import pycocotools.mask as mask_util
import pandas as pd

# from https://github.com/crockwell/rel_pose
# rot: w first
def eval_camera(pred_poses, gt_poses, pair_names, acc_threshold, using_sub = False, save_path = None):
    """_summary_

    Args:
        pred_poses (_type_): SE3(torch.size([n,2,7])) position+x,y,z,w
        gt_poses (_type_): SE3(torch.size([n,2,7])) position+x,y,z,w
        if_sub (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
     # threshold for translation and rotation error to say prediction is correct.

    if using_sub: 
        sub_pred_pose = pred_poses[:,1]*pred_poses[:,0].inv() 
        sub_pred_pose = sub_pred_pose.data.cpu().numpy() # [n,7]?
        pred_tran = sub_pred_pose[:,:3] # [n,3]
        pred_rot = sub_pred_pose[:,3:] # [n,4]

        sub_gt_pose = gt_poses[:,1]*gt_poses[:,0].inv()
        sub_gt_pose = sub_gt_pose.data.cpu().numpy() 
        gt_tran = sub_gt_pose[:,:3]
        gt_rot = sub_gt_pose[:,3:]
    else:
        pred_poses = pred_poses.data.cpu().numpy()
        pred_tran = pred_poses[:,1,:3]
        pred_rot = pred_poses[:,1,3:] #[n,4]
        
        gt_poses = gt_poses.data.cpu().numpy()
        gt_tran = gt_poses[:,1,:3]
        gt_rot = gt_poses[:,1,3:]

    # w last -> w fist, x,y,z,w -> w,x,y,z
    pred_rot = pred_rot[:, [3,0,1,2]]
    gt_rot = gt_rot[:, [3,0,1,2]]
    
    top1_error = {
        "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
        "rot": 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1)), -1.0, 1.0)) * 180 / np.pi,
    }

    top1_accuracy = {
        "tran": [
            (top1_error["tran"] <= i).sum()
        / len(top1_error["tran"]) for i in acc_threshold["tran"]
        ],
        "rot": [
            (top1_error["rot"] <= i).sum()
        / len(top1_error["rot"]) for i in acc_threshold["rot"]
        ],
    }
    camera_metrics = {
        f"top1 T err < {acc_threshold['tran'][0]}": top1_accuracy["tran"][0] * 100,
        f"top1 R err < {acc_threshold['rot'][0]}": top1_accuracy["rot"][0] * 100,
        f"top1 T err < {acc_threshold['tran'][1]}": top1_accuracy["tran"][1] * 100,
        f"top1 R err < {acc_threshold['rot'][1]}": top1_accuracy["rot"][1] * 100,
        f"top1 T err < {acc_threshold['tran'][2]}": top1_accuracy["tran"][2] * 100,
        f"top1 R err < {acc_threshold['rot'][2]}": top1_accuracy["rot"][2] * 100,
        f"T mean err": np.mean(top1_error["tran"]),
        f"R mean err": np.mean(top1_error["rot"]),
        f"T median err": np.median(top1_error["tran"]),
        f"R median err": np.median(top1_error["rot"]),
    }
    
    if save_path != None:
        gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(np.abs(gt_rot[:,0])) * 180 / np.pi} # add np.abs -> [-180, 180]

        tran_graph = np.stack([gt_mags['tran'], top1_error['tran']],axis=1)
        tran_graph_df = pd.DataFrame(tran_graph, index = pair_names)
        tran_graph_name = os.path.join(save_path, 'gt_translation_magnitude_vs_error.csv')
        tran_graph_df = tran_graph_df.applymap(lambda x:('%1.5f')%x)
        tran_graph_df.to_csv(tran_graph_name)
        # np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

        rot_graph = np.stack([gt_mags['rot'], top1_error['rot']],axis=1)
        rot_graph_df = pd.DataFrame(rot_graph, index = pair_names)
        rot_graph_name = os.path.join(save_path, 'gt_rotation_magnitude_vs_error.csv')
        rot_graph_df = rot_graph_df.applymap(lambda x:('%1.5f')%x)
        rot_graph_df.to_csv(rot_graph_name)
        # np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
        
    return camera_metrics

def compute_IPAA(pred_assignment_m, gt_assignment_list, IPAA_dict):
    wrong_count = 0
    gt_assignment_list = np.array(gt_assignment_list)
    if len(gt_assignment_list) != 0:
        common_row_idxs = gt_assignment_list[:, 0]
        common_colomn_idxs = gt_assignment_list[:, 1]
    else:
        common_row_idxs = []
        common_colomn_idxs = []
    if len(gt_assignment_list) != 0:
        for [row, column] in gt_assignment_list:
            if pred_assignment_m[row, column] != 1:
                wrong_count += 1
    for i in range(pred_assignment_m.shape[0]):
        if i not in common_row_idxs:
            if sum(pred_assignment_m[i, :]) != 0:
                wrong_count += 1
    for j in range(pred_assignment_m.shape[1]):
        if j not in common_colomn_idxs:
            if sum(pred_assignment_m[:, j]) != 0:
                wrong_count += 1
    p = float(wrong_count) / (
        pred_assignment_m.shape[0]
        + pred_assignment_m.shape[1]
        - len(gt_assignment_list)
    )
    for key in IPAA_dict.keys():
        if (1 - p) * 100 >= key:
            IPAA_dict[key] += 1





def evaluate_matching_from_iou(
    match_statistics,
    pred_segmentations,  #{"0":[480,640], "1":[480, 640]
    gt_seg_masks, #{"0":[n1,480,640], "1":[n2,480,640]}
    pred_assignment_matrixs,  # [2+2, n1, n2]
    gt_corr, # [num_corr,2]
    pred_plane_num, # {"0":, "1":}
    iou_thresh=0.5
):
    # match_statistics = {}
    key_list = ["attn_corr0", "attn_corr1", "final_corr0", "final_corr1", "attn_combine_corr", "final_combine_corr"]
    # for i in range(pred_assignment_matrixs.shape[0]):
    #     # if "assignment" in key:
    #     key = key_list[i]
    #     match_statistics[key] = {
    #         "all_correct_num": 0,
    #         "all_matched_num": 0
    #     }
    key_list = key_list[:pred_assignment_matrixs.shape[0]]

    # all_gt_num = 0

    # all_gt_num += len(gt_corr)

    matched_iou_list = []
    matched_gtidx_list = []
    for img_idx in ['0', '1']:
        # get gt masks
        gt_mask_rles = []
        for gt_seg_mask in gt_seg_masks[str(img_idx)]:
            # image_height = ann['height']
            # image_width = ann['width']
            # if isinstance(ann["segmentation"], list):
            #     polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
            #     try:
            #         rles = mask_util.frPyObjects(polygons, image_height, image_width)
            #     except:
            #         polygons = []
            #         for p in ann["segmentation"]:
            #             if len(p) > 4:
            #                 polygons.append(np.array(p, dtype=np.float64))
            #             else:
            #                 raise ValueError
            #     rle = mask_util.merge(rles)
            # elif isinstance(ann["segmentation"], dict):  # RLE
            #     rle = ann["segmentation"]
            # else:
            #     raise TypeError(
            #         f"Unknown segmentation type {type(ann['segmentation'])}!"
            #     )
            rle = mask_util.encode(np.asfortranarray(gt_seg_mask))
            gt_mask_rles.append(rle)

        # get predicted masks
        pred_mask_rles = []
        for pred_i in range(pred_plane_num[str(img_idx)]):
            rle = mask_util.encode(np.asfortranarray(pred_segmentations[str(img_idx)] == pred_i))
            pred_mask_rles.append(rle)

        

        # get mask ious
        miou = mask_util.iou(pred_mask_rles, gt_mask_rles, [0] * len(gt_mask_rles))  # shape: [n_pred, n_gt]
        # m_iou_tensor = torch.from_numpy(miou)  # shape: [n_pred, n_gt]

        # assign pred mask to gt mask
        # matched_iou, matched_gtidx = m_iou_tensor.max(-1)  # idx of gt plane which has max iou with predicted plane
        matched_iou = miou.max(-1)  # pred_num1/2
        matched_gtidx = miou.argmax(-1)  # pred_num1/2
        matched_iou_list.append(matched_iou)
        matched_gtidx_list.append(matched_gtidx)

    # for key in pred:
    #     if "assignment" in key:
    # pred_corr_matrix = pred[key]  # pred_n, pred_n
    # pred_corr = torch.nonzero(pred_corr_matrix).reshape(-1, 2).detach().cpu().numpy()  # n, 2
    for pred_assignment_m, key in zip(pred_assignment_matrixs, key_list):
        pred_corr = np.stack(np.nonzero(pred_assignment_m), axis = -1) # pred_num_corr,2
        pred_matched_num = pred_corr.shape[0]
        correct_num = 0
        for i in range(pred_matched_num):
            m_idxs = pred_corr[i]
            pred_idx0 = m_idxs[0]
            pred_idx1 = m_idxs[1]
            # if the iou of plane mask of pred_idx0，pred_idx1 between the gt mask > TH
            if matched_iou_list[0][pred_idx0] >= iou_thresh and matched_iou_list[1][pred_idx1] >= iou_thresh:
                gt_idx0 = matched_gtidx_list[0][pred_idx0]
                gt_idx1 = matched_gtidx_list[1][pred_idx1]
                #! if [gt_idx0, gt_idx1] in gt_corr:
                if [gt_idx0, gt_idx1] in gt_corr.tolist():
                    correct_num += 1

        match_statistics[key]["all_matched_num"] += pred_matched_num
        match_statistics[key]["all_correct_num"] += correct_num
    # return matching_metrics