import argparse
import numpy as np
import time
import torch
from torch.nn import functional as F
import os
import cv2
import pickle
import quaternion
import math
from tqdm import tqdm
from scipy.linalg import eigh
import multiprocessing
from multiprocessing import Pool, Process, Queue
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
# from NopeSAC_Net.modeling.meta_arch.camera_branch import Camera_Branch
# from NopeSAC_Net.utils.mesh_utils import get_plane_params_in_global
# from NopeSAC_Net.visualization import create_instances
# from NopeSAC_Net.config import get_sparseplane_cfg_defaults
from PlaneRecTR import add_PlaneRecTR_config
from detectron2.projects.deeplab import add_deeplab_config
# from NopeSAC_Net.data import PlaneRCNNMapper
# from tools.sparseplane_planeloss import GeoConsistencyLoss
from collections import defaultdict
from scipy.special import softmax
import seaborn as sns
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='3'


# EP_mask_delta_thresh =      [0.5,   0.5,        0.5,        0.,       0.5]
# EP_normal_delta_thresh =    [30.,   30.,        1000.,      30.,      1000.]
# EP_offset_delta_thresh =    [1.,    1000.,      1.,         1.,       1000.]

# EP_mask_delta_thresh =      [0.5,   0.5,        0.5,        0.,       0.5]
# EP_normal_delta_thresh =    [15.,   15.,        1000.,      15.,      1000.]
# EP_offset_delta_thresh =    [0.5,    1000.,      0.5,       0.5,       1000.]

EP_mask_delta_thresh =      [0.5,   0.5,        0.5,        0.,       0.5]
EP_normal_delta_thresh =    [5.,    5.,        1000.,       5.,      1000.]
EP_offset_delta_thresh =    [0.2,    1000.,      0.2,       0.2,       1000.]

EP_ap_str =                 ['all', '-offset',  '-normal',  '-mask', '-normal-offset']



def quaternion2rotmatrix(quan):
    assert isinstance(quan, torch.Tensor)
    assert quan.shape[-1] == 4
    if quan.dim() == 1:
        quan = quan.unsqueeze(0)  # 1, 4
    elif quan.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError

    bs = quan.shape[0]
    rot_matrix = torch.zeros([bs, 3, 3]).to(device=quan.device, dtype=quan.dtype)  # bs, 3, 3

    w = quan[:, 0]  # bs
    x = quan[:, 1]
    y = quan[:, 2]
    z = quan[:, 3]

    m1 = 1 - 2 * y * y - 2 * z * z
    m2 = 2 * x * y - 2 * w * z
    m3 = 2 * x * z + 2 * w * y
    m4 = 2 * x * y + 2 * w * z
    m5 = 1 - 2 * x * x - 2 * z * z
    m6 = 2 * y * z - 2 * w * x
    m7 = 2 * x * z - 2 * w * y
    m8 = 2 * y * z + 2 * w * x
    m9 = 1 - 2 * x * x - 2 * y * y

    rot_matrix[:, 0, 0] = m1
    rot_matrix[:, 0, 1] = m2
    rot_matrix[:, 0, 2] = m3

    rot_matrix[:, 1, 0] = m4
    rot_matrix[:, 1, 1] = m5
    rot_matrix[:, 1, 2] = m6

    rot_matrix[:, 2, 0] = m7
    rot_matrix[:, 2, 1] = m8
    rot_matrix[:, 2, 2] = m9

    rot_matrix = rot_matrix.contiguous()  # bs, 3, 3

    return rot_matrix

def warp_single_view_plane_param_to_global(plane, rot_quan=None, tran=None, pose_n=None):
    """
    plane: bs, plane_n, 3
    rot_quan: bs, pose_n, 4
    tran: bs, pose_n, 3
    """
    if rot_quan is not None and tran is not None:
        bs, pose_n, _ = rot_quan.shape
        plane_n = plane.shape[1]

        # get normal and offset
        plane0 = plane.unsqueeze(1).repeat(1, pose_n, 1, 1).view(bs*pose_n, plane_n, 3)  # bs*pose_n, plane_n, 3

        # get rot matrix
        rot_quan = rot_quan.view(-1, 4)  # bs*pose_n, 4
        rot_matrix = quaternion2rotmatrix(rot_quan)  # bs*pose_n, 3, 3

        tran = tran.unsqueeze(2).repeat(1, 1, plane_n, 1).view(bs*pose_n, plane_n, 3)  # bs*pose_n, plane_n, 3

        # convert plane of the first view
        start = tran  # bs*pose_n, plane_n, 3
        end = plane0 * (torch.tensor([1, -1, -1]).reshape(1, 1, 3).to(tran.device))  # suncg2habitat # bs*pose_n, plane_n, 3
        end = end.permute(0, 2, 1)  # bs*pose_n, 3, plane_n
        end = (torch.bmm(rot_matrix, end)).permute(0, 2, 1) + tran  # cam2world  # bs*pose_n, plane_n, 3
        a = end  # bs*pose_n, plane_n, 3
        b = end - start  # bs*pose_n, plane_n, 3
        plane0 = ((a * b).sum(dim=-1) / (torch.norm(b, dim=-1) + 1e-5) ** 2).view(bs*pose_n, plane_n, 1) * b  # bs*pose_n, plane_n, 3
        plane0 = plane0.reshape(bs, pose_n, plane_n, 3).contiguous()

        return plane0
    else:
        assert pose_n is not None
        bs = plane.shape[0]
        plane_n = plane.shape[1]
        # convert plane of the second view
        plane1 = plane.unsqueeze(1).repeat(1, pose_n, 1, 1).view(bs*pose_n, plane_n, 3)  # bs*pose_n, plane_n, 3
        plane1 = plane1 * (torch.tensor([1, -1, -1]).reshape(1, 1, 3).to(plane1.device))  # bs*n, n, 3
        plane1 = plane1.reshape(bs, pose_n, plane_n, 3).contiguous()

        return plane1



def update_assignment_matrixs(assignment_matrix, 
                              planeParam1, 
                              planeParam2, 
                              ref_rot_soft,
                              ref_trans_soft,
                              normal_th=45.,
                              offset_th=1.,
                              ):
    # !planeparam: n*d

    # calculate warped plane parameters  #! *(1,-1,-1)
    parameters2_warped = warp_single_view_plane_param_to_global(
        planeParam2, pose_n=1)[:, 0, :, :]  # bs, n2, 3
    offset2_warped = torch.norm(parameters2_warped, dim=2, keepdim=True, p=2)  # bs, n2, 1
    normal2_warped = F.normalize(parameters2_warped, dim=-1, p=2)  # bs, n2, 3


    parameters1_warped_r = warp_single_view_plane_param_to_global(
            planeParam1, ref_rot_soft.unsqueeze(1), ref_trans_soft.unsqueeze(1) * 0.)[:, 0, :, :]  # bs, n1, 3
    normal1_warped_r = F.normalize(parameters1_warped_r, dim=-1, p=2)  # bs, n1, 3
    nTn_r = torch.bmm(normal1_warped_r, normal2_warped.transpose(1, 2))  # bs, n1, n2
    normal_dist = torch.acos(torch.clamp(nTn_r, -1, 1))  # in rad
    normal_dist = normal_dist / np.pi * 180.  # b, n1, n2
    # calculate offset dist
    parameters1_warped_rt = warp_single_view_plane_param_to_global(
        planeParam1, ref_rot_soft.unsqueeze(1), ref_trans_soft.unsqueeze(1))[:, 0, :, :]  # bs, n1, 3
    offset1_warped_rt = torch.norm(parameters1_warped_rt, dim=2, keepdim=True, p=2)  # bs, n1, 1
    normal1_warped_rt = F.normalize(parameters1_warped_rt, dim=-1, p=2)  # bs, n1, 3
    nTn_rt = torch.bmm(normal1_warped_rt, normal2_warped.transpose(1, 2))  # bs, n1, n2
    offset_dist = torch.abs(offset1_warped_rt - offset2_warped.transpose(1, 2))  # b, n1, n2
    offset_dist[nTn_rt < 0] = torch.abs(offset1_warped_rt + offset2_warped.transpose(1, 2))[nTn_rt < 0]
    offset_dist = torch.clamp(offset_dist, min=1e-4, max=10)  # b, n1, n2
    # calculate mask
    normal_dist_mask = normal_dist < normal_th
    offset_dist_mask = offset_dist < offset_th
    Ass_mask = normal_dist_mask & offset_dist_mask
    new_assignment_matrix = assignment_matrix * Ass_mask.float()
    return new_assignment_matrix

    # assignment_matrix_list.append(assignment_matrix)
    # output_planeAss["pred_assignment_afterRef0"] = assignment_matrix.clone()
    # output_planeAss["pred_assignment"] = assignment_matrix.clone()



def get_plane_params_in_global(planes, camera_info):
    """
    input:
    @planes: plane params
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    plane parameters in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    start = np.ones((len(planes), 3)) * tran
    end = planes * np.array([1, -1, -1])  # suncg2habitat
    end = (quaternion.as_rotation_matrix(rot) @ (end).T).T + tran  # cam2world
    a = end
    b = end - start
    planes_world = ((a * b).sum(axis=1) / np.linalg.norm(b, axis=1) ** 2).reshape(-1, 1) * b
    return planes_world

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def assignment_inference(attn_corrs, match_threshold, th_method = 0, label_mask0=None, label_mask1=None):
    assignment_matrix_list = []
    if len(attn_corrs.shape) <= 2:
        attn_corrs = [attn_corrs]
    for a_i, attn_corr in enumerate(attn_corrs):  # torch.Size([8/9, 30, 30]) 
        attn_corr = attn_corr.clone() 
        # if a_i in [4,5,6,7]:
        #     attn_corr = attn_corr.T
        attn_corr = attn_corr.unsqueeze(0)
        if th_method == 0:
            max_sum = min(torch.sum(attn_corr, dim=-2).max(), torch.sum(attn_corr, dim=-1).max())
            curr_match_threshold = max_sum * match_threshold
        elif th_method == 1:
            max_sum = max(torch.sum(attn_corr, dim=-2).max(), torch.sum(attn_corr, dim=-1).max())
            curr_match_threshold = max_sum * match_threshold
        elif th_method == 2:
            curr_match_threshold = match_threshold
        else:
            print("Error: th_method")

        if True:
            # label_mask0 = label_mask[:self.num_queries]
            # label_mask1 = label_mask[self.num_queries:]
            # attn_corrs = attn_corrs[label_mask0][:,label_mask1]  # (nq, nq) -> n1, n2
            
            attn_corr[:, ~label_mask0] = 0.
            attn_corr[:, :, ~label_mask1] = 0.
        
        n1, n2 = attn_corr.shape[-2], attn_corr.shape[-1]
        

        max0, max1 = attn_corr.max(2), attn_corr.max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = attn_corr.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values, zero)
        valid0 = mutual0 & (mscores0 > curr_match_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))

        assignment_matrix = torch.zeros((1, n1+1, n2+1), dtype=torch.float32, device = attn_corr.device)  # n1+1, n2+1
        indices0_ = indices0.clone()
        indices0_[indices0_ == -1] = assignment_matrix.shape[-1] - 1 # 1, n1(nq)
        idxs0 = torch.arange(0, indices0_.shape[1]).to(indices0_.device) # n1(nq)
        assignment_matrix[0, idxs0, indices0_] = 1
        assignment_matrix = assignment_matrix[:, :-1, :-1]  # n1, n2 (nq, nq)
        assignment_matrix_list.append(assignment_matrix)

    assignment_matrixs = torch.cat(assignment_matrix_list)
    return assignment_matrixs

def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_PlaneRecTR_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


class Evaluator:
    def __init__(self, args, dataset="mp3d_test", match_threshold = 0.2, corr_idx = 0, th_method = 0,
                 normal_threshold = 45., offset_threshold = 1.,):
        cfg = setup(args)
        self.score_threshold = 0.1

        self.match_threshold = match_threshold
        self.corr_idx = corr_idx
        self.th_method = th_method

        self.normal_threshold = normal_threshold
        self.offset_threshold = offset_threshold

        #!
        self.root_dir = cfg.DATASETS.ROOT_DIR



        rcnn_cached_file = args.rcnn_cached_file
        with open(rcnn_cached_file, "rb") as f:
            print('loading rcnn cached file from {}'.format(rcnn_cached_file))

            self.rcnn_data = list(pickle.load(f).values())  # dict-> list

            print("rcnn cached file has been loaded")

        self.metadata = MetadataCatalog.get(dataset)
        self.load_input_dataset(dataset)
        self.sanity_check()
        self.update_pred_corr_list = []

    def sanity_check(self):
        for idx, key in enumerate(self.dataset_dict.keys()):
            assert self.rcnnidx2datasetkey(idx) == key

    def rcnnidx2datasetkey(self, idx):
        key0 = self.rcnn_data[idx]["0"]["image_id"]
        key1 = self.rcnn_data[idx]["1"]["image_id"]
        key = key0 + "__" + key1
        return key

    def load_input_dataset(self, dataset):
        dataset_dict = {}
        dataset_list = list(DatasetCatalog.get(dataset))
        for dic in dataset_list:
            key0 = dic["0"]["image_id"]
            key1 = dic["1"]["image_id"]
            key = key0 + "__" + key1
            dataset_dict[key] = dic
        self.dataset_dict = dataset_dict

    def get_gt_affinity(self, idx, rtnformat="matrix", gtbox=True):
        """
        return gt affinity.
        If gtbox is True, return gt affinity for gt boxes;
        else return gt affinity for pred boxes.
        """
        if gtbox:
            key0 = self.rcnn_data[idx]["0"]["image_id"]
            key1 = self.rcnn_data[idx]["1"]["image_id"]
            key = key0 + "__" + key1
            corrlist = np.array(self.dataset_dict[key]["gt_corrs"])
        else:
            corrlist = self.get_gt_affinity_from_pred_box(idx)
        if rtnformat == "list":
            return corrlist
        elif rtnformat == "matrix":
            if gtbox:
                mat = torch.zeros(
                    (
                        len(self.dataset_dict[key]["0"]["annotations"]),
                        len(self.dataset_dict[key]["1"]["annotations"]),
                    )
                )
            else:
                mat = torch.zeros(
                    (
                        len(self.rcnn_data[idx]["0"]["instances"]),
                        len(self.rcnn_data[idx]["1"]["instances"]),
                    )
                )
            for i in corrlist:
                mat[i[0], i[1]] = 1
            return mat
        else:
            raise NotImplementedError


    def evaluate_camera(self, out_path=''):
        tran_errs = []
        rot_errs = []
        for idx in tqdm(range(len(self.rcnn_data))):

            key = self.rcnnidx2datasetkey(idx)
            gt_cam = self.dataset_dict[key]["rel_pose"]
            gt_cam = {
                "position": np.array(gt_cam["position"]),
                "rotation": np.array(gt_cam["rotation"]),
            }

            pred_cam = self.rcnn_data[idx]["pred_pose"]
            pred_cam = {
                    "position": pred_cam["position"],
                    "rotation": pred_cam["rotation"], # w, x, y, z
                }


            # Error - translation
            tran_errs.append(np.linalg.norm(pred_cam["position"] - np.array(gt_cam["position"])))
            # Error - rotation
            if type(pred_cam["rotation"]) != np.ndarray:
                print("Need to convert quaternion to np array")
                raise
            d = np.abs(np.sum(np.multiply(pred_cam["rotation"], np.array(gt_cam["rotation"]))))
            d = np.clip(d, -1, 1)
            rot_errs.append(2 * np.arccos(d) * 180 / np.pi)

        tran_acc = sum(_ < 1 for _ in tran_errs) / len(tran_errs)
        rot_acc = sum(_ < 30 for _ in rot_errs) / len(rot_errs)

        tran_acc2 = sum(_ < 0.5 for _ in tran_errs) / len(tran_errs)
        rot_acc2 = sum(_ < 15 for _ in rot_errs) / len(rot_errs)

        tran_acc3 = sum(_ < 0.2 for _ in tran_errs) / len(tran_errs)
        rot_acc3 = sum(_ < 10 for _ in rot_errs) / len(rot_errs)

        tran_acc4 = sum(_ < 0.1 for _ in tran_errs) / len(tran_errs)
        rot_acc4 = sum(_ < 5 for _ in rot_errs) / len(rot_errs)

        median_tran_err = np.median(np.array(tran_errs))
        mean_tran_err = np.mean(np.array(tran_errs))
        median_rot_err = np.median(np.array(rot_errs))
        mean_rot_err = np.mean(np.array(rot_errs))

        print(
            "Median Error [tran, rot]:            {:.2f}, {:.2f}".format(
                median_tran_err, median_rot_err
            )
        )
        print(
            "Mean Error   [tran, rot]:            {:.2f}, {:.2f}".format(
                mean_tran_err, mean_rot_err)
        )
        # print(
        #     "Accuracy     [tran(1m), rot(30')]:   {:.2f}, {:.2f}".format(
        #         tran_acc * 100, rot_acc * 100)
        # )
        # print(
        #     "Accuracy     [tran(0.5m), rot(15')]: {:.2f}, {:.2f}".format(
        #         tran_acc2 * 100, rot_acc2 * 100)
        # )
        # print(
        #     "Accuracy     [tran(0.2m), rot(10')]: {:.2f}, {:.2f}".format(
        #         tran_acc3 * 100, rot_acc3 * 100)
        # )
        # print(
        #     "Accuracy     [tran(0.1m), rot(5')]:  {:.2f}, {:.2f}".format(
        #         tran_acc4 * 100, rot_acc4 * 100)
        # )
        print(
            "Accuracy     [tran(1m), rot(30')]:   {:.1f}, {:.1f}".format(
                tran_acc * 100, rot_acc * 100)
        )
        print(
            "Accuracy     [tran(0.5m), rot(15')]: {:.1f}, {:.1f}".format(
                tran_acc2 * 100, rot_acc2 * 100)
        )
        print(
            "Accuracy     [tran(0.2m), rot(10')]: {:.1f}, {:.1f}".format(
                tran_acc3 * 100, rot_acc3 * 100)
        )
        print(
            "Accuracy     [tran(0.1m), rot(5')]:  {:.1f}, {:.1f}".format(
                tran_acc4 * 100, rot_acc4 * 100)
        )

        camera_eval_dict = {
            "tran_errs": np.array(tran_errs),
            "rot_errs": np.array(rot_errs),
            "mean_tran_err": mean_tran_err,
            "mean_rot_err": mean_rot_err,
            "median_tran_err": median_tran_err,
            "median_rot_err": median_rot_err,
            "tran_acc": tran_acc,
            "rot_acc": rot_acc,
        }

        # save_histplot(np.array(rot_errs), save_dir=out_path, th=50, less=False)

        return camera_eval_dict

    def evaluate_matching_precision_recall(self, data_num):
        all_matched_num = 0.
        all_gt_num = 0.
        all_correct_num = 0.
        pre = 0.
        recall = 0.

        update_all_matched_num = 0.
        update_all_gt_num = 0.
        update_all_correct_num = 0.

        for idx in tqdm(range(len(self.rcnn_data))):
            key = self.rcnnidx2datasetkey(idx)  # e.g. "2t7WUuJeko7_0_0_26__2t7WUuJeko7_0_0_40"

            label_mask0=torch.from_numpy(self.rcnn_data[idx]["0"]["label_mask"])
            label_mask1=torch.from_numpy(self.rcnn_data[idx]["1"]["label_mask"])
            if "singleview_attn" in self.rcnn_data[idx]:
                singleview_attn = torch.from_numpy(self.rcnn_data[idx]["singleview_attn"])
                singleview_attn_corr = singleview_attn.softmax(dim=-1) * singleview_attn.softmax(dim=-2)
                assignment_matrixs = assignment_inference(singleview_attn_corr,
                                                      self.match_threshold, self.th_method, 
                                                      label_mask0=label_mask0,
                                                      label_mask1=label_mask1
                                                      )
                valid_assignment_matrixs = assignment_matrixs[:,label_mask0][:,:,label_mask1]
            
                pred_corr = np.argwhere(valid_assignment_matrixs.numpy()[self.corr_idx])
            else:
                
                attn_fundamentals = torch.from_numpy(self.rcnn_data[idx]["attn_fundamentals"])
                
                # for Nocombineattn
                # attn_fundamentals = torch.mean(attn_fundamentals, dim=0).unsqueeze(0) # 1,30,30
                                                     
                assignment_matrixs = assignment_inference(attn_fundamentals,
                                                        self.match_threshold, self.th_method, 
                                                        label_mask0=label_mask0,
                                                        label_mask1=label_mask1
                                                        )
                valid_assignment_matrixs = assignment_matrixs[:,label_mask0][:,:,label_mask1]
                
                pred_corr = np.argwhere(valid_assignment_matrixs.numpy()[self.corr_idx])


                valid_params0 = torch.from_numpy(self.rcnn_data[idx]["0"]["valid_params"]).unsqueeze(0) # (1, n1, 3)
                valid_params0 = valid_params0/torch.sum(valid_params0**2, dim=-1).unsqueeze(-1) # # (1, n1, 3)/ (1, n1, 1)
                valid_params1 = torch.from_numpy(self.rcnn_data[idx]["1"]["valid_params"]).unsqueeze(0)
                valid_params1 = valid_params1/torch.sum(valid_params1**2, dim=-1).unsqueeze(-1)
                rot = torch.from_numpy(self.rcnn_data[idx]["pred_pose"]["rotation"].astype(np.float32)).unsqueeze(0)
                trans = torch.from_numpy(self.rcnn_data[idx]["pred_pose"]["position"].astype(np.float32)).unsqueeze(0)
                update_valid_assignment_matrix = update_assignment_matrixs(valid_assignment_matrixs[self.corr_idx],
                                                                        planeParam1 = valid_params0,
                                                                        planeParam2 = valid_params1,
                                                                        ref_rot_soft = rot,
                                                                        ref_trans_soft = trans,
                                                                        normal_th = self.normal_threshold,
                                                                        offset_th = self.offset_threshold,
                                                                        )
                update_pred_corr = np.argwhere(update_valid_assignment_matrix.numpy()[0])

                self.update_pred_corr_list.append(update_pred_corr)
            
            
            
            gt_corr = np.array(self.dataset_dict[key]["gt_corrs"])
            gt_corr = gt_corr.tolist() #!!!
            individual_miou = self.get_maskiou(idx)

            # print(key)

            m_iou0 = individual_miou["0"]
            m_iou1 = individual_miou["1"]

            m_iou0 = torch.from_numpy(m_iou0)
            m_iou1 = torch.from_numpy(m_iou1)

            matched_gtiou0, matched_gtidx0 = m_iou0.max(-1)
            matched_gtiou1, matched_gtidx1 = m_iou1.max(-1)

            pred_matched_num = len(pred_corr)
            correct_num = 0
            for i in range(pred_matched_num):
                m_idxs = pred_corr[i]
                pred_idx0 = m_idxs[0]
                pred_idx1 = m_idxs[1]

                if matched_gtiou0[pred_idx0] >= 0.5 and matched_gtiou1[pred_idx1] >= 0.5:
                    gt_idx0 = matched_gtidx0[pred_idx0]
                    gt_idx1 = matched_gtidx1[pred_idx1]
                    if [gt_idx0, gt_idx1] in gt_corr:
                        correct_num += 1

            all_matched_num += pred_matched_num
            all_correct_num += correct_num
            all_gt_num += len(gt_corr)

            if "singleview_attn" not in self.rcnn_data[idx]:

                #! UPDATE
                update_pred_matched_num = len(update_pred_corr)
                update_correct_num = 0
                for i in range(update_pred_matched_num):
                    update_m_idxs = update_pred_corr[i]
                    update_pred_idx0 = update_m_idxs[0]
                    update_pred_idx1 = update_m_idxs[1]

                    if matched_gtiou0[update_pred_idx0] >= 0.5 and matched_gtiou1[update_pred_idx1] >= 0.5:
                        update_gt_idx0 = matched_gtidx0[update_pred_idx0]
                        update_gt_idx1 = matched_gtidx1[update_pred_idx1]
                        if [update_gt_idx0, update_gt_idx1] in gt_corr:
                            update_correct_num += 1

                update_all_matched_num += update_pred_matched_num
                update_all_correct_num += update_correct_num
                update_all_gt_num += len(gt_corr)

            if pred_matched_num == 0:
                continue
            else:
                pre += float(correct_num) / float(pred_matched_num)

            if len(gt_corr) == 0:
                continue
            else:
                recall += float(correct_num) / float(len(gt_corr))


            # new_iou_thresh = 0.5
            # new_key = self.rcnnidx2datasetkey(idx)
    
            # new_pred_assignment_m = self.rcnn_data[idx]["valid_assignment_matrixs"][self.corr_idx]


            # new_matched_iou_list = []
            # new_matched_gtidx_list = []
            # for img_idx in ['0', '1']:
            #     # get gt masks
            #     new_gt_mask_rles = []             
            #     for new_ann in self.dataset_dict[new_key][img_idx]["annotations"]:
            #         if isinstance(new_ann["segmentation"], list):
            #             new_polygons = [
            #                 np.array(p, dtype=np.float64) for p in new_ann["segmentation"]
            #             ]
            #             new_rles = mask_util.frPyObjects(
            #                 new_polygons,
            #                 self.dataset_dict[new_key][img_idx]["height"],
            #                 self.dataset_dict[new_key][img_idx]["width"],
            #             )
            #             new_rle = mask_util.merge(new_rles)
            #         elif isinstance(new_ann["segmentation"], dict):  # RLE
            #             new_rle = new_ann["segmentation"]
            #         else:
            #             raise TypeError(
            #                 f"Unknown segmentation type {type(new_ann['segmentation'])}!"
            #             )
            #         new_gt_mask_rles.append(new_rle)
            #     new_pred_mask_rles = [mask_util.encode(np.asfortranarray(self.rcnn_data[idx][img_idx]["sem_seg"]==segi))
            #                   for segi in range(len(self.rcnn_data[idx][img_idx]["valid_params"]))]

            #     # get mask ious
            #     new_miou = mask_util.iou(new_pred_mask_rles, new_gt_mask_rles, [0] * len(new_gt_mask_rles))  # shape: [n_pred, n_gt]
            #     # m_iou_tensor = torch.from_numpy(miou)  # shape: [n_pred, n_gt]

            #     # assign pred mask to gt mask
            #     # matched_iou, matched_gtidx = m_iou_tensor.max(-1)  # idx of gt plane which has max iou with predicted plane
            #     new_matched_iou = new_miou.max(-1)  # pred_num1/2
            #     new_matched_gtidx = new_miou.argmax(-1)  # pred_num1/2
            #     new_matched_iou_list.append(new_matched_iou)
            #     new_matched_gtidx_list.append(new_matched_gtidx)

            # new_pred_corr = np.stack(np.nonzero(new_pred_assignment_m), axis = -1) # pred_num_corr,2
            # new_pred_matched_num = new_pred_corr.shape[0]

            # new_gt_corr = np.array(self.dataset_dict[new_key]["gt_corrs"])
            # # correct_num = 0
            # for i in range(new_pred_matched_num):
            #     new_m_idxs = new_pred_corr[i]
            #     new_pred_idx0 = new_m_idxs[0]
            #     new_pred_idx1 = new_m_idxs[1]
            #     if new_matched_iou_list[0][new_pred_idx0] >= new_iou_thresh and new_matched_iou_list[1][new_pred_idx1] >= new_iou_thresh:
            #         new_gt_idx0 = new_matched_gtidx_list[0][new_pred_idx0]
            #         new_gt_idx1 = new_matched_gtidx_list[1][new_pred_idx1]
            #         if [new_gt_idx0, new_gt_idx1] in new_gt_corr:
            #             new_all_correct_num += 1
            #             pass

            # new_all_matched_num += pred_matched_num
            # new_all_gt_num += len(gt_corr)
            # 

        # print("self.match_threshold:", self.match_threshold)
        precision = float(all_correct_num) / float(all_matched_num)
        recall = float(all_correct_num) / float(all_gt_num)
        F_score = 2 * precision * recall / (precision + recall)

        print('precision2 = ', precision)
        print('recall2 = ', recall)
        print('F-score = ', F_score)
        print("TP = ", all_correct_num)
        print("Pred Num = ", all_matched_num)
        print("GT Num:", all_gt_num)

        if "singleview_attn" not in self.rcnn_data[idx]:
            update_precision = float(update_all_correct_num) / float(update_all_matched_num)
            update_recall = float(update_all_correct_num) / float(update_all_gt_num)
            update_F_score = 2 * update_precision * update_recall / (update_precision + update_recall)

            print('update_precision2 = ', update_precision)
            print('update_recall2 = ', update_recall)
            print('update_F-score = ', update_F_score)
            print("update_TP = ", update_all_correct_num)
            print("update_Pred Num = ", update_all_matched_num)
            print("update_GT Num:", update_all_gt_num)

    def evaluate_matching_from_iou(self, iou_thresh=0.5):
        
        all_matched_num = 0.
        all_gt_num = 0.
        all_correct_num = 0.
        pre = 0.
        recall = 0.
        for idx in tqdm(range(len(self.rcnn_data))):
            key = self.rcnnidx2datasetkey(idx)
        

            pred_assignment_m = self.rcnn_data[idx]["valid_assignment_matrixs"][self.corr_idx]


            matched_iou_list = []
            matched_gtidx_list = []
            for img_idx in ['0', '1']:
                # get gt masks
                gt_mask_rles = []
                
                for ann in self.dataset_dict[key][img_idx]["annotations"]:
                    if isinstance(ann["segmentation"], list):
                        polygons = [
                            np.array(p, dtype=np.float64) for p in ann["segmentation"]
                        ]
                        rles = mask_util.frPyObjects(
                            polygons,
                            self.dataset_dict[key][img_idx]["height"],
                            self.dataset_dict[key][img_idx]["width"],
                        )
                        rle = mask_util.merge(rles)
                    elif isinstance(ann["segmentation"], dict):  # RLE
                        rle = ann["segmentation"]
                    else:
                        raise TypeError(
                            f"Unknown segmentation type {type(ann['segmentation'])}!"
                        )
                    gt_mask_rles.append(rle)

                pred_mask_rles = [mask_util.encode(np.asfortranarray(self.rcnn_data[idx][img_idx]["sem_seg"]==segi))
                              for segi in range(len(self.rcnn_data[idx][img_idx]["valid_params"]))]

                

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
            
            # for pred_assignment_m, key in zip(pred_assignment_matrixs):
            pred_corr = np.stack(np.nonzero(pred_assignment_m), axis = -1) # pred_num_corr,2
            pred_matched_num = pred_corr.shape[0]

            gt_corr = np.array(self.dataset_dict[key]["gt_corrs"])
            #!!!
            gt_corr = gt_corr.tolist()
            # correct_num = 0
            for i in range(pred_matched_num):
                m_idxs = pred_corr[i]
                pred_idx0 = m_idxs[0]
                pred_idx1 = m_idxs[1]
                if matched_iou_list[0][pred_idx0] >= iou_thresh and matched_iou_list[1][pred_idx1] >= iou_thresh:
                    gt_idx0 = matched_gtidx_list[0][pred_idx0]
                    gt_idx1 = matched_gtidx_list[1][pred_idx1]
                    if [gt_idx0, gt_idx1] in gt_corr:
                        all_correct_num += 1

            all_matched_num += pred_matched_num
            all_gt_num += len(gt_corr)
        precision = float(all_correct_num) / float(all_matched_num)
        recall = float(all_correct_num) / float(all_gt_num)
        F_score = 2 * precision * recall / (precision + recall)

        print('precision2 = ', precision)
        print('recall2 = ', recall)
        print('F-score = ', F_score)
        print("TP = ", all_correct_num)
        print("Pred Num = ", all_matched_num)
        print("GT Num:", all_gt_num)

    def evaluate_ap_by_idx(self, idx):
        """
        get plane errors and mask errors
        """
        key = self.rcnnidx2datasetkey(idx)  # e.g. "2t7WUuJeko7_0_0_26__2t7WUuJeko7_0_0_40"

        update_pred_corr = self.update_pred_corr_list[idx]

        pred_camera = self.rcnn_data[idx]["pred_pose"]
        pred_camera = {
                "position": np.array(pred_camera["position"]),
                "rotation": quaternion.from_float_array(pred_camera["rotation"]), # w, x, y, z
            }


        """
        PRED
        """
        # Load predict camera
        # if pred_camera is None:
        #     pred_camera = self.get_camera_info(idx, tran_topk, rot_topk)
        #     pred_camera = {
        #         "position": np.array(pred_camera["position"]),
        #         "rotation": quaternion.from_float_array(pred_camera["rotation"]),
        #     }
        # else:
        #     assert tran_topk == -2 and rot_topk == -2
        #     pred_camera = {
        #         "position": np.array(pred_camera["position"]),
        #         "rotation": quaternion.from_float_array(pred_camera["rotation"]),  # convert to quaternion type
        #     }

        # Load single view prediction
        pred = {
            "0": {},
            "1": {},
            "merged": {},
            # "corrs": pred_corr,
            "corrs": update_pred_corr,
            "camera": pred_camera,
            "0_local": {},
            "1_local": {},
        }
        for i in range(2):
            if i == 0:
                camera_info = pred_camera
            else:
                camera_info = {
                    "position": np.array([0, 0, 0]),
                    "rotation": np.quaternion(1, 0, 0, 0),
                }
            
            # (n/d)
            pred_plane_single = self.rcnn_data[idx][str(i)]["valid_params"]

            offset = np.maximum(
                1/ np.linalg.norm(pred_plane_single, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)

            # normal = pred_plane_single / offset
            normal = pred_plane_single * offset
            pred[str(i) + "_local"]["offset"] = offset
            pred[str(i) + "_local"]["normal"] = normal
            pred[str(i) + "_local"]["scores"] = self.rcnn_data[idx][str(i)]["valid_scores"]

            # Global frame
            # convert from local to global
            # !(n*d)
            plane_global = get_plane_params_in_global(pred_plane_single*offset*offset, camera_info)
            offset = np.maximum(
                np.linalg.norm(plane_global, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = plane_global / offset

            # offset = np.maximum(
            #     1/ np.linalg.norm(pred_plane_single, ord=2, axis=1), 1e-5
            # ).reshape(-1, 1)
            # normal = pred_plane_single * offset

            pred[str(i)]["offset"] = offset
            pred[str(i)]["normal"] = normal
            pred[str(i)]["scores"] = self.rcnn_data[idx][str(i)]["valid_scores"]
        # Merge prediction
        merged_offset = []
        merged_normal = []
        merged_score = []
        for i in range(2):
            for ann_id in range(len(pred[str(i)]["scores"])):
                if len(pred["corrs"]) == 0 or ann_id not in pred["corrs"][:, i]:
                    # current plane matched to no plane
                    merged_offset.append(pred[str(i)]["offset"][ann_id])  # in global frame
                    merged_normal.append(pred[str(i)]["normal"][ann_id])
                    merged_score.append(pred[str(i)]["scores"][ann_id])

        for ann_id in pred["corrs"]:
            # average normal
            normal_pair = np.vstack(
                (pred["0"]["normal"][ann_id[0]], pred["1"]["normal"][ann_id[1]])
            )
            w, v = eigh(normal_pair.T @ normal_pair)
            avg_normals = v[:, np.argmax(w)]
            if (avg_normals @ normal_pair.T).sum() < 0:
                avg_normals = -avg_normals
            # average offset
            avg_offset = (
                pred["0"]["offset"][ann_id[0]] + pred["1"]["offset"][ann_id[1]]
            ) / 2
            merged_offset.append(avg_offset)
            merged_normal.append(avg_normals)
            # max score
            merged_score.append(
                max(pred["0"]["scores"][ann_id[0]], pred["1"]["scores"][ann_id[1]])
            )
        pred["merged"] = {
            "merged_offset": np.array(merged_offset),
            "merged_normal": np.array(merged_normal),
            "merged_score": np.array(merged_score)[:, np.newaxis],
        }
        """
        GT
        """
        gt_camera = self.dataset_dict[key]["rel_pose"]
        gt_camera = {
            "position": np.array(gt_camera["position"]),
            "rotation": quaternion.from_float_array(gt_camera["rotation"]),
        }
        gt_corr = np.array(self.dataset_dict[key]["gt_corrs"])

        # Load single view gt
        gt = {
            "0": {},
            "1": {},
            "merged": {},
            "corrs": gt_corr,
            "camera": gt_camera,
            "0_local": {},
            "1_local": {},
        }
        for i in range(2):
            if i == 0:
                camera_info = gt_camera
            else:
                camera_info = {
                    "position": np.array([0, 0, 0]),
                    "rotation": np.quaternion(1, 0, 0, 0),
                }
            #!
            plane_params = np.array(
                [ann["plane"] for ann in self.dataset_dict[key][str(i)]["annotations"]]
            )

            #!
            # Local frame
            offset = np.maximum(
                np.linalg.norm(plane_params, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = plane_params / offset
            gt[str(i) + "_local"]["offset"] = offset
            gt[str(i) + "_local"]["normal"] = normal

            # Global frame
            plane_global = get_plane_params_in_global(plane_params, camera_info)
            offset = np.maximum(
                np.linalg.norm(plane_global, ord=2, axis=1), 1e-5
            ).reshape(-1, 1)
            normal = plane_global / offset
            gt[str(i)]["offset"] = offset
            gt[str(i)]["normal"] = normal
        # Merge gt
        merged_offset = []
        merged_normal = []
        for i in range(2):
            for ann_id in range(len(gt[str(i)]["offset"])):
                if len(gt["corrs"]) == 0 or ann_id not in gt["corrs"][:, i]:
                    merged_offset.append(gt[str(i)]["offset"][ann_id])
                    merged_normal.append(gt[str(i)]["normal"][ann_id])
        for ann_id in gt["corrs"]:
            # average normal
            assert (
                np.linalg.norm(
                    gt["0"]["normal"][ann_id[0]] - gt["1"]["normal"][ann_id[1]]
                )
                < 1e-3
            )
            assert (
                np.abs(gt["0"]["offset"][ann_id[0]] - gt["1"]["offset"][ann_id[1]])
                < 1e-3
            )
            merged_offset.append(gt["0"]["offset"][ann_id[0]])
            merged_normal.append(gt["0"]["normal"][ann_id[0]])
        gt["merged"] = {
            "merged_offset": np.array(merged_offset),
            "merged_normal": np.array(merged_normal),
        }
        """
        ERRORs
        """
        # compute individual error in its own frame
        individual_error_offset = {}
        individual_error_normal = {}
        for i in range(2):
            individual_error_offset[str(i)] = np.abs(
                pred[str(i) + "_local"]["offset"] - gt[str(i) + "_local"]["offset"].T
            )  # n_pred, n_gt
            individual_error_normal[str(i)] = (
                np.arccos(
                    np.clip(
                        np.abs(
                            pred[str(i) + "_local"]["normal"]
                            @ gt[str(i) + "_local"]["normal"].T
                        ),
                        -1,
                        1,
                    )
                )
                / np.pi
                * 180
            )  # n_pred, n_gt

        """
        individual_miou = {'0': array-shape[n_pred, n_gt], 
                           '1': ...}
        """
        individual_miou = self.get_maskiou(idx)

        # compute merged error
        err_offsets = np.abs(
            pred["merged"]["merged_offset"] - gt["merged"]["merged_offset"].T
        )
        err_normals = (
            np.arccos(
                np.clip(
                    np.abs(
                        pred["merged"]["merged_normal"]
                        @ gt["merged"]["merged_normal"].T
                    ),
                    -1,
                    1,
                )
            )
            / np.pi
            * 180
        )

        mask_iou = self.get_maskiou_merged(
            idx, pred_corr=pred["corrs"], gt_corr=gt["corrs"]
        )

        output = {
            "err_offsets": err_offsets,  # n_pred_all, n_gt_all
            "err_normals": err_normals,  # n_pred_all, n_gt_all
            "mask_iou": mask_iou,  # n_pred_all, n_gt_all
            "scores": pred["merged"]["merged_score"],  # n_pred_all
            "individual_error_offset": individual_error_offset,
            "individual_error_normal": individual_error_normal,
            "individual_miou": individual_miou,
            "individual_score": {
                "0": pred["0"]["scores"].reshape(-1, 1),
                "1": pred["1"]["scores"].reshape(-1, 1),
            },
        }
        return output

    def get_maskiou(self, idx):
        """
        calculate mask iou between predicted mask and gt masks
        """
        key0 = self.rcnn_data[idx]["0"]["image_id"]
        key1 = self.rcnn_data[idx]["1"]["image_id"]
        key = key0 + "__" + key1
        mious = {}
        for i in range(2):
            gt_mask_rles = []
            for ann in self.dataset_dict[key][str(i)]["annotations"]:
                if isinstance(ann["segmentation"], list):
                    polygons = [
                        np.array(p, dtype=np.float64) for p in ann["segmentation"]
                    ]
                    rles = mask_util.frPyObjects(
                        polygons,
                        self.dataset_dict[key][str(i)]["height"],
                        self.dataset_dict[key][str(i)]["width"],
                    )
                    rle = mask_util.merge(rles)
                elif isinstance(ann["segmentation"], dict):  # RLE
                    rle = ann["segmentation"]
                else:
                    raise TypeError(
                        f"Unknown segmentation type {type(ann['segmentation'])}!"
                    )
                gt_mask_rles.append(rle)

            pred_mask_rles = [mask_util.encode(np.asfortranarray(self.rcnn_data[idx][str(i)]["sem_seg"]==segi))
                              for segi in range(len(self.rcnn_data[idx][str(i)]["valid_params"]))]

            miou = mask_util.iou(pred_mask_rles, gt_mask_rles, [0] * len(gt_mask_rles))
            mious[str(i)] = miou
        return mious

    def get_maskiou_merged(self, idx, pred_corr=None, gt_corr=None):
        """
        calculate mask iou between merged pred and merged gt
                gt_1    gt_2    gt_m
        pred_1  miou    0       miou(1)
        pred_2  0       miou    miou(2)
        pred_m  miou(1)  miou(2)  avg_miou(1,2)
        """
        mious = self.get_maskiou(idx)
        single2merge_dict = self.get_single2merge(
            idx, pred_corr=pred_corr, gt_corr=gt_corr
        )

        entry2gt_single_view = single2merge_dict["entry2gt_single_view"]
        gt_single_view2entry = single2merge_dict["gt_single_view2entry"]
        entry2pred_single_view = single2merge_dict["entry2pred_single_view"]
        pred_single_view2entry = single2merge_dict["pred_single_view2entry"]

        num_pred_entry = len(entry2pred_single_view.keys())
        num_gt_entry = len(entry2gt_single_view.keys())
        # pred_gt_merged_mask
        mask_iou = np.zeros((num_pred_entry, num_gt_entry))
        for r in range(num_pred_entry):
            for c in range(num_gt_entry):
                pred_merged = entry2pred_single_view[r]["merged"]
                gt_merged = entry2gt_single_view[c]["merged"]
                pair_id_pred = entry2pred_single_view[r]["pair"]
                pair_id_gt = entry2gt_single_view[c]["pair"]
                ann_id_pred = entry2pred_single_view[r]["ann_id"]
                ann_id_gt = entry2gt_single_view[c]["ann_id"]
                if not pred_merged and not gt_merged:
                    # pred_single & gt_single
                    # Should be in the same image
                    if pair_id_pred != pair_id_gt:
                        continue
                    else:
                        miou_single = mious[pair_id_pred]
                        mask_iou[r][c] = miou_single[ann_id_pred, ann_id_gt]
                elif pred_merged and not gt_merged:
                    # pred_merged & gt_single
                    miou_single = mious[pair_id_gt]
                    mask_iou[r][c] = miou_single[
                        ann_id_pred[int(pair_id_gt)], ann_id_gt
                    ]
                elif not pred_merged and gt_merged:
                    # pred_single & gt_merged
                    miou_single = mious[pair_id_pred]
                    mask_iou[r][c] = miou_single[
                        ann_id_pred, ann_id_gt[int(pair_id_pred)]
                    ]
                elif pred_merged and gt_merged:
                    # pred_merge & gt_merged, average both
                    miou_single = mious[str(0)]
                    iou0 = miou_single[ann_id_pred[0], ann_id_gt[0]]
                    miou_single = mious[str(1)]
                    iou1 = miou_single[ann_id_pred[1], ann_id_gt[1]]
                    mask_iou[r][c] = (iou0 + iou1) / 2
                else:
                    raise "BUG"

        return mask_iou

    def get_single2merge(self, idx, pred_corr=None, gt_corr=None):
        key = self.rcnnidx2datasetkey(idx)
        # GT merged mapping
        entry2gt_single_view = {}
        gt_single_view2entry = {"0": {}, "1": {}}
        if gt_corr is not None:
            gt_entry_id = 0
            for i in range(2):
                single_gt_idx = len(self.dataset_dict[key][str(i)]["annotations"])
                for s_i in range(single_gt_idx):
                    if s_i not in gt_corr[:, i]:
                        # process unmatched plane
                        entry2gt_single_view[gt_entry_id] = {
                            "pair": str(i),
                            "ann_id": s_i,
                            "merged": False,
                        }
                        gt_single_view2entry[str(i)][s_i] = gt_entry_id
                        gt_entry_id += 1
            for pair in gt_corr:
                # process matched plane
                entry2gt_single_view[gt_entry_id] = {
                    "pair": ["0", "1"],
                    "ann_id": pair,
                    "merged": True,
                }
                gt_single_view2entry["0"][pair[0]] = gt_entry_id
                gt_single_view2entry["1"][pair[1]] = gt_entry_id
                gt_entry_id += 1

        # Pred merged mapping
        entry2pred_single_view = {}
        pred_single_view2entry = {"0": {}, "1": {}}
        if pred_corr is not None:
            pred_entry_id = 0
            for i in range(2):
                
                single_idx = len(self.rcnn_data[idx][str(i)]["valid_params"])
                for s_i in range(single_idx):
                    if len(pred_corr) == 0 or s_i not in pred_corr[:, i]:
                        entry2pred_single_view[pred_entry_id] = {
                            "pair": str(i),
                            "ann_id": s_i,
                            "merged": False,
                        }
                        pred_single_view2entry[str(i)][s_i] = pred_entry_id
                        pred_entry_id += 1
            for pair in pred_corr:
                entry2pred_single_view[pred_entry_id] = {
                    "pair": ["0", "1"],
                    "ann_id": pair,
                    "merged": True,
                }
                pred_single_view2entry["0"][pair[0]] = pred_entry_id
                pred_single_view2entry["1"][pair[1]] = pred_entry_id
                pred_entry_id += 1
        return {
            "entry2gt_single_view": entry2gt_single_view,
            "gt_single_view2entry": gt_single_view2entry,
            "entry2pred_single_view": entry2pred_single_view,
            "pred_single_view2entry": pred_single_view2entry,
        }

    def evaluate_by_list(self, idxs, return_dict):
        for idx in idxs:
            rtn = self.evaluate_ap_by_idx(idx)
            return_dict[idx] = rtn

def multiprocess_by_list(ev, num_process, idx_list, evaluate, optimize=False, args=None):
    max_iter = len(idx_list)
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    per_thread = int(np.ceil(max_iter / num_process))
    split_by_thread = [
        idx_list[i * per_thread : (i + 1) * per_thread] for i in range(num_process)
    ]
    for i in range(num_process):
        p = Process(
            target=ev.evaluate_by_list, args=(split_by_thread[i], return_dict)
        )
        p.start()
        jobs.append(p)

    prev = 0
    with tqdm(total=max_iter) as pbar:
        while True:
            time.sleep(0.1)
            curr = len(return_dict.keys())
            pbar.update(curr - prev)
            prev = curr
            if curr == max_iter:
                break

    for job in jobs:
        job.join()

    return return_dict


def save_dict(return_dict, folder, prefix=None):
    os.makedirs(folder, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if prefix is None:
        save_path = os.path.join(folder, f"optimized_{timestr}.pkl")
    else:
        save_path = os.path.join(folder, prefix + ".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(return_dict.copy(), f)


def evaluate_by_idx(eval_dict):
    ndt, ngt = eval_dict["mask_iou"].shape
    if ndt == 0:
        stats = []
        for i in range(len(EP_ap_str)):
            tp = np.zeros((0, 1), dtype=bool)
            fp = np.zeros((0, 1), dtype=bool)
            sc = np.zeros((0, 1), dtype=bool)
            num_inst = ngt
            stats.append([tp, fp, sc, num_inst, None, None, None])
        # tqdm.write(str(0.0))
        return stats
    # Run the benchmarking code here.
    threshs = [EP_mask_delta_thresh, EP_normal_delta_thresh, EP_offset_delta_thresh]
    fn = [np.greater_equal, np.less_equal, np.less_equal]
    overlaps = [
        eval_dict["mask_iou"],
        eval_dict["err_normals"],
        eval_dict["err_offsets"],
    ]

    _dt = {"sc": eval_dict["scores"]}
    _gt = {"diff": np.zeros((ngt, 1), dtype=np.bool)}
    _bopts = {"minoverlap": 0.5}
    stats = []
    for i in range(len(EP_ap_str)):
        # Compute a single overlap that ands all the thresholds.
        ov = []
        for j in range(len(overlaps)):
            ov.append(fn[j](overlaps[j], threshs[j][i]))
        _ov = np.all(np.array(ov), 0).astype(np.float32)
        # Benchmark for this setting.
        tp, fp, sc, num_inst, dup_det, inst_id, ov = inst_bench_image(
            _dt, _gt, _bopts, _ov
        )
        stats.append([tp, fp, sc, num_inst, dup_det, inst_id, ov])
    return stats


def inst_bench_image(dt, gt, bOpts, overlap=None):
    nDt = len(dt["sc"])
    nGt = len(gt["diff"])
    numInst = np.sum(gt["diff"] == False)

    # if overlap is None:
    #  overlap = bbox_utils.bbox_overlaps(dt['boxInfo'].astype(np.float), gt['boxInfo'].astype(np.float))
    # assert(issorted(-dt.sc), 'Scores are not sorted.\n');
    sc = dt["sc"]

    det = np.zeros((nGt, 1)).astype(np.bool)
    tp = np.zeros((nDt, 1)).astype(np.bool)
    fp = np.zeros((nDt, 1)).astype(np.bool)
    dupDet = np.zeros((nDt, 1)).astype(np.bool)
    instId = np.zeros((nDt, 1)).astype(np.int32)
    ov = np.zeros((nDt, 1)).astype(np.float32)

    # Walk through the detections in decreasing score
    # and assign tp, fp, fn, tn labels
    for i in range(nDt):
        # assign detection to ground truth object if any
        if nGt > 0:
            maxOverlap = overlap[i, :].max()
            maxInd = overlap[i, :].argmax()
            instId[i] = maxInd
            ov[i] = maxOverlap
        else:
            maxOverlap = 0
            instId[i] = -1
            maxInd = -1
        # assign detection as true positive/don't care/false positive
        if maxOverlap >= bOpts["minoverlap"]:
            if gt["diff"][maxInd] == False:
                if det[maxInd] == False:
                    # true positive
                    tp[i] = True
                    det[maxInd] = True
                else:
                    # false positive (multiple detection)
                    fp[i] = True
                    dupDet[i] = True
        else:
            # false positive
            fp[i] = True
    return tp, fp, sc, numInst, dupDet, instId, ov


def inst_bench(dt, gt, bOpts, tp=None, fp=None, score=None, numInst=None):
    """
    ap, rec, prec, npos, details = inst_bench(dt, gt, bOpts, tp = None, fp = None, sc = None, numInst = None)
    dt  - a list with a dict for each image and with following fields
        .boxInfo - info that will be used to cpmpute the overlap with ground truths, a list
        .sc - score
    gt
        .boxInfo - info used to compute the overlap,  a list
        .diff - a logical array of size nGtx1, saying if the instance is hard or not
    bOpt
        .minoverlap - the minimum overlap to call it a true positive
    [tp], [fp], [sc], [numInst]
        Optional arguments, in case the inst_bench_image is being called outside of this function
    """
    details = None
    if tp is None:
        # We do not have the tp, fp, sc, and numInst, so compute them from the structures gt, and out
        tp = []
        fp = []
        numInst = []
        score = []
        dupDet = []
        instId = []
        ov = []
        for i in range(len(gt)):
            # Sort dt by the score
            sc = dt[i]["sc"]
            bb = dt[i]["boxInfo"]
            ind = np.argsort(sc, axis=0)
            ind = ind[::-1]
            if len(ind) > 0:
                sc = np.vstack((sc[i, :] for i in ind))
                bb = np.vstack((bb[i, :] for i in ind))
            else:
                sc = np.zeros((0, 1)).astype(np.float)
                bb = np.zeros((0, 4)).astype(np.float)

            dtI = dict({"boxInfo": bb, "sc": sc})
            tp_i, fp_i, sc_i, numInst_i, dupDet_i, instId_i, ov_i = inst_bench_image(
                dtI, gt[i], bOpts
            )
            tp.append(tp_i)
            fp.append(fp_i)
            score.append(sc_i)
            numInst.append(numInst_i)
            dupDet.append(dupDet_i)
            instId.append(instId_i)
            ov.append(ov_i)

        details = {
            "tp": list(tp),
            "fp": list(fp),
            "score": list(score),
            "dupDet": list(dupDet),
            "numInst": list(numInst),
            "instId": list(instId),
            "ov": list(ov),
        }

    tp = np.vstack(tp[:])
    fp = np.vstack(fp[:])
    sc = np.vstack(score[:])
    cat_all = np.hstack((tp, fp, sc))
    ind = np.argsort(cat_all[:, 2])  # from low score to high score
    cat_all = cat_all[ind[::-1], :]
    tp = np.cumsum(cat_all[:, 0], axis=0)
    fp = np.cumsum(cat_all[:, 1], axis=0)
    thresh = cat_all[:, 2]
    npos = np.sum(numInst, axis=0)

    # Compute precision/recall
    rec = tp / npos
    prec = np.divide(tp, (fp + tp))
    ap = VOCap(rec, prec)
    return ap, rec, prec, npos, details


def VOCap(rec, prec):
    rec = rec.reshape(rec.size, 1)
    prec = prec.reshape(prec.size, 1)
    z = np.zeros((1, 1))
    o = np.ones((1, 1))
    mrec = np.vstack((z, rec, o))
    mpre = np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    I = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in I:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def main(args):
    ev = Evaluator(args, dataset=args.dataset_phase, match_threshold = args.match_threshold, corr_idx=args.corr_idx, th_method=args.th_method,
                   normal_threshold = args.normal_threshold, offset_threshold = args.offset_threshold)
    path_seg = (args.rcnn_cached_file).split("/")
    out_dir = os.path.join(*path_seg[:-1])
    if args.rcnn_cached_file[0] == '/':
        out_dir = '/' + out_dir

    # if args.evaluate == "AP":
    ev.evaluate_matching_precision_recall(len(ev.rcnn_data))

    error_dict = multiprocess_by_list(
        ev, args.num_process, np.arange(len(ev.rcnn_data)), args.evaluate, False
    )
    bench_stats = []
    for idx in tqdm(range(len(ev.rcnn_data))):
        errs = error_dict[idx]
        bench_image_stats = evaluate_by_idx(errs)
        bench_stats.append(bench_image_stats)

    # Accumulate stats
    bb = list(zip(*bench_stats))
    bench_summarys = []
    print("mask IoU th:", EP_mask_delta_thresh[0], " offset th:", EP_offset_delta_thresh[0], " normal th:", EP_normal_delta_thresh[0])
    for i in range(len(EP_ap_str)):
        tp, fp, sc, num_inst, dup_det, inst_id, ov = zip(*bb[i])
        ap, rec, prec, npos, details = inst_bench(
            None, None, None, tp, fp, sc, num_inst
        )
        bench_summary = {
            "prec": prec.tolist(),
            "rec": rec.tolist(),
            "ap": ap[0],
            "npos": npos,
        }
        
        print("{:>20s}: {:5.2f}".format(EP_ap_str[i], ap[0] * 100.0))
        bench_summarys.append(bench_summary)
# # else:
#     # elif args.evaluate == "camera":
    cam_dict = ev.evaluate_camera(out_path=out_dir)
    # elif args.evaluate == "matching":
        # ev.optimized_dict = optimized_dict
    
    # ev.evaluate_matching_from_iou()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--rcnn-cached-file", required=True, help="path to instances_predictions.pth"
    )
    parser.add_argument(
        "--evaluate", default="correspondence", help="AP / camera / correspondence"
    )
    parser.add_argument(
        "--num-process",
        default=50,
        type=int,
        help="number of process for multiprocessing",
    )
    parser.add_argument(
        "--camera-cached-file", default='', required=False, help="path to summary.pkl"
    )
    parser.add_argument(
        "--num-data",
        default=-1,
        type=int,
        help="number of data to process, if -1 then all.",
    )
    parser.add_argument(
        "--dataset-phase", default="mp3d_test", type=str, help="dataset and phase"
    )

    parser.add_argument(
        "--match-threshold", default=0.2, type=float, help=""
    )

    parser.add_argument(
        "--corr-idx", default=0, type=int, help="idx of assignment matrixs of attn_fundamentals"
    )
    parser.add_argument(
        "--th-method", default=0, type=int, help="method of curr_match_threshold"
    )

    parser.add_argument(
        "--normal-threshold", default=45., type=float, help="normal_threshold for updating assignment matrixs"
    )

    parser.add_argument(
        "--offset-threshold", default=1., type=float, help="offset_threshold for updating assignment matrixs"
    )

    parser.add_argument("--opts", default=[])
    args = parser.parse_args()
    print(args)
    main(args)