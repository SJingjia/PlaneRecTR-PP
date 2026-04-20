import itertools
import json
import logging
import numpy as np
import os
from os.path import join as pjoin
from collections import OrderedDict
import torch

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

from lietorch import SE3
import lietorch

from PlaneRecTR.utils.misc import get_coordinate_map, save_list_to_file

import pycocotools.mask as mask_util
import cv2
import quaternion
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage.measurements import center_of_mass

from detectron2.utils.visualizer import Visualizer

from pytorch3d.structures import join_meshes_as_batch
from PlaneRecTR.utils.mesh_utils import (
    save_obj,
    get_camera_meshes,
    transform_meshes,
    rotate_mesh_for_webview,
    get_plane_params_in_global,
    get_plane_params_in_local,
    get_single_image_mesh_plane,
)
from PlaneRecTR.utils.pycococreatortools import binary_mask_to_polygon
from PlaneRecTR.utils.visualization import create_instances, get_labeled_seg, draw_match
from PlaneRecTR.utils.misc import get_intrinsics

from detectron2.config import get_cfg
from PlaneRecTR import add_PlaneRecTR_config
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
from tqdm import tqdm
from eval import assignment_inference, update_assignment_matrixs
import argparse
from detectron2.data import detection_utils as utils
import time
import seaborn as sns

def merge_plane_params_from_local_params(plane_locals, corr_list, camera_pose):
    """
    input: plane parameters in camera frame: #!n*d
    output: merged plane parameters using corr_list
    """
    param1, param2 = plane_locals["0"], plane_locals["1"]
    param1_global = get_plane_params_in_global(param1, camera_pose)
    param2_global = get_plane_params_in_global(
        param2, {"position": np.array([0, 0, 0]), "rotation": np.quaternion(1, 0, 0, 0)}
    )
    param1_global, param2_global = merge_plane_params_from_global_params(
        param1_global, param2_global, corr_list
    )
    param1 = get_plane_params_in_local(param1_global, camera_pose)
    param2 = get_plane_params_in_local(
        param2_global,
        {"position": np.array([0, 0, 0]), "rotation": np.quaternion(1, 0, 0, 0)},
    )
    # import pdb; pdb.set_trace()
    return {"0": param1, "1": param2}


def merge_plane_params_from_global_params(param1, param2, corr_list):
    """
    input: plane parameters in global frame #!n*d
    output: merged plane parameters using corr_list
    """
    pred = {"0": {}, "1": {}}
    pred["0"]["offset"] = np.maximum(
        np.linalg.norm(param1, ord=2, axis=1), 1e-5
    ).reshape(-1, 1)
    pred["0"]["normal"] = param1 / pred["0"]["offset"]
    pred["1"]["offset"] = np.maximum(
        np.linalg.norm(param2, ord=2, axis=1), 1e-5
    ).reshape(-1, 1)
    pred["1"]["normal"] = param2 / pred["1"]["offset"]
    for ann_id in corr_list:
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
        avg_plane = avg_normals * avg_offset
        param1[ann_id[0]] = avg_plane
        param2[ann_id[1]] = avg_plane
    return param1, param2


def save_matching(
    # img_file1,
    # img_file2,
    pred_dict,
    assignment,
    output_dir,
    prefix="",
    paper_img=False,
    score_threshold=0.7,
):
    """
    fp: whether show fp or fn
    gt_box: whether use gtbox
    """
    # image_paths = {"0": img_file1, "1": img_file2}
    blended = {}
    # centroids for matching
    centroids = {"0": [], "1": []}

    idxs1, idxs2 = np.where(assignment > 0)
    matched_num = idxs1.shape[0]
    idxs_all = [idxs1, idxs2]

    for i in range(2):
        # img = cv2.imread(image_paths[str(i)], cv2.IMREAD_COLOR)[:, :, ::-1]
        # img = cv2.resize(img, (640, 480))
        img = pred_dict[str(i)]["image"]  #! need RGB!!!!!!!!!!!
        height, width, _ = img.shape
        vis = Visualizer(img)

        # plane = pred_dict[str(i)]["pred_plane"].numpy()  # n, 3
        # ins = pred_dict[str(i)]["instances"]
        seg = pred_dict[str(i)]["segmentation"] # (h, w); gt: nonplane num_queries ; pred: nonplane num_queries+1
        scores = pred_dict[str(i)]["valid_scores"]
        num_planes = len(scores)
        segs = (np.expand_dims(seg, -1) == np.arange(num_planes)).astype(np.uint8) # (h.w,n)
        segs = segs.transpose((2,0,1)) # (n, h, w)
        
        p_instance_align = create_instances(
            segs,
            scores,
            img.shape[:2],
            # pred_planes=plane_new,
            conf_threshold=score_threshold,
        )  # <class 'detectron2.structures.instances.Instances'>


        seg_blended = get_labeled_seg(
            p_instance_align, score_threshold, vis, paper_img=paper_img
        )
        blended[str(i)] = seg_blended
        # centroid of mask
        # for ann in pred_dict[str(i)]["instances"]:
        # for oneseg in segs_new:
        for oneseg in segs:
            # M = center_of_mass(mask_util.decode(ann["segmentation"]))
            M = center_of_mass(oneseg)
            centroids[str(i)].append(M[::-1])  # reverse for opencv
        centroids[str(i)] = np.array(centroids[str(i)])

    pred_corr_list = np.array(torch.FloatTensor(assignment).nonzero().tolist())

    correct_list_pred = [True for pair in pred_corr_list]
    pred_matching_fig = draw_match(
        blended["0"],
        blended["1"],
        centroids["0"],
        centroids["1"],
        np.array(pred_corr_list),
        correct_list_pred,
        vertical=False,
        factor=1, 
        dotsize= int(20*height/480),
    )
    os.makedirs(output_dir, exist_ok=True)
    pred_matching_fig.save(os.path.join(output_dir, prefix + ".png"))


def save_pair_objects(
    pred_dict,
    output_dir,
    prefix="",
    pred_camera=None,
    plane_param_override=None,
    show_camera=True,
    assignment=None,
    webvis=False,
    save_mesh=True,
    camera_K=-1,
    radius = 0.01,
):
    meshes_list = []

    uv_maps = []
    cam_list = []
    # get plane parameters
    plane_locals = {}
    for i in range(2):
        if plane_param_override is None:
            plane_locals[str(i)] = pred_dict[str(i)]["valid_params"]  # n,3
        else:
            plane_locals[str(i)] = plane_param_override[str(i)]
        #! n/d -> n*d
        plane_locals[str(i)] = plane_locals[str(i)]/np.sum(plane_locals[str(i)]**2, axis = 1, keepdims = True)
    # get camera 1 to 2
    camera1to2 = {
        "position": np.array(pred_camera["position"]),
        "rotation": quaternion.from_float_array(pred_camera["rotation"]),
    }

    # Merge planes if they are in correspondence
    corr_list = np.argwhere(assignment)
    if len(corr_list) != 0:
        plane_locals = merge_plane_params_from_local_params(
            plane_locals, corr_list, camera1to2
        )

    height, width, _ = pred_dict["0"]["image"].shape

    os.makedirs(output_dir, exist_ok=True)
    for i in range(2):
        if i == 0:
            camera_info = camera1to2
        else:
            camera_info = {
                "position": np.array([0, 0, 0]),
                "rotation": np.quaternion(1, 0, 0, 0),
            }
        plane_params = plane_locals[str(i)]
        seg = pred_dict[str(i)]["segmentation"] # (h, w); gt: nonplane num_queries ; pred: nonplane num_queries+1
        scores = pred_dict[str(i)]["valid_scores"]
        num_planes = len(scores)
        segs = (np.expand_dims(seg, -1) == np.arange(num_planes)).astype(np.uint8) # (h.w,n)
        segs = segs.transpose((2,0,1)) # (n, h, w)
        tolerance = 0.0
        poly_segs = [binary_mask_to_polygon(bm, tolerance) for bm in segs]
        meshes, uv_map = get_single_image_mesh_plane( 
            plane_params,
            poly_segs,
            image = pred_dict[str(i)]["image"], # ! RGB
            height=height,
            width=width,
            webvis=False,
            camera_K=camera_K
        )
        uv_maps.extend(uv_map)
        meshes = transform_meshes(meshes, camera_info)
        meshes_list.append(meshes)
        cam_list.append(camera_info)

    joint_mesh = join_meshes_as_batch(meshes_list)
    if webvis:
        joint_mesh = rotate_mesh_for_webview(joint_mesh)

    # add camera into the mesh
    if show_camera:
        cam_meshes = get_camera_meshes(cam_list, radius=radius)
        if webvis:
            cam_meshes = rotate_mesh_for_webview(cam_meshes)
    else:
        cam_meshes = None

    # save obj
    if len(prefix) == 0:
        prefix = "pred"
    save_obj(
        folder=output_dir,
        prefix=prefix,
        meshes=joint_mesh,
        cam_meshes=cam_meshes,
        decimal_places=10,
        blend_flag=True,
        map_files=None,
        uv_maps=uv_maps,
        save_mesh=save_mesh
    )



def setup(args):
    cfg = get_cfg()

    add_deeplab_config(cfg)
    add_PlaneRecTR_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

class SparseViewsVisualizer():
    def __init__(
        self,
        args,
        dataset_name,
        output_dir=None,
        # num_planes=None,
        vis_matching = False,
        vis_3dmesh = False,
        vis_frustum = False,
        vis_period = 50,
        match_threshold = 0.2, 
        corr_idx = 0, 
        th_method = 0,
        normal_threshold = 45., 
        offset_threshold = 1.,
        radius = 0.01, 
    ):
        cfg = setup(args)

        self._dataset_name = dataset_name
        if output_dir is not None:  
            self._output_dir = output_dir
        else:
            self._output_dir = pjoin(os.path.split(args.rcnn_cached_file)[0], "outputs") 


        self._cpu_device = torch.device("cpu")

        self._num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.vis_matching = vis_matching
        self.vis_3dmesh = vis_3dmesh
        self.vis_frustum = vis_frustum
        self.vis_period = vis_period

        self.image_size = cfg.INPUT.IMAGE_SIZE
        self.k_inv_dot_xy1 = get_coordinate_map(dataset_name, self._cpu_device,
                                                h = self.image_size[0], w = self.image_size[1]).numpy()

        self.match_threshold = match_threshold
        self.corr_idx = corr_idx
        self.th_method = th_method

        self.normal_threshold = normal_threshold
        self.offset_threshold = offset_threshold

        self.radius = radius

        self.save_prefix = "_".join([
                                     "matchTh="+str(self.match_threshold),
                                     "thMethod="+str(self.th_method),
                                     "nTh="+str(self.normal_threshold),
                                     "oTh="+str(self.offset_threshold)
                                     ]).replace(".","-")


        self.root_dir = cfg.DATASETS.ROOT_DIR
        self.img_format = cfg.INPUT.FORMAT

        rcnn_cached_file = args.rcnn_cached_file
        with open(rcnn_cached_file, "rb") as f:
            print('loading rcnn cached file from {}'.format(rcnn_cached_file))
            self.rcnn_data = list(pickle.load(f).values())  # dict-> list
            print("rcnn cached file has been loaded")
       
        self.metadata = MetadataCatalog.get(dataset_name)
        self.load_input_dataset(dataset_name)
        self.sanity_check()
        self.cal_assignment_matrixs()
        self.evaluate_camera()

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

    def cal_assignment_matrixs(self):
        print("Calculating assignment matrixs...")
        self.update_valid_assignment_matrixs = {}
        for idx in tqdm(range(len(self.rcnn_data))):
            key = self.rcnnidx2datasetkey(idx)
            label_mask0=torch.from_numpy(self.rcnn_data[idx]["0"]["label_mask"])
            label_mask1=torch.from_numpy(self.rcnn_data[idx]["1"]["label_mask"])
            assignment_matrixs = assignment_inference(torch.from_numpy(self.rcnn_data[idx]["attn_fundamentals"]),
                                                        self.match_threshold, self.th_method, 
                                                        label_mask0=label_mask0,
                                                        label_mask1=label_mask1
                                                        )
            valid_assignment_matrixs = assignment_matrixs[:,label_mask0][:,:,label_mask1]
            
            # pred_corr = np.argwhere(valid_assignment_matrixs.numpy()[self.corr_idx])


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
            # update_pred_corr = np.argwhere(update_valid_assignment_matrix.numpy()[0])
            self.update_valid_assignment_matrixs[key] = update_valid_assignment_matrix.numpy()[0]
        print("Done.")

    def evaluate_camera(self):
        print("Evaluating camera ...")
        tran_errs = []
        rot_errs = []
        self.camera_eval_dict = {}
        self.gt_mags = {}
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
            tran_err = np.linalg.norm(pred_cam["position"] - np.array(gt_cam["position"]))
            tran_errs.append(tran_err)
            # Error - rotation
            if type(pred_cam["rotation"]) != np.ndarray:
                print("Need to convert quaternion to np array")
                raise
            d = np.abs(np.sum(np.multiply(pred_cam["rotation"], np.array(gt_cam["rotation"]))))
            d = np.clip(d, -1, 1)
            rot_err = 2 * np.arccos(d) * 180 / np.pi
            rot_errs.append(rot_err)


            self.camera_eval_dict[key] = {"tran_err": tran_err, "rot_err":rot_err}

            self.gt_mags[key] = {"gttran": np.linalg.norm(gt_cam["position"]), "gtrot": 2 * np.arccos(gt_cam["rotation"][0]) * 180 / np.pi}

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

        print("Done.")

    def evaluate_matching_precision_recall(self):

        print("Evaluating matching precision and recall...")
        self.matching_pr_dict = {}

        update_all_matched_num = 0.
        update_all_gt_num = 0.
        update_all_correct_num = 0.

        for idx in tqdm(range(len(self.rcnn_data))):
            key = self.rcnnidx2datasetkey(idx)  # e.g. "2t7WUuJeko7_0_0_26__2t7WUuJeko7_0_0_40"
            
            update_pred_corr = np.argwhere(self.update_valid_assignment_matrixs[key])

                    
            gt_corr = np.array(self.dataset_dict[key]["gt_corrs"])
            gt_corr = gt_corr.tolist() #!
            individual_miou = self.get_maskiou(idx)

            # print(key)

            m_iou0 = individual_miou["0"]
            m_iou1 = individual_miou["1"]

            m_iou0 = torch.from_numpy(m_iou0)
            m_iou1 = torch.from_numpy(m_iou1)

            matched_gtiou0, matched_gtidx0 = m_iou0.max(-1)
            matched_gtiou1, matched_gtidx1 = m_iou1.max(-1)

            # pred_matched_num = len(pred_corr)
            # correct_num = 0
            # for i in range(pred_matched_num):
            #     m_idxs = pred_corr[i]
            #     pred_idx0 = m_idxs[0]
            #     pred_idx1 = m_idxs[1]

            #     if matched_gtiou0[pred_idx0] >= 0.5 and matched_gtiou1[pred_idx1] >= 0.5:
            #         gt_idx0 = matched_gtidx0[pred_idx0]
            #         gt_idx1 = matched_gtidx1[pred_idx1]
            #         if [gt_idx0, gt_idx1] in gt_corr:
            #             correct_num += 1

            # all_matched_num += pred_matched_num
            # all_correct_num += correct_num
            # all_gt_num += len(gt_corr)

            #UPDATE
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


            self.matching_pr_dict[key] = {}
            if update_pred_matched_num == 0:
                self.matching_pr_dict[key]["precision"] = -1
                # continue
            else:
                self.matching_pr_dict[key]["precision"] = float(update_correct_num) / float(update_pred_matched_num)
                # pre += float(update_correct_num) / float(update_pred_matched_num)

            if len(gt_corr) == 0:
                self.matching_pr_dict[key]["recall"] = -1
                # continue
            else:
                self.matching_pr_dict[key]["recall"] = float(update_correct_num) / float(len(gt_corr))
                # recall += float(update_correct_num) / float(len(gt_corr))

        update_precision = float(update_all_correct_num) / float(update_all_matched_num)
        update_recall = float(update_all_correct_num) / float(update_all_gt_num)
        update_F_score = 2 * update_precision * update_recall / (update_precision + update_recall)

        print('update_precision2 = ', update_precision)
        print('update_recall2 = ', update_recall)
        print('update_F-score = ', update_F_score)
        print("update_TP = ", update_all_correct_num)
        print("update_Pred Num = ", update_all_matched_num)
        print("update_GT Num:", update_all_gt_num)

        print("Done.")

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

    def process(self):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        inputs = self.dataset_dict
        outputs = self.rcnn_data


        vis_iter = 0
        # for input, output in tqdm(zip(inputs, outputs)):
        for output in tqdm(outputs):

            if vis_iter % self.vis_period != 0:
                vis_iter += 1
                continue

            # single image evaluation
            key0 = output["0"]["image_id"]
            key1 = output["1"]["image_id"]
            key = key0 + "__" + key1

            input = inputs[key]

            cam_dict = self.camera_eval_dict[key]
            gtcam_dict = self.gt_mags[key]
            # cam_error_prefix = "IND" + str(vis_iter) +"_R{:.1f}_t{:.1f}_gtR{:.1f}_gtt{:.1f}".format(cam_dict["rot_err"], cam_dict["tran_err"], gtcam_dict["gtrot"], gtcam_dict["gttran"]).replace(".","-")
            cam_error_prefix = ""

            gt_vis_dicts = []
            pred_vis_dicts = []
            file_names = [input["0"]["image_id"], input["1"]["image_id"]]

            for i in range(2):
                pred = output[str(i)]["sem_seg"] # (480, 640)
       
                valid_params = output[str(i)]["valid_params"]
                valid_scores = output[str(i)]["valid_scores"]

                if "sparseviews" in self._dataset_name:
                    
                    if "scannetv2" in self._dataset_name:

                        ps = input[str(i)]["file_name"].split("/")
                        ps.insert(4, "frames")
                        recent_file_name = os.path.join(self.root_dir, "/".join(ps[2:]))
                        input[str(i)]["file_name"] = recent_file_name

                        # 2.1 read image
                        image = utils.read_image(
                            input[str(i)]["file_name"], format=self.img_format
                        )
                        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

                        image_id = input[str(i)]["image_id"]
                        scene_idx, image_idx = image_id.split('-')
                        masks_path = os.path.join(
                            self.root_dir,
                            "twoView_Anns",
                            scene_idx,
                            image_idx + ".pkl",
                        )
                        with open(masks_path, "rb") as f:
                            obs = pickle.load(f) # dict_keys(['plane_masks', 'camera_K'])
                        gt_plane_masks = obs['plane_masks'] # BitMasks(num_instances=10)

                        gt_plane_masks = gt_plane_masks.tensor.numpy()
                        # gt = np.ones_like(gt_plane_masks[0])*20
                        gt = np.ones_like(gt_plane_masks[0])*self._num_queries
                        for idx,gpm in enumerate(gt_plane_masks):
                            gt[gpm==1]=idx


                    elif "mp3d" in self._dataset_name:                        
                        ps = input[str(i)]["file_name"].split("/")
                        # ps.insert(4, "frames")
                        recent_file_name = os.path.join(self.root_dir, "/".join(ps[6:]))
                        input[str(i)]["file_name"] = recent_file_name

                        # 2.1 read image
                        image = utils.read_image(
                            input[str(i)]["file_name"], format=self.img_format
                        )
                        
                        house, img_id = input[str(i)]["image_id"].split("_", 1)
                        masks_path = os.path.join(
                            self.root_dir,
                            "observations",
                            house,
                            img_id + ".pkl",
                        )

                        with open(masks_path, "rb") as f:
                            obs = pickle.load(f) # dict_keys(['color_sensor', 'depth_sensor', 'semantic_sensor'])
                        # This assertion is to check dataset is clean
                        # assert((obs['color_sensor'][:,:,:3][:,:,::-1].transpose(2, 0, 1)-dataset_dict[str(i)]["image"].numpy()).sum()==0)
                        semantic_map = obs["semantic_sensor"] # 0: non-plane (480, 640) uint32
                        plane_ids = np.unique(semantic_map)
                        if plane_ids[0] == 0:
                            plane_ids = plane_ids[1:]
                        plane_num = len(plane_ids)

                        plane_seg_gt = np.ones_like(semantic_map) * self._num_queries
                        labels = []
                        for label_id, pid in enumerate(plane_ids):
                            plane_seg_gt[semantic_map==pid] = label_id
                            labels.append(label_id)
                        gt = plane_seg_gt.astype(np.uint8)



                    annos = [
                    obj
                    for obj in input[str(i)].pop("annotations")
                        if obj.get("iscrowd", 0) == 0
                    ]
                    if len(annos) and "plane" in annos[0]:
                        plane = [np.array(obj["plane"]).reshape(1,3) for obj in annos] # ! n*X - d = 0  obj["plane"]: n*d
                        gt_params = np.concatenate(plane)
                        gt_params /= np.sum(gt_params**2, axis = 1).reshape(-1,1) # n*d -> n/d
                        gt_params = gt_params.astype(np.float32)

                    assert len(gt_params) == (len(np.unique(gt))-1)
                    
                else:
                    print(self._dataset_name)

                gt_vis_dicts.append({
                    'image': image, # (h, w, 3)
                    'segmentation': gt,
                    # 'depth_GTplane': gt_plane_depth,
                    # 'depth_GTori': gt_ori_depth,
                    'valid_params': gt_params,
                    'valid_scores': np.ones(len(gt_params)),
                    'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                    })
                pred_vis_dicts.append({
                    'image': image,
                    'segmentation': pred,
                    # 'depth_predplane': plane_depth,
                    'valid_params': valid_params,
                    'valid_scores': valid_scores,
                    'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                })
                
            # pred_poses = output["pred_poses"] # SE3 [2,6]?
            pred_camera_dict = output["pred_pose"] # {"rotation": w,x,y,z, ...}
            gt_camera_dict = input['rel_pose']


            # assignment_matrixs = output["valid_assignment_matrixs"].to(self._cpu_device).numpy()
            pred_assignment_matrix = self.update_valid_assignment_matrixs[key]
            gt_corrs = np.array(input["gt_corrs"])
            gt_assignment_matrix = np.zeros((len(gt_vis_dicts[-2]["valid_params"]), len(gt_vis_dicts[-1]["valid_params"])))
            gt_assignment_matrix[(gt_corrs[:,0], gt_corrs[:,1])] = 1

            if self.vis_matching:
                save_matching(
                                {"0": gt_vis_dicts[-2], "1": gt_vis_dicts[-1]},
                                gt_assignment_matrix,
                                pjoin(self._output_dir, "corr_" + self.save_prefix),
                                prefix= file_names[-2] + "__" + file_names[-1] + "_gt_" + cam_error_prefix,
                                paper_img=True,
                                score_threshold=0.5,
                                )
                
                pscoreth = 0.0
                # for ai in range(len(assignment_matrixs)):
                #     save_matching(
                #                     {"0": pred_vis_dicts[-2], "1": pred_vis_dicts[-1]},
                #                     assignment_matrixs[ai],
                #                     pjoin(self._output_dir, "corr_" + self.save_prefix),
                #                     prefix= "_".join([file_names[-2],file_names[-1], "pred", "scoreth"+str(pscoreth), "attn"+str(ai)]),
                #                     paper_img=True,
                #                     score_threshold=pscoreth,
                #                     )
                save_matching(
                                {"0": pred_vis_dicts[-2], "1": pred_vis_dicts[-1]},
                                pred_assignment_matrix,
                                pjoin(self._output_dir, "corr_" + self.save_prefix),
                                prefix= file_names[-2] + "__" + file_names[-1] + "_pred_" + cam_error_prefix,
                                paper_img=True,
                                score_threshold=pscoreth,
                                )
                

            camera_K = get_intrinsics(self._dataset_name, h = image.shape[0], w = image.shape[1]).cpu().numpy()
            focal_x, focal_y, offset_x, offset_y = camera_K[0][0]
            camera_K = np.array([[focal_x, 0, offset_x], [0, focal_y, offset_y], [0, 0, 1]])

            

            if self.vis_3dmesh:
                save_pair_objects(
                                    pred_dict = {"0": pred_vis_dicts[-2], "1": pred_vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, file_names[-2] + "__" + file_names[-1] +"_"+ self.save_prefix + "_" + cam_error_prefix, "attn0_pred"),
                                    prefix = file_names[-2] + "__" + file_names[-1] + "_attn0_pred",
                                    pred_camera = pred_camera_dict,
                                    plane_param_override=None,
                                    show_camera = False,
                                    # assignment = assignment_matrixs[0],
                                    assignment=pred_assignment_matrix,
                                    webvis=False,
                                    save_mesh=True,
                                    camera_K=camera_K,
                                    radius = self.radius,
                                )
            
                save_pair_objects(
                                    pred_dict = {"0": gt_vis_dicts[-2], "1": gt_vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, file_names[-2] + "__" + file_names[-1] +"_"+ self.save_prefix + "_" + cam_error_prefix, "attn0_gt"),
                                    prefix = file_names[-2] + "__" + file_names[-1] + "_attn0_gt",
                                    pred_camera = gt_camera_dict,
                                    plane_param_override=None,
                                    show_camera = False,
                                    assignment = gt_assignment_matrix,
                                    webvis=False,
                                    save_mesh=True,
                                    camera_K=camera_K,
                                    radius=self.radius,
                                )
                cv2.imwrite(pjoin(self._output_dir, file_names[-2] + "__" + file_names[-1] +"_"+ self.save_prefix + "_" + cam_error_prefix, file_names[-2] + "__" + file_names[-1] + "_1.png"), pred_vis_dicts[-2]["image"][:,:,::-1])
                cv2.imwrite(pjoin(self._output_dir, file_names[-2] + "__" + file_names[-1] +"_"+ self.save_prefix + "_" + cam_error_prefix, file_names[-2] + "__" + file_names[-1] + "_2.png"), pred_vis_dicts[-1]["image"][:,:,::-1])  #! RGB-> BGR
                
            
            if self.vis_frustum:
                save_pair_objects(
                                    pred_dict = {"0": pred_vis_dicts[-2], "1": pred_vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, file_names[-2] + "__" + file_names[-1] +"_"+ self.save_prefix + "_" + cam_error_prefix, "attn0_predfrustum"),
                                    prefix = file_names[-2] + "__" + file_names[-1] + "attn0_predfrustum",
                                    pred_camera = pred_camera_dict,
                                    plane_param_override=None,
                                    show_camera = True,
                                    # assignment = assignment_matrixs[0],
                                    assignment = pred_assignment_matrix,
                                    webvis=False,
                                    save_mesh=False,
                                    camera_K=camera_K,
                                    radius=self.radius,
                                )

                save_pair_objects(
                                    pred_dict = {"0": gt_vis_dicts[-2], "1": gt_vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, file_names[-2] + "__" + file_names[-1] +"_"+ self.save_prefix + "_" + cam_error_prefix, "attn0_gtfrustum"),
                                    prefix = file_names[-2] + "__" + file_names[-1] + "attn0_gtfrustum",
                                    pred_camera = gt_camera_dict,
                                    plane_param_override=None,
                                    show_camera = True,
                                    assignment = gt_assignment_matrix,
                                    webvis=False,
                                    save_mesh=False,
                                    camera_K=camera_K,
                                    radius=self.radius,
                                )
        
            vis_iter += 1
        
        


def main(args):
    ev = SparseViewsVisualizer(args, 
                               dataset_name=args.dataset_phase,
                               output_dir = args.output_dir, 
                               vis_matching = args.vis_matching,
                               vis_3dmesh = args.vis_3dmesh,
                               vis_frustum = args.vis_frustum,
                               vis_period = args.vis_period,
                               match_threshold = args.match_threshold, 
                               corr_idx=args.corr_idx, 
                               th_method=args.th_method,
                               normal_threshold = args.normal_threshold, 
                               offset_threshold = args.offset_threshold,
                               radius=args.radius,
                               )
    ev.process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--rcnn-cached-file", required=True, help="path to PlaneRecTRpp_outputs.pkl"
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
        "--output-dir", default=None, type=str, help=""
    )
    
    parser.add_argument(
        "--vis-matching", default=False, type=bool, help=""
    )

    parser.add_argument(
        "--vis-3dmesh", default=False, type=bool, help=""
    )

    parser.add_argument(
        "--vis-frustum", default=False, type=bool, help=""
    )

    parser.add_argument(
        "--vis-period", default=50, type=int, help=""
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

    parser.add_argument(
        "--radius", default=0.01, type=float, help="")

    parser.add_argument("--opts", default=[])
    args = parser.parse_args()
    print(args)
    main(args)