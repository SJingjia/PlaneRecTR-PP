# https://github.com/facebookresearch/Mask2Former
# 

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

from ..utils.misc import get_coordinate_map

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
    # get_single_image_mesh_with_calibration,
)
from PlaneRecTR.utils.pycococreatortools import binary_mask_to_polygon
from PlaneRecTR.utils.visualization import create_instances, get_labeled_seg, draw_match
from PlaneRecTR.utils.misc import get_intrinsics


def merge_plane_params_from_local_params(plane_locals, corr_list, camera_pose):
    """
    input: plane parameters in camera frame: #!n*d
    output: merged plane parameters using corr_list
    """
    param1, param2 = plane_locals["0"], plane_locals["1"]
    param1_global = get_plane_params_in_global(param1, camera_pose) #! cam1 to world(cam2)
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
    blended = {}
    # centroids for matching
    centroids = {"0": [], "1": []}

    idxs1, idxs2 = np.where(assignment > 0)
    matched_num = idxs1.shape[0]
    idxs_all = [idxs1, idxs2]

    for i in range(2):
        img = pred_dict[str(i)]["image"]
        height, width, _ = img.shape
        vis = Visualizer(img)

        seg = pred_dict[str(i)]["segmentation"] # (h, w); gt: nonplane num_queries ; pred: nonplane num_queries+1
        scores = pred_dict[str(i)]["valid_scores"]
        num_planes = len(scores)
        segs = (np.expand_dims(seg, -1) == np.arange(num_planes)).astype(np.uint8) # (h.w,n)
        segs = segs.transpose((2,0,1)) # (n, h, w)

        p_instance_align = create_instances(
            segs,
            scores,
            img.shape[:2],
            conf_threshold=score_threshold,
        )  # <class 'detectron2.structures.instances.Instances'>


        seg_blended = get_labeled_seg(
            p_instance_align, score_threshold, vis, paper_img=paper_img
        )
        blended[str(i)] = seg_blended
        # centroid of mask
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
    camera_K=-1
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
        # p_instance = p_instances[str(i)]
        plane_params = plane_locals[str(i)]
        # segmentations = p_instance.pred_masks
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
            image = pred_dict[str(i)]["image"][:,:,::-1], # ! BGR -> RGB
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
    # import pdb;pdb.set_trace()
    if webvis:
        joint_mesh = rotate_mesh_for_webview(joint_mesh)

    # add camera into the mesh
    if show_camera:
        cam_meshes = get_camera_meshes(cam_list)
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


class SparseViewsVisualizer(DatasetEvaluator):
    """
    Evaluate plane segmentation metrics.
    """
    eval_iter = 0
    vis_iter = 0

    def __init__(
        self,
        dataset_name,
        output_dir=None,
        *,
        num_planes=None,
        vis_matching = False,
        vis_3dmesh = False,
        vis_frustum = False,
        vis_period = 50,
        eval_period = 500,
        image_size = (480, 640),
        predict_poses = True,
    ):
        self._logger = logging.getLogger(__name__)
        if num_planes is not None:
            self._logger.warn(
                "SparseViewsVisualizer(num_planes) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._num_planes = num_planes
        self._num_queries = num_planes + 1 if "npr" in dataset_name else num_planes # TODO: add npr
        self.vis_matching = vis_matching
        self.vis_3dmesh = vis_3dmesh
        self.vis_frustum = vis_frustum
        self.vis_period = vis_period
        self.eval_period = eval_period

        self.k_inv_dot_xy1 = get_coordinate_map(dataset_name, self._cpu_device,
                                                h = image_size[0], w = image_size[1]).numpy()
        self.predict_poses = predict_poses

    def reset(self):
        
        self.vis_dicts = []
        self.gt_vis_dicts = []
        self.file_names = []
        # !
        SparseViewsVisualizer.eval_iter += self.eval_period
        SparseViewsVisualizer.vis_iter = 0

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """


        for input, output in zip(inputs, outputs):
            # output, indices, gt_corrs = output_tuple
            if SparseViewsVisualizer.vis_iter % self.vis_period != 0:
                continue

            # single image evaluation
            for i in range(2):
                sem_seg = output[str(i)]["sem_seg"].argmax(dim=0).to(self._cpu_device) # torch.Size([480, 640]) # sem_seg 21, 192, 256
                pred = np.array(sem_seg, dtype=np.int) # (480, 640)
                plane_depth = output[str(i)]["planes_depth"].to(self._cpu_device).numpy()
                # seg_depth = output[str(i)]["seg_depth"].to(self._cpu_device).numpy()
                valid_params = output[str(i)]["valid_params"].to(self._cpu_device).numpy()
                valid_scores = output[str(i)]["valid_scores"].to(self._cpu_device).numpy()

                if "sparseviews" in self._dataset_name:
                    gt_plane_masks = input[str(i)]["plane_masks"].tensor.to(self._cpu_device).numpy()
                    # gt = np.ones_like(gt_plane_masks[0])*20
                    gt = np.ones_like(gt_plane_masks[0])*self._num_queries
                    for idx,gpm in enumerate(gt_plane_masks):
                        gt[gpm==1]=idx
                    gt_plane_depth = torch.sum(input[str(i)]["plane_depths"], dim = 0).to(self._cpu_device).numpy()
                    gt_ori_depth = torch.sum(input[str(i)]["depths"], dim = 0).to(self._cpu_device).numpy()
                    gt_params = input[str(i)]["params"].to(self._cpu_device).numpy()   
                else:
                    print(self._dataset_name)

                        
                if "sparseviews" in self._dataset_name:
                    image = input[str(i)]["image"].to(self._cpu_device).numpy().transpose((1,2,0)) # (h,w,3)
                    # file_name = "_".join(input[str(i)]["file_name"].split("/")[6:]).split(".")[0]
                    # if "7scenes" in self._dataset_name:
                    file_name = input[str(i)]["image_id"]


                self.gt_vis_dicts.append({
                    'image': image, # (h, w, 3)
                    'segmentation': gt,
                    'depth_GTplane': gt_plane_depth,
                    'depth_GTori': gt_ori_depth,
                    'valid_params': gt_params,
                    'valid_scores': np.ones(len(gt_params)),
                    'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                    })
                self.vis_dicts.append({
                    'image': image,
                    'segmentation': pred,
                    'depth_predplane': plane_depth,
                    'valid_params': valid_params,
                    'valid_scores': valid_scores,
                    'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                })
                self.file_names.append(file_name)


            if self.predict_poses:
                pred_poses = output["pred_poses"] # SE3 [2,6]?
                rel_pose = input['rel_pose']
                

                assignment_matrixs = output["valid_assignment_matrixs"].to(self._cpu_device).numpy()
                gt_corrs = np.array(input["gt_corrs"])
                gt_assignment_matrix = np.zeros((len(self.gt_vis_dicts[-2]["valid_params"]), len(self.gt_vis_dicts[-1]["valid_params"])))
                gt_assignment_matrix[(gt_corrs[:,0], gt_corrs[:,1])] = 1


                pp1,pp2,pp3,px,py,pz,pw = pred_poses[1].data.cpu().numpy()
                pred_camera_dict = {
                    "rotation": np.array([pw,px,py,pz]),
                    "position": np.array([pp1,pp2,pp3])
                }
                gt_camera_dict = rel_pose

            if self.vis_matching:
                save_matching(
                                {"0": self.gt_vis_dicts[-2], "1": self.gt_vis_dicts[-1]},
                                gt_assignment_matrix,
                                pjoin(self._output_dir, "corr_" + str(SparseViewsVisualizer.eval_iter)),
                                prefix= "_".join([self.file_names[-2],self.file_names[-1], "gt"]),
                                paper_img=True,
                                score_threshold=0.5,
                                )
                
                pscoreth = 0.0
                for ai in range(len(assignment_matrixs)):
                    save_matching(
                                    {"0": self.vis_dicts[-2], "1": self.vis_dicts[-1]},
                                    assignment_matrixs[ai],
                                    pjoin(self._output_dir, "corr_" + str(SparseViewsVisualizer.eval_iter)),
                                    prefix= "_".join([self.file_names[-2],self.file_names[-1], "pred", "scoreth"+str(pscoreth), "attn"+str(ai)]),
                                    paper_img=True,
                                    score_threshold=pscoreth,
                                    )
                cv2.imwrite(pjoin(self._output_dir, "corr_" + str(SparseViewsVisualizer.eval_iter), "_".join([self.file_names[-2], self.file_names[-1], "1", ".png"])), self.vis_dicts[-2]["image"][:,:,::-1])
                cv2.imwrite(pjoin(self._output_dir, "corr_" + str(SparseViewsVisualizer.eval_iter), "_".join([self.file_names[-2], self.file_names[-1], "2", ".png"])), self.vis_dicts[-1]["image"][:,:,::-1])

            camera_K = get_intrinsics(self._dataset_name, h = image.shape[0], w = image.shape[1]).cpu().numpy()
            focal_x, focal_y, offset_x, offset_y = camera_K[0][0]
            camera_K = np.array([[focal_x, 0, offset_x], [0, focal_y, offset_y], [0, 0, 1]])

            

            if self.vis_3dmesh:
                save_pair_objects(
                                    pred_dict = {"0": self.vis_dicts[-2], "1": self.vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, "mesh_" + str(SparseViewsVisualizer.eval_iter), "_".join([self.file_names[-2],self.file_names[-1], "attn"+str(0), "pred"])),
                                    prefix = "_".join([self.file_names[-2],self.file_names[-1], "attn"+str(0), "pred"]),
                                    pred_camera = pred_camera_dict,
                                    plane_param_override=None,
                                    show_camera = False,
                                    assignment = assignment_matrixs[0],
                                    webvis=False,
                                    save_mesh=True,
                                    camera_K=camera_K,
                                )
            
                save_pair_objects(
                                    pred_dict = {"0": self.gt_vis_dicts[-2], "1": self.gt_vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, "mesh_" + str(SparseViewsVisualizer.eval_iter), "_".join([self.file_names[-2],self.file_names[-1], "gt"]),),
                                    prefix = "_".join([self.file_names[-2],self.file_names[-1], "gt"]),
                                    pred_camera = gt_camera_dict,
                                    plane_param_override=None,
                                    show_camera = False,
                                    assignment = gt_assignment_matrix,
                                    webvis=False,
                                    save_mesh=True,
                                    camera_K=camera_K,
                                )
                
                
            
            if self.vis_frustum:
                save_pair_objects(
                                    pred_dict = {"0": self.vis_dicts[-2], "1": self.vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, "mesh_" + str(SparseViewsVisualizer.eval_iter), "_".join([self.file_names[-2],self.file_names[-1], "attn"+str(0), "predfrustum"])),
                                    prefix = "_".join([self.file_names[-2],self.file_names[-1], "attn"+str(0), "predfrustum"]),
                                    pred_camera = pred_camera_dict,
                                    plane_param_override=None,
                                    show_camera = True,
                                    assignment = assignment_matrixs[0],
                                    webvis=False,
                                    save_mesh=False,
                                    camera_K=camera_K,
                                )

                save_pair_objects(
                                    pred_dict = {"0": self.gt_vis_dicts[-2], "1": self.gt_vis_dicts[-1]},
                                    output_dir = pjoin(self._output_dir, "mesh_" + str(SparseViewsVisualizer.eval_iter), "_".join([self.file_names[-2],self.file_names[-1], "gtfrustum"])),
                                    prefix = "_".join([self.file_names[-2],self.file_names[-1], "gtfrustum"]),
                                    pred_camera = gt_camera_dict,
                                    plane_param_override=None,
                                    show_camera = True,
                                    assignment = gt_assignment_matrix,
                                    webvis=False,
                                    save_mesh=False,
                                    camera_K=camera_K,
                                )
                
        
        # !
        SparseViewsVisualizer.vis_iter += 1

    def evaluate(self):

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        res = {}
                
        results = OrderedDict({"vis": res})
        self._logger.info(results)

        return results
