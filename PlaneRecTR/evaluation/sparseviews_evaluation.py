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


_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

from ..utils.disp import (
    visualizationBatch, 
    plot_depth_recall_curve,
    plot_normal_recall_curve,
    plot_offset_recall_curve,
)
from ..utils.misc import get_coordinate_map, save_dict, save_list_to_file
from ..utils.metrics import (
    evaluateMasks, 
    eval_plane_recall_depth, 
    eval_plane_recall_normal,
    eval_plane_recall_offset,
)
from ..utils.metrics_de import evaluateDepths

from ..utils.metrics_onlyparams import eval_plane_bestmatch_normal_offset

from ..utils.metrics_pose import eval_camera


class SparseViewsEvaluator(DatasetEvaluator):
    """
    Evaluate plane segmentation metrics.
    """
    eval_iter = 0

    def __init__(
        self,
        dataset_name,
        output_dir=None,
        *,
        num_planes=None,
        vis = False,
        vis_period = 50,
        eval_period = 500,
        image_size = (480, 640),
        predict_poses = True,
        predict_inverse_poses = True,
        # outputdir_for_nopesac = None,
        save_output = False,
    ):
        self._logger = logging.getLogger(__name__)
        if num_planes is not None:
            self._logger.warn(
                "SparseViewsEvaluator(num_planes) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._num_planes = num_planes
        self._num_queries = num_planes + 1 if "npr" in dataset_name else num_planes # TODO: add npr
        self.vis = vis
        self.vis_period = vis_period
        self.eval_period = eval_period

        self.k_inv_dot_xy1 = get_coordinate_map(dataset_name, self._cpu_device,
                                                h = image_size[0], w = image_size[1]).numpy()
        self.predict_poses = predict_poses
        self.predict_inverse_poses = predict_inverse_poses
        # self.outputdir_for_nopesac = outputdir_for_nopesac
        self.save_output = save_output

    def reset(self):
        
        self.RI_VI_SC = []
        self.pixelDepth_recall_curve_of_GTpd = np.zeros((13))
        self.planeDepth_recall_curve_of_GTpd = np.zeros((13, 3))
    
        self.pixelNorm_recall_curve = np.zeros((13))
        self.planeNorm_recall_curve = np.zeros((13, 3))

        self.pixelOff_recall_curve = np.zeros((13))
        self.planeOff_recall_curve = np.zeros((13, 3))

        self.bestmatch_normal_errors = []
        self.bestmatch_offset_errors = []

        self.pred_poses_list = []
        self.gt_poses_list = []
        self.pose_eval_file_names = []

        self.pred_inv_poses_list = []

        if self.save_output:
            self._predictions = {}        

        if self.vis:
            self.vis_dicts = []
            self.gt_vis_dicts = []
            self.file_names = []

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

            # single image evaluation
            prediction = {"0":{}, "1":{}}

            for i in range(2):
                sem_seg = output[str(i)]["sem_seg"].argmax(dim=0).to(self._cpu_device) # torch.Size([480, 640]) # sem_seg 21, 192, 256
                pred = np.array(sem_seg, dtype=np.int) # (480, 640)
                plane_depth = output[str(i)]["planes_depth"].to(self._cpu_device).numpy()
                # seg_depth = output[str(i)]["seg_depth"].to(self._cpu_device).numpy()
                valid_params = output[str(i)]["valid_params"].to(self._cpu_device)

                if self.save_output:
                    prediction[str(i)]["image_id"] = input[str(i)]["image_id"]
                    prediction[str(i)]["valid_params"] = valid_params.numpy()
                    prediction[str(i)]["sem_seg"] = pred
                    prediction[str(i)]["label_mask"] = output[str(i)]["label_mask"].to(self._cpu_device).numpy()
                    prediction[str(i)]["valid_scores"] = output[str(i)]["valid_scores"].to(self._cpu_device).numpy()

                # if self.outputdir_for_nopesac is not None:
                #     valid_query_embs = output[str(i)]["valid_query_embs"].to(self._cpu_device).numpy()
                #     save_results = {"valid_query_embs": valid_query_embs, "valid_params": valid_params}
                #     save_dict(save_results, folder=self.outputdir_for_nopesac, prefix=input[str(i)]["image_id"])


                # if self._dataset_name=="scannetv1_plane" or self._dataset_name=="nyuv2_plane":
                #     gt_filename = input["npz_file_name"]
                #     npz_data = np.load(gt_filename)
                #     gt = npz_data["segmentation"]

                #     gt_plane_depth = npz_data["depth"][0] # b#??, h, w
                #     gt_raw_depth = npz_data["raw_depth"]
                #     gt_params = npz_data["plane"]  
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

                if self.vis:
                    # if self._dataset_name=="scannetv1_plane" or self._dataset_name=="nyuv2_plane":
                    #     image = npz_data["image"] #BGR
                    #     file_name = os.path.split(input["npz_file_name"])[-1].split(".")[0]
                        
                    if "sparseviews" in self._dataset_name:
                        image = input[str(i)]["image"].to(self._cpu_device).numpy().transpose((1,2,0)) # (h,w,3)
                        # file_name = "_".join(input[str(i)]["file_name"].split("/")[6:]).split(".")[0]
                        # if "7scenes" in self._dataset_name:
                        file_name = input[str(i)]["image_id"]


                    self.gt_vis_dicts.append({
                        'image': image[:,:,::-1], # (h, w, 3)
                        'segmentation': gt,
                        'depth_GTplane': gt_plane_depth,
                        'depth_GTori': gt_ori_depth,
                        'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                        })
                    self.vis_dicts.append({
                        'image': image[:,:,::-1],
                        'segmentation': pred,
                        'depth_predplane': plane_depth,
                        'K_inv_dot_xy_1': self.k_inv_dot_xy1,
                    })
                    self.file_names.append(file_name)

                self.RI_VI_SC.append(evaluateMasks(pred, gt, device = "cuda",  pred_non_plane_idx = self._num_planes+1, gt_non_plane_idx=self._num_planes, printInfo=False))

                # ----------------------------------------------------- evaluation
                # 1 evaluation: plane segmentation
                valid_plane_num = len(valid_params)
                pixelStatistics, planeStatistics = eval_plane_recall_depth(
                    pred, gt, plane_depth, gt_plane_depth, valid_plane_num, self._num_queries)
                self.pixelDepth_recall_curve_of_GTpd += np.array(pixelStatistics)
                self.planeDepth_recall_curve_of_GTpd += np.array(planeStatistics)

                # 2 evaluation: plane segmentation
                instance_param = valid_params.cpu().numpy()
                plane_recall, pixel_recall = eval_plane_recall_normal(pred, gt,
                                            instance_param, gt_params,
                                            )
                self.pixelNorm_recall_curve += pixel_recall
                self.planeNorm_recall_curve += plane_recall


                # 3 evaluation: plane offset
                instance_param = valid_params.cpu().numpy()
                plane_recall, pixel_recall = eval_plane_recall_offset(pred, gt,
                                            instance_param, gt_params,
                                            )
                self.pixelOff_recall_curve += pixel_recall
                self.planeOff_recall_curve += plane_recall

                instance_param = valid_params.numpy()
                normal_error, offset_error = eval_plane_bestmatch_normal_offset(instance_param, gt_params)
                self.bestmatch_normal_errors.append(normal_error)
                self.bestmatch_offset_errors.append(offset_error)

            if self.predict_poses:
                pred_poses = output["pred_poses"] # SE3 [2,6]?
                rel_pose = input['rel_pose']
                w,x,y,z = rel_pose['rotation']
                rel_pose = np.array(rel_pose['position']+[x,y,z,w])
                base_pose = np.array([0,0,0,0,0,0,1])
                gt_poses = SE3(torch.from_numpy(np.vstack([base_pose, rel_pose]))) # ? SE3[2,6]

                if self.save_output:
                    ptx,pty,ptz,px,py,pz,pw = pred_poses.data.cpu().numpy()[1]
                    prediction["pred_pose"] = {
                        "rotation": np.array([pw, px, py, pz]),
                        "position": np.array([ptx, pty, ptz])
                    }

                    prediction["attn_fundamentals"] = output["attn_fundamentals"].to(self._cpu_device).numpy()
                    prediction["valid_assignment_matrixs"] = output["valid_assignment_matrixs"].to(self._cpu_device).numpy()
                    prediction["assignment_matrixs"] = output["assignment_matrixs"].to(self._cpu_device).numpy()
                    prediction["pred2gt_indices"] = output["pred2gt_indices"]
                    
                    prediction["query1"] = output["query1"].to(self._cpu_device).numpy()
                    prediction["key2"] = output["key2"].to(self._cpu_device).numpy()


                self.pred_poses_list.append(pred_poses)
                self.gt_poses_list.append(gt_poses)
                # self.pose_eval_file_names.append("_".join(input['0']["file_name"].split("/")[6:]).split(".")[0] + "_" + "_".join(input['1']["file_name"].split("/")[6:]).split(".")[0])
                self.pose_eval_file_names.append(input['0']["image_id"] + "_" + input['1']["image_id"])
            
            else:
                if self.save_output:
                    prediction["singleview_attn"] = output["singleview_attn"].to(self._cpu_device).numpy()  # (1, 30, 30)
                    prediction["pred2gt_indices"] = output["pred2gt_indices"]

                    prediction["query1"] = output["query1"].to(self._cpu_device).numpy()
                    prediction["key2"] = output["key2"].to(self._cpu_device).numpy()

            if self.predict_inverse_poses:
                pred_inv_poses = output["pred_inverse_poses"]
                self.pred_inv_poses_list.append(pred_inv_poses)

            if self.save_output:
                
                # prediction["attn_fundamentals"] = output["attn_fundamentals"].to(self._cpu_device).numpy()
                # prediction["valid_assignment_matrixs"] = output["valid_assignment_matrixs"].to(self._cpu_device).numpy()
                # prediction["assignment_matrixs"] = output["assignment_matrixs"].to(self._cpu_device).numpy()

                self._predictions[input['0']['image_id'] + "__" + input['1']['image_id']] = prediction

    def evaluate(self):

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        SparseViewsEvaluator.eval_iter += self.eval_period
        res = {}
        
        # print("len(RI_VI_SC)",len(self.RI_VI_SC))
        res_RI_VI_SC = np.sum(self.RI_VI_SC, axis = 0)/len(self.RI_VI_SC)
        res["RI"] = res_RI_VI_SC[0]
        res["VI"] = res_RI_VI_SC[1]
        res["SC"] = res_RI_VI_SC[2]
        if "nyuv2_plane" in self._dataset_name: # rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3
            res_depth_estimation_metrics = self.depth_estimation_metrics/len(self.RI_VI_SC)
            res["DE_rel"], res["DE_rel_sqr"], res["DE_log10"], res["DE_rmse"], \
            res["DE_rmse_log"], res["DE_accuracy_1"], res["DE_accuracy_2"], res["DE_accuracy_3"] = res_depth_estimation_metrics
        
        
        
        if self._output_dir:
            
            file_path = pjoin(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        if self.save_output:
            save_file_path = pjoin(self._output_dir, "PlaneRecTRpp_outputs.pkl")
            # with PathManager.open(save_file_path, "wb") as f:
            #     torch.save(self._predictions, f)
            save_dict(self._predictions, self._output_dir, "PlaneRecTRpp_outputs")
            

    
        vis_path = pjoin(self._output_dir, "vis_" + str(SparseViewsEvaluator.eval_iter))

        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        if self.vis: 
                
            for i in range(len(self.vis_dicts)):
                
                if i % self.vis_period == 0:
                    
                    visualizationBatch(root_path = vis_path, idx = self.file_names[i], info = "gt",
                    data_dict = self.gt_vis_dicts[i], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                    save_depth = True, save_ply = True, save_cloud = False) 
                    visualizationBatch(root_path = vis_path, idx = self.file_names[i], info = "pred",
                    data_dict = self.vis_dicts[i], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                    save_depth = True, save_ply = True, save_cloud = False)

        recall_curve_save_path = pjoin(vis_path, "recall_curve")
        if not os.path.exists(recall_curve_save_path):
            os.makedirs(recall_curve_save_path)
        
        mine_recalls_pixel = {"PlaneRecTR (Ours)": self.pixelDepth_recall_curve_of_GTpd / len(self.RI_VI_SC) * 100}
        mine_recalls_plane = {"PlaneRecTR (Ours)": self.planeDepth_recall_curve_of_GTpd[:, 0] / self.planeDepth_recall_curve_of_GTpd[:, 1] * 100}
        res['per_pixel_depth_01'] = mine_recalls_pixel["PlaneRecTR (Ours)"][2]
        res['per_pixel_depth_06'] = mine_recalls_pixel["PlaneRecTR (Ours)"][-1]
        res['per_plane_depth_01'] = mine_recalls_plane["PlaneRecTR (Ours)"][2]
        res['per_plane_depth_06'] = mine_recalls_plane["PlaneRecTR (Ours)"][-1]
        # # print("mine_recalls_pixel (pred_planed vs gt_planed)", mine_recalls_pixel)
        # # print("mine_recalls_plane (pred_planed vs gt_planed)", mine_recalls_plane)
        # plot_depth_recall_curve(mine_recalls_pixel, type='pixel (pred_planed vs gt_planed)', save_path=recall_curve_save_path)
        # plot_depth_recall_curve(mine_recalls_plane, type='plane (pred_planed vs gt_planed)', save_path=recall_curve_save_path)

        normal_recalls_pixel = {"PlaneRecTR": self.pixelNorm_recall_curve / len(self.RI_VI_SC) * 100}
        normal_recalls_plane = {"PlaneRecTR": self.planeNorm_recall_curve[:, 0] / self.planeNorm_recall_curve[:, 1] * 100}
        res['per_pixel_normal_5'] = normal_recalls_pixel["PlaneRecTR"][2]
        res['per_pixel_normal_30'] = normal_recalls_pixel["PlaneRecTR"][-1]
        res['per_plane_normal_5'] = normal_recalls_plane["PlaneRecTR"][2]
        res['per_plane_normal_30'] = normal_recalls_plane["PlaneRecTR"][-1]
        # # print("normal_recalls_pixel", normal_recalls_pixel)
        # # print("normal_recalls_plane", normal_recalls_plane)
        # plot_normal_recall_curve(normal_recalls_pixel, type='pixel', save_path=recall_curve_save_path)
        # plot_normal_recall_curve(normal_recalls_plane, type='plane', save_path=recall_curve_save_path)

        offset_recalls_pixel = {"PlaneRecTR": self.pixelOff_recall_curve / len(self.RI_VI_SC) * 100}
        offset_recalls_plane = {"PlaneRecTR": self.planeOff_recall_curve[:, 0] / self.planeOff_recall_curve[:, 1] * 100}
        res['per_pixel_offset_17'] = offset_recalls_pixel["PlaneRecTR"][2]
        res['per_pixel_offset_100'] = offset_recalls_pixel["PlaneRecTR"][-1]
        res['per_plane_offset_17'] = offset_recalls_plane["PlaneRecTR"][2]
        res['per_plane_offset_100'] = offset_recalls_plane["PlaneRecTR"][-1]
        # # print("offset_recalls_pixel", offset_recalls_pixel)
        # # print("offset_recalls_plane", offset_recalls_plane)
        # plot_offset_recall_curve(offset_recalls_pixel, type='pixel', save_path=recall_curve_save_path)
        # plot_offset_recall_curve(offset_recalls_plane, type='plane', save_path=recall_curve_save_path)

        res["normal_error"] = np.mean(self.bestmatch_normal_errors)
        res["offset_error"] = np.mean(self.bestmatch_offset_errors)
        # print("bestmatch_normal_errors", res["normal_error"])
        # print("bestmatch_offset_errors", res["offset_error"])

        if self.predict_poses:
            vis_pose_path = pjoin(self._output_dir, "vis_" + str(SparseViewsEvaluator.eval_iter), "camera_pose")

            if not os.path.exists(vis_pose_path):
                os.makedirs(vis_pose_path)

            acc_threshold = {"tran": [1.0, 0.5, 0.2], "rot": [30, 15, 10],}
            camera_metrics = eval_camera(pred_poses = lietorch.stack(self.pred_poses_list, dim = 0), 
                                         gt_poses = lietorch.stack(self.gt_poses_list, dim = 0),
                                         pair_names = self.pose_eval_file_names,
                                         acc_threshold = acc_threshold,
                                         using_sub = False,
                                         save_path = vis_pose_path,
                                         )
            
            torch.save(lietorch.stack(self.pred_poses_list, dim = 0), pjoin(vis_pose_path, "pred_poses.pt"))
            torch.save(lietorch.stack(self.gt_poses_list, dim = 0), pjoin(vis_pose_path, "gt_poses.pt"))
            save_list_to_file(self.pose_eval_file_names, pjoin(vis_pose_path, "pair_names.txt"))
            
            res[f"top1 T err < {acc_threshold['tran'][0]}"] = camera_metrics[f"top1 T err < {acc_threshold['tran'][0]}"]
            res[f"top1 R err < {acc_threshold['rot'][0]}"] = camera_metrics[f"top1 R err < {acc_threshold['rot'][0]}"]
            res[f"top1 T err < {acc_threshold['tran'][1]}"] = camera_metrics[f"top1 T err < {acc_threshold['tran'][1]}"]
            res[f"top1 R err < {acc_threshold['rot'][1]}"] = camera_metrics[f"top1 R err < {acc_threshold['rot'][1]}"]
            res[f"top1 T err < {acc_threshold['tran'][2]}"] = camera_metrics[f"top1 T err < {acc_threshold['tran'][2]}"]
            res[f"top1 R err < {acc_threshold['rot'][2]}"] = camera_metrics[f"top1 R err < {acc_threshold['rot'][2]}"]
            res[f"T mean err"] = camera_metrics[f"T mean err"]
            res[f"R mean err"] = camera_metrics[f"R mean err"]
            res[f"T median err"] = camera_metrics[f"T median err"]
            res[f"R median err"] = camera_metrics[f"R median err"]

        if self.predict_inverse_poses:
            vis_inv_pose_path = pjoin(self._output_dir, "vis_" + str(SparseViewsEvaluator.eval_iter), "inv_camera_pose")

            if not os.path.exists(vis_inv_pose_path):
                os.makedirs(vis_inv_pose_path)

            acc_threshold = {"tran": [1.0, 0.5, 0.2], "rot": [30, 15, 10],}
            camera_metrics = eval_camera(pred_poses = lietorch.stack(self.pred_inv_poses_list, dim = 0), 
                                         gt_poses = lietorch.stack(self.gt_poses_list, dim = 0),
                                         pair_names = self.pose_eval_file_names,
                                         acc_threshold = acc_threshold,
                                         using_sub = True,
                                         save_path = vis_inv_pose_path,
                                         )
            
            torch.save(lietorch.stack(self.pred_inv_poses_list, dim = 0), pjoin(vis_inv_pose_path, "pred_inv_poses.pt"))
            # torch.save(lietorch.stack(self.gt_poses_list, dim = 0), pjoin(vis_inv_pose_path, "gt_poses.pt"))
            # save_list_to_file(self.pose_eval_file_names, pjoin(vis_inv_pose_path, "pair_names.txt"))
            
            res[f"[INV]top1 T err < {acc_threshold['tran'][0]}"] = camera_metrics[f"top1 T err < {acc_threshold['tran'][0]}"]
            res[f"[INV]top1 R err < {acc_threshold['rot'][0]}"] = camera_metrics[f"top1 R err < {acc_threshold['rot'][0]}"]
            res[f"[INV]top1 T err < {acc_threshold['tran'][1]}"] = camera_metrics[f"top1 T err < {acc_threshold['tran'][1]}"]
            res[f"[INV]top1 R err < {acc_threshold['rot'][1]}"] = camera_metrics[f"top1 R err < {acc_threshold['rot'][1]}"]
            res[f"[INV]top1 T err < {acc_threshold['tran'][2]}"] = camera_metrics[f"top1 T err < {acc_threshold['tran'][2]}"]
            res[f"[INV]top1 R err < {acc_threshold['rot'][2]}"] = camera_metrics[f"top1 R err < {acc_threshold['rot'][2]}"]
            res[f"[INV]T mean err"] = camera_metrics[f"T mean err"]
            res[f"[INV]R mean err"] = camera_metrics[f"R mean err"]
            res[f"[INV]T median err"] = camera_metrics[f"T median err"]
            res[f"[INV]R median err"] = camera_metrics[f"R median err"]
                
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)

        return results
