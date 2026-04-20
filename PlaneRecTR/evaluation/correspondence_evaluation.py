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
from detectron2.utils.logger import create_small_table

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

from ..utils.metrics_pose import compute_IPAA, evaluate_matching_from_iou

from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns


def draw_heatmap(corr_matrix, save_path):
    # fig = sns.heatmap(corr_matrix, cmap='RdBu', xticklabels=1, yticklabels=1)
    fig = sns.heatmap(corr_matrix, cmap='YlGnBu', xticklabels=1, yticklabels=1)
    heatmap = fig.get_figure()
    heatmap.savefig(save_path, dpi = 400)
    plt.close()


def tsne_vis(x, y, fig_name, n_components=2):
    tsne = TSNE(n_components=n_components)
    result = tsne.fit_transform(x)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    result = scaler.fit_transform(result)

    if n_components==2:
        plt.figure(figsize=(20, 20))
        plt.scatter(result[:,0], result[:,1], c=y, s=500)
    elif n_components==3:
        fig = plt.figure(figsize=(20, 20))
        ax = Axes3D(fig)
        ax.scatter(result[:,0], result[:,1], result[:,2], c=y, s=500)
    plt.savefig(fig_name + "_n=" + str(n_components) +".png", dpi=300)



class CorrEvaluator(DatasetEvaluator):
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
        predict_corrs = False,
    ):
        self._logger = logging.getLogger(__name__)
        if num_planes is not None:
            self._logger.warn(
                "CorrEvaluator(num_planes) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._num_planes = num_planes
        self._num_queries = num_planes + 1 if "npr" in dataset_name else num_planes # TODO: add npr
        self.vis = vis
        self.vis_period = vis_period
        self.eval_period = eval_period
        self.predict_corrs = predict_corrs

    def reset(self):
        
        self.gt_corrs = []
        self.IPAA_dict_list = []
        for i in range(9):
            IPAA_dict = {}
            for i in range(11):
                IPAA_dict[i * 10] = 0
            self.IPAA_dict_list.append(IPAA_dict)

        # matching from iou < modified by Nope-SAC
        self.match_statistics = {}
        key_list = ["attn_corr0", "attn_corr1",
                    #  "final_corr0", "final_corr1", "attn_combine_corr", "final_combine_corr"
                     ]
        for key in key_list:
            self.match_statistics[key] = {
                "all_correct_num": 0,
                "all_matched_num": 0
            }
        self.all_gt_corr_num = 0

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        
        self.fig_path = pjoin(self._output_dir, 'attn_fundamentals_heatmap/')

        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)


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
            gt_corrs = np.array(input["gt_corrs"])
            fig_name = self.fig_path
            attn_fundamentals = output["attn_fundamentals"].to(self._cpu_device).numpy() #[8,30,30]
            if self.predict_corrs:
                attn_corr = output["pred_corrs"].to(self._cpu_device).numpy()
            assignment_matrixs = output["assignment_matrixs"].to(self._cpu_device).numpy()
            indice = output["pred2gt_indices"]
            # single image evaluation
            gt_pred_ind = {"0":{}, "1":{}}
            for i in range(2):
                fig_name += "_".join(input[str(i)]["file_name"].split("/")[6:]).split(".")[0]
                pred_ind, gt_ind = indice[str(i)]
                # pred_ind = pred_ind.to(self._cpu_device).numpy()
                # gt_ind = gt_ind.to(self._cpu_device).numpy()
                for pred_i, gt_i in zip(pred_ind, gt_ind):
                    gt_pred_ind[str(i)][gt_i] = pred_i

            predorder_gt_corrs = [] 
            for gt_corr in gt_corrs:
                gt_i0, gt_i1 = gt_corr
                pred_i0, pred_i1 = gt_pred_ind["0"][gt_i0], gt_pred_ind["1"][gt_i1]
                predorder_gt_corrs.append([pred_i0, pred_i1])
            predorder_gt_corrs = np.array(predorder_gt_corrs)
            
            predorder_gt_corr_matrx = np.zeros((attn_fundamentals.shape[-2], attn_fundamentals.shape[-1]))
            predorder_gt_corr_matrx[(predorder_gt_corrs[:,0], predorder_gt_corrs[:,1])] = 1

            self.gt_corrs.append(predorder_gt_corr_matrx)
          
            # update IPAA:
            for i in range(len(assignment_matrixs)):
                compute_IPAA(assignment_matrixs[i], predorder_gt_corrs, self.IPAA_dict_list[i])
            


            valid_assignment_matrixs = output["valid_assignment_matrixs"].to(self._cpu_device).numpy()
            pred_assignment_matrixs = valid_assignment_matrixs[[0,4]] if len(valid_assignment_matrixs)>2 else valid_assignment_matrixs[[0,1]]
            # pred_assignment_matrixs = valid_assignment_matrixs[[0,1]]
            pred_segmentations = {}
            gt_seg_masks = {}
            pred_plane_num = {}
            for img_idx in range(2):
                sem_seg = output[str(img_idx)]["sem_seg"].argmax(dim=0).to(self._cpu_device) # torch.Size([480, 640]) # sem_seg 21, 192, 256
                pred = np.array(sem_seg, dtype=np.int) # (480, 640)
                num = output[str(img_idx)]["valid_params"].shape[0]

                pred_segmentations[str(img_idx)] = pred
                pred_plane_num[str(img_idx)] = num

                if "sparseviews" in self._dataset_name:
                    gt_plane_masks = input[str(img_idx)]["plane_masks"].tensor.to(self._cpu_device).numpy()

                    # gt = np.ones_like(gt_plane_masks[0])*self._num_queries
                    # for idx,gpm in enumerate(gt_plane_masks):
                    #     gt[gpm==1]=idx
                    gt_seg_masks[str(img_idx)] = gt_plane_masks

            self.all_gt_corr_num += len(gt_corrs)
            evaluate_matching_from_iou(self.match_statistics, pred_segmentations, gt_seg_masks, pred_assignment_matrixs,  gt_corrs, pred_plane_num, iou_thresh=0.5)
            

       
            # self.attn_fundamentals.append(attn_fundamentals)
            # self.gt_fundmentals.append(predorder_gt_corr_matrx)

    def evaluate(self):

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        CorrEvaluator.eval_iter += self.eval_period
        res = {}
        
        for i in range(len(self.IPAA_dict_list)):
            for key in self.IPAA_dict_list[i].keys():
                self.IPAA_dict_list[i][key] /= len(self.gt_corrs)
                if i < 8:
                    res["IPAA_attn_f"+str(i)+"_"+str(key)] = self.IPAA_dict_list[i][key]
                    # print("IPAA_attn_f"+str(i)+"_"+str(key))
                    # print(self.IPAA_dict_list[i][key])
                else:
                    res["IPAA_attn_c"+"_"+str(key)] = self.IPAA_dict_list[i][key] 
                    # print("IPAA_attn_c"+"_"+str(key))
                    # print(self.IPAA_dict_list[i][key])

        results = OrderedDict({"corr": res})
        self._logger.info(results)

        # all_matching_metrics = {}
        for key in self.match_statistics:
            all_correct_num = self.match_statistics[key]['all_correct_num']
            all_matched_num = self.match_statistics[key]['all_matched_num']
            precision = float(all_correct_num) / float(all_matched_num)
            recall = float(all_correct_num) / float(self.all_gt_corr_num)
            F_score = 2 * precision * recall / (precision + recall)

            matching_metrics = {}
            matching_metrics['precision'] = precision
            matching_metrics['recall'] = recall
            matching_metrics['F-score'] = F_score
            matching_metrics['TP'] = all_correct_num
            matching_metrics['Pred. Num.'] = all_matched_num
            matching_metrics['GT Num.'] = self.all_gt_corr_num
            # self._logger.info("Plane metrics (%s): \n"%(key) + create_small_table(matching_metrics))
            # self._logger.info(OrderedDict({"iou match metrics (%s)"%(key): matching_metrics}))
            # all_matching_metrics[key] = matching_metrics
            match_results = OrderedDict({"iou match metrics (%s)"%(key): matching_metrics})
            results.update(match_results)
            self._logger.info("iou match metrics (%s): \n"%(key) + create_small_table(matching_metrics))
        
        

        return results

        


