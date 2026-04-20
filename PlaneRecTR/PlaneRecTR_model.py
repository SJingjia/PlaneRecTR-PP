# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by https://github.com/facebookresearch/Mask2Former
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

from .utils.misc import get_coordinate_map, get_intrinsics, save_dict
from .modeling.camera_decoder.camera_vit_ess import build_camera_decoder

import numpy as np
from lietorch import SE3
# import time

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


@META_ARCH_REGISTRY.register()
class PlaneRecTR(nn.Module):
    """
    Main class for plane segmentation and reconstruction architectures.
    """

    @configurable
    def __init__(        
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        camera_decoder,
        criterion: nn.Module,
        num_queries: int,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        k_inv_dot_xy1,
        # intrinsics,
        predict_param: bool,
        predict_depth: bool,
        predict_center: bool,
        predict_poses: bool,
        predict_corrs: bool,
        predict_inverse_poses: bool,
        predict_joint_corr_pose_param: bool,
        predict_joint_gtcorr_pose_gtparam: bool,
        predict_joint_01corr_pose_paramdetach: bool,
        filter_cam_planes: bool,
        filter_corr_planes: bool,
        data_pair2single: bool,
        data_multiviews: bool,
        match_threshold: float,
        plane_mask_threshold: float,
        plane_score_threshold: float,
        mask_prob_threshold: float,
        overlap_threshold: float,
        mask_embeddings: bool,
        mask2center: bool,
        # outputdir_for_results,
        matcher: nn.Module,
        # Gs,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            k_inv_dot_xy1:
            predict_param: bool,
            predict_depth: bool,
            predict_poses: bool,
            plane_mask_threshold: float, 
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.camera_decoder = camera_decoder
        self.criterion = criterion
        self.num_queries = num_queries
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.k_inv_dot_xy1 = k_inv_dot_xy1
        # self.intrinsics = intrinsics
        
        self.predict_param = predict_param
        self.predict_depth = predict_depth
        self.predict_center = predict_center
        self.predict_poses = predict_poses
        self.predict_corrs = predict_corrs
        self.predict_inverse_poses = predict_inverse_poses
        self.predict_joint_corr_pose_param = predict_joint_corr_pose_param
        self.predict_joint_gtcorr_pose_gtparam = predict_joint_gtcorr_pose_gtparam
        self.predict_joint_01corr_pose_paramdetach = predict_joint_01corr_pose_paramdetach
        self.filter_cam_planes = filter_cam_planes
        self.filter_corr_planes = filter_corr_planes
        self.data_pair2single = data_pair2single
        self.data_multiviews = data_multiviews
        self.match_threshold = match_threshold
        self.plane_mask_threshold = plane_mask_threshold
        self.plane_score_threshold = plane_score_threshold
        self.mask_prob_threshold = mask_prob_threshold
        self.overlap_threshold = overlap_threshold
        self.mask_embeddings = mask_embeddings
        self.mask2center = mask2center
        # self.outputdir_for_results = outputdir_for_results
        # ! add for checking query embeddings' similarity
        self.matcher = matcher

        # self.Gs = Gs

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
            # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        param_l1_weight = cfg.MODEL.MASK_FORMER.PARAM_L1_WEIGHT
        param_cos_weight = cfg.MODEL.MASK_FORMER.PARAM_COS_WEIGHT
        q_weight = cfg.MODEL.MASK_FORMER.Q_WEIGHT
        center_weight = cfg.MODEL.MASK_FORMER.CENTER_WEIGHT
        plane_depths_weight = cfg.MODEL.MASK_FORMER.PLANE_DEPTHS_WEIGHT
        whole_depth_weight = cfg.MODEL.MASK_FORMER.WHOLE_DEPTH_WEIGHT

        pose_tr_weight = cfg.MODEL.CAMERA_MODULE.POSE_TR_WEIGHT
        pose_rot_weight = cfg.MODEL.CAMERA_MODULE.POSE_ROT_WEIGHT

        corrs_weight = cfg.MODEL.CAMERA_MODULE.CORRS_WEIGHT

        inverse_pose_tr_weight = cfg.MODEL.CAMERA_MODULE.INVERSE_POSE_TR_WEIGHT
        inverse_pose_rot_weight = cfg.MODEL.CAMERA_MODULE.INVERSE_POSE_ROT_WEIGHT

        joint_corr_pose_param_l1_weight = cfg.MODEL.CAMERA_MODULE.JOINT_CORR_POSE_PARAM_L1_WEIGHT
        joint_corr_pose_param_cos_weight = cfg.MODEL.CAMERA_MODULE.JOINT_CORR_POSE_PARAM_COS_WEIGHT
        joint_gtcorr_pose_gtparam_l1_weight = cfg.MODEL.CAMERA_MODULE.JOINT_GTCORR_POSE_GTPARAM_L1_WEIGHT
        joint_gtcorr_pose_gtparam_cos_weight = cfg.MODEL.CAMERA_MODULE.JOINT_GTCORR_POSE_GTPARAM_COS_WEIGHT
        joint_01corr_pose_paramdetach_offset_weight = cfg.MODEL.CAMERA_MODULE.JOINT_01CORR_POSE_PARAMDETACH_OFFSET_WEIGHT
        joint_01corr_pose_paramdetach_cos_weight = cfg.MODEL.CAMERA_MODULE.JOINT_01CORR_POSE_PARAMDETACH_COS_WEIGHT
        # predict bool
        predict_param = cfg.MODEL.MASK_FORMER.PREDICT_PARAM
        predict_depth = cfg.MODEL.MASK_FORMER.PREDICT_DEPTH
        predict_center = cfg.MODEL.MASK_FORMER.PREDICT_CENTER
        predict_poses = cfg.MODEL.CAMERA_MODULE.PREDICT_POSES
        predict_corrs = cfg.MODEL.CAMERA_MODULE.PREDICT_CORRS
        predict_inverse_poses = cfg.MODEL.CAMERA_MODULE.PREDICT_INVERSE_POSES
        predict_joint_corr_pose_param = cfg.MODEL.CAMERA_MODULE.PREDICT_JOINT_CORR_POSE_PARAM
        predict_joint_gtcorr_pose_gtparam = cfg.MODEL.CAMERA_MODULE.PREDICT_JOINT_GTCORR_POSE_GTPARAM
        predict_joint_01corr_pose_paramdetach = cfg.MODEL.CAMERA_MODULE.PREDICT_JOINT_01CORR_POSE_PARAMDETACH
        filter_cam_planes = cfg.MODEL.CAMERA_MODULE.FILTER_CAM_PLANES
        filter_corr_planes = cfg.MODEL.CAMERA_MODULE.TEST.FILTER_CORR_PLANES
        match_threshold = cfg.MODEL.CAMERA_MODULE.TEST.MATCH_THRESHOLD
        

        if predict_poses:
            camera_decoder = build_camera_decoder(cfg)
        else:
            camera_decoder = None

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_param = param_l1_weight,  
            cost_depth = plane_depths_weight,
            cost_center = center_weight,
            predict_param = predict_param,
            predict_depth = predict_depth,
            predict_center = predict_center,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                        "loss_param_l1": param_l1_weight, "loss_param_cos": param_cos_weight,
                        "loss_Q": q_weight, "loss_center": center_weight, 
                        "loss_plane_depths": plane_depths_weight, "loss_whole_depth": whole_depth_weight,
                        "loss_pose_tr": pose_tr_weight, "loss_pose_rot": pose_rot_weight,
                        "loss_corrs": corrs_weight,
                        "loss_inverse_pose_tr": inverse_pose_tr_weight, 
                        "loss_inverse_pose_rot": inverse_pose_rot_weight,
                        "loss_joint_corr_pose_param_l1": joint_corr_pose_param_l1_weight,
                        "loss_joint_corr_pose_param_cos": joint_corr_pose_param_cos_weight,
                        "loss_joint_gtcorr_pose_gtparam_l1": joint_gtcorr_pose_gtparam_l1_weight,
                        "loss_joint_gtcorr_pose_gtparam_cos": joint_gtcorr_pose_gtparam_cos_weight,
                        "loss_joint_01corr_pose_paramdetach_offset": joint_01corr_pose_paramdetach_offset_weight,
                        "loss_joint_01corr_pose_paramdetach_cos": joint_01corr_pose_paramdetach_cos_weight,
                        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = [
                    "labels", 
                    "masks",
                    'plane_depths',
                ]

        if predict_param:
            losses.extend([
                'params', 
                'Q'
            ])
          
        if predict_depth:
            losses.extend([
                'plane_depths',
            ])

        if predict_poses:
            losses.extend([
                'pose', 
                ])
            
        if predict_corrs:
            losses.extend([
                'corrs'
            ])
        
        if predict_center:
            losses.extend([
                'center'
            ])

        if predict_inverse_poses:
            losses.extend([
                'inverse_pose'
            ])
        
        # The useless loss from the previous experiment, to be deleted
        if predict_joint_corr_pose_param:
            losses.extend([
                'joint_corr_pose_param'
            ])
        # The useless loss from the previous experiment, to be deleted
        if predict_joint_gtcorr_pose_gtparam:
            losses.extend([
                'joint_gtcorr_pose_gtparam'
            ])
        # The useless loss from the previous experiment, to be deleted
        if predict_joint_01corr_pose_paramdetach:
            losses.extend([
                'joint_01corr_pose_paramdetach'
            ])

        
        k_inv_dot_xy1 = get_coordinate_map(cfg.INPUT.DATASET_MAPPER_NAME, torch.device("cpu"), 
                                           h = cfg.INPUT.IMAGE_SIZE[0], w = cfg.INPUT.IMAGE_SIZE[1])
        # intrinsics = get_intrinsics(cfg.INPUT.DATASET_MAPPER_NAME)

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            k_inv_dot_xy1 = k_inv_dot_xy1, 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        # base_pose = np.array([0,0,0,0,0,0,1])
        # poses = np.vstack([base_pose, base_pose]).astype(np.float32)
        # poses = torch.from_numpy(poses).unsqueeze(0).cuda() # (1, 2, 7)
        # Gs = SE3(poses)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "camera_decoder": camera_decoder,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN)>0 else None,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "k_inv_dot_xy1": k_inv_dot_xy1,
            # "intrinsics": intrinsics,
            "predict_param": predict_param,
            "predict_depth": predict_depth,
            "predict_center": predict_center,
            "predict_poses": predict_poses,
            "predict_corrs": predict_corrs,
            "predict_inverse_poses": predict_inverse_poses,
            "predict_joint_corr_pose_param": predict_joint_corr_pose_param,
            "predict_joint_gtcorr_pose_gtparam": predict_joint_gtcorr_pose_gtparam,
            "predict_joint_01corr_pose_paramdetach": predict_joint_01corr_pose_paramdetach,
            "filter_cam_planes": filter_cam_planes,
            "filter_corr_planes": filter_corr_planes,
            "data_pair2single": cfg.INPUT.DATA_PAIR2SINGLE,
            "data_multiviews": cfg.INPUT.DATA_MULTIVIEWS,
            "match_threshold": match_threshold,
            "plane_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.PLANE_MASK_THRESHOLD,
            #! from nope-sac
            "plane_score_threshold": cfg.MODEL.MASK_FORMER.TEST.PLANE_SCORE_THRESHOLD,
            "mask_prob_threshold": cfg.MODEL.MASK_FORMER.TEST.MASK_PROB_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "mask_embeddings": cfg.MODEL.CAMERA_MODULE.MASK_EMBEDDINGS,
            "mask2center": cfg.MODEL.CAMERA_MODULE.MASK2CENTER,
            # "outputdir_for_results": cfg.MODEL.MASK_FORMER.TEST.OUTPUTDIR_FOR_RESULTS,
            "matcher": matcher,
            # "Gs": Gs,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        if self.data_pair2single:
            return self.forward_singleview(batched_inputs)
        if self.data_multiviews:
            return self.inference_multiviews(batched_inputs)
        # Common interface 
        return self.forward_sparseviews(batched_inputs)

    def forward_singleview(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs] 
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor) 
        
        outputs = self.sem_seg_head(features)

        if self.training:
            targets = self.prepare_targets_singleview(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict: # {'loss_ce': 2.0, 'loss_mask': 5.0, 'loss_dice': 5.0, 'loss_ce_0': 2.0, 'loss_mask_0': 5.0, 'loss_dice_0': 5.0, 'loss_ce_1': 2.0, 'loss_mask_1': 5.0, 'loss_dice_1': 5.0, 'loss_ce_2': 2.0, 'loss_mask_2': 5.0, 'loss_dice_2': 5.0, 'loss_ce_3': 2.0, 'loss_mask_3': 5.0, ...}
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"] # torch.Size([b, num_queries, num_classes + 1])
            mask_pred_results = outputs["pred_masks"] # torch.Size([b, num_queries, h/4, w/4])
            param_pred_results = outputs["pred_params"] # torch.Size([b, num_queries, 3])
            depth_pred_results = outputs["pred_depths"] # torch.Size([b, num_queries, h, w]
            query_embedding_results = outputs["query_embeddings"]
            # upsample masks
            if not mask_pred_results.shape[-1] == images.tensor.shape[-1]:
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                ) # torch.Size([b,num_queries,h,w])

            if not depth_pred_results.shape[-1] == images.tensor.shape[-1]:
                depth_pred_results = F.interpolate(
                    depth_pred_results,
                    size = (images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode = "bilinear",
                    align_corners=False,
                ) # torch.Size([b,num_queries,h,w])

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, param_pred_result, depth_pred_result, query_embedding_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, param_pred_results, depth_pred_results, query_embedding_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0]) # ep 349
                width = input_per_image.get("width", image_size[1]) # ep 640
                processed_results.append({})
                # Return semantic segmentation predictions in the original resolution.
                # if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                ) 
                mask_cls_result = mask_cls_result.to(mask_pred_result) # torch.Size([num_queries, num_classes, num_classes + 1])

                # plane inference
                if self.semantic_on:
                    
                    plane_seg, inferred_planes_depth, inferred_seg_depth, valid_param, valid_score, label_mask = retry_if_cuda_oom(self.plane_inference)(mask_cls_result, mask_pred_result, param_pred_result, depth_pred_result)
                    processed_results[-1]["sem_seg"] = plane_seg
                    processed_results[-1]["planes_depth"] = inferred_planes_depth
                    # processed_results[-1]["seg_depth"] = inferred_seg_depth
                    processed_results[-1]["valid_params"] = valid_param

                    # if self.outputdir_for_results is not None:
                    #     # processed_results[-1]["valid_query_embs"] = query_embedding_result[label_mask, :] # [valid_plane_num, 256]
                    #     valid_query_emb = query_embedding_result[label_mask, :].detach().cpu().numpy()
                    #     save_results = {"sem_seg":plane_seg, "valid_query_embs": valid_query_emb, "valid_params": valid_param, "label_mask": label_mask}
                    #     save_dict(save_results, folder=self.outputdir_for_results, prefix=input_per_image["image_id"])

            return processed_results
    
    def forward_sparseviews(self, batched_inputs):
        images = {}
        outputs = {}
        # gt_corrs = [len(x["gt_corrs"]) for x in batched_inputs]
        # imgs = [x["0"]["image"].to(self.device) for x in batched_inputs]
        # imgs = ImageList.from_tensors(imgs, self.size_divisibility)
        # features = self.backbone(imgs.tensor)
        # loss = torch.sum(features['res2'])
        # max_corr_plane_num = max(gt_corrs)
        # if max_corr_plane_num > 20:
        #     print("max_corr_plane_num: ", max_corr_plane_num)

        # planes = [len(x["0"]["params"]) for x in batched_inputs] + [len(x["1"]["params"]) for x in batched_inputs]
        # max_plane_num = max(planes)
        # if max_plane_num > 20:
        #     print("max_plane_num: ", max_plane_num)
        # start_time = time.time()

        for i in range(2):
            imgs = [x[str(i)]["image"].to(self.device) for x in batched_inputs] 
            imgs = [(x - self.pixel_mean) / self.pixel_std for x in imgs]
            # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            imgs = ImageList.from_tensors(imgs, self.size_divisibility)
            images[str(i)] = imgs

            features = self.backbone(imgs.tensor)
        
            outputs[str(i)] = self.sem_seg_head(features)

        targets = self.prepare_targets_sparseviews(batched_inputs, images)

        if self.predict_poses:
            Ps = SE3(torch.stack(targets["gt_poses"]))
            Gs = SE3.IdentityLike(Ps)
            Ps_out = SE3(Ps.data.clone())
            
            intrinsics = torch.stack(targets["intrinsics"]) # (B,4)

            if self.mask_embeddings:
                # (b,2,nq,512)
                twoviews_features = torch.stack([
                    torch.cat([outputs["0"]["query_embeddings"], outputs["0"]["mask_embeddings"].detach()], dim=-1), 
                    torch.cat([outputs["1"]["query_embeddings"], outputs["1"]["mask_embeddings"].detach()], dim=-1)
                    ], dim=1)
            else:
                #  (b, 2, num_queries, 256)
                twoviews_features = torch.stack([outputs["0"]["query_embeddings"], 
                                            outputs["1"]["query_embeddings"]], dim=1)

            #  (b, 2, num_queries, 3)
            plane_params = torch.stack([outputs["0"]["pred_params"], 
                                         outputs["1"]["pred_params"]], dim=1)

            plane_centers = None
            if self.predict_center:
                plane_centers = torch.stack([outputs["0"]["pred_centers"], 
                                 outputs["1"]["pred_centers"]], dim=1).detach()
                plane_centers = plane_centers[:,:,:,[1,0]]  # format: (y,x)
            elif self.mask2center:
                plane_centers = self.mask_prob2center([outputs["0"]["pred_masks"], 
                                 outputs["1"]["pred_masks"]])  # format: (y,x)


            outputs["pred_poses"], outputs["pred_inverse_poses"], outputs["attn_fundamentals"], outputs["pred_corrs"], outputs["query1"], outputs["key2"] = \
            self.camera_decoder(features = twoviews_features, Gs = Gs,
                                 label_masks = self.cal_label_masks([outputs["0"]["pred_logits"], 
                                 outputs["1"]["pred_logits"]]) if self.filter_cam_planes else None, 
                                 plane_params = plane_params, 
                                 plane_centers = plane_centers, 
                                 intrinsics = intrinsics, inference=True) # SE3(b, 2, 6)
        else:
            f_dim = outputs["0"]["query_embeddings"].shape[-1]
            outputs["singleview_attns"] = outputs["0"]["query_embeddings"] @ outputs["1"]["query_embeddings"].transpose(-2, -1) * (f_dim ** -0.5) # (b,nq,256) @ (b, 256, nq) -> (b, nq, nq)

        if self.training:
            outputs, targets = self.process_for_sparseviews(outputs, targets)

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict: # {'loss_ce': 2.0, 'loss_mask': 5.0, 'loss_dice': 5.0, 'loss_ce_0': 2.0, 'loss_mask_0': 5.0, 'loss_dice_0': 5.0, 'loss_ce_1': 2.0, 'loss_mask_1': 5.0, 'loss_dice_1': 5.0, 'loss_ce_2': 2.0, 'loss_mask_2': 5.0, 'loss_dice_2': 5.0, 'loss_ce_3': 2.0, 'loss_mask_3': 5.0, ...}
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

        else:
            
            # output_Gs = self.camera_decoder(twoviews_features, self.Gs, self.intrinsics, inference=True) # (2, 7)

            indices = {}
            for i in range(2):
                outputs_without_aux = {k: v for k, v in outputs[str(i)].items() if k != "aux_outputs"} 
                # {'pred_logits': torch.Size([1, 100, 61]), 'pred_masks': torch.Size([1, 100, 120, 160])}
                # Retrieve the matching between the outputs of the last layer and the targets
                indices[str(i)] = self.matcher(outputs_without_aux, targets[str(i)])
            
            new_indices = []
            for inds0, inds1 in zip(indices["0"], indices["1"]):
                new_indices.append({"0":(inds0[0].cpu().numpy(), inds0[1].cpu().numpy()), "1":(inds1[0].cpu().numpy(), inds1[1].cpu().numpy())})


            mask_cls_results = {}
            mask_pred_results = {}
            param_pred_results = {}
            depth_pred_results = {}
            query_embedding_results = {}
            processed_results = []

            for i in range(2):
                mask_cls_results[str(i)] = outputs[str(i)]["pred_logits"] # torch.Size([b, num_queries, num_classes + 1])
                mask_pred_results[str(i)] = outputs[str(i)]["pred_masks"] # torch.Size([b, num_queries, h/4, w/4])
                param_pred_results[str(i)] = outputs[str(i)]["pred_params"] # torch.Size([b, num_queries, 3])
                if self.predict_depth:
                    depth_pred_results[str(i)] = outputs[str(i)]["pred_depths"] # torch.Size([b, num_queries, h, w]
                    if not depth_pred_results[str(i)].shape[-1] == images[str(i)].tensor.shape[-1]:
                        depth_pred_results[str(i)] = F.interpolate(
                            depth_pred_results[str(i)],
                            size = (images[str(i)].tensor.shape[-2], images[str(i)].tensor.shape[-1]),
                            mode = "bilinear",
                            align_corners=False,
                        ) # torch.Size([b,num_queries,h,w])
                query_embedding_results[str(i)] = outputs[str(i)]["query_embeddings"] # torch.Size([b, num_queries, 256])
                # upsample masks
                if not mask_pred_results[str(i)].shape[-1] == images[str(i)].tensor.shape[-1]:
                    mask_pred_results[str(i)] = F.interpolate(
                        mask_pred_results[str(i)],
                        size=(images[str(i)].tensor.shape[-2], images[str(i)].tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    ) # torch.Size([b,num_queries,h,w])
            for ir, item in enumerate(zip(batched_inputs, images["0"].image_sizes)):
                input_per_image, image_size = item
                processed_results.append({"0":{}, "1":{}})

                label_masks = []
                for i in range(2):    
                    height = input_per_image[str(i)].get("height", image_size[0]) # ep 349
                    width = input_per_image[str(i)].get("width", image_size[1]) # ep 640
                    
 
                    # Return semantic segmentation predictions in the original resolution.
                    # if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_results[str(i)][ir], image_size, height, width
                    ) 
                    mask_cls_result = mask_cls_results[str(i)][ir].to(mask_pred_result) # torch.Size([num_queries, num_classes, num_classes + 1])

                    # plane inference
                    # if self.semantic_on:
                        
                    plane_seg, inferred_planes_depth, inferred_seg_depth, valid_param, valid_score, label_mask = retry_if_cuda_oom(self.plane_inference)(mask_cls_result,
                     mask_pred_result, param_pred_results[str(i)][ir], depth_pred_results[str(i)][ir] if self.predict_depth else None)
                    
                    label_masks.append(label_mask)

                    
                    processed_results[-1][str(i)]["sem_seg"] = plane_seg # (num_queries+1, h, w)
                    processed_results[-1][str(i)]["planes_depth"] = inferred_planes_depth
                    # processed_results[-1][str(i)]["seg_depth"] = inferred_seg_depth
                    processed_results[-1][str(i)]["valid_params"] = valid_param
                    processed_results[-1][str(i)]["valid_scores"] = valid_score
                    processed_results[-1][str(i)]["label_mask"] = label_mask

                    # if self.outputdir_for_results is not None:
                    #     # processed_results[-1][str(i)]["valid_query_embs"] = query_embedding_results[str(i)][ir][label_mask, :] # [valid_plane_num, 256]
                    #     valid_query_emb =  query_embedding_results[str(i)][ir][label_mask, :].cpu().numpy()
                    #     save_results = {"sem_seg": plane_seg.cpu().numpy(), 
                    #                     "valid_query_embs": valid_query_emb, 
                    #                     "valid_params": valid_param.cpu().numpy(), 
                    #                     "label_mask": label_mask.cpu().numpy()
                    #                     }
                    #     save_dict(save_results, folder=self.outputdir_for_results, prefix=input_per_image[str(i)]["image_id"])

                    # processed_results[-1][str(i)]["pred2gt_indices"] = indices[str(i)][ir]
                    # processed_results[-1][str(i)]["query_embeddings"] = query_embedding_results[str(i)][ir].to(mask_pred_result)
                if self.predict_poses:

                    # assignment_matrixs = retry_if_cuda_oom(self.assignment_inference)(
                    # torch.cat([outputs["attn_fundamentals"][ir], outputs["pred_corrs"][ir].unsqueeze(0)], 
                    #             dim=0) if self.predict_corrs else outputs["attn_fundamentals"][ir],
                    #             label_mask0 = label_masks[0], label_mask1 = label_masks[1])
                    assignment_matrixs = retry_if_cuda_oom(self.assignment_inference)(
                                outputs["attn_fundamentals"][ir],
                                label_mask0 = label_masks[0], label_mask1 = label_masks[1])

                    processed_results[-1]["pred_poses"] = outputs["pred_poses"][ir]
                    #!  rot: [-pi, pi]
                    if outputs["pred_poses"][ir].data[1][-1] < 0:
                        processed_results[-1]["pred_poses"].data[1][3:] = - processed_results[-1]["pred_poses"].data[1][3:]

                    processed_results[-1]["assignment_matrixs"] = assignment_matrixs
                    processed_results[-1]["valid_assignment_matrixs"] = assignment_matrixs[:,label_masks[0]][:,:,label_masks[1]] # (n, valid_num1, valid_num2)
                    processed_results[-1]["attn_fundamentals"] = outputs["attn_fundamentals"][ir]
                    processed_results[-1]["pred2gt_indices"] = new_indices[ir]

                    #! for visualization
                    processed_results[-1]["query1"] = outputs["query1"][ir]
                    processed_results[-1]["key2"] = outputs["key2"][ir]
                
                else:
                    processed_results[-1]["singleview_attn"] = outputs["singleview_attns"][ir]
                    processed_results[-1]["pred2gt_indices"] = new_indices[ir]
                    processed_results[-1]["query1"] = outputs["0"]["query_embeddings"][ir]
                    processed_results[-1]["key2"] = outputs["1"]["query_embeddings"][ir]

                if self.predict_inverse_poses:
                    processed_results[-1]["pred_inverse_poses"] = outputs["pred_inverse_poses"][ir]
                    if outputs["pred_inverse_poses"][ir].data[1][-1] < 0:
                        processed_results[-1]["pred_inverse_poses"].data[1][3:] = - processed_results[-1]["pred_inverse_poses"].data[1][3:]
                
                if self.predict_corrs:
                    processed_results[-1]["pred_corrs"] = outputs["pred_corrs"][ir]

                # if self.outputdir_for_results is not None:
                    
                #     save_results = {
                #         "attn_fundamentals": processed_results[-1]["attn_fundamentals"].cpu().numpy(), 
                #         "valid_assignment_matrixs": processed_results[-1]["valid_assignment_matrixs"].cpu().numpy(), 
                #         "assignment_matrixs": processed_results[-1]["assignment_matrixs"].cpu().numpy(),
                #         "pred_poses": processed_results[-1]["pred_poses"].cpu().data.numpy(),
                #         }
                #     save_dict(save_results, folder=self.outputdir_for_results, prefix=input_per_image["0"]["image_id"]+"_"+input_per_image["1"]["image_id"])
            # print("inference time:", time.time()-start_time)
            del outputs

            # return processed_results, new_indices, gt_corrs
            return processed_results
        # return loss

    def inference_multiviews(self, batched_inputs): # for >=3 views, #!batch=1
        
        num_views = batched_inputs[0]['num_views']
        edges = batched_inputs[0]['edges']

        # edge_results = []
        edge_results = {str(v):{} for v in range(num_views)} # {"edge_point":} X num_views
        edge_cameras = {}
        edge_valid_assignment_matrixs = {}

        for ed in edges:        
            images = {} # images: {"0/1":} X 2
            outputs = {}  # outputs: {"0/1":} X 2
            ed_key = "_".join([str(ed[0]), str(ed[1])])
            for e_i,e in enumerate(ed):
                imgs = [x[str(e)]["image"].to(self.device) for x in batched_inputs] 
                imgs = [(x - self.pixel_mean) / self.pixel_std for x in imgs]
                imgs = ImageList.from_tensors(imgs, self.size_divisibility)
                images[str(e_i)] = imgs

                features = self.backbone(imgs.tensor)
            
                outputs[str(e_i)] = self.sem_seg_head(features) 

            targets = self.prepare_targets_multiviews(batched_inputs, images, ed) # targets: {"0/1":} X 2

            if self.predict_poses:
                Ps = SE3(torch.stack(targets["gt_poses"]))
                Gs = SE3.IdentityLike(Ps)
                Ps_out = SE3(Ps.data.clone())
                
                intrinsics = torch.stack(targets["intrinsics"]) # (B,4)

                if self.mask_embeddings:
                    # (b,2,nq,512)
                    twoviews_features = torch.stack([
                        torch.cat([outputs["0"]["query_embeddings"], outputs["0"]["mask_embeddings"].detach()], dim=-1), 
                        torch.cat([outputs["1"]["query_embeddings"], outputs["1"]["mask_embeddings"].detach()], dim=-1)
                        ], dim=1)
                else:
                    #  (b, 2, num_queries, 256)
                    twoviews_features = torch.stack([outputs["0"]["query_embeddings"], 
                                                outputs["1"]["query_embeddings"]], dim=1)

                #  (b, 2, num_queries, 3)
                plane_params = torch.stack([outputs["0"]["pred_params"], 
                                            outputs["1"]["pred_params"]], dim=1)

                plane_centers = None
                if self.predict_center:
                    plane_centers = torch.stack([outputs["0"]["pred_centers"], 
                                    outputs["1"]["pred_centers"]], dim=1).detach()
                    plane_centers = plane_centers[:,:,:,[1,0]]  # format: (y,x)
                elif self.mask2center:
                    plane_centers = self.mask_prob2center([outputs["0"]["pred_masks"], 
                                    outputs["1"]["pred_masks"]])  # format: (y,x)


                outputs["pred_poses"], outputs["pred_inverse_poses"], outputs["attn_fundamentals"], outputs["pred_corrs"], outputs["query1"], outputs["key2"] = \
                self.camera_decoder(features = twoviews_features, Gs = Gs,
                                    label_masks = self.cal_label_masks([outputs["0"]["pred_logits"], 
                                    outputs["1"]["pred_logits"]]) if self.filter_cam_planes else None, 
                                    plane_params = plane_params, 
                                    plane_centers = plane_centers, 
                                    intrinsics = intrinsics, inference=True) # SE3(b, 2, 6)
            else:
                f_dim = outputs["0"]["query_embeddings"].shape[-1]
                outputs["singleview_attns"] = outputs["0"]["query_embeddings"] @ outputs["1"]["query_embeddings"].transpose(-2, -1) * (f_dim ** -0.5) # (b,nq,256) @ (b, 256, nq) -> (b, nq, nq)

            if self.training:
                return 

            else:
                
                # output_Gs = self.camera_decoder(twoviews_features, self.Gs, self.intrinsics, inference=True) # (2, 7)

                indices = {}
                for i in range(2):
                    outputs_without_aux = {k: v for k, v in outputs[str(i)].items() if k != "aux_outputs"} 
                    # {'pred_logits': torch.Size([1, 100, 61]), 'pred_masks': torch.Size([1, 100, 120, 160])}
                    # Retrieve the matching between the outputs of the last layer and the targets
                    indices[str(i)] = self.matcher(outputs_without_aux, targets[str(i)])
                
                new_indices = []
                for inds0, inds1 in zip(indices["0"], indices["1"]):
                    new_indices.append({"0":(inds0[0].cpu().numpy(), inds0[1].cpu().numpy()), "1":(inds1[0].cpu().numpy(), inds1[1].cpu().numpy())})

                # gt_corrs = [x["gt_corrs"] for x in batched_inputs]

                mask_cls_results = {}
                mask_pred_results = {}
                param_pred_results = {}
                depth_pred_results = {}
                query_embedding_results = {}
                # processed_results = []

                for i in range(2):
                    mask_cls_results[str(i)] = outputs[str(i)]["pred_logits"] # torch.Size([b, num_queries, num_classes + 1])
                    mask_pred_results[str(i)] = outputs[str(i)]["pred_masks"] # torch.Size([b, num_queries, h/4, w/4])
                    param_pred_results[str(i)] = outputs[str(i)]["pred_params"] # torch.Size([b, num_queries, 3])
                    if self.predict_depth:
                        depth_pred_results[str(i)] = outputs[str(i)]["pred_depths"] # torch.Size([b, num_queries, h, w]
                        if not depth_pred_results[str(i)].shape[-1] == images[str(i)].tensor.shape[-1]:
                            depth_pred_results[str(i)] = F.interpolate(
                                depth_pred_results[str(i)],
                                size = (images[str(i)].tensor.shape[-2], images[str(i)].tensor.shape[-1]),
                                mode = "bilinear",
                                align_corners=False,
                            ) # torch.Size([b,num_queries,h,w])
                    query_embedding_results[str(i)] = outputs[str(i)]["query_embeddings"] # torch.Size([b, num_queries, 256])
                    # upsample masks
                    if not mask_pred_results[str(i)].shape[-1] == images[str(i)].tensor.shape[-1]:
                        mask_pred_results[str(i)] = F.interpolate(
                            mask_pred_results[str(i)],
                            size=(images[str(i)].tensor.shape[-2], images[str(i)].tensor.shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                        ) # torch.Size([b,num_queries,h,w])
                
                for ir, item in enumerate(zip(batched_inputs, images["0"].image_sizes)):
                    input_per_image, image_size = item
                    label_masks = []
                    for i in range(2):    
                        height = input_per_image[str(i)].get("height", image_size[0]) # ep 349
                        width = input_per_image[str(i)].get("width", image_size[1]) # ep 640
                        
    
                        # Return semantic segmentation predictions in the original resolution.
                        # if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_results[str(i)][ir], image_size, height, width
                        ) 
                        mask_cls_result = mask_cls_results[str(i)][ir].to(mask_pred_result) # torch.Size([num_queries, num_classes, num_classes + 1])

                        # plane inference
                        # if self.semantic_on:
                            
                        plane_seg, inferred_planes_depth, inferred_seg_depth, valid_param, valid_score, label_mask = retry_if_cuda_oom(self.plane_inference)(mask_cls_result,
                        mask_pred_result, param_pred_results[str(i)][ir], depth_pred_results[str(i)][ir] if self.predict_depth else None)
                        
                        label_masks.append(label_mask)


                        edge_results[str(ed[i])]["sem_seg"] = plane_seg # (num_queries+1, h, w)
                        edge_results[str(ed[i])]["planes_depth"] = inferred_planes_depth
                        # processed_results[-1][str(i)]["seg_depth"] = inferred_seg_depth
                        edge_results[str(ed[i])]["valid_params"] = valid_param
                        edge_results[str(ed[i])]["valid_scores"] = valid_score
                        edge_results[str(ed[i])]["label_mask"] = label_mask

                        # if self.outputdir_for_results is not None:
                        #     # processed_results[-1][str(i)]["valid_query_embs"] = query_embedding_results[str(i)][ir][label_mask, :] # [valid_plane_num, 256]
                        #     valid_query_emb =  query_embedding_results[str(i)][ir][label_mask, :].cpu().numpy()
                        #     save_results = {"sem_seg": plane_seg.cpu().numpy(), 
                        #                     "valid_query_embs": valid_query_emb, 
                        #                     "valid_params": valid_param.cpu().numpy(), 
                        #                     "label_mask": label_mask.cpu().numpy()
                        #                     }
                        #     save_dict(save_results, folder=self.outputdir_for_results, prefix=input_per_image[str(i)]["image_id"])

                        # processed_results[-1][str(i)]["pred2gt_indices"] = indices[str(i)][ir]
                        # processed_results[-1][str(i)]["query_embeddings"] = query_embedding_results[str(i)][ir].to(mask_pred_result)
                    if self.predict_poses:

                        # assignment_matrixs = retry_if_cuda_oom(self.assignment_inference)(
                        # torch.cat([outputs["attn_fundamentals"][ir], outputs["pred_corrs"][ir].unsqueeze(0)], 
                        #             dim=0) if self.predict_corrs else outputs["attn_fundamentals"][ir],
                        #             label_mask0 = label_masks[0], label_mask1 = label_masks[1])
                        assignment_matrixs = retry_if_cuda_oom(self.assignment_inference)(
                                    outputs["attn_fundamentals"][ir],
                                    label_mask0 = label_masks[0], label_mask1 = label_masks[1])

                        processed_pred_poses = outputs["pred_poses"][ir]
                        #!  rot: [-pi, pi]
                        if outputs["pred_poses"][ir].data[1][-1] < 0:
                            processed_pred_poses.data[1][3:] = - processed_pred_poses.data[1][3:]

                        processed_assignment_matrixs = assignment_matrixs
                        processed_valid_assignment_matrixs = assignment_matrixs[:,label_masks[0]][:,:,label_masks[1]] # (n, valid_num1, valid_num2)
                        processed_attn_fundamentals = outputs["attn_fundamentals"][ir]
                        processed_pred2gt_indices = new_indices[ir]

                        #! for visualization
                        # processed_results[-1]["query1"] = outputs["query1"][ir]
                        # processed_results[-1]["key2"] = outputs["key2"][ir]
                    
                    # else:
                        # processed_results[-1]["singleview_attn"] = outputs["singleview_attns"][ir]
                        # processed_results[-1]["pred2gt_indices"] = new_indices[ir]
                        # processed_results[-1]["query1"] = outputs["0"]["query_embeddings"][ir]
                        # processed_results[-1]["key2"] = outputs["1"]["query_embeddings"][ir]

                    # if self.predict_inverse_poses:
                    #     processed_results[-1]["pred_inverse_poses"] = outputs["pred_inverse_poses"][ir]
                    #     if outputs["pred_inverse_poses"][ir].data[1][-1] < 0:
                    #         processed_results[-1]["pred_inverse_poses"].data[1][3:] = - processed_results[-1]["pred_inverse_poses"].data[1][3:]
                    
                    # if self.predict_corrs:
                    #     processed_pred_corrs = outputs["pred_corrs"][ir]

                    # if self.outputdir_for_results is not None:
                        
                    #     save_results = {
                    #         "attn_fundamentals": processed_results[-1]["attn_fundamentals"].cpu().numpy(), 
                    #         "valid_assignment_matrixs": processed_results[-1]["valid_assignment_matrixs"].cpu().numpy(), 
                    #         "assignment_matrixs": processed_results[-1]["assignment_matrixs"].cpu().numpy(),
                    #         "pred_poses": processed_results[-1]["pred_poses"].cpu().data.numpy(),
                    #         }
                    #     save_dict(save_results, folder=self.outputdir_for_results, prefix=input_per_image["0"]["image_id"]+"_"+input_per_image["1"]["image_id"])
                # print("inference time:", time.time()-start_time)
                del outputs

                # # return processed_results, new_indices, gt_corrs
                # return processed_results
            edge_cameras[ed_key] = processed_pred_poses
            edge_valid_assignment_matrixs[ed_key] = processed_valid_assignment_matrixs
        
        edge_results["edge_cameras"] = edge_cameras
        edge_results["edge_valid_assignment_matrixs"] = edge_valid_assignment_matrixs

        return [edge_results]

    def prepare_targets_multiviews(self, batched_inputs, images, edge):
        # batched_inputs: {"edge_point":} X num_views
        # images: {"0/1":} X 2
        # new_targets: {"0/1":} X 2
        h_pad, w_pad = images["0"].tensor.shape[-2:]
        edge_key = "_".join([str(e) for e in edge])
        new_targets = {"0":[], "1":[], "gt_poses":[], "intrinsics": [], "gt_corrs": []}
        for i in range(2):
            for targets_per_image in batched_inputs:
                # pad gt
                gt_masks = targets_per_image[str(edge[i])]["plane_masks"].tensor.to(self.device)
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

                new_targets[str(i)].append(
                    {
                        "labels": targets_per_image[str(edge[i])]["classes"].to(self.device),
                        "masks": padded_masks,
                        "params": targets_per_image[str(edge[i])]["params"].to(self.device),
                        "plane_centers": targets_per_image[str(edge[i])]["plane_centers"].to(self.device) if self.predict_center else None,
                        "plane_depths": targets_per_image[str(edge[i])]["plane_depths"].to(self.device),
                        "resize14_plane_depths": targets_per_image[str(edge[i])]["resize14_plane_depths"].to(self.device),
                        #TODO: to be deleted
                        "K_inv_dot_xy_1": targets_per_image[str(edge[i])]["K_inv_dot_xy_1"].to(self.device) if self.training else None,
                    }
                )
                if i == 1 and self.predict_poses:
                    rel_pose = targets_per_image['edge_rel_poses'][edge_key]
                    w,x,y,z = rel_pose['rotation']
                    rel_pose = np.array(rel_pose['position']+[x,y,z,w])
                    base_pose = np.array([0,0,0,0,0,0,1])
                    new_targets["gt_poses"].append(
                        torch.from_numpy(np.vstack([base_pose, rel_pose])).to(self.device)
                        )
                    new_targets["intrinsics"].append(targets_per_image["intrinsics"].to(self.device))
                    new_targets["gt_corrs"].append(torch.tensor(targets_per_image["edge_gt_corrs"][edge_key]).to(self.device))
            
        return new_targets

    def mask_prob2center(self, output_mask_probs):
        plane_centers = []

        eps = 1e-6
        h,w = output_mask_probs[0].shape[-2:] 
        x = torch.arange(1, w+1, dtype=torch.float32).view(1, w) / (w+eps)
        y = torch.arange(1, h+1, dtype=torch.float32).view(h, 1) / (h+eps)
        x = x.to(output_mask_probs[0].device)
        y = y.to(output_mask_probs[0].device)
        xx = x.repeat(h, 1)[None,None,:,:]  # (1,1,h/4,w/4)
        yy = y.repeat(1, w)[None,None,:,:]  # (1,1,h/4,w/4)
        
        for i in range(2):
            mask_probs = output_mask_probs[i].detach()  #! detach() (b, nq, h/4, w/4)
            mask_probs = mask_probs.sigmoid()
            mask_probs_hwnorm = F.normalize(mask_probs, p=1, dim = [-2, -1])
            x_centers = torch.sum(mask_probs_hwnorm * xx, dim = [-2, -1]) # (b, nq, h/4, w/4) -> (b, nq)
            y_centers = torch.sum(mask_probs_hwnorm * yy, dim = [-2, -1])
            yx_centers = torch.stack([y_centers, x_centers], dim = -1) # (b, nq, 2)
            plane_centers.append(yx_centers)

        plane_centers = torch.stack(plane_centers, dim = 1) # (b,n_imgs = 2,nq,2)

        return plane_centers
            
            


    def prepare_targets(self, targets, images, K_inv_dot_xy_1s, random_scales):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, K_inv_dot_xy_1, scale in zip(targets, K_inv_dot_xy_1s, random_scales):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "params": targets_per_image.gt_params,
                    "plane_depths": targets_per_image.gt_plane_depths,
                    "resize14_plane_depths": targets_per_image.gt_resize14_plane_depths,
                    "K_inv_dot_xy_1": K_inv_dot_xy_1,
                    "random_scale": scale,
                }
            )
        return new_targets

    def prepare_targets_singleview(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in batched_inputs:
            # pad gt
            gt_masks = targets_per_image["plane_masks"].tensor.to(self.device)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image["classes"].to(self.device),
                    "masks": padded_masks,
                    "params": targets_per_image["params"].to(self.device),
                    "plane_depths": targets_per_image["plane_depths"].to(self.device),
                    "resize14_plane_depths": targets_per_image["resize14_plane_depths"].to(self.device),
                    "K_inv_dot_xy_1": targets_per_image["K_inv_dot_xy_1"].to(self.device) if self.training else None,
                }
            )
            
        return new_targets
    
    def prepare_targets_sparseviews(self, batched_inputs, images):
        h_pad, w_pad = images["0"].tensor.shape[-2:]
        new_targets = {"0":[], "1":[], "gt_poses":[], "intrinsics": [], "gt_corrs": []}
        for i in range(2):
            for targets_per_image in batched_inputs:
                # pad gt
                gt_masks = targets_per_image[str(i)]["plane_masks"].tensor.to(self.device)
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

                new_targets[str(i)].append(
                    {
                        "labels": targets_per_image[str(i)]["classes"].to(self.device),
                        "masks": padded_masks,
                        "params": targets_per_image[str(i)]["params"].to(self.device),
                        "plane_centers": targets_per_image[str(i)]["plane_centers"].to(self.device) if self.predict_center else None,
                        "plane_depths": targets_per_image[str(i)]["plane_depths"].to(self.device),
                        "resize14_plane_depths": targets_per_image[str(i)]["resize14_plane_depths"].to(self.device),
                        "K_inv_dot_xy_1": targets_per_image[str(i)]["K_inv_dot_xy_1"].to(self.device) if self.training else None,
                    }
                )
                if i == 1 and self.predict_poses:
                    rel_pose = targets_per_image['rel_pose']
                    w,x,y,z = rel_pose['rotation']
                    rel_pose = np.array(rel_pose['position']+[x,y,z,w])
                    base_pose = np.array([0,0,0,0,0,0,1])
                    new_targets["gt_poses"].append(
                        torch.from_numpy(np.vstack([base_pose, rel_pose])).to(self.device)
                        )
                    new_targets["intrinsics"].append(targets_per_image["intrinsics"].to(self.device))
                    new_targets["gt_corrs"].append(torch.tensor(targets_per_image["gt_corrs"]).to(self.device))
            
        return new_targets
    
    def process_for_sparseviews(self, outputs, targets):
        keys = outputs["0"].keys()
        new_outputs = {} # {"pred_logits":[(2*b)], ..., "pred_poses":[(b)]}
        for key in keys: 
            if key == 'aux_outputs':
                new_outputs[key] = []
                for aux_i in range(len(outputs["0"][key])):
                    new_outputs[key].append({})
                    for son_key in outputs["0"][key][aux_i]:
                        if son_key=="pred_centers" and not self.predict_center:
                            continue
                        new_outputs[key][-1][son_key] = torch.cat([
                            outputs["0"][key][aux_i][son_key], 
                            outputs["1"][key][aux_i][son_key]
                            ])
            else:
                if key=="pred_centers" and not self.predict_center:
                    continue
                new_outputs[key] = torch.cat([outputs["0"][key], outputs["1"][key]]) # torch.Size([b, num_queries, num_classes + 1])
        if self.predict_poses:
            new_outputs["pred_poses"] = outputs["pred_poses"]
        if self.predict_corrs:
            new_outputs["pred_corrs"] = outputs["pred_corrs"]
        if self.predict_inverse_poses:
            new_outputs["pred_inverse_poses"] = outputs["pred_inverse_poses"]
        if self.predict_joint_corr_pose_param or self.predict_joint_01corr_pose_paramdetach:
            new_outputs["attn_fundamentals"] = outputs["attn_fundamentals"]

        keys = targets["0"][0].keys()
        new_targets = [] # (2*b), [{"labels":,...,"gt_poses":}]
        for i in range(2):
            for j,t in enumerate(targets[str(i)]):
                new_targets.append(t)
                if i == 0 and self.predict_poses:
                    new_targets[-1]["gt_poses"] = targets["gt_poses"][j]
                    new_targets[-1]["gt_corrs"] = targets["gt_corrs"][j]

        return new_outputs, new_targets
    
    def cal_label_masks(self, mask_cls_results):
        
        for i in range(2):
            mask_cls = mask_cls_results[i] # [b, 30, 3]
            mask_cls = F.softmax(mask_cls, dim=-1) # torch.Size([b, num_queries, num_classes + 1 = 3])
            score, labels = mask_cls.max(dim=-1)  # torch.Size([b, num_queries])
            labels[labels != 1] = 0 # [b, num_queries]
            labels = labels > 0  # [b, num_queries]
        
            if i == 0:
                labels_1 = labels.unsqueeze(2).repeat((1,1,self.num_queries)) # [b, num_queries, num_queries]
            if i == 1:
                labels_2 = labels.unsqueeze(2).repeat((1,1,self.num_queries)) # [b, num_queries, num_queries]
                labels_2 = labels_2.permute((0,2,1))

        label_masks = torch.logical_and(labels_1, labels_2)

        label_masks = label_masks.detach()
        
        return label_masks

    def plane_inference(self, mask_cls, mask_pred, param_pred, depth_pred):
        mask_cls = F.softmax(mask_cls, dim=-1) # torch.Size([num_queries, num_classes + 1 = 3])
        score, labels = mask_cls.max(dim=-1)
        labels[labels != 1] = 0 # [num_queries]
        label_mask = labels > 0  # [num_queries]
        if sum(label_mask) == 0:
            _, max_pro_idx = mask_cls[:, 1].max(dim=0)
            label_mask[max_pro_idx] = 1
        valid_param = param_pred[label_mask, :]  # valid_plane_num, 3
        valid_score = score[label_mask] # valid_plane_num
        
        mask_pred = mask_pred.sigmoid() # torch.Size([num_queries, h, w])
        valid_mask_pred = mask_pred[label_mask] # [valid_plane_num,h,w]
        tmp = torch.zeros((self.num_queries + 1 - valid_mask_pred.shape[0], valid_mask_pred.shape[1], valid_mask_pred.shape[2]),
                        dtype = valid_mask_pred.dtype, device = valid_mask_pred.device)
        
        non_plane_mask = (valid_mask_pred > self.plane_mask_threshold).sum(0) == 0
        tmp[-1][non_plane_mask] = 1 
        plane_seg = torch.cat((valid_mask_pred, tmp), dim = 0)
        plane_seg = plane_seg.sigmoid() # [num_queries, h, w]
  
        valid_num = valid_mask_pred.shape[0]
        inferred_planes_depth = None
        inferred_seg_depth = None
        if self.predict_param:
            # get depth map
            h, w = plane_seg.shape[-2:]
            
            depth_maps_inv = torch.matmul(valid_param, self.k_inv_dot_xy1.to(self.pixel_mean.device))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)
            inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg[:valid_num].argmax(dim=0).view(-1)] # plane depth [h,w]
            inferred_planes_depth = inferred_planes_depth.view(h, w)
            inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions
        
        if self.predict_depth:
            valid_depth_pred = depth_pred[label_mask] # [valid_plane_num, h, w]
            segmentation = (plane_seg[:valid_num].argmax(dim=0)[:,:,None] == torch.arange(valid_num).to(plane_seg)).permute(2, 0, 1) # [h, w, 1] == []  -> [valid_plane_num, h, w]
            inferred_seg_depth = (segmentation * valid_depth_pred).sum(0) # [h, w]
            inferred_seg_depth[non_plane_mask] = 0.0 # del non-plane regions
        
        
        return plane_seg, inferred_planes_depth, inferred_seg_depth, valid_param, valid_score, label_mask
    
    def assignment_inference(self, attn_corrs, label_mask0=None, label_mask1=None ):
        assignment_matrix_list = []
        if len(attn_corrs.shape) <= 2:
            attn_corrs = [attn_corrs]

        for a_i, attn_corr in enumerate(attn_corrs):  # torch.Size([8/9, 30, 30]) 
            attn_corr = attn_corr.clone()  #!
            attn_corr = attn_corr.unsqueeze(0)
            max_sum = min(torch.sum(attn_corr, dim=-2).max(), torch.sum(attn_corr, dim=-1).max())
            curr_match_threshold = max_sum * self.match_threshold

            if self.filter_corr_planes:
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
    
    #! from nope-sac
    # def nopesac_inference(self, pred_logits, pred_masks_logits, pred_param, pred_depth):
        
    #     pred_masks_prob = torch.sigmoid(pred_masks_logits)  # nq, h=120, w=160
    #     # pred_masks_prob = F.interpolate(pred_masks_prob[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    #     # oriIdx = torch.arange(0, self.num_queries).to(pred_logits.device)
        
        
    #     pred_prob = F.softmax(pred_logits, dim=-1) # torch.Size([num_queries, num_classes + 1 = 3])
    #     score, labels = pred_prob.max(dim=-1)
    #     labels[labels != 1] = 0 # [num_queries]
    #     # label_mask = labels > 0  # [num_queries]
    #     label_mask = (labels > 0) & (score > self.plane_score_threshold) # ! th=0.6
        
    #     zero_flag = False
    #     if sum(label_mask) == 0:
    #         _, max_pro_idx = pred_prob[:, 1].max(dim=0)
    #         label_mask[max_pro_idx] = 1
    #         score[max_pro_idx] = pred_prob[max_pro_idx, 1]
    #         zero_flag = True

    #     valid_param = pred_param[label_mask, :]  # valid_plane_num, 3
    #     valid_plane_prob = score[label_mask] # valid_plane_num
    #     valid_masks_prob_ori = pred_masks_prob[label_mask]  # valid_plane_num, h, w
    #     valid_masks_prob = valid_plane_prob.view(-1, 1, 1) * valid_masks_prob_ori  # valid_plane_num, h, w
    #     valid_plane_num = valid_param.shape[0]
    #     # valid_plane_feat = query_feat[i, label_mask]  # valid_plane_num, c
    #     # valid_plane_oriIdx = oriIdx[label_mask]

    #     assert valid_plane_num > 0
    #     # get plane segmentation
    #     valid_mask_ids = valid_masks_prob.argmax(0)  # h, w

    #     processed_plane = []
    #     processed_score = []

    #     # valid_plane_idxs = []
    #     valid_plane_masks = []
    #     max_overlap_id = 0
    #     max_overlap = 0.
    #     for pi in range(valid_plane_num):
    #         plane_mask_pi = (valid_mask_ids == pi) & (valid_masks_prob[pi] > self.mask_prob_threshold)  # h, w; #!th=0.5
    #         # plane_mask_pi_np = np.asfortranarray(plane_mask_pi.cpu().numpy())

    #         mask_area = plane_mask_pi.sum().item()
    #         original_area = (valid_masks_prob_ori[pi] >= self.mask_prob_threshold).sum().item()
    #         if zero_flag is False:
    #             if mask_area < 1 or original_area < 1:
    #                 continue
    #             overlap = mask_area / original_area
    #             if overlap > max_overlap:
    #                 max_overlap = overlap
    #                 max_overlap_id = pi
    #             if overlap < self.overlap_threshold: #! th=0.6
    #                 continue
    #         else:
    #             if mask_area == 0:
    #                 # plane_mask_pi_np[0, 0] = 1
    #                 plane_mask_pi[0, 0] = 1
    #         # rle_pi = mask_util.encode(plane_mask_pi_np)
    #         # bbox_i = mask_util.toBbox(rle_pi).tolist()  # x, y, w, h
    #         # plane_param_pi = valid_param[pi].cpu()  # shape: [3]
    #         plane_param_pi = valid_param[pi]
    #         score_pi = valid_plane_prob[pi]
    #         # segmentation = {}
    #         # segmentation["size"] = [height, width]
    #         # segmentation["counts"] = rle_pi["counts"]

    #         processed_plane.append(plane_param_pi)
    #         processed_score.append(score_pi)
    #         # ins_pi = {}
    #         # ins_pi["image_id"] = batched_inputs[i]["image_id"]
    #         # ins_pi["file_name"] = batched_inputs[i]["file_name"]
    #         # ins_pi["category_id"] = 0
    #         # ins_pi["score"] = score_pi.item()
    #         # ins_pi["segmentation"] = segmentation
    #         # ins_pi["bbox"] = bbox_i
    #         # ins_pi["bbox_mode"] = 1
    #         # instances.append(ins_pi)
    #         # valid_query_feat_pi = valid_plane_feat[pi]  # c
    #         # valid_query_feats.append(valid_query_feat_pi)
    #         # valid_plane_idxs.append(valid_plane_oriIdx[pi])

    #         valid_plane_masks.append(plane_mask_pi)

    #         # get plane center
    #         # plane_mask = plane_mask_pi_np.astype(np.float)
    #         # pixel_num = plane_mask.sum()
    #         # x_map = self.normalized_xy_map[0] * plane_mask
    #         # y_map = self.normalized_xy_map[1] * plane_mask
    #         # x_sum = x_map.sum()
    #         # y_sum = y_map.sum()
    #         # plane_x = x_sum / (pixel_num + 1e-10)
    #         # plane_y = y_sum / (pixel_num + 1e-10)

    #         # instance_center = np.zeros([2]).astype(np.float32)
    #         # instance_center[0] = plane_x
    #         # instance_center[1] = plane_y
    #         # valid_plane_ins_centers.append(instance_center)

    #     if len(processed_plane) == 0:
    #         pi = max_overlap_id
    #         plane_mask_pi = valid_mask_ids == pi  # h, w
    #         # plane_mask_pi = np.asfortranarray(plane_mask_pi.cpu().numpy())
    #         mask_area = plane_mask_pi.sum()
    #         original_area = (valid_masks_prob_ori[pi] >= 0.5).sum().item()
    #         # rle_pi = mask_util.encode(plane_mask_pi)
    #         # bbox_i = mask_util.toBbox(rle_pi).tolist()  # x, y, w, h
    #         plane_param_pi = valid_param[pi]
    #         score_pi = valid_plane_prob[pi]

    #         # segmentation = {}
    #         # segmentation["size"] = [height, width]
    #         # segmentation["counts"] = rle_pi["counts"]

    #         processed_plane.append(plane_param_pi)
    #         processed_score.append(score_pi)
    #         # ins_pi = {}
    #         # ins_pi["image_id"] = batched_inputs[i]["image_id"]
    #         # ins_pi["file_name"] = batched_inputs[i]["file_name"]
    #         # ins_pi["category_id"] = 0
    #         # ins_pi["score"] = score_pi.item()
    #         # ins_pi["segmentation"] = segmentation
    #         # ins_pi["bbox"] = bbox_i
    #         # ins_pi["bbox_mode"] = 1

    #         # instances.append(ins_pi)

    #         # valid_query_feat_pi = valid_plane_feat[pi]  # c
    #         # valid_query_feats.append(valid_query_feat_pi)

    #         # valid_plane_idxs.append(valid_plane_oriIdx[pi])

    #         # valid_plane_masks.append(torch.from_numpy(plane_mask_pi))
    #         valid_plane_masks.append(plane_mask_pi)

    #         # # get plane center
    #         # plane_mask = plane_mask_pi.astype(np.float)
    #         # pixel_num = plane_mask.sum()
    #         # x_map = self.normalized_xy_map[0] * plane_mask
    #         # y_map = self.normalized_xy_map[1] * plane_mask
    #         # x_sum = x_map.sum()
    #         # y_sum = y_map.sum()
    #         # plane_x = x_sum / pixel_num
    #         # plane_y = y_sum / pixel_num

    #         # instance_center = np.zeros([2]).astype(np.float32)
    #         # instance_center[0] = plane_x
    #         # instance_center[1] = plane_y
    #         # valid_plane_ins_centers.append(instance_center)


    #     # mask_pred = mask_pred.sigmoid() # torch.Size([num_queries, h, w])
    #     # valid_mask_pred = mask_pred[label_mask] # [valid_plane_num,h,w]
        
            
    #     valid_plane_masks = torch.stack(valid_plane_masks)
    #     tmp = torch.zeros((self.num_queries + 1 - valid_plane_masks.shape[0], valid_plane_masks.shape[1], valid_plane_masks.shape[2]),
    #                     dtype = valid_plane_masks.dtype, device = valid_plane_masks.device)
        
    #     non_plane_mask = (valid_plane_masks > 0).sum(0) == 0
    #     tmp[-1][non_plane_mask] = 1 
    #     plane_seg = torch.cat((valid_plane_masks, tmp), dim = 0)
    #     plane_seg = plane_seg.sigmoid() # [num_queries, h, w]

    #     processed_plane = torch.stack(processed_plane)
    #     processed_score = torch.score(processed_score)

    #     # new_label_mask = label_mask

    #     valid_num = valid_plane_masks.shape[0]
    #     inferred_planes_depth = None
    #     inferred_seg_depth = None
    #     if self.predict_param:
    #         # get depth map
    #         h, w = plane_seg.shape[-2:]
            
    #         # depth_maps_inv = torch.matmul(valid_param, self.k_inv_dot_xy1.to(self.pixel_mean.device))
    #         depth_maps_inv = torch.matmul(processed_plane, self.k_inv_dot_xy1.to(self.pixel_mean.device))
    #         depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
    #         depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)
    #         inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg[:valid_num].argmax(dim=0).view(-1)] # plane depth [h,w]
    #         inferred_planes_depth = inferred_planes_depth.view(h, w)
    #         inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions
        
    #     # if self.predict_depth:
    #     #     valid_depth_pred = depth_pred[label_mask] # [valid_plane_num, h, w]
    #     #     segmentation = (plane_seg[:valid_num].argmax(dim=0)[:,:,None] == torch.arange(valid_num).to(plane_seg)).permute(2, 0, 1) # [h, w, 1] == []  -> [valid_plane_num, h, w]
    #     #     inferred_seg_depth = (segmentation * valid_depth_pred).sum(0) # [h, w]
    #     #     inferred_seg_depth[non_plane_mask] = 0.0 # del non-plane regions
            
        
        
    #     return plane_seg, inferred_planes_depth, inferred_seg_depth, processed_plane, processed_score, label_mask

