# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# and by https://github.com/facebookresearch/Mask2Former
import logging

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

from lietorch import SE3


# from detectron2.utils.memory import retry_if_cuda_oom
# from detectron2.modeling.postprocessing import sem_seg_postprocess

def argshuffle(x, y):
    """x->y indexs

    Args:
        x (_type_): (n,) tensor of shape
        y (_type_): (m,) tensor of shape
    """
    length = x.shape[0]
    # assert length == y.shape[0]
    m = torch.stack([y for i in range(length)]).T
    return torch.where((x-m)==0)[1]

def get_tensorbatch_plane_params_in_global(planes, pose_matrixs):
    """
    input:
    @planes: plane params [b, n, 3] n/d
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys 
    @pose_matrixs: [b, 4, 4]
    output:
    plane parameters in global frame. n/d
    """
    # ! n/d -> n*d
    planends =  planes/torch.sum(planes**2, dim=-1).unsqueeze(2) # (b,n,3)/(b,n,1)
    start = torch.ones_like(planends) * pose_matrixs[:,:3,3].unsqueeze(1) # (b, n, 3)* (b,1,3) -> b,n,3
    coor = torch.tensor([1, -1, -1], dtype=planends.dtype, device=planends.device) 
    end = planends * coor # suncg2habitat (b,n,3)*(3,) -> (b,n,3)
    # end = planes
    # end = (quaternion.as_rotation_matrix(rot) @ (end).T).T + tran  # cam2world
    end = torch.bmm(pose_matrixs[:,:3,:3], end.permute(0,2,1)).permute(0,2,1) + pose_matrixs[:,:3,3].unsqueeze(1)  # (b,3,3)*(b,3,n) -> (b,3,n)->(b,n,3)+(b,1,3)
    a = end # (b,n,3)
    b = end - start
    planends_world = ((a * b).sum(dim=-1) / torch.norm(b, dim=-1) ** 2).unsqueeze(-1) * b  # ((b,n)/(b,n))**2 -> (b,n,1)*(b,n,3) -> (b.n.3)
    planends_world *= coor
    planes_world = planends_world/torch.sum(planends_world**2, dim=-1).unsqueeze(2)
    return planes_world

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def assignment_inference(filter_corr_planes, attn_corrs, match_threshold = 0.2, label_mask0=None, label_mask1=None ):
    assignment_matrix_list = []
    if len(attn_corrs.shape) <= 2:
        attn_corrs = [attn_corrs]
    # !
    for a_i, attn_corr in enumerate(attn_corrs):  # torch.Size([8/9, 30, 30]) 
        
        attn_corr = attn_corr.unsqueeze(0)
        max_sum = min(torch.sum(attn_corr, dim=-2).max(), torch.sum(attn_corr, dim=-1).max())
        curr_match_threshold = max_sum * match_threshold

        if filter_corr_planes:
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
        #!
        # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
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


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def l1_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):

    return torch.mean(torch.sum(torch.abs(targets - inputs), dim=1))

l1_loss_jit = torch.jit.script(
    l1_loss
)

def cos_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    ):
    similarity = torch.nn.functional.cosine_similarity(inputs, targets, dim=1)  # N
    return torch.mean(1-similarity)

cos_loss_jit = torch.jit.script(
    cos_loss
)


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, k_inv_dot_xy1,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.k_inv_dot_xy1 = k_inv_dot_xy1
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float() # # torch.Size([b, num_queries, 3])

        idx = self._get_src_permutation_idx(indices) # (batch_idx, indices[0] for src)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # target["labels"][indice[1](for tgt)]
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices) # (batch_idx[b*num_tgt_planes], src_idx)
        tgt_idx = self._get_tgt_permutation_idx(indices) # (batch_idx, tgt_idx)
        src_masks = outputs["pred_masks"] # [b, num_queries, h/4, w/4]
        src_masks = src_masks[src_idx] # [b*num_tgt_planes_i, h/4, w/4]
        masks = [t["masks"] for t in targets] # [num_tgt_planes_i, h, w] * b
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose() # [b, max([num_tgt_planes_i]), h, w]
        target_masks = target_masks.to(src_masks) 
        target_masks = target_masks[tgt_idx] 

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords# (h, w, b*3) -> (b, h, w, 3)
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ) # torch.Size([b*num_tgt_planes, num_points, 2])
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1) # torch.Size([b*num_tgt_planes, num_points])

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1) # 

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_params(self, outputs, targets, indices, num_planes, log=True):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_params' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_param = outputs['pred_params'][idx]  # N, 3
        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        # l1 loss
        loss_param_l1 = torch.mean(torch.sum(torch.abs(target_param - src_param), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(src_param, target_param, dim=1)  # N
        loss_param_cos = torch.mean(1-similarity)
        angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))

        losses = {}
        losses['loss_param_l1'] = loss_param_l1 
        losses['loss_param_cos'] = loss_param_cos 
        if log:
            losses['mean_angle'] = angle * 180.0 / np.pi

        return losses

    def loss_centers(self, outputs, targets, indices, num_planes, log=True):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_centers' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_center = outputs['pred_centers'][idx]  # N, 2
        target_center = torch.cat([tgt["plane_centers"][J] for tgt, (_, J) in zip(targets, indices)], dim=0)


        # l1 loss
        delta_xy = torch.abs(target_center - src_center)  # N, 2
        dist = torch.norm(delta_xy, dim=-1)  # N
        loss_center_l2 = torch.mean(dist)

        losses = {}
        losses['loss_center'] = loss_center_l2

        # if 'gt_plane_pixel_centers' in outputs.keys():
        #     gt_plane_pixel_centers = outputs['gt_plane_pixel_centers']
        #     pixel_center = outputs['pixel_center']  # b, 2, h, w
        #     valid_region = outputs['valid_region']  # b, 1, h, w
        #     mask = valid_region > 0
        #     pixel_dist = torch.norm(torch.abs(gt_plane_pixel_centers - pixel_center), dim=1, keepdim=True)  #b, 1, h, w
        #     loss_pixel_center = torch.mean(pixel_dist[mask])
        #     losses['loss_center_pixel'] = loss_pixel_center

        return losses

    def loss_Q(self, outputs, targets, indices, num_planes_sum, log=True):
        
        gt_depths = torch.stack([t["plane_depths"].sum(dim = 0) for t in targets]) # b , h, w
        
        b, h, w = gt_depths.shape

        assert b == len(targets)

        losses = 0.

        for bi in range(b):
            
            segmentation = targets[bi]['masks']  # num_tgt_planes, h, w
            num_planes = segmentation.shape[0]
            device = segmentation.device

            depth = gt_depths[bi]  # 1, h, w
            # k_inv_dot_xy1_map = (self.k_inv_dot_xy1).clone().view(3, h, w).to(device)
            k_inv_dot_xy1_map = targets[bi]["K_inv_dot_xy_1"].view(3, h, w) # 3, h, w
            gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w

            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]
            # num_planes = idx_tgt.max() + 1
            assert idx_tgt.max() + 1 == num_planes

            # select pixel with segmentation
            loss_bi = 0.
            # tmp = {}
            for i in range(num_planes):
                if i > 29:
                    print("i, idx_tgt", i, indices, num_planes)
                gt_plane_idx = int(idx_tgt[i])
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0

                # delete noise！
                gt_param = targets[bi]["params"][gt_plane_idx].view(1, 3) # 
                gt_dist = torch.abs(torch.einsum('ik,kmn->imn', gt_param, gt_pts_map) - 1.0)  # 1, h, w
                mask = (gt_dist < 0.2) & mask

                pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num

                pred_plane_idx = int(idx_out[i])
                param = outputs['pred_params'][bi][pred_plane_idx].view(1, 3)
                # 
                if pts.shape[1]<1:
                    continue
                loss = torch.abs(torch.matmul(param, pts) - 1)  # 1, plane_pt_num
                if torch.sum(torch.isnan(loss))>0 or torch.sum(torch.isinf(loss))>0:
                    print(loss)
                    loss = loss[~torch.isnan(loss)]
                    loss = loss[~torch.isinf(loss)]
                    # continue
                loss = loss.mean()
                # tmp[i] = (loss, param, pts)
                
                loss_bi += loss
            loss_bi = loss_bi / float(num_planes)
            losses += loss_bi

            if torch.isnan(losses) or torch.isinf(losses):
                print("Q!Q: ",losses)
            # exit()

        losses_dict = {}
        losses_dict['loss_Q'] = losses / float(b)

        return losses_dict            
      
    
    def loss_plane_depths(self, outputs, targets, indices, num_planes_sum, log=True):

        src_idx = self._get_src_permutation_idx(indices) # [batch_idx, src_idx]
        
        src_depths = outputs["pred_depths"] # [b, num_queries, h, w]
        src_depths = src_depths[src_idx] # [b * num_tgt_planes_i, h, w]
        target_plane_depths = torch.cat([t["plane_depths"][J] for t, (_, J) in zip(targets, indices)]) # [b * num_tgt_planes_i, height, width]
        if src_depths.shape[-1] != target_plane_depths.shape[-1]:
            target_plane_depths = torch.cat([t["resize14_plane_depths"][J] for t,(_, J) in zip(targets, indices)]) #

        mask = target_plane_depths > 1e-4
        src_plane_depths = mask * src_depths # [b * num_tgt_planes_i, h, w]
        
        loss = {
                "loss_plane_depths":  torch.sum(torch.abs((src_plane_depths - target_plane_depths)*mask))/ torch.clamp(mask.sum(), min=1)
                }
        
        return loss
    
    # The useless loss from the previous experiment, to be deleted
    def loss_joint_corr_pose_param(self, outputs, targets, indices, num_planes_sum, log=True):
        
        T = outputs["pred_poses"].matrix().to(torch.float32) # (b,2,4,4)
        T = T[:,1] # (b,4,4)

        # pred_corrs_matrix = outputs["pred_corrs"] # (b, 30, 30)
        pred_corrs_matrix = outputs["attn_fundamentals"] # (b, 8, 30, 30)
        # pred_corrs_matrix = attn_fundamentals[:,[0,4]] # (b, 30, 30)
        b = pred_corrs_matrix.shape[0]
        src_param = outputs['pred_params'] # (2*b, 30, 3)

        param1 = src_param[:b] # (b, 30, 3) format (n/d)  plane:nX-d=0

        param2 = src_param[b:] # (b, 30, 3)
        # param2_ = torch.cat([param2, -torch.ones((b, nq, 1), dtype=param2.dtype)]) # (b, 30, 4)
        # param2_ = torch.bmm(param2_, T) # (b,30,4)*(b,4,4) -> (b, 30, 4)
        # param2_ = -param2_[:,:,:3]/param2_[:,:,3:] # (b, 30, 3)  ->format (n/d, -1)
        newparam1_ = get_tensorbatch_plane_params_in_global(param1, T) 

        loss_pose_param_l1 = torch.sum(torch.abs(newparam1_.unsqueeze(2) - param2.unsqueeze(1)), dim=-1) # (b,30,1,3)-(b,1,30,3)->(b,30,30,3)->(b,30,30)
        # loss_pose_param_delta_offset = torch.abs(offset1_.unsqueeze(2) - offset2.unsqueeze(1)) # (b, n1, n2)
        similarity = torch.nn.functional.cosine_similarity(newparam1_.unsqueeze(2), param2.unsqueeze(1), dim=-1)  # (b,30,1,3)-(b,1,30,3) -> (b,30,30)
        loss_pose_param_cos = 1-similarity

        pred_corrs_num = torch.sum(pred_corrs_matrix, dim = [-2,-1]) # (b, 8)
        loss_joint_corr_pose_param_l1 = torch.sum(torch.mul(pred_corrs_matrix, loss_pose_param_l1.unsqueeze(1)), dim = [-2,-1])/ pred_corrs_num  # (b, 8, 30, 30)*(b, 1, 30, 30) -> (b,8)/(b, 8)  -> (b,8)
        loss_joint_corr_pose_param_cos = torch.sum(torch.mul(pred_corrs_matrix, loss_pose_param_cos.unsqueeze(1)), dim = [-2,-1])/ pred_corrs_num

        loss_joint_corr_pose_param_l1 = torch.mean(loss_joint_corr_pose_param_l1) 
        loss_joint_corr_pose_param_cos = torch.mean(loss_joint_corr_pose_param_cos) 

        return {'loss_joint_corr_pose_param_l1': loss_joint_corr_pose_param_l1,
                'loss_joint_corr_pose_param_cos': loss_joint_corr_pose_param_cos}

    # The useless loss from the previous experiment, to be deleted
    def loss_joint_gtcorr_pose_gtparam(self, outputs, targets, indices, num_planes_sum, log=True):

        T = outputs["pred_poses"].matrix().to(torch.float32) # (b,2,4,4)
        T = T[:,1] # (b,4,4)
        b = T.shape[0]
        loss_joint_gtcorr_pose_gtparam_l1 = 0.
        loss_joint_gtcorr_pose_gtparam_cos = 0.


        for bi in range(b):
            gt_corrs = targets[bi]["gt_corrs"]
            gt_corr_params = []
            for i in range(2):
                gt_param = targets[int(bi+i*b)]["params"] # (planei_num,3)
                gt_corr_i = gt_corrs[:,i] # (corr_num,)
                gt_corr_param = gt_param[gt_corr_i] # (corr_num,3)
                gt_corr_params.append(gt_corr_param.unsqueeze(0)) # (1, corr_num, 3)
            Ti = T[[bi]]  # (1,4,4)
            newparam1_ = get_tensorbatch_plane_params_in_global(gt_corr_params[0], Ti) 

            loss_joint_gtcorr_pose_gtparam_l1 += torch.mean(torch.sum(torch.abs(newparam1_ - gt_corr_params[1]), dim=-1))  # (1, corr_num, 3) -> (1, corr_num) -> (1,)
            # loss_pose_param_delta_offset = torch.abs(offset1_.unsqueeze(2) - offset2.unsqueeze(1)) # (b, n1, n2)
            similarity = torch.nn.functional.cosine_similarity(newparam1_,  gt_corr_params[1], dim=-1)  # (1, corr_num, 3) -> (1, corr_num,)
            loss_joint_gtcorr_pose_gtparam_cos += torch.mean(1-similarity) # (1,)
            
            
        return {'loss_joint_gtcorr_pose_gtparam_l1': loss_joint_gtcorr_pose_gtparam_l1 / b,
                'loss_joint_gtcorr_pose_gtparam_cos': loss_joint_gtcorr_pose_gtparam_cos / b}
    
    # The useless loss from the previous experiment, to be deleted
    def loss_joint_01corr_pose_paramdetach(self, outputs, targets, indices, num_planes_sum, log=True):
        

        T = outputs["pred_poses"].matrix().to(torch.float32) # (b,2,4,4)
        T = T[:,1] # (b,4,4)

        # pred_corrs_matrix = outputs["pred_corrs"] # (b, 30, 30)
        pred_corrs_matrix = outputs["attn_fundamentals"] # (b, 8, 30, 30)
        # pred_corrs_matrix = attn_fundamentals[:,[0,4]] # (b, 30, 30)
        b = pred_corrs_matrix.shape[0]

        pred_assignments_matrix_list = [assignment_inference(False, pred_corrs_matrix[bi]) for bi in range(b)]
        pred_assignments_matrix = torch.stack(pred_assignments_matrix_list).to(torch.bool) #?(b,8,30,30)
        
        src_param = outputs['pred_params'].detach() # (2*b, 30, 3)

        param1 = src_param[:b] # (b, 30, 3) format (n/d)  plane:nX-d=0

        param2 = src_param[b:] # (b, 30, 3)
        abs_offset2 = 1/torch.abs(torch.norm(param2, dim = -1)) # (b, 30)
        
        newparam1_ = get_tensorbatch_plane_params_in_global(param1, T) # (b, 30, 3)
        abs_newoffset1_ = 1/torch.abs(torch.norm(newparam1_, dim = -1)) # (b, 30)


        # loss_pose_param_l1 = torch.sum(torch.abs(newparam1_.unsqueeze(2) - param2.unsqueeze(1)), dim=-1) # (b,30,1,3)-(b,1,30,3)->(b,30,30,3)->(b,30,30)
        loss_pose_param_offset = torch.abs(abs_newoffset1_.unsqueeze(2) - abs_offset2.unsqueeze(1)) # (b, n1, n2)
        similarity = torch.nn.functional.cosine_similarity(newparam1_.unsqueeze(2), param2.unsqueeze(1), dim=-1)  # (b,30,1,3)-(b,1,30,3) -> (b,30,30)
        loss_pose_param_cos = 1-similarity

        
        loss_joint_01corr_pose_paramdetach_offset = torch.masked_select(loss_pose_param_offset.unsqueeze(1), pred_assignments_matrix).mean()
        loss_joint_01corr_pose_paramdetach_cos = torch.masked_select(loss_pose_param_cos.unsqueeze(1), pred_assignments_matrix).mean()

        

        return {
            'loss_joint_01corr_pose_paramdetach_offset': loss_joint_01corr_pose_paramdetach_offset,
            'loss_joint_01corr_pose_paramdetach_cos': loss_joint_01corr_pose_paramdetach_cos}

    
    # from rel_pose
    def loss_pose(self, outputs, targets, indices, num_planes_sum, log=True):
        Gs = outputs["pred_poses"] # ?[b, 2, 7]
        Ps = SE3(torch.stack([targets[i]["gt_poses"] for i in range(Gs.shape[0])])) #? [b,2,7]

        ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0]) 

        dP = Ps[:,jj] * Ps[:,ii].inv()
        dG = Gs[:,jj] * Gs[:,ii].inv()
        d = (dG * dP.inv()).log() # torch.Size([16, 2, 6])

        tau, phi = d.split([3,3], dim=-1)  # torch.Size([16, 2, 3])
        geodesic_loss_tr = tau.norm(dim=-1).mean()
        geodesic_loss_rot = phi.norm(dim=-1).mean()

        losses = {
            'loss_pose_tr': geodesic_loss_tr,
            'loss_pose_rot': geodesic_loss_rot,
        }

        return losses
    
        # from rel_pose
    def loss_inverse_pose(self, outputs, targets, indices, num_planes_sum, log=True):
        Gs = outputs["pred_inverse_poses"] # ?[b, 2, 7] 
        Ps = SE3(torch.stack([targets[i]["gt_poses"] for i in range(Gs.shape[0])])) #? [b,2,7]

        ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0]) 

        dP = Ps[:,jj] * Ps[:,ii].inv()
        dG = Gs[:,jj] * Gs[:,ii].inv()
        d = (dG * dP.inv()).log() # torch.Size([16, 2, 6])

        tau, phi = d.split([3,3], dim=-1)  # torch.Size([16, 2, 3])
        geodesic_loss_tr = tau.norm(dim=-1).mean()
        geodesic_loss_rot = phi.norm(dim=-1).mean()

        losses = {
            'loss_inverse_pose_tr': geodesic_loss_tr,
            'loss_inverse_pose_rot': geodesic_loss_rot,
        }

        return losses
    
    
    def loss_corrs(self, outputs, targets, indices, num_planes_sum, log=True):
        pred_corrs_matrix = outputs["pred_corrs"] # (b, nc, 30, 30)
        b, nc, nq, _ = pred_corrs_matrix.shape
        #
        loss = 0
        for bi in range(b):
            gt_corrs = targets[bi]["gt_corrs"].to(torch.device("cpu"))
            
            new_idx_out = []
            for i in range(2):
                gt_corr = gt_corrs[:,i]
                indices_bi = indices[int(bi+i*b)]
                idx_out = indices_bi[0]
                idx_tgt = indices_bi[1]

                idxtgt_corr_index = argshuffle(idx_tgt, gt_corr)
                new_idx_out.append(idx_out[idxtgt_corr_index])

            # loss += torch.mean(-torch.log(pred_corrs_matrix[bi][(new_idx_out[0], new_idx_out[1])]))
            #!
            loss_nc = 0
            for ci in range(nc):
                loss_nc += torch.mean(-torch.log(pred_corrs_matrix[bi][ci][(new_idx_out[0], new_idx_out[1])]))
            loss += (loss_nc/nc)
        

        return {'loss_corrs': loss}


        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'params': self.loss_params,
            'Q': self.loss_Q,
            'center': self.loss_centers,
            'plane_depths': self.loss_plane_depths,
            'pose': self.loss_pose,
            'corrs': self.loss_corrs,
            'inverse_pose': self.loss_inverse_pose,
            'joint_corr_pose_param': self.loss_joint_corr_pose_param,
            'joint_gtcorr_pose_gtparam': self.loss_joint_gtcorr_pose_gtparam,
            'joint_01corr_pose_paramdetach': self.loss_joint_01corr_pose_paramdetach,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"} 
        # {'pred_logits': torch.Size([1, 100, 61]), 'pred_masks': torch.Size([1, 100, 120, 160])}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) # targets: {"label":tensor([0, 3, 1, 4, 6, 5, 2, 7, 8], device='cuda:0'), "masks": torch.Size([9, 480, 640]) }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets) # num_tgt_planes
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    # ! no pred_poses in aux_outputs
                    if loss in ["pose", "corrs", "inverse_pose", "joint_corr_pose_param", "joint_gtcorr_pose_gtparam", "joint_01corr_pose_paramdetach"]:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
