import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from .modules.extractor import ResidualBlock
from .vision_transformer import _create_vision_transformer
from lietorch import SE3
from detectron2.utils.registry import Registry
from detectron2.config import configurable

CAMERA_DECODER_REGISTRY = Registry("CAMERA_MODULE")
CAMERA_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in PlaneRecTR.
"""

def build_camera_decoder(cfg):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.CAMERA_MODULE.CAMERA_DECODER_NAME
    return CAMERA_DECODER_REGISTRY.get(name)(cfg)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class Plane2DPositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    all pixel positions -> plane level 2d positions
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, plane_centers):
        # ! plane_centers size: (b, nq, 2)  format: (y+1/h+eps,x+1/w+eps)
        
        y_embed = plane_centers[:,:,0]  # (b, nq)  from0-> from1
        x_embed = plane_centers[:,:,1] # (b, nq)

        if self.normalize:
            y_embed = y_embed * self.scale # (b, nq)
            x_embed = x_embed * self.scale # (b, nq)


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=plane_centers.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        
        pos_x = x_embed[:, :, None] / dim_t  # (b, nq, 1)/ (128,) -> (b, nq, 128)
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)  # (b, nq, 64, 2) -> (b, nq, 128)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2) # (b, nq, 64, 2) -> (b, nq, 128)
        
        pos = torch.cat((pos_y, pos_x), dim=2)  # (b, nq, 256)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

@CAMERA_DECODER_REGISTRY.register()
class ViTEss(nn.Module):
    @configurable
    def __init__(self, 
                 total_num_features,
                 num_patches,
                 feature_image_scale,
                 noess,
                 fc_hidden_size,
                 fusion_transformer,
                 num_heads,
                 cross_features,
                 use_single_softmax,
                 transformer_depth,
                 param_encoding,
                 center_encoding,
                 decoupling,
                 predict_corrs,
                 attn1_transpose,
                 attn2_transpose,
                 filter_cam_planes,
                 one_head_kq,
                 predict_inverse_poses,
                 consistent_mlp_layers,
                 cross_pp_embedding,
                 ):
        super().__init__()

        # hyperparams
        self.noess = noess
        self.total_num_features = total_num_features
        self.num_images = 2
        self.pose_size = 7
        
        self.num_patches = num_patches # num_queries
        self.feature_image_scale = feature_image_scale
        
        self.H2 = fc_hidden_size # 512

        self.transformer_depth = transformer_depth
        self.cross_features = cross_features 
        self.use_single_softmax = use_single_softmax
        
        self.param_encoding = param_encoding
        self.center_encoding = center_encoding
        
        self.attn1_transpose = attn1_transpose
        self.attn2_transpose = attn2_transpose
        self.filter_cam_planes = filter_cam_planes
        self.one_head_kq = one_head_kq
        self.predict_inverse_poses = predict_inverse_poses
        self.consistent_mlp_layers = consistent_mlp_layers
        self.cross_pp_embedding = cross_pp_embedding


        
        self.fusion_transformer = fusion_transformer
        if self.fusion_transformer: #True
            self.num_heads = num_heads
            model_kwargs = dict(patch_size=16, embed_dim=self.total_num_features, depth=self.transformer_depth, 
                                num_heads=self.num_heads, 
                                cross_features=self.cross_features,
                                use_single_softmax=self.use_single_softmax,
                                
                                noess=self.noess, 
                                
                                param_encoding=self.param_encoding,
                                center_encoding=self.center_encoding,
                                
                                attn1_transpose=self.attn1_transpose, attn2_transpose=self.attn2_transpose,
                                filter_cam_planes=self.filter_cam_planes,
                                one_head_kq=self.one_head_kq,
                                consistent_mlp_layers=self.consistent_mlp_layers,
                                cross_pp_embedding=self.cross_pp_embedding,
                                )
            self.fusion_transformer = _create_vision_transformer('vit_tiny_patch16_384', **model_kwargs)

            self.fusion_transformer.blocks = self.fusion_transformer.blocks[:self.transformer_depth]
            self.fusion_transformer.patch_embed = nn.Identity()
            self.fusion_transformer.head = nn.Identity() 
            self.fusion_transformer.cls_token = None
            self.pos_encoding = None

            # we overwrite pos_embedding as we don't have class token
            self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,self.num_patches,self.total_num_features])) 
            # randomly initialize as usual 
            nn.init.xavier_uniform_(self.fusion_transformer.pos_embed) 

            
            
            pos_enc = 0
            if self.param_encoding:
                pos_enc = 3
            if self.center_encoding:
                pos_enc = 6
            
            self.H = int(self.num_heads*2*(self.total_num_features//self.num_heads + pos_enc) * (self.total_num_features//self.num_heads)) # 26880
        
        self.pose_regressor = nn.Sequential(
            nn.Linear(self.H, self.H2),  # 26880 -> 512
            nn.ReLU(), 
            nn.Linear(self.H2, self.H2), 
            nn.ReLU(), 
            nn.Linear(self.H2, self.num_images * self.pose_size), # 14
            nn.Unflatten(1, (self.num_images, self.pose_size))
        )

        self.decoupling = decoupling
        if decoupling>0:
            self.pose_embed = MLP(total_num_features, total_num_features, total_num_features, decoupling)

        self.predict_corrs = predict_corrs
        if predict_corrs:
            self.corr_predictor = nn.Linear(num_heads*2, 1)

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["total_num_features"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_patches"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["feature_image_scale"] = (1/cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE, 
                              1/cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE)

        ret["noess"] = cfg.MODEL.CAMERA_MODULE.NOESS
        ret["fc_hidden_size"] = cfg.MODEL.CAMERA_MODULE.FC_HIDDEN_SIZE
        ret["fusion_transformer"] = cfg.MODEL.CAMERA_MODULE.FUSION_TRANSFORMER
        ret["num_heads"] = cfg.MODEL.CAMERA_MODULE.NUM_HEADS
        ret["cross_features"] = cfg.MODEL.CAMERA_MODULE.CROSS_FEATURES
        ret["use_single_softmax"] = cfg.MODEL.CAMERA_MODULE.USE_SINGLE_SOFTMAX
        
        ret["transformer_depth"] = cfg.MODEL.CAMERA_MODULE.TRANSFORMER_DEPTH
        
        ret["param_encoding"] = cfg.MODEL.CAMERA_MODULE.PARAM_ENCODING
        ret["center_encoding"] = cfg.MODEL.CAMERA_MODULE.CENTER_ENCODING
        
        ret["decoupling"] = cfg.MODEL.CAMERA_MODULE.DECOUPLING
        ret["predict_corrs"] = cfg.MODEL.CAMERA_MODULE.PREDICT_CORRS
        ret["attn1_transpose"] = cfg.MODEL.CAMERA_MODULE.ATTN1_TRANSPOSE
        ret["attn2_transpose"] = cfg.MODEL.CAMERA_MODULE.ATTN2_TRANSPOSE
        ret["filter_cam_planes"] = cfg.MODEL.CAMERA_MODULE.FILTER_CAM_PLANES
        ret["one_head_kq"] = cfg.MODEL.CAMERA_MODULE.ONE_HEAD_KQ
        ret["predict_inverse_poses"] = cfg.MODEL.CAMERA_MODULE.PREDICT_INVERSE_POSES
        ret["consistent_mlp_layers"] = cfg.MODEL.CAMERA_MODULE.CONSISTENT_MLP_LAYERS
        ret["cross_pp_embedding"] = cfg.MODEL.CAMERA_MODULE.CROSS_PP_EMBEDDING
        return ret
    
    def normalize_preds(self, Gs, pose_preds):
        pred_out_Gs = SE3(pose_preds)
        
        normalized = pred_out_Gs.data[:,:,3:].norm(dim=-1).unsqueeze(2)
        eps = torch.ones_like(normalized) * .01
        pred_out_Gs_new = SE3(torch.clone(pred_out_Gs.data))
        pred_out_Gs_new.data[:,:,3:] = pred_out_Gs.data[:,:,3:] / torch.max(normalized, eps)
        these_out_Gs = SE3(torch.cat([Gs[:,:1].data, pred_out_Gs_new.data[:,1:]], dim=1))
        
        out_Gs = these_out_Gs   # 1,2,6 ?

        inv_out_Gs = None
        if self.predict_inverse_poses:
            inv_out_Gs = SE3(torch.cat([pred_out_Gs_new.data[:,:1], Gs[:,:1].data], dim=1))

        return out_Gs, inv_out_Gs

   
    def forward(self, features, Gs, label_masks, plane_params, plane_centers, intrinsics=None, inference=False):
        """ Estimates SE3 between pair of frames """
        if not isinstance(Gs, SE3):
            Gs = SE3(torch.from_numpy(Gs).unsqueeze(0).cuda().float())

        if self.decoupling > 0:
            features = self.pose_embed(features)

       
        B, NI, NQ, NF = features.shape  # (1, 2, 20, 256)?
        features = features.reshape(B*NI, NQ, NF)

        if self.fusion_transformer is not None:
            x = features[:,:,:self.total_num_features] # torch.Size([2, 576, 192])
            x = self.fusion_transformer.patch_embed(x) # ? torch.Size([2, 576, 192])
            x = x + self.fusion_transformer.pos_embed # torch.Size([2, 576, 192]) + torch.Size([1, 576, 192])
            x = self.fusion_transformer.pos_drop(x) # ? torch.Size([2, 576, 192])

            for layer in range(self.transformer_depth):
                x, attn_fundamentals, query1, key2 = self.fusion_transformer.blocks[layer](x, label_masks, plane_params, plane_centers, intrinsics=intrinsics)

            features = self.fusion_transformer.norm(x) # torch.Size([b_s, 64, 256])
            
            attn_corr = None
            if self.predict_corrs:
                # attn_corr = self.corr_predictor(attn_fundamentals.permute(0,2,3,1))[:,:,:,0] # (b,8,30,30) -> (b,30,30)
                # attn_corr = torch.mean(attn_fundamentals, dim= 1) # (b,30,30)
                #!
                attn_corr = attn_fundamentals # (b,n,30,30)
        else:
            reshaped_features = features.reshape([-1,self.feature_resolution[0],self.feature_resolution[1],self.total_num_features])
            features = self.pool_transformer_output(reshaped_features.permute(0,3,1,2))

        if self.noess:
            # 12, 576, 192
            features = features.reshape([B,self.feature_resolution[0], self.feature_resolution[1],-1]).permute([0,3,1,2])
            pooled_features = self.pool_attn(features)
            pose_preds = self.pose_regressor(pooled_features.reshape([B, -1]))
        else: # √
            pose_preds = self.pose_regressor(features.reshape([B, -1])) # torch.Size([1, 2, 7])

        out_Gs, inv_out_Gs = self.normalize_preds(Gs, pose_preds)
        return out_Gs, inv_out_Gs, attn_fundamentals, attn_corr, query1, key2 # torch.Size([1, 2, 7])
