
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_helpers import build_model_with_cfg, named_apply
from .vit_layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_



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


def get_center_positional_encodings(B, N, centers, intrinsics=None):

    xs = centers[:,:,0]*2-1 # B,N  [0,1] -> [-1,1]
    ys = centers[:,:,1]*2-1 # B,N  [0,1] -> [-1,1]
    p3 = xs
    p4 = ys

    if intrinsics is not None:

        fx, fy, cx, cy = intrinsics.unbind(dim=-1) # (B,) (B,), (B,), (B,)

        hpix = cy * 2
        wpix = cx * 2
        # map to between -1 and 1
        fx_normalized = (fx / wpix) * 2
        cx_normalized = (cx / wpix) * 2 - 1 
        fy_normalized = (fy / hpix) * 2
        cy_normalized = (cy / hpix) * 2 - 1
        # in fixed case, if we are mapping rectangular img with width > height,
        # then fy will be > fx and therefore p3 will be both greater than -1 and less than 1. ("y is zoomed out")
        # p4 will be -1 to 1.

        K = torch.zeros([B,3,3], device = centers.device)
        K[:,0,0] = fx_normalized
        K[:,1,1] = fy_normalized
        K[:,0,2] = cx_normalized
        K[:,1,2] = cy_normalized
        K[:,2,2] = 1
    
        Kinv = torch.inverse(K) # (B,3,3)

        uv1 = torch.stack([xs, ys, torch.ones_like(xs)]).permute((1,0,2)) # (3,B,N)->(B,3,N)
        k_inv_xy1 = torch.bmm(Kinv, uv1) # (B,3,N)

        p3 = torch.div(k_inv_xy1[:,0], k_inv_xy1[:,2])
        p4 = torch.div(k_inv_xy1[:,1], k_inv_xy1[:,2]) # (B,N)

    p2 = torch.mul(p3, p4)
    p1 = torch.mul(p4, p4)
    p0 = torch.mul(p3, p3)
    positional = torch.stack([p0,p1,p2,p3,p4,torch.ones_like(p0)], dim=2) # (B,N,6)

    return positional # torch.Size([1, 576, 6])

class CrossAttention(nn.Module):
    """
    Our custom Cross-Attention Block. Have options to use dual softmax, 
    add positional encoding and use bilinear attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., 
                proj_drop=0., cross_features=False, 
                use_single_softmax=False, 
                
                noess=False, 
                
                param_encoding=False,
                center_encoding=False,
                
                filter_cam_planes=False,
                one_head_kq=True,
                consistent_mlp_layers=0,
                cross_pp_embedding=False,
                ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_scale = head_dim ** -0.5
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if noess:
            self.proj = nn.Linear(dim, dim)
        else:
            # if no_pos_encoding:
            if param_encoding:
                self.proj_fundamental = nn.Linear(dim+int(3*self.num_heads), dim)
            elif center_encoding:
                self.proj_fundamental = nn.Linear(dim+int(6*self.num_heads), dim)
            else:
                self.proj_fundamental = nn.Linear(dim, dim)
            # else:
            #     self.proj_fundamental = nn.Linear(dim+int(6*self.num_heads), dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cross_pp_embedding = cross_pp_embedding
        if self.cross_pp_embedding:
            N_steps = dim // 2
            self.p2pe_layer = Plane2DPositionEmbeddingSine(N_steps, normalize=True)

        self.cross_features = cross_features
        self.use_single_softmax = use_single_softmax
        
        self.noess = noess
        
        self.param_encoding = param_encoding
        self.center_encoding = center_encoding
        
        self.filter_cam_planes = filter_cam_planes
        self.one_head_kq = one_head_kq
        self.consistent_mlp_layers = consistent_mlp_layers
        if self.consistent_mlp_layers > 0:
            self.consisten_mlp = MLP(dim, dim*2, dim, self.consistent_mlp_layers)


    def forward(self, x1, x2, label_masks=None, plane_params=None, plane_centers=None, intrinsics=None):
        # plane_centers: range:[0-1], size:(b, n_images=2, nq, 2)
        B, N, C = x1.shape # 1, 576, 192, 192/3=64
        # torch.Size([3, 1, 3, 576, 64]) -> (3, 1, 4, 30, 64)
        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]   # make torchscript happy (cannot use tensor as tuple)

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)

        if not self.noess:
            
            if self.one_head_kq:
                new_q1 = q1.permute(0,2,1,3).reshape(B,N,-1)
                new_q2 = q2.permute(0,2,1,3).reshape(B,N,-1)
                new_k1 = k1.permute(0,2,1,3).reshape(B,N,-1)
                new_k2 = k2.permute(0,2,1,3).reshape(B,N,-1)
                if self.consistent_mlp_layers > 0:
                    new_k1 = self.consisten_mlp(new_k1)
                    new_k2 = self.consisten_mlp(new_k2)
                    new_q1 = self.consisten_mlp(new_q1)
                    new_q2 = self.consisten_mlp(new_q2)
                
                if self.cross_pp_embedding:
                    pp_pos1 = self.p2pe_layer(plane_centers[:,0])
                    pp_pos2 = self.p2pe_layer(plane_centers[:,1])
                    new_q1 += pp_pos1
                    new_q2 += pp_pos2
                    new_k1 += pp_pos1
                    new_k2 += pp_pos2

                attn_1 = (new_q2 @ new_k1.transpose(-2, -1)) * self.scale # torch.Size([1,30, 30])
                attn_2 = (new_q1 @ new_k2.transpose(-2, -1)) * self.scale 
                attn_1 = attn_1.unsqueeze(1).repeat(1,self.num_heads,1,1)
                attn_2 = attn_2.unsqueeze(1).repeat(1,self.num_heads,1,1)
            else:
                attn_1 = (q2 @ k1.transpose(-2, -1)) * self.head_scale # torch.Size([1, 4, 30, 30])
                attn_2 = (q1 @ k2.transpose(-2, -1)) * self.head_scale
                #!!
                new_q1 = q1
                new_k2 = k2


            if self.filter_cam_planes:
                # label_mask: (b, 30, 30)
                label_masks = label_masks.unsqueeze(1).repeat(1,self.num_heads,1,1) # (1,4,30,30)
                attn_1 *= label_masks.permute((0,1,3,2)) # !
                attn_2 *= label_masks

            if self.use_single_softmax: 
                attn_fundamental_1 = attn_1.softmax(dim=-1)
                attn_fundamental_2 = attn_2.softmax(dim=-1)
            else: # √ A = torch.Size([1, 3, 576, 576]) -> torch.Size([1, 4, num_queries, num_queries])
                attn_fundamental_1 = attn_1.softmax(dim=-1) * attn_1.softmax(dim=-2)
                attn_fundamental_2 = attn_2.softmax(dim=-1) * attn_2.softmax(dim=-2)

            if self.param_encoding:
                # (b, 2, num_queries, 3)
                ext_param1 = plane_params[:,0]
                ext_param2 = plane_params[:,1]
                # (b, num_heads=4, num_queries, 64+3)
                v1 = torch.cat([v1,ext_param1.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3) 
                v2 = torch.cat([v2,ext_param2.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)
            
            if self.center_encoding:
                center_p1 = get_center_positional_encodings(B, N, plane_centers[:,0], intrinsics=intrinsics) # shape B,N,6
                center_p2 = get_center_positional_encodings(B, N, plane_centers[:,1], intrinsics=intrinsics)

                v1 = torch.cat([v1,center_p1.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)
                v2 = torch.cat([v2,center_p2.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)
                
            if self.cross_features:
                fundamental_1 = (v2.transpose(-2, -1) @ attn_fundamental_1) @ v1
                fundamental_2 = (v1.transpose(-2, -1) @ attn_fundamental_2) @ v2
            
            else: # √
                fundamental_1 = (v1.transpose(-2, -1) @ attn_fundamental_1) @ v1 # torch.Size([1, 4, 64, 64])
                fundamental_2 = (v2.transpose(-2, -1) @ attn_fundamental_2) @ v2

            # if self.no_pos_encoding:
            if self.param_encoding: # torch.Size([1, 67, 268])
                fundamental_1 = fundamental_1.reshape(B, int(C+3*self.num_heads), int((C+3*self.num_heads)/self.num_heads)).transpose(-2,-1)           
                fundamental_2 = fundamental_2.reshape(B, int(C+3*self.num_heads), int((C+3*self.num_heads)/self.num_heads)).transpose(-2,-1)
            elif self.center_encoding:
                fundamental_1 = fundamental_1.reshape(B, int(C+6*self.num_heads), int((C+6*self.num_heads)/self.num_heads)).transpose(-2,-1)           
                fundamental_2 = fundamental_2.reshape(B, int(C+6*self.num_heads), int((C+6*self.num_heads)/self.num_heads)).transpose(-2,-1)
            else:
                # torch.Size([1, 64, 256])
                fundamental_1 = fundamental_1.reshape(B, int(C), int(C/self.num_heads)).transpose(-2,-1)           
                fundamental_2 = fundamental_2.reshape(B, int(C), int(C/self.num_heads)).transpose(-2,-1)
           
            # fundamental is C/3+6,C/3+6 (for each head)
            # torch.Size([1, 70, 210]) -> (1, 64, 256)
            fundamental_2 = self.proj_fundamental(fundamental_2)
            fundamental_1 = self.proj_fundamental(fundamental_1)

            # we flip these: we want x1 to be (q1 @ k2) @ v2
            # impl is similar to ViLBERT
            return fundamental_2, fundamental_1, attn_fundamental_2, attn_fundamental_1, new_q1, new_k2
        else:
            # q2, k1, v1
            attn_1 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn_1 = attn_1.softmax(dim=-1)
            attn_1 = self.attn_drop(attn_1)

            x1 = (attn_1 @ v1).transpose(1, 2).reshape(B, N, C)

            # q1, k2, v2
            attn_2 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn_2 = attn_2.softmax(dim=-1)
            attn_2 = self.attn_drop(attn_2)

            x2 = (attn_2 @ v2).transpose(1, 2).reshape(B, N, C)
            
            x1 = self.proj(x1)
            x2 = self.proj(x2)

            x1 = self.proj_drop(x1)
            x2 = self.proj_drop(x2)

            # we flip these: we want x1 to be (q1 @ k2) @ v2
            # impl is similar to ViLBERT
            return x2, x1 


class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, cross_features=False,
                 use_single_softmax=False, 
                 noess=False, 
                 param_encoding=False,
                 center_encoding=False,
                 attn1_transpose=False, attn2_transpose=False, 
                 filter_cam_planes=False,
                 one_head_kq=True,
                 consistent_mlp_layers=0,
                 cross_pp_embedding=False,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop,
                                cross_features=cross_features, 
                                use_single_softmax=use_single_softmax, 
                                
                                noess=noess, 
                                
                                param_encoding=param_encoding,
                                center_encoding=center_encoding,
                                
                                filter_cam_planes=filter_cam_planes,
                                one_head_kq=one_head_kq,
                                consistent_mlp_layers=consistent_mlp_layers,
                                cross_pp_embedding=cross_pp_embedding,
                                )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.noess = noess
        self.attn1_transpose = attn1_transpose
        self.attn2_transpose = attn2_transpose

    def forward(self, x, label_masks=None, plane_params=None, plane_centers=None, intrinsics=None):
        b_s, h_w, nf = x.shape # 2, 576,192
        x = x.reshape([-1, 2, h_w, nf])
        x1_in = x[:,0]
        x2_in = x[:,1]

        if not self.noess: # Not false
            fundamental1, fundamental2, attn_fundamental_1, attn_fundamental_2, query1, key2 = self.cross_attn(self.norm1(x1_in), self.norm1(x2_in), label_masks, plane_params, plane_centers, intrinsics=intrinsics) # torch.Size([1, 64, 256])
            
            attn_fundamental_1 = attn_fundamental_1.permute((0,1,3,2)) if self.attn1_transpose else attn_fundamental_1
            attn_fundamental_2 = attn_fundamental_2.permute((0,1,3,2)) if self.attn2_transpose else attn_fundamental_2
            attn_fundamentals = torch.cat([attn_fundamental_1, attn_fundamental_2], dim = 1) # [1, num_heads*2, num_queries, num_queries]
            
            fundamental_inter = torch.cat([fundamental1.unsqueeze(1), fundamental2.unsqueeze(1)], dim=1) # torch.Size([1, 2, 64, 256])
            fundamental = fundamental_inter.reshape(b_s, -1, nf)
            fundamental = fundamental + self.drop_path(self.mlp(self.norm2(fundamental)))
            
            return fundamental, attn_fundamentals, query1, key2 # torch.Size([2, 64/64+3, 256])
        
        else:
            x1, x2, _, _ = self.cross_attn(self.norm1(x1_in), self.norm1(x2_in), label_masks, plane_params, plane_centers, intrinsics=intrinsics)
            x_inter = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
            x_inter = x_inter.reshape(b_s, h_w, nf)
            x = x.reshape(b_s, h_w, nf)
            x = x + self.drop_path(x_inter)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, None


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim_in = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_in, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    
    def forward(self, x, label_masks=None, plane_params=None, plane_centers=None, intrinsics=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)           

    def forward(self, x, label_masks=None, plane_params=None, plane_centers=None, intrinsics=None):
        identity = x 
        out = x
        tmp, attn = self.drop_path(self.attn(self.norm1(out), label_masks, plane_params, plane_centers, intrinsics))
        x = identity + tmp
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', cross_features=False,
                 use_single_softmax=False, 
                 noess=False, 
                 param_encoding=False,
                 center_encoding=False,
                 attn1_transpose=False, attn2_transpose=False,
                 filter_cam_planes=False,
                 one_head_kq=True,
                 consistent_mlp_layers=0,
                 cross_pp_embedding=False,
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_list = []
        for i in range(depth):
            if i == depth - 1:
                this_block = CrossBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                            drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                            cross_features=cross_features,
                            use_single_softmax=use_single_softmax, 
                            
                            noess=noess, 
                            
                            param_encoding=param_encoding,
                            center_encoding=center_encoding,
                            
                            attn1_transpose=attn1_transpose,
                            attn2_transpose=attn2_transpose,
                            filter_cam_planes=filter_cam_planes,
                            one_head_kq=one_head_kq,
                            consistent_mlp_layers=consistent_mlp_layers,
                            cross_pp_embedding=cross_pp_embedding,
                            )
            else:
                this_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                            drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            block_list.append(this_block)
        self.blocks = nn.Sequential(*block_list)
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
}

def _create_vision_transformer(variant, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)

    model = build_model_with_cfg(
        VisionTransformer, variant,
        default_cfg=default_cfg,
        **kwargs)
    return model