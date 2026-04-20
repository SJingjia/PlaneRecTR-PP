import copy
import numpy as np
import os
import torch
import pickle
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)
import pycocotools.mask as mask_util
from PIL import Image
import torchvision.transforms as transforms
from .augmentation import GaussianBlur
import cv2
from .scannetv1_plane_dataset_mapper import (
    NewFixedSizeCrop, 
    transforms_apply_intrinsic,
    build_transform_gen,
    dataset_precompute_K_inv_dot_xy_1,
    test_tfm_K,
    after_transform_apply_K_inv_dot_xy_1,
    get_plane_parameters,
    )

import logging
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform, TransformList

__all__ = ["PlaneRCNNMapper"]

import open3d as o3d
def fitting_planes(K_inv_dot_xy1, 
                   depth, 
                   segmentation, 
                   num_queries,
                   method='ransac',
                   ):
    
    xyz = K_inv_dot_xy1*depth

    planes = []
    svd_planes = []

    for i in np.unique(segmentation):
        if i == num_queries:
            break
        mask = segmentation==i
        masked_xyz = xyz[:,mask].copy().reshape(3, -1).T
        if method=='ransac':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(masked_xyz)

            # plane_model=[a,b,c,d'] ax + by + cz + d' = 0, d = -d'
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=50 if masked_xyz.shape[0]>1500 else 20,
                                         num_iterations=1000)
            plane_model = plane_model/np.linalg.norm(plane_model[:3])
            plane_model = plane_model[:3]/(-plane_model[3]) 
            planes.append(plane_model)

        else:
            print("method must be svd or ransac, not", method)

    return np.array(planes), np.array(svd_planes)


def phase2_img_transforms():
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    augmentation = [
        transforms.RandomApply([color_jitter], p=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
    ]
    img_transform = transforms.Compose(augmentation)
    return img_transform

def compute_xy_map(h, w):
    xy_map = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            xy_map[0, y, x] = float(x) / w
            xy_map[1, y, x] = float(y) / h

    return xy_map


from lietorch import SE3

import quaternion

def get_plane_params_in_global(planes, camera_info, pose_matrix):
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
    # end = planes
    end = (quaternion.as_rotation_matrix(rot) @ (end).T).T + tran  # cam2world
    newend = (pose_matrix[1,:3,:3] @ (end).T).T + tran
    a = end
    b = end - start
    planes_world = ((a * b).sum(axis=1) / np.linalg.norm(b, axis=1) ** 2).reshape(-1, 1) * b
    return planes_world

def get_tensorbatch_plane_params_in_global(planes, pose_matrixs):
    """
    input:
    @planes: plane params [b, n, 3] n/d
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys 
    @pose_matrixs: [b, 4, 4]
    output:
    plane parameters in global frame. n/d
    """

    planends =  planes/torch.sum(planes**2, dim=-1).unsqueeze(2)
    start = torch.ones_like(planends) * pose_matrixs[:,:3,3].unsqueeze(1) # (b, n, 3)* (b,1,3) -> b,n,3
    coor = torch.tensor([1, -1, -1], dtype=planends.dtype, device=planends.device) 
    end = planends * coor # suncg2habitat (b,n,3)*(3,) -> (b,n,3)

    end = torch.bmm(pose_matrixs[:,:3,:3], end.permute(0,2,1)).permute(0,2,1) + pose_matrixs[:,:3,3].unsqueeze(1)  # (b,3,3)*(b,3,n) -> (b,3,n)->(b,n,3)+(b,1,3)
    a = end # (b,n,3)
    b = end - start
    planends_world = ((a * b).sum(dim=-1) / torch.norm(b, dim=-1) ** 2).unsqueeze(-1) * b  # ((b,n)/(b,n))**2 -> (b,n,1)*(b,n,3) -> (b.n.3)
    planends_world *= coor
    planes_world = planends_world/torch.sum(planends_world**2, dim=-1).unsqueeze(2)
    return planes_world



# def test_loss_joint_corr_pose_param(pose_matrics, gt_corrs, gt_params1, gt_params2, camera_info):
        
#     T = pose_matrics.to(torch.float32) # (b,2,4,4)
#     T = T[:,1] # (b,4,4)

#     pred_corrs_matrix = gt_corrs # (b, 30, 30) (b, n1, n2)
#     b, n1, n2 = pred_corrs_matrix.shape
#     # src_param = gt_params # (2*b, 30, 3)

#     param1 = gt_params1 # (b, 30, 3) (b, n1, 3) format (n/d)  plane:nX-d=0  

#     param2 = gt_params2  # (b, 30, 3) (b, n2, 3)
#     # param1_ = torch.cat([param1, -torch.ones((b, n1, 1), dtype=param2.dtype)], dim=2) # (b, 30, 4) (b, n2, 4)
#     # param1_ = torch.bmm(param1_, T) # (b,30,4)*(b,4,4) -> (b, 30, 4) (b,n2,4)*(b,4,4) -> (b, n2, 4)
#     # param1_ = -param1_[:,:,:3]/param1_[:,:,3:] # (b, 30, 3)  (n, -d)->format (n/d, -1)   (b, n2, 3)

#     newparam1_ = get_tensorbatch_plane_params_in_global(param1, T) 


#     # param1nd = param1/torch.sum(param1**2, dim=-1).unsqueeze(2) # (n/d)/(1/d**2) = n*d
#     # param2nd = param2/torch.sum(param2**2, dim=-1).unsqueeze(2) # (n/d)/(1/d**2) = n*d
#     # nopesac_param1nd_ = torch.from_numpy(get_plane_params_in_global(param1nd[0].cpu().numpy(), camera_info, pose_matrix= pose_matrics[0].cpu().numpy())).unsqueeze(0)
#     # nopesac_param1_ = nopesac_param1nd_/torch.sum(nopesac_param1nd_**2, dim=-1).unsqueeze(2)  # (n*d)/(d**2) = n/d

#     # offset1_ = 1/torch.sqrt(torch.sum(param1_**2, dim=-1)) # (b, n1)
#     # offset2 = 1/torch.sqrt(torch.sum(param2**2, dim=-1))

#     # loss_pose_param_l1 = torch.sum(torch.abs(param1_.unsqueeze(2) - param2.unsqueeze(1)), dim=-1) # (b,30,1,3)-(b,1,30,3)->(b,30,30,3)->(b,30,30)
#     # loss_pose_param_delta_offset = torch.abs(offset1_.unsqueeze(2) - offset2.unsqueeze(1)) # (b, n1, n2)
#     # similarity = torch.nn.functional.cosine_similarity(param1_.unsqueeze(2), param2.unsqueeze(1), dim=-1)  # (b,30,1,3)-(b,1,30,3) -> (b,30,30)
#     # loss_pose_param_cos = 1-similarity
#     loss_pose_param_l1 = torch.sum(torch.abs(newparam1_.unsqueeze(2) - param2.unsqueeze(1)), dim=-1) # (b,30,1,3)-(b,1,30,3)->(b,30,30,3)->(b,30,30)
#     # loss_pose_param_delta_offset = torch.abs(offset1_.unsqueeze(2) - offset2.unsqueeze(1)) # (b, n1, n2)
#     similarity = torch.nn.functional.cosine_similarity(newparam1_.unsqueeze(2), param2.unsqueeze(1), dim=-1)  # (b,30,1,3)-(b,1,30,3) -> (b,30,30)
#     loss_pose_param_cos = 1-similarity

#     # nopesac_loss_pose_param_l1 = torch.sum(torch.abs(nopesac_param1_.unsqueeze(2) - param2.unsqueeze(1)), dim=-1) # (b,30,1,3)-(b,1,30,3)->(b,30,30,3)->(b,30,30)
#     # nopesac_similarity = torch.nn.functional.cosine_similarity(nopesac_param1_.unsqueeze(2), param2.unsqueeze(1), dim=-1)  # (b,30,1,3)-(b,1,30,3) -> (b,30,30)
#     # nopesac_loss_pose_param_cos = 1-nopesac_similarity

#     loss_joint_corr_pose_param_l1 = torch.sum(torch.mul(pred_corrs_matrix, loss_pose_param_l1))/ torch.sum(pred_corrs_matrix)
#     loss_joint_corr_pose_param_cos = torch.sum(torch.mul(pred_corrs_matrix, loss_pose_param_cos))/ torch.sum(pred_corrs_matrix)

#     # nopesac_loss_joint_corr_pose_param_l1 = torch.sum(torch.mul(pred_corrs_matrix, nopesac_loss_pose_param_l1))/ torch.sum(pred_corrs_matrix)
#     # nopesac_loss_joint_corr_pose_param_cos = torch.sum(torch.mul(pred_corrs_matrix, nopesac_loss_pose_param_cos))/ torch.sum(pred_corrs_matrix)


#     return {'loss_joint_corr_pose_param_l1': loss_joint_corr_pose_param_l1,
#             'loss_joint_corr_pose_param_cos': loss_joint_corr_pose_param_cos}

class PlaneRCNNMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.

    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True, dataset_names=None):
        self.cfg = cfg
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        
        self.root_dir = cfg.DATASETS.ROOT_DIR
        self.dataset_names = dataset_names
        assert dataset_names is not None

        self.use_mp3d = False
        self.use_scannet = False
        # self.use_7scenes = False
        for data_name in self.dataset_names:
            if 'mp3d' in data_name:
                self.use_mp3d = True
            if 'scannet' in data_name:
                self.use_scannet = True
            # if '7scenes' in data_name:
            #     self.use_7scenes = True


        assert (self.use_scannet & self.use_mp3d) is False

        self.is_train = is_train

        self.image_size = cfg.INPUT.IMAGE_SIZE
        self.common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE

        self.tfm_gens = build_transform_gen(cfg, is_train) if is_train else []
        logging.getLogger(__name__).info(
            "[class PlaneRCNNMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.phase2_tfm = cfg.MODEL.CAMERA_MODULE.PHASE2_TFM
        if self.phase2_tfm:
            self.phase2_img_transforms = phase2_img_transforms()

        self.predict_center = cfg.MODEL.MASK_FORMER.PREDICT_CENTER
        if self.predict_center:
            self.xy_map = compute_xy_map(h = int(self.image_size[0]), 
                                     w = int(self.image_size[1]))

    def __call__(self, dataset_dict):
        if self.use_mp3d:
            return self.call_mp3d(dataset_dict)
        elif self.use_scannet:
            return self.call_scannet(dataset_dict)
        # elif self.use_7scenes:
        #     return self.call_7scenes(dataset_dict)
        else:
            print("error in dataset_names:", self.dataset_names)
            return None


    def call_mp3d(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict) # dict_keys(['0', '1', 'rel_pose', 'gt_corrs'])
        # "0":dict_keys(['file_name', 'image_id', 'height', 'width', 'camera', 'annotations'])
        # ["annotations"][0]: dict_keys(['id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation', 'width', 'height', 'bbox_mode', 'plane'])
        for i in range(2):
            
            # 1. replace file name
            ps = dataset_dict[str(i)]["file_name"].split("/")
            # ps.insert(4, "frames")
            recent_file_name = os.path.join(self.root_dir, "/".join(ps[6:]))
            dataset_dict[str(i)]["file_name"] = recent_file_name

            # 2.1 read image
            image = utils.read_image(
                dataset_dict[str(i)]["file_name"], format=self.img_format
            )
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            dataset_dict[str(i)]["width"] = self.image_size[1]
            dataset_dict[str(i)]["height"] = self.image_size[0]
            utils.check_image_size(dataset_dict[str(i)], image) #!important
            # 2.2 transform image 
            if self.is_train:
                if self.phase2_tfm:
                    image = Image.fromarray(image)
                    dataset_dict[str(i)]["image"] = self.phase2_img_transforms(image) * 255.0
                else:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                    dataset_dict[str(i)]["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            else:
                dataset_dict[str(i)]["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            # 4. load mask maps
            house, img_id = dataset_dict[str(i)]["image_id"].split("_", 1)
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
            # dataset_dict[str(i)]["semantic_map"] = torch.as_tensor(
            #     semantic_map.astype("int32")
            # )
            plane_ids = np.unique(semantic_map)
            if plane_ids[0] == 0:
                plane_ids = plane_ids[1:]
            plane_num = len(plane_ids)

            plane_seg_gt = np.ones_like(semantic_map) * self.num_queries
            labels = []
            for label_id, pid in enumerate(plane_ids):
                plane_seg_gt[semantic_map==pid] = label_id
                labels.append(label_id)
            plane_seg_gt = plane_seg_gt.astype(np.uint8)
            plane_seg_gt = cv2.resize(plane_seg_gt, (self.image_size[1], self.image_size[0]), 
                                            interpolation=cv2.INTER_NEAREST) 

             # 3. compute K_inv_dot_xy_1
            focal_length = 517.97
            offset_x = 320
            offset_y = 240
            camera_K = np.array([[focal_length, 0, offset_x],
                                [0, focal_length, offset_y],
                                [0, 0, 1]]
                        )
            intrinsic_inv = np.linalg.inv(camera_K)
            
            dataset_K_inv_dot_xy_1, dataset_xy1 = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, 
                                                image_h = self.image_size[0], image_w = self.image_size[1])
            

            # 6. read plane params
            annos = [
                    obj
                    for obj in dataset_dict[str(i)].pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
            if len(annos) and "plane" in annos[0]:
                plane = [np.array(obj["plane"]).reshape(1,3) for obj in annos] # ! n*X - d = 0  obj["plane"]: n*d
                params = np.concatenate(plane)
                params /= np.sum(params**2, axis = 1).reshape(-1,1) # n*d -> n/d
                params = params.astype(np.float32)

            # 5. load depth map
            depth_gt = obs["depth_sensor"]
            depth_gt = cv2.resize(depth_gt.copy(), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
            
            plane_depth_gt = depth_gt
            
            if (not self.is_train) or self.phase2_tfm:
                            
                tfm_plane_depth_gt = plane_depth_gt
                tfm_labels = labels
                dataset_dict[str(i)]["K_inv_dot_xy_1"] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32))

            else:
                # 7. del ColorTransform for seg, depth and dataset_K_inv_dot_xy_1
                new_transforms = []
                for t in transforms.transforms:
                    if t.__class__!=T.ColorTransform:
                        new_transforms.append(t)
                new_transforms = TransformList(new_transforms)
                # 8. apply new_transforms
                plane_seg_gt = new_transforms.apply_segmentation(plane_seg_gt)
                plane_depth_gt = new_transforms.apply_image(plane_depth_gt) # [ h, w]  # interp="nearest"
                # depth_gt = new_transforms.apply_image(depth_gt) 

                tfm_intrinsic, random_scale, new_h, new_w = transforms_apply_intrinsic(transforms, camera_K)
                dataset_K_inv_dot_xy_1 = dataset_K_inv_dot_xy_1.transpose(1,2,0) # (image_h, image_w, 3)
                tfm_dataset_K_inv_dot_xy_1 = new_transforms.apply_image(dataset_K_inv_dot_xy_1) # interp = 'bilinear'
                tfm_dataset_K_inv_dot_xy_1 = tfm_dataset_K_inv_dot_xy_1.transpose(2, 0, 1) # (3, 480, 640)
                dataset_dict[str(i)]["K_inv_dot_xy_1"] = torch.from_numpy(tfm_dataset_K_inv_dot_xy_1.astype(np.float32))
                
                tfm_plane_depth_gt = plane_depth_gt
                
                tfm_labels = np.unique(plane_seg_gt)
                tfm_labels = tfm_labels[tfm_labels<self.num_queries]
                
                params = params[tfm_labels]
                


            # 9. add data to dataset_dict
            # 9.1 classes params
            dataset_dict[str(i)]["classes"] = torch.tensor([1]*len(tfm_labels), dtype=torch.int64)
            dataset_dict[str(i)]["params"] = torch.from_numpy(params)
            # 9.2 
            dsize = (int(self.image_size[1]/self.common_stride), 
                     int(self.image_size[0]/self.common_stride))
            d_ = cv2.resize(tfm_plane_depth_gt.copy(), dsize, interpolation=cv2.INTER_AREA)
            
            masks = []
            seg_depths = []
            tfm_plane_depths = []
            tfm_resize14_plane_depths = []


            if self.is_train:
                if np.sum(np.abs(tfm_plane_depth_gt) > 200)>0:
                    print("WARNING:", np.sum(np.abs(tfm_plane_depth_gt) > 200))
                valid_region = np.abs(tfm_plane_depth_gt) < 200.0
                resize_valid_region = np.abs(d_) < 200.0
            else:
                valid_region = np.ones_like(tfm_plane_depth_gt)
                resize_valid_region = np.ones_like(d_)

            for label_id in tfm_labels:
                mask = (plane_seg_gt==label_id) * valid_region
                m_ = cv2.resize(mask.copy().astype(np.float32), dsize, interpolation=cv2.INTER_NEAREST)
                m_ *= resize_valid_region
                masks.append(mask)
                tfm_plane_depths.append(torch.from_numpy((mask*tfm_plane_depth_gt).astype(np.float32)))
                seg_depths.append(torch.from_numpy(mask*depth_gt))
                tfm_resize14_plane_depths.append(torch.from_numpy(m_*d_))
            
            # del d_

            plane_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(m.copy())) for m in masks])
                )
            dataset_dict[str(i)]["plane_masks"] = plane_masks
            tfm_plane_depths = torch.stack(tfm_plane_depths, dim = 0)
            seg_depths = torch.stack(seg_depths, dim = 0)
            dataset_dict[str(i)]["plane_depths"] = tfm_plane_depths
            dataset_dict[str(i)]["depths"] = seg_depths
            tfm_resize14_plane_depths = torch.stack(tfm_resize14_plane_depths, dim = 0)
            dataset_dict[str(i)]["resize14_plane_depths"] = tfm_resize14_plane_depths
            
        focal_x = camera_K[0][0] * self.image_size[1] / 640
        focal_y = camera_K[1][1] * self.image_size[0] / 480
        offset_x = camera_K[0][2] * self.image_size[1] / 640
        offset_y = camera_K[1][2] * self.image_size[0] / 480
        dataset_dict["intrinsics"] = torch.from_numpy(np.stack(
            [np.array([[focal_x, focal_y, offset_x, offset_y], 
                        [focal_x, focal_y, offset_x, offset_y]])]).astype(np.float32))

        dataset_dict["width"] = self.image_size[1]
        dataset_dict["height"] = self.image_size[0]

        
        return dataset_dict

    def call_scannet(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict) # dict_keys(['0', '1', 'rel_pose', 'gt_corrs'])
        # "0":dict_keys(['image_id', 'file_name', 'height', 'width', 'annotations', 'gt_plane_num'])
        # ["annotations"][0]: dict_keys(['id', 'image_id', 'category_id', 'area', 'segmentation', 'width', 'height', 'plane', 'iscrowd', 'bbox', 'bbox_mode'])
        for i in range(2):
            
            # 1. replace file name
            ps = dataset_dict[str(i)]["file_name"].split("/")
            ps.insert(4, "frames")
            recent_file_name = os.path.join(self.root_dir, "/".join(ps[2:]))
            dataset_dict[str(i)]["file_name"] = recent_file_name

            # 2.1 read image
            image = utils.read_image(
                dataset_dict[str(i)]["file_name"], format=self.img_format
            )
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            dataset_dict[str(i)]["width"] = self.image_size[1]
            dataset_dict[str(i)]["height"] = self.image_size[0]
            utils.check_image_size(dataset_dict[str(i)], image) #!important
            # 2.2 transform image 
            if self.is_train:
                if self.phase2_tfm:
                    image = Image.fromarray(image)
                    dataset_dict[str(i)]["image"] = self.phase2_img_transforms(image) * 255.0
                else:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                    dataset_dict[str(i)]["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            else:
                dataset_dict[str(i)]["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            # 4. load mask maps
            image_id = dataset_dict[str(i)]["image_id"]
            scene_idx, image_idx = image_id.split('-')
            masks_path = os.path.join(
                self.root_dir,
                "twoView_Anns",
                scene_idx,
                image_idx + ".pkl",
            )
            with open(masks_path, "rb") as f:
                obs = pickle.load(f) # dict_keys(['plane_masks', 'camera_K'])
            plane_masks = obs['plane_masks'] # BitMasks(num_instances=10)
            # !    
            # if plane_masks.shape[-1]!= self.image_size[1]:                       
            plane_masks = np.transpose(plane_masks.tensor.numpy(), (1,2,0))
            plane_masks = cv2.resize(plane_masks.astype(np.uint8), (self.image_size[1], self.image_size[0]), 
                                            interpolation=cv2.INTER_NEAREST) 
            plane_masks = np.transpose(plane_masks, (2,0,1)).astype(np.bool)
            
            plane_seg_gt = np.ones_like(plane_masks[0]) * self.num_queries
            plane_seg_gt = plane_seg_gt.astype(np.uint8)
            labels = []
            for label_id, pm in enumerate(plane_masks):
                plane_seg_gt[pm] = label_id
                labels.append(label_id)

            # plane_num = len(plane_masks)
            # assert plane_num == dataset_dict[str(i)]["gt_plane_num"]

            # dataset_dict[str(i)]["plane_masks"] = plane_masks

             # 3. compute K_inv_dot_xy_1
            camera_K = obs['camera_K']  # 3, 3
            intrinsic_inv = np.linalg.inv(camera_K)
            
            dataset_K_inv_dot_xy_1, dataset_xy1 = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, 
                                                image_h = self.image_size[0], image_w = self.image_size[1])
            

            # 6. read plane params
            annos = [
                    obj
                    for obj in dataset_dict[str(i)].pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
            if len(annos) and "plane" in annos[0]:
                plane = [np.array(obj["plane"]).reshape(1,3) for obj in annos] # ! n*X - d = 0  obj["plane"]: n*d
                params = np.concatenate(plane)
                params /= np.sum(params**2, axis = 1).reshape(-1,1) # n*d -> n/d
                params = params.astype(np.float32)
            # dataset_dict[str(i)]["params"] = gt_planes


            # 5. load depth map
            depth_path = dataset_dict[str(i)]["file_name"]
            depth_path = depth_path.replace('color', 'depth').replace('.jpg', '.png')
            depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
            depth_gt = cv2.resize(depth_gt.copy(), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
            
            plane_depth_gt = depth_gt
       
            if (not self.is_train) or self.phase2_tfm:
                            
                tfm_plane_depth_gt = plane_depth_gt
                tfm_labels = labels
                dataset_dict[str(i)]["K_inv_dot_xy_1"] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32))

            else:
                # 7. del ColorTransform for seg, depth and dataset_K_inv_dot_xy_1
                new_transforms = []
                for t in transforms.transforms:
                    if t.__class__!=T.ColorTransform:
                        new_transforms.append(t)
                new_transforms = TransformList(new_transforms)
                # 8. apply new_transforms
                plane_seg_gt = new_transforms.apply_segmentation(plane_seg_gt)
                plane_depth_gt = new_transforms.apply_image(plane_depth_gt) # [ h, w]  # interp="nearest"
                # depth_gt = new_transforms.apply_image(depth_gt) 

                tfm_intrinsic, random_scale, new_h, new_w = transforms_apply_intrinsic(transforms, camera_K)
                dataset_K_inv_dot_xy_1 = dataset_K_inv_dot_xy_1.transpose(1,2,0) # (image_h, image_w, 3)
                tfm_dataset_K_inv_dot_xy_1 = new_transforms.apply_image(dataset_K_inv_dot_xy_1) # interp = 'bilinear'
                tfm_dataset_K_inv_dot_xy_1 = tfm_dataset_K_inv_dot_xy_1.transpose(2, 0, 1) # (3, 480, 640)
                dataset_dict[str(i)]["K_inv_dot_xy_1"] = torch.from_numpy(tfm_dataset_K_inv_dot_xy_1.astype(np.float32))
                
                tfm_plane_depth_gt = plane_depth_gt
                
                tfm_labels = np.unique(plane_seg_gt)
                tfm_labels = tfm_labels[tfm_labels<self.num_queries]
                
                params = params[tfm_labels]
                


            # 9. add data to dataset_dict
            # 9.1 classes params
            dataset_dict[str(i)]["classes"] = torch.tensor([1]*len(tfm_labels), dtype=torch.int64)
            dataset_dict[str(i)]["params"] = torch.from_numpy(params)
            # 9.2 
            dsize = (int(self.image_size[1]/self.common_stride), 
                     int(self.image_size[0]/self.common_stride))
            d_ = cv2.resize(tfm_plane_depth_gt.copy(), dsize, interpolation=cv2.INTER_AREA)
            
            masks = []
            seg_depths = []
            tfm_plane_depths = []
            tfm_resize14_plane_depths = []
            # resize14_plane_centers = []
            plane_centers = []  #!

            #!
            if self.is_train:
                if np.sum(np.abs(tfm_plane_depth_gt) > 200)>0:
                    print("WARNING:", np.sum(np.abs(tfm_plane_depth_gt) > 200))
                valid_region = np.abs(tfm_plane_depth_gt) < 200.0
                resize_valid_region = np.abs(d_) < 200.0
            else:
                valid_region = np.ones_like(tfm_plane_depth_gt)
                resize_valid_region = np.ones_like(d_)

            for label_id in tfm_labels:
                mask = (plane_seg_gt==label_id) * valid_region
                m_ = cv2.resize(mask.copy().astype(np.float32), dsize, interpolation=cv2.INTER_NEAREST)
                m_ *= resize_valid_region

                if self.predict_center:
                    pixel_num = mask.sum()
                    x_map = self.xy_map[0] * mask
                    y_map = self.xy_map[1] * mask
                    x_sum = x_map.sum()
                    y_sum = y_map.sum()
                    plane_x = x_sum / pixel_num
                    plane_y = y_sum / pixel_num
                    plane_centers.append(torch.FloatTensor([plane_x, plane_y]))

                masks.append(mask)
                tfm_plane_depths.append(torch.from_numpy((mask*tfm_plane_depth_gt).astype(np.float32)))
                seg_depths.append(torch.from_numpy(mask*depth_gt))
                tfm_resize14_plane_depths.append(torch.from_numpy(m_*d_))
            
            # del d_

            plane_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(m.copy())) for m in masks])
                )
            dataset_dict[str(i)]["plane_masks"] = plane_masks
            tfm_plane_depths = torch.stack(tfm_plane_depths, dim = 0)
            seg_depths = torch.stack(seg_depths, dim = 0)
            dataset_dict[str(i)]["plane_depths"] = tfm_plane_depths
            dataset_dict[str(i)]["depths"] = seg_depths
            tfm_resize14_plane_depths = torch.stack(tfm_resize14_plane_depths, dim = 0)
            dataset_dict[str(i)]["resize14_plane_depths"] = tfm_resize14_plane_depths
            if self.predict_center:
                dataset_dict[str(i)]["plane_centers"] = torch.stack(plane_centers, dim = 0)
            
        focal_x = camera_K[0][0] * self.image_size[1] / 640
        focal_y = camera_K[1][1] * self.image_size[0] / 480
        offset_x = camera_K[0][2] * self.image_size[1] / 640
        offset_y = camera_K[1][2] * self.image_size[0] / 480
    
        dataset_dict["intrinsics"] = torch.tensor([focal_x, focal_y, offset_x, offset_y], dtype = torch.float32) # (4)
        # ?
        dataset_dict["width"] = self.image_size[1]
        dataset_dict["height"] = self.image_size[0]

        
        return dataset_dict
    
    

if __name__ == "__main__":
    pass