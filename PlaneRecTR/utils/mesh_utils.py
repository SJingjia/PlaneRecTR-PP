import cv2
import os
import shutil
import quaternion
import torch
import numpy as np
from typing import Optional
import imageio
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import utils as struct_utils
from PlaneRecTR.utils.camera import (
    create_cylinder_mesh,
    create_color_palette,
    get_cone_edges,
)


import mapbox_earcut as earcut
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures

from pycocotools import mask
from .pycococreatortools import binary_mask_to_polygon
from detectron2.structures.masks import polygons_to_bitmask


def transform_meshes(meshes, camera_info):
    """
    input:
    @meshes: mesh in local frame
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    mesh in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    verts_packed = meshes.verts_packed()
    verts_packed = verts_packed * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )  # suncg2habitat
    faces_list = meshes.faces_list()
    tex = meshes.textures
    rot_matrix = torch.tensor(quaternion.as_rotation_matrix(rot), dtype=torch.float32)
    verts_packed = torch.mm(rot_matrix, verts_packed.T).T + torch.tensor(
        tran, dtype=torch.float32
    )
    verts_list = list(verts_packed.split(meshes.num_verts_per_mesh().tolist(), dim=0))
    return Meshes(verts=verts_list, faces=faces_list, textures=tex)


def rotate_mesh_for_webview(meshes):
    """
    input:
    @meshes: mesh in global (habitat) frame
    output:
    mesh is rotated around x axis by -11 degrees such that floor is horizontal
    """
    verts_packed = meshes.verts_packed()
    faces_list = meshes.faces_list()
    tex = meshes.textures
    rot_matrix = torch.FloatTensor(
        np.linalg.inv(
            np.array([[1, 0, 0], [0, 0.9816272, -0.1908090], [0, 0.1908090, 0.9816272]])
        )
    )
    verts_packed = torch.mm(rot_matrix, verts_packed.T).T
    verts_list = list(verts_packed.split(meshes.num_verts_per_mesh().tolist(), dim=0))
    return Meshes(verts=verts_list, faces=faces_list, textures=tex)


def transform_verts_list(verts_list, camera_info):
    """
    input:
    @meshes: verts_list in local frame
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    verts_list in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    verts_list_to_packed = struct_utils.list_to_packed(verts_list)
    verts_packed = verts_list_to_packed[0]
    num_verts_per_mesh = verts_list_to_packed[1]
    verts_packed = verts_packed * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )  # suncg2habitat
    rot_matrix = torch.tensor(quaternion.as_rotation_matrix(rot), dtype=torch.float32)
    verts_packed = torch.mm(rot_matrix, verts_packed.T).T + torch.tensor(
        tran, dtype=torch.float32
    )
    verts_list = list(verts_packed.split(num_verts_per_mesh.tolist(), dim=0))
    return verts_list


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


def get_plane_params_in_local(planes, camera_info):
    """
    input:
    @planes: plane params
    @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
    output:
    plane parameters in global frame.
    """
    tran = camera_info["position"]
    rot = camera_info["rotation"]
    b = planes
    a = np.ones((len(planes), 3)) * tran
    planes_world = (
        a
        + b
        - ((a * b).sum(axis=1) / np.linalg.norm(b, axis=1) ** 2).reshape(-1, 1) * b
    )
    end = (
        quaternion.as_rotation_matrix(rot.inverse()) @ (planes_world - tran).T
    ).T  # world2cam
    planes_local = end * np.array([1, -1, -1])  # habitat2suncg
    return planes_local


def save_obj(
    folder,
    prefix,
    meshes,
    cam_meshes=None,
    decimal_places=None,
    blend_flag=False,
    map_files=None,
    uv_maps=None,
    save_mesh=True
):
    os.makedirs(folder, exist_ok=True)

    # pytorch3d does not support map_files
    # map_files = meshes.textures.map_files()
    # assert map_files is not None
    if map_files is None and uv_maps is None:
        raise RuntimeError("either map_files or uv_maps should be set!")

    # generate map_files from uv_map
    if uv_maps is not None and map_files is None:
        map_files = []
        uv_dir = os.path.join(folder, "uv_maps")
        if not os.path.exists(uv_dir):
            os.mkdir(uv_dir)
        for map_id, uv_map in enumerate(uv_maps):
            uv_path = os.path.join(uv_dir, "{}_uv_plane_{}.png".format(prefix, map_id))
            imageio.imwrite(uv_path, uv_map)
            map_files.append(uv_path)

    f_mtl = open(os.path.join(folder, prefix + ".mtl"), "w")
    f = open(os.path.join(folder, prefix + ".obj"), "w")
    try:
        seen = set()
        uniq_map_files = [
            m for m in list(map_files) if m not in seen and not seen.add(m)
        ]
        for map_id, map_file in enumerate(uniq_map_files):
            if uv_maps is not None:
                # we do not need to copy map_files,
                # they are already in uv_maps/...
                f_mtl.write(
                    _get_mtl_map(
                        os.path.basename(map_file).split(".")[0],
                        os.path.join("uv_maps", os.path.basename(map_file)),
                    )
                )
                continue

            if not blend_flag:
                shutil.copy(map_file, folder)
                os.chmod(os.path.join(folder, os.path.basename(map_file)), 0o755)
                f_mtl.write(
                    _get_mtl_map(
                        os.path.basename(map_file).split(".")[0],
                        os.path.basename(map_file),
                    )
                )
            else:
                rgb = cv2.imread(map_file, cv2.IMREAD_COLOR)
                if cam_meshes is not None:
                    blend_color = (
                        np.array(
                            cam_meshes.textures.verts_features_packed()
                            .numpy()
                            .tolist()[map_id]
                        )
                        * 255
                    )
                else:
                    blend_color = np.array(create_color_palette()[map_id + 10])
                alpha = 0.7
                blend = (rgb * alpha + blend_color[::-1] * (1 - alpha)).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        folder, os.path.basename(map_file).split(".")[0] + "_debug.png"
                    ),
                    blend,
                )
                f_mtl.write(
                    _get_mtl_map(
                        os.path.basename(map_file).split(".")[0],
                        os.path.basename(map_file).split(".")[0] + "_debug.png",
                    )
                )

        f.write(f"mtllib {prefix}.mtl\n\n")
        # we want [list]    verts, vert_uvs, map_files;
        #         [packed]  faces;
        #         face per mesh

        if meshes is not None:
            verts_list = meshes.verts_list()
            verts_uvs_list = meshes.textures.verts_uvs_list()
            faces_list = meshes.faces_packed().split(
                meshes.num_faces_per_mesh().tolist(), dim=0
            )
            if save_mesh:
                for idx, (verts, verts_uvs, faces, map_file) in enumerate(
                    zip(verts_list, verts_uvs_list, faces_list, map_files)
                ):
                    f.write(f"# mesh {idx}\n")
                    trunc_verts_uvs = verts_uvs[: verts.shape[0]]
                    _save(
                        f,
                        verts,
                        faces,
                        verts_uv=trunc_verts_uvs,
                        map_file=map_file,
                        idx=idx,
                        decimal_places=decimal_places,
                    )

        if cam_meshes:
            if save_mesh:
                face_offset = np.sum([len(v) for v in verts_list])
            else:
                face_offset = 0
            cam_verts_list = cam_meshes.verts_list() #[tensor(880,3), tensor(880,3)]
            cam_verts_rgbs_list = (
                cam_meshes.textures.verts_features_packed().numpy().tolist() # [[3]*1760]
            )
            cam_verts_rgbs_list = (cam_meshes.textures.verts_features_packed()).split(
                [cam_verts_list[i].shape[0] for i in range(len(cam_verts_list))], dim=0) #??????(tensor(880,3), tensor(880,3))

            cam_faces_list = (cam_meshes.faces_packed() + face_offset).split(
                cam_meshes.num_faces_per_mesh().tolist(), dim=0
            ) # (tensor(1600,3), tensor(1600,3))
            assert len(cam_verts_rgbs_list) == len(cam_verts_list), "len(cam_verts_rgbs_list) = %d, len(cam_verts_list) = %d"%(len(cam_verts_rgbs_list), len(cam_verts_list))
            for idx, (verts, faces, rgb) in enumerate(
                zip(cam_verts_list, cam_faces_list, cam_verts_rgbs_list)
            ):
                f.write(f"# camera {idx}\n")
                f_mtl.write(_get_mtl_rgb(idx, rgb))
                _save(f, verts, faces, rgb=rgb, idx=idx, decimal_places=decimal_places)
    finally:
        f.close()
        f_mtl.close()


def _get_mtl_map(material_name, map_Kd):
    return f"""newmtl {material_name}
map_Kd {map_Kd}
# Test colors
Ka 1.000 1.000 1.000  # white
Kd 1.000 1.000 1.000  # white
Ks 0.000 0.000 0.000  # black
Ns 10.0\n"""


def _get_mtl_rgb(material_idx, rgb):
    return f"""newmtl color_{material_idx}
Kd {rgb[0]} {rgb[1]} {rgb[2]}
Ka 0.000 0.000 0.000\n"""


def _save(
    f,
    verts,
    faces,
    verts_uv=None,
    map_file=None,
    rgb=None,
    idx=None,
    double_sided=True,
    decimal_places: Optional[int] = None,
):
    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places

    lines = ""

    V, D = verts.shape
    for i in range(V):
        vert = [float_str % verts[i, j] for j in range(D)]
        lines += "v %s\n" % " ".join(vert)

    if verts_uv is not None:
        V, D = verts_uv.shape
        for i in range(V):
            vert_uv = [float_str % verts_uv[i, j] for j in range(D)]
            lines += "vt %s\n" % " ".join(vert_uv)

    if map_file is not None:
        lines += f"usemtl {os.path.basename(map_file).split('.')[0]}\n"
    elif rgb is not None:
        lines += f"usemtl color_{idx}\n"

    if faces != []:
        F, P = faces.shape
        for i in range(F):
            if verts_uv is not None:
                face = ["%d/%d" % (faces[i, j] + 1, faces[i, j] + 1) for j in range(P)]
            else:
                face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            lines += "f %s\n" % " ".join(face)
            if double_sided:
                if verts_uv is not None:
                    face = [
                        "%d/%d" % (faces[i, j] + 1, faces[i, j] + 1)
                        for j in reversed(range(P))
                    ]
                else:
                    face = ["%d" % (faces[i, j] + 1) for j in reversed(range(P))]
                lines += "f %s\n" % " ".join(face)
    else:
        tqdm.write(f"face = []")
    f.write(lines)


def get_camera_meshes(camera_list, radius=0.01):
    # 0.0035 network fig
    # 0.0005 for pose compare
    # print("set came radius for pose compare")
    # radius = 0.005  # onePP
    # print("set came size for 3D plane..........")
    # radius = 0.01

    verts_list = []
    faces_list = []
    color_list = []
    rots = np.array(
        [
            quaternion.as_rotation_matrix(camera_info["rotation"])
            for camera_info in camera_list
        ]
    )

    # ai habitat frame
    lookat = np.array([0, 0, -1])
    vertical = np.array([0, 1, 0])

    positions = np.array([camera_info["position"] for camera_info in camera_list])
    lookats = rots @ lookat.T
    verticals = rots @ vertical.T
    # predetermined_color = [
    #     [0.10196, 0.32157, 1.0],
    #     [1.0, 0.0667, 0.1490],
    # ]  # old

    predetermined_color = [
        [0, 0, 1],
        [0, 0, 0],
    ]

    # predetermined_color = [
    #     [0.10196, 0.32157, 1.0],
    #     [1.0, 0.0667, 0.1490],
    #     [197/255, 181/255, 24/255],
    #     [73/255, 145/255, 115/255],
    #     [198/255, 120/255, 221/255],
    # ][:len(camera_list)]
    if len(camera_list) > 2:
        predetermined_color = [[0, 0, 0]] +[[0, 0, 1]]*(len(camera_list)-1)
        # predetermined_color = [[0, 0, 1]]*len(camera_list)
        # predetermined_color = [
        #     # [0.10196, 0.32157, 1.0],
        #     [0., 0., 1.],#blue
        #     # [1.0, 0.0667, 0.1490],
        #     [205/255, 79/255, 57/255], # Tomato3
        #     # [197/255, 181/255, 24/255],
        #     [255/255, 185/255, 15/255], #DarkGoldenrod1
        #     [255/255, 181/255, 197/255], #pink
        #     # [73/255, 145/255, 115/255],
        #     [131/255, 111/255, 255/255], # SlateBlue1 
        #     # [198/255, 120/255, 221/255],
        # ][:len(camera_list)]

    for idx, (position, lookat, vertical, color) in enumerate(
        zip(positions, lookats, verticals, predetermined_color)
    ):
        cur_num_verts = 0
        edges = get_cone_edges(position, lookat, vertical)
        cam_verts = []
        cam_inds = []
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cyl_verts = [x for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            cur_num_verts += len(cyl_verts)
            cam_verts.extend(cyl_verts)
            cam_inds.extend(cyl_ind)
        # Create a textures object
        verts_list.append(torch.tensor(cam_verts, dtype=torch.float32))
        faces_list.append(torch.tensor(cam_inds, dtype=torch.float32))
        color_list.append(color)

    color_tensor = torch.tensor(color_list, dtype=torch.float32).unsqueeze_(1) # [2, 1, 3]
    color_tensor_ = color_tensor.repeat(1, verts_list[0].shape[0], 1) # [2, 880, 3]
    tex = TexturesVertex(verts_features=color_tensor_)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    # meshes = Meshes(verts=verts_list, faces=faces_list)

    return meshes


#################### vis.py

def precompute_K_inv_dot_xy_1(h=480, w=640):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x], [0, focal_length, offset_y], [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640

            ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]

    # precompute to speed up processing
    return K_inv_dot_xy_1


def project2D(pcd, h=480, w=640, focal_length=517.97):
    assert h == 480 and w==640
    # pcd is Nx3
    offset_x = w / 2
    offset_y = h / 2
    K = [[focal_length, 0, offset_x], [0, focal_length, offset_y], [0, 0, 1]]

    # proj is Nx2
    proj = (np.array(K) @ (pcd.T)).T
    proj = proj[:, :2] / proj[:, 2][:, None]
    return proj


def get_pcd(verts, normal, offset, h=480, w=640, focal_length=517.97, K=-1):
    assert h == 480 and w==640
    """
    convert 2d verts to 3d point cloud based on plane normal and offset
    depth = offset / n \dot K^{-1}q
    """

    if isinstance(K, int):
        print("-----------------------------------------------------")
        import pdb; pdb.set_trace()
    if K is None:
        offset_x = w / 2
        offset_y = h / 2
        K = [[focal_length, 0, offset_x], [0, focal_length, offset_y], [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    homogeneous = np.hstack((verts, np.ones(len(verts)).reshape(-1, 1)))
    ray = K_inv @ homogeneous.T
    depth = offset / np.dot(normal, ray)
    pcd = depth.reshape(-1, 1) * ray.T

    return pcd


def get_pcd_depth(verts, depth, h=480, w=640, focal_length=517.97):
    assert h == 480 and w==640
    """
    convert 2d verts to 3d point cloud based on depth map
    depth = offset / n \dot K^{-1}q
    """
    offset_x = w / 2
    offset_y = h / 2
    K = [[focal_length, 0, offset_x], [0, focal_length, offset_y], [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    homogeneous = np.hstack((verts, np.ones(len(verts)).reshape(-1, 1)))
    ray = K_inv @ homogeneous.T
    pcd = depth[tuple(np.transpose(verts))].reshape(-1, 1) * ray.T
    return pcd


def rle2polygon(segmentations, tolerance=0):
    """
    convert rle format segmentation to polygon
    """
    assert isinstance(segmentations[0], dict)
    # decode
    binary_masks = mask.decode(segmentations).transpose(2, 0, 1)
    # binary masks 2 polygon masks
    poly_masks = [binary_mask_to_polygon(bm, tolerance) for bm in binary_masks]
    return poly_masks


def get_single_image_mesh_plane(
    plane_params,
    segmentations,
    # !img_file,
    image,   # ! RGB
    height=480,
    width=640,
    focal_length=517.97,
    webvis=False,
    tolerance=0,
    camera_K=-1
):
    plane_params = np.array(plane_params)
    offsets = np.linalg.norm(plane_params, ord=2, axis=1)
    norms = plane_params / offsets.reshape(-1, 1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations, tolerance)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    uv_maps = []
    imgs = []

    for segm, normal, offset in zip(poly_segmentations, norms, offsets):
        if len(segm) == 0:
            continue
        #! I = np.array(imageio.imread(img_file)) 
        I = image.copy()

        HUse = None

        # save uv_map
        tmp_verts = []
        for s in segm:
            tmp_verts.extend(s)
        tmp_verts = np.array(tmp_verts).reshape(-1, 2)
        # pick an arbitrary point
        # get 3d pointcloud
        #!  tmp_pcd = get_pcd(tmp_verts, normal, offset, K=camera_K)  # !
        tmp_pcd = get_pcd(tmp_verts, normal, offset, K=camera_K)  # !
        point0 = tmp_pcd[0, :]
        # pick the furthest point from here
        dPoint0 = np.sum((tmp_pcd - point0[np.newaxis, :]) ** 2, axis=1)
        point1 = tmp_pcd[np.argmax(dPoint0), :]

        # dir1 and dir2 are orthogonal to the normal
        dir1 = point1 - point0
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = np.cross(dir1, normal)

        # control points in 3D
        control3D = [point0, point0 + dir1, point0 + dir2, point0 + dir1 + dir2]
        control3D = np.vstack([p[None, :] for p in control3D])
        control3DProject = project2D(control3D)

        # pick an arbitrary square
        targetSize = 300
        fakePoints = np.array(
            [[0, 0], [0, targetSize], [targetSize, 0], [targetSize, targetSize]]
        ).astype(np.float32)

        # fit, then adjust
        H = cv2.getPerspectiveTransform(control3DProject.astype(np.float32), fakePoints)
        # this maps the control points to the square; now make sure the full mask warps in
        P = cv2.perspectiveTransform(tmp_verts.reshape(1, -1, 2), H)[0, :, :]
        xTrans, yTrans = P[:, 0].min(), P[:, 1].min()
        maxScale = max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min())
        HShuffle = np.array(
            [
                [targetSize / maxScale, 0, -xTrans * targetSize / maxScale],
                [0, targetSize / maxScale, -yTrans * targetSize / maxScale],
                [0, 0, 1],
            ]
        )
        HUse = HShuffle @ H

        # warped_image is now the rectified image; warped_image2 has it with a 100px fudge factor
        warped_image = cv2.warpPerspective(I, HUse, (targetSize, targetSize))

        uv_maps.append(warped_image)

        verts_3d = []
        faces = []
        uvs = []

        for ring in segm:
            verts = np.array(ring).reshape(-1, 2)
            # get 3d pointcloud
            pcd = get_pcd(verts, normal, offset, K=camera_K)

            if webvis:
                # Rotate by 11 degree around x axis to push things on the ground.
                pcd = (
                    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    @ np.array(
                        [
                            [1, 0, 0],
                            [0, 0.9816272, -0.1908090],
                            [0, 0.1908090, 0.9816272],
                        ]
                    )
                    @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    @ pcd.T
                ).T

            uvsRectified = cv2.perspectiveTransform(
                verts.astype(np.float32).reshape(1, -1, 2), HUse
            )[0, :, :]
            uvsRectified = np.array([0, 1]) + np.array(
                [1, -1]
            ) * uvsRectified / np.array([targetSize, targetSize])
            uvs.extend(uvsRectified)

            # triangulate polygon using earcut algorithm
            triangles = earcut.triangulate_float32(verts, [len(verts)])
            # add base index of vertice
            triangles += len(verts_3d)
            triangles = triangles.reshape(-1, 3)
            # convert to counter-clockwise
            triangles[:, [0, 2]] = triangles[:, [2, 0]]

            if triangles.shape[0] == 0:
                continue

            verts_3d.extend(pcd)
            faces.extend(triangles)

        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        # imgs.append(torch.FloatTensor(imageio.imread(img_file)))
        imgs.append(torch.FloatTensor(I)) #?

    # pytorch3d mesh
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)

    return meshes, uv_maps


def get_single_image_mesh(
    plane_params,
    segmentations,
    img_file,
    height=480,
    width=640,
    focal_length=517.97,
    webvis=False,
    reduce_size=True,
):
    plane_params = np.array(plane_params)
    offsets = np.linalg.norm(plane_params, ord=2, axis=1)
    norms = plane_params / offsets.reshape(-1, 1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    img_files = []
    imgs = []

    for segm, normal, offset in zip(poly_segmentations, norms, offsets):
        if len(segm) == 0:
            continue
        verts_3d = []
        faces = []
        uvs = []
        if reduce_size:
            for ring in segm:
                verts = np.array(ring).reshape(-1, 2)
                # get 3d pointcloud
                pcd = get_pcd(verts, normal, offset)
                if webvis:
                    # Rotate by 11 degree around x axis to push things on the ground.
                    pcd = (
                        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                        @ np.array(
                            [
                                [1, 0, 0],
                                [0, 0.9816272, -0.1908090],
                                [0, 0.1908090, 0.9816272],
                            ]
                        )
                        @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                        @ pcd.T
                    ).T
                # triangulate polygon using earcut algorithm
                triangles = earcut.triangulate_float32(verts, [len(verts)])
                # add base index of vertice
                triangles += len(verts_3d)
                triangles = triangles.reshape(-1, 3)
                # convert to counter-clockwise
                triangles[:, [0, 2]] = triangles[:, [2, 0]]
                verts_3d.extend(pcd)
                faces.extend(triangles)
                uvs.extend(
                    np.array([0, 1])
                    + np.array([1, -1]) * verts / np.array([width, height])
                )

        else:
            bitmask = polygons_to_bitmask(segm, height=height, width=width)
            verts = np.transpose(bitmask.nonzero())
            vert_id_map = defaultdict(dict)
            for idx, vert in enumerate(verts):
                vert_id_map[vert[0]][vert[1]] = idx + len(verts_3d)

            verts_3d = get_pcd(verts[:, ::-1], normal, offset)
            if webvis:
                # Rotate by 11 degree around x axis to push things on the ground.
                verts_3d = (
                    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    @ np.array(
                        [
                            [1, 0, 0],
                            [0, 0.9816272, -0.1908090],
                            [0, 0.1908090, 0.9816272],
                        ]
                    )
                    @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    @ pcd.T
                ).T
            triangles = []
            for vert in verts:
                # upper right triangle
                if (
                    vert[0] < height - 1
                    and vert[1] < width - 1
                    and bitmask[vert[0]][vert[1] + 1]
                    and bitmask[vert[0] + 1][vert[1] + 1]
                ):
                    triangles.append(
                        [
                            vert_id_map[vert[0]][vert[1]],
                            vert_id_map[vert[0] + 1][vert[1] + 1],
                            vert_id_map[vert[0]][vert[1] + 1],
                        ]
                    )
                # bottom left triangle
                if (
                    vert[0] < height - 1
                    and vert[1] < width - 1
                    and bitmask[vert[0] + 1][vert[1]]
                    and bitmask[vert[0] + 1][vert[1] + 1]
                ):
                    triangles.append(
                        [
                            vert_id_map[vert[0]][vert[1]],
                            vert_id_map[vert[0] + 1][vert[1]],
                            vert_id_map[vert[0] + 1][vert[1] + 1],
                        ]
                    )
            triangles = np.array(triangles)
            faces.extend(triangles)
            uvs.extend(
                np.array([0, 1])
                + np.array([1, -1]) * verts[:, ::-1] / np.array([width, height])
            )
        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        img_files.append(img_file)
        imgs.append(torch.FloatTensor(imageio.imread(img_file)))
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)

    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    return meshes, img_files


def get_single_image_pcd(plane_params, segmentations, height=480, width=640):
    plane_params = np.array(plane_params)
    offsets = np.maximum(np.linalg.norm(plane_params, ord=2, axis=1), 1e-5)
    norms = plane_params / offsets.reshape(-1, 1)

    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []

    for segm, normal, offset in zip(poly_segmentations, norms, offsets):
        if len(segm) == 0:
            verts_list.append(torch.tensor([[0, 0, 0]], dtype=torch.float32))
            continue
        verts_3d = []
        bitmask = polygons_to_bitmask(segm, height=height, width=width)
        verts = np.transpose(bitmask.nonzero())
        verts_3d = get_pcd(verts[:, ::-1], normal, offset)
        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
    return verts_list


def get_single_image_mesh_depth(
    depth, segmentations, img_file, height=480, width=640, webvis=True
):
    if type(segmentations[0]) == dict:
        poly_segmentations = rle2polygon(segmentations)
    else:
        poly_segmentations = segmentations
    verts_list = []
    faces_list = []
    verts_uvs = []
    img_files = []
    imgs = []

    for segm in poly_segmentations:
        if len(segm) == 0:
            continue
        verts_3d = []
        faces = []
        uvs = []
        bitmask = polygons_to_bitmask(segm, height=height, width=width)
        verts = np.transpose(bitmask.nonzero())
        vert_id_map = defaultdict(dict)
        for idx, vert in enumerate(verts):
            vert_id_map[vert[0]][vert[1]] = idx + len(verts_3d)
        pcd = get_pcd_depth(verts[:, ::-1], depth.T)
        if webvis:
            # Rotate by 11 degree around x axis to push things on the ground.
            pcd = (
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                @ np.array(
                    [[1, 0, 0], [0, 0.9816272, -0.1908090], [0, 0.1908090, 0.9816272]]
                )
                @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                @ pcd.T
            ).T
        triangles = []
        for vert in verts:
            # upper right triangle
            if (
                vert[0] < height - 1
                and vert[1] < width - 1
                and bitmask[vert[0]][vert[1] + 1]
                and bitmask[vert[0] + 1][vert[1] + 1]
            ):
                triangles.append(
                    [
                        vert_id_map[vert[0]][vert[1]],
                        vert_id_map[vert[0] + 1][vert[1] + 1],
                        vert_id_map[vert[0]][vert[1] + 1],
                    ]
                )
            # bottom left triangle
            if (
                vert[0] < height - 1
                and vert[1] < width - 1
                and bitmask[vert[0] + 1][vert[1]]
                and bitmask[vert[0] + 1][vert[1] + 1]
            ):
                triangles.append(
                    [
                        vert_id_map[vert[0]][vert[1]],
                        vert_id_map[vert[0] + 1][vert[1]],
                        vert_id_map[vert[0] + 1][vert[1] + 1],
                    ]
                )
        triangles = np.array(triangles)
        verts_3d.extend(pcd)
        faces.extend(triangles)
        uvs.extend(
            np.array([0, 1])
            + np.array([1, -1]) * verts[:, ::-1] / np.array([width, height])
        )
        verts_list.append(torch.tensor(verts_3d, dtype=torch.float32))
        faces_list.append(torch.tensor(faces, dtype=torch.int32))
        verts_uvs.append(torch.tensor(uvs, dtype=torch.float32))
        img_files.append(img_file)
        imgs.append(torch.FloatTensor(imageio.imread(img_file)))
    verts_uvs = pad_sequence(verts_uvs, batch_first=True)
    faces_uvs = pad_sequence(faces_list, batch_first=True, padding_value=-1)
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=imgs)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    return meshes, img_files

# # for 7scenes
# def get_single_image_mesh_with_calibration(
#     depth, rgb, height=480, width=640, 
#     optimized_params=[5.98836000e+02,  5.87618669e+02, 2.34490289e-02, 6.17665950e-03, 1.05253125e-02], # [fx_depth, fy_depth, tx, ty, tz]
#     rgb_focals = [532.57, 531.54],
# ):
#     import open3d as o3d
#     K = np.array([[rgb_focals[0], 0.0, width/2],
#               [0.0, rgb_focals[1], height/2],
#               [0.0, 0.0, 1.0]])

#     # # [fx, fy, tx, ty, tz]
#     # optimized_params = [ 5.98836000e+02,  5.87618669e+02, 2.34490289e-02, 6.17665950e-03, 1.05253125e-02]

#     # depth intrinsics
#     K_depth = np.array([[optimized_params[0], 0.0, width/2],
#                         [0.0, optimized_params[1], height/2],
#                         [0.0, 0.0, 1.0]])
#     K_depth_inv = np.linalg.inv(K_depth)



#     # idx = 0
#     # rgb = cv2.imread(os.path.join(folder, "frame-%06d.color.png" % idx))
#     # depth = cv2.imread(os.path.join(folder, "frame-%06d.depth.png" % idx), cv2.IMREAD_UNCHANGED)

#     x, y = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
#     pts = np.vstack((x.flatten(), y.flatten(), np.ones(depth.shape[0] * depth.shape[1]))) # [u_depth, v_depth, 1]

#     depth = depth.flatten()
#     # valid_depth_indices = np.where(depth < 65535)[0]
#     valid_depth_indices = np.where(depth < 65535/1000)[0]
#     # depth = depth.astype(np.float64) / 1000
#     pts = pts[:, valid_depth_indices]
#     depth = depth[valid_depth_indices]


#     X = np.multiply(depth.flatten(),  K_depth_inv @ pts) #[Xc, Yc, Zc] 3xn

#     # Get color by projecting in rgb image
#     Trgb_depth = np.array([[1.0, 0.0, 0.0, -optimized_params[2]],
#                         [0.0, 1.0, 0.0, -optimized_params[3]],
#                         [0.0, 0.0, 1.0, -optimized_params[4]]])

#     uvs = K @ Trgb_depth @ np.vstack((X, np.ones(pts.shape[1]))) # Zc*[u_rgb, v_rgb, 1]
#     uvs /= uvs[2, :]
#     uvs = np.round(uvs).astype(int)[:2, :].T  #[u_rgb, v_rgb]

#     colors = []
#     for uv in uvs:
#         if uv[0] >= 0 and uv[0] < 640 and uv[1] >= 0 and uv[1] < 480:
#             colors.append(rgb[uv[1], uv[0], :])
#         else:
#             colors.append((0, 0, 0))
#     colors = np.vstack(colors)/255



#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(X.T)
#     # pcd.colors = o3d.utility.Vector3dVector(colors/255)
#     # # if not pcd.has_normals():
#     # pcd.estimate_normals(
#     #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
#     # ball pivoting
#     # distances = pcd.compute_nearest_neighbor_distance()
#     # avg_dist = np.mean(distances)
#     # radius = 1.5 * avg_dist   

#     # oed_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     #         pcd,
#     #         o3d.utility.DoubleVector([radius, radius * 2]))
    
#     # poisson
#     # o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
#     # o3d.io.write_triangle_mesh("tmp.obj", o3d_mesh)

#     # meshes = pytorch3d.io.load_objs_as_meshes(["tmp.obj"])



#     # meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
#     return X.T, colors



        
    


