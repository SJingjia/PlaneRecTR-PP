import os
import logging
import json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
import random 


# logger = logging.getLogger(__name__)
# if not logger.isEnabledFor(logging.INFO):
#     setup_logger(name=__name__)


def load_mp3d_json(json_file, dataset_name=None, vis_on=False, mini_on=False):
    """
    Load a json file with mp3d's instances annotation format.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See .md)

    Notes:
        1. This function does not read the image files.c
           The results do not have the "image" field.
    """
    with open(json_file, "r") as f:
        summary = json.load(f)
    meta = MetadataCatalog.get(dataset_name)
    cats = summary["categories"]
    try:
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    except:
        import pdb; pdb.set_trace()
    meta.thing_classes = thing_classes
    return summary["data"]

def get_mp3d_metadata():
    meta = [
        {"name": "plane", "color": [230, 25, 75], "id": 1},  # noqa
    ]
    return meta


SPLITS = {
    "sparseviews_scannetv2_plane_train": ("scannet", "scannet_json/cached_set_trainV2.json"),
    "sparseviews_scannetv2_plane_test": ("scannet", "scannet_json/cached_set_testV2.json"),
    
    "sparseviews_mp3d_plane_train": ("mp3d", "mp3d_json/cached_set_train.json"),
    "sparseviews_mp3d_plane_test": ("mp3d", "mp3d_json/cached_set_test.json"), # 7996

    # ! pair to single
    "sparseviews_scannetv2_single_plane_train": ("scannet_single", "scannet_single_json/cached_set_trainV2.json"), # 34474
    "sparseviews_scannetv2_single_plane_test": ("scannet_single", "scannet_single_json/cached_set_testV2.json"),  # 4391

    "sparseviews_mp3d_single_plane_train": ("mp3d_single", "mp3d_single_json/cached_set_train.json"),
    "sparseviews_mp3d_single_plane_test": ("mp3d_single", "mp3d_single_json/cached_set_test.json"), 


}

def register_mp3d(dataset_name, json_file, image_root):
    if 'mp3d' in dataset_name:
        # root = "/home/jingjia/data/mp3d"
        root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), 'mp3d')
    elif 'scannet' in dataset_name:
        # root = "/home/jingjia/data/scannetv2_multiview/"
        root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), 'scannetv2_multiview') # 26 
    elif '7scenes' in dataset_name:
        # root = "/home/jingjia/data/7scenes/"
        root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), '7scenes')
    else:
        raise NotImplementedError

    json_file = os.path.join(root, json_file)

    DatasetCatalog.register(
        dataset_name, lambda: load_mp3d_json(json_file, dataset_name)
    )
    things_ids = [k["id"] for k in get_mp3d_metadata()]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in get_mp3d_metadata()]
    thing_colors = [k["color"] for k in get_mp3d_metadata()]
    
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }
    # evaluator_type = "sparseviews2single" if "single" in dataset_name else "sparseviews"
    if "single" in dataset_name:
        evaluator_type = "sparseviews2single"
    elif "sparseviews" in dataset_name:
        evaluator_type = "sparseviews"
    elif "multiviews" in dataset_name:
        evaluator_type = "multiviews"

    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type=evaluator_type, **metadata
    )


for key, (data_root, anno_file) in SPLITS.items():
    register_mp3d(key, anno_file, data_root)

if __name__ == "__main__":
    """
    Test the mp3d json dataset loader.

    Usage:
        python -m planercnn.data.datasets.mp3d \
            datasets/mp3d/cached_set_val.json mp3d_val

        "dataset_name" can be "coco", "coco_person", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    import cv2
    import sys

    logger = setup_logger(name=__name__)
    dataset_name = "scannet_test"
    meta = MetadataCatalog.get(dataset_name)
    dataset_custom = DatasetCatalog.get(dataset_name)

    # dicts = load_mp3d_json(sys.argv[1], dataset_name)
    # logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "mp3d-data-vis"
    os.makedirs(dirname, exist_ok=True)

    root = "/home/jingjia/data/scannetv2_multiview/"
    for s in random.sample(dataset_custom, 3):
        ps = s["0"]["file_name"].split("/")
        ps.insert(4, "frames")
        recent_file_name = os.path.join(root, "/".join(ps[2:]))
        img = cv2.imread(recent_file_name)
        img = cv2.resize(img, (640, 480))
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.5)
        vis = visualizer.draw_dataset_dict(s["0"])
        fpath = os.path.join(dirname, os.path.basename(s["0"]["file_name"]))
        vis.save(fpath)
