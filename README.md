# PlaneRecTR++

**PlaneRecTR++: Unified Query Learning for Joint 3D Planar Reconstruction and Pose Estimation**

[[paper](https://arxiv.org/abs/2307.13756)]

###  Installation

Please refer to [PlaneRecTR](https://github.com/SJingjia/PlaneRecTR).

### Data Preparation

Please download the required sparse-view datasets from [NOPE-SAC](https://github.com/IceTTTb/NopeSAC).  
Then set the location for builtin datasets by `export DETECTRON2_DATASETS=YOUR_DATASETS_FOLDER/`, and detectron2 will look for datasets in the following directory structure:

```bash
YOUR_DATASETS_FOLDER/
├── scannetv2_multiview/
└── mp3d/
```

For MP3D single-view evaluation, please additionally prepare [mp3d_single_json](https://drive.google.com/drive/folders/1bzIZNEsHPnpC030cQkFyQBKdwchnweuP), which removes duplicated images appearing in image pairs.

### Training and Evaluation

#### 1.1 Pre-training of intra-frame query learning

**ScanNetv2**

```bash
python sparseviews_train_net.py --num-gpus 1 --config-file configs/SparseViews/scannetv2_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/scannetv2_multiview/ OUTPUT_DIR ./output_train_singleview_scannetv2
```

**MP3D**

```bash
python sparseviews_train_net.py --num-gpus 1 --config-file configs/SparseViews_mp3d/mp3d_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/mp3d/ OUTPUT_DIR ./output_train_singleview_mp3d
```

#### 1.2  Evaluation for single-view metrics

You can downlaod our trained models from [here](https://drive.google.com/drive/folders/1bzIZNEsHPnpC030cQkFyQBKdwchnweuP).

**ScanNetv2**

```bash
python sparseviews_train_net.py --eval-only --num-gpus 1 --config-file configs/SparseViews/scannetv2_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/scannetv2_multiview/ OUTPUT_DIR ./output_eval_singleview_scannetv2 MODEL.WEIGHTS output_train_singleview_scannetv2/model_final.pth
```

**MP3D**

> For MP3D single-view evaluation, use `mp3d_single_size480.yaml`.

```bash
python sparseviews_train_net.py --eval-only --num-gpus 1 --config-file configs/SparseViews_mp3d/mp3d_single_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/mp3d/ OUTPUT_DIR ./output_eval_singleview_mp3d MODEL.WEIGHTS output_train_singleview_mp3d/model_final.pth
```

#### 2.1 Training the whole PlaneRecTR++

**ScanNetv2**

```bash
python sparseviews_train_net.py --num-gpus 1 --config-file configs/SparseViews/scannetv2_pose_phase_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/scannetv2_multiview/ OUTPUT_DIR ./output_train_sparseview_scannetv2 MODEL.WEIGHTS output_train_singleview_scannetv2/model_final.pth
```

**MP3D**

```bash
python sparseviews_train_net.py --num-gpus 1 --config-file configs/SparseViews_mp3d/mp3d_pose_phase_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/mp3d/ OUTPUT_DIR ./output_train_sparseview_mp3d MODEL.WEIGHTS output_train_singleview_mp3d/model_final.pth
```

#### 2.2 Evaluation

You can downlaod our trained models from [here](https://drive.google.com/drive/folders/1bzIZNEsHPnpC030cQkFyQBKdwchnweuP).

**ScanNetv2**

```bash
python sparseviews_train_net.py --eval-only --num-gpus 1 --config-file configs/SparseViews/scannetv2_pose_phase_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/scannetv2_multiview/  MODEL.MASK_FORMER.TEST.SAVE_OUTPUT True OUTPUT_DIR ./output_eval_sparseview_scannetv2 TEST.VIS_PERIOD 500 MODEL.WEIGHTS output_train_sparseview_scannetv2/model_final.pth
```
set MODEL.MASK_FORMER.TEST.SAVE_OUTPUT=True to save PlaneRecTRpp_outputs.pkl for evaluations.

```bash
python eval.py --config-file configs/SparseViews/scannetv2_pose_phase_size480.yaml --rcnn-cached-file output_eval_sparseview_scannetv2/inference/PlaneRecTRpp_outputs.pkl --dataset-phase sparseviews_scannetv2_plane_test --match-threshold 0.17 --corr-idx 0 --th-method 0
```


**MP3D**

```bash
python sparseviews_train_net.py --eval-only --num-gpus 1 --config-file configs/SparseViews_mp3d/mp3d_pose_phase_size480.yaml DATASETS.ROOT_DIR YOUR_DATASETS_FOLDER/mp3d/ MODEL.MASK_FORMER.TEST.SAVE_OUTPUT True OUTPUT_DIR ./output_eval_sparseview_mp3d TEST.VIS_PERIOD 500 MODEL.WEIGHTS output_train_sparseview_mp3d/model_final.pth
python eval.py --config-file configs/SparseViews_mp3d/mp3d_pose_phase_size480.yaml --rcnn-cached-file output_eval_sparseview_mp3d/inference/PlaneRecTRpp_outputs.pkl --dataset-phase sparseviews_mp3d_plane_test --match-threshold 0.08 --corr-idx 0 --th-method 0
```
In addition, vis_planerec.py can be used to provide further visualizations presented in the paper.

### Acknowledgements

We would like to thank [NOPE-SAC](https://github.com/IceTTTb/NopeSAC), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [rel-pose](https://github.com/crockwell/rel_pose) and other related open-source projects.





