# Cross-Descriptor Visual Localization and Mapping

This repository contains the implementation of the following paper:

```text
"Cross-Descriptor Visual Localization and Mapping".
M. Dusmanu, O. Miksik, J.L. Schönberger, and M. Pollefeys. ICCV 2021.
```

[[Paper on arXiv]](https://arxiv.org/abs/2012.01377)


## Requirements

### COLMAP

We use COLMAP for DoG keypoint extraction as well as localization and mapping.
Please follow the installation instructions available on the [official webpage](https://colmap.github.io).
Before proceeding, we recommend setting an environmental variable to the COLMAP executable folder by running `export COLMAP_PATH=path_to_colmap_executable_folder`.

### Python

The environment can be set up directly using conda:
```
conda env create -f env.yml
conda activate cross-descriptor-vis-loc-map
```

### Training data

We provide a script for downloading the raw training data:
```
bash scripts/download_training_data.sh
```

### Evaluation data

We provide a script for downloading the LFE dataset along with the GT used for evaluation as well as the Aachen Day-Night dataset:
```
bash scripts/download_evaluation_data.sh
```


## Training

### Data preprocessing

First step is extracting keypoints and descriptors on the training data downloaded above.
```
bash scripts/preprocess_training_data.sh
```
Alternatively, you can directly download the processed training data by running:
```
bash scripts/download_processed_training_data.sh
```

### Training

To run training with the default architecture and hyper-parameters, execute the following:
```
python train.py --dataset_path data/train/colmap --features brief sift-kornia hardnet sosnet
```

### Pretrained models

We provide two pretrained models trained on descriptors extracted from COLMAP SIFT and OpenCV SIFT keypoints, respectively.
These models can be downloaded by running:
```
bash scripts/download_checkpoints.sh
```


## Evaluation

### Demo Notebook

### Local Feature Evaluation Benchmark

### Aachen Day-Night


## BibTeX

If you use this code in your project, please cite the following paper:
```
@InProceedings{Dusmanu2021Cross,
    author = {Dusmanu, Mihai and Miksik, Ondrej and Sch\"onberger, Johannes L. and Pollefeys, Marc},
    title = {{Cross Descriptor Visual Localization and Mapping}},
    booktitle = {Proceedings of the International Conference on Computer Vision},
    year = {2021}
}
```
