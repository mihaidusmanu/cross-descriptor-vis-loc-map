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
python train.py \
    --dataset_path data/train/colmap \
    --features brief sift-kornia hardnet sosnet
```

### Pretrained models

We provide two pretrained models trained on descriptors extracted from COLMAP SIFT and OpenCV SIFT keypoints, respectively.
These models can be downloaded by running:
```
bash scripts/download_checkpoints.sh
```


## Evaluation

### Demo Notebook

<details>
<summary>Click for details...</summary>

</details>

### Local Feature Evaluation Benchmark

<details>
<summary>Click for details...</summary>

First step is extracting descriptors on all datasets:
```
bash scripts/process_LFE_data.sh
```

We provide examples below for running reconstruction on Madrid Metrpolis in each different evaluation scenario.

#### Reconstruction using a single descriptor (standard)

```
python local-feature-evaluation/reconstruction_pipeline_progressive.py \
    --dataset_path data/eval/LFE-release/Madrid_Metropolis \
    --colmap_path $COLMAP_PATH \
    --features sift-kornia \
    --exp_name sift-kornia-single
```

#### Reconstruction using the progressive approach (ours)

```
python local-feature-evaluation/reconstruction_pipeline_progressive.py \
    --dataset_path data/eval/LFE-release/Madrid_Metropolis \
    --colmap_path $COLMAP_PATH \
    --features brief sift-kornia hardnet sosnet \
    --exp_name progressive
```

#### Reconstruction using the joint embedding approach (ours)

```
python local-feature-evaluation/reconstruction_pipeline_embed.py \
    --dataset_path data/eval/LFE-release/Madrid_Metropolis \
    --colmap_path $COLMAP_PATH \
    --features brief sift-kornia hardnet sosnet \
    --exp_name embed
```

#### Reconstruction using a single descriptor on the associated split (real-world)

```
python local-feature-evaluation/reconstruction_pipeline_subset.py \
    --dataset_path data/eval/LFE-release/Madrid_Metropolis/ \
    --colmap_path $COLMAP_PATH \
    --features brief sift-kornia hardnet sosnet \
    --feature sift-kornia \
    --exp_name sift-kornia-subset
```

#### Evaluation of a reconstruction w.r.t. metric pseudo-ground-truth

```
python local-feature-evaluation/align_and_compare.py \
    --colmap_path $COLMAP_PATH \
    --reference_model_path data/eval/LFE-release/Madrid_Metropolis/sparse-reference/filtered-metric/ \
    --model_path data/eval/LFE-release/Madrid_Metropolis/sparse-sift-kornia-single/0/
```

</details>

### Aachen Day-Night

<details>
<summary>Click for details...</summary>

</details>

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
