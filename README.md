# ThermalNeRF: Thermal Radiance Fields

[Webpage](https://yvette256.github.io/thermalnerf/) | [Paper](TODO)

[//]: # (<img src="https://user-images.githubusercontent.com/3310961/194017985-ade69503-9d68-46a2-b518-2db1a012f090.gif" width="52%"/> <img src="https://user-images.githubusercontent.com/3310961/194020648-7e5f380c-15ca-461d-8c1c-20beb586defe.gif" width="46%"/>)

This repository contains the official authors' implementation associated with the paper _ThermalNeRF: Thermal Radiance Fields_.

Abstract:
_Thermal imaging has a variety of applications, from agricultural monitoring to building inspection to imaging under
poor visibility, such as in low light, fog, and rain. However, reconstructing thermal scenes in 3D presents several
challenges due to the comparatively lower resolution and limited features present in long-wave infrared (LWIR) images.
To overcome these challenges, we propose a unified framework for scene reconstruction from a set of LWIR and RGB images,
using a multispectral radiance field to represent a scene viewed by both visible and infrared cameras, thus leveraging
information across both spectra. We calibrate the RGB and infrared cameras with respect to each other, as a
preprocessing step using a simple calibration target. We demonstrate our method on real-world sets of RGB and LWIR
photographs captured from a handheld thermal camera, showing the effectiveness of our method at scene representation
across the visible and infrared spectra. We show that our method is capable of thermal super-resolution, as well as
visually removing obstacles to reveal objects that are occluded in either the RGB or thermal channels._

This work's codebase is built on top of the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) project.
As a result, the official [Nerfstudio documentation](https://docs.nerf.studio/) might also be a helpful resource for any
additional questions unanswered by this document, which also adapts parts of the Nerfstudio README.

## Setup

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This project has been tested with version 11.8 of CUDA.

### Create environment

This project requires `python >= 3.8`. We recommend using conda to manage dependencies.

```bash
conda create --name thermalnerf -y python=3.8
conda activate thermalnerf 
python -m pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Additionally, install [COLMAP](https://github.com/colmap/colmap):

```bash
conda install -c conda-forge colmap
```


### Installing

```bash
git clone git@github.com:yvette256/nerfstudio-thermal.git
cd nerfstudio-thermal
pip install --upgrade pip setuptools
pip install -e .
```

## Running

### Processing data

To process FLIR images (in a directory `FLIR_DATA_PATH`) for training, run the following command.
This command computes the RGB and thermal camera intrinsics/extrinsics using the calibration
images in `CALIBRATION_DATA_PATH` and populates the directory `DATA_PATH`.

```bash
python nerfstudio/scripts/process_data.py rgbt --data FLIR_DATA_PATH --output-dir DATA_PATH --calibration-data CALIBRATION_DATA_PATH
```

**IMPORTANT:** This script does currently hard-code some assumptions.
Changing the hard-coded nature of these quirks is on our to-do list, but in the meantime, users should know about the following:

- The calibration pattern of the target we used (4 x 11 asymmetric grid of circular cutouts, each with a diameter
  of 15mm and a center-center distance of 38mm) is hard-coded in this [function](https://github.com/yvette256/nerfstudio-thermal/blob/9d347c21ffbc7293e8dd4109483800b8021784bc/nerfstudio/process_data/calibration_utils.py#L11).
  If you wish to use your own custom calibration pattern, you will need to edit the code in the function.
- We hard-code the assumption that the 3rd and 4th (in lexicographic order) images in `FLIR_DATA_PATH` are taken from
  camera positions 1m apart [here](https://github.com/yvette256/nerfstudio-thermal/blob/9d347c21ffbc7293e8dd4109483800b8021784bc/nerfstudio/process_data/rgbt_to_nerfstudio_dataset.py#L222).
  This is to resolve the global scale ambiguity in the output of COLMAP, which is used to estimate RGB pose.
  Precisely, to resolve this ambiguity, it is sufficient to know the distance between camera positions of any two images,
  so this can be edited to reflect any two images and any distance when using custom data.

#### Additional configuration options

Use the `--help` command to see the full list of configuration options.
This includes additional configuration options stemming from the original Nerfstudio image processing scripts on which
our script is built.
Most of these should still work, but have not been extensively tested.

```bash
python nerfstudio/scripts/process_data.py rgbt --help
```

### Training

To train the _thermal-nerfacto_ model on the (processed) thermal data, run

```bash
python nerfstudio/scripts/train.py thermal-nerfacto --data DATA_PATH
```
<details>
<summary><span style="font-weight: bold;">Configuration options for thermal-nerfacto</span></summary>

##### --pipeline.model.density_mode {rgb_only,shared,separate}
How to treat density between RGB/T (`rgb_only` only reconstructs RGB field).
##### --pipeline.model.density_loss_mult FLOAT
Density loss (L1 norm of `<rgb density> - <thermal density>`) multiplier.
##### --pipeline.model.rgb_density_loss_mult FLOAT
Relative influence on RGB density in the L1 density loss (applied on top of `density_loss_mult`).
##### --pipeline.model.cross_channel_loss_mult FLOAT
Cross-channel gradient loss multiplier.
##### --pipeline.model.thermal_loss_mult FLOAT
Thermal pixel-wise reconstruction loss multiplier.
##### --pipeline.model.tv_pixel_loss_mult FLOAT
Pixelwise thermal TV loss multiplier.

</details>

#### Tensorboard / WandB / Viewer

We support four different methods to track training progress, using the viewer: [tensorboard](https://www.tensorflow.org/tensorboard), [Weights and Biases](https://wandb.ai/site), and [Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=github). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command.

#### Resume from checkpoint

It is possible to load a pretrained model by running

```bash
python nerfstudio/scripts/train.py nerfacto --data DATA_PATH --load-dir MODEL_PATH
```

#### Additional configuration options

Use the `--help` command to see the full list of configuration options.
This includes additional configuration options stemming from the original Nerfstudio _nerfacto_ model on which
_thermal-nerfacto_ is built.
Most of these should still work, but have not been extensively tested with _thermal-nerfacto_.

```bash
python nerfstudio/scripts/train.py thermal-nerfacto --help
```

## Visualization

### Visualize existing run

When training, navigating to the link at the end of the terminal will load the webviewer.
Otherwise, given a pretrained model checkpoint, you can start the viewer by running

```bash
python nerfstudio/scripts/viewer/run_viewer.py --load-config {outputs/.../config.yml}
```

In the viewer, the thermal outputs are named similarly to the RGB outputs, but with `_thermal` appended.
So `rgb` is the RGB view while `rgb_thermal` is the thermal view, `depth` is the RGB depth while `depth_thermal` is the
thermal depth, etc.
(We know this results in some unintuitive names, sorry about that, it may be changed in the future.)

### Render results

To render outputs from train/test views, run

```bash
python nerfstudio/scripts/render.py dataset --load-config {outputs/.../config.yml}
```

<details>
<summary><span style="font-weight: bold;">Command-line arguments</span></summary>

##### --split {train,val,test,train+test}
Split to render. (default: test)
##### --rendered_output_names \<list of output names\>
Name of the renderer outputs to use. As described previously, the thermal outputs are named similarly to the RGB outputs, but with `_thermal` appended.
So `rgb` is the RGB view while `rgb_thermal` is the thermal view, `depth` is the RGB depth while `depth_thermal` is the
thermal depth, etc.

</details>

#### Removing hidden objects

We can remove occluding objects from RGB or thermal views, thus revealing objects hidden behind other objects, by
rendering only the parts of the scene with RGB and thermal densities sufficiently similar to each other.
To demonstrate this, use `--rendered-output-names removal` to render hidden RGB objects and/or
`--rendered-output-names removal_thermal` to render hidden thermal objects.
If desired, use `--removal_min_density_diff FLOAT` to specify the minimum difference between rgb and thermal densities
allowed for removal rendering.

## Acknowledgements

This material is based upon work supported by the National Science Foundation under award number 2303178 to SFK. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## Citation

If you find this repo useful for your projects, please consider citing:

```
@inproceedings{lin2024thermalnerf,
	title        = {{ThermalNeRF}: Thermal Radiance Fields},
	author       = {Lin, Yvette Y and Pan, Xin-Yi and Fridovich-Keil, Sara and Wetzstein, Gordon},
	year         = {2024},
	booktitle    = {IEEE International Conference on Computational Photography (ICCP)},
	organization = {IEEE}
}
```
