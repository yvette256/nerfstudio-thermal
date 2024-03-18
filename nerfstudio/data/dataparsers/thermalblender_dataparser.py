from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.blender_dataparser import Blender, BlenderDataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.utils.io import load_from_json


@dataclass
class ThermalBlenderDataParserConfig(BlenderDataParserConfig):
    """Thermal Blender dataset config"""

    _target: Type = field(default_factory=lambda: ThermalBlender)
    """target class to instantiate"""
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training.."""


@dataclass
class ThermalBlender(Blender):
    """Thermal Blender DatasetParser"""

    config = ThermalBlenderDataParserConfig

    def __init__(self, config: ThermalBlenderDataParserConfig):
        super().__init__(config=config)
        self.thermal_data = self.data / (self.data.stem + '_thermal')
        self.data = self.data / (self.data.stem + '_rgb')

    def _generate_dataparser_outputs_(self, split="train"):
        meta = load_from_json(self.data / f"transforms.json")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        image_filenames = [image_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        image_filenames = self._hack_filenames(image_filenames, self.data)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs

    def _hack_filenames(self, image_filenames, data):  # HACK
        return [Path('/home/owo/thermal-nerf/data') / data / f.name for f in image_filenames]

    def _generate_dataparser_outputs(self, split="train"):
        # return self._generate_dataparser_outputs_(split=split)
        dpo_rgb = self._generate_dataparser_outputs_(split=split)
        self.data, self.thermal_data = self.thermal_data, self.data  # HACK
        dpo_thermal = self._generate_dataparser_outputs_(split=split)
        self.data, self.thermal_data = self.thermal_data, self.data  # HACK

        cameras = Cameras(
            camera_to_worlds=torch.cat((dpo_rgb.cameras.camera_to_worlds,
                                        dpo_thermal.cameras.camera_to_worlds)),
            fx=torch.cat((dpo_rgb.cameras.fx, dpo_thermal.cameras.fx)),
            fy=torch.cat((dpo_rgb.cameras.fy, dpo_thermal.cameras.fy)),
            cx=torch.cat((dpo_rgb.cameras.cx, dpo_thermal.cameras.cx)),
            cy=torch.cat((dpo_rgb.cameras.cy, dpo_thermal.cameras.cy)),
            camera_type=CameraType.PERSPECTIVE,
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=dpo_rgb.image_filenames + dpo_thermal.image_filenames,
            cameras=cameras,
            # alpha_color=self.alpha_color_tensor,
            scene_box=dpo_rgb.scene_box,
            dataparser_scale=self.scale_factor,
        )
        dataparser_outputs.metadata["is_thermal"] = [0 for _ in range(len(dpo_rgb.image_filenames))]\
                                                    + [1 for _ in range(len(dpo_thermal.image_filenames))]
        return dataparser_outputs
