# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Processes a video or image sequence to a nerfstudio compatible dataset."""


import sys
import zipfile
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Union

import exiftool
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from typing import Literal
from typing_extensions import Annotated

from nerfstudio.process_data import (
    metashape_utils,
    odm_utils,
    polycam_utils,
    process_data_utils,
    realitycapture_utils,
    record3d_utils,
)
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.process_data.video_to_nerfstudio_dataset import VideoToNerfstudioDataset
from nerfstudio.process_data.rgbt_to_nerfstudio_dataset import RGBTToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ProcessRecord3D(BaseConverterToNerfstudioDataset):
    """Process Record3D data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Record3D poses into the nerfstudio format.
    """

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 300
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        record3d_image_dir = self.data / "rgb"

        if not record3d_image_dir.exists():
            raise ValueError(f"Image directory {record3d_image_dir} doesn't exist")

        record3d_image_filenames = []
        for f in record3d_image_dir.iterdir():
            if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    record3d_image_filenames.append(f)

        record3d_image_filenames = sorted(record3d_image_filenames, key=lambda fn: int(fn.stem))
        num_images = len(record3d_image_filenames)
        idx = np.arange(num_images)
        if self.max_dataset_size != -1 and num_images > self.max_dataset_size:
            idx = np.round(np.linspace(0, num_images - 1, self.max_dataset_size)).astype(int)

        record3d_image_filenames = list(np.array(record3d_image_filenames)[idx])
        # Copy images to output directory
        copied_image_paths = process_data_utils.copy_images_list(
            record3d_image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
            num_downscales=self.num_downscales,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        summary_log.append(f"Used {num_frames} images out of {num_images} total")
        if self.max_dataset_size > 0:
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
            )

        metadata_path = self.data / "metadata.json"
        record3d_utils.record3d_to_json(copied_image_paths, metadata_path, self.output_dir, indices=idx)
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessPolycam(BaseConverterToNerfstudioDataset):
    """Process Polycam data into a nerfstudio dataset.

    To capture data, use the Polycam app on an iPhone or iPad with LiDAR. The capture must be in LiDAR or ROOM mode.
    Developer mode must be enabled in the app settings, this will enable a raw data export option in the export menus.
    The exported data folder is used as the input to this script.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Polycam poses into the nerfstudio format.
    """

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    use_uncorrected_images: bool = False
    """If True, use the raw images from the polycam export. If False, use the corrected images."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""
    min_blur_score: float = 25
    """Minimum blur score to use an image. If the blur score is below this value, the image will be skipped."""
    crop_border_pixels: int = 15
    """Number of pixels to crop from each border of the image. Useful as borders may be black due to undistortion."""
    use_depth: bool = False
    """If True, processes the generated depth maps from Polycam"""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        if self.data.suffix == ".zip":
            with zipfile.ZipFile(self.data, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
                extracted_folder = zip_ref.namelist()[0].split("/")[0]
            self.data = self.output_dir / extracted_folder

        if (self.data / "keyframes" / "corrected_images").exists() and not self.use_uncorrected_images:
            polycam_image_dir = self.data / "keyframes" / "corrected_images"
            polycam_cameras_dir = self.data / "keyframes" / "corrected_cameras"
        else:
            polycam_image_dir = self.data / "keyframes" / "images"
            polycam_cameras_dir = self.data / "keyframes" / "cameras"
            if not self.use_uncorrected_images:
                CONSOLE.print("[bold yellow]Corrected images not found, using raw images.")

        if not polycam_image_dir.exists():
            raise ValueError(f"Image directory {polycam_image_dir} doesn't exist")

        if not (self.data / "keyframes" / "depth").exists():
            depth_dir = self.data / "keyframes" / "depth"
            raise ValueError(f"Depth map directory {depth_dir} doesn't exist")

        (image_processing_log, polycam_image_filenames) = polycam_utils.process_images(
            polycam_image_dir,
            image_dir,
            crop_border_pixels=self.crop_border_pixels,
            max_dataset_size=self.max_dataset_size,
            num_downscales=self.num_downscales,
            verbose=self.verbose,
        )

        summary_log.extend(image_processing_log)

        polycam_depth_filenames = []
        if self.use_depth:
            polycam_depth_image_dir = self.data / "keyframes" / "depth"
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            (depth_processing_log, polycam_depth_filenames) = polycam_utils.process_depth_maps(
                polycam_depth_image_dir,
                depth_dir,
                num_processed_images=len(polycam_image_filenames),
                crop_border_pixels=self.crop_border_pixels,
                max_dataset_size=self.max_dataset_size,
                num_downscales=self.num_downscales,
                verbose=self.verbose,
            )
            summary_log.extend(depth_processing_log)

        summary_log.extend(
            polycam_utils.polycam_to_json(
                image_filenames=polycam_image_filenames,
                depth_filenames=polycam_depth_filenames,
                cameras_dir=polycam_cameras_dir,
                output_dir=self.output_dir,
                min_blur_score=self.min_blur_score,
                crop_border_pixels=self.crop_border_pixels,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class _NoDefaultProcessMetashape:
    """Private class to order the parameters of ProcessMetashape in the right order for default values."""

    xml: Path
    """Path to the Metashape xml file."""


@dataclass
class ProcessMetashape(BaseConverterToNerfstudioDataset, _NoDefaultProcessMetashape):
    """Process Metashape data into a nerfstudio dataset.

    This script assumes that cameras have been aligned using Metashape. After alignment, it is necessary to export the
    camera poses as a `.xml` file. This option can be found under `File > Export > Export Cameras`.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Metashape poses into the nerfstudio format.
    """

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        if self.xml.suffix != ".xml":
            raise ValueError(f"XML file {self.xml} must have a .xml extension")
        if not self.xml.exists:
            raise ValueError(f"XML file {self.xml} doesn't exist")
        if self.eval_data is not None:
            raise ValueError("Cannot use eval_data since cameras were already aligned with Metashape.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        image_filenames, num_orig_images = process_data_utils.get_image_filenames(self.data, self.max_dataset_size)
        copied_image_paths = process_data_utils.copy_images_list(
            image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
            num_downscales=self.num_downscales,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        original_names = [image_path.stem for image_path in image_filenames]
        image_filename_map = dict(zip(original_names, copied_image_paths))

        if self.max_dataset_size > 0 and num_frames != num_orig_images:
            summary_log.append(f"Started with {num_frames} images out of {num_orig_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
            )
        else:
            summary_log.append(f"Started with {num_frames} images")

        # Save json
        if num_frames == 0:
            CONSOLE.print("[bold red]No images found, exiting")
            sys.exit(1)
        summary_log.extend(
            metashape_utils.metashape_to_json(
                image_filename_map=image_filename_map,
                xml_filename=self.xml,
                output_dir=self.output_dir,
                verbose=self.verbose,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class _NoDefaultProcessRealityCapture:
    """Private class to order the parameters of ProcessRealityCapture in the right order for default values."""

    csv: Path
    """Path to the RealityCapture cameras CSV file."""


@dataclass
class ProcessRealityCapture(BaseConverterToNerfstudioDataset, _NoDefaultProcessRealityCapture):
    """Process RealityCapture data into a nerfstudio dataset.

    This script assumes that cameras have been aligned using RealityCapture. After alignment, it is necessary to
    export the camera poses as a `.csv` file using the `Internal/External camera parameters` option.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts RealityCapture poses into the nerfstudio format.
    """

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        if self.csv.suffix != ".csv":
            raise ValueError(f"CSV file {self.csv} must have a .csv extension")
        if not self.csv.exists:
            raise ValueError(f"CSV file {self.csv} doesn't exist")
        if self.eval_data is not None:
            raise ValueError("Cannot use eval_data since cameras were already aligned with RealityCapture.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        image_filenames, num_orig_images = process_data_utils.get_image_filenames(self.data, self.max_dataset_size)
        copied_image_paths = process_data_utils.copy_images_list(
            image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
            num_downscales=self.num_downscales,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        original_names = [image_path.stem for image_path in image_filenames]
        image_filename_map = dict(zip(original_names, copied_image_paths))

        if self.max_dataset_size > 0 and num_frames != num_orig_images:
            summary_log.append(f"Started with {num_frames} images out of {num_orig_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
            )
        else:
            summary_log.append(f"Started with {num_frames} images")

        # Save json
        if num_frames == 0:
            CONSOLE.print("[bold red]No images found, exiting")
            sys.exit(1)
        summary_log.extend(
            realitycapture_utils.realitycapture_to_json(
                image_filename_map=image_filename_map,
                csv_filename=self.csv,
                output_dir=self.output_dir,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessODM(BaseConverterToNerfstudioDataset):
    """Process ODM data into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts ODM poses into the nerfstudio format.
    """

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        orig_images_dir = self.data / "images"
        cameras_file = self.data / "cameras.json"
        shots_file = self.data / "odm_report" / "shots.geojson"
        reconstruction_file = self.data / "opensfm" / "reconstruction.json"

        if not shots_file.exists:
            raise ValueError(f"shots file {shots_file} doesn't exist")
        if not shots_file.exists:
            raise ValueError(f"cameras file {cameras_file} doesn't exist")

        if not orig_images_dir.exists:
            raise ValueError(f"Images dir {orig_images_dir} doesn't exist")

        if self.eval_data is not None:
            raise ValueError("Cannot use eval_data since cameras were already aligned with ODM.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        image_filenames, num_orig_images = process_data_utils.get_image_filenames(
            orig_images_dir, self.max_dataset_size
        )
        copied_image_paths = process_data_utils.copy_images_list(
            image_filenames,
            image_dir=image_dir,
            verbose=self.verbose,
            num_downscales=self.num_downscales,
        )
        num_frames = len(copied_image_paths)

        copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
        original_names = [image_path.stem for image_path in image_filenames]
        image_filename_map = dict(zip(original_names, copied_image_paths))

        if self.max_dataset_size > 0 and num_frames != num_orig_images:
            summary_log.append(f"Started with {num_frames} images out of {num_orig_images} total")
            summary_log.append(
                "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
            )
        else:
            summary_log.append(f"Started with {num_frames} images")

        # Save json
        if num_frames == 0:
            CONSOLE.print("[bold red]No images found, exiting")
            sys.exit(1)
        summary_log.extend(
            odm_utils.cameras2nerfds(
                image_filename_map=image_filename_map,
                cameras_file=cameras_file,
                shots_file=shots_file,
                reconstruction_file=reconstruction_file,
                output_dir=self.output_dir,
                verbose=self.verbose,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessSkydio(BaseConverterToNerfstudioDataset):
    skip_image_processing: bool = False
    """If True, skips copying of images"""
    coordinate_convention: Literal["NED", "FLU"] = "NED"

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        thermal_image_dir = self.output_dir / "images_thermal"
        thermal_image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        transforms = {
            "camera_model": "OPENCV",
            "frames": [],
        }
        n_rgb = 0
        n_thermal = 0

        files = process_data_utils.list_images(self.data, recursive=True)
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(files)
        for i, (file, md) in enumerate(zip(files, metadata)):
            frame = {}
            frame["is_thermal"] = 1 if md["XMP:CameraSource"] == "INFRARED" else 0
            if frame["is_thermal"] and "APP1:AtmosphericTransAlpha1" not in md:
                continue

            # roll = md["XMP:CameraOrientationNEDRoll"]
            # pitch = md["XMP:CameraOrientationNEDPitch"]
            # yaw = md["XMP:CameraOrientationNEDYaw"]
            # R_yaw = np.array([
            #     [np.cos(yaw), -np.sin(yaw), 0],
            #     [np.sin(yaw), np.cos(yaw), 0],
            #     [0, 0, 1],
            # ])
            # R_pitch = np.array([
            #     [np.cos(pitch), 0, np.sin(pitch)],
            #     [0, 1, 0],
            #     [-np.sin(pitch), 0, np.cos(pitch)],
            # ])
            # R_roll = np.array([
            #     [1, 0, 0],
            #     [0, np.cos(roll), -np.sin(roll)],
            #     [0, np.sin(roll), np.cos(roll)],
            # ])
            # R = R_yaw @ R_pitch @ R_roll

            quat_x = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}X"]
            quat_y = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}Y"]
            quat_z = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}Z"]
            quat_w = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}W"]
            R = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w]).as_matrix()

            cam_x = md[f"XMP:CameraPosition{self.coordinate_convention}X"]
            cam_y = md[f"XMP:CameraPosition{self.coordinate_convention}Y"]
            cam_z = md[f"XMP:CameraPosition{self.coordinate_convention}Z"]
            t = np.array([cam_x, cam_y, cam_z])
            T = np.identity(4)
            T[:3, 3] = t

            M = np.identity(4)
            M[:3, :3] = R
            M = M @ T

            frame["transform_matrix"] = M.tolist()
            frame["fl_x"] = md["XMP:CalibratedFocalLengthX"]
            frame["fl_y"] = md["XMP:CalibratedFocalLengthY"]
            frame["cx"] = md["XMP:CalibratedOpticalCenterX"]
            frame["cy"] = md["XMP:CalibratedOpticalCenterY"]
            frame["p1"], frame["p2"], frame["k4"], frame["k5"], frame["k6"] = (0. for _ in range(5))
            frame["k1"], frame["k2"], frame["k3"] = map(float, md["XMP:DewarpData"].split(','))
            frame["w"] = md["File:ImageWidth"]
            frame["h"] = md["File:ImageHeight"]

            if not self.skip_image_processing:
                dst = thermal_image_dir if frame["is_thermal"] else image_dir
                filename = f"frame_{n_thermal if frame['is_thermal'] else n_rgb:05d}.jpg"
                subdir = "images_thermal" if frame["is_thermal"] else "images"
                frame["file_path"] = Path(subdir) / filename
                shutil.copy(file, dst / filename)
            else:
                frame["file_path"] = file
            frame["file_path"] = str(frame["file_path"])

            if frame["is_thermal"]:
                n_thermal += 1
            else:
                n_rgb += 1

            transforms["frames"].append(frame)

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(transforms, f, indent=4)

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class NotInstalled:
    def main(self) -> None:
        ...


Commands = Union[
    Annotated[ImagesToNerfstudioDataset, tyro.conf.subcommand(name="images")],
    Annotated[VideoToNerfstudioDataset, tyro.conf.subcommand(name="video")],
    Annotated[RGBTToNerfstudioDataset, tyro.conf.subcommand(name="rgbt")],
    Annotated[ProcessPolycam, tyro.conf.subcommand(name="polycam")],
    Annotated[ProcessMetashape, tyro.conf.subcommand(name="metashape")],
    Annotated[ProcessRealityCapture, tyro.conf.subcommand(name="realitycapture")],
    Annotated[ProcessRecord3D, tyro.conf.subcommand(name="record3d")],
    Annotated[ProcessODM, tyro.conf.subcommand(name="odm")],
    Annotated[ProcessSkydio, tyro.conf.subcommand(name="skydio")],
]

# Add aria subcommand if projectaria_tools is installed.
try:
    import projectaria_tools
except ImportError:
    projectaria_tools = None

if projectaria_tools is not None:
    from nerfstudio.scripts.datasets.process_project_aria import ProcessProjectAria

    # Note that Union[A, Union[B, C]] == Union[A, B, C].
    Commands = Union[Commands, Annotated[ProcessProjectAria, tyro.conf.subcommand(name="aria")]]
else:
    Commands = Union[
        Commands,
        Annotated[
            NotInstalled,
            tyro.conf.subcommand(
                name="aria",
                description="**Not installed.** Processing Project Aria data requires `pip install projectaria_tools'[all]'`.",
            ),
        ],
    ]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # type: ignore
