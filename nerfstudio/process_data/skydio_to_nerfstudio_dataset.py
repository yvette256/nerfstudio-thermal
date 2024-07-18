from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Literal, Optional

import exiftool
import numpy as np
from rich.progress import track
from scipy.spatial.transform import Rotation

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SkydioToNerfstudioDataset(ImagesToNerfstudioDataset):
    """Process images into a thermal nerfstudio dataset."""

    # TODO: [eval_][thermal_]data should mimic whatever the behavior of RGBTToNerfstudioDataset is
    #  currently, probably broken if self.skip_image_processing
    thermal_data: Optional[Path] = None
    """Path to directory of thermal images."""
    eval_thermal_data: Optional[Path] = None
    """Path to eval thermal data."""
    coordinate_convention: Literal["NED", "FLU"] = "NED"
    """Coordinate convention for camera pose."""
    max_num_images: int = -1
    """Maximum number of images to process. If -1, no maximum."""
    rgb_only: bool = False
    """Whether to process only RGB images."""
    use_quat_poses: bool = False
    """Whether to use quaternion poses. If False, uses RPY poses."""
    colmap_transforms_file: Optional[Path] = None
    """Name of colmap transforms json file. If None and skip-colmap, poses in metadata are used."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.thermal_image_dir.mkdir(parents=True, exist_ok=True)

    @property
    def thermal_image_dir(self) -> Path:
        return self.output_dir / "images_thermal"

    def main(self) -> None:
        """Process images into a thermal nerfstudio dataset."""

        summary_log = []

        # Create transforms from Skydio metadata
        transforms = {
            "camera_model": "OPENCV",
            "frames": [],
        }
        n_rgb = 0
        n_thermal = 0

        files = process_data_utils.list_images(self.data, recursive=self.skip_image_processing)
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(files)
        for i, (file, md) in track(enumerate(zip(files, metadata)), description="Reading metadata", transient=True):
            frame = {}
            try:
                frame["is_thermal"] = 1 if md["XMP:CameraSource"] == "INFRARED" else 0
            except KeyError:
                continue
            # Only use thermal frames w/ radiometric data (non-tonemapped)
            if frame["is_thermal"] and "APP1:AtmosphericTransAlpha1" not in md:
                continue
            if self.rgb_only and frame["is_thermal"]:
                continue

            if not self.use_quat_poses:
                # Rotation from yaw, pitch, roll
                roll = md["XMP:CameraOrientationNEDRoll"] * np.pi / 180.
                pitch = md["XMP:CameraOrientationNEDPitch"] * np.pi / 180.
                yaw = md["XMP:CameraOrientationNEDYaw"] * np.pi / 180.
                R_yaw = yaw_matrix(yaw)
                R_pitch = pitch_matrix(pitch)
                R_roll = roll_matrix(roll)
                R = R_yaw @ R_pitch @ R_roll
            else:
                # Rotation from quaternion
                quat_x = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}X"]
                quat_y = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}Y"]
                quat_z = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}Z"]
                quat_w = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}W"]
                R = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w]).as_matrix()

            # Adjust rotation by camera orientation in gimbal
            roll_cam = 90. * np.pi / 180.
            if not self.use_quat_poses:
                pitch_cam = 180. * np.pi / 180.
            else:
                pitch_cam = 0. * np.pi / 180.
            yaw_cam = 270. * np.pi / 180.
            R_cam2gimbal = yaw_matrix(yaw_cam) @ pitch_matrix(pitch_cam) @ roll_matrix(roll_cam)
            R = R @ R_cam2gimbal

            # Camera translation
            cam_x = md[f"XMP:CameraPosition{self.coordinate_convention}X"]
            cam_y = md[f"XMP:CameraPosition{self.coordinate_convention}Y"]
            cam_z = md[f"XMP:CameraPosition{self.coordinate_convention}Z"]
            t = np.array([cam_x, cam_y, cam_z])
            T = np.identity(4)
            T[:3, 3] = t

            M = np.identity(4)
            M[:3, :3] = R
            M[:3, 3] = t
            frame["transform_matrix"] = M.tolist()

            # Camera intrinsics + distortion coeffs + image w, h
            frame["fl_x"] = md["XMP:CalibratedFocalLengthX"]
            frame["fl_y"] = md["XMP:CalibratedFocalLengthY"]
            frame["cx"] = md["XMP:CalibratedOpticalCenterX"]
            frame["cy"] = md["XMP:CalibratedOpticalCenterY"]
            frame["p1"], frame["p2"], frame["k4"], frame["k5"], frame["k6"] = (0. for _ in range(5))
            frame["k1"], frame["k2"], frame["k3"] = map(float, md["XMP:DewarpData"].split(','))
            frame["w"] = md["File:ImageWidth"]
            frame["h"] = md["File:ImageHeight"]

            # file_path of image
            if not self.skip_image_processing:
                # Copy images to self.output_dir
                dst = self.thermal_image_dir if frame["is_thermal"] else self.image_dir
                filename = f"frame_{n_thermal + 1 if frame['is_thermal'] else n_rgb + 1:05d}.jpg"
                subdir = "images_thermal" if frame["is_thermal"] else "images"
                frame["file_path"] = Path(subdir) / filename
                shutil.copy(file, dst / filename)
            else:
                frame["file_path"] = file  # TODO: should be relative to self.data?
            frame["file_path"] = str(frame["file_path"])

            if frame["is_thermal"]:
                n_thermal += 1
            else:
                n_rgb += 1

            transforms["frames"].append(frame)

            if -1 < self.max_num_images <= n_thermal + n_rgb:
                break

        # Downscale RGB images
        if not self.skip_image_processing and self.num_downscales > 0:
            for (image_dir, n) in ((self.image_dir, n_rgb), (self.thermal_image_dir, n_thermal)):
                image_filenames = [image_dir / f"frame_{i:05d}.jpg" for i in range(1, n + 1)]
                copied_image_paths = process_data_utils.copy_images_list(
                    image_filenames,
                    image_dir=image_dir,
                    verbose=self.verbose,
                    num_downscales=self.num_downscales,
                    keep_image_dir=True,
                )
                assert len(copied_image_paths) == n

        # Run COLMAP
        if not self.skip_colmap:
            require_cameras_exist = True
            self._run_colmap()
            # Colmap uses renamed images
            image_rename_map = None

            # Export depth maps
            image_id_to_depth_path, log_tmp = self._export_depth()
            summary_log += log_tmp

            if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
                raise RuntimeError(f"Could not find existing COLMAP results ({self.colmap_model_path / 'cameras.bin'}).")

            summary_log += self._save_transforms(
                n_rgb,
                image_id_to_depth_path,
                None,
                image_rename_map,
            )

        colmap_transforms_path = None
        if not self.skip_colmap:
            colmap_transforms_path = "transforms.json"
        elif self.colmap_transforms_file:
            colmap_transforms_path = self.colmap_transforms_file

        # Use RGB + thermal COLMAP poses
        if colmap_transforms_path:
            with open(self.output_dir / colmap_transforms_path, "r", encoding="utf-8") as f:
                colmap_transforms = json.load(f)
            colmap_transforms["frames"].sort(key=lambda x: x["file_path"])

            # Transform thermal poses into COLMAP space
            metadata_rgb_ind = -1
            colmap_rgb_ind = -1
            for i, frame in enumerate(transforms["frames"]):
                if not frame["is_thermal"]:
                    metadata_rgb_ind = i
                    colmap_rgb_ind += 1
                else:
                    # Latest RGB camera poses
                    M_rgb2metadata = np.array(transforms["frames"][metadata_rgb_ind]["transform_matrix"])
                    M_rgb2colmap = np.array(colmap_transforms["frames"][colmap_rgb_ind]["transform_matrix"])

                    # Thermal poses from metadata
                    M_thermal2metadata = np.array(frame["transform_matrix"])

                    M_thermal2colmap = M_rgb2colmap @ np.linalg.inv(M_rgb2metadata) @ M_thermal2metadata

                    frame["transform_matrix"] = M_thermal2colmap.tolist()
                    frame["is_thermal"] = 1

            # Copy over COLMAP RGB poses
            colmap_camera_params = {k: colmap_transforms[k]
                                    for k in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")}
            colmap_ind = 0
            for i, frame in enumerate(transforms["frames"]):
                if not frame["is_thermal"]:
                    transforms["frames"][i] = colmap_transforms["frames"][colmap_ind]
                    transforms["frames"][i].update(colmap_camera_params)
                    transforms["frames"][i]["is_thermal"] = 0
                    colmap_ind += 1

        assert len(transforms["frames"]) == n_rgb + n_thermal

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(transforms, f, indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)


def yaw_matrix(rad):
    return np.array([
        [np.cos(rad), -np.sin(rad), 0.],
        [np.sin(rad), np.cos(rad), 0.],
        [0., 0., 1.],
    ])


def pitch_matrix(rad):
    return np.array([
        [np.cos(rad), 0., np.sin(rad)],
        [0., 1., 0.],
        [-np.sin(rad), 0., np.cos(rad)],
    ])

def roll_matrix(rad):
    return np.array([
        [1., 0., 0.],
        [0., np.cos(rad), -np.sin(rad)],
        [0., np.sin(rad), np.cos(rad)],
    ])
