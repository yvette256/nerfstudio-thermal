from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Literal, Optional

import exiftool
import numpy as np
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.thermal_image_dir.mkdir(parents=True, exist_ok=True)

    def _rgb_to_thermal_path(self, path: str) -> str:
        # HACK: not robust to different data / thermal_data paths
        return path.replace("images", "images_thermal")

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
        group_to_rgb_frame_inds = {}
        frame_path_to_group = {}
        frame_groups = []

        files = process_data_utils.list_images(self.data, recursive=True)
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(files)
        for i, (file, md) in enumerate(zip(files, metadata)):
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

            # Rotation from yaw, pitch, roll
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
            # R = R_yaw @ R_roll @ R_pitch
            # R = R_roll @ R_yaw @ R_pitch
            # R = R_roll @ R_pitch @ R_yaw
            # R = R_pitch @ R_roll @ R_yaw
            # R = R_pitch @ R_yaw @ R_roll

            # Rotation from quaternion
            quat_x = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}X"]
            quat_y = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}Y"]
            quat_z = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}Z"]
            quat_w = md[f"XMP:CameraOrientationQuat{self.coordinate_convention}W"]
            R = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w]).as_matrix()

            # Camera translation
            cam_x = md[f"XMP:CameraPosition{self.coordinate_convention}X"]
            cam_y = md[f"XMP:CameraPosition{self.coordinate_convention}Y"]
            cam_z = md[f"XMP:CameraPosition{self.coordinate_convention}Z"]
            t = np.array([cam_x, cam_y, cam_z])
            T = np.identity(4)
            T[:3, 3] = t

            M = np.identity(4)
            M[:3, :3] = R
            # TODO: what are the correct c2w matrices?
            # M = M @ T
            M[:3, 3] = t
            # M = np.linalg.inv(M)
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
                filename = f"frame_{n_thermal if frame['is_thermal'] else n_rgb:05d}.jpg"
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

            # Separate frames into groups w/ no relative transform ambiguity (according to subdir)
            if not frame["is_thermal"]:
                if file.parent not in group_to_rgb_frame_inds:
                    group_to_rgb_frame_inds[file.parent] = []
                group_to_rgb_frame_inds[file.parent].append(i)
            frame_path_to_group[frame["file_path"]] = file.parent
            frame_groups.append(file.parent)

            transforms["frames"].append(frame)

            if -1 < self.max_num_images <= n_thermal + n_rgb:
                break

        # Skip COLMAP if no relative transform ambiguity
        if len(group_to_rgb_frame_inds) < 2:
            self.skip_colmap = True

        if not self.skip_colmap:
            # Run COLMAP on all collected RGB images
            require_cameras_exist = True
            self._run_colmap()

            # Export depth maps
            image_id_to_depth_path, log_tmp = self._export_depth()
            summary_log += log_tmp

            if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
                raise RuntimeError(f"Could not find existing COLMAP results ({self.colmap_model_path / 'cameras.bin'}).")

            summary_log += self._save_transforms(
                n_rgb,
                image_id_to_depth_path,
                None,
                None,
            )

            # Compute transform between COLMAP poses and groups of metadata poses
            with open(self.output_dir / "transforms.json", "r", encoding="utf-8") as f:
                colmap_data = json.load(f)

            colmap_group_to_frame_inds = {k: [] for k in group_to_rgb_frame_inds.keys()}
            for i, frame in enumerate(colmap_data["frames"]):
                file_path = frame["file_path"]
                colmap_group_to_frame_inds[frame_path_to_group[file_path]].append(i)

            for group, frame_inds in group_to_rgb_frame_inds.items():
                group_transforms = [transforms["frames"][i]["transform_matrix"] for i in frame_inds]
                colmap_frame_inds = colmap_group_to_frame_inds[group]
                colmap_transforms = [colmap_data["frames"][i]["transform_matrix"] for i in colmap_frame_inds]

            # TODO: resolve relative transform ambiguity: either remove this if not needed or finish it lol

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(transforms, f, indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)
