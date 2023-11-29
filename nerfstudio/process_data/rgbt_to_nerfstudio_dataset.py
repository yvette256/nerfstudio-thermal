from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from nerfstudio.process_data import calibration_utils, flir_utils, process_data_utils
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class RGBTToNerfstudioDataset(ImagesToNerfstudioDataset):
    """Process images into a thermal nerfstudio dataset."""

    calibration_data: Path = None
    thermal_data: Path = None
    eval_thermal_data: Optional[Path] = None

    def __post_init__(self) -> None:
        flir_utils.extract_raws_from_dir(self.data)
        CONSOLE.log("[bold green]:tada: Extracted raw RGB/T images from FLIR data.")
        self.data = self.data.parent / (self.data.name + "_raw") / "rgb"  # HACK: redefines self.data unintuitively

        super().__post_init__()

        if self.thermal_data is None:
            self.thermal_data = self.data.parent / Path(str(self.data.name).replace('rgb', 'thermal'))

    def _rgb_to_thermal_path(self, path: str) -> str:
        if self.skip_image_processing:
            return path.replace(self.data.as_posix(), self.thermal_data.as_posix())
        else:
            return path.replace("images", "images_thermal")

    @property
    def thermal_image_dir(self) -> Path:
        return self.output_dir / "images_thermal"

    def main(self) -> None:
        """Process images into a thermal nerfstudio dataset."""
        super().main()

        if not self.skip_image_processing:
            # Copy thermal images to output directory
            process_data_utils.copy_images(
                self.thermal_data,
                image_dir=self.thermal_image_dir,
                crop_factor=self.crop_factor,
                image_prefix="frame_train_" if self.eval_data is not None else "frame_",
                verbose=self.verbose,
                num_downscales=0,
                same_dimensions=self.same_dimensions,
                keep_image_dir=False,
            )
            if self.eval_data is not None:
                process_data_utils.copy_images(
                    self.eval_thermal_data,
                    image_dir=self.thermal_image_dir,
                    crop_factor=self.crop_factor,
                    image_prefix="frame_eval_",
                    verbose=self.verbose,
                    num_downscales=0,
                    same_dimensions=self.same_dimensions,
                    keep_image_dir=True,
                )

        # Edit transforms.json for RGBT dataset
        with open(self.output_dir / "transforms.json", "r", encoding="utf-8") as f:
            file_data = json.load(f)

        # Get h, w of thermal image
        h_thermal, w_thermal = cv2.imread(
            (self.output_dir / Path(self._rgb_to_thermal_path(file_data["frames"][0]["file_path"]))).as_posix()
        ).shape[:2]  # FIX: pretty sure this breaks if self.skip_image_processing

        rgb_camera_params = {"w": file_data["w"], "h": file_data["h"]}
        thermal_camera_params = {"w": w_thermal, "h": h_thermal}

        # Calibrate RGB and thermal cameras for transforms.json
        colmap_calibration_transform = np.identity(4)  # from colmap world space to our calibration world space
        rgb_thermal_transform = np.identity(4)  # from rgb camera pose to thermal camera pose in calibration world
        if self.calibration_data is not None:
            flir_utils.extract_raws_from_dir(self.calibration_data)
            cal_rgb_dir = f"{self.calibration_data}_raw/rgb"
            cal_thermal_dir = f"{self.calibration_data}_raw/thermal"
            cal_result = calibration_utils.calibrate_rgb_thermal(
                cal_rgb_dir,
                cal_thermal_dir,
                force_radial_distortion_coeff_K3_to_zero=True,
            )

            # Get intrinsics and distortion coeffs
            mat_rgb, mat_thermal = cal_result["camera_matrix_rgb"], cal_result["camera_matrix_thermal"]
            fx_rgb, fy_rgb, cx_rgb, cy_rgb = mat_rgb[0,0], mat_rgb[1,1], mat_rgb[0,2], mat_rgb[1,2]
            fx_thermal, fy_thermal, cx_thermal, cy_thermal = \
                mat_thermal[0,0], mat_thermal[1,1], mat_thermal[0,2], mat_thermal[1,2]

            dist_rgb, dist_thermal = cal_result["distortion_coeffs_rgb"], cal_result["distortion_coeffs_thermal"]
            k1_rgb, k2_rgb, p1_rgb, p2_rgb = dist_rgb
            k1_thermal, k2_thermal, p1_thermal, p2_thermal = dist_thermal

            scale = (fx_rgb + fy_rgb) / (file_data["fl_x"] + file_data["fl_y"])

            rgb_camera_params["fl_x"] = fx_rgb
            rgb_camera_params["fl_y"] = fy_rgb
            rgb_camera_params["cx"] = cx_rgb
            rgb_camera_params["cy"] = cy_rgb
            rgb_camera_params["k1"] = k1_rgb
            rgb_camera_params["k2"] = k2_rgb
            rgb_camera_params["p1"] = p1_rgb
            rgb_camera_params["p2"] = p2_rgb

            thermal_camera_params["fl_x"] = fx_thermal
            thermal_camera_params["fl_y"] = fy_thermal
            thermal_camera_params["cx"] = cx_thermal
            thermal_camera_params["cy"] = cy_thermal
            thermal_camera_params["k1"] = k1_thermal
            thermal_camera_params["k2"] = k2_thermal
            thermal_camera_params["p1"] = p1_thermal
            thermal_camera_params["p2"] = p2_thermal

            # Get extrinsics
            colmap_calibration_transform[0, 0], colmap_calibration_transform[1, 1], colmap_calibration_transform[2, 2] = scale
            tvec, rmat = cal_result["tvec_relative"], cal_result["rmat_relative"]
            T, R = np.identity(4), np.identity(4)
            T[:3, 3] = tvec
            R[:3, :3] = rmat
            rgb_thermal_transform = T @ R @ rgb_thermal_transform

        camera_params = thermal_camera_params.keys()  # camera params to set as per-frame rather than fixed

        # Build frames for thermal images
        thermal_frames = []
        for i, frame in enumerate(file_data["frames"]):
            thermal_frame_name = self._rgb_to_thermal_path(frame["file_path"])

            # Set params for thermal frame
            thermal_frame = {
                "file_path": thermal_frame_name,
                "transform_matrix":
                    rgb_thermal_transform @ colmap_calibration_transform @ np.array(frame["transform_matrix"]),
                "colmap_im_id": frame["colmap_im_id"],  # NOTE: not sure what this field is used for
                "is_thermal": 1,
            }
            for param in camera_params:
                thermal_frame[param] = thermal_camera_params[param]
            thermal_frames.append(thermal_frame)

            # Set params for RGB frame
            file_data["frames"][i]["is_thermal"] = 0
            for param in camera_params:
                file_data["frames"][i][param] = rgb_camera_params[param]
            file_data["frames"][i]["transform_matrix"] =\
                colmap_calibration_transform @ np.array(frame["transform_matrix"])

        file_data["frames"] += thermal_frames

        # Remove (now) unfixed camera params
        for param in camera_params:
            del file_data[param]

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)

        CONSOLE.log("[bold green]:tada: Done processing thermal data.")
