from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from nerfstudio.process_data import (
    calibration_utils, colmap_utils, flir_utils, hloc_utils, process_data_utils
)
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class RGBTToNerfstudioDataset(ImagesToNerfstudioDataset):
    """Process images into a thermal nerfstudio dataset."""

    # calibration_data: Optional[Union[Path, List[Path]]] = None
    calibration_data: Optional[List[Path]] = None
    """Paths to directories containing calibration images."""
    thermal_data: Optional[Path] = None
    """Path to directory of thermal images."""
    eval_thermal_data: Optional[Path] = None
    """Path to eval thermal data."""
    upsample_thermal: bool = False
    """If true, upsample thermal images to same resolution as RGB images when extracting raws."""
    skip_calibration_processing: bool = False
    """If true, skip extracting raw RGB/thermal from calibration FLIR data."""

    def __post_init__(self) -> None:
        if not self.skip_image_processing:
            flir_utils.extract_raws_from_dir(self.data, upsample_thermal=self.upsample_thermal)
            CONSOLE.log("[bold green]:tada: Extracted raw RGB/T images from FLIR data.")
            self.data = self.data.parent / (self.data.name + "_raw") / "rgb"  # HACK: redefines self.data unintuitively
        else:
            self.data = self.data / "images"  # HACK: redefines self.data unintuitively

        super().__post_init__()

        if self.thermal_data is None:
            if not self.skip_image_processing:
                self.thermal_data = self.data.parent / "thermal"
            else:
                self.thermal_data = self.data.parent / "images_thermal"

        self.mat_rgb = None  # RGB camera intrinsic matrix
        self.dist_rgb = None  # RGB camera distortion coefficients

    def _rgb_to_thermal_path(self, path: str) -> str:
        # HACK: not robust to different data / thermal_data paths
        return path.replace("images", "images_thermal")

    @property
    def thermal_image_dir(self) -> Path:
        return self.output_dir / "images_thermal"

    def _run_colmap(self, mask_path: Optional[Path] = None):
        """
        Args:
            mask_path: Path to the camera mask. Defaults to None.
        """
        self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)

        (
            sfm_tool,
            feature_type,
            matcher_type,
        ) = process_data_utils.find_tool_feature_matcher_combination(
            self.sfm_tool, self.feature_type, self.matcher_type
        )
        # check that sfm_tool is hloc if using refine_pixsfm
        if self.refine_pixsfm:
            assert sfm_tool == "hloc", "refine_pixsfm only works with sfm_tool hloc"

        # set the image_dir if didn't copy
        if self.skip_image_processing:
            image_dir = self.data
        else:
            image_dir = self.image_dir

        if sfm_tool == "colmap":
            colmap_utils.run_colmap(
                image_dir=image_dir,
                colmap_dir=self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                camera_mask_path=mask_path,
                gpu=self.gpu,
                verbose=self.verbose,
                matching_method=self.matching_method,
                colmap_cmd=self.colmap_cmd,
                camera_matrix=self.mat_rgb,
                dist_coeffs=self.dist_rgb,
            )
        elif sfm_tool == "hloc":
            if mask_path is not None:
                raise RuntimeError("Cannot use a mask with hloc. Please remove the cropping options " "and try again.")

            assert feature_type is not None
            assert matcher_type is not None
            assert matcher_type != "NN"  # Only used for colmap.
            hloc_utils.run_hloc(
                image_dir=image_dir,
                colmap_dir=self.absolute_colmap_path,
                camera_model=CAMERA_MODELS[self.camera_type],
                verbose=self.verbose,
                matching_method=self.matching_method,
                feature_type=feature_type,
                matcher_type=matcher_type,
                refine_pixsfm=self.refine_pixsfm,
            )
        else:
            raise RuntimeError("Invalid combination of sfm_tool, feature_type, and matcher_type, " "exiting")

    def main(self) -> None:
        """Process images into a thermal nerfstudio dataset."""
        # Calibrate RGB and thermal cameras
        if self.calibration_data is not None:
            if not self.skip_calibration_processing:
                for path in self.calibration_data:
                    flir_utils.extract_raws_from_dir(path, normalize_per_image=True)
            cal_rgb_dirs = [f"{path}_raw/rgb" for path in self.calibration_data]
            cal_thermal_dirs = [f"{path}_raw/thermal" for path in self.calibration_data]
            cal_result = calibration_utils.calibrate_rgb_thermal(
                cal_rgb_dirs,
                cal_thermal_dirs,
                intrinsic_calibration_mode=4,
                # force_tangential_distortion_coeffs_to_zero=True,
                force_radial_distortion_coeff_K3_to_zero=True,
                upsample_thermal=self.upsample_thermal,
                show_preview=False,
            )

            self.mat_rgb, mat_thermal = cal_result["camera_matrix_rgb"], cal_result["camera_matrix_thermal"]
            self.dist_rgb, dist_thermal = cal_result["distortion_coeffs_rgb"], cal_result["distortion_coeffs_thermal"]

        # RGB image data processing, runs COLMAP with calibrated RGB camera if applicable
        super().main()

        # Copy thermal images
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
        ).shape[:2]  # HACK: pretty sure this breaks if self.skip_image_processing

        print('==============================')
        print(f"fl_x: {file_data['fl_x']}")
        print(f"fl_y: {file_data['fl_y']}")
        print(f"cx: {file_data['cx']}")
        print(f"cy: {file_data['cy']}")
        print('==============================')

        # Get calibrated RGB and thermal camera params for transforms.json
        rgb_camera_params = {"w": file_data["w"], "h": file_data["h"]}
        thermal_camera_params = {"w": w_thermal, "h": h_thermal}

        M_colmap_world = np.identity(4)  # transform from colmap world space to our calibration world space
        M_world_colmap = np.identity(4)  # transform from our calibration world space to colmap world space
        M_rgb_thermal = np.identity(4)  # transform from rgb camera pose to thermal camera pose in calibration world
        M_thermal_rgb = np.identity(4)  # transform from thermal camera pose to rgb camera pose in calibration world
        world_colmap_scale = 1
        if self.calibration_data is not None:
            # Get intrinsics and distortion coeffs
            fx_rgb, fy_rgb, cx_rgb, cy_rgb = self.mat_rgb[0,0], self.mat_rgb[1,1], self.mat_rgb[0,2], self.mat_rgb[1,2]
            fx_thermal, fy_thermal, cx_thermal, cy_thermal = \
                mat_thermal[0,0], mat_thermal[1,1], mat_thermal[0,2], mat_thermal[1,2]

            k1_rgb, k2_rgb, p1_rgb, p2_rgb, k3_rgb = self.dist_rgb.squeeze()
            k1_thermal, k2_thermal, p1_thermal, p2_thermal, k3_thermal = dist_thermal.squeeze()

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
            # HACK: Hardcoding some values here to get scale estimation to work, will only work for
            #  specific scene. Change method of calculating scale later.
            img1 = "images/frame_00003.png"
            img2 = "images/frame_00004.png"
            world_distance = 12. * 2.54  # 1 ft in cm
            frame1 = next((x for x in file_data["frames"] if x["file_path"] == img1), None)
            frame2 = next((x for x in file_data["frames"] if x["file_path"] == img2), None)
            M1 = np.array(frame1["transform_matrix"])
            M2 = np.array(frame2["transform_matrix"])
            colmap_distance = np.linalg.norm((M1 - M2) @ np.array([0., 0., 0., 1.]))
            world_colmap_scale = colmap_distance / world_distance
            print(f"world_colmap_scale: {world_colmap_scale}")
            M_world_colmap[0,0], M_world_colmap[1,1], M_world_colmap[2,2] = (world_colmap_scale for _ in range(3))
            M_colmap_world[0,0], M_colmap_world[1,1], M_colmap_world[2,2] = (1 / world_colmap_scale for _ in range(3))

            M_thermal_rgb = cal_result["thermal_rgb_transform"]
            # M_rgb_thermal = cal_result["rgb_thermal_transform"]

        camera_params = thermal_camera_params.keys()  # camera params to set as per-frame rather than fixed

        # Build frames for thermal images
        thermal_frames = []
        for i, frame in enumerate(file_data["frames"]):
            thermal_frame_name = self._rgb_to_thermal_path(frame["file_path"])

            # Set params for thermal frame
            M_thermal_rgb_colmap = M_thermal_rgb.copy()
            M_thermal_rgb_colmap[:3,3] *= world_colmap_scale
            thermal_frame = {
                "file_path": thermal_frame_name,
                "transform_matrix":
                    (np.array(frame["transform_matrix"]) @ M_world_colmap @ M_thermal_rgb @ M_colmap_world).tolist(),
                # "transform_matrix":
                #     (np.array(frame["transform_matrix"]) @ M_thermal_rgb_colmap).tolist(),
                "colmap_im_id": frame["colmap_im_id"] + len(file_data["frames"]),  # NOTE: not sure what this field is used for
                "is_thermal": 1,
            }
            for param in camera_params:
                thermal_frame[param] = thermal_camera_params[param]
            thermal_frames.append(thermal_frame)

            # Set params for RGB frame
            file_data["frames"][i]["is_thermal"] = 0
            for param in camera_params:
                file_data["frames"][i][param] = rgb_camera_params[param]
            # file_data["frames"][i]["transform_matrix"] = M_colmap_world @ np.array(frame["transform_matrix"]),

        file_data["frames"] += thermal_frames

        # Remove (now) unfixed camera params
        for param in camera_params:
            del file_data[param]

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)

        CONSOLE.log("[bold green]:tada: Done processing thermal data.")
