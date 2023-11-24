from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import cv2

from nerfstudio.process_data import process_data_utils, flir_utils
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class RGBTToNerfstudioDataset(ImagesToNerfstudioDataset):
    """Process images into a thermal nerfstudio dataset."""

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

        with open(self.output_dir / "transforms.json", "r", encoding="utf-8") as f:
            file_data = json.load(f)

        # FIX: pretty sure this breaks if self.skip_image_processing
        h_thermal, w_thermal = cv2.imread(
            (self.output_dir / Path(self._rgb_to_thermal_path(file_data["frames"][0]["file_path"]))).as_posix()
        ).shape[:2]
        thermal_camera_params = {"w": w_thermal, "h": h_thermal}
        camera_params = thermal_camera_params.keys()  # camera params to set as per-frame rather than fixed

        thermal_frames = []
        for i, frame in enumerate(file_data["frames"]):
            thermal_frame_name = self._rgb_to_thermal_path(frame["file_path"])

            for param in camera_params:
                file_data["frames"][i][param] = file_data[param]

            file_data["frames"][i]["is_thermal"] = 0
            thermal_frame = {
                "file_path": thermal_frame_name,
                "transform_matrix": frame["transform_matrix"],
                "colmap_im_id": frame["colmap_im_id"],  # NOTE: not sure what this field is used for
                "is_thermal": 1,
            }
            for param in camera_params:
                thermal_frame[param] = thermal_camera_params[param]
            thermal_frames.append(thermal_frame)

        file_data["frames"] += thermal_frames

        for param in camera_params:
            del file_data[param]

        with open(self.output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)

        CONSOLE.log("[bold green]:tada: Done processing thermal data.")
