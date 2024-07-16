from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig


@dataclass
class ThermalNerfDataParserConfig(NerfstudioDataParserConfig):
    """Thermal Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: ThermalNerf)
    """target class to instantiate"""


@dataclass
class ThermalNerf(Nerfstudio):
    """Thermal Nerfstudio DatasetParser"""

    config = ThermalNerfDataParserConfig

    def _generate_dataparser_outputs(self, split="train", metadata_keys=()):
        dataparser_outputs = super()._generate_dataparser_outputs(
            split=split, metadata_keys=("is_thermal",) + metadata_keys)
        return dataparser_outputs

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        if downsample_folder_prefix == "images_":
            downsample_folder_prefix = f"{filepath.parent.name}_"
        return super()._get_fname(filepath, data_dir, downsample_folder_prefix=downsample_folder_prefix)
