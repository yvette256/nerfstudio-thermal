from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.utils.io import load_from_json


@dataclass
class ThermalNerfDataParserConfig(NerfstudioDataParserConfig):
    """Thermal Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: ThermalNerf)
    """target class to instantiate"""


@dataclass
class ThermalNerf(Nerfstudio):
    """Thermal Nerfstudio DatasetParser"""

    config = ThermalNerfDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        dataparser_outputs = super()._generate_dataparser_outputs(split=split)

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
        else:
            meta = load_from_json(self.config.data / "transforms.json")

        is_thermal = [frame["is_thermal"] for frame in meta["frames"]]
        dataparser_outputs.metadata["is_thermal"] = is_thermal
        return dataparser_outputs
