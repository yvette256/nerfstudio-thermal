from typing import Dict

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class ThermalDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs=dataparser_outputs, scale_factor=scale_factor)

    def get_metadata(self, data: Dict) -> Dict:
        return {"is_thermal": self.metadata["is_thermal"][data["image_idx"]]}
