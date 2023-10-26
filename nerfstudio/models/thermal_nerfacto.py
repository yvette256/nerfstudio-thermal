from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class ThermalNerfactoModelConfig(NerfactoModelConfig):
    """Additional parameters for thermal model."""

    _target: Type = field(default_factory=lambda: ThermalNerfactoModel)


class ThermalNerfactoModel(NerfactoModel):
    """Thermal data augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config = ThermalNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
