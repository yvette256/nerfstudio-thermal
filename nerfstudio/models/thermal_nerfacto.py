from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import torch
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.thermal_nerfacto_field import ThermalNerfactoField
from nerfstudio.model_components.renderers import RGBTRenderer
from nerfstudio.model_components.losses import (
    distortion_loss,
    interlevel_loss,
    MSELossRGBT,
    ssim_rgbt,
    LearnedPerceptualImagePatchSimilarityRGBT,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps


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

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = ThermalNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        # renderers
        self.renderer_rgb = RGBTRenderer(background_color=self.config.background_color)

        # losses
        self.rgb_loss = MSELossRGBT()

        # metrics
        self.ssim = ssim_rgbt
        self.lpips = LearnedPerceptualImagePatchSimilarityRGBT(normalize=True)

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb, batch["is_thermal"])  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb, batch["is_thermal"])

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
            is_thermal=batch["is_thermal"],
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb, batch["is_thermal"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        rgbt = outputs["rgb"]
        outputs["rgb_actual"] = rgbt[..., :3]
        outputs["thermal"] = rgbt[..., 3:]
        return outputs

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb, batch["is_thermal"])
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        # psnr = self.psnr(gt_rgb, predicted_rgb, batch["is_thermal"])
        # ssim = self.ssim(gt_rgb, predicted_rgb, batch["is_thermal"])
        # lpips = self.lpips(gt_rgb, predicted_rgb, batch["is_thermal"])
        psnr_rgb = Tensor([-1])
        psnr_thermal = Tensor([-1])
        if not hasattr(batch["is_thermal"], "__len__"):  # HACK: want better extension to if is_thermal is tensor
            if batch["is_thermal"] < 1:
                psnr_rgb = self.psnr(gt_rgb[:, :3, :, :], predicted_rgb[:, :3, :, :])
            else:
                psnr_thermal = self.psnr(gt_rgb[:, 3:, :, :], predicted_rgb[:, 3:, :, :])

        # all of these metrics will be logged as scalars
        # metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict = {}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)
        metrics_dict["psnr_rgb"] = float(psnr_rgb.item())
        metrics_dict["psnr_thermal"] = float(psnr_thermal.item())

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
