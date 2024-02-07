from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.thermal_nerfacto_field import ThermalNerfactoField
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import RGBRenderer, RGBTRenderer
from nerfstudio.model_components.losses import (
    distortion_loss,
    interlevel_loss,
    MSELoss,
    MSELossRGBT,
    scale_gradients_by_distance_squared,
    ssim_rgbt,
    L1Loss,
    LearnedPerceptualImagePatchSimilarityRGBT,
    compute_TVloss,  # PXY
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps


@dataclass
class ThermalNerfactoModelConfig(NerfactoModelConfig):
    """Thermal Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ThermalNerfactoModel)
    density_loss_mult: float = 1e-3  # NOTE: 1e-6 is good
    """Density loss (L1 norm of [rgb density] - [thermal density]) multiplier."""
    density_mode: Literal["rgb_only", "shared", "separate"] = "separate"
    """How to treat density between RGB/T (rgb_only only reconstructs RGB field)."""


class ThermalNerfactoModel(NerfactoModel):
    """Thermal Nerfacto model

    Args:
        config: Thermal Nerfacto configuration to instantiate model
    """
    # TODO: Warning: ambiguous variable names in this class; sometimes "rgb" means "rgbt."
    #  This is b/c it's built on the NerfactoModel class and some attribute/method names might need to be
    #  preserved; still this is probably overused and should be refactored eventually.

    config = ThermalNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        if self.config.density_mode == "separate":
            self.output_suffixes = ("", "_thermal")
        else:
            self.output_suffixes = ("",)

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
            num_channels=3 + (self.config.density_mode == "shared"),
        )
        if self.config.density_mode == "separate":
            self.field_thermal = ThermalNerfactoField(
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
                num_channels=1,
            )

        # Build the thermal proposal network
        self.density_fns_thermal = []

        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks_thermal = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks_thermal.append(network)
            self.density_fns_thermal.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks_thermal.append(network)
            self.density_fns_thermal.extend([network.density_fn for network in self.proposal_networks_thermal])

        # Samplers  # HACK: I don't think I need to redefine this? just pass the correct density_fns
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler_thermal = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # renderers
        self.renderer_rgbt = RGBTRenderer(background_color=self.config.background_color)
        self.renderer_thermal = RGBRenderer(background_color=self.config.background_color,
                                            num_channels=1)

        # losses
        self.rgbt_loss = MSELossRGBT()
        self.density_loss = L1Loss()

        self.tvloss = compute_TVloss

        # metrics
        # XXX: these are untested, but not strictly necessary for model to train
        self.ssim = ssim_rgbt
        self.lpips = LearnedPerceptualImagePatchSimilarityRGBT(normalize=True)

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgbt.blend_background(gt_rgb, batch["is_thermal"])  # Blend if RGBA

        predicted_rgb = outputs["rgb"]
        # metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb, batch["is_thermal"])
        metrics_dict["psnr_rgb"] = self.psnr(
            gt_rgb[..., :3] * (1 - batch["is_thermal"])[:, None],
            predicted_rgb * (1 - batch["is_thermal"])[:, None]
        )
        if not self.config.density_mode == "rgb_only":
            predicted_thermal = outputs["rgb_thermal"]
            metrics_dict["psnr_thermal"] = self.psnr(
                gt_rgb[..., 3:] * batch["is_thermal"][:, None],
                predicted_thermal * batch["is_thermal"][:, None]
            )

        if self.training:
            metrics_dict["distortion"] = 0
            for s in self.output_suffixes:
                metrics_dict["distortion"] += distortion_loss(
                    outputs[f"weights_list{s}"], outputs[f"ray_samples_list{s}"]
                )

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        if not self.config.density_mode == "rgb_only":
            pred_rgb, gt_rgb = self.renderer_rgbt.blend_background_for_loss_computation(
                pred_image=torch.cat((outputs["rgb"], outputs["rgb_thermal"]), dim=1),
                pred_accumulation=outputs["accumulation"],  # XXX: which accumulation should this be?
                gt_image=image,
                is_thermal=batch["is_thermal"],
            )
        else:
            pred_rgb, gt_rgb = self.renderer_rgbt.blend_background_for_loss_computation(
                pred_image=torch.cat((outputs["rgb"], torch.zeros(outputs["rgb"].shape[0], 1).to("cuda")), dim=1),
                pred_accumulation=outputs["accumulation"],  # XXX: which accumulation should this be?
                gt_image=image,
                is_thermal=batch["is_thermal"],
            )


        num_samples = int(5000) # can be changed accordingly
        lambda_tv = 1 # tuning parameter
        loss_dict["tv_loss_combined"] = self.tvloss(self.field.get_density_only(num_points=num_samples, voxelSize = int(2048*lambda_tv)), num_samples=num_samples)
        loss_dict["tv_loss_separated"] = self.tvloss(self.field_thermal.get_density_only(num_points=num_samples, voxelSize=int(2048*lambda_tv)), num_samples=num_samples)
        loss_dict["rgb_loss"] = self.rgb_loss(
            gt_rgb[..., :3] * (1 - batch["is_thermal"])[:, None],
            pred_rgb[..., :3] * (1 - batch["is_thermal"])[:, None]
        )
        if not self.config.density_mode == "rgb_only":
            loss_dict["thermal_loss"] = self.rgb_loss(
                gt_rgb[..., 3:] * batch["is_thermal"][:, None],
                pred_rgb[..., 3:] * batch["is_thermal"][:, None]
            )
        # rgbt_loss = self.rgbt_loss(gt_rgb, pred_rgb, batch["is_thermal"])
        # assert torch.allclose(rgbt_loss, loss_dict["rgb_loss"] + loss_dict["thermal_loss"])

        if self.config.density_mode == "separate" and self.config.density_loss_mult > 0:
            loss_dict["density_loss"] = self.config.density_loss_mult * self.density_loss(
                outputs["density2"], outputs["density_thermal"])
            loss_dict["density_loss"] += self.config.density_loss_mult * self.density_loss(
                outputs["density"], outputs["density2_thermal"])

        if self.training:
            loss_dict["interlevel_loss"] = 0
            loss_dict["distortion_loss"] = 0
            if self.config.predict_normals:
                loss_dict["orientation_loss"] = 0
                loss_dict["pred_normal_loss"] = 0

            for s in self.output_suffixes:
                loss_dict["interlevel_loss"] += self.config.interlevel_loss_mult * interlevel_loss(
                    outputs[f"weights_list{s}"], outputs[f"ray_samples_list{s}"]
                )
                assert metrics_dict is not None and "distortion" in metrics_dict
                loss_dict["distortion_loss"] += self.config.distortion_loss_mult * metrics_dict["distortion"]
                if self.config.predict_normals:
                    # orientation loss for computed normals
                    loss_dict["orientation_loss"] += self.config.orientation_loss_mult * torch.mean(
                        outputs[f"rendered_orientation_loss{s}"]
                    )

                    # ground truth supervision for normals
                    loss_dict["pred_normal_loss"] += self.config.pred_normal_loss_mult * torch.mean(
                        outputs[f"rendered_pred_normal_loss{s}"]
                    )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if self.config.density_mode == "separate":
            # param_groups["proposal_networks"] += list(self.proposal_networks_thermal.parameters())
            # param_groups["fields"] += list(self.field_thermal.parameters())
            param_groups["proposal_networks_thermal"] = list(self.proposal_networks_thermal.parameters())
            param_groups["fields_thermal"] = list(self.field_thermal.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        renderer_rgb = self.renderer_rgb
        if self.config.density_mode == "shared":
            renderer_rgb = self.renderer_rgbt

        outputs = super()._get_outputs(ray_bundle, self.field, renderer_rgb,
                                       ray_samples, weights_list, ray_samples_list)

        if self.config.density_mode == "shared":
            rgbt = outputs["rgb"]
            outputs["rgbt"] = rgbt
            outputs["rgb"] = rgbt[..., :3]
            outputs["rgb_thermal"] = rgbt[..., 3:]

        elif self.config.density_mode == "separate":
            # Thermal outputs
            ray_samples_thermal: RaySamples
            ray_samples_thermal, weights_list_thermal, ray_samples_list_thermal = \
                self.proposal_sampler_thermal(ray_bundle, density_fns=self.density_fns_thermal)
            thermal_outputs = super()._get_outputs(
                ray_bundle, self.field_thermal, self.renderer_thermal,
                ray_samples_thermal, weights_list_thermal, ray_samples_list_thermal
            )
            for k, v in thermal_outputs.items():
                outputs[f"{k}_thermal"] = v

            if self.config.density_loss_mult > 0:
                # Sample rays w/ rgb + thermal fields in same regions for density regularizer
                #   density2 corresponds to same raysamples as density_thermal (and vice versa)
                field_outputs = self.field.forward(ray_samples_thermal, compute_normals=self.config.predict_normals)
                if self.config.use_gradient_scaling:
                    field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
                outputs["density2"] = field_outputs[FieldHeadNames.DENSITY]

                field_outputs = self.field_thermal.forward(ray_samples, compute_normals=self.config.predict_normals)
                if self.config.use_gradient_scaling:
                    field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
                outputs["density2_thermal"] = field_outputs[FieldHeadNames.DENSITY]

        return outputs

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        # FIXME: separate rgb/t
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb, batch["is_thermal"])
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        if self.config.density_mode == "separate":
            combined_rgb = torch.cat([gt_rgb, predicted_rgb, outputs["rgb_thermal"].expand(-1, -1, 3)], dim=1)
            combined_acc = torch.cat([acc, colormaps.apply_colormap(outputs["accumulation_thermal"])], dim=1)
            combined_depth = torch.cat([
                depth,
                colormaps.apply_depth_colormap(
                    outputs["depth_thermal"],
                    accumulation=outputs["accumulation_thermal"],
                )
            ], dim=1)
        else:
            if self.config.density_mode == "rgb_only":
                combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
            else:
                combined_rgb = torch.cat([gt_rgb, predicted_rgb, outputs["rgb_thermal"].expand(-1, -1, 3)], dim=1)
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
                psnr_rgb = self.psnr(gt_rgb[:, :3, :, :], predicted_rgb)
            elif not self.config.density_mode == "rgb_only":
                predicted_thermal = outputs["rgb_thermal"]
                predicted_thermal = torch.moveaxis(predicted_thermal, -1, 0)[None, ...]
                psnr_thermal = self.psnr(gt_rgb[:, 3:, :, :], predicted_thermal)

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
