import torch
from jaxtyping import Float, Int
from torch import Tensor


def rgb_to_rgbt_image(
        image: Float[Tensor, "*bs 3"],
        is_thermal: Float[Tensor, "*bs"],
) -> Float[Tensor, "*bs 4"]:
    """Turn batched image w/ thermal information into RGBT image.

    Populate empty information with 0 -- this will not be used in the loss anyway.

    Args:
        image: RGB or T(x3) per pixel.
        is_thermal: 0 if RGB, 1 if thermal, per pixel.

    Returns:
        RGBT per pixel.
    """
    n_images = image.shape[0]
    rgbt = torch.zeros(n_images, 4).to(image.device)
    is_rgb = 1 - is_thermal
    rgbt[..., :3] = torch.einsum("ij,i->ij", image, is_rgb)
    rgbt[..., 3] = image[..., 0] * is_thermal
    return rgbt


def align_gt_with_pred_rgbt(
        gt_rgbt: Float[Tensor, "*bs 4"],
        pred_rgbt: Float[Tensor, "*bs 4"],
        is_thermal: Float[Tensor, "*bs"],
) -> Float[Tensor, "*bs 4"]:
    is_rgb = 1 - is_thermal
    # if gt is thermal, use predicted rgb values for 0 rgb loss
    gt_rgbt[..., :3] += torch.einsum("ij,i->ij", pred_rgbt[..., :3], is_thermal)
    # if gt is rgb, use predicted thermal values for 0 thermal loss
    gt_rgbt[..., 3] += pred_rgbt[..., 3] * is_rgb
    return gt_rgbt


