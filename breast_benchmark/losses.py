from __future__ import annotations

import torch
import torch.nn.functional as F


def structure_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Boundary-weighted BCE plus weighted IoU used by the shared comparator protocol."""
    weight = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    probability = torch.sigmoid(logits)
    intersection = (probability * mask * weight).sum(dim=(2, 3))
    union = ((probability + mask) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()


def supervised_loss(outputs: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for output in outputs:
        if output.shape[-2:] != mask.shape[-2:]:
            output = F.interpolate(output, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(structure_loss(output, mask))
    return torch.stack(losses).mean()


def dual_supervised_loss(
    outputs: list[torch.Tensor],
    mask: torch.Tensor,
    background_weight: float = 0.8,
) -> torch.Tensor:
    """PraNet-V2 foreground structure loss plus paired background supervision."""
    if len(outputs) == 0 or len(outputs) % 2 != 0:
        raise ValueError("Dual supervision expects paired foreground/background outputs")

    branch_count = len(outputs) // 2
    foreground_outputs = outputs[:branch_count]
    background_outputs = outputs[branch_count:]
    background_mask = 1.0 - mask
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, 31, stride=1, padding=15) - mask
    )
    losses = []
    for foreground, background in zip(foreground_outputs, background_outputs):
        if foreground.shape[-2:] != mask.shape[-2:]:
            foreground = F.interpolate(
                foreground, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        if background.shape[-2:] != mask.shape[-2:]:
            background = F.interpolate(
                background, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        background_bce = F.binary_cross_entropy_with_logits(
            background, background_mask, reduction="none"
        )
        weighted_background_bce = (
            (weight * background_bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        ).mean()
        losses.append(
            structure_loss(foreground, mask)
            + background_weight * weighted_background_bce
        )
    return torch.stack(losses).mean()
