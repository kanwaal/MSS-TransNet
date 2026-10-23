from __future__ import annotations

import math

import torch


def decode_area_percent(log_prediction: torch.Tensor) -> torch.Tensor:
    """Convert log1p area-percentage predictions to the valid [0, 100] range."""

    maximum = math.log1p(100.0)
    return torch.expm1(log_prediction.float().clamp(0.0, maximum))


def tumor_size_summary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_targets: torch.Tensor,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    predictions = predictions.float().flatten()
    targets = targets.float().flatten()
    class_targets = class_targets.long().flatten()
    if not (len(predictions) == len(targets) == len(class_targets)):
        raise ValueError("Tumor-size predictions, targets, and classes must align")
    if not len(predictions):
        raise ValueError("Tumor-size summary requires at least one sample")

    errors = predictions - targets
    absolute_errors = errors.abs()
    target_variance = ((targets - targets.mean()) ** 2).sum()
    r_squared = (
        1.0 - float((errors**2).sum() / target_variance)
        if float(target_variance) > 0.0
        else float("nan")
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        selected = class_targets == class_index
        per_class[class_name] = {
            "mae_percentage_points": (
                float(absolute_errors[selected].mean())
                if bool(selected.any())
                else float("nan")
            ),
            "samples": int(selected.sum()),
        }
    lesions = targets > 0.0
    return {
        "target": "union-mask foreground area divided by original image area",
        "unit": "percentage points of image area",
        "physical_size": False,
        "samples": len(predictions),
        "mae_percentage_points": float(absolute_errors.mean()),
        "rmse_percentage_points": float(torch.sqrt((errors**2).mean())),
        "mean_bias_percentage_points": float(errors.mean()),
        "r_squared": r_squared,
        "lesion_only_mae_percentage_points": (
            float(absolute_errors[lesions].mean())
            if bool(lesions.any())
            else float("nan")
        ),
        "per_class": per_class,
    }
