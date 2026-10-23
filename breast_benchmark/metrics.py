from __future__ import annotations

import torch


def binary_metrics(logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:
    prediction = torch.sigmoid(logits) >= threshold
    target = mask >= 0.5
    dims = (1, 2, 3)
    intersection = (prediction & target).sum(dim=dims).float()
    pred_sum = prediction.sum(dim=dims).float()
    target_sum = target.sum(dim=dims).float()
    union = pred_sum + target_sum - intersection
    true_negative = ((~prediction) & (~target)).sum(dim=dims).float()
    false_positive = (prediction & (~target)).sum(dim=dims).float()
    false_negative = ((~prediction) & target).sum(dim=dims).float()

    both_empty = (pred_sum == 0) & (target_sum == 0)
    dice = torch.where(both_empty, 1.0, (2.0 * intersection) / (pred_sum + target_sum).clamp_min(1.0))
    iou = torch.where(both_empty, 1.0, intersection / union.clamp_min(1.0))
    precision = torch.where(pred_sum == 0, (target_sum == 0).float(), intersection / pred_sum)
    recall = torch.where(target_sum == 0, (pred_sum == 0).float(), intersection / target_sum)
    specificity = true_negative / (true_negative + false_positive).clamp_min(1.0)
    accuracy = (intersection + true_negative) / (
        intersection + true_negative + false_positive + false_negative
    ).clamp_min(1.0)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
    }
