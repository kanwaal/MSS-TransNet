from __future__ import annotations

import torch


def multiclass_summary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    """Return aggregate and per-class metrics from integer predictions."""
    predictions = predictions.detach().long().cpu().reshape(-1)
    targets = targets.detach().long().cpu().reshape(-1)
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError("Predictions and targets must be non-empty vectors with equal shape")

    class_count = len(class_names)
    if targets.min() < 0 or targets.max() >= class_count:
        raise ValueError("Target index is outside the configured class range")
    if predictions.min() < 0 or predictions.max() >= class_count:
        raise ValueError("Prediction index is outside the configured class range")

    encoded = targets * class_count + predictions
    confusion = torch.bincount(encoded, minlength=class_count**2).reshape(class_count, class_count)
    confusion_float = confusion.float()
    true_positive = confusion_float.diag()
    support = confusion_float.sum(dim=1)
    predicted_count = confusion_float.sum(dim=0)
    precision = true_positive / predicted_count.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)

    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(confusion[index].sum()),
        }
        for index, name in enumerate(class_names)
    }
    return {
        "samples": int(targets.numel()),
        "accuracy": float((predictions == targets).float().mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "class_names": list(class_names),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }
