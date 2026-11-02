import torch
import torch.nn.functional as F


def segmentation_loss(logits, mask):
    weights = 1+5*(F.avg_pool2d(mask,31,stride=1,padding=15)-mask).abs()
    bce = F.binary_cross_entropy_with_logits(logits,mask,reduction="none")
    bce = (weights*bce).sum((2,3))/weights.sum((2,3))
    probability = logits.sigmoid()
    intersection = (weights*probability*mask).sum((2,3))
    union = (weights*(probability+mask-probability*mask)).sum((2,3))
    return (bce+1-(intersection+1)/(union+1)).mean()


def reconstruction_loss(prediction, target, pixels):
    # Mean squared error over hidden-patch pixels only.
    error = (prediction.float()-target.float()).square()
    expanded = pixels.expand_as(error)
    if not expanded.any():
        raise ValueError("Masked reconstruction requires at least one hidden patch")
    return error.masked_select(expanded).mean()


def class_weights(labels, classes):
    counts = torch.bincount(torch.as_tensor(labels),minlength=classes).float()
    if (counts==0).any():
        raise ValueError("Every diagnostic class must occur in optimization training")
    return len(labels)/(classes*counts)


def decode_area(log_area):
    return torch.expm1(log_area.clamp(0.,torch.log(torch.tensor(101.)).item()))


def joint_losses(outputs, batch, weights, classification_weights, rec_loss=None):
    losses = {}
    if "segmentation" in weights:
        losses["segmentation"] = segmentation_loss(outputs["segmentation"],batch["mask"])
    if "classification" in weights:
        losses["classification"] = F.cross_entropy(outputs["classification"],batch["class_index"],
                                                    weight=classification_weights)
    if "area" in weights:
        losses["area"] = F.smooth_l1_loss(outputs["area"],batch["log_area"],beta=1.)
    if "reconstruction" in weights:
        losses["reconstruction"] = rec_loss
    return sum(weights[k]*v for k,v in losses.items()), losses
