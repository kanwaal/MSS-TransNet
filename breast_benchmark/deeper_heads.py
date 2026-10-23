"""Two-hidden-layer auxiliary heads; encoder and decoder paths are unchanged."""
import torch
from torch import nn


def make_deeper_head(in_features: int, out_features: int, *, seed: int, regression: bool) -> nn.Sequential:
    # New heads must not perturb initialization or RNG state outside the heads.
    # Seed the CPU generator only: torch.manual_seed also changes CUDA generators.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Dropout(0.2),
            nn.Linear(in_features, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, out_features),
        )
        if regression:
            # Preserve the baseline's zero initial log-area output.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
    return head


def description(out_features: int) -> str:
    return ("mean token pooling -> Dropout(0.2) -> Linear(768, 256) -> GELU -> "
            "Dropout(0.2) -> Linear(256, 128) -> GELU -> Dropout(0.2) -> "
            f"Linear(128, {out_features})")


def install_deeper_heads(model: nn.Module, *, classifier: bool, area: bool) -> nn.Module:
    if model.bottleneck_channels != 768 or not model.supports_tumor_size:
        raise ValueError("These heads require the 768-dimensional TransUNetMAESize bottleneck")
    seed = torch.initial_seed()
    if classifier:
        model.classifier = make_deeper_head(768, len(model.classification_classes), seed=seed + 1000, regression=False)
        model.classification_head_description = description(len(model.classification_classes))
    if area:
        model.tumor_size_regressor = make_deeper_head(768, 1, seed=seed + 2000, regression=True)
        model.tumor_size_head_description = description(1)
    return model
