from __future__ import annotations


import importlib


import sys


from dataclasses import asdict, dataclass


from pathlib import Path


from typing import Callable


import numpy as np


import torch


import torch.nn as nn


from .losses import dual_supervised_loss


import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    exact_configuration: str
    sizes: tuple[int, ...]
    batch_size: int
    expected_parameters: int
    builder: Callable[[bool], nn.Module]

    def metadata(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("builder")
        return result


class ComparatorAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        normalize_imagenet: bool = False,
        reverse_outputs: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize_imagenet = normalize_imagenet
        self.reverse_outputs = reverse_outputs
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        if self.normalize_imagenet:
            image = (image - self.mean) / self.std
        outputs = self.model(image)
        if isinstance(outputs, torch.Tensor):
            result = [outputs]
        elif isinstance(outputs, (tuple, list)):
            result = list(outputs)
        else:
            raise TypeError(f"Unsupported model output type: {type(outputs)!r}")
        if self.reverse_outputs:
            result.reverse()
        return result


class BottleneckMultiTaskAdapter(ComparatorAdapter):
    """Segmentation adapter with a three-class head on an encoder bottleneck."""

    supports_classification = True
    supports_tumor_size = False
    classification_classes = ("normal", "benign", "malignant")

    def __init__(
        self,
        model: nn.Module,
        bottleneck_module: nn.Module,
        bottleneck_channels: int,
        bottleneck_name: str,
        *,
        feature_selector: Callable[[object], torch.Tensor] | None = None,
        normalize_imagenet: bool = False,
        reverse_outputs: bool = False,
        predict_tumor_size: bool = False,
    ) -> None:
        super().__init__(
            model,
            normalize_imagenet=normalize_imagenet,
            reverse_outputs=reverse_outputs,
        )
        self.bottleneck_channels = bottleneck_channels
        self.bottleneck_name = bottleneck_name
        self._feature_selector = feature_selector
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
            nn.Linear(bottleneck_channels, len(self.classification_classes)),
        )
        self.supports_tumor_size = predict_tumor_size
        if predict_tumor_size:
            self.tumor_size_regressor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(p=0.2),
                nn.Linear(bottleneck_channels, 1),
            )
            nn.init.zeros_(self.tumor_size_regressor[-1].weight)
            nn.init.zeros_(self.tumor_size_regressor[-1].bias)
        self._bottleneck: torch.Tensor | None = None
        self._bottleneck_hook = bottleneck_module.register_forward_hook(self._capture_bottleneck)

    def _capture_bottleneck(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: object,
    ) -> None:
        feature = self._feature_selector(output) if self._feature_selector else output
        if not isinstance(feature, torch.Tensor):
            raise TypeError(
                f"{self.bottleneck_name} produced {type(feature)!r}, expected a tensor"
            )
        self._bottleneck = feature

    def _classifier_input(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 4 and feature.shape[1] == self.bottleneck_channels:
            return feature
        if feature.ndim == 4 and feature.shape[-1] == self.bottleneck_channels:
            pooled = feature.mean(dim=(1, 2))
        elif feature.ndim == 3 and feature.shape[-1] == self.bottleneck_channels:
            pooled = feature.mean(dim=1)
        elif feature.ndim == 3 and feature.shape[1] == self.bottleneck_channels:
            pooled = feature.mean(dim=2)
        elif feature.ndim == 2 and feature.shape[1] == self.bottleneck_channels:
            pooled = feature
        else:
            raise ValueError(
                f"Cannot find {self.bottleneck_channels} bottleneck channels in "
                f"{self.bottleneck_name} output shape {tuple(feature.shape)}"
            )
        return pooled[:, :, None, None]

    def forward_multitask(
        self, image: torch.Tensor
    ) -> (
        tuple[list[torch.Tensor], torch.Tensor]
        | tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
    ):
        self._bottleneck = None
        segmentation_outputs = super().forward(image)
        bottleneck = self._bottleneck
        self._bottleneck = None
        if bottleneck is None:
            raise RuntimeError(
                f"{self.bottleneck_name} hook did not receive a bottleneck feature map"
            )
        head_input = self._classifier_input(bottleneck)
        class_logits = self.classifier(head_input)
        if self.supports_tumor_size:
            size_prediction = self.tumor_size_regressor(head_input).squeeze(1)
            return segmentation_outputs, class_logits, size_prediction
        return segmentation_outputs, class_logits


class TransUNetMAEAdapter(BottleneckMultiTaskAdapter):
    """TransUNet multitask adapter with a visible-token MAE decoder."""

    supports_reconstruction = True
    reconstruction_patch_size = 16
    reconstruction_grid_size = 14
    reconstruction_loss_name = "masked-patch RGB mean squared error"

    def __init__(
        self,
        model: nn.Module,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        predict_tumor_size: bool = False,
    ) -> None:
        super().__init__(
            model,
            model.transformer.encoder.encoder_norm,
            768,
            "TransUNet transformer encoder norm",
            normalize_imagenet=True,
            predict_tumor_size=predict_tumor_size,
        )
        patch_count = self.reconstruction_grid_size**2
        patch_values = 3 * self.reconstruction_patch_size**2
        self.encoder_to_decoder = nn.Linear(self.bottleneck_channels, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position_embeddings = nn.Parameter(
            torch.zeros(1, patch_count, decoder_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.reconstruction_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_depth,
            enable_nested_tensor=False,
        )
        self.reconstruction_norm = nn.LayerNorm(decoder_dim)
        self.reconstruction_head = nn.Linear(decoder_dim, patch_values)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_position_embeddings, std=0.02)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.zeros_(self.encoder_to_decoder.bias)
        nn.init.xavier_uniform_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    @property
    def reconstruction_patch_count(self) -> int:
        return self.reconstruction_grid_size**2

    def random_patch_mask(
        self,
        batch_size: int,
        mask_ratio: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("MAE mask ratio must be between zero and one")
        patch_count = self.reconstruction_patch_count
        keep_count = patch_count - int(round(patch_count * mask_ratio))
        noise = torch.rand(batch_size, patch_count, device=device)
        shuffled = noise.argsort(dim=1)
        visible_indices = shuffled[:, :keep_count]
        patch_mask = torch.ones(
            batch_size, patch_count, dtype=torch.bool, device=device
        )
        patch_mask.scatter_(1, visible_indices, False)
        return patch_mask, visible_indices

    def _pixel_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        batch_size = patch_mask.shape[0]
        pixel_mask = patch_mask.view(
            batch_size,
            self.reconstruction_grid_size,
            self.reconstruction_grid_size,
        )
        return pixel_mask.repeat_interleave(
            self.reconstruction_patch_size, dim=1
        ).repeat_interleave(self.reconstruction_patch_size, dim=2)[:, None]

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        patch_size = self.reconstruction_patch_size
        grid_size = self.reconstruction_grid_size
        expected_size = patch_size * grid_size
        if image.shape[1:] != (3, expected_size, expected_size):
            raise ValueError(
                f"MAE input must have shape NCHW with C=3 and H=W={expected_size}; "
                f"received {tuple(image.shape)}"
            )
        batch_size = image.shape[0]
        patches = image.reshape(
            batch_size, 3, grid_size, patch_size, grid_size, patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch_size, grid_size**2, 3 * patch_size**2)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        patch_size = self.reconstruction_patch_size
        grid_size = self.reconstruction_grid_size
        batch_size = patches.shape[0]
        expected_shape = (grid_size**2, 3 * patch_size**2)
        if patches.shape[1:] != expected_shape:
            raise ValueError(
                f"MAE patches must have trailing shape {expected_shape}; "
                f"received {tuple(patches.shape)}"
            )
        image = patches.reshape(
            batch_size, grid_size, grid_size, 3, patch_size, patch_size
        )
        image = image.permute(0, 3, 1, 4, 2, 5)
        return image.reshape(
            batch_size, 3, grid_size * patch_size, grid_size * patch_size
        )

    def forward_reconstruction(
        self,
        image: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_mask, visible_indices = self.random_patch_mask(
            image.shape[0], mask_ratio, image.device
        )
        normalized = (image - self.mean) / self.std
        normalized = normalized.masked_fill(self._pixel_mask(patch_mask), 0.0)
        embeddings, _ = self.model.transformer.embeddings(normalized)
        gather_indices = visible_indices.unsqueeze(-1).expand(
            -1, -1, embeddings.shape[-1]
        )
        visible_embeddings = embeddings.gather(1, gather_indices)
        encoded_visible, _ = self.model.transformer.encoder(visible_embeddings)
        self._bottleneck = None
        visible_tokens = self.encoder_to_decoder(encoded_visible)

        decoder_tokens = self.mask_token.to(dtype=visible_tokens.dtype).expand(
            image.shape[0], self.reconstruction_patch_count, -1
        )
        decoder_indices = visible_indices.unsqueeze(-1).expand(
            -1, -1, visible_tokens.shape[-1]
        )
        decoder_tokens = decoder_tokens.scatter(1, decoder_indices, visible_tokens)
        decoder_tokens = decoder_tokens + self.decoder_position_embeddings.to(
            dtype=decoder_tokens.dtype
        )
        decoded = self.reconstruction_decoder(decoder_tokens)
        predicted_patches = self.reconstruction_head(
            self.reconstruction_norm(decoded)
        )
        return predicted_patches, patch_mask

    def compute_reconstruction_loss(
        self,
        predicted_patches: torch.Tensor,
        target_image: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_patches = self.patchify(target_image)
        patch_losses = (predicted_patches - target_patches).square().mean(dim=-1)
        return patch_losses.masked_select(patch_mask).mean()


def _external_path(name: str, child: str | None = None) -> Path:
    path = ROOT / "external" / name
    if child:
        path /= child
    if not path.exists():
        raise FileNotFoundError(f"Missing bundled TransUNet source {path}; restore external/TransUNet from this repository")
    return path


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _create_transunet(pretrained: bool) -> nn.Module:
    root = _external_path("TransUNet")
    _prepend_import_path(root)
    module = importlib.import_module("networks.vit_seg_modeling")
    config = module.CONFIGS["R50-ViT-B_16"]
    config.n_classes = 1
    config.n_skip = 3
    config.patches.grid = (14, 14)
    model = module.VisionTransformer(config, img_size=224, num_classes=1)
    if pretrained:
        weight_path = ROOT / "weights" / "R50+ViT-B_16.npz"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Optional initialization missing: {weight_path}. Supply it separately or use --no-pretrained")
        model.load_from(np.load(weight_path))
    return model


def _build_transunet_mae_size(pretrained: bool) -> nn.Module:
    return TransUNetMAEAdapter(
        _create_transunet(pretrained), predict_tumor_size=True
    )


def _build_transunet_mae_deep_classifier(pretrained: bool) -> nn.Module:
    from .deeper_heads import install_deeper_heads
    return install_deeper_heads(_build_transunet_mae_size(pretrained), classifier=True, area=False)


def _build_transunet_mae_deep_area(pretrained: bool) -> nn.Module:
    from .deeper_heads import install_deeper_heads
    return install_deeper_heads(_build_transunet_mae_size(pretrained), classifier=False, area=True)


def _build_transunet_mae_deep_heads(pretrained: bool) -> nn.Module:
    from .deeper_heads import install_deeper_heads
    return install_deeper_heads(_build_transunet_mae_size(pretrained), classifier=True, area=True)


MODEL_SPECS = (
    ModelSpec("TransUNetMAESize", "TransUNet MAE with tumor type and size multitask heads", "R50-ViT-B/16 with segmentation, three-class transformer head, lesion-area regression head, and masked-patch reconstruction decoder", (224,), 24, 108_883_221, _build_transunet_mae_size),
    ModelSpec("TransUNetMAESizeDeepClassifier", "TransUNet MAE with deeper classifier", "Original TransUNetMAESize with classifier MLP 768-256-128-3; original linear area head", (224,), 24, 109_111_061, _build_transunet_mae_deep_classifier),
    ModelSpec("TransUNetMAESizeDeepArea", "TransUNet MAE with deeper area head", "Original TransUNetMAESize with area MLP 768-256-128-1; original linear classifier", (224,), 24, 109_112_341, _build_transunet_mae_deep_area),
    ModelSpec("TransUNetMAESizeDeepHeads", "TransUNet MAE with deeper classifier and area heads", "Original TransUNetMAESize with independent classifier MLP 768-256-128-3 and area MLP 768-256-128-1", (224,), 24, 109_340_181, _build_transunet_mae_deep_heads),
)


MODEL_REGISTRY = {spec.key: spec for spec in MODEL_SPECS}


def build_model(key: str, pretrained: bool = True) -> tuple[ComparatorAdapter, ModelSpec]:
    try:
        spec = MODEL_REGISTRY[key]
    except KeyError as error:
        raise KeyError(f"Unknown model {key!r}; choose from {', '.join(MODEL_REGISTRY)}") from error
    model = spec.builder(pretrained)
    return model, spec


def configure_classification_classes(
    model: nn.Module,
    class_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Resize a multitask classifier and retain weights for shared class names."""

    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Classification classes must be non-empty and unique")
    if not bool(getattr(model, "supports_classification", False)):
        return ()
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Sequential) or not isinstance(
        classifier[-1], nn.Linear
    ):
        raise TypeError("Expected the multitask classifier to end in nn.Linear")

    previous_names = tuple(getattr(model, "classification_classes", ()))
    previous_head = classifier[-1]
    if previous_head.out_features == len(class_names) and previous_names == class_names:
        return class_names

    replacement = nn.Linear(
        previous_head.in_features,
        len(class_names),
        bias=previous_head.bias is not None,
        device=previous_head.weight.device,
        dtype=previous_head.weight.dtype,
    )
    transferred: list[str] = []
    with torch.no_grad():
        for new_index, class_name in enumerate(class_names):
            if class_name not in previous_names:
                continue
            previous_index = previous_names.index(class_name)
            replacement.weight[new_index].copy_(previous_head.weight[previous_index])
            if replacement.bias is not None and previous_head.bias is not None:
                replacement.bias[new_index].copy_(previous_head.bias[previous_index])
            transferred.append(class_name)
    classifier[-1] = replacement
    model.classification_classes = class_names
    return tuple(transferred)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
