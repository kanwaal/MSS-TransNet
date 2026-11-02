"""MSS-Trans-Net: shared CNN-transformer encoder with segmentation, classification,
relative-area and masked-reconstruction branches."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Encoder, CNN decoder and masked decoder build on TransUNet (Chen et al., 2024); see NOTICE.md.
from ._vendor.transunet.vit_seg_modeling import VisionTransformer, CONFIGS
from ._vendor.transunet2024.decoder2d import MultiScaleMaskedTransformerDecoder2d


def deep_head(outputs):
    # Pooled 768-d features -> 256 -> 128 (GELU and dropout) -> 64 -> task readout.
    return nn.Sequential(nn.Dropout(.2), nn.Linear(768, 256), nn.GELU(),
                         nn.Dropout(.2), nn.Linear(256, 128), nn.GELU(),
                         nn.Dropout(.2), nn.Linear(128, 64), nn.Linear(64, outputs))


class CNNPyramid(nn.Module):
    """CNN decoding with encoder skips, returning the multiscale feature maps."""
    def __init__(self, cup):
        super().__init__()
        self.cup = cup

    def forward(self, tokens, skips):
        batch, length, channels = tokens.shape
        side = math.isqrt(length)
        if side * side != length:
            raise ValueError("The full spatial token grid must be retained")
        x = self.cup.conv_more(tokens.transpose(1, 2).reshape(batch, channels, side, side))
        pyramid = []
        for i, block in enumerate(self.cup.blocks):
            x = block(x, skips[i] if i < self.cup.config.n_skip else None)
            pyramid.append(x)
        return pyramid


class CNNReconstruction(nn.Module):
    """CNN reconstruction decoder: 14 -> 28 -> 56 -> 112 -> 224 RGB in [0, 1]."""
    def __init__(self, channels=(256, 128, 64, 32)):
        super().__init__()
        blocks = []
        previous = 768
        for width in channels:
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(previous, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True)))
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.rgb = nn.Conv2d(previous, 3, 1)

    def forward(self, tokens):
        x = tokens.transpose(1, 2).reshape(tokens.shape[0], 768, 14, 14)
        return torch.sigmoid(self.rgb(self.blocks(x)))


class QueryRefinement(nn.Module):
    """Prediction-guided transformer refinement of the CNN decoder features.

    A learnable foreground query attends to three CNN decoder levels; each layer
    restricts cross-attention to the previously predicted foreground.
    """
    def __init__(self, width=192, heads=8, feedforward=1536, layers=3, queries=1,
                 layer_scale=True, attention_float32=True):
        super().__init__()
        if queries != 1:
            raise ValueError("The binary lesion mask uses one foreground query")
        if width % 32 or width % heads or layers < 1:
            raise ValueError("Use a width divisible by 32 and the number of attention heads")
        # Project the 28-, 56- and 112-pixel CNN decoder levels to the refinement width.
        self.memory_projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, width, 1), nn.GroupNorm(32, width))
            for c in (256,128,64)])
        self.pixel_projection = nn.Conv2d(16, width, 1)
        self.predictor = MultiScaleMaskedTransformerDecoder2d(
            in_channels=width, mask_classification=False, num_classes=1,
            hidden_dim=width, num_queries=1, nheads=heads,
            dim_feedforward=feedforward, dec_layers=layers, pre_norm=False,
            mask_dim=width, enforce_input_project=False, non_object=False,
            num_feature_levels=3, is_masking=True, is_masking_argmax=False,
            is_mhsa_float32=attention_float32, no_max_hw_pe=False,
            use_layer_scale=layer_scale)

    def forward(self, pyramid):
        levels = [project(feature) for project,feature in zip(self.memory_projections,pyramid[:3])]
        pixels = self.pixel_projection(pyramid[-1])
        # The final mask logits are supervised with the boundary-weighted structure loss.
        return self.predictor(levels,pixels)["pred_masks"]


class MSS_TransNet(nn.Module):
    def __init__(self, classes=3, pretrained_path=None,
                 refinement_width=192, refinement_heads=8, refinement_ffn=1536,
                 refinement_layers=3, foreground_queries=1,
                 refinement_layer_scale=True, refinement_attention_float32=True,
                 reconstruction_channels=(256, 128, 64, 32)):
        super().__init__()
        config = deepcopy(CONFIGS["R50-ViT-B_16"])
        config.n_classes, config.n_skip, config.patches.grid = 1, 3, (14, 14)
        base = VisionTransformer(config, img_size=224, num_classes=1)
        if pretrained_path is not None:
            path = Path(pretrained_path)
            if not path.is_file():
                raise FileNotFoundError(f"Supply the ImageNet hybrid initialization: {path}")
            with np.load(path) as weights:
                base.load_from(weights)
        self.encoder = base.transformer
        self.segmentation_cnn = CNNPyramid(base.decoder)
        self.segmentation = QueryRefinement(refinement_width, refinement_heads,
                                            refinement_ffn, refinement_layers, foreground_queries,
                                            refinement_layer_scale, refinement_attention_float32)
        self.reconstruction = CNNReconstruction(reconstruction_channels)
        self.classifier = deep_head(classes)
        self.area = deep_head(1)
        self.register_buffer("mean", torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std", torch.tensor([.229,.224,.225])[None,:,None,None])

    def encode(self, image):
        if image.shape[1:] != (3,224,224):
            raise ValueError("Expected RGB images with shape B x 3 x 224 x 224")
        tokens, _, skips = self.encoder((image-self.mean)/self.std)
        return tokens, skips

    def forward(self, image, tasks=None):
        tokens, skips = self.encode(image)
        tasks = {"segmentation","classification","area"} if tasks is None else set(tasks)
        result = {}
        if "segmentation" in tasks:
            pyramid = self.segmentation_cnn(tokens, skips)
            result["segmentation"] = self.segmentation(pyramid)
        pooled = tokens.mean(1)
        if "classification" in tasks:result["classification"] = self.classifier(pooled)
        if "area" in tasks:result["area"] = self.area(pooled).squeeze(-1)
        return result

    def reconstruct(self, image, mask_ratio=.30, generator=None):
        count = round(196*mask_ratio)
        if not 0 <= count < 196:
            raise ValueError("Masking must leave visible context")
        # An optional CPU generator gives fixed, reproducible validation masks.
        random_device = "cpu" if generator is not None else image.device
        noise = torch.rand(image.shape[0],196,device=random_device,generator=generator)
        indices = noise.argsort(1)[:,:count].to(image.device)
        patches = torch.zeros(image.shape[0],196,device=image.device,dtype=torch.bool)
        patches.scatter_(1,indices,True)
        pixels = patches.reshape(-1,1,14,14).repeat_interleave(16,2).repeat_interleave(16,3)
        # Hidden patches are set to black before ImageNet normalization.
        masked = image.masked_fill(pixels,0.)
        tokens, _ = self.encode(masked)  # all 196 tokens, including hidden locations
        return self.reconstruction(tokens), pixels

    def set_trainable(self, phase, active, freeze_encoder=False):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules = [self.encoder, self.reconstruction] if phase=="ssl" else [
            *([] if freeze_encoder else [self.encoder]),
            *([self.segmentation_cnn,self.segmentation] if "segmentation" in active else []),
            *([self.classifier] if "classification" in active else []),
            *([self.area] if "area" in active else []),
            *([self.reconstruction] if "reconstruction" in active else [])]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
