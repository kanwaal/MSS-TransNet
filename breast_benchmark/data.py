from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


CLASS_NAMES = ("normal", "benign", "malignant")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
ULTRASOUND_NOISE_TYPES = (
    "speckle",
    "electronic",
    "clutter",
    "shadow",
    "reverberation",
)


class UltrasoundNoiseAugmentation:
    """Stochastic ultrasound corruptions with a normalized severity parameter."""

    def __init__(
        self,
        probability: float = 0.8,
        intensity_range: tuple[float, float] = (0.15, 0.85),
        secondary_probability: float = 0.35,
    ) -> None:
        minimum, maximum = intensity_range
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Noise probability must be between 0 and 1")
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError("Noise intensity range must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= secondary_probability <= 1.0:
            raise ValueError("Secondary noise probability must be between 0 and 1")
        self.probability = probability
        self.intensity_range = intensity_range
        self.secondary_probability = secondary_probability

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image
        count = 2 if random.random() < self.secondary_probability else 1
        noise_types = random.sample(ULTRASOUND_NOISE_TYPES, k=count)
        result = image
        for noise_type in noise_types:
            intensity = random.uniform(*self.intensity_range)
            result = self.apply_named(result, noise_type, intensity)
        return result

    @staticmethod
    def apply_named(
        image: torch.Tensor,
        noise_type: str,
        intensity: float,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Apply one corruption, optionally using a deterministic local RNG."""
        if noise_type not in ULTRASOUND_NOISE_TYPES:
            raise ValueError(
                f"Unknown ultrasound noise type {noise_type!r}; "
                f"choose from {ULTRASOUND_NOISE_TYPES}"
            )
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("Noise intensity must be between 0 and 1")
        if image.ndim != 3 or not image.is_floating_point():
            raise ValueError("Ultrasound noise expects a floating-point CHW image")

        rng = random if seed is None else random.Random(seed)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=image.device)
            generator.manual_seed(seed)

        channels, height, width = image.shape
        field_shape = (1, height, width)
        rand_kwargs = {
            "dtype": image.dtype,
            "device": image.device,
            "generator": generator,
        }

        if noise_type == "speckle":
            uniform = torch.rand(field_shape, **rand_kwargs).clamp_(1e-6, 1.0 - 1e-6)
            rayleigh = torch.sqrt(-2.0 * torch.log1p(-uniform))
            rayleigh_mean = math.sqrt(math.pi / 2.0)
            rayleigh_std = math.sqrt((4.0 - math.pi) / 2.0)
            standardized = (rayleigh - rayleigh_mean) / rayleigh_std
            sigma = 0.02 + 0.24 * intensity
            result = image * (1.0 + sigma * standardized)
        elif noise_type == "electronic":
            sigma = 0.004 + 0.055 * intensity
            result = image + sigma * torch.randn(field_shape, **rand_kwargs)
        elif noise_type == "clutter":
            field = torch.randn(field_shape, **rand_kwargs)
            kernel = max(5, 2 * int(round(min(height, width) * 0.035)) + 1)
            field = F.avg_pool2d(
                field.unsqueeze(0), kernel_size=kernel, stride=1, padding=kernel // 2
            ).squeeze(0)
            field = (field - field.mean()) / field.std().clamp_min(1e-6)
            depth_weight = torch.linspace(
                0.45, 1.0, height, dtype=image.dtype, device=image.device
            ).view(1, height, 1)
            amplitude = 0.01 + 0.075 * intensity
            result = image + amplitude * depth_weight * field
        elif noise_type == "shadow":
            center = rng.uniform(0.15, 0.85) * max(width - 1, 1)
            spread = rng.uniform(0.06, 0.18) * max(width, 1)
            start = rng.uniform(0.08, 0.52) * max(height - 1, 1)
            x = torch.arange(width, dtype=image.dtype, device=image.device)
            y = torch.arange(height, dtype=image.dtype, device=image.device)
            lateral = torch.exp(-0.5 * ((x - center) / max(spread, 1.0)) ** 2)
            depth = ((y - start) / max(height * 0.12, 1.0)).clamp(0.0, 1.0)
            shadow = depth[:, None] * lateral[None, :]
            attenuation = 0.10 + 0.75 * intensity
            result = image * (1.0 - attenuation * shadow.unsqueeze(0))
        else:  # reverberation
            first_echo = rng.uniform(0.08, 0.28) * max(height - 1, 1)
            spacing = rng.uniform(0.055, 0.13) * max(height, 1)
            line_width = rng.uniform(0.7, 1.8)
            y = torch.arange(height, dtype=image.dtype, device=image.device)
            echoes = torch.zeros(height, dtype=image.dtype, device=image.device)
            echo_index = 0
            position = first_echo
            while position < height:
                echoes += math.exp(-0.30 * echo_index) * torch.exp(
                    -0.5 * ((y - position) / line_width) ** 2
                )
                echo_index += 1
                position += spacing
            echoes = echoes.clamp_max(1.0).view(1, height, 1)
            amplitude = 0.015 + 0.18 * intensity
            result = image + amplitude * echoes * (1.0 - image)

        if channels > 1 and result.shape[0] == 1:
            result = result.expand(channels, -1, -1)
        return result.clamp(0.0, 1.0)


class BUSIDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: int = 224,
        augment: bool = False,
        ultrasound_noise: bool = False,
        noise_probability: float = 0.8,
        noise_intensity_range: tuple[float, float] = (0.15, 0.85),
        noise_secondary_probability: float = 0.35,
        class_names: tuple[str, ...] = CLASS_NAMES,
    ) -> None:
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            self.records = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.records:
            raise ValueError(f"No {split!r} records in {manifest}")
        self.class_names = class_names
        self.class_to_index = {
            name: class_index for class_index, name in enumerate(class_names)
        }
        unknown_classes = {
            row["class_name"] for row in self.records
        } - self.class_to_index.keys()
        if unknown_classes:
            raise ValueError(
                f"Manifest contains classes outside {class_names}: "
                f"{sorted(unknown_classes)}"
            )
        self.image_size = image_size
        self.augment = augment
        self.noise_augmenter = (
            UltrasoundNoiseAugmentation(
                probability=noise_probability,
                intensity_range=noise_intensity_range,
                secondary_probability=noise_secondary_probability,
            )
            if ultrasound_noise
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_union_mask(mask_paths: str) -> Image.Image:
        union: np.ndarray | None = None
        for path in mask_paths.split(";"):
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            union = mask if union is None else np.logical_or(union, mask)
        if union is None:
            raise ValueError("A BUSI sample must have at least one mask")
        return Image.fromarray((union.astype(np.uint8) * 255), mode="L")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        mask = self._load_union_mask(record["mask_paths"])

        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        mask_tensor = TF.pil_to_tensor(mask).float().div_(255.0)
        lesion_area_fraction = (mask_tensor > 0.5).float().mean()
        lesion_area_percent = lesion_area_fraction * 100.0
        image_tensor = TF.resize(
            image_tensor,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_tensor = TF.resize(
            mask_tensor,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if random.random() < 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)
            turns = random.randrange(4)
            if turns:
                image_tensor = torch.rot90(image_tensor, turns, dims=(-2, -1))
                mask_tensor = torch.rot90(mask_tensor, turns, dims=(-2, -1))

        if self.noise_augmenter is not None:
            image_tensor = self.noise_augmenter(image_tensor)

        return {
            "image": image_tensor.contiguous(),
            "mask": (mask_tensor > 0.5).float().contiguous(),
            "sample_id": record["sample_id"],
            "class_name": record["class_name"],
            "class_index": self.class_to_index[record["class_name"]],
            "lesion_area_fraction": lesion_area_fraction,
            "lesion_size_log_target": torch.log1p(lesion_area_percent),
        }


class BrEaSTUnlabeledDataset(Dataset):
    """Primary BrEaST ultrasound images with all annotation files excluded."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        augment: bool = True,
        reconstruction_input_noise: bool = False,
        noise_probability: float = 0.8,
        noise_intensity_range: tuple[float, float] = (0.15, 0.85),
        noise_secondary_probability: float = 0.35,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Missing BrEaST image directory {self.root}; "
                "run scripts/download_datasets.sh"
            )
        self.image_paths = tuple(
            path
            for path in sorted(self.root.glob("case*.png"))
            if len(path.stem) == 7 and path.stem[4:].isdigit()
        )
        if not self.image_paths:
            raise ValueError(f"No primary BrEaST case images found in {self.root}")
        self.image_size = image_size
        self.augment = augment
        self.noise_augmenter = (
            UltrasoundNoiseAugmentation(
                probability=noise_probability,
                intensity_range=noise_intensity_range,
                secondary_probability=noise_secondary_probability,
            )
            if reconstruction_input_noise
            else None
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        image_tensor = TF.resize(
            image_tensor,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.augment:
            if random.random() < 0.5:
                image_tensor = TF.hflip(image_tensor)
            if random.random() < 0.5:
                image_tensor = TF.vflip(image_tensor)
            turns = random.randrange(4)
            if turns:
                image_tensor = torch.rot90(image_tensor, turns, dims=(-2, -1))
        clean_image = image_tensor.contiguous()
        reconstruction_input = (
            self.noise_augmenter(clean_image)
            if self.noise_augmenter is not None
            else clean_image
        )
        sample = {
            "image": reconstruction_input.contiguous(),
            "sample_id": f"BrEaST/{image_path.stem}",
        }
        if self.noise_augmenter is not None:
            sample["reconstruction_target"] = clean_image
        return sample
