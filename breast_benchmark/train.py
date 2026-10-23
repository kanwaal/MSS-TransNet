from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .classification import multiclass_summary
from .data import (
    BUSIDataset,
    BrEaSTUnlabeledDataset,
    CLASS_NAMES,
    ULTRASOUND_NOISE_TYPES,
)
from .losses import supervised_loss
from .metrics import binary_metrics
from .models import (
    MODEL_REGISTRY,
    build_model,
    configure_classification_classes,
    trainable_parameter_count,
)
from .size_regression import decode_area_percent, tumor_size_summary


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(paths: tuple[Path, ...], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def training_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    classification_enabled: bool,
    activation_checkpointing: bool,
    tumor_size_enabled: bool = False,
) -> (
    tuple[list[torch.Tensor], torch.Tensor | None]
    | tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
):
    """Run a training forward, optionally recomputing activations in backward."""

    if classification_enabled:
        forward = model.forward_multitask
    else:
        forward = model
    if activation_checkpointing:
        result = checkpoint(forward, image, use_reentrant=False)
    else:
        result = forward(image)
    if classification_enabled:
        if tumor_size_enabled:
            outputs, class_logits, size_prediction = result
            return outputs, class_logits, size_prediction
        outputs, class_logits = result
        return outputs, class_logits
    return result, None


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "segmentation_loss",
                "classification_loss",
                "tumor_size_loss",
                "reconstruction_loss",
                "classification_accuracy",
                "learning_rate",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, float | int]],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_train_loss": best_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def update_early_stopping(
    current_loss: float,
    best_loss: float,
    epochs_without_improvement: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    improved = current_loss < best_loss - min_delta
    if improved:
        return current_loss, 0, True
    return best_loss, epochs_without_improvement + 1, False


def restore_rng(payload: dict[str, Any]) -> None:
    rng = payload.get("rng")
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    # Checkpoints are loaded onto the training device, but the default RNG
    # generator always requires a CPU byte tensor.
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])


def train_one(args: argparse.Namespace) -> Path:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.early_stopping_patience is not None and args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta cannot be negative")
    if not 0.0 <= args.noise_probability <= 1.0:
        raise ValueError("--noise-probability must be between 0 and 1")
    if not 0.0 <= args.noise_min_intensity <= args.noise_max_intensity <= 1.0:
        raise ValueError(
            "Noise intensity range must satisfy 0 <= min intensity <= max intensity <= 1"
        )
    if not 0.0 <= args.noise_secondary_probability <= 1.0:
        raise ValueError("--noise-secondary-probability must be between 0 and 1")
    if not 0.0 < args.mae_mask_ratio < 1.0:
        raise ValueError("--mae-mask-ratio must be between zero and one")
    if args.reconstruction_loss_weight <= 0.0:
        raise ValueError("--reconstruction-loss-weight must be positive")
    if args.segmentation_loss_weight <= 0.0:
        raise ValueError("--segmentation-loss-weight must be positive")
    if args.tumor_size_loss_weight <= 0.0:
        raise ValueError("--tumor-size-loss-weight must be positive")
    if args.unlabeled_batch_size is not None and args.unlabeled_batch_size <= 0:
        raise ValueError("--unlabeled-batch-size must be positive")
    if args.microbatch_size is not None and args.microbatch_size <= 0:
        raise ValueError("--microbatch-size must be positive")
    seed_everything(args.seed)
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; run python -m breast_benchmark.prepare")
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_records = list(csv.DictReader(handle))
    if not manifest_records:
        raise ValueError(f"Manifest is empty: {manifest}")
    present_classes = {row["class_name"] for row in manifest_records}
    class_names = tuple(name for name in CLASS_NAMES if name in present_classes) + tuple(
        sorted(present_classes - set(CLASS_NAMES))
    )
    dataset_names = {row.get("dataset", "").strip() for row in manifest_records}
    dataset_names.discard("")
    if len(dataset_names) > 1:
        raise ValueError(f"Manifest mixes datasets: {sorted(dataset_names)}")
    dataset_name = next(iter(dataset_names), "BUSI")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, spec = build_model(args.model, pretrained=args.pretrained)
    initial_checkpoint_metadata = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = args.initial_checkpoint.resolve()
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(f"Missing initial checkpoint {initial_checkpoint}")
        initial_payload = torch.load(
            initial_checkpoint, map_location="cpu", weights_only=False
        )
        initial_state = (
            initial_payload["model"]
            if isinstance(initial_payload, dict) and "model" in initial_payload
            else initial_payload
        )
        model.load_state_dict(initial_state)
        initial_checkpoint_metadata = {
            "path": str(initial_checkpoint),
            "sha256": file_sha256(initial_checkpoint),
            "source_epoch": (
                int(initial_payload["epoch"]) + 1
                if isinstance(initial_payload, dict) and "epoch" in initial_payload
                else None
            ),
            "source_best_train_loss": (
                float(initial_payload["best_train_loss"])
                if isinstance(initial_payload, dict)
                and "best_train_loss" in initial_payload
                else None
            ),
            "model_parameters_loaded": True,
            "optimizer_and_scheduler_loaded": False,
        }
    transferred_class_rows = configure_classification_classes(model, class_names)
    if initial_checkpoint_metadata is not None:
        initial_checkpoint_metadata["classification_head_reconfigured"] = {
            "classes": list(class_names),
            "transferred_rows": list(transferred_class_rows),
            "new_output_features": len(class_names),
        }
    model.to(device)
    parameter_count = trainable_parameter_count(model)
    classification_enabled = bool(getattr(model, "supports_classification", False))
    tumor_size_enabled = bool(getattr(model, "supports_tumor_size", False))
    reconstruction_enabled = bool(getattr(model, "supports_reconstruction", False))
    if tumor_size_enabled and not classification_enabled:
        raise ValueError("Tumor-size regression requires the multitask forward path")
    if reconstruction_enabled and spec.sizes != (224,):
        raise ValueError("MAE reconstruction currently requires a single 224-pixel input")
    if args.reconstruction_input_noise and not reconstruction_enabled:
        raise ValueError(
            "--reconstruction-input-noise requires a model with a reconstruction branch"
        )
    run_key = f"{spec.key}MultiTask" if classification_enabled else spec.key
    if reconstruction_enabled:
        run_key += "SemiSupervised"
    if args.ultrasound_noise:
        run_key += "NoiseAug"
    if args.reconstruction_input_noise:
        run_key += "Denoising"
    dataset_slug = dataset_name.replace(" ", "_").replace("/", "-")
    output_dir = args.output_root / f"{dataset_slug}_{run_key}_seed{args.seed}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "COMPLETE"
    if args.resume and complete_path.is_file():
        print(f"Skipping completed run {output_dir}")
        return output_dir

    batch_size = args.batch_size or spec.batch_size
    microbatch_size = args.microbatch_size or batch_size
    if microbatch_size > batch_size:
        raise ValueError("--microbatch-size cannot exceed --batch-size")
    execution_notes: list[str] = []
    if microbatch_size < batch_size:
        execution_notes.append(
            f"Each effective batch of {batch_size} is split into microbatches of "
            f"at most {microbatch_size}; weighted gradients are accumulated before "
            "one optimizer step."
        )
    if args.activation_checkpointing:
        execution_notes.append(
            "Training-only non-reentrant activation checkpointing reduces peak GPU "
            "memory by recomputing the multitask forward during backward."
        )
    if args.reconstruction_input_noise:
        execution_notes.append(
            "The MAE encoder receives stochastically corrupted BrEaST inputs while "
            "masked-patch MSE uses the geometrically matched clean image as its target."
        )
    scan_backend = getattr(model, "scan_backend", None)
    if scan_backend == "pytorch_chunked":
        execution_notes.append(
            "Mamba CUDA selective-scan is unavailable; the differentiable PyTorch scan "
            "fallback runs with upstream VSSM block checkpointing."
        )
    elif scan_backend == "triton_recurrence":
        execution_notes.append(
            "Mamba CUDA selective-scan is unavailable; an output- and gradient-checked "
            "Triton recurrence kernel runs with upstream VSSM block checkpointing."
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = BUSIDataset(
        manifest,
        "train",
        image_size=224,
        augment=True,
        ultrasound_noise=args.ultrasound_noise,
        noise_probability=args.noise_probability,
        noise_intensity_range=(args.noise_min_intensity, args.noise_max_intensity),
        noise_secondary_probability=args.noise_secondary_probability,
        class_names=class_names,
    )
    test_dataset = BUSIDataset(
        manifest,
        "test",
        image_size=spec.sizes[0] if len(spec.sizes) == 1 else 224,
        class_names=class_names,
    )
    unlabeled_dataset = (
        BrEaSTUnlabeledDataset(
            args.unlabeled_root,
            image_size=224,
            augment=True,
            reconstruction_input_noise=args.reconstruction_input_noise,
            noise_probability=args.noise_probability,
            noise_intensity_range=(
                args.noise_min_intensity,
                args.noise_max_intensity,
            ),
            noise_secondary_probability=args.noise_secondary_probability,
        )
        if reconstruction_enabled
        else None
    )
    classification_class_weights = None
    if classification_enabled:
        class_counts = torch.bincount(
            torch.tensor(
                [train_dataset.class_to_index[row["class_name"]] for row in train_dataset.records],
                dtype=torch.long,
            ),
            minlength=len(class_names),
        ).float()
        classification_class_weights = (
            len(train_dataset) / (len(class_names) * class_counts)
        ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size, args.test_batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    unlabeled_batch_size = args.unlabeled_batch_size or batch_size
    unlabeled_loader = (
        DataLoader(
            unlabeled_dataset,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + 1),
            drop_last=False,
        )
        if unlabeled_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    tasks = ["segmentation"]
    loss_components = ["segmentation"]
    if classification_enabled:
        tasks.append("tumor_type_classification")
        loss_components.append("tumor type classification")
    if tumor_size_enabled:
        tasks.append("tumor_size_regression")
        loss_components.append("tumor size regression")
    if reconstruction_enabled:
        tasks.append("masked_image_reconstruction")
        loss_components.append("reconstruction")
    combined_loss_name = ", ".join(loss_components)
    task_loss_weights: dict[str, float] = {
        "segmentation": args.segmentation_loss_weight
    }
    if classification_enabled:
        task_loss_weights["tumor_type_classification"] = (
            args.classification_loss_weight
        )
    if tumor_size_enabled:
        task_loss_weights["tumor_size_regression"] = args.tumor_size_loss_weight
    if reconstruction_enabled:
        task_loss_weights["masked_image_reconstruction"] = (
            args.reconstruction_loss_weight
        )

    config = {
        "dataset": dataset_name,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "partition": {
            "train": len(train_dataset),
            "test": len(test_dataset),
            "train_groups": len(
                {row.get("group_id", row["sample_id"]) for row in train_dataset.records}
            ),
            "test_groups": len(
                {row.get("group_id", row["sample_id"]) for row in test_dataset.records}
            ),
            "seed": args.seed,
        },
        "run_key": run_key,
        "model": spec.metadata(),
        "actual_trainable_parameters": parameter_count,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip,
        "segmentation_loss": getattr(
            model,
            "segmentation_loss_name",
            "mean deep-supervision boundary-weighted BCE plus weighted IoU",
        ),
        "scheduler": "CosineAnnealingLR",
        "epochs": args.epochs,
        "early_stopping": (
            {
                "monitor": f"combined {combined_loss_name} training loss",
                "mode": "min",
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "test_set_used": False,
            }
            if args.early_stopping_patience is not None
            else None
        ),
        "batch_size": batch_size,
        "gradient_accumulation": {
            "effective_batch_size": batch_size,
            "microbatch_size": microbatch_size,
            "maximum_accumulation_steps": (batch_size + microbatch_size - 1)
            // microbatch_size,
        },
        "activation_checkpointing": {
            "enabled": args.activation_checkpointing,
            "training_only": True,
            "implementation": "torch.utils.checkpoint.checkpoint(use_reentrant=False)",
        },
        "scan_backend": scan_backend,
        "execution_notes": execution_notes,
        "amp": amp_enabled,
        "pretrained_initialization": args.pretrained,
        "initial_checkpoint": initial_checkpoint_metadata,
        "ultrasound_noise_augmentation": {
            "enabled": args.ultrasound_noise,
            "training_only": True,
            "types": list(ULTRASOUND_NOISE_TYPES),
            "application_probability": args.noise_probability,
            "secondary_noise_probability": args.noise_secondary_probability,
            "minimum_intensity": args.noise_min_intensity,
            "maximum_intensity": args.noise_max_intensity,
            "test_set_corruption": None,
        },
        "tasks": tasks,
        "task_loss_weights": task_loss_weights,
        "classification": (
            {
                "classes": list(class_names),
                "head": getattr(model, "classification_head_description", (
                    "global average pooling -> Dropout(0.2) -> "
                    f"Linear({model.bottleneck_channels}, {len(class_names)})"
                )),
                "attachment": model.bottleneck_name,
                "loss": "inverse-frequency weighted cross-entropy",
                "loss_weight": args.classification_loss_weight,
                "class_weights": classification_class_weights.detach().cpu().tolist(),
            }
            if classification_enabled
            else None
        ),
        "tumor_size_regression": (
            {
                "enabled": True,
                "target": "log1p(100 * union-mask foreground pixels / image pixels)",
                "reported_unit": "percentage points of original image area",
                "physical_size_claimed": False,
                "physical_size_limitation": (
                    f"{dataset_name} does not provide reliable pixel spacing; this is a "
                    "two-dimensional relative lesion-area proxy, not a centimeter value"
                ),
                "head": getattr(model, "tumor_size_head_description", (
                    "global average pooling -> Dropout(0.2) -> "
                    f"Linear({model.bottleneck_channels}, 1)"
                )),
                "attachment": model.bottleneck_name,
                "loss": "Smooth L1 on log1p lesion-area percentage",
                "loss_weight": args.tumor_size_loss_weight,
            }
            if tumor_size_enabled
            else None
        ),
        "semi_supervised_reconstruction": (
            {
                "enabled": True,
                "unlabeled_dataset": "BrEaST / Breast-Lesions-USG",
                "unlabeled_root": str(unlabeled_dataset.root),
                "unlabeled_images": len(unlabeled_dataset),
                "unlabeled_images_sha256": dataset_sha256(
                    unlabeled_dataset.image_paths, unlabeled_dataset.root
                ),
                "included_files": "exact caseNNN.png primary ultrasound images",
                "excluded_annotations": "all *_tumor.png and *_otherN.png files",
                "labels_or_masks_used": False,
                "license": "CC BY 4.0",
                "batch_size": unlabeled_batch_size,
                "patch_size": model.reconstruction_patch_size,
                "mask_ratio": args.mae_mask_ratio,
                "loss": model.reconstruction_loss_name,
                "loss_weight": args.reconstruction_loss_weight,
                "input_corruption": {
                    "enabled": args.reconstruction_input_noise,
                    "training_only": True,
                    "types": list(ULTRASOUND_NOISE_TYPES),
                    "application_probability": args.noise_probability,
                    "secondary_noise_probability": args.noise_secondary_probability,
                    "minimum_intensity": args.noise_min_intensity,
                    "maximum_intensity": args.noise_max_intensity,
                },
                "reconstruction_target": (
                    "clean geometrically matched BrEaST RGB patches before input noise"
                    if args.reconstruction_input_noise
                    else "clean geometrically augmented BrEaST RGB patches"
                ),
                "decoder": (
                    "encoder projection -> four-layer transformer decoder -> "
                    "16x16 RGB patch prediction"
                ),
            }
            if reconstruction_enabled
            else None
        ),
        "checkpoint_selection": (
            f"lowest combined {combined_loss_name} training loss; "
            "test set is never used for selection"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "command": " ".join(sys.argv),
    }
    save_json(output_dir / "config.json", config)

    start_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    last_path = checkpoint_dir / "last_model.pt"
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload["best_train_loss"])
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        history = payload["history"]
        restore_rng(payload)
        print(f"Resuming {spec.key} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        segmentation_loss_sum = 0.0
        classification_loss_sum = 0.0
        tumor_size_loss_sum = 0.0
        reconstruction_loss_sum = 0.0
        classification_correct = 0
        classification_examples = 0
        updates = 0
        unlabeled_iterator = iter(unlabeled_loader) if unlabeled_loader is not None else None
        progress = tqdm(train_loader, desc=f"{spec.key} {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            base_image = batch["image"].to(device, non_blocking=True)
            base_mask = batch["mask"].to(device, non_blocking=True)
            class_target = batch["class_index"].to(device, non_blocking=True)
            tumor_size_target = batch["lesion_size_log_target"].to(
                device, non_blocking=True
            )
            unlabeled_image = None
            unlabeled_reconstruction_target = None
            if unlabeled_iterator is not None:
                try:
                    unlabeled_batch = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iterator)
                unlabeled_image = unlabeled_batch["image"].to(
                    device, non_blocking=True
                )
                unlabeled_reconstruction_target = unlabeled_batch.get(
                    "reconstruction_target", unlabeled_batch["image"]
                ).to(device, non_blocking=True)
            for size in spec.sizes:
                image = F.interpolate(base_image, (size, size), mode="bilinear", align_corners=False)
                mask = F.interpolate(base_mask, (size, size), mode="nearest")
                optimizer.zero_grad(set_to_none=True)
                logical_batch_size = image.shape[0]
                segmentation_loss_value = 0.0
                classification_loss_value = 0.0
                tumor_size_loss_value = 0.0
                classification_denominator = (
                    classification_class_weights[class_target].sum()
                    if classification_enabled
                    else None
                )
                for start in range(0, logical_batch_size, microbatch_size):
                    stop = min(start + microbatch_size, logical_batch_size)
                    image_chunk = image[start:stop]
                    mask_chunk = mask[start:stop]
                    class_chunk = class_target[start:stop]
                    tumor_size_chunk = tumor_size_target[start:stop]
                    sample_fraction = (stop - start) / logical_batch_size
                    with autocast_context(device, amp_enabled):
                        forward_result = training_forward(
                            model,
                            image_chunk,
                            classification_enabled,
                            args.activation_checkpointing,
                            tumor_size_enabled,
                        )
                        if tumor_size_enabled:
                            outputs, class_logits, tumor_size_prediction = forward_result
                        else:
                            outputs, class_logits = forward_result
                            tumor_size_prediction = None
                        segmentation_loss_function = getattr(
                            model, "compute_segmentation_loss", supervised_loss
                        )
                        segmentation_loss = segmentation_loss_function(
                            outputs, mask_chunk
                        )
                        scaled_segmentation_loss = segmentation_loss * sample_fraction
                        if class_logits is not None:
                            classification_loss = F.cross_entropy(
                                class_logits,
                                class_chunk,
                                weight=classification_class_weights,
                                reduction="sum",
                            ) / classification_denominator
                            loss = (
                                args.segmentation_loss_weight
                                * scaled_segmentation_loss
                                + args.classification_loss_weight
                                * classification_loss
                            )
                        else:
                            classification_loss = None
                            loss = (
                                args.segmentation_loss_weight
                                * scaled_segmentation_loss
                            )
                        if tumor_size_prediction is not None:
                            tumor_size_loss = F.smooth_l1_loss(
                                tumor_size_prediction.float(),
                                tumor_size_chunk.float(),
                                reduction="sum",
                            ) / logical_batch_size
                            loss = (
                                loss
                                + args.tumor_size_loss_weight * tumor_size_loss
                            )
                        else:
                            tumor_size_loss = None
                    scaler.scale(loss).backward()
                    segmentation_loss_value += sample_fraction * float(
                        segmentation_loss.detach()
                    )
                    if classification_loss is not None and class_logits is not None:
                        classification_loss_value += float(classification_loss.detach())
                        classification_correct += int(
                            (class_logits.detach().argmax(dim=1) == class_chunk).sum()
                        )
                        classification_examples += int(class_chunk.numel())
                    if tumor_size_loss is not None:
                        tumor_size_loss_value += float(tumor_size_loss.detach())

                reconstruction_loss_value = 0.0
                if unlabeled_image is not None:
                    unlabeled_examples = unlabeled_image.shape[0]
                    for start in range(0, unlabeled_examples, microbatch_size):
                        stop = min(start + microbatch_size, unlabeled_examples)
                        unlabeled_chunk = unlabeled_image[start:stop]
                        reconstruction_target_chunk = (
                            unlabeled_reconstruction_target[start:stop]
                        )
                        sample_fraction = (stop - start) / unlabeled_examples
                        with autocast_context(device, amp_enabled):
                            predicted_patches, patch_mask = model.forward_reconstruction(
                                unlabeled_chunk, args.mae_mask_ratio
                            )
                            reconstruction_loss = model.compute_reconstruction_loss(
                                predicted_patches,
                                reconstruction_target_chunk,
                                patch_mask,
                            )
                        scaler.scale(
                            args.reconstruction_loss_weight
                            * sample_fraction
                            * reconstruction_loss
                        ).backward()
                        reconstruction_loss_value += sample_fraction * float(
                            reconstruction_loss.detach()
                        )
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                combined_loss = (
                    args.segmentation_loss_weight * segmentation_loss_value
                )
                if classification_enabled:
                    combined_loss += (
                        args.classification_loss_weight * classification_loss_value
                    )
                if tumor_size_enabled:
                    combined_loss += (
                        args.tumor_size_loss_weight * tumor_size_loss_value
                    )
                if unlabeled_image is not None:
                    combined_loss += (
                        args.reconstruction_loss_weight * reconstruction_loss_value
                    )
                    reconstruction_loss_sum += reconstruction_loss_value
                loss_sum += combined_loss
                segmentation_loss_sum += segmentation_loss_value
                classification_loss_sum += classification_loss_value
                tumor_size_loss_sum += tumor_size_loss_value
                updates += 1
            postfix = {"loss": f"{loss_sum / updates:.4f}"}
            if classification_examples:
                postfix["class_acc"] = f"{classification_correct / classification_examples:.3f}"
            if tumor_size_enabled:
                postfix["size"] = f"{tumor_size_loss_sum / updates:.4f}"
            if reconstruction_enabled:
                postfix["recon"] = f"{reconstruction_loss_sum / updates:.4f}"
            progress.set_postfix(postfix)
        scheduler.step()
        epoch_loss = loss_sum / max(updates, 1)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "segmentation_loss": segmentation_loss_sum / max(updates, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        if classification_enabled:
            row["classification_loss"] = classification_loss_sum / max(updates, 1)
            row["classification_accuracy"] = classification_correct / max(
                classification_examples, 1
            )
        if tumor_size_enabled:
            row["tumor_size_loss"] = tumor_size_loss_sum / max(updates, 1)
        if reconstruction_enabled:
            row["reconstruction_loss"] = reconstruction_loss_sum / max(updates, 1)
        history.append(row)
        best_loss, epochs_without_improvement, improved = update_early_stopping(
            epoch_loss,
            best_loss,
            epochs_without_improvement,
            args.early_stopping_min_delta,
        )
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            epochs_without_improvement,
            history,
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, checkpoint_dir / "best_model.pt")
        save_history(output_dir / "history.csv", history)
        print(f"{spec.key} epoch {epoch + 1}: loss={epoch_loss:.6f}, seconds={row['seconds']:.1f}")
        if (
            args.early_stopping_patience is not None
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping {spec.key} at epoch {epoch + 1}: "
                f"no improvement for {epochs_without_improvement} epochs"
            )
            break

    config["completed_epochs"] = len(history)
    config["best_train_loss"] = best_loss
    config["stopped_early"] = len(history) < args.epochs
    save_json(output_dir / "config.json", config)

    best = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    evaluate(model, test_loader, device, amp_enabled, output_dir)
    complete_path.write_text("completed\n", encoding="ascii")
    return output_dir


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    output_dir: Path,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, str | float]] = []
    metric_values: dict[str, list[float]] = {}
    classification_enabled = bool(getattr(model, "supports_classification", False))
    tumor_size_enabled = bool(getattr(model, "supports_tumor_size", False))
    class_names = tuple(getattr(model, "classification_classes", CLASS_NAMES))
    classification_rows: list[dict[str, str | float | int]] = []
    classification_predictions: list[torch.Tensor] = []
    classification_targets: list[torch.Tensor] = []
    tumor_size_rows: list[dict[str, str | float]] = []
    tumor_size_predictions: list[torch.Tensor] = []
    tumor_size_targets: list[torch.Tensor] = []
    tumor_size_classes: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="test", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            if classification_enabled:
                multitask_outputs = model.forward_multitask(image)
                if tumor_size_enabled:
                    segmentation_outputs, class_logits, tumor_size_log_prediction = (
                        multitask_outputs
                    )
                else:
                    segmentation_outputs, class_logits = multitask_outputs
                    tumor_size_log_prediction = None
            else:
                segmentation_outputs = model(image)
                class_logits = None
                tumor_size_log_prediction = None
            logits = segmentation_outputs[0]
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, mask.shape[-2:], mode="bilinear", align_corners=False)
        metrics = binary_metrics(logits.float(), mask)
        for index, sample_id in enumerate(batch["sample_id"]):
            row: dict[str, str | float] = {
                "sample_id": sample_id,
                "class_name": batch["class_name"][index],
            }
            for name, values in metrics.items():
                value = float(values[index].cpu())
                row[name] = value
                metric_values.setdefault(name, []).append(value)
            rows.append(row)

        if class_logits is not None:
            class_probabilities = torch.softmax(class_logits.float(), dim=1).cpu()
            class_predictions = class_probabilities.argmax(dim=1)
            class_targets = batch["class_index"].long()
            classification_predictions.append(class_predictions)
            classification_targets.append(class_targets)
            for index, sample_id in enumerate(batch["sample_id"]):
                target_index = int(class_targets[index])
                prediction_index = int(class_predictions[index])
                classification_row: dict[str, str | float | int] = {
                    "sample_id": sample_id,
                    "true_class": class_names[target_index],
                    "predicted_class": class_names[prediction_index],
                    "correct": int(target_index == prediction_index),
                }
                for class_index, class_name in enumerate(class_names):
                    classification_row[f"probability_{class_name}"] = float(
                        class_probabilities[index, class_index]
                    )
                classification_rows.append(classification_row)

        if tumor_size_log_prediction is not None:
            size_predictions = decode_area_percent(
                tumor_size_log_prediction
            ).cpu()
            size_targets = batch["lesion_area_fraction"].float() * 100.0
            size_classes = batch["class_index"].long()
            tumor_size_predictions.append(size_predictions)
            tumor_size_targets.append(size_targets)
            tumor_size_classes.append(size_classes)
            for index, sample_id in enumerate(batch["sample_id"]):
                target = float(size_targets[index])
                prediction = float(size_predictions[index])
                tumor_size_rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": batch["class_name"][index],
                        "target_area_percent": target,
                        "predicted_area_percent": prediction,
                        "absolute_error_percentage_points": abs(prediction - target),
                    }
                )

    result_dir = output_dir / "test_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {name: float(np.mean(values)) for name, values in metric_values.items()}
    summary["samples"] = len(rows)
    save_json(output_dir / "test_summary.json", summary)
    printed_summary: dict[str, object] = {"segmentation": summary}
    if classification_rows:
        with (result_dir / "per_sample_classification.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(classification_rows[0]))
            writer.writeheader()
            writer.writerows(classification_rows)
        classification_summary = multiclass_summary(
            torch.cat(classification_predictions),
            torch.cat(classification_targets),
            class_names,
        )
        save_json(output_dir / "classification_summary.json", classification_summary)
        printed_summary["classification"] = classification_summary
    if tumor_size_rows:
        with (result_dir / "per_sample_tumor_size.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tumor_size_rows[0]))
            writer.writeheader()
            writer.writerows(tumor_size_rows)
        size_summary = tumor_size_summary(
            torch.cat(tumor_size_predictions),
            torch.cat(tumor_size_targets),
            torch.cat(tumor_size_classes),
            class_names,
        )
        save_json(output_dir / "tumor_size_summary.json", size_summary)
        printed_summary["tumor_size"] = size_summary
    print(json.dumps(printed_summary, indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Train the proposed deeper-head framework on an ultrasound manifest"
    )
    result.add_argument("--model", choices=[*MODEL_REGISTRY, "all"], default="TransUNetMAESizeDeepHeads")
    result.add_argument("--manifest", type=Path, default=Path("data/splits/busi_seed42.csv"))
    result.add_argument("--output-root", type=Path, default=Path("outputs"))
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--early-stopping-patience", type=int, default=40)
    result.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    result.add_argument("--batch-size", type=int, default=24)
    result.add_argument("--microbatch-size", type=int)
    result.add_argument("--test-batch-size", type=int, default=16)
    result.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--gradient-clip", type=float, default=0.5)
    result.add_argument("--classification-loss-weight", type=float, default=1.0 / 30.0)
    result.add_argument("--segmentation-loss-weight", type=float, default=0.90)
    result.add_argument("--tumor-size-loss-weight", type=float, default=1.0 / 30.0)
    result.add_argument(
        "--unlabeled-root",
        type=Path,
        default=Path("data/raw/BrEaST/BrEaST-Lesions_USG-images_and_masks"),
    )
    result.add_argument("--unlabeled-batch-size", type=int)
    result.add_argument("--mae-mask-ratio", type=float, default=0.30)
    result.add_argument("--reconstruction-loss-weight", type=float, default=1.0 / 30.0)
    result.add_argument(
        "--reconstruction-input-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Corrupt unlabeled MAE inputs using the configured ultrasound noise "
            "while retaining the clean image as the reconstruction target"
        ),
    )
    result.add_argument(
        "--ultrasound-noise", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--noise-probability", type=float, default=0.8)
    result.add_argument("--noise-min-intensity", type=float, default=0.15)
    result.add_argument("--noise-max-intensity", type=float, default=0.85)
    result.add_argument("--noise-secondary-probability", type=float, default=0.35)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument(
        "--initial-checkpoint",
        type=Path,
        help=(
            "Warm-start model parameters from a prior checkpoint while creating a "
            "fresh optimizer, scheduler, history, and early-stopping state"
        ),
    )
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return result


def run_all(args: argparse.Namespace) -> None:
    for key in MODEL_REGISTRY:
        command = [sys.executable, "-m", "breast_benchmark.train"]
        for name, value in vars(args).items():
            if name == "model":
                continue
            option = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                command.append(option if value else "--no-" + name.replace("_", "-"))
            elif value is not None:
                command.extend([option, str(value)])
        command.extend(["--model", key])
        print(f"Running {key}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    args = parser().parse_args()
    if args.model == "all":
        run_all(args)
    else:
        train_one(args)


if __name__ == "__main__":
    main()
