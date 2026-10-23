from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean(rows: list[dict[str, str]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows])) if rows else float("nan")


def summarize_run(run_dir: Path) -> dict[str, str | int | float]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "test_summary.json").read_text(encoding="utf-8"))
    classification_path = run_dir / "classification_summary.json"
    classification = (
        json.loads(classification_path.read_text(encoding="utf-8"))
        if classification_path.is_file()
        else None
    )
    tumor_size_path = run_dir / "tumor_size_summary.json"
    tumor_size = (
        json.loads(tumor_size_path.read_text(encoding="utf-8"))
        if tumor_size_path.is_file()
        else None
    )
    with (run_dir / "test_results" / "per_sample_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        samples = list(csv.DictReader(handle))
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        by_class[sample["class_name"]].append(sample)
    lesions = by_class["benign"] + by_class["malignant"]
    history_rows = list(
        csv.DictReader((run_dir / "history.csv").open(newline="", encoding="utf-8"))
    )

    def class_recall(class_name: str) -> float | str:
        if classification is None:
            return ""
        class_metrics = classification["per_class"].get(class_name)
        return class_metrics["recall"] if class_metrics is not None else float("nan")

    result: dict[str, str | int | float] = {
        "dataset": config.get("dataset", "BUSI"),
        "key": config.get("run_key", config["model"]["key"]),
        "label": config["model"]["label"],
        "parameters": config["actual_trainable_parameters"],
        "epochs": config["epochs"],
        "completed_epochs": config.get("completed_epochs", len(history_rows)),
        "stopped_early": config.get(
            "stopped_early", len(history_rows) < config["epochs"]
        ),
        "early_stopping_patience": config.get("early_stopping", {}).get("patience", ""),
        "best_train_loss": config.get(
            "best_train_loss", min(float(row["train_loss"]) for row in history_rows)
        ),
        "batch_size": config["batch_size"],
        "final_train_loss": float(history_rows[-1]["train_loss"]),
        "dice": summary["dice"],
        "iou": summary["iou"],
        "lesion_dice": mean(lesions, "dice"),
        "lesion_iou": mean(lesions, "iou"),
        "normal_dice": mean(by_class["normal"], "dice"),
        "benign_dice": mean(by_class["benign"], "dice"),
        "malignant_dice": mean(by_class["malignant"], "dice"),
        "precision": summary["precision"],
        "recall": summary["recall"],
        "specificity": summary["specificity"],
        "accuracy": summary["accuracy"],
        "classification_accuracy": classification["accuracy"] if classification else "",
        "classification_balanced_accuracy": (
            classification["balanced_accuracy"] if classification else ""
        ),
        "classification_macro_f1": classification["macro_f1"] if classification else "",
        "normal_class_recall": class_recall("normal"),
        "benign_class_recall": class_recall("benign"),
        "malignant_class_recall": class_recall("malignant"),
        "tumor_size_mae_percentage_points": (
            tumor_size["mae_percentage_points"] if tumor_size else ""
        ),
        "tumor_size_rmse_percentage_points": (
            tumor_size["rmse_percentage_points"] if tumor_size else ""
        ),
        "tumor_size_lesion_mae_percentage_points": (
            tumor_size["lesion_only_mae_percentage_points"] if tumor_size else ""
        ),
        "tumor_size_r_squared": tumor_size["r_squared"] if tumor_size else "",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed comparator runs")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison.csv"))
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.output_root.glob("*_seed*")):
        if (run_dir / "COMPLETE").is_file():
            rows.append(summarize_run(run_dir))
    if not rows:
        raise RuntimeError(f"No completed runs found under {args.output_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
