from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split


CLASSES = ("normal", "benign", "malignant")
BUSBRA_CLASSES = ("benign", "malignant")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    dataset_root: Path,
    output: Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> list[dict[str, str]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between zero and one")
    records: list[dict[str, str]] = []
    for class_name in CLASSES:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing BUSI class directory: {class_dir}")
        for image_path in sorted(class_dir.glob("*.png")):
            if "_mask" in image_path.stem:
                continue
            masks = sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))
            if not masks:
                raise FileNotFoundError(f"No mask found for {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            for mask_path in masks:
                with Image.open(mask_path) as mask:
                    if mask.size != (width, height):
                        raise ValueError(f"Mask size mismatch: {mask_path}")
            records.append(
                {
                    "dataset": "BUSI",
                    "sample_id": f"{class_name}/{image_path.stem}",
                    "group_id": f"{class_name}/{image_path.stem}",
                    "class_name": class_name,
                    "image_path": image_path.resolve().as_posix(),
                    "mask_paths": ";".join(p.resolve().as_posix() for p in masks),
                    "width": str(width),
                    "height": str(height),
                    "image_sha256": _sha256(image_path),
                }
            )

    labels = [record["class_name"] for record in records]
    train_indices, test_indices = train_test_split(
        list(range(len(records))),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_set = set(train_indices)
    test_set = set(test_indices)
    if train_set & test_set or len(train_set | test_set) != len(records):
        raise RuntimeError("Invalid train/test partition")
    for index, record in enumerate(records):
        record["split"] = "train" if index in train_set else "test"

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records


def build_busbra_manifest(
    dataset_root: Path,
    output: Path,
    seed: int = 42,
    test_size: float = 0.05,
) -> list[dict[str, str]]:
    """Build a pathology-stratified, patient-level BUS-BRA partition."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between zero and one")
    metadata_path = dataset_root / "bus_data.csv"
    image_root = dataset_root / "Images"
    mask_root = dataset_root / "Masks"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing BUS-BRA metadata: {metadata_path}")

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))
    if not metadata:
        raise ValueError(f"No BUS-BRA records in {metadata_path}")

    case_classes: dict[str, str] = {}
    for row in metadata:
        case_id = row["Case"]
        class_name = row["Pathology"].lower()
        if class_name not in BUSBRA_CLASSES:
            raise ValueError(f"Unknown BUS-BRA pathology {class_name!r}")
        previous = case_classes.setdefault(case_id, class_name)
        if previous != class_name:
            raise ValueError(
                f"BUS-BRA case {case_id} has mixed pathology labels: "
                f"{previous!r} and {class_name!r}"
            )

    case_ids = sorted(case_classes, key=int)
    train_cases, test_cases = train_test_split(
        case_ids,
        test_size=test_size,
        random_state=seed,
        stratify=[case_classes[case_id] for case_id in case_ids],
    )
    train_case_set = set(train_cases)
    test_case_set = set(test_cases)
    if train_case_set & test_case_set:
        raise RuntimeError("BUS-BRA patient split overlaps")

    records: list[dict[str, str]] = []
    for row in metadata:
        sample_name = row["ID"]
        image_path = image_root / f"{sample_name}.png"
        mask_name = sample_name.replace("bus_", "mask_", 1)
        mask_path = mask_root / f"{mask_name}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing BUS-BRA image/mask pair: {image_path}, {mask_path}"
            )
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask:
            if mask.size != (width, height):
                raise ValueError(f"Mask size mismatch: {mask_path}")
        case_id = row["Case"]
        records.append(
            {
                "dataset": "BUS-BRA",
                "sample_id": f"BUS-BRA/{sample_name}",
                "group_id": case_id,
                "class_name": row["Pathology"].lower(),
                "image_path": image_path.resolve().as_posix(),
                "mask_paths": mask_path.resolve().as_posix(),
                "width": str(width),
                "height": str(height),
                "image_sha256": _sha256(image_path),
                "histology": row["Histology"],
                "birads": row["BIRADS"],
                "device": row["Device"],
                "side": row["Side"],
                "bbox": row["BBOX"],
                "split": "train" if case_id in train_case_set else "test",
            }
        )

    if len(records) != len(metadata):
        raise RuntimeError("BUS-BRA manifest record count changed unexpectedly")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("BUSI", "BUS-BRA"),
        default="BUSI",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if args.dataset == "BUSI":
        dataset_root = args.dataset_root or Path(
            "data/raw/BUSI/Dataset_BUSI_with_GT"
        )
        output = args.output or Path("data/splits/busi_seed42.csv")
        records = build_manifest(
            dataset_root,
            output,
            seed=args.seed,
            test_size=args.test_size,
        )
        class_names = CLASSES
    else:
        dataset_root = args.dataset_root or Path("data/raw/BUS-BRA/BUSBRA")
        output = args.output or Path("data/splits/busbra_seed42_test5.csv")
        records = build_busbra_manifest(
            dataset_root,
            output,
            seed=args.seed,
            test_size=args.test_size,
        )
        class_names = BUSBRA_CLASSES
    counts = Counter((row["split"], row["class_name"]) for row in records)
    print(f"Wrote {len(records)} samples to {output}")
    for split in ("train", "test"):
        values = ", ".join(f"{name}={counts[split, name]}" for name in class_names)
        print(
            f"{split}: total={sum(counts[split, name] for name in class_names)}, "
            f"{values}"
        )


if __name__ == "__main__":
    main()
