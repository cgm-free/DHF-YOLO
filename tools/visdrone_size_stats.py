from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


def load_dataset_yaml(dataset_yaml: Path) -> dict:
    """Load a dataset YAML file."""
    with dataset_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_dir(dataset: dict, dataset_yaml: Path, split: str) -> Path:
    """Resolve an image split directory from dataset YAML."""
    root = Path(dataset["path"])
    if not root.is_absolute():
        root = (dataset_yaml.parents[3] / root).resolve()
    return (root / dataset[split]).resolve()


def area_to_bin(area_px: float, tiny_thr: float, small_thr: float) -> str:
    """Map pixel area to a size bin."""
    if area_px < tiny_thr:
        return "tiny"
    if area_px < small_thr:
        return "small"
    return "medium"


def parse_label_file(label_path: Path) -> list[tuple[int, float, float]]:
    """Parse a YOLO label file and return (cls, w_norm, h_norm)."""
    rows: list[tuple[int, float, float]] = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        w = float(parts[3])
        h = float(parts[4])
        rows.append((cls_id, w, h))
    return rows


def main() -> None:
    """Compute VisDrone size-bin statistics from YOLO labels."""
    parser = argparse.ArgumentParser(description="Compute tiny/small/medium GT statistics for VisDrone-style YOLO labels.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to inspect.")
    parser.add_argument("--tiny-thr", type=float, default=16 * 16, help="Tiny upper area threshold in pixels.")
    parser.add_argument("--small-thr", type=float, default=32 * 32, help="Small upper area threshold in pixels.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path.")
    args = parser.parse_args()

    dataset_yaml = Path(args.data).resolve()
    dataset = load_dataset_yaml(dataset_yaml)
    image_dir = resolve_split_dir(dataset, dataset_yaml, args.split)
    label_dir = image_dir.parent / "labels"
    names = dataset.get("names", {})

    size_counter: Counter[str] = Counter()
    class_size_counter: dict[str, Counter[str]] = {}
    image_count = 0
    instance_count = 0

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for image_path in image_paths:
        image_count += 1
        with Image.open(image_path) as im:
            w_img, h_img = im.size
        label_path = label_dir / f"{image_path.stem}.txt"
        for cls_id, w_norm, h_norm in parse_label_file(label_path):
            area_px = (w_norm * w_img) * (h_norm * h_img)
            size_bin = area_to_bin(area_px, args.tiny_thr, args.small_thr)
            size_counter[size_bin] += 1
            class_name = names.get(cls_id, str(cls_id))
            class_size_counter.setdefault(class_name, Counter())[size_bin] += 1
            instance_count += 1

    rows = []
    for size_bin in ("tiny", "small", "medium"):
        count = size_counter[size_bin]
        ratio = count / instance_count if instance_count else 0.0
        rows.append({"scope": "all", "class": "all", "size_bin": size_bin, "count": count, "ratio": f"{ratio:.4f}"})

    for class_name in sorted(class_size_counter):
        total = sum(class_size_counter[class_name].values())
        for size_bin in ("tiny", "small", "medium"):
            count = class_size_counter[class_name][size_bin]
            ratio = count / total if total else 0.0
            rows.append(
                {"scope": "class", "class": class_name, "size_bin": size_bin, "count": count, "ratio": f"{ratio:.4f}"}
            )

    print(f"dataset_yaml: {dataset_yaml}")
    print(f"split: {args.split}")
    print(f"images: {image_count}")
    print(f"instances: {instance_count}")
    print(f"thresholds_px2: tiny<{args.tiny_thr:.0f}, small<{args.small_thr:.0f}, medium>=small")
    print()
    print("overall:")
    for size_bin in ("tiny", "small", "medium"):
        count = size_counter[size_bin]
        ratio = count / instance_count if instance_count else 0.0
        print(f"  {size_bin:>6}: {count:>7} ({ratio:.2%})")

    if args.csv_out:
        csv_path = Path(args.csv_out).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["scope", "class", "size_bin", "count", "ratio"])
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"csv_saved: {csv_path}")


if __name__ == "__main__":
    main()
