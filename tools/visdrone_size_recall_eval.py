from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


@dataclass
class BoxRecord:
    """Container for a single box."""

    cls_id: int
    xyxy: np.ndarray
    area_px: float
    conf: float = 1.0


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


def yolo_label_to_xyxy(row: list[float], width: int, height: int) -> BoxRecord:
    """Convert one YOLO label row to absolute xyxy."""
    cls_id, x_c, y_c, w, h = row
    bw = w * width
    bh = h * height
    x1 = (x_c * width) - bw / 2
    y1 = (y_c * height) - bh / 2
    x2 = x1 + bw
    y2 = y1 + bh
    return BoxRecord(int(cls_id), np.array([x1, y1, x2, y2], dtype=np.float32), float(bw * bh))


def parse_gt_boxes(label_path: Path, width: int, height: int) -> list[BoxRecord]:
    """Load GT boxes from a YOLO txt file."""
    boxes: list[BoxRecord] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        row = [float(v) for v in parts]
        boxes.append(yolo_label_to_xyxy(row, width, height))
    return boxes


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area1 = max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0)
    area2 = max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0)
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def area_to_bin(area_px: float, tiny_thr: float, small_thr: float) -> str:
    """Map pixel area to a size bin."""
    if area_px < tiny_thr:
        return "tiny"
    if area_px < small_thr:
        return "small"
    return "medium"


def main() -> None:
    """Run size-binned recall evaluation on a YOLO model."""
    parser = argparse.ArgumentParser(description="Evaluate size-binned GT recall for a YOLO model on VisDrone-style labels.")
    parser.add_argument("--model", required=True, help="Path to model weights or model YAML.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="GT matching IoU threshold.")
    parser.add_argument("--device", default="", help="Inference device, e.g. 'cpu' or '0'.")
    parser.add_argument("--tiny-thr", type=float, default=16 * 16, help="Tiny upper area threshold in pixels.")
    parser.add_argument("--small-thr", type=float, default=32 * 32, help="Small upper area threshold in pixels.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for quick debugging.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path.")
    parser.add_argument("--metadata-out", default="", help="Checkpoint-linked JSON with per-image counts.")
    args = parser.parse_args()

    dataset_yaml = Path(args.data).resolve()
    dataset = load_dataset_yaml(dataset_yaml)
    image_dir = resolve_split_dir(dataset, dataset_yaml, args.split)
    label_dir = image_dir.parent / "labels"

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    model = YOLO(args.model)

    gt_counts = {"tiny": 0, "small": 0, "medium": 0}
    matched_counts = {"tiny": 0, "small": 0, "medium": 0}
    pred_total = 0
    pred_matched = 0
    per_image = []
    label_hash = hashlib.sha256()

    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        batch=1,
        half=False,
        rect=True,
        iou=0.7,
        max_det=300,
        device=args.device if args.device else None,
        stream=True,
        verbose=False,
        save=False,
    )

    for image_path, result in zip(image_paths, results):
        if Path(result.path).resolve() != image_path.resolve():
            raise RuntimeError(f'Prediction/label ordering mismatch: {result.path}, {image_path}')
        with Image.open(image_path) as im:
            width, height = im.size

        gt_boxes = parse_gt_boxes(label_dir / f"{image_path.stem}.txt", width, height)
        label_path = label_dir / f"{image_path.stem}.txt"
        label_hash.update(image_path.name.encode())
        label_hash.update(label_path.read_bytes() if label_path.exists() else b'<missing>')
        matched_gt = [False] * len(gt_boxes)

        pred_boxes: list[BoxRecord] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            order = np.argsort(-conf)
            for idx in order:
                area_px = max(xyxy[idx][2] - xyxy[idx][0], 0.0) * max(xyxy[idx][3] - xyxy[idx][1], 0.0)
                pred_boxes.append(BoxRecord(int(cls[idx]), xyxy[idx].astype(np.float32), float(area_px), float(conf[idx])))

        pred_total += len(pred_boxes)

        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt in enumerate(gt_boxes):
                if matched_gt[gi] or gt.cls_id != pred.cls_id:
                    continue
                iou = compute_iou(pred.xyxy, gt.xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_gt_idx >= 0 and best_iou >= args.iou:
                matched_gt[best_gt_idx] = True
                pred_matched += 1

        for gi, gt in enumerate(gt_boxes):
            size_bin = area_to_bin(gt.area_px, args.tiny_thr, args.small_thr)
            gt_counts[size_bin] += 1
            if matched_gt[gi]:
                matched_counts[size_bin] += 1
        per_image.append(dict(image=image_path.name, ground_truth=len(gt_boxes),
                              predictions=len(pred_boxes), matched=sum(matched_gt)))

    rows = []
    for size_bin in ("tiny", "small", "medium"):
        gt_count = gt_counts[size_bin]
        matched = matched_counts[size_bin]
        recall = matched / gt_count if gt_count else 0.0
        rows.append({"size_bin": size_bin, "gt_count": gt_count, "matched": matched, "recall@0.5": f"{recall:.4f}"})

    overall_recall = sum(matched_counts.values()) / sum(gt_counts.values()) if sum(gt_counts.values()) else 0.0
    matched_precision = pred_matched / pred_total if pred_total else 0.0

    print(f"model: {args.model}")
    print(f"data: {dataset_yaml}")
    print(f"split: {args.split}")
    print(f"images: {len(image_paths)}")
    print(f"conf: {args.conf}")
    print(f"iou_match: {args.iou}")
    print(f"thresholds_px2: tiny<{args.tiny_thr:.0f}, small<{args.small_thr:.0f}, medium>=small")
    print()
    for row in rows:
        print(
            f"{row['size_bin']:>6}: gt={row['gt_count']:>7}, matched={row['matched']:>7}, recall@0.5={row['recall@0.5']}"
        )
    print()
    print(f"overall_gt_recall@0.5: {overall_recall:.4f}")
    print(f"matched_precision@conf{args.conf:.2f}: {matched_precision:.4f}")

    if args.csv_out:
        csv_path = Path(args.csv_out).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["size_bin", "gt_count", "matched", "recall@0.5"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv_saved: {csv_path}")

    if args.metadata_out:
        import torch
        import ultralytics
        metadata_path = Path(args.metadata_out)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(checkpoint_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        labels_manifest_sha256=label_hash.hexdigest(),
                        torch=torch.__version__, ultralytics=ultralytics.__version__,
                        settings=dict(imgsz=args.imgsz, conf=args.conf, matching_iou=args.iou,
                                      batch=1, half=False, rect=True, postprocess_iou=0.7, max_det=300,
                                      tiny_area=args.tiny_thr, small_area=args.small_thr, split=args.split),
                        rows=rows, total_gt=sum(gt_counts.values()), total_matched=sum(matched_counts.values()),
                        overall_recall=overall_recall, per_image=per_image,
                        note='Legacy bin name medium means all area >= small_area, not COCO medium.')
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == "__main__":
    main()
