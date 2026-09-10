"""Create GT-checked diagnostic crops from real validation images and retained checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from visdrone_size_recall_eval import YOLO, parse_gt_boxes, compute_iou


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--dhf', required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    models = [YOLO(args.baseline), YOLO(args.dhf)]
    scenes = ['0000295_02400_d_0000033.jpg', '0000295_02000_d_0000031.jpg']
    evidence = dict(selection='Two previously shown frames; densest GT-only tile in a fixed 3x3 grid.',
                    checkpoint_sha256=[hashlib.sha256(Path(x).read_bytes()).hexdigest()
                                       for x in (args.baseline, args.dhf)],
                    settings=dict(imgsz=640, conf=0.25, half=False, rect=True, batch=1,
                                  max_det=300, postprocess_iou=0.7, matching_iou=0.5), scenes=[])
    for row, name in enumerate(scenes):
        path = args.images / name
        im = np.asarray(Image.open(path).convert('RGB'))
        h, w = im.shape[:2]
        gt = parse_gt_boxes(args.images.parent / 'labels' / path.with_suffix('.txt').name, w, h)
        centres = np.array([(g.xyxy[:2] + g.xyxy[2:]) / 2 for g in gt])
        cells = [(w*x/3, h*y/3, w*(x+1)/3, h*(y+1)/3) for y in range(3) for x in range(3)]
        inside = lambda c: ((centres[:, 0] >= c[0]) & (centres[:, 0] < c[2]) &
                            (centres[:, 1] >= c[1]) & (centres[:, 1] < c[3]))
        cell = max(cells, key=lambda c: int(inside(c).sum()))
        selected = inside(cell)
        info = dict(image=name, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    crop_xyxy=cell, gt_in_crop=int(selected.sum()), models=[])
        for col in range(3):
            fig, ax = plt.subplots(figsize=(4, 2.4), dpi=250)
            ax.imshow(im)
            boxes = []
            if col == 0:
                boxes = [(g.xyxy, '#00b9e8', '-') for i, g in enumerate(gt) if selected[i]]
            else:
                result = models[col-1].predict(str(path), device=0, imgsz=640, conf=0.25,
                    batch=1, half=False, rect=True, iou=0.7, max_det=300, verbose=False)[0]
                xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                matched = np.zeros(len(gt), dtype=bool)
                pairs, fps = [], []
                for pi in np.argsort(-confidences):
                    best_iou, best_gt = 0., -1
                    for gi, g in enumerate(gt):
                        if matched[gi] or g.cls_id != classes[pi]:
                            continue
                        iou = compute_iou(xyxy[pi], g.xyxy)
                        if iou > best_iou:
                            best_iou, best_gt = iou, gi
                    if best_iou >= .5:
                        matched[best_gt] = True
                        pairs.append((int(pi), best_gt))
                    else:
                        fps.append(int(pi))
                pred_inside = lambda b: cell[0] <= (b[0]+b[2])/2 < cell[2] and cell[1] <= (b[1]+b[3])/2 < cell[3]
                tp = [(pi, gi) for pi, gi in pairs if selected[gi]]
                fp = [pi for pi in fps if pred_inside(xyxy[pi])]
                fn = np.where(selected & ~matched)[0].tolist()
                boxes += [(xyxy[pi], '#00a65a', '-') for pi, _ in tp]
                boxes += [(xyxy[pi], '#c7009b', '-') for pi in fp]
                boxes += [(gt[gi].xyxy, '#e32636', '--') for gi in fn]
                info['models'].append(dict(name=('P2', 'DHF')[col-1], tp=len(tp), fp=len(fp), fn=len(fn),
                                           matched_gt_indices=[gi for _, gi in tp], missed_gt_indices=fn))
            for b, color, style in boxes:
                ax.add_patch(Rectangle(b[:2], b[2]-b[0], b[3]-b[1], fill=False,
                                       edgecolor=color, linewidth=.75, linestyle=style))
            ax.set_xlim(cell[0], cell[2])
            ax.set_ylim(cell[3], cell[1])
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.savefig(args.output / f'crop_{row}_{col}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        evidence['scenes'].append(info)
    (args.output / 'crop_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
