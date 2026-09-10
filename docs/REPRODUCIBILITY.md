# DHF-YOLO Reproducibility Record

## Current evidence

The September 2026 audit re-evaluated the retained `best.pt` checkpoints on the same VisDrone2019-DET validation labels. The authoritative table is `experiments/dhf_yolo26_p2_visdrone/verified_submission_results.csv`, which includes full-precision metrics, actual training batches and checkpoint SHA-256 values. No weights or datasets are redistributed.

Use only within-budget comparisons. Runs 18/19, 40/41, and 42/43 are paired 300-epoch P2/DHF runs with seeds 0, 1, and 2. Runs 11/13/14/15/16/17/34/35/37/38/39 are 100-epoch seed-0 comparisons. Run 39 is the batch-16 D+H result used in the manuscript; the older run 36 is retained only as provenance for a batch-8 pilot after an OOM. The manuscript includes the processed UAVDT subset only as a bounded exploratory check after removing the exact train--validation duplicate; it is not treated as an official sequence-disjoint benchmark.

## Processed UAVDT exploratory check

The processed YOLO-format archive was traced to Zenodo record 14575517. It contains 1,266 training, 271 validation, and 272 test images for car, truck, and bus. A file-level audit found one exact train--validation duplicate and one exact train--test duplicate. The reported validation manifest removes the train--validation duplicate, leaving 270 images and 7,014 boxes. Frames sharing the same source-sequence name remain distributed across subsets, so the split is not sequence-disjoint.

The manuscript reports the final-epoch `last.pt` checkpoints rather than checkpoints selected on the original validation split. All three runs use seed 0, 100 epochs, image size 640, training batch 16, CIoU, Ultralytics 8.4.51, and RTX 4090D. The clean-validation results are:

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLO26n | 0.599 | 0.628 | 0.599 | 0.310 |
| YOLO26n-P2 | 0.662 | 0.670 | 0.672 | 0.373 |
| DHF-YOLO | 0.749 | 0.634 | 0.707 | 0.404 |

The portable manifest, checkpoint hashes, and interpretation boundary are in `experiments/dhf_yolo26_p2_visdrone/uavdt_exploratory_20260905/`. These results are suitable only as an exploratory robustness check.

## Environment and training

The audited environment was Ultralytics 8.4.51 with PyTorch 2.11.0+cu128 on RTX 4090D. The repository must be installed editable so the custom classes can be loaded. The recorded runs initialize models from YAML, not from a pretrained checkpoint. Use the nano name, e.g. `dhf-yolo26n-p2.yaml`; the loader resolves it to the shared YAML and selects scale `n`.

Training uses image size 640, CIoU, batch 16, AMP, deterministic mode, linear LR scheduling, initial LR 0.01, final LR fraction 0.01, momentum 0.9, weight decay 0.0005, and mosaic disabled in the final ten epochs. The seed-0 runs recorded `optimizer=auto`, which selected MuSGD. The seed-1 and seed-2 queue fixed the resolved MuSGD settings explicitly and refused automatic batch reduction. The pretrained YOLO26n model used by Ultralytics' AMP compatibility check is separate from initialization of the trained model.

## Validation

For a retained checkpoint:

```bash
yolo detect val model=/path/to/best.pt data=ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml imgsz=640 batch=32 device=0 half=True rect=True split=val conf=0.001 iou=0.7 max_det=300 plots=True
```

Set the dataset YAML `path` to the actual dataset root. P/R are class-averaged at the validator's F1-selected threshold, not at confidence 0.25. AP uses the converted ten-class validation labels (38,759 boxes across 548 images); this is not the official challenge evaluator's ignored-region protocol. Category AP and PR curves must come from the same validation invocation.

## Completed submission-control experiments

All seven queue jobs completed successfully. Full-precision values and checkpoint SHA-256 hashes are in `experiments/dhf_yolo26_p2_visdrone/verified_submission_results.csv`.

| Job | Epochs | Seed | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| 37: full model, HDFM global logits disabled | 100 | 0 | 0.4475 | 0.3485 | 0.3206 | 0.1838 |
| 38: full model, local-mean subtraction disabled | 100 | 0 | 0.4448 | 0.3499 | 0.3214 | 0.1836 |
| 39: D+H, fixed batch 16 | 100 | 0 | 0.4442 | 0.3520 | 0.3210 | 0.1832 |
| 40: P2 | 300 | 1 | 0.4521 | 0.3483 | 0.3270 | 0.1882 |
| 41: DHF-YOLO | 300 | 1 | 0.4793 | 0.3707 | 0.3478 | 0.2021 |
| 42: P2 | 300 | 2 | 0.4634 | 0.3544 | 0.3325 | 0.1928 |
| 43: DHF-YOLO | 300 | 2 | 0.4773 | 0.3656 | 0.3440 | 0.2012 |

The first two controls retain branch parameters and computation but set the tested contribution to zero. They are mechanism controls, not reduced-cost models. Against run 13, disabling the HDFM global logits lowers mAP@0.5/mAP@0.5:0.95 by 1.43/0.92 percentage points; disabling C3k2-DHF local contrast lowers them by 1.34/0.94 points. Default control values are 1 and preserve checkpoint compatibility. The runner checks nano scale, parameter count, completed epoch count, actual batch, and checkpoint hash; it stops rather than silently treating a failed run as valid.

Across paired seeds 0, 1, and 2, P2 obtains 0.3309 +/- 0.0034 mAP@0.5 and 0.1920 +/- 0.0034 mAP@0.5:0.95; DHF-YOLO obtains 0.3485 +/- 0.0050 and 0.2036 +/- 0.0034. The paired gains are 1.76 +/- 0.53 and 1.16 +/- 0.29 percentage points (mean +/- sample SD), and both metrics improve for every seed. With only three pairs, these are descriptive repeatability statistics rather than a population-level significance claim.

Linux launch, using a new output directory:

```bash
python tools/submission_revision_queue.py --output runs/submission_revision --prepare --data-root '/path/to/VisDrone2019-DET Dataset'
python tools/submission_revision_queue.py --output runs/submission_revision --smoke --job 37_dhf_local_only_e100_s0
python tools/submission_revision_queue.py --output runs/submission_revision
```

Run the queue in `tmux` or another persistent shell for unattended training. Worker logs, frozen configuration files, checkpoint hashes, actual batches, and complete validation records are written under the output directory. The published rows were added only after those records passed the checks above.

## Diagnostic scripts

`tools/visdrone_size_recall_eval.py` computes same-class, confidence-ordered greedy matching at IoU 0.5, using original-image box areas. Use `--metadata-out` to save checkpoint/script hashes, label-manifest hash, settings and per-image counts. Its legacy bin name `medium` means **all area >= 32 squared pixels**, not COCO medium. Inference is batch 1, FP32, rectangular padding, confidence 0.25, maximum 300 detections.

`tools/benchmark_dhf_forward.py --baseline /path/to/p2.pt --dhf /path/to/dhf.pt --output forward.json` measures fused, eager PyTorch forward-only latency on a preallocated batch-1 640-square GPU tensor. It runs FP32/FP16, three counterbalanced timing blocks, each with 50 warmups and 200 measured iterations. It excludes image decoding, transfer, preprocessing and external post-processing; do not label it end-to-end FPS.

The measured raw timings are in `experiments/dhf_yolo26_p2_visdrone/forward_latency_verified.json`: aggregate medians are P2/DHF **2.93/3.87 ms (FP32)** and **3.21/4.25 ms (FP16)**. FP16 was not faster in this eager configuration. No optimized-engine or end-to-end speed claim follows from these measurements.

The `analysis_records/scale_p2.json` and `scale_dhf.json` files under the experiment directory provide the matched-checkpoint scale audit. Explicit single-image FP32 inference gives 16,032/17,367 matched GT instances out of 38,759, respectively. Earlier scale CSVs used incompletely recorded inference batching and are superseded, not silently relabeled as the same experiment.

`tools/make_verified_detection_crops.py` selects the most GT-dense cell in a fixed 3x3 grid of each of the two previously illustrated frames. Detection and class-aware IoU matching are performed on full images before cropping. It displays matched predictions, unmatched predictions, and missed GT separately. The selection rule and counts are saved in `analysis_records/crop_evidence.json`; the rendered dataset images are not redistributed here.
