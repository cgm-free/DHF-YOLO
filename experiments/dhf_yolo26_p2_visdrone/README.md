# DHF-YOLO Experiment Scripts

This directory stores reproducible scripts and result summaries for the DHF-YOLO VisDrone/UAVDT experiments.

## Dataset Configs

Use repository-relative dataset YAML files:

```text
ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml
ultralytics/cfg/datasets/UAVDT-repo.yaml
```

Datasets are not included in this repository.

Expected VisDrone path:

```text
<repo-root>/VisDrone2019-DET Dataset
```

Expected UAVDT path:

```text
<repo-root>/UAVDT Dataset
```

## Main Scripts

| Script | Purpose |
|---|---|
| `11_train_yolo26n_p2_baseline_ciou_lab4090.sh` | YOLO26n-P2 baseline on VisDrone |
| `13_train_dhf_yolo26n_p2_ciou_lab4090.sh` | Final DHF-YOLO on VisDrone |
| `14_train_yolo26n_no_p2_ciou_lab4090.sh` | YOLO26n without P2 on VisDrone |
| `15_train_spd_yolo26n_p2_ciou_lab4090.sh` | SPDConv-only ablation |
| `16_train_dhfblock_yolo26n_p2_ciou_lab4090.sh` | C3k2-DHF-only ablation |
| `17_train_hdfm_yolo26n_p2_ciou_lab4090.sh` | HDFM-only ablation |
| `34_train_spd_dhf_yolo26n_p2_ciou_lab4090.sh` | SPDConv + C3k2-DHF ablation |
| `35_train_spd_hdfm_yolo26n_p2_ciou_lab4090.sh` | SPDConv + HDFM ablation |
| `36_train_dhf_hdfm_yolo26n_p2_ciou_lab4090.sh` | C3k2-DHF + HDFM ablation |
| `31_train_yolo26n_p2_baseline_ciou_uavdt_lab4090.sh` | YOLO26n-P2 baseline on UAVDT |
| `32_train_dhf_yolo26n_p2_ciou_uavdt_lab4090.sh` | DHF-YOLO on UAVDT |
| `33_train_yolo26n_baseline_ciou_uavdt_lab4090.sh` | YOLO26n without P2 on UAVDT |

## Script Parameters

Most scripts support environment-variable overrides:

```bash
PYTHON_BIN=python \
MODEL=ultralytics/cfg/models/26/dhf-yolo26-p2.yaml \
DATA=ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml \
EPOCHS=100 \
IMGSZ=640 \
BATCH=16 \
DEVICE=0 \
WORKERS=8 \
PROJECT=runs/visdrone_ablation \
NAME=custom_run_name \
bash experiments/dhf_yolo26_p2_visdrone/13_train_dhf_yolo26n_p2_ciou_lab4090.sh
```

Logs are written to:

```text
runs/visdrone_ablation_logs/*.out.log
runs/visdrone_ablation_logs/*.err.log
```

Ultralytics run outputs are written to:

```text
runs/detect/<project>/<name>
```

## Result Summary

`results_summary.csv` records all important experimental runs, including model, loss, epoch count, precision, recall, mAP@0.5, mAP@0.5:0.95, parameters, GFLOPs and result directory.

The final manuscript mainly uses:

| ID | Model | Dataset | Epochs | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---:|---:|---:|
| 14 | YOLO26n | VisDrone | 100 | 0.292 | 0.161 |
| 19 | YOLO26n-P2 | VisDrone | 300 | 0.333 | 0.195 |
| 18 | DHF-YOLO | VisDrone | 300 | 0.354 | 0.207 |
| 33 | YOLO26n | processed UAVDT clean validation | 100 | 0.599 | 0.310 |
| 31 | YOLO26n-P2 | processed UAVDT clean validation | 100 | 0.672 | 0.373 |
| 32 | DHF-YOLO | processed UAVDT clean validation | 100 | 0.707 | 0.404 |

## Notes

- `yolo26-p2.yaml` is kept as the baseline configuration.
- The final structure contribution is evaluated with CIoU. MPDIoU and NWD scripts are kept as supplementary negative/orthogonal trials.
- The processed UAVDT rows use final-epoch checkpoints on the deduplicated 270-image validation manifest and are exploratory rather than official UAVDT benchmark scores.
- Do not commit local datasets, `.pt` weights, logs, or private infrastructure notes.
