# DHF-YOLO for Small-Object Detection in UAV Imagery

This repository contains the implementation and experiment scripts for **DHF-YOLO**, a compact detector for small-object detection in UAV imagery. The codebase is derived from Ultralytics YOLO26 and keeps the original YOLO26-P2 baseline configuration intact for matched comparisons.

## Highlights

DHF-YOLO is built on the Ultralytics YOLO26n-P2 detection configuration and introduces three structural changes:

1. **SPDConv**: replaces selected stride-2 convolutions with detail-preserving space-to-depth downsampling.
2. **C3k2-DHF**: enhances high-resolution P2/P3 neck features with edge detail, local context and spatial gating.
3. **HDFM**: performs hierarchical dynamic feature fusion before the detection head to generate enhanced P2' and P3' features.

The final recommended configuration is:

```text
DHF-YOLO = YOLO26n-P2 + SPDConv + C3k2-DHF + HDFM + CIoU
```

Optional MPDIoU and NWD implementations are retained for exploratory work. The manuscript experiments use CIoU; the optional losses are not part of the reported method.

## Repository Status

This repository contains code, model YAML files, dataset YAML files, training scripts, analysis utilities and summary tables.

It does **not** include:

- VisDrone2019-DET images or labels
- UAVDT images or labels
- trained `.pt` weights
- `runs/` training outputs
- manuscript drafts, journal templates, or generated paper figures

These files are intentionally excluded so that the repository contains only redistributable source code and reproducibility records.

## Environment

The audited experiments used an editable installation (Ultralytics 8.4.51), PyTorch 2.11.0+cu128, and an RTX 4090D. Models were initialized from YAML, not pretrained checkpoints. See [the reproducibility record](docs/REPRODUCIBILITY.md) for validation settings and evidence boundaries.

Recommended setup:

```bash
conda create -n yolo26 python=3.10 -y
conda activate yolo26
pip install -e .
```

A quick model loading check:

```bash
python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/26/dhf-yolo26n-p2.yaml').info()"
```

## Datasets

### VisDrone2019-DET

Main benchmark dataset: **VisDrone2019-DET**.

Sources:

- VisDrone official repository: https://github.com/VisDrone/VisDrone-Dataset
- DatasetNinja page: https://datasetninja.com/vis-drone-2019-det

Expected local layout:

```text
<repo-root>/VisDrone2019-DET Dataset/
  VisDrone2019-DET-train/images
  VisDrone2019-DET-train/annotations
  VisDrone2019-DET-val/images
  VisDrone2019-DET-val/annotations
  VisDrone2019-DET-test-dev/images
  VisDrone2019-DET-test-dev/annotations
```

Dataset YAML:

```text
ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml
```

Class mapping used in this project:

| ID | Class |
|---:|---|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

Available split sizes (the manuscript uses train and val only):

| Split | Images |
|---|---:|
| train | 6471 |
| val | 548 |
| test-dev | 1610 |

Reported AP values are Ultralytics metrics on the 548-image validation split, not official test-server scores. Original categories 1--10 map to YOLO classes 0--9; ignored regions and category 11 are excluded by the label converter. This is not the official VisDrone ignored-region evaluation protocol.

Convert original VisDrone annotations to YOLO labels:

```bash
python -c "from ultralytics.data.converter import convert_visdrone_det_to_yolo; convert_visdrone_det_to_yolo('VisDrone2019-DET Dataset')"
```

### UAVDT

The manuscript includes a strictly bounded exploratory check on a processed UAVDT derivative traced to Zenodo record 14575517. This archive is not a verified official sequence-disjoint UAVDT split. Its 1,266/271/272-image train/validation/test partitions contain one exact train--validation duplicate and one exact train--test duplicate, and frames with the same source-sequence name occur across subsets. The paper therefore reports only a deduplicated validation check and does not present it as official UAVDT performance or independent cross-dataset generalization.

Expected processed layout:

```text
<repo-root>/UAVDT Dataset/
  train/images
  train/labels
  val/images
  val/labels
  test/images
  test/labels
```

Dataset YAML:

```text
ultralytics/cfg/datasets/UAVDT-repo.yaml
ultralytics/cfg/datasets/UAVDT-repo-clean-val.yaml
```

The portable 270-image validation manifest and its audit record are stored in:

```text
experiments/dhf_yolo26_p2_visdrone/uavdt_exploratory_20260905/
```

Copy `val_clean_no_train_overlap.txt` into the extracted `UAVDT Dataset` root before using `UAVDT-repo-clean-val.yaml`.

Class mapping:

| ID | Class |
|---:|---|
| 0 | car |
| 1 | truck |
| 2 | bus |

## Main Model Configurations

| File | Purpose |
|---|---|
| `ultralytics/cfg/models/26/yolo26-p2.yaml` | Official YOLO26-P2 baseline, unchanged |
| `ultralytics/cfg/models/26/dhf-yolo26-p2.yaml` | Final DHF-YOLO configuration |
| `ultralytics/cfg/models/26/spd-yolo26-p2.yaml` | SPDConv-only ablation |
| `ultralytics/cfg/models/26/dhfblock-yolo26-p2.yaml` | C3k2-DHF-only ablation |
| `ultralytics/cfg/models/26/hdfm-yolo26-p2.yaml` | HDFM-only ablation |
| `ultralytics/cfg/models/26/spd-dhfblock-yolo26-p2.yaml` | SPDConv + C3k2-DHF ablation |
| `ultralytics/cfg/models/26/spd-hdfm-yolo26-p2.yaml` | SPDConv + HDFM ablation |
| `ultralytics/cfg/models/26/dhfblock-hdfm-yolo26-p2.yaml` | C3k2-DHF + HDFM ablation |
| `ultralytics/cfg/models/26/dhfv2-yolo26-p2.yaml` | Experimental DHFv2 variant |

## Training Commands

The scripts are written so that they can run from any working directory. They set `PYTHONPATH` to the repository root and save stdout/stderr logs under `runs/visdrone_ablation_logs/`.

### VisDrone Main Experiments

YOLO26n without P2:

```bash
bash experiments/dhf_yolo26_p2_visdrone/14_train_yolo26n_no_p2_ciou_lab4090.sh
```

YOLO26n-P2 baseline:

```bash
bash experiments/dhf_yolo26_p2_visdrone/11_train_yolo26n_p2_baseline_ciou_lab4090.sh
```

DHF-YOLO:

```bash
bash experiments/dhf_yolo26_p2_visdrone/13_train_dhf_yolo26n_p2_ciou_lab4090.sh
```

Run 300 epochs by overriding environment variables:

```bash
EPOCHS=300 NAME=dhf_yolo26n_p2_ciou_e300 bash experiments/dhf_yolo26_p2_visdrone/13_train_dhf_yolo26n_p2_ciou_lab4090.sh
EPOCHS=300 NAME=yolo26n_p2_baseline_ciou_e300 bash experiments/dhf_yolo26_p2_visdrone/11_train_yolo26n_p2_baseline_ciou_lab4090.sh
```

### VisDrone Ablation Experiments

```bash
bash experiments/dhf_yolo26_p2_visdrone/15_train_spd_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/16_train_dhfblock_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/17_train_hdfm_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/34_train_spd_dhf_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/35_train_spd_hdfm_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/36_train_dhf_hdfm_yolo26n_p2_ciou_lab4090.sh
```

The submission-control queue reproduces the batch-matched D+H run, two internal mechanism controls, and paired seed-1/seed-2 300-epoch runs. It refuses automatic training-batch reduction and records frozen arguments, metrics, checkpoint hashes, parameters, and GFLOPs:

```bash
python tools/submission_revision_queue.py --output runs/submission_revision --prepare --data-root '/path/to/VisDrone2019-DET Dataset'
python tools/submission_revision_queue.py --output runs/submission_revision --smoke --job 37_dhf_local_only_e100_s0
python tools/submission_revision_queue.py --output runs/submission_revision
```

### Exploratory UAVDT Scripts

```bash
bash experiments/dhf_yolo26_p2_visdrone/30_smoke_uavdt_yolo26n_p2_ciou.sh
bash experiments/dhf_yolo26_p2_visdrone/33_train_yolo26n_baseline_ciou_uavdt_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/31_train_yolo26n_p2_baseline_ciou_uavdt_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/32_train_dhf_yolo26n_p2_ciou_uavdt_lab4090.sh
```

The manuscript's exploratory Table 6 uses the final-epoch `last.pt` checkpoints re-evaluated on the deduplicated 270-image validation manifest. See the audit record above for checkpoint hashes, metrics, and interpretation limits.

## Reported Results

### VisDrone2019-DET, 300 Epochs, Three Paired Seeds

Values are mean +/- sample SD over deterministic seeds 0, 1, and 2.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Fused params | Fused GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-P2 | 0.4578 +/- 0.0057 | 0.3546 +/- 0.0064 | 0.3309 +/- 0.0034 | 0.1920 +/- 0.0034 | 2.403M | 6.5 |
| DHF-YOLO | 0.4778 +/- 0.0014 | 0.3697 +/- 0.0038 | 0.3485 +/- 0.0050 | 0.2036 +/- 0.0034 | 2.631M | 9.9 |

The paired mAP gains are **1.76 +/- 0.53** and **1.16 +/- 0.29 percentage points**; both are positive for every tested seed. Three pairs characterize repeatability but are not a population-level significance test. Parameter and GFLOP values are for the 10-class VisDrone models after Ultralytics convolution--batch-normalization fusion. Fused GFLOPs increase by 52.3%, and measured fused eager forward time increases by approximately 32% in FP32. The available no-P2 checkpoint was trained for **100**, not 300, epochs and is excluded from this table.

### VisDrone2019-DET Ablation, 100 Epochs

`S` denotes SPDConv, `D` denotes C3k2-DHF, and `H` denotes HDFM.

| Variant | Train batch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 16 | 0.4364 | 0.3414 | 0.3117 | 0.1791 |
| S | 16 | 0.4433 | 0.3446 | 0.3178 | 0.1830 |
| D | 16 | 0.4465 | 0.3520 | 0.3254 | 0.1882 |
| H | 16 | 0.4396 | 0.3473 | 0.3169 | 0.1808 |
| S+D | 16 | 0.4499 | 0.3509 | 0.3269 | 0.1887 |
| S+H | 16 | 0.4539 | 0.3514 | 0.3238 | 0.1850 |
| D+H, batch-matched rerun | 16 | 0.4442 | 0.3520 | 0.3210 | 0.1832 |
| Full, HDFM global logits disabled | 16 | 0.4475 | 0.3485 | 0.3206 | 0.1838 |
| Full, C3k2-DHF local contrast disabled | 16 | 0.4448 | 0.3499 | 0.3214 | 0.1836 |
| Full, S+D+H | 16 | 0.4684 | 0.3532 | 0.3348 | 0.1930 |

The batch-16 D+H rerun supersedes the earlier batch-8 pilot. Disabling HDFM global logits or C3k2-DHF local contrast reduces mAP@0.5:0.95 by 0.92 and 0.94 percentage points, respectively, relative to the full model while retaining the same 2.631M parameters and 9.9 GFLOPs. All rows were revalidated at batch 32 with the same settings. The no-P2 100-epoch result is P=0.4246, R=0.3137, AP50=0.2916, AP50-95=0.1604.

The [audited full-precision CSV](experiments/dhf_yolo26_p2_visdrone/verified_submission_results.csv) is the numerical source. Checkpoint hashes, evaluation boundaries, and the completed submission-control queue are documented in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Code Changes

Main implementation changes relative to upstream Ultralytics:

| File | Change |
|---|---|
| `ultralytics/nn/modules/conv.py` | Adds `SPDConv` for detail-preserving downsampling |
| `ultralytics/nn/modules/block.py` | Adds `C3k2DHF`, `C3k2DHFv2`, `HDFM`, and `HDFMv2` modules used by DHF-YOLO and its controls |
| `ultralytics/nn/modules/__init__.py` | Exports custom modules |
| `ultralytics/nn/tasks.py` | Registers custom modules and supports multi-input fusion modules in model parsing |
| `ultralytics/utils/loss.py` | Adds optional MPDIoU/NWD loss experiments while keeping CIoU as default |
| `ultralytics/utils/metrics.py` | Adds metric helpers needed by optional loss experiments |
| `ultralytics/cfg/default.yaml` | Adds `box_loss` configuration option |
| `ultralytics/cfg/models/26/*.yaml` | Adds DHF-YOLO and ablation model configurations |
| `ultralytics/cfg/datasets/*.yaml` | Adds portable dataset configs for VisDrone and UAVDT |
| `experiments/dhf_yolo26_p2_visdrone/*.sh` | Reproducible training scripts and logging commands |
| `tools/*.py` | Analysis and visualization utilities for paper figures and scale recall |

## License

This project is derived from Ultralytics and follows the upstream **AGPL-3.0** license.

## Citation

If this repository is useful for your research, please cite the project URL:

```bibtex
@misc{dhfyolo2026,
  title  = {DHF-YOLO for UAV Aerial Small Object Detection},
  author = {Chu, Guoming and Liu, Shenghui and Luo, Xuhong and Chen, Yaoyao},
  year   = {2026},
  url    = {https://github.com/cgm-free/DHF-YOLO}
}
```
