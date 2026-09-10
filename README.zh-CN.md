# DHF-YOLO：面向无人机航拍小目标检测的轻量化改进模型

本仓库提供 **DHF-YOLO** 的代码、模型配置、训练脚本和实验记录。项目派生自 Ultralytics YOLO26，并保留官方 YOLO26-P2 基线配置用于同口径比较。

## 方法概述

DHF-YOLO 基于 Ultralytics YOLO26n-P2，主要包含三个结构改进：

1. **SPDConv**：细节保留下采样模块，用空间到深度变换替代部分步长卷积，降低浅层小目标细节损失。
2. **C3k2-DHF**：高分辨率细节增强模块，用边缘分支、上下文分支和空间门控增强 P2/P3 高分辨率特征。
3. **HDFM**：层次动态融合模块，在检测头前对多尺度特征进行动态融合，生成增强后的 P2' 和 P3' 特征。

最终推荐配置为：

```text
DHF-YOLO = YOLO26n-P2 + SPDConv + C3k2-DHF + HDFM + CIoU
```

MPDIoU 和 NWD 实现保留用于探索性实验。论文实验使用 CIoU，扩展损失不是论文方法的组成部分。

## 仓库包含与不包含的内容

本仓库包含：

- 模型代码
- 模型 YAML 配置
- 数据集 YAML 配置
- 训练脚本
- 消融实验脚本
- 可视化和分析工具
- 实验结果汇总表

本仓库不包含：

- VisDrone2019-DET 原始数据集
- UAVDT 原始数据集
- 训练得到的 `.pt` 权重
- `runs/` 训练输出目录
- 投稿论文 Word/LaTeX 模板和生成图件

这些文件被排除，以使仓库只包含允许再分发的源代码和复现记录。

## 环境安装

已核验实验使用 Ultralytics 8.4.51、PyTorch 2.11.0+cu128 和 RTX 4090D，从 YAML 随机初始化，不加载预训练权重。完整验证口径见 [复现记录](docs/REPRODUCIBILITY.md)。安装示例：

```bash
conda create -n yolo26 python=3.10 -y
conda activate yolo26
pip install -e .
```

模型加载测试：

```bash
python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/26/dhf-yolo26n-p2.yaml').info()"
```

## 数据集说明

### VisDrone2019-DET

主实验数据集为 **VisDrone2019-DET**。

数据来源：

- VisDrone 官方仓库：https://github.com/VisDrone/VisDrone-Dataset
- DatasetNinja 页面：https://datasetninja.com/vis-drone-2019-det

本项目期望的数据目录：

```text
<repo-root>/VisDrone2019-DET Dataset/
  VisDrone2019-DET-train/images
  VisDrone2019-DET-train/annotations
  VisDrone2019-DET-val/images
  VisDrone2019-DET-val/annotations
  VisDrone2019-DET-test-dev/images
  VisDrone2019-DET-test-dev/annotations
```

数据集配置文件：

```text
ultralytics/cfg/datasets/VisDrone2019-DET-repo.yaml
```

类别映射：

| ID | 类别 |
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

数据划分规模（论文只使用 train 和 val）：

| 划分 | 图片数 |
|---|---:|
| train | 6471 |
| val | 548 |
| test-dev | 1610 |

论文 AP 为 548 张验证图上的 Ultralytics 指标，不是官方测试服务器成绩。转换器将原类别 1--10 映射至 0--9，排除忽略区域及类别 11；此口径不同于官方 VisDrone 忽略区域评测协议。

将 VisDrone 原始标注转换为 YOLO 格式：

```bash
python -c "from ultralytics.data.converter import convert_visdrone_det_to_yolo; convert_visdrone_det_to_yolo('VisDrone2019-DET Dataset')"
```

### UAVDT

论文将处理后的 UAVDT 子集作为边界明确的探索性稳健性检查。数据来源追溯到 Zenodo 记录 14575517，但其划分不符合已核验的官方序列互斥协议。该副本 train/val/test 分别为 1266/271/272 张，并存在 1 张训练-验证完全重复图及 1 张训练-测试完全重复图。论文结果删除训练-验证重复图后，在 270 张验证图上重新评估；这些分数不表述为官方 UAVDT 性能，也不用于宣称独立跨数据集泛化。

期望的处理后目录：

```text
<repo-root>/UAVDT Dataset/
  train/images
  train/labels
  val/images
  val/labels
  test/images
  test/labels
```

数据集配置文件：

```text
ultralytics/cfg/datasets/UAVDT-repo.yaml
ultralytics/cfg/datasets/UAVDT-repo-clean-val.yaml
```

类别映射：

| ID | 类别 |
|---:|---|
| 0 | car |
| 1 | truck |
| 2 | bus |

## 主要模型配置

| 文件 | 作用 |
|---|---|
| `ultralytics/cfg/models/26/yolo26-p2.yaml` | 官方 YOLO26-P2 基线，不修改 |
| `ultralytics/cfg/models/26/dhf-yolo26-p2.yaml` | 最终 DHF-YOLO 配置 |
| `ultralytics/cfg/models/26/spd-yolo26-p2.yaml` | 仅 SPDConv 消融 |
| `ultralytics/cfg/models/26/dhfblock-yolo26-p2.yaml` | 仅 C3k2-DHF 消融 |
| `ultralytics/cfg/models/26/hdfm-yolo26-p2.yaml` | 仅 HDFM 消融 |
| `ultralytics/cfg/models/26/spd-dhfblock-yolo26-p2.yaml` | SPDConv + C3k2-DHF 消融 |
| `ultralytics/cfg/models/26/spd-hdfm-yolo26-p2.yaml` | SPDConv + HDFM 消融 |
| `ultralytics/cfg/models/26/dhfblock-hdfm-yolo26-p2.yaml` | C3k2-DHF + HDFM 消融 |
| `ultralytics/cfg/models/26/dhfv2-yolo26-p2.yaml` | DHFv2 扩展实验 |

## 训练命令

训练脚本会自动进入仓库根目录，设置 `PYTHONPATH`，并将日志保存到 `runs/visdrone_ablation_logs/`。

### VisDrone 主实验

YOLO26n 不带 P2：

```bash
bash experiments/dhf_yolo26_p2_visdrone/14_train_yolo26n_no_p2_ciou_lab4090.sh
```

YOLO26n-P2 基线：

```bash
bash experiments/dhf_yolo26_p2_visdrone/11_train_yolo26n_p2_baseline_ciou_lab4090.sh
```

DHF-YOLO：

```bash
bash experiments/dhf_yolo26_p2_visdrone/13_train_dhf_yolo26n_p2_ciou_lab4090.sh
```

300 epoch 训练：

```bash
EPOCHS=300 NAME=dhf_yolo26n_p2_ciou_e300 bash experiments/dhf_yolo26_p2_visdrone/13_train_dhf_yolo26n_p2_ciou_lab4090.sh
EPOCHS=300 NAME=yolo26n_p2_baseline_ciou_e300 bash experiments/dhf_yolo26_p2_visdrone/11_train_yolo26n_p2_baseline_ciou_lab4090.sh
```

### VisDrone 消融实验

```bash
bash experiments/dhf_yolo26_p2_visdrone/15_train_spd_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/16_train_dhfblock_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/17_train_hdfm_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/34_train_spd_dhf_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/35_train_spd_hdfm_yolo26n_p2_ciou_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/36_train_dhf_hdfm_yolo26n_p2_ciou_lab4090.sh
```

投稿补充实验队列用于复现 batch 一致的 D+H、两组内部机制对照，以及 seed 1/2 的 300 轮成对重复。队列禁止自动降低训练 batch，并保存冻结参数、指标、检查点哈希、参数量和 GFLOPs：

```bash
python tools/submission_revision_queue.py --output runs/submission_revision --prepare --data-root '/path/to/VisDrone2019-DET Dataset'
python tools/submission_revision_queue.py --output runs/submission_revision --smoke --job 37_dhf_local_only_e100_s0
python tools/submission_revision_queue.py --output runs/submission_revision
```

### UAVDT 历史脚本（非当前论文证据）

```bash
bash experiments/dhf_yolo26_p2_visdrone/30_smoke_uavdt_yolo26n_p2_ciou.sh
bash experiments/dhf_yolo26_p2_visdrone/33_train_yolo26n_baseline_ciou_uavdt_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/31_train_yolo26n_p2_baseline_ciou_uavdt_lab4090.sh
bash experiments/dhf_yolo26_p2_visdrone/32_train_dhf_yolo26n_p2_ciou_uavdt_lab4090.sh
```

## 实验结果

### VisDrone2019-DET，300 epoch，三组成对随机种子

下表为确定性随机种子 0、1、2 的均值 +/- 样本标准差。

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 融合后参数量 | 融合后 GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-P2 | 0.4578 +/- 0.0057 | 0.3546 +/- 0.0064 | 0.3309 +/- 0.0034 | 0.1920 +/- 0.0034 | 2.403M | 6.5 |
| DHF-YOLO | 0.4778 +/- 0.0014 | 0.3697 +/- 0.0038 | 0.3485 +/- 0.0050 | 0.2036 +/- 0.0034 | 2.631M | 9.9 |

两项 mAP 的成对提升为 **1.76 +/- 0.53、1.16 +/- 0.29 个百分点**，每个随机种子均为正提升。三组成对运行用于描述可重复性，不构成总体显著性检验。参数量和 GFLOPs 对应 VisDrone 10 类模型，均在执行 Ultralytics 卷积与批归一化融合后统计。融合后 GFLOPs 增加 52.3%；在 FP32 下，实测融合后 eager 前向时间约增加 32%。已有无 P2 检查点训练了 **100 轮而非 300 轮**，不放入该表混比。

### VisDrone2019-DET 消融实验，100 epoch

`S` 表示 SPDConv，`D` 表示 C3k2-DHF，`H` 表示 HDFM。

| 变体 | 训练 batch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 16 | 0.4364 | 0.3414 | 0.3117 | 0.1791 |
| S | 16 | 0.4433 | 0.3446 | 0.3178 | 0.1830 |
| D | 16 | 0.4465 | 0.3520 | 0.3254 | 0.1882 |
| H | 16 | 0.4396 | 0.3473 | 0.3169 | 0.1808 |
| S+D | 16 | 0.4499 | 0.3509 | 0.3269 | 0.1887 |
| S+H | 16 | 0.4539 | 0.3514 | 0.3238 | 0.1850 |
| D+H，batch 一致复跑 | 16 | 0.4442 | 0.3520 | 0.3210 | 0.1832 |
| 完整模型，关闭 HDFM 全局 logits | 16 | 0.4475 | 0.3485 | 0.3206 | 0.1838 |
| 完整模型，关闭 C3k2-DHF 局部对比 | 16 | 0.4448 | 0.3499 | 0.3214 | 0.1836 |
| 完整模型，S+D+H | 16 | 0.4684 | 0.3532 | 0.3348 | 0.1930 |

D+H 的 batch=16 复跑结果取代了早期 batch=8 试验。相对完整模型，关闭 HDFM 全局 logits 或 C3k2-DHF 局部对比分别使 mAP@0.5:0.95 降低 0.92、0.94 个百分点，同时参数量和 GFLOPs 保持为 2.631M、9.9。上表所有检查点均以验证 batch=32 及相同设置重新评估。无 P2 的 100 轮结果为 P=0.4246、R=0.3137、AP50=0.2916、AP50-95=0.1604。

当前数值来源为[已核验的全精度 CSV](experiments/dhf_yolo26_p2_visdrone/verified_submission_results.csv)。检查点哈希、评估边界和已完成的投稿补充实验队列见[复现记录](docs/REPRODUCIBILITY.md)。

## 代码改动说明

| 文件 | 改动 |
|---|---|
| `ultralytics/nn/modules/conv.py` | 增加 `SPDConv` |
| `ultralytics/nn/modules/block.py` | 增加 DHF-YOLO 及其对照实验使用的 `C3k2DHF`、`C3k2DHFv2`、`HDFM`、`HDFMv2` |
| `ultralytics/nn/modules/__init__.py` | 导出自定义模块 |
| `ultralytics/nn/tasks.py` | 注册自定义模块，并支持多输入融合模块解析 |
| `ultralytics/utils/loss.py` | 增加 MPDIoU/NWD 扩展损失实验，默认仍使用 CIoU |
| `ultralytics/utils/metrics.py` | 增加扩展损失所需的度量函数 |
| `ultralytics/cfg/default.yaml` | 增加 `box_loss` 参数 |
| `ultralytics/cfg/models/26/*.yaml` | 增加 DHF-YOLO 和消融模型配置 |
| `ultralytics/cfg/datasets/*.yaml` | 增加 VisDrone 和 UAVDT 可移植数据集配置 |
| `experiments/dhf_yolo26_p2_visdrone/*.sh` | 增加可复现实验训练脚本 |
| `tools/*.py` | 增加论文图件、检测对比和尺度召回分析工具 |

## 开源协议

本项目派生自 Ultralytics，遵循上游 **AGPL-3.0** 协议。

## 引用

如果本项目对你的研究有帮助，可以引用本仓库：

```bibtex
@misc{dhfyolo2026,
  title  = {DHF-YOLO for UAV Aerial Small Object Detection},
  author = {Chu, Guoming and Liu, Shenghui and Luo, Xuhong and Chen, Yaoyao},
  year   = {2026},
  url    = {https://github.com/cgm-free/DHF-YOLO}
}
```
