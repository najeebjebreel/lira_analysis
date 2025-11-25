# Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

This repository provides an official implementation of the paper **"Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions"**.
We show that prior work often overestimated attack success by using overconfident models, target-data threshold calibration, and balanced membership priors.

**Key Contributions:**
- Evaluation with **anti-overfitting** and **transfer learning** defenses
- **Shadow-only threshold calibration** and precision under skewed priors (π ≤ 10%)
- **Per-sample reproducibility analysis** across architectures and configurations
- Comprehensive implementation with 5 LiRA variants

**Main Findings:**
- Defenses reduce LiRA success while maintaining utility
- Shadow-calibrated thresholds and realistic priors substantially lower precision
- Membership predictions are unstable; reproducibility requires support thresholding

---

## Installation

```bash
Download this anonymized repository
Navigate into the project directory:
cd lira_analysis

# Create environment (Python 3.8+)
conda create -n lira python=3.11
conda activate lira

# Install package
pip install .

# Optional: development tools and notebooks
pip install .[dev,notebooks]
```

---

## How to Run the Codes and Reproduce Results
Follow these steps to reproduce the full pipeline: **train/finetune shadow models → run LiRA attack → perform analysis**.

### 1) Train or Finetune Shadow Models (using a config file)

Choose the config that matches your dataset:

- **Images**: `configs/config_train_image.yaml` (for models trained from scratch); `configs/config_finetune.yaml` (for transfer learning models)
- **Tabular (Purchase-100)**: `configs/config_train_tabular.yaml`

Run training (defaults to shadow-model training with `training.num_shadow_models=256`; adjust this value as needed):

```bash
# Image example: CIFAR-10 with ResNet-18
python train.py --config configs/config_train_image.yaml \
  --override dataset.name=cifar10 model.architecture=resnet18

# Tabular example: Purchase-100 with a fully connected network
python train.py --config configs/config_train_tabular.yaml
```

- **Output**: `experiments/{dataset}/{model}/{timestamp}/` containing checkpoints, saved `keep_indices` and saved `train_config.yaml`.

### 2) Run the LiRA Attack (using the matching attack config)

Use the attack config that corresponds to your training run and pass the checkpoint directory from step 1:

```bash
python attack.py --config configs/config_attack.yaml \
  --override experiment.checkpoint_dir=experiments/cifar10/resnet18/YYYY-MM-DD_HHMM \
  --override attack.evaluation_mode=leave_one_out  # or "single" / "both"
```

- **Output**: Adds `attack_config.yaml`, ROC curves, per-variant metrics, and LiRA score files inside the same experiment directory.
- **Note**: The attack automatically loads all shadow model checkpoints found under `experiment.checkpoint_dir`.

### 3) Analyze Results (notebooks)

```bash
# Primary analysis: aggregates metrics, thresholds, and vulnerability rankings
jupyter notebook comprehensive_analysis/comprehensive_analysis.ipynb

# Optional: threshold stability and reproducibility studies
jupyter notebook comprehensive_analysis/threshold_distribution.ipynb
jupyter notebook comprehensive_analysis/reproducibility.ipynb
```

- In each notebook, set `EXP_PATH` to the experiment directory from steps 1–2.
- Outputs are written to `analysis_results/{dataset}/{model}/{config}/` (see the analysis README for details).

See [comprehensive_analysis/README.md](comprehensive_analysis/README.md) for notebook-specific guidance.

**Which notebook generates which paper asset?**

- `comprehensive_analysis.ipynb`: Tables **3–13** and Figures **4** & **11**.
- `threshold_distribution.ipynb`: Figure **1** (threshold stability across targets).
- `reproducibility.ipynb`: Figures **2, 3, 5, 6, 9, 10** plus the corresponding per-FPR reproducibility tables.
- `loss_ratio_tpr.ipynb`: Figure **7** (loss ratio vs. TPR).
- `plot_benchmark_distribution.ipynb`: Figure **8** (score distributions across benchmarks).

---

## Repository Structure

```
lira_analysis/
├── train.py                    # Shadow model training
├── attack.py                   # LiRA attack evaluation
├── configs/                    # YAML configurations
│   ├── config_train_image.yaml
│   ├── config_train_tabular.yaml
│   └── config_attack.yaml
├── attacks/
│   └── lira.py                 # LiRA implementation (5 variants)
├── utils/                      # Training, data loading, models
├── comprehensive_analysis/     # Analysis notebooks
│   ├── comprehensive_analysis.ipynb
│   ├── threshold_distribution.ipynb
│   ├── reproducibility.ipynb
│   └── ...
├── data/                       # Dataset directory
└── experiments/                # Auto-generated outputs
```

---

## Datasets

| Dataset | Type | Classes | Samples | Models |
|---------|------|---------|---------|--------|
| CIFAR-10 / CIFAR-100 | Image | 10 / 100 | 60K | ResNet-18, WideResNet, EfficientNet-V2 |
| GTSRB | Image | 43 | 51K | ResNet-18, EfficientNet-V2 |
| Purchase-100 | Tabular | 100 | 197K | FCN |

**Note:** Image datasets download automatically. Purchase-100 must be downloaded manually (see [data/Readme.md](data/Readme.md)).

---

## Attack Variants

This implementation provides 5 LiRA variants:

1. **LiRA (Online)** — Uses in/out distribution modeling (strongest)
2. **LiRA (Online, Fixed Variance)** — Global variance (more stable)
3. **LiRA (Offline)** — Out-distribution only (realistic)
4. **LiRA (Offline, Fixed Variance)** — Simplest offline variant
5. **Global Threshold** — Fixed threshold baseline

---

## Configuration

Key parameters in `configs/config_train_image.yaml`:
- `num_shadow_models`: Number of shadow models (default: 10, use 256+ for full experiments)
- `dataset.pkeep`: Member probability (default: 0.5)
- `training.epochs`: Training epochs per model
- `train_data_augmentation`: Augmentation strategies

Key parameters in `configs/config_attack.yaml`:
- `evaluation_mode`: `single`, `leave_one_out`, or `both`
- `target_fprs`: False positive rates for evaluation (e.g., [0.001, 0.01])
- `prior`: Membership prior for precision computation

---

## Citation


---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Original LiRA implementation: [Carlini et al., 2022](https://ieeexplore.ieee.org/document/9833649)
- Built with PyTorch and TIMM
- Code optimization: Claude Sonnet 4.5
