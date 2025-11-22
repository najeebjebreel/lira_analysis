# Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

This repository provides an official implementation of **"Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions"**.
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
git clone https://github.com/najeebjebreel/lira_analysis.git
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

## Quick Start

**1. Train Shadow Models**

```bash
python train.py --config configs/config_train_image.yaml
```

This trains shadow models (default: 10 for testing, increase `num_shadow_models` for full experiments) and saves checkpoints to `experiments/{dataset}/{model}/{timestamp}/`.

**2. Run LiRA Attack**

```bash
python attack.py --config configs/config_attack.yaml \
  --override experiment.checkpoint_dir=experiments/cifar10/resnet18/YYYY-MM-DD_HHMM
```

Evaluates 5 attack variants and saves ROC curves, metrics, and vulnerability rankings.

**3. Analyze Results**

```bash
jupyter notebook comprehensive_analysis/comprehensive_analysis.ipynb
```

See [comprehensive_analysis/README.md](comprehensive_analysis/README.md) for detailed analysis workflows

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

```bibtex
@article{yourpaper2025,
  title={Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions},
  author={Your Name and Others},
  journal={Under Review},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Original LiRA implementation: [Carlini et al., 2022](https://arxiv.org/abs/2112.03570)
- Built with PyTorch and TIMM
- Code optimization: Claude Sonnet 4.5
