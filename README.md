# LiRA: Likelihood Ratio Attack for Membership Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research implementation of "Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions"**

This repository provides a production-quality, reproducible implementation of the Likelihood Ratio Attack (LiRA) for membership inference. The codebase follows modern research engineering best practices with modular design, comprehensive documentation, and extensive configurability.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Shadow Models](#training-shadow-models)
  - [Running Attacks](#running-attacks)
  - [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Supported Datasets and Models](#supported-datasets-and-models)
- [Attack Variants](#attack-variants)
- [Evaluation Modes](#evaluation-modes)
- [Results and Visualization](#results-and-visualization)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Membership Inference Attacks (MIAs)** determine whether a specific data point was used to train a machine learning model. This poses serious privacy risks, especially when training data contains sensitive information.

**LiRA (Likelihood Ratio Attack)** is a state-of-the-art membership inference technique that leverages likelihood ratios computed from shadow models trained on different data subsets. This implementation extends the original work with:

- **Realistic threat models**: Online and offline attack variants
- **Multiple evaluation modes**: Single-target and leave-one-out cross-validation
- **Spatial augmentation support**: For improved attack robustness
- **Comprehensive metrics**: AUC, accuracy, TPR@FPR, precision

---

## Features

✅ **Modular architecture**: Clean separation of training, attack, and analysis code
✅ **Multiple datasets**: CIFAR-10/100, GTSRB (image), Purchase-100 (tabular)
✅ **Flexible model zoo**: ResNet, WideResNet, FCN, and 200+ models via TIMM
✅ **Attack variants**: Online, offline, fixed/adaptive variance, global threshold
✅ **Evaluation modes**: Single-target and leave-one-out with uncertainty quantification
✅ **Data augmentation**: Training augmentations (MixUp, CutMix, etc.) and inference augmentations
✅ **Numerical stability**: scipy-based stable softmax and log-sum-exp
✅ **GPU acceleration**: Full CUDA support with automatic mixed precision (AMP)
✅ **Reproducibility**: Seed management and configuration tracking
✅ **Professional logging**: Detailed experiment logs and progress tracking
✅ **PEP8 compliant**: Clean, readable, well-documented code

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- 16GB+ RAM (32GB recommended for large experiments)

### Option 1: pip install (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/lira-analysis.git
cd lira-analysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Development install with optional dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with notebook support
pip install -e ".[notebooks]"

# Install everything
pip install -e ".[dev,notebooks]"
```

---

## Quick Start

### 1. Train Shadow Models on CIFAR-10

```bash
python train.py --config configs/config_train_image.yaml
```

This will:
- Train 256 shadow models on CIFAR-10 (default configuration)
- Save models to `experiments/cifar10/resnet18/YYYY-MM-DD_HHMM/`
- Generate `keep_indices.npy` tracking training membership

### 2. Run LiRA Attack

```bash
python attack.py --config configs/config_attack.yaml \
    --override experiment.checkpoint_dir=experiments/cifar10/resnet18/YYYY-MM-DD_HHMM
```

This will:
- Generate logits for all shadow models
- Compute membership scores
- Run all attack variants (online, offline, global)
- Generate ROC curves and save metrics to CSV

### 3. Analyze Results

Results are saved in the experiment directory:
```
experiments/cifar10/resnet18/YYYY-MM-DD_HHMM/
├── roc_curve_single.pdf                  # ROC curves for all attacks
├── attack_results_single.csv             # Metrics (AUC, Acc, TPR@FPR)
├── attack_results_leave_one_out_summary.csv  # Leave-one-out results
├── train_test_stats.csv                  # Training/test loss and accuracy
└── model_0/, model_1/, ..., model_255/  # Shadow models and scores
```

---

## Usage

### Training Shadow Models

The training script supports both image and tabular datasets:

#### Image Datasets (CIFAR-10, CIFAR-100, GTSRB)

```bash
python train.py --config configs/config_train_image.yaml \
    --override dataset.name=cifar10 \
               training.num_shadow_models=256 \
               training.epochs=100 \
               model.architecture=resnet18
```

#### Tabular Datasets (Purchase-100)

First, download the Purchase-100 dataset following instructions in `data/Readme.md`, then:

```bash
python train.py --config configs/config_train_tabular.yaml \
    --override dataset.name=purchase \
               training.num_shadow_models=20 \
               training.epochs=20
```

**Key training parameters:**
- `training.num_shadow_models`: Number of shadow models (256 for images, 20 for tabular)
- `training.epochs`: Training epochs (100 for images, 20 for tabular)
- `training.batch_size`: Batch size (256 for images, 128 for tabular)
- `training.optimizer`: Optimizer choice (sgd, adam, adamw)
- `training.lr`: Learning rate (0.1 for SGD, 0.001 for Adam)
- `training.data_augmentations`: List of augmentations (e.g., [mixup, cutmix, random_crop])

### Running Attacks

#### Basic Attack

```bash
python attack.py --config configs/config_attack.yaml \
    --override experiment.checkpoint_dir=PATH_TO_EXPERIMENT
```

#### Advanced Attack Configuration

```bash
python attack.py --config configs/config_attack.yaml \
    --override experiment.checkpoint_dir=PATH_TO_EXPERIMENT \
               attack.evaluation_mode=both \
               attack.target_fprs=[0.00001,0.0001,0.001,0.01] \
               inference_data_augmentations.spatial_shift=2 \
               inference_data_augmentations.horizontal_flip=true
```

**Key attack parameters:**
- `attack.evaluation_mode`: 'single', 'leave_one_out', or 'both'
- `attack.ntest`: Number of target models (for single mode)
- `attack.target_fprs`: List of FPR thresholds for TPR computation
- `attack.target_model`: 'best' or specific epoch number
- `inference_data_augmentations.spatial_shift`: Pixels to shift (0-4)
- `inference_data_augmentations.horizontal_flip`: Enable horizontal flip

### Configuration

Configuration is done via YAML files with command-line overrides:

```bash
python train.py --config configs/config_train_image.yaml \
    --override dataset.name=cifar100 \
               model.architecture=resnet34 \
               training.epochs=150 \
               training.lr=0.05
```

#### Override Syntax

- Single level: `key=value`
- Nested keys: `key.subkey=value`
- Lists: `key=[value1,value2,value3]`
- Booleans: `key=true` or `key=false`
- Numbers: `key=123` or `key=0.001`

---

## Project Structure

```
lira_analysis/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project metadata and build configuration
├── train.py                     # Main training script
├── attack.py                    # Main attack script
│
├── configs/                     # Configuration files
│   ├── config_train_image.yaml      # Image dataset training config
│   ├── config_train_tabular.yaml    # Tabular dataset training config
│   ├── config_finetune.yaml         # Fine-tuning config
│   └── config_attack.yaml           # Attack config
│
├── utils/                       # Utility modules
│   ├── __init__.py                  # Package initialization
│   ├── utils.py                     # General utilities (logging, seeding, etc.)
│   ├── data_utils.py                # Dataset loading and preprocessing
│   ├── model_utils.py               # Model architectures and factory
│   └── train_utils.py               # Training functions and optimizers
│
├── attacks/                     # Attack implementations
│   ├── __init__.py                  # Package initialization
│   └── lira.py                      # LiRA attack implementation
│
├── analysis_results/            # Analysis scripts and notebooks
│   ├── threshold_dist.py            # Threshold distribution analysis
│   ├── loss_ratio_tpr.ipynb         # Loss ratio vs TPR analysis
│   ├── plot_benchmark+distribution.ipynb
│   ├── agreement.ipynb              # Attack agreement analysis
│   └── post_analysis.ipynb          # Post-hoc analysis
│
├── data/                        # Data directory (auto-populated)
│   ├── Readme.md                    # Data download instructions
│   ├── cifar10/                     # Auto-downloaded by torchvision
│   ├── cifar100/                    # Auto-downloaded by torchvision
│   ├── gtsrb/                       # Auto-downloaded by torchvision
│   └── purchase/                    # Manual download required
│
└── experiments/                 # Experiment outputs (auto-generated)
    └── {dataset}/{model}/{timestamp}/
        ├── train_config.yaml           # Saved training configuration
        ├── attack_config.yaml          # Saved attack configuration
        ├── train_log.log               # Training logs
        ├── attack_log.log              # Attack logs
        ├── keep_indices.npy            # Training membership indices
        ├── roc_curve_single.pdf        # ROC curves
        ├── attack_results_single.csv   # Attack metrics
        ├── membership_labels.npy       # Ground truth labels
        └── model_0/, model_1/, ...     # Shadow model directories
            ├── best_model.pth              # Best model checkpoint
            ├── checkpoint_epochN.pth       # Per-epoch checkpoints
            ├── logits/
            │   └── logits.npy              # Model logits [N, 1, A, C]
            └── scores/
                └── scores.npy              # Membership scores [N]
```

---

## Supported Datasets and Models

### Datasets

| Dataset | Type | Classes | Samples | Input Size | Auto-Download |
|---------|------|---------|---------|------------|---------------|
| CIFAR-10 | Image | 10 | 60,000 | 32×32 | ✅ Yes |
| CIFAR-100 | Image | 100 | 60,000 | 32×32 | ✅ Yes |
| GTSRB | Image | 43 | ~50,000 | 32×32 | ✅ Yes |
| Purchase-100 | Tabular | 100 | 197,324 | 600 | ❌ Manual |

**Note**: For Purchase-100 dataset, see `data/Readme.md` for download instructions.

### Models

#### Image Models
- **ResNet family**: resnet18, resnet34, resnet50, resnet101, resnet152
- **WideResNet**: Custom implementation (WideResNet-28-10)
- **200+ via TIMM**: efficientnet, vit, convnext, etc.

#### Tabular Models
- **FCN**: Fully Connected Network with configurable hidden layers

---

## Attack Variants

This implementation includes five attack variants:

### 1. LiRA (Online)
- **Description**: Uses both in-distribution and out-of-distribution statistics
- **Threat Model**: Attacker has access to shadow model training
- **Performance**: Best overall performance

### 2. LiRA (Online, Fixed Variance)
- **Description**: Online LiRA with global variance pooling
- **Advantage**: More stable with fewer shadow models
- **Performance**: Slightly lower than adaptive variance

### 3. LiRA (Offline)
- **Description**: Uses only out-of-distribution statistics
- **Threat Model**: More realistic, attacker only knows non-member distribution
- **Performance**: Lower than online variants but still effective

### 4. LiRA (Offline, Fixed Variance)
- **Description**: Offline LiRA with global variance
- **Advantage**: Most realistic threat model
- **Performance**: Baseline for realistic attacks

### 5. Global Threshold
- **Description**: Simple threshold on raw membership scores
- **Purpose**: Baseline comparison
- **Performance**: Weakest attack, useful as sanity check

---

## Evaluation Modes

### Single-Target Mode

Use the last `ntest` shadow models as targets:

```bash
python attack.py --config configs/config_attack.yaml \
    --override attack.evaluation_mode=single \
               attack.ntest=1
```

**Pros**: Fast, standard evaluation
**Cons**: Less robust estimates, no uncertainty quantification

### Leave-One-Out Mode

Each shadow model serves as target once:

```bash
python attack.py --config configs/config_attack.yaml \
    --override attack.evaluation_mode=leave_one_out
```

**Pros**: Robust estimates with mean ± std across targets
**Cons**: Slower (M evaluations for M shadow models)

### Both Modes

Run both evaluations:

```bash
python attack.py --config configs/config_attack.yaml \
    --override attack.evaluation_mode=both
```

---

## Results and Visualization

### Metrics Reported

For each attack variant, the following metrics are computed:

- **AUC**: Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
- **Accuracy**: Maximum accuracy across all thresholds
- **TPR@FPR**: True Positive Rate at specific False Positive Rates (e.g., 0.001%)
- **Precision**: Precision at specific FPR thresholds

### Output Files

1. **ROC Curves** (`roc_curve_single.pdf`, `roc_curve_leave_one_out.pdf`):
   - Log-log plot of FPR vs TPR
   - Separate curves for each attack variant
   - AUC or accuracy shown in legend

2. **Metrics Tables** (`attack_results_*.csv`):
   - Rows: Attack variants
   - Columns: AUC, Accuracy, TPR@various FPRs, Precision@various FPRs

3. **Training Statistics** (`train_test_stats.csv`):
   - Train/test loss and accuracy
   - Mean ± std across shadow models

4. **Threshold Information** (`threshold_info_leave_one_out.csv`):
   - Per-target, per-attack threshold analysis
   - Useful for understanding attack calibration

### Analysis Notebooks

The `analysis_results/` directory contains Jupyter notebooks for deeper analysis:

- `loss_ratio_tpr.ipynb`: Analyze relationship between loss ratios and TPR
- `plot_benchmark+distribution.ipynb`: Compare attacks and visualize score distributions
- `agreement.ipynb`: Analyze agreement between attack variants
- `post_analysis.ipynb`: Custom post-hoc analysis

---

## Advanced Usage

### Fine-tuning Pre-trained Models

```bash
python train.py --config configs/config_finetune.yaml \
    --override model.pretrained=true \
               model.architecture=resnet50 \
               training.epochs=10 \
               training.lr=0.0005
```

### Custom Data Augmentation

Edit `configs/config_train_image.yaml`:

```yaml
training:
  data_augmentations:
    - random_crop
    - random_flip
    - color_jitter
    - mixup          # alpha=0.2
    - cutmix         # alpha=1.0
    - cutout         # size=16
```

### Spatial Augmentation for Attacks

```bash
python attack.py --config configs/config_attack.yaml \
    --override inference_data_augmentations.spatial_shift=4 \
               inference_data_augmentations.spatial_stride=2 \
               inference_data_augmentations.horizontal_flip=true
```

This generates 2×(4×4+1) = 34 augmented views per sample.

### Using Custom Models

Add your model to `utils/model_utils.py`:

```python
def get_model(num_classes, architecture='resnet18', **kwargs):
    if architecture == 'my_custom_model':
        return MyCustomModel(num_classes=num_classes, **kwargs)
    # ... existing code
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{lira2022,
  title={Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions},
  author={Author Names},
  booktitle={Conference/Journal Name},
  year={2022}
}
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Follow PEP8 (enforced by `black` and `flake8`)
2. **Documentation**: Add docstrings for all functions and classes
3. **Testing**: Add tests for new features
4. **Pull Requests**: Submit PRs with clear descriptions

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black .

# Check code style
flake8 .

# Run tests (when available)
pytest
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Original LiRA paper and authors
- PyTorch and torchvision teams
- TIMM library for model zoo
- Open-source community

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/lira-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/lira-analysis/discussions)
- **Documentation**: See inline code documentation and notebooks

---

**Note**: This implementation is for research purposes. Membership inference attacks can reveal private information about training data. Always consider privacy implications and ethical guidelines when conducting such research.
