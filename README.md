# Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a **reproducible implementation** of our paper  
***“Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions”***  
(currently **under review** at a peer-reviewed conference).

It re-evaluates the Likelihood Ratio Attack (LiRA) under **practical training and attack assumptions**, introducing analyses for **realistic threshold calibration**, **skewed membership priors**, and **per-sample reproducibility** of membership inference outcomes.

---

## Overview

**Membership Inference Attacks (MIAs)** test whether a data point was used to train a model.  
**LiRA** is a strong black-box MIA when many shadow models are available (e.g., M=256).  
However, prior work often **overestimated attack success** by:
- Evaluating overconfident models,
- Calibrating thresholds on target data,
- Assuming balanced membership priors,
- Ignoring reproducibility across runs.

This implementation provides a **realistic and reproducible evaluation** of LiRA with:

- **Anti-Overfitting (AOF)** and **Transfer Learning (TL)** target models,  
  preserving accuracy while reducing overconfidence.  
- **Shadow-only threshold calibration** and **precision (PPV)** under realistic priors (π ≤ 10%).  
- **Per-sample reproducibility analysis** across architectures, seeds, and configurations.  

**Key findings:**
- AOF and TL significantly **reduce LiRA’s success** while maintaining model utility.  
- **Shadow-calibrated thresholds** and **skewed priors** substantially lower PPV.  
- **Membership predictions are unstable per sample**; reproducibility requires support-qualified reporting.


---

## 🧩 Installation

### 🌟 Recommended: Using Miniconda

1. **Install Miniconda**
   Download and install from the official site:
   👉 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. **Clone the repository**

   ```bash
   git clone https://github.com/najeebjebreel/lira_analysis.git
   cd lira_analysis
   ```

3. **Create and activate the environment**

   ```bash
   conda create -n lira python=3.11
   conda activate lira
   ```

4. **Install LiRA Analysis**

   ```bash
   pip install .
   ```

   > 💡 To include optional dependencies for development or notebooks:
   >
   > ```bash
   > pip install .[dev]
   > pip install .[notebooks]
   > ```

---


## Quick Start

1. **Train Shadow Models (example: CIFAR-10)**

   ```bash
   python train.py --config configs/config_train_image.yaml
   ```

   Trains **256** shadow models and saves artifacts under:
   `experiments/cifar10/resnet18/YYYY-MM-DD_HHMM/`

2. **Run LiRA Attack**

   ```bash
   python attack.py --config configs/config_attack.yaml \
     --override experiment.checkpoint_dir=experiments/cifar10/resnet18/YYYY-MM-DD_HHMM
   ```

   Runs online, offline, and global variants; saves metrics and ROC curves.

3. **Train and Attack Results** — see [OUTPUTS.md](./OUTPUTS.md)

---

## Project Structure

```
lira_analysis/
├── train.py                  # Model training
├── attack.py                 # LiRA evaluation (online/offline)
├── configs/                  # YAML configs (training / attack)
├── attacks/                  # LiRA implementations
├── utils/                    # Helpers (I/O, logging, models, seeding, etc.)
├── analysis_results/         # Analysis notebooks & scripts
│   ├── threshold_dist.py
│   ├── compare_attacks.py
│   ├── vulnerability_analysis.py
│   ├── loss_ratio_tpr.ipynb
│   ├── plot_benchmark_distribution.ipynb
│   ├── agreement.ipynb
│   └── post_analysis.ipynb
└── experiments/              # Auto-generated outputs
```

---

## Datasets & Models

| Dataset              | Type    | Classes  | Samples | Models                                      |
| -------------------- | ------- | -------- | ------- | --------------------------------------------|
| CIFAR-10 / CIFAR-100 | Image   | 10 / 100 | 60,000  | ResNet-18, WideResNet, EfficientNet-V2 (TL) |
| GTSRB                | Image   | 43       | ~51,000 | ResNet-18, , EfficientNet-V2 (TL)           |
| Purchase-100         | Tabular | 100      | 197,324 | FCN                                         |

---
 - All image datasets will be downloaded automatically.
 - Purchase can be downloaded to this folder via this link https://drive.proton.me/urls/25C1HJ14S8#3uJjfOAAPblu. Please put it inside a folder named ''purchase'' in the ''data'' folder.
---

## Attack Variants

1. **LiRA (Online)** — in/out modeling; strongest with many shadows.
2. **LiRA (Online, Fixed Variance)** — global variance; more stable for small shadow sets.
3. **LiRA (Offline)** — out-only; realistic but weaker.
4. **LiRA (Offline, Fixed Variance)** — simplest offline baseline.
5. **Global Threshold** — single fixed threshold; sanity baseline.

---

## Analysis and Visualization

All post-attack analyses are in [`analysis_results/`](analysis_results/), which reproduces our main rresults, and includes code scripts and interactive notebooks.

| File                                | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| `threshold_dist.py`                 | Threshold distributions & boxplots        |
| `compare_attacks.py`                | Multi-attack ROC & metric comparison      |
| `vulnerability_analysis.py`         | Identify and visualize vulnerable samples |
| `loss_ratio_tpr.ipynb`              | Loss-ratio vs TPR correlation             |
| `plot_benchmark+distribution.ipynb` | Score distributions & benchmark plots     |
| `agreement.ipynb`                   | Cross-attack agreement analysis           |
| `post_analysis.ipynb`               | Full paper figure and table generation    |

For details, see [`analysis_results/README.md`](analysis_results/README.md).

---

## Citation


---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## Acknowledgments

We thank the original LiRA authors and the open-source community (PyTorch, TIMM, etc.).
