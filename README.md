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

## Installation

**Prerequisites**
- Python ≥ 3.8  
- CUDA-capable GPU (recommended)  
- 16 GB+ RAM (32 GB recommended for large grids)

```bash
git clone https://github.com/najeebjebreel/lira-analysis.git
cd lira-analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

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

3. **Analyze Results**

   ```
   experiments/cifar10/resnet18/YYYY-MM-DD_HHMM/
   ├── roc_curve_single.pdf
   ├── attack_results_single.csv
   ├── attack_results_leave_one_out_summary.csv
   ├── train_test_stats.csv
   └── model_0/, model_1/, ...
   ```

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
│   ├── plot_benchmark+distribution.ipynb
│   ├── agreement.ipynb
│   └── post_analysis.ipynb
└── experiments/              # Auto-generated outputs
```

---

## Datasets & Models

| Dataset              | Type    | Classes  | Samples | Models                                      |
| -------------------- | ------- | -------- | ------- | --------------------------------------------|
| CIFAR-10 / CIFAR-100 | Image   | 10 / 100 | 60,000  | ResNet-18, WideResNet, EfficientNet-V2 (TL) |
| GTSRB                | Image   | 43       | ~51,000 | ResNet-18, TL                               |
| Purchase-100         | Tabular | 100      | 197,324 | FCN                                         |

---

## Attack Variants

1. **LiRA (Online)** — in/out modeling; strongest with many shadows.
2. **LiRA (Online, Fixed Variance)** — global variance; more stable for small shadow sets.
3. **LiRA (Offline)** — out-only; realistic but weaker.
4. **LiRA (Offline, Fixed Variance)** — simplest offline baseline.
5. **Global Threshold** — single fixed threshold; sanity baseline.

---

## Evaluation Modes

* **Single-Target** — fast single-model evaluation.
* **Leave-One-Out (default)** — each shadow model serves once as a target; reports mean ± std.
* Metrics: **AUC**, **TPR @ low FPRs (0.001–0.1%)**, **PPV** under priors π ∈ {1%, 10%, 50%}.
* Thresholds and PPV use **shadow-only calibration** (realistic evaluation).

---

## Analysis and Visualization

All post-attack analyses are in [`analysis_results/`](analysis_results/), which includes reusable modules, standalone scripts, and interactive notebooks.

| File                                | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| `threshold_dist.py`                 | Threshold distributions & boxplots        |
| `compare_attacks.py`                | Multi-attack ROC & metric comparison      |
| `vulnerability_analysis.py`         | Identify and visualize vulnerable samples |
| `loss_ratio_tpr.ipynb`              | Loss-ratio vs TPR correlation             |
| `plot_benchmark+distribution.ipynb` | Score distributions & benchmark plots     |
| `agreement.ipynb`                   | Cross-attack agreement analysis           |
| `post_analysis.ipynb`               | Full paper figure and table generation    |

Each analysis script produces publication-quality figures and LaTeX-ready tables (AUC, TPR, PPV, thresholds).
For details, see [`analysis_results/README.md`](analysis_results/README.md).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Jebreel2025LiRARealistic,
  title     = {Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions},
  author    = {Najeeb Jebreel and Mona Khalil and David S{\'a}nchez and Josep Domingo-Ferrer},
  booktitle = {Under Review},
  year      = {2025}
}
```

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## Acknowledgments

We thank the original LiRA authors and the open-source community (PyTorch, TIMM, Hydra, etc.).
Please use this code **for research and educational purposes only** and follow appropriate ethical and privacy guidelines.
