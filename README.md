
# Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a **reproducible implementation** of our paper
***“Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions”***
(currently **under review** at a peer-reviewed conference).

It re-evaluates the Likelihood Ratio Attack (LiRA) under **practical training and attack assumptions**, with analyses for **shadow-only threshold calibration**, **skewed membership priors**, and **per-sample reproducibility** of membership inferences.

---

## Overview

**Membership Inference Attacks (MIAs)** test whether a data point was used to train a model. **LiRA** is a strong black-box MIA when many shadow models are available (e.g., M=256). Prior evaluations often **overestimated risk** by (i) attacking overconfident models, (ii) calibrating thresholds on target data, (iii) assuming balanced priors, and (iv) overlooking per-sample reproducibility.

This repo follows a **realistic evaluation protocol**:

* Targets trained with **Anti-Overfitting (AOF)** and, where applicable, **Transfer Learning (TL)** to reduce loss-wise overconfidence while preserving accuracy.
* **Thresholds calibrated only on shadow models**; **PPV** evaluated under skewed priors (e.g., π ≤ 10%).
* **Reproducibility** quantified across seeds, hyperparameters, and architectures.

**Key findings**:

* AOF and TL substantially **reduce LiRA success** while maintaining (often improving) utility.
* Under shadow-calibrated thresholds and skewed priors, **PPV drops** and varies across targets.
* **Per-sample positives are unstable** across runs; support-qualified reporting is recommended.

---

## Installation

**Prerequisites**

* Python ≥ 3.8
* CUDA-capable GPU (recommended)
* 16 GB+ RAM (32 GB recommended for large grids)

```bash
git clone https://github.com/najeebjebreel/lira-analysis.git
cd lira-analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

1. **Train shadow models (example: CIFAR-10)**

   ```bash
   python train.py --config configs/config_train_image.yaml
   ```

   This trains **256** shadow models by default and saves artifacts under:
   `experiments/cifar10/resnet18/YYYY-MM-DD_HHMM/`

2. **Run LiRA attack**

   ```bash
   python attack.py --config configs/config_attack.yaml \
     --override experiment.checkpoint_dir=experiments/cifar10/resnet18/YYYY-MM-DD_HHMM
   ```

   This generates logits/scores, runs online/offline/global variants, and writes metrics/curves.

3. **Analyze results**
   Outputs are saved to the experiment directory, e.g.:

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
├── analysis_results/         # Notebooks & scripts for figures
│   ├── threshold_dist.py
│   ├── loss_ratio_tpr.ipynb
│   ├── plot_benchmark+distribution.ipynb
│   ├── agreement.ipynb
│   └── post_analysis.ipynb
└── experiments/              # Auto-generated outputs
```

---

## Datasets & Models

| Dataset              | Type    | Classes  | Samples | Models                    |
| -------------------- | ------- | -------- | ------- | ------------------------- |
| CIFAR-10 / CIFAR-100 | Image   | 10 / 100 | 60,000  | ResNet-18, WideResNet, TL |
| GTSRB                | Image   | 43       | ~51,000 | ResNet-18, TL             |
| Purchase-100         | Tabular | 100      | 197,324 | FCN                       |

---

## Attack Variants

1. **LiRA (Online)** — IN/OUT modeling; strongest with many shadows
2. **LiRA (Online, Fixed Variance)** — global variance; more stable with fewer shadows
3. **LiRA (Offline)** — OUT-only; lighter but weaker
4. **LiRA (Offline, Fixed Variance)**
5. **Global Threshold** — no shadows; baseline sanity check

---

## Evaluation Modes

* **Single-Target** — quick smoke tests
* **Leave-One-Out** — each shadow acts once as “target” (default; aggregated mean ± std)
* Metrics: **AUC**, **TPR at ultra-low FPRs** (e.g., 0.001%, 0.1%), and **PPV under priors** π ∈ {1%, 10%, 50%}

Thresholds and PPV use **shadow-only calibration** (achieved FPR/TPR, not nominal).

---

## Results & Visualization

Use the notebooks and scripts in `analysis_results/` to reproduce paper figures and tables:

| File                                | Purpose                                      |
| ----------------------------------- | -------------------------------------------- |
| `threshold_dist.py`                 | Threshold variability & boxplots             |
| `agreement.ipynb`                   | Per-sample reproducibility (Jaccard overlap) |
| `loss_ratio_tpr.ipynb`              | Loss-ratio vs attack success correlation     |
| `plot_benchmark+distribution.ipynb` | Benchmark & score-distribution plots         |
| `post_analysis.ipynb`               | Aggregated paper figures and summaries       |

---

## Citation

If you use this code in your research, please cite our work:

```bibtex
@inproceedings{Jebreel2025LiRARealistic,
  title     = {Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions},
  author    = {Najeeb Jebreel and Mona Khalil and David S{\'a}nchez and Josep Domingo-Ferrer},
  booktitle = {Under Review},
  year      = {2025},
}
```



---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## Acknowledgments

We thank the original LiRA authors and the open-source community (PyTorch, TIMM, Hydra, etc.).
Use this code **for research and educational purposes only** and follow applicable ethical guidelines and policies.
