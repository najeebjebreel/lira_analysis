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

**Membership Inference Attacks (MIAs)** aim to infer whether a data point was used to train a model.  
**LiRA** is widely regarded as the state-of-the-art black-box MIA when many shadow models are available (e.g., M=256).  
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

All post-attack comprehensive analyses are in [`comprehensive_analysis/`](comprehensive_analysis/), which reproduces our main rresults, and includes code scripts and interactive notebooks.

For details, see [`comprehensive_analysis/README.md`](comprehensive_analysis/README.md).

---



## Example Results: 📊 CIFAR-10 under Target vs. Shadow Calibration

*Target FPR = 0.001 %*

| **Benchmark**                              | **Attack**   |      **TPR′ (%)** |  **FPR′ (%)** |  **PPV @ π = 1%** | **PPV @ π = 10%** | **PPV @ π = 50%** |
| :----------------------------------------- | :----------- | ----------------: | ------------: | ----------------: | ----------------: | ----------------: |
| ***Target-based thresholds (optimistic)*** |              |                   |               |                   |                   |                   |
| **Baseline**                               | Online       |     3.956 ± 1.061 | 0.000 ± 0.000 |     100.00 ± 0.00 |     100.00 ± 0.00 |     100.00 ± 0.00 |
|                                            | Online (FV)  |     2.876 ± 1.064 | 0.000 ± 0.000 |     100.00 ± 0.00 |     100.00 ± 0.00 |     100.00 ± 0.00 |
|                                            | Offline      |     0.762 ± 0.348 | 0.000 ± 0.000 |     100.00 ± 0.00 |     100.00 ± 0.00 |     100.00 ± 0.00 |
|                                            | Offline (FV) |     0.948 ± 0.526 | 0.000 ± 0.000 |     100.00 ± 0.00 |     100.00 ± 0.00 |     100.00 ± 0.00 |
| ***Shadow-based thresholds (realistic)***  |              |                   |               |                   |                   |                   |
| **Baseline**                               | Online       |     3.990 ± 0.161 | 0.002 ± 0.003 |      94.73 ± 6.10 |      99.46 ± 0.65 |      99.94 ± 0.07 |
|                                            | Online (FV)  |     2.912 ± 0.142 | 0.002 ± 0.003 |      93.10 ± 8.03 |      99.26 ± 0.91 |      99.92 ± 0.10 |
|                                            | Offline      |     0.713 ± 0.052 | 0.002 ± 0.003 |     81.31 ± 20.20 |      97.24 ± 3.33 |      99.67 ± 0.40 |
|                                            | Offline (FV) |     0.918 ± 0.068 | 0.003 ± 0.005 |     81.13 ± 21.29 |      97.03 ± 4.06 |      99.64 ± 0.52 |
| **AOF**                                    | Online       |     0.224 ± 0.482 | 0.033 ± 0.466 |     66.52 ± 34.88 |     90.93 ± 12.18 |      98.42 ± 5.25 |
|                                            | Online (FV)  |     0.636 ± 0.101 | 0.002 ± 0.003 |     80.13 ± 21.52 |      96.69 ± 6.53 |      99.46 ± 2.99 |
|                                            | Offline      |     0.290 ± 4.134 | 0.262 ± 4.138 |     55.17 ± 46.40 |     73.31 ± 28.84 |      93.37 ± 9.32 |
|                                            | Offline (FV) |     0.310 ± 1.179 | 0.077 ± 1.192 |     67.96 ± 34.53 |     91.18 ± 12.71 |      98.36 ± 6.23 |
| **AOF + TL**                               | Online       |     0.084 ± 0.048 | 0.017 ± 0.045 |     49.13 ± 44.90 |     70.75 ± 30.40 |     91.49 ± 12.35 |
|                                            | Online (FV)  |     0.084 ± 0.021 | 0.002 ± 0.003 |     59.25 ± 42.01 |     83.54 ± 18.38 |      97.22 ± 3.42 |
|                                            | Offline      |     0.027 ± 0.085 | 0.026 ± 0.085 |     32.73 ± 46.50 |     37.30 ± 43.64 |     56.17 ± 36.15 |
|                                            | Offline (FV) |     0.044 ± 0.089 | 0.033 ± 0.089 |     42.39 ± 48.08 |     52.67 ± 40.53 |     78.43 ± 21.99 |

**Notes:**

* *FV = Fixed Variance variant*
* *AOF = Anti-Overfitting training*
* *TL = Transfer Learning*
* Values are **mean ± standard deviation** across 5 seeds.
* *Target-based calibration* assumes perfect knowledge of the target model (**optimistic**).
* *Shadow-based calibration* represents realistic, deployable attack conditions (**realistic**).

---

### 🧩 Figure 1 — Reproducibility of LiRA Membership Inferences

![Reproducibility, stability, and coverage vs seeds, training variations, and runs (TP≥1)](figures/reproducibility_cifar10.png)

**Caption:**
*as the number of combined runs increases, the intersection of vulnerable samples (those identified in all runs) shrinks
sharply, while the union (samples identified in any run) expands rapidly.*

---


## Citation


---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## Acknowledgments

We thank the original LiRA authors and the open-source community (PyTorch, TIMM, etc.).

This work was partly supported by the Government of Catalonia (ICREA Acad`emia Prizes to D. S´anchez and J. Domingo-Ferrer, and grant 2021SGR-00115), MCIN/AEI under grant PID2024-157271NB-I00 “CLEARING-IT”, and the EU’s NextGenerationEU/PRTR via INCIBE (project “HERMES” and INCIBE-URV cybersecurity chair).
