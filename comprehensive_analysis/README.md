# Comprehensive Analysis

This directory contains Jupyter notebooks for analyzing LiRA attack results and reproducing paper figures.

---

## Notebooks

### Primary Analysis

#### `comprehensive_analysis.ipynb` ⭐ **Start Here**

Main analysis pipeline for LiRA evaluation.

**Features:**
- Per-model metrics (AUC, TPR@FPR, precision at multiple priors)
- Two-threshold evaluation:
  - **Target mode**: Model uses its own threshold (upper bound)
  - **Shadow mode**: Model uses median threshold from others (realistic)
- Per-sample vulnerability ranking (FP=0, high TP = most vulnerable)
- Visualization of top vulnerable samples

**Inputs:**
- `membership_labels.npy`: Ground truth [M × N]
- `*_scores_leave_one_out.npy`: Attack scores for each variant

**Outputs** (saved to `analysis_results/{dataset}/{model}/{config}/`):
```
per_model_metrics_two_modes.csv          # Detailed per-model results
summary_statistics_two_modes.csv         # Aggregated mean ± std
samples_vulnerability_ranked_*.csv       # All samples ranked by vulnerability
samples_highly_vulnerable_*.csv          # Subset with FP=0, TP>0
top20_vulnerable_*.png                   # Visualization
```

**Usage:**
```python
# Edit Config class in notebook
EXP_PATH = "experiments/cifar10/resnet18/2025-01-15_1200"
TARGET_FPRS = [0.00001, 0.001]
PRIORS = [0.01, 0.1, 0.5]
```

---

### Secondary Analysis (Optional)

#### `threshold_distribution.ipynb`
Visualizes threshold stability across models. Requires output from `comprehensive_analysis.ipynb`.

**Plots:**
- Threshold distributions per attack variant
- Target vs shadow threshold comparison
- Multi-run stability analysis

---

#### `reproducibility.ipynb`
Analyzes detection consistency across multiple runs (reproduces paper Fig. 2).

**Metrics:**
- Jaccard index (agreement)
- Intersection size (always detected)
- Union size (ever detected)

**Requires:** Multiple runs with `samples_vulnerability_ranked_*.csv` from each.

---

#### `loss_ratio_tpr.ipynb`
Analyzes relationship between loss ratios and TPR (paper Fig. 5).

**Requires:** Custom CSV with aggregated loss and TPR statistics.

---

#### `plot_benchmark_distribution.ipynb`
Compares score distributions across training configurations (paper Fig. 6).

**Usage:** Specify sample ID to visualize across baseline/AOF/TL benchmarks.

---

## Quick Start

```bash
# 1. Run primary analysis
jupyter notebook comprehensive_analysis.ipynb

# 2. Optional: threshold analysis
jupyter notebook threshold_distribution.ipynb

# 3. Optional: reproducibility (requires multiple runs)
jupyter notebook reproducibility.ipynb
```

---

## Key Concepts

### Threshold Modes
- **Target**: Optimal from model's own ROC (assumes perfect knowledge)
- **Shadow**: Median from other models (realistic transferability)

### Vulnerability Ranking
Samples sorted by `(FP ↑, TP ↓)`:
- Most vulnerable: FP=0 (never false alarm), high TP (reliably detected)
- Least vulnerable: High FP (often false alarm), low TP (rarely detected)

### Membership Priors
Precision = (prior × TPR) / (prior × TPR + (1-prior) × FPR)
- Prior=0.5: Balanced dataset
- Prior=0.01: Realistic scenario (1% members)
- **Always report multiple priors** for realistic interpretation

---

## Output Structure

```
analysis_results/
└── {dataset}/              # cifar10, cifar100, gtsrb, purchase
    └── {model}/            # resnet18, wrn28-2, etc.
        └── {config}/       # weak_aug, strong_aug, etc.
            ├── per_model_metrics_two_modes.csv
            ├── summary_statistics_two_modes.csv
            ├── samples_vulnerability_ranked_online_shadow_0p001pct.csv
            ├── samples_highly_vulnerable_online_shadow_0p001pct.csv
            └── top20_vulnerable_online_shadow_0p001pct.png
```

---

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision pyyaml
```

Or use the main package installation:
```bash
pip install .[notebooks]
```
