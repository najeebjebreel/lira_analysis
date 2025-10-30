# Analysis Results and Visualization

This directory contains Jupyter notebooks and scripts for analyzing LiRA attack results.

## Contents

### Jupyter Notebooks

#### `loss_ratio_tpr.ipynb`
**Purpose**: Analyze the relationship between loss ratios and True Positive Rate (TPR).

---

#### `plot_benchmark_distribution.ipynb`
**Purpose**: Compare attack performance and visualize score distributions

---

#### `agreement.ipynb`
**Purpose**: Analyze agreement between different attack variants


---

#### `post_analysis.ipynb`
**Purpose**: Advanced post-hoc analysis with two-threshold modes and per-sample vulnerability

**What it does**:
- **Two-threshold evaluation modes:**
  - **Target mode**: Threshold computed from target model's own ROC curve
  - **Shadow mode**: Threshold = median of other shadow models' target thresholds (more realistic)
- **Per-model metrics:** Computes confusion matrices (TP/FP/TN/FN) for each model and threshold
- **Precision at multiple priors:** Evaluates precision assuming different membership priors (1%, 10%, 50%)
- **Per-sample vulnerability analysis:**
  - Computes TP/FP/TN/FN for each sample across all leave-one-out models
  - Ranks samples by vulnerability (high TP, low FP = reliably detected when member)
  - Identifies highly vulnerable samples (FP=0, TP>0)
- **Visualization:**
  - Grid of most vulnerable samples with TP/FP annotations
  - Customizable sample visualization for image datasets
- **LaTeX table generation:**
  - Creates publication-ready tables comparing benchmarks
  - Includes reduction factors (×N) vs baseline

**Outputs generated:**
- `per_model_metrics_two_modes.csv`: Detailed per-model metrics
- `summary_statistics_two_modes.csv`: Aggregated statistics with mean ± std
- `samples_vulnerability_ranked_online_shadow_0p001pct.csv`: All samples ranked by vulnerability
- `samples_highly_vulnerable_online_shadow_0p001pct.csv`: Subset of highly vulnerable samples
- `top20_vulnerable_online_shadow_0p001pct.png`: Grid visualization of top 20 vulnerable samples

---

### Python Modules

The analysis code has been refactored into reusable modules:

#### `analysis_utils.py`
**Purpose**: Core utilities for loading and processing experiment data

**Key functions**:
- `load_experiment_config()`: Load train/attack configurations
- `load_attack_scores()`: Load all attack scores from experiment
- `load_membership_labels()`: Load ground truth labels
- `load_dataset_for_analysis()`: Load datasets for visualization
- `compute_per_sample_confusion_matrix()`: Compute TP/FP/TN/FN per sample
- `rank_samples_by_vulnerability()`: Rank samples by attack vulnerability
- `get_highly_vulnerable_samples()`: Filter highly vulnerable samples


#### `metrics.py`
**Purpose**: ROC curve computation and evaluation metrics

**Key functions**:
- `compute_roc_metrics()`: Compute full ROC curve, AUC, and optimal thresholds
- `compute_tpr_at_fpr()`: Find TPR at specific FPR values
- `compute_precision()`: Compute precision from TPR/FPR with prior
- `compute_confusion_matrix_at_threshold()`: Get confusion matrix at threshold
- `compute_median_and_rmad()`: Robust statistics for threshold distributions


#### `visualization.py`
**Purpose**: Publication-quality plotting functions

**Key functions**:
- `plot_roc_curves()`: Multi-attack ROC curve comparison
- `plot_score_distributions()`: Histogram of member vs non-member scores
- `plot_threshold_distribution_boxplot()`: Box plots of thresholds across models
- `plot_vulnerable_samples_grid()`: Grid visualization of vulnerable samples

---

### Python Scripts

#### `threshold_dist.py`
**Purpose**: Create box plots of threshold distributions across targets

**What it does**:
- Loads threshold information from leave-one-out evaluation
- Creates box plots showing threshold distribution for each attack variant
- Uses colorblind-safe Okabe-Ito palette
- Saves figures

---

#### `compare_attacks.py`
**Purpose**: Compare all attack variants for a single experiment

**What it does**:
- Loads all attack scores (online, offline, global)
- Computes ROC curves and metrics for each attack
- Generates side-by-side visualizations
- Creates comparison table with AUC, TPR@FPR, and precision

**Usage**:
```bash
python compare_attacks.py --experiment_dir PATH_TO_EXPERIMENT \
                          --target_fprs 0.001 0.01 0.1 \
                          --prior 0.5
```

**Outputs**:
- `attack_comparison_roc.pdf`: ROC curves
- `attack_comparison_scores.pdf`: Score distributions
- `attack_comparison_metrics.csv`: Performance metrics table

---

#### `vulnerability_analysis.py`
**Purpose**: Identify and visualize samples vulnerable to membership inference

**What it does**:
- Computes per-sample confusion matrix across all leave-one-out models
- Ranks samples by vulnerability (high TP, low FP)
- Identifies highly vulnerable samples (TP>0, FP=0)
- Creates grid visualization of most vulnerable samples (for image datasets)

**Usage**:
```bash
python vulnerability_analysis.py --experiment_dir PATH_TO_EXPERIMENT \
                                  --attack "LiRA (online)" \
                                  --threshold 0.0 \
                                  --k_samples 20
```

**Outputs**:
- `vulnerability_ranked_{attack}.csv`: All samples ranked by vulnerability
- `highly_vulnerable_{attack}.csv`: Subset meeting vulnerability criteria
- `top{k}_vulnerable_grid.png`: Visual grid of top vulnerable samples

---

## Running Notebooks

### Prerequisites

Install notebook dependencies:
```bash
pip install jupyter seaborn ipywidgets
```

### Launch Jupyter

```bash
jupyter notebook
```

Then navigate to the notebook you want to run.

### Expected Inputs

All notebooks expect experiment results in the standard format:

```
experiments/{dataset}/{model}/{timestamp}/
├── attack_results_single.csv
├── attack_results_leave_one_out_summary.csv
├── membership_labels.npy
├── online_scores_leave_one_out.npy
├── offline_scores_leave_one_out.npy
└── threshold_info_leave_one_out.csv
```

Update the `experiment_dir` variable in each notebook to point to your experiment directory.

---

## Common Analysis Tasks

### 1. Compare Attack Variants

**Quick**: Use `compare_attacks.py` script
```bash
python compare_attacks.py --experiment_dir PATH --target_fprs 0.001 0.01
```

**Interactive**: Use `plot_benchmark+distribution.ipynb`
1. Load results from multiple experiments
2. Plot ROC curves side-by-side
3. Create score distribution histograms
4. Generate comparison tables

### 2. Analyze Sample Vulnerability

**Quick**: Use `vulnerability_analysis.py` script
```bash
python vulnerability_analysis.py --experiment_dir PATH --attack "LiRA (online)"
```

**Interactive**: Use `post_analysis.ipynb` for detailed vulnerability analysis

### 3. Understand Sample Difficulty

Use `loss_ratio_tpr.ipynb`:
1. Identify "easy" vs "hard" samples
2. Analyze correlation with model confidence
3. Find outliers and edge cases

### 4. Evaluate Attack Agreement

Use `agreement.ipynb`:
1. Compute confusion matrices between attacks
2. Find samples where attacks disagree
3. Analyze complementary strengths

### 5. Threshold Analysis

**Quick**: Use `threshold_dist.py` script
```bash
python threshold_dist.py --experiment_dir PATH --target_fpr 0.001
```

**Purpose**:
1. Understand threshold variability across targets
2. Identify robust vs sensitive attacks
3. Calibrate attack parameters

---






