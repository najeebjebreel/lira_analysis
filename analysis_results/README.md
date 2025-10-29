# Analysis Results and Visualization

This directory contains Jupyter notebooks and scripts for analyzing LiRA attack results.

## Structure

The analysis code is organized into **reusable Python modules** and **standalone scripts**:

- **Modules** (`analysis_utils.py`, `metrics.py`, `visualization.py`, `latex_utils.py`):
  - Reusable functions for common analysis tasks
  - Import these in your own scripts and notebooks
  - Well-documented with docstrings and type hints

- **Scripts** (`threshold_dist.py`, `compare_attacks.py`, `vulnerability_analysis.py`):
  - Ready-to-run analysis pipelines
  - Command-line interfaces with argparse
  - Use these for quick analysis without writing code

- **Notebooks** (`post_analysis.ipynb`, `plot_benchmark+distribution.ipynb`, etc.):
  - Interactive exploration and visualization
  - Can leverage the reusable modules
  - Good for custom analysis and paper figures

## Contents

### Jupyter Notebooks

#### `loss_ratio_tpr.ipynb`
**Purpose**: Analyze the relationship between loss ratios and True Positive Rate (TPR)

**What it does**:
- Loads membership scores and ground truth labels
- Computes loss ratios for members vs non-members
- Plots TPR as a function of loss ratio
- Helps understand which samples are easier/harder to attack

**When to use**: After running attacks, to understand attack behavior on different samples

---

#### `plot_benchmark+distribution.ipynb`
**Purpose**: Compare attack performance and visualize score distributions

**What it does**:
- Loads results from multiple attack variants
- Creates benchmark comparison plots
- Visualizes score distributions for members vs non-members
- Generates publication-ready figures

**When to use**: For final analysis and paper figures

---

#### `agreement.ipynb`
**Purpose**: Analyze agreement between different attack variants

**What it does**:
- Compares predictions across attack variants (online, offline, global)
- Computes agreement matrices and correlation scores
- Identifies samples where attacks disagree (interesting edge cases)
- Visualizes agreement patterns

**When to use**: To understand how different attacks complement each other

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
  - Formatted for academic papers

**Outputs generated:**
- `per_model_metrics_two_modes.csv`: Detailed per-model metrics
- `summary_statistics_two_modes.csv`: Aggregated statistics with mean ± std
- `samples_vulnerability_ranked_online_shadow_0p001pct.csv`: All samples ranked by vulnerability
- `samples_highly_vulnerable_online_shadow_0p001pct.csv`: Subset of highly vulnerable samples
- `top20_vulnerable_online_shadow_0p001pct.png`: Grid visualization of top 20 vulnerable samples
- LaTeX tables (printed to output, ready to copy into paper)

**When to use**:
- For paper submissions requiring detailed metrics
- To understand which samples are most vulnerable to membership inference
- To compare attack performance across different configurations
- To evaluate precision under realistic prior assumptions
- To generate publication-ready figures and tables

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

**Usage example**:
```python
from analysis_results.analysis_utils import load_attack_scores, load_membership_labels

scores_dict = load_attack_scores('experiments/cifar10/resnet18/...')
labels = load_membership_labels('experiments/cifar10/resnet18/...')
```

#### `metrics.py`
**Purpose**: ROC curve computation and evaluation metrics

**Key functions**:
- `compute_roc_metrics()`: Compute full ROC curve, AUC, and optimal thresholds
- `compute_tpr_at_fpr()`: Find TPR at specific FPR values
- `compute_precision()`: Compute precision from TPR/FPR with prior
- `compute_confusion_matrix_at_threshold()`: Get confusion matrix at threshold
- `compute_median_and_rmad()`: Robust statistics for threshold distributions

**Usage example**:
```python
from analysis_results.metrics import compute_roc_metrics, compute_tpr_at_fpr

fpr, tpr, thresholds, auc, _ = compute_roc_metrics(scores, labels)
tpr_val, threshold = compute_tpr_at_fpr(fpr, tpr, thresholds, target_fpr=0.01)
```

#### `visualization.py`
**Purpose**: Publication-quality plotting functions

**Key functions**:
- `plot_roc_curves()`: Multi-attack ROC curve comparison
- `plot_score_distributions()`: Histogram of member vs non-member scores
- `plot_threshold_distribution_boxplot()`: Box plots of thresholds across models
- `plot_vulnerable_samples_grid()`: Grid visualization of vulnerable samples
- `setup_paper_style()`: Configure matplotlib for publication

**Color palette**: Colorblind-safe Okabe-Ito palette

**Usage example**:
```python
from analysis_results.visualization import plot_roc_curves, setup_paper_style

setup_paper_style()
roc_data = {'LiRA (online)': (fpr, tpr, auc)}
fig = plot_roc_curves(roc_data, save_path='roc_curves.pdf')
```

#### `latex_utils.py`
**Purpose**: LaTeX table generation for research papers

**Key functions**:
- `format_mean_std()`: Format values as "mean ± std" for LaTeX
- `format_multiplier()`: Format reduction factors as "(×5.2)"
- `create_metrics_table()`: Generate metrics table for single benchmark
- `create_comparison_table()`: Generate multi-benchmark comparison table
- `save_latex_table()`: Save table to .tex file

**Usage example**:
```python
from analysis_results.latex_utils import create_metrics_table

latex_str = create_metrics_table(
    results_df=df,
    attacks=['LiRA (online)', 'LiRA (offline)'],
    target_fprs=[0.001, 0.01],
    caption='Attack performance on CIFAR-10',
    label='tab:results'
)
print(latex_str)  # Ready to copy into paper
```

---

### Python Scripts

#### `threshold_dist.py`
**Purpose**: Create box plots of threshold distributions across targets

**What it does**:
- Loads threshold information from leave-one-out evaluation
- Creates box plots showing threshold distribution for each attack variant
- Uses colorblind-safe Okabe-Ito palette
- Saves publication-ready figures

**Usage**:
```bash
python threshold_dist.py --experiment_dir PATH_TO_EXPERIMENT \
                         --target_fpr 0.001 \
                         --output_dir custom_output
```

**Outputs**:
- `threshold_distribution.pdf`: Box plots of thresholds
- `threshold_statistics.csv`: Summary statistics

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

## Adding Custom Analysis

To add your own analysis, you can leverage the reusable modules:

### Option 1: Using the Modules (Recommended)

```python
import sys
sys.path.append('..')  # If running from analysis_results directory

from analysis_results.analysis_utils import load_attack_scores, load_membership_labels
from analysis_results.metrics import compute_roc_metrics, compute_tpr_at_fpr
from analysis_results.visualization import plot_roc_curves, setup_paper_style
import matplotlib.pyplot as plt

# Setup publication-quality plots
setup_paper_style()

# Load experiment results using utility functions
experiment_dir = "experiments/cifar10/resnet18/2024-01-01_0000"
scores_dict = load_attack_scores(experiment_dir, mode='leave_one_out')
labels = load_membership_labels(experiment_dir)

# Compute metrics using reusable functions
fpr, tpr, thresholds, auc, _ = compute_roc_metrics(
    scores_dict['LiRA (online)'].flatten(),
    labels.flatten()
)

# Visualize using plotting functions
roc_data = {'LiRA (online)': (fpr, tpr, auc)}
fig = plot_roc_curves(roc_data, save_path=f"{experiment_dir}/custom_roc.pdf")

print(f"AUC: {auc:.4f}")
```

### Option 2: Manual Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results manually
experiment_dir = "experiments/cifar10/resnet18/2024-01-01_0000"
membership_labels = np.load(f"{experiment_dir}/membership_labels.npy")
online_scores = np.load(f"{experiment_dir}/online_scores_leave_one_out.npy")

# Your custom analysis here
# ...

# Save results
plt.savefig(f"{experiment_dir}/custom_analysis.pdf")
results_df.to_csv(f"{experiment_dir}/custom_results.csv")
```

### Creating Standalone Analysis Scripts

Follow the pattern in `compare_attacks.py` or `vulnerability_analysis.py`:

1. Import from analysis modules
2. Add argparse for command-line interface
3. Implement main analysis function
4. Save outputs with descriptive names
5. Add docstrings and usage examples

---

## Visualization Best Practices

### Color Schemes

Use colorblind-safe palettes:
- **Okabe-Ito**: Default in `threshold_dist.py`
- **Viridis**: Good for continuous data
- **Seaborn colorblind**: Safe categorical palette

### Figure Formats

- **PDF**: Vector graphics for papers (recommended)
- **PNG**: Raster graphics for presentations (use high DPI, e.g., 300)
- **SVG**: Editable vector graphics

### Font Compatibility

All scripts set `matplotlib.rcParams['pdf.fonttype'] = 42` for PDF font embedding compatibility.

---

## Troubleshooting

### Import Errors

If you get import errors in notebooks:
```python
import sys
sys.path.append('..')  # Add parent directory to path
from utils.utils import *
from attacks.lira import LiRA
```

### Memory Issues

For large experiments (256+ shadow models):
- Load data incrementally
- Use `np.load(..., mmap_mode='r')` for memory mapping
- Clear variables after use: `del variable; gc.collect()`

### Missing Data

Ensure you've run both training and attack scripts before analysis:
1. `python train.py --config CONFIG`
2. `python attack.py --config CONFIG`

---

## Citation

If you use these analysis scripts in your research, please cite the main paper (see main README.md).

---

## Contributing

To contribute new analysis notebooks or scripts:

1. Follow the naming convention: `descriptive_name.ipynb` or `descriptive_name.py`
2. Add clear documentation at the top of the notebook/script
3. Include example usage and expected outputs
4. Update this README with your additions
5. Submit a pull request

---

## Support

For issues with analysis scripts:
- Check main README.md for general setup
- Ensure all dependencies are installed
- Verify experiment results are in the expected format
- Open an issue on GitHub if problems persist
