# Analysis Results and Visualization

This directory contains Jupyter notebooks and scripts for analyzing LiRA attack results.

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
**Purpose**: Custom post-hoc analysis template

**What it does**:
- Template for custom analyses
- Load experiment results and run additional experiments
- Compute custom metrics
- Generate custom visualizations

**When to use**: For exploratory analysis beyond standard metrics

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
python threshold_dist.py --experiment_dir PATH_TO_EXPERIMENT
```

**Outputs**:
- `threshold_distribution.pdf`: Box plots of thresholds
- `threshold_statistics.csv`: Summary statistics

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

Use `plot_benchmark+distribution.ipynb`:
1. Load results from multiple experiments
2. Plot ROC curves side-by-side
3. Create score distribution histograms
4. Generate comparison tables

### 2. Understand Sample Difficulty

Use `loss_ratio_tpr.ipynb`:
1. Identify "easy" vs "hard" samples
2. Analyze correlation with model confidence
3. Find outliers and edge cases

### 3. Evaluate Attack Agreement

Use `agreement.ipynb`:
1. Compute confusion matrices between attacks
2. Find samples where attacks disagree
3. Analyze complementary strengths

### 4. Threshold Analysis

Use `threshold_dist.py`:
1. Understand threshold variability across targets
2. Identify robust vs sensitive attacks
3. Calibrate attack parameters

---

## Adding Custom Analysis

To add your own analysis:

1. Copy `post_analysis.ipynb` to a new file
2. Update the experiment directory path
3. Load relevant data (scores, labels, metrics)
4. Implement your analysis
5. Generate visualizations and save results

### Example Template

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results
experiment_dir = "experiments/cifar10/resnet18/2024-01-01_0000"
membership_labels = np.load(f"{experiment_dir}/membership_labels.npy")
online_scores = np.load(f"{experiment_dir}/online_scores_leave_one_out.npy")

# Your custom analysis here
# ...

# Save results
plt.savefig(f"{experiment_dir}/custom_analysis.pdf")
results_df.to_csv(f"{experiment_dir}/custom_results.csv")
```

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
