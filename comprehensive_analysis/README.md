# Comprehensive Analysis

This directory contains the post-processing layer for the LiRA paper artifact.
All scripts are run from the repo root unless noted otherwise.

## Main Scripts

### `run_analysis.py` — Phase 3: per-run post-analysis

Computes two-mode (target / shadow) TPR@FPR, PPV at multiple priors,
per-sample vulnerability counts, and image grids for one experiment directory.

Outputs under `analysis_results/{dataset}/{model}/{run_name}/`:
- `per_model_metrics_two_modes.csv`
- `summary_statistics_two_modes.csv`
- `samples_vulnerability_ranked_online_shadow_<tag>.csv`
- `samples_highly_vulnerable_online_shadow_<tag>.csv`
- `top9_vulnerable_online_shadow_<tag>.png`

```bash
lira analyze --exp-path experiments/cifar10/resnet18/seed1 \
             --target-fprs 0.00001 0.001 \
             --priors 0.01 0.1 0.5

# Skip image grid generation on headless machines
python comprehensive_analysis/run_analysis.py \
  --exp-path experiments/cifar10/resnet18/seed1 \
  --skip-visualization

# Custom output root
python comprehensive_analysis/run_analysis.py \
  --exp-path experiments/cifar10/resnet18/seed1 \
  --out-root /tmp/analysis_out
```

---

### `reproducibility_analysis.py` — Phase 4: cross-run analysis

Requires Phase 3 outputs from all 12 baseline seeds and 4 variant runs.
Generates Jaccard / intersection / union panels (Figures 2, 3, 5, 6),
TP-support heatmaps, TP-support-at-0FP heatmaps, rank-stability figures,
and LaTeX tables.

Outputs: `analysis_results/figures/`, `analysis_results/tables/`.

```bash
# Full run (uses default paths)
python comprehensive_analysis/reproducibility_analysis.py

# Custom roots
python comprehensive_analysis/reproducibility_analysis.py \
  --analysis-root comprehensive_analysis/analysis_results/cifar10/resnet18 \
  --arch-analysis-root comprehensive_analysis/analysis_results/cifar10/wrn28-2/seed42

# Skip individual sections
python comprehensive_analysis/reproducibility_analysis.py --skip-rank
python comprehensive_analysis/reproducibility_analysis.py --skip-heatmaps
python comprehensive_analysis/reproducibility_analysis.py --skip-threshold-panels
```

---

### `threshold_distribution.py` — Figure 1

Generates threshold distribution boxplots from `per_model_metrics_two_modes.csv`.

```bash
python comprehensive_analysis/threshold_distribution.py
```

---

### `compose_top_vulnerable.py` — Figure 4 (collect)

Collects all per-run `top*_vulnerable*.png` files from `analysis_results/` into
`analysis_results/figures/topk_vulnerable_images/`, preserving the `<dataset>/<arch>/<run>/`
subfolder hierarchy. The paper panel (Figure 4) is then assembled manually from these images.

```bash
python comprehensive_analysis/compose_top_vulnerable.py
```

Override destination:

```bash
python comprehensive_analysis/compose_top_vulnerable.py --dest /path/to/output/dir
```

---

### `loss_ratio_tpr.py` — Figure 7

Plots loss ratio vs TPR@FPR scatter given two lists of values collected
from per-run summaries across experiments. Assemble `analysis_results/loss_ratio.csv`
(columns: `Loss Ratio`, `TPR`) manually before running.

```bash
python comprehensive_analysis/loss_ratio_tpr.py
```

---

### `plot_benchmark_distribution.py` — Figure 8

Generates per-sample in/out score distribution panels across benchmark settings.
Requires running the benchmarks listed in the panel manifests under `configs/figure_panels/`.
Output is written to `analysis_results/figures/`.

```bash
# Using recorded manifests
python comprehensive_analysis/plot_benchmark_distribution.py \
  --config configs/figure_panels/figure8_scores.yaml
python comprehensive_analysis/plot_benchmark_distribution.py \
  --config configs/figure_panels/figure8_ratios.yaml

# Manual panel specification
python comprehensive_analysis/plot_benchmark_distribution.py \
  --panel "Baseline=/path/to/baseline_run" \
  --panel "AOF=/path/to/aof_run" \
  --sample-idx 21 \
  --score-file global_scores_leave_one_out.npy \
  --out figures/sample_inout_score.pdf
```

---

## Supporting Modules

- `analysis_utils.py` — loading, thresholding, vulnerability analysis, set operations
- `rank_stability_utils.py` — rank-stability implementation used by `reproducibility_analysis.py`
- `metrics.py` — ROC, threshold selection, confusion matrices, PPV
- `plot_style.py` — common paper plot styling

---

## Typical Workflow

```bash
# Phase 3: analyze each seed run
for i in $(seq 1 12); do
  python comprehensive_analysis/run_analysis.py \
    --exp-path "experiments/cifar10/resnet18/seed${i}"
done

# Phase 4: cross-run analysis (Figures 2, 3, 5, 6)
python comprehensive_analysis/reproducibility_analysis.py

# Remaining figures
python comprehensive_analysis/threshold_distribution.py   # Figure 1
python comprehensive_analysis/compose_top_vulnerable.py   # Figure 4 (collect grids; assemble manually)
python comprehensive_analysis/loss_ratio_tpr.py           # Figure 7
```
