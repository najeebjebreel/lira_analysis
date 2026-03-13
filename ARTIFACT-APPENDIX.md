# Artifact Appendix

Paper title: **Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions**

Requested Badges:
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

---

## Description

### Paper Reference

Najeeb Jebreel et al., "Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions", PoPETs 2026 (3).

### Artifact Summary

This artifact contains the full implementation of the experimental pipeline described in the paper.
It covers all five stages of the workflow: shadow-model training, LiRA attack (four variants plus a
global-threshold baseline), per-run post-analysis, cross-run reproducibility analysis, and
paper-figure generation.  All 10 paper benchmarks (Table 1) are manifest-driven and reproducible
from a single command.

### Security/Privacy Issues and Ethical Concerns

This artifact is purely academic research software.  It implements the LiRA membership inference attack
against ML models trained on public benchmark datasets (CIFAR-10, CIFAR-100, GTSRB, Purchase-100).
No real personal data is involved.  Running the code poses no security or privacy risk to the
reviewer's machine.

---

## Basic Requirements

### Hardware Requirements

**Minimum:** A CUDA-capable GPU (≥ 8 GB VRAM) is required for all image-dataset benchmarks.
The Purchase-100 tabular benchmarks can run on CPU but benefit from a GPU.

**Authors' machine (paper results):**
- GPU: NVIDIA GeForce RTX 4080, 16 GB VRAM
- OS: Ubuntu 20.04 LTS (WSL2 on Windows 11 Education)
- CUDA: 12.6 / Driver 560.94
- RAM: 16 GB
- CPU: Intel Core i7-12700 (10 cores / 20 threads)


### Software Requirements

| Component | Version used |
|---|---|
| OS | Ubuntu 20.04 LTS (WSL2) |
| Python | 3.11 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| numpy | 1.26.4 |
| scipy | 1.14.1 |
| pandas | 2.2.3 |
| seaborn | 0.13.2 |
| scikit-learn | 1.5.2 |
| timm | 1.0.15 |
| matplotlib | 3.9.2 |
| Pillow | 10.4.0 |
| pyyaml | 6.0.2 |
| tqdm | 4.67.1 |

All versions are pinned in `requirements-lock.txt`.  The full pinned environment is reproducible
via `environment.yml` (see [Set up the environment](#set-up-the-environment)).

The artifact uses a Conda environment with pinned
`requirements-lock.txt` as the build environment.  

### Estimated Time and Storage Consumption

**Overall human time:** ~30 minutes (environment setup + launching benchmark commands).

**Overall compute time:** varies by benchmark; see the table in
[Experiments](#experiments).  Full reproduction of all 10 benchmarks requires roughly 10–14 days
of sequential GPU compute on an RTX 4080.  Reviewers are encouraged to start with `purchase100_baseline` (≤ ~16 h), then
rerun the core image benchmark (e.g. `cifar10_baseline`, ~33–34 h) for deeper verification.
Actual time sometimes decreased by early stopping.

**Disk space:**
- Code + environment: ~2 GB
- One benchmark run (e.g. `cifar10_baseline`): ~10–15 GB
- All 10 benchmarks: ~100–150 GB

---

## Environment

### Accessibility

The artifact is publicly available at:

**https://github.com/najeebjebreel/lira_analysis**

The link above points to the `main` branch.  After the artifact evaluation is finalized, a stable
commit tag will be collected by the artifact chairs.

### Set up the Environment

```bash
# 1. Clone the repository
git clone https://github.com/najeebjebreel/lira_analysis.git
cd lira_analysis

# 2. Create the pinned Conda environment
conda env create -f environment.yml
conda activate lira-repro

# 3. Stage the Purchase-100 dataset (image datasets download automatically)
#    Download features_labels.npy from: https://drive.proton.me/urls/25C1HJ14S8#3uJjfOAAPblu
#    Then:
mkdir -p data/purchase
mv features_labels.npy data/purchase/
```

The `environment.yml` installs Python 3.11 and all packages from `requirements-lock.txt` in a
single step.  After activation, all `python *.py` invocations are available.

### Testing the Environment

Run the following to verify the environment is correctly set up:

```bash
# Dry-run: print the commands that would be executed, without training anything
python scripts/run_benchmark.py --benchmark purchase100_baseline --dry-run
```

Expected output: three `[run]` lines (train → attack → analysis) with no errors, e.g.:

```
[run] cwd=...
[run] ... python train.py --config .../configs/generated/purchase100_baseline.train.yaml
[run] cwd=...
[run] ... python attack.py --config .../configs/generated/purchase100_baseline.attack.yaml
[run] cwd=...
[run] ... python .../comprehensive_analysis/run_analysis.py --exp-path ...
```

For a deeper smoke test (trains 10 shadow models instead of 256, completes in ~5 minutes):

```bash
python train.py --config configs/train_tabular.yaml \
  --override seed=42 \
             training.start_shadow_model_idx=0 \
             training.end_shadow_model_idx=9 \
             experiment.checkpoint_dir=experiments/smoke_test
```

Expected output: 10 model checkpoints under `experiments/smoke_test/model_{0..9}/`.

---

## Artifact Evaluation

### Main Results and Claims

All results below are produced by the five-phase pipeline described in `README.md`.
The paper's claims are anchored in **Section 5** and the corresponding tables and figures.

#### Main Result 1: AOF and TL reduce LiRA success (Tables 3–6, Figure 8)

Under anti-overfitting (AOF) regularisation and transfer-learning (TL) fine-tuning, the true
positive rate (TPR) of all four LiRA variants drops substantially compared to the baseline.  This
holds across all datasets (CIFAR-10, CIFAR-100, GTSRB, Purchase-100).  The independent variable is
the training regime (baseline / AOF / TL); the dependent variable is TPR at fixed low FPR
thresholds (0.001% and 0.1%).

#### Main Result 2: Shadow-only threshold calibration degrades attack precision (Tables 7–10)

When thresholds are calibrated on shadow models only (the realistic threat model), TPR and positive
predictive value (PPV) under skewed priors (1%, 10% membership prior) are significantly lower than
under the optimistic leave-one-out setting.  The independent variable is the calibration mode
(target-and-shadow vs. shadow-only); the dependent variables are TPR and PPV at 0.001% and 0.1%
FPR.

#### Main Result 3: Thresholded vulnerable sets are unstable across runs (Figures 2, 3, 6)

The set of samples classified as vulnerable at low FPR (0.001%) changes substantially across
independent training runs: Jaccard similarity between pairs of runs is low and the intersection
shrinks rapidly.  Rankings of samples by LiRA score, however, are more stable.  The independent
variable is the number of runs compared; the dependent variables are Jaccard index, intersection
size, and union size.

#### Main Result 4: Loss ratio correlates with LiRA success (Figure 7)

Samples with a high loss ratio (train loss / test loss, averaged over shadow models) are more
consistently identified as vulnerable.  The independent variable is the loss ratio decile; the
dependent variable is mean online LiRA TPR.

### Paper-to-Artifact Map

The table below maps every paper item to the primary script and its expected output path.

| Paper item | Claim | Script | Expected output path |
|---|---|---|---|
| Table 1 | benchmark configurations | benchmark YAMLs | `configs/benchmarks/` |
| Table 2 | model utility and loss ratio | `train.py` | `experiments/<dataset>/<arch>/<run>/train_test_stats.csv`, `shadow_metrics_aggregate.csv` |
| Tables 3–6 | optimistic LiRA performance (baseline / AOF / TL) | `attack.py` | `experiments/<dataset>/<arch>/<run>/attack_results_leave_one_out_summary.csv` |
| Tables 7–10 | realistic shadow thresholds, skewed priors (0.001% FPR) | `run_analysis.py` | `analysis_results/<dataset>/<arch>/<run>/summary_statistics_two_modes.csv`, `summary_shadow_prior0p01.tex`, … |
| Table 11 | reproducibility of ranking-based sets (median gap) | `reproducibility_analysis.py` | `analysis_results/tables/topq_tail_table_median_gap_rank_scores.tex` |
| Table 12 | realistic shadow thresholds, skewed priors (0.1% FPR) | `run_analysis.py` | same `.tex` outputs as Tables 7–10 (0.1% FPR columns) |
| Table 13 *(appendix)* | reproducibility of ranking-based sets (mean gap + FPR-thresholded) | `reproducibility_analysis.py` | `analysis_results/tables/topq_tail_table_mean_gap_rank_scores.tex` |
| Table 14 *(appendix)* | ablation: WideResNet-28-2 on CIFAR-10 | `run_analysis.py` | `analysis_results/cifar10/wrn28-2/<run>/summary_shadow_prior*.tex` |
| Table 15 *(appendix)* | CIFAR-10 under TL (EfficientNetV2) | `run_analysis.py` | `analysis_results/cifar10/<efficientnetv2>/<run>/summary_shadow_prior*.tex` *(requires `cifar10_tl` benchmark)* |
| Figure 1 | threshold variability at 0.001% vs 0.1% FPR | `threshold_distribution.py` | `analysis_results/figures/thresh_boxplot_single.pdf`, `thresh_boxplot_multi.pdf` |
| Figure 2 | reproducibility, stability, coverage — 0.001% FPR (TP≥1) | `reproducibility_analysis.py` | `analysis_results/figures/jaccard_noleg_0p001pct.pdf`, `intersection_0p001pct.pdf`, `union_0p001pct.pdf` |
| Figure 3 | zero-FP reproducibility heatmap — 0.001% FPR | `reproducibility_analysis.py` | `analysis_results/figures/tpgeq_x_0fp_identical_heatmaps_0p001pct.pdf` |
| Figure 4 | top-9 vulnerable samples across runs and training variations | `run_analysis.py`, `compose_top_vulnerable.py` | `analysis_results/figures/topk_vulnerable_images/<dataset>/<arch>/<run>/top9_vulnerable_online_shadow_0p001pct.png` *(assembled manually)* |
| Figure 5 | rank displacement under run-to-run variability | `reproducibility_analysis.py` | `analysis_results/figures/inside_run_displacement_simple/` |
| Figure 6 | reproducibility, stability, coverage — 0.1% FPR (TP≥1) | `reproducibility_analysis.py` | `analysis_results/figures/jaccard_noleg_0p1pct.pdf`, `intersection_0p1pct.pdf`, `union_0p1pct.pdf` |
| Figure 7 | loss ratio vs online LiRA TPR correlation | `loss_ratio_tpr.py` | `analysis_results/figures/lossratio_tpr.pdf` |
| Figure 8 | in/out score and ratio distributions across benchmarks | `plot_benchmark_distribution.py` | `analysis_results/figures/sample_inout_score.pdf`, `sample_inout_ratio.pdf` |
| Figure 9 *(appendix)* | all-positives reproducibility heatmap — 0.001% FPR | `reproducibility_analysis.py` | `analysis_results/figures/tpgeq_x_identical_heatmaps_0p001pct.pdf` |
| Figure 10 *(appendix)* | all-positives reproducibility heatmap — 0.1% FPR | `reproducibility_analysis.py` | `analysis_results/figures/tpgeq_x_identical_heatmaps_0p1pct.pdf` |
| Figure 11 *(appendix)* | top-16 vulnerable samples: 3 seeds × 2 FPR settings | `run_analysis.py --top-k 16 --nrow 4` (per seed), then assemble manually | `analysis_results/cifar10/resnet18/<run>/top16_vulnerable_online_shadow_{0p001pct,0p1pct}.png` |
| Figure 12 *(appendix)* | zero-FP reproducibility heatmap — 0.1% FPR | `reproducibility_analysis.py` | `analysis_results/figures/tpgeq_x_0fp_identical_heatmaps_0p1pct.pdf` |
| Figure 13 *(appendix)* | distribution of median gap vulnerability scores | `reproducibility_analysis.py` | `analysis_results/figures/bin_boxplot_scores_median_gap_rank_scores.pdf` |
| Figure 14 *(appendix)* | top-1 vulnerable sample median gap score across runs | `reproducibility_analysis.py` | `analysis_results/figures/top1_across_runs/top1_across_runs_boxplots_median_gap_rank_scores.pdf` |
| Figure 15 *(appendix)* | top-1 sample identity across runs | `reproducibility_analysis.py` | `analysis_results/figures/top1_across_runs/top1_samples_grid_median_gap_rank_scores.pdf` |

### Experiments

#### Experiment 1: Quick Check — `purchase100_baseline`

- **Time:** ~5 human-minutes + **≤ ~16 compute-hours** (RTX 4080)
- **Storage:** ~2 GB

This is the **recommended entry point** for reviewers with limited time or compute.
It runs the full end-to-end pipeline (train → attack → analysis) for the Purchase-100 baseline FCN
benchmark (tabular data, 1 augmentation, up to 100 epochs × 256 shadow models).

```bash
python scripts/run_benchmark.py --benchmark purchase100_baseline
```

Expected outputs under `experiments/purchase/fcn/purchase100_baseline/`:
- `shadow_metrics_aggregate.csv` — per-model train/test accuracy and loss ratio
- `attack_results_leave_one_out_summary.csv` — AUC, TPR@0.001% FPR, TPR@0.1% FPR → **Tables 3, 5**
- `membership_labels.npy`, `online_scores_leave_one_out.npy`, and other score arrays

Expected outputs under `analysis_results/purchase/fcn/purchase100_baseline/`:
- `summary_statistics_two_modes.csv` — TPR/PPV under both calibration modes and all three priors → **Tables 7, 9**
- `samples_vulnerability_ranked_*.csv` — per-sample vulnerability scores

**Expected numerical values** (from paper Tables 6 and 10 — Purchase-100 Baseline):

*Table 6 — optimistic (leave-one-out) thresholds, π = 50%:*

| Attack | AUC (%) | TPR @ 0.001% FPR (%) | TPR @ 0.1% FPR (%) |
|---|---|---|---|
| LiRA (online) | 70.16 ± 0.29 | 0.523 ± 0.243 | 4.491 ± 0.281 |
| LiRA (online, fixed var) | 69.52 ± 0.28 | 0.180 ± 0.110 | 3.089 ± 0.188 |
| LiRA (offline) | 55.11 ± 0.48 | 0.007 ± 0.007 | 0.500 ± 0.077 |
| LiRA (offline, fixed var) | 56.11 ± 0.51 | 0.022 ± 0.017 | 0.713 ± 0.078 |
| Global threshold | 54.83 ± 0.15 | 0.001 ± 0.001 | 0.100 ± 0.015 |

*Table 10 — realistic (shadow-only) thresholds at 0.001% FPR:*

| Attack | TPR' (%) | FPR' (%) | PPV @ π=1% | PPV @ π=10% | PPV @ π=50% |
|---|---|---|---|---|---|
| LiRA (online) | 0.516 ± 0.047 | 0.001 ± 0.001 | 89.93 ± 11.15 | 98.84 ± 1.37 | 99.87 ± 0.16 |
| LiRA (online, fixed var) | 0.159 ± 0.015 | 0.001 ± 0.001 | 78.87 ± 22.27 | 96.70 ± 3.80 | 99.61 ± 0.47 |
| LiRA (offline) | 0.004 ± 0.002 | 0.001 ± 0.001 | 55.82 ± 48.27 | 66.20 ± 37.65 | 87.37 ± 16.56 |
| LiRA (offline, fixed var) | 0.018 ± 0.004 | 0.001 ± 0.001 | 57.27 ± 43.60 | 80.46 ± 21.53 | 96.32 ± 4.79 |


#### Experiment 2: Core CIFAR-10 Benchmark — `cifar10_baseline`

- **Time:** ~10 human-minutes + **~33–34 compute-hours** (RTX 4080; ~29 h training ≈ 7 min/model, plus 15–17% attack overhead with 18 augmentations)
- **Storage:** ~10–15 GB

This reproduces the primary benchmark used throughout the paper.  It covers Main Results 1–4 for
the CIFAR-10 / ResNet-18 / baseline setting, and its outputs feed into the cross-run reproducibility
figures (Experiments 3 and 4) when run under multiple seeds.

```bash
python scripts/run_benchmark.py --benchmark cifar10_baseline
```

The benchmark manifest sets `experiment_dir` to `experiments/cifar10/resnet18/cifar10_baseline`.
For a custom named run (e.g. when running multiple seeds), use:

```bash
python train.py --config configs/train_image.yaml \
  --override seed=1 experiment.run_name=seed1
python attack.py --config configs/attack.yaml \
  --override experiment.checkpoint_dir=experiments/cifar10/resnet18/seed1
python comprehensive_analysis/run_analysis.py \
  --exp-path experiments/cifar10/resnet18/seed1 \
  --target-fprs 0.00001 0.001 --priors 0.01 0.1 0.5
```

Key outputs to verify against the paper (Tables 3–4, 7–8):
- `experiments/cifar10/resnet18/seed1/attack_results_leave_one_out_summary.csv` → Tables 3–4
- `analysis_results/cifar10/resnet18/seed1/summary_statistics_two_modes.csv` → Tables 7–8
- `experiments/cifar10/resnet18/seed1/roc_curve_single.pdf` → visual sanity check

**Expected numerical values** (from paper Tables 3 and 7 — CIFAR-10 Baseline):

*Table 3 — optimistic (leave-one-out) thresholds, π = 50%:*

| Attack | AUC (%) | TPR @ 0.001% FPR (%) | TPR @ 0.1% FPR (%) |
|---|---|---|---|
| LiRA (online) | 76.48 ± 0.32 | 3.956 ± 1.061 | 10.268 ± 0.555 |
| LiRA (online, fixed var) | 76.28 ± 0.31 | 2.876 ± 1.064 | 9.135 ± 0.508 |
| LiRA (offline) | 55.58 ± 0.92 | 0.762 ± 0.348 | 3.262 ± 0.338 |
| LiRA (offline, fixed var) | 56.64 ± 0.89 | 0.948 ± 0.526 | 4.540 ± 0.424 |
| Global threshold | 59.97 ± 0.32 | 0.003 ± 0.004 | 0.097 ± 0.027 |

*Table 7 — realistic (shadow-only) thresholds at 0.001% FPR:*

| Attack | TPR' (%) | FPR' (%) | PPV @ π=1% | PPV @ π=10% | PPV @ π=50% |
|---|---|---|---|---|---|
| LiRA (online) | 3.990 ± 0.161 | 0.002 ± 0.003 | 94.73 ± 6.10 | 99.46 ± 0.65 | 99.94 ± 0.07 |
| LiRA (online, fixed var) | 2.912 ± 0.142 | 0.002 ± 0.003 | 93.10 ± 8.03 | 99.26 ± 0.91 | 99.92 ± 0.10 |
| LiRA (offline) | 0.713 ± 0.052 | 0.002 ± 0.003 | 81.31 ± 20.20 | 97.24 ± 3.33 | 99.67 ± 0.40 |
| LiRA (offline, fixed var) | 0.918 ± 0.068 | 0.003 ± 0.005 | 81.13 ± 21.29 | 97.03 ± 4.06 | 99.64 ± 0.52 |

Key qualitative claims to verify: **(i) LiRA significantly degrades with AOF/AOF+TL** (core claim, Table 3); 
**(ii) PPV drops sharply with AOF/TL + shadow-based threshold calibration + skewed priors** ( Table 7).

---

**Simplified option — Mini CIFAR-10 (~7–9 h on RTX 4080):**

For reviewers who cannot afford the full 34–39 h run, use 64 shadow models instead of 256.
This validates the full pipeline end-to-end and qualitatively reproduces the attack ordering,
but the numerical values (especially TPR at very low FPR) will differ from the paper:

```bash
python train.py --config configs/train_image.yaml \
  --override seed=1 experiment.run_name=seed1_mini \
             training.end_shadow_model_idx=63
python attack.py --config configs/attack.yaml \
  --override experiment.checkpoint_dir=experiments/cifar10/resnet18/seed1_mini
python comprehensive_analysis/run_analysis.py \
  --exp-path experiments/cifar10/resnet18/seed1_mini \
  --target-fprs 0.00001 0.001 --priors 0.01 0.1 0.5
```

#### Experiment 3: Reproducibility and Rank Stability (Figures 2, 3, 5, 6)

The script auto-discovers all Phase 3 outputs under `--analysis-root` — any number of runs is
supported.  Choose the tier that fits your available compute:

---

**Option A — Partial rerun with 3–4 seeds (~100–160 compute-hours on RTX 4080)**

Run Experiment 2 for 3–4 seeds, then feed those results into the analysis.
This is enough to verify the trend (Jaccard decreasing with k, intersection shrinking).

```bash
# Train + attack + per-run analysis for seeds 1–4
for i in 1 2 3 4; do
  python train.py --config configs/train_image.yaml \
    --override seed=${i} experiment.run_name=seed${i}
  python attack.py --config configs/attack.yaml \
    --override experiment.checkpoint_dir=experiments/cifar10/resnet18/seed${i}
  python comprehensive_analysis/run_analysis.py \
    --exp-path experiments/cifar10/resnet18/seed${i} \
    --target-fprs 0.00001 0.001 --priors 0.01 0.1 0.5
done

# Reproducibility analysis over the runs
python comprehensive_analysis/reproducibility_analysis.py \
  --analysis-root analysis_results/cifar10/resnet18 \
  --skip-rank
```

The `--skip-rank` flag omits the rank-stability section, which requires experiment
directories with raw score files (not just analysis outputs).

---

**Option B — Full paper reproduction (12 seeds + 4 variants, ~500+ compute-hours)**

Run all 12 seeds of `cifar10_baseline` plus the four training-variation runs
(`bs512_drp0.2`, `mixup_drp0.15`, `tl`, `wrn28-2/seed42`), then:

```bash
python comprehensive_analysis/reproducibility_analysis.py
```

---

Expected outputs:

`analysis_results/figures/`:
- `jaccard_noleg_0p001pct.pdf`, `intersection_0p001pct.pdf`, `union_0p001pct.pdf` → **Figure 2**
- `tpgeq_x_0fp_identical_heatmaps_0p001pct.pdf` → **Figure 3**
- `jaccard_noleg_0p1pct.pdf`, `intersection_0p1pct.pdf`, `union_0p1pct.pdf` → **Figure 6**
- `tpgeq_x_identical_heatmaps_0p001pct.pdf` → **Figure 9** *(appendix)*
- `tpgeq_x_identical_heatmaps_0p1pct.pdf` → **Figure 10** *(appendix)*
- `tpgeq_x_0fp_identical_heatmaps_0p1pct.pdf` → **Figure 12** *(appendix)*

`analysis_results/tables/`:
- `reproducibility_<N>runs_<M>variants_0p001pct.csv` — Jaccard/intersection/union (N=12, M=4 for full paper set)

`analysis_results/figures/` (rank-stability additions):
- `inside_run_displacement_simple/` → **Figure 5**
- `bin_boxplot_scores_median_gap_rank_scores.pdf` → **Figure 13** *(appendix)*
- `top1_across_runs/top1_across_runs_boxplots_median_gap_rank_scores.pdf` → **Figure 14** *(appendix)*
- `top1_across_runs/top1_samples_grid_median_gap_rank_scores.pdf` → **Figure 15** *(appendix)*

`analysis_results/tables/` (rank-stability additions):
- `topq_tail_table_median_gap_rank_scores.tex` → **Table 11** *(appendix)*
- `topq_tail_table_mean_gap_rank_scores.tex` → **Table 13** *(appendix)*

Verify generated PDFs and CSVs against the values reported in the paper.

#### Experiment 4: Paper Figure Scripts (Figures 1, 4, 7, 8)

- **Time:** ~5 human-minutes + < 15 compute-minutes (requires Phase 3 outputs to be present)

All scripts read Phase 3/4 outputs and write PDFs to `analysis_results/figures/`.

**Figure 1** — threshold variability boxplots:
```bash
python comprehensive_analysis/threshold_distribution.py
```
Expected: `analysis_results/figures/thresh_boxplot_single.pdf`, `thresh_boxplot_multi.pdf`

**Figure 4** — collect per-run top-vulnerable grids (requires Phase 3 grids; panel assembled manually):
```bash
python comprehensive_analysis/compose_top_vulnerable.py
```
Expected: `analysis_results/figures/topk_vulnerable_images/<dataset>/<arch>/<run>/top9_vulnerable_*.png`

**Figure 7** — loss ratio vs TPR correlation:
```bash
python comprehensive_analysis/loss_ratio_tpr.py
```
Expected: `analysis_results/figures/lossratio_tpr.pdf`

**Figure 8** — per-benchmark score and ratio distributions:
```bash
python comprehensive_analysis/plot_benchmark_distribution.py \
  --config configs/figure_panels/figure8_scores.yaml
python comprehensive_analysis/plot_benchmark_distribution.py \
  --config configs/figure_panels/figure8_ratios.yaml
```
Expected: `analysis_results/figures/sample_inout_score.pdf`, `sample_inout_ratio.pdf`

**Figure 11** *(appendix)* — top-16 vulnerable samples: 3 seeds × 2 FPR settings (manual assembly):

```bash
# Regenerate top-16 grids for seeds 1–3
for i in 1 2 3; do
  python comprehensive_analysis/run_analysis.py \
    --exp-path experiments/cifar10/resnet18/seed${i} \
    --target-fprs 0.00001 0.001 --top-k 16 --nrow 4
done
```

This produces six PNG grids:
- `analysis_results/cifar10/resnet18/seed{1,2,3}/top16_vulnerable_online_shadow_0p001pct.png`
- `analysis_results/cifar10/resnet18/seed{1,2,3}/top16_vulnerable_online_shadow_0p1pct.png`

Arrange them in a 2 × 3 layout (rows = FPR threshold, columns = seed) using any image editor or
LaTeX `\includegraphics` to reproduce Figure 11 as shown in the paper.

Verify generated PDFs against the figures in the paper.

---

### Results Not Reproducible During Evaluation

The following paper items cannot be reproduced within a typical artifact evaluation timeframe or require steps beyond automated pipeline execution.  All are documented with the reason.

| Paper item | Reproducible? | Reason |
|---|---|---|
| Tables 3–10, Figures 2–7 (CIFAR-10 baseline) | **Yes**, with sufficient compute | One seed: ~33–34 h; mini (64 models): ~8–9 h |
| Tables 3–10 (Purchase-100) | **Yes** | ≤ ~16 h end-to-end (upper bound; reduced by early stopping) |
| Tables 3–6, Figures 2–3 (CIFAR-100, GTSRB) | Partial | ~38–39 h per CIFAR-100 benchmark; omit if time-constrained |
| Figure 8 (all-benchmark score distributions) | Partial | Requires all 10 benchmarks; run a subset to get a partial panel |
| Figure 11 (top-16 grid, 3 seeds × 2 FPR) | Partial — **manual step** | Grids generated by script; final layout requires manual image arrangement |
| Table 15 (CIFAR-10 TL, EfficientNetV2) | Not in quick path | Requires `cifar10_tl` benchmark (~8–13 h) |
| Figures 2, 3, 5, 6 (full reproducibility, 12 seeds) | Partial | 3–4 seeds show the trend; full 12-seed set needs ~500+ compute-hours |

---

## Limitations

- **Full-paper rerun is GPU-intensive.** Reproducing all 10 benchmarks from scratch requires
  roughly 10–14 days of sequential compute on an RTX 4080.  The quick-check path
  (`purchase100_baseline`, ~16 h; `cifar10_baseline`, ~33–34 h) is sufficient to verify the
  pipeline end-to-end and cross-check the paper's primary numerical claims.

- **Purchase-100 requires manual dataset staging.** The dataset file must be downloaded from the
  link in `data/Readme.md` and placed under `data/purchase/` before running any Purchase benchmark.
  All image datasets (CIFAR-10, CIFAR-100, GTSRB) download automatically on first use.

- **Figure 8 requires multiple benchmarks.** The panel is composed from all 10 benchmarks
  (see `configs/figure_panels/`).  Reviewers can regenerate it by running the benchmarks listed
  in the figure-panel YAML configs, or run only a subset to get a partial figure.

- **Figure 11 (appendix) requires a top-k 16 rerun and manual assembly.**  The default
  `run_analysis.py` generates `top9_*` grids.  Reproducing Figure 11 requires rerunning Phase 3
  for seeds 1–3 with `--top-k 16 --nrow 4`, then arranging the six resulting PNG files in a
  2 × 3 grid (rows = FPR threshold, columns = seed) manually or via LaTeX.

- **Table 15 (appendix) requires the `cifar10_tl` benchmark.** Table 15 reports CIFAR-10 results
  under TL (EfficientNetV2).  Reproducing it requires running `cifar10_tl` (~10–18 h) and then
  `run_analysis.py` on the resulting experiment directory.

- **Windows / WSL2 environment.** The artifact was developed and measured on WSL2 / Ubuntu 20.04.
  It should work on any Linux system with CUDA, but has not been tested on macOS or native Windows.

---

## Notes on Reusability

The pipeline is fully config-driven and designed for extension:

- **New datasets:** Add a dataset loader in `utils/data_utils.py` and a benchmark manifest YAML
  under `configs/benchmarks/`.
- **New architectures:** Register the model in `utils/model_utils.py`; no other changes are needed.
- **New attack variants:** Add a scorer in `attacks/lira.py`; the post-analysis scripts pick up
  any score file matching the naming convention in `OUTPUTS.md`.
- **Partial reruns:** The `--stages`, `--skip-existing`, and `--override` flags on
  `python scripts/run_benchmark.py` allow fine-grained control over which pipeline stages are (re-)executed.
- **Sharded training:** The `training.start_shadow_model_idx` / `training.end_shadow_model_idx`
  overrides allow shadow-model training to be distributed across multiple machines and merged
  afterward.

The benchmark manifest system (`configs/benchmarks/`) is a lightweight, self-contained contract
format that can be adapted to replicate the evaluation protocol of this paper on entirely new
settings.
