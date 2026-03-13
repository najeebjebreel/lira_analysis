# Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions

This repository provides the **official implementation** of the paper
**["Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions"](https://arxiv.org/abs/2603.07567)**
(Jebreel, Khalil, Sánchez, Domingo-Ferrer — PoPETs 2026).

We show that prior work often overestimated attack success by using overconfident models,
target-data threshold calibration, and balanced membership priors.

## Overview

**Key Contributions:**
- Evaluation under **anti-overfitting** (AoF) and **transfer-learning** (TL) settings
- **Shadow-only threshold calibration** and precision under skewed priors (π ≤ 10%)
- **Per-sample reproducibility analysis** across architectures and configurations
- Comprehensive implementation of 5 LiRA variants across 4 datasets and 10 benchmarks

**Main Findings:**
- AoF and TL defenses reduce LiRA success while maintaining model utility
- Shadow-calibrated thresholds and realistic priors substantially lower attack precision
- Membership predictions are unstable; reproducibility requires support thresholding


## Installation

Recommended environment:

```bash
conda create -n lira python=3.11
conda activate lira
pip install .
```


Pinned reproduction environment:

```bash
conda env create -f environment.yml
conda activate lira-repro
```

Notes:
- Python 3.10+ is required.
- The default configs favor deterministic reruns for paper reproduction.

Optional dev extras:

```bash
pip install .[dev]
```


## Pipeline

The pipeline has five phases. All commands are run from the repo root.
The full command reference is in [cli_commands.txt](./cli_commands.txt).

### Phase 1 — Train shadow models

```bash
python train.py --config configs/train_image.yaml
# or fine-tune from ImageNet pretrained weights:
python train.py --config configs/finetune.yaml
```

Use `experiment.run_name` to give the experiment a readable directory name instead of a
timestamp — this is the recommended convention for all named runs:

```bash
python train.py --config configs/train_image.yaml \
  --override seed=1 experiment.run_name=seed1
```

Output: `experiments/<dataset>/<arch>/<run_name>/model_0/ … model_255/`,
`train_test_stats.csv` (per-model train/test accuracy → **Table 2**),
`shadow_metrics_aggregate.csv` (aggregate loss ratio → **Table 2**).

If `experiment.run_name` is not set, the directory falls back to a timestamp
(`experiments/<dataset>/<arch>/<timestamp>/`).

Shard across machines by splitting the model index range (all shards must share
the same `keep_indices.npy` from the first shard):

```bash
python train.py --config configs/train_image.yaml \
  --override seed=1 experiment.run_name=seed1 \
             training.start_shadow_model_idx=0 \
             training.end_shadow_model_idx=63
```

### Phase 2 — LiRA attack

```bash
python attack.py --config configs/attack.yaml

# Point to a specific run on the fly
python attack.py --config configs/attack.yaml \
  --override experiment.checkpoint_dir=experiments/cifar10/resnet18/seed1

# Skip recomputing cached logits or scores
python attack.py --config configs/attack.yaml \
  --override attack.compute_logits=false \
             attack.compute_scores=false
```

Output inside `experiments/<dataset>/<arch>/<run>/`:
- `online_scores_leave_one_out.npy`, `offline_scores_leave_one_out.npy`, … — raw attack scores
- `membership_labels.npy` — ground-truth member/non-member labels
- `roc_curve_single.pdf` — ROC curve visual sanity check
- `attack_results_leave_one_out_summary.csv` — AUC, TPR@FPR per attack → **Tables 3–6**

### Phase 3 — Per-run post-analysis

```bash
python comprehensive_analysis/run_analysis.py \
  --exp-path experiments/cifar10/resnet18/seed1 \
  --target-fprs 0.00001 0.001 \
  --priors 0.01 0.1 0.5

# Skip image grid generation on headless machines
python comprehensive_analysis/run_analysis.py \
  --exp-path experiments/cifar10/resnet18/seed1 \
  --skip-visualization
```

Output: `analysis_results/<dataset>/<arch>/<run>/`
- `summary_statistics_two_modes.csv` + `summary_<mode>_prior<p>.tex` — TPR/PPV under shadow thresholds and skewed priors → **Tables 7–10**
- `per_model_metrics_two_modes.csv` — full per-model detail
- `samples_vulnerability_ranked_*.csv`, `samples_highly_vulnerable_*.csv` — per-sample vulnerability
- `top9_vulnerable_online_shadow_*.png` — top-9 vulnerable sample grids → **Figure 4** (assembled in Phase 5)

### Phase 4 — Cross-run reproducibility and rank stability

The script auto-discovers all completed Phase 3 runs under `--analysis-root`
(any subdirectory containing `summary_statistics_two_modes.csv`).
Subdirectories matching `seed\d+` are treated as baseline runs; everything else
as variants.  **Any number of runs is supported** — the paper used 12 seeds + 4
variants, but 3–4 seeds already show the reproducibility trend.

```bash
# Uses default analysis-root (analysis_results/cifar10/resnet18)
python comprehensive_analysis/reproducibility_analysis.py

# Point to a different set of runs
python comprehensive_analysis/reproducibility_analysis.py \
  --analysis-root analysis_results/cifar10/resnet18

# Skip individual sections
python comprehensive_analysis/reproducibility_analysis.py --skip-heatmaps
python comprehensive_analysis/reproducibility_analysis.py --skip-rank
```

Output:
- `analysis_results/figures/` — panels, heatmaps, and rank-stability figures → **Figures 2, 3, 5, 6** (main); **9, 10, 12, 13, 14, 15** (appendix)
- `analysis_results/tables/` — `reproducibility_*runs_*variants_*.csv` and rank-stability LaTeX tables → **Table 11** (appendix); **Table 13** (appendix)

### Phase 5 — Paper figure scripts

All scripts read Phase 3/4 outputs and write PDFs to `analysis_results/figures/`.

| Script | Paper output | Output file |
|---|---|---|
| `threshold_distribution.py` | **Figure 1** | `analysis_results/figures/thresh_boxplot_single.pdf`, `thresh_boxplot_multi.pdf` |
| `compose_top_vulnerable.py` | **Figure 4** *(collect)* | `analysis_results/figures/topk_vulnerable_images/<dataset>/<arch>/<run>/` |
| `loss_ratio_tpr.py` | **Figure 7** | `analysis_results/figures/lossratio_tpr.pdf` |
| `plot_benchmark_distribution.py` | **Figure 8** | `analysis_results/figures/sample_inout_score.pdf`, `sample_inout_ratio.pdf` |
| *(manual assembly)* | **Figure 11** *(appendix)* | Assemble manually — see note below |

```bash
python comprehensive_analysis/threshold_distribution.py

python comprehensive_analysis/compose_top_vulnerable.py

python comprehensive_analysis/loss_ratio_tpr.py

python comprehensive_analysis/plot_benchmark_distribution.py \
  --config configs/figure_panels/figure8_scores.yaml
python comprehensive_analysis/plot_benchmark_distribution.py \
  --config configs/figure_panels/figure8_ratios.yaml
```

> **Figure 11 (appendix — manual):** Run Phase 3 for seeds 1–3 with `--top-k 16 --nrow 4`.
> This produces `analysis_results/cifar10/resnet18/seed{1,2,3}/top16_vulnerable_online_shadow_{0p001pct,0p1pct}.png`.
> Place the six images in a 2 × 3 grid (rows = FPR threshold, columns = seed) using any image editor or LaTeX `\includegraphics`.

Figures 2, 3, 5, 6 and appendix Figures 9, 10, 12, 13, 14, 15 are produced by Phase 4 (`reproducibility_analysis.py`).
Tables 11 and 13 (appendix) are also produced by Phase 4 (rank-stability `.tex` outputs).


## Estimated Reproduction Time

All benchmarks used 256 shadow models. Timings measured on the authors' machine:
**NVIDIA GeForce RTX 4080** (16 GB), Ubuntu 20.04 (WSL2), CUDA 12.6, Windows 11 Education.
Training ran sequentially on a single GPU; Phase 2 used `aug=18` for all image benchmarks.

Times marked † are from the paper (Appendix C, full early-stopping run on RTX 4080);
times marked ★ are extrapolated from 1-epoch timing runs on the same machine.

| Benchmark | Arch | Epochs | Phase 1 training | Phase 2 attack | Total |
|---|---|---|---|---|---|
| `purchase100_baseline` | FCN (tabular) | 100 | ≤ ~16 h  | < 30 min  | ≤ ~16 h |
| `purchase100_aof` | FCN (tabular) | 100 | ≤ ~15 h  | < 30 min  | ≤ ~15 h |
| `cifar10_baseline`, `cifar10_aof` | ResNet-18 | 100 | **~29 h** † (≈ 7 min/model) | **~4–5 h** † (15–17% overhead) | **~33–34 h** |
| `cifar100_baseline`, `cifar100_aof` | WRN-28-2 | 100 | **~33 h** † (≈ 8 min/model) | **~5–6 h** † (15–17% overhead) | **~38–39 h** |
| `cifar10_tl`, `cifar100_tl`, `gtsrb_tl` | EfficientNetV2 (pretrained) | 5 | ~4–8 h | ~3–5 h | ~8–13 h |
| `gtsrb_baseline` | ResNet-18 | 100 | ~20–28 h | ~3–4 h | ~23–32 h |

Phases 3–4 (post-analysis, reproducibility figures) take < 1 h combined.
If logits are already cached, Phase 2 re-runs in < 30 min (Purchase) or ~1 h (image datasets).




## Benchmark Manifests (recommended for paper runs)

Each manifest under `configs/benchmarks/` encodes the exact train + attack + analysis
config for one Table 1 row. This is the preferred way to reproduce paper results.

```bash
# List available benchmarks
python scripts/run_benchmark.py --list

# Full run (train → attack → analyze)
python scripts/run_benchmark.py --benchmark cifar10_baseline

# Dry-run: print commands without executing
python scripts/run_benchmark.py --benchmark cifar10_baseline --dry-run

# Skip stages whose output markers already exist
python scripts/run_benchmark.py --benchmark cifar10_baseline --skip-existing

# Run only specific stages
python scripts/run_benchmark.py --benchmark cifar10_baseline --stages attack analysis
```

Available benchmarks:
`cifar10_baseline`, `cifar10_aof`, `cifar10_tl`,
`cifar100_baseline`, `cifar100_aof`, `cifar100_tl`,
`gtsrb_baseline`, `gtsrb_tl`,
`purchase100_baseline`, `purchase100_aof`



## Project Structure

```text
lira_analysis/
├── train.py
├── attack.py
├── attacks/
├── utils/
│   ├── cli.py
│   └── benchmark_cli.py
├── configs/
│   ├── train_image.yaml
│   ├── train_tabular.yaml
│   ├── finetune.yaml
│   ├── attack.yaml
│   ├── benchmarks/               # one YAML per Table 1 row
│   └── figure_panels/            # Figure 8 panel manifests
├── scripts/
│   └── run_benchmark.py
├── comprehensive_analysis/
│   ├── run_analysis.py           # Phase 3: per-run post-analysis
│   ├── reproducibility_analysis.py  # Phase 4: cross-run analysis
│   ├── threshold_distribution.py    # Figure 1
│   ├── compose_top_vulnerable.py    # collect per-run top-vulnerable grids
│   ├── loss_ratio_tpr.py            # Figure 7
│   └── plot_benchmark_distribution.py  # Figure 8
└── experiments/
```


## Datasets and Models

| Dataset | Type | Classes | Samples | Models |
| --- | --- | --- | --- | --- |
| CIFAR-10 / CIFAR-100 | image | 10 / 100 | 60K | ResNet-18, WideResNet, EfficientNet-V2 |
| GTSRB | image | 43 | 51K | ResNet-18, EfficientNet-V2 |
| Purchase-100 | tabular | 100 | 197K | FCN |

Notes:
- Image datasets download automatically on first use.
- Purchase-100 must be staged under `data/purchase/`, as documented in [data/Readme.md](./data/Readme.md).


## Attack Variants

1. `LiRA (online)` — uses both in- and out-of-distribution shadow statistics
2. `LiRA (online, fixed var)` — online with global variance
3. `LiRA (offline)` — uses only out-of-distribution statistics
4. `LiRA (offline, fixed var)` — offline with global variance
5. `Global threshold` — score baseline, no shadow statistics


## Analysis

All paper-facing post analysis is under [comprehensive_analysis/](./comprehensive_analysis/).

Scripts:
- [run_analysis.py](./comprehensive_analysis/run_analysis.py) — per-run TPR, PPV, vulnerability analysis
- [reproducibility_analysis.py](./comprehensive_analysis/reproducibility_analysis.py) — Jaccard, heatmaps, rank stability
- [threshold_distribution.py](./comprehensive_analysis/threshold_distribution.py) — Figure 1
- [compose_top_vulnerable.py](./comprehensive_analysis/compose_top_vulnerable.py) — collect per-run top-vulnerable grids (Figure 4 assembled manually)
- [loss_ratio_tpr.py](./comprehensive_analysis/loss_ratio_tpr.py) — Figure 7
- [plot_benchmark_distribution.py](./comprehensive_analysis/plot_benchmark_distribution.py) — Figure 8
- Figure 11 *(appendix)* — assembled manually from per-run top-16 grids (see Phase 5 note)

For the paper-to-artifact map and badge requirements, see [ARTIFACT-APPENDIX.md](./ARTIFACT-APPENDIX.md).
For the expected output path index, see [RESULTS_INDEX.md](./RESULTS_INDEX.md).
For the complete CLI command reference, see [cli_commands.txt](./cli_commands.txt).


## Citation

```bibtex
@misc{jebreel2026revisitingliramembershipinference,
  title         = {Revisiting the {LiRA} Membership Inference Attack Under Realistic Assumptions},
  author        = {Najeeb Jebreel and Mona Khalil and David S\'{a}nchez and Josep Domingo-Ferrer},
  year          = {2026},
  eprint        = {2603.07567},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2603.07567},
}
```

> The proceedings citation (PoPETs 2026 volume/issue/DOI) will be added after publication.


## Acknowledgments

Partial support to this work has been received from the Government of Catalonia (ICREA Acadèmia Prizes to J. Domingo-Ferrer and to D. Sánchez), MCIN/AEI under grant PID2024-157271NB-I00 "CLEARING-IT", and the European Commission under project HORIZON-101292277 "SoBigData IP".

We used Claude Sonnet 4.5 to optimize code implementation and WriteFull and ChatGPT to correct typos, grammatical errors, and awkward phrasing throughout the article.

- Original LiRA: [Carlini et al., 2022](https://ieeexplore.ieee.org/document/9833649)


## License

Released under the MIT License. See [LICENSE](./LICENSE).
