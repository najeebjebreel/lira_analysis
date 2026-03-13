# Figure Panel Manifests

This directory stores named input manifests for figure-regeneration scripts.

Current manifests:
- `figure8_scores.yaml`
- `figure8_ratios.yaml`

These record the intended panel titles, sample index, source experiment directories, and output paths for Figure 8.

Important:
- the manifests store source experiment directories and panel titles for Figure 8
- Figure 8 requires running all 10 benchmarks; the `--experiments-root` flag can point to any experiments tree

Use with:

```bash
python comprehensive_analysis/plot_benchmark_distribution.py ^
  --config configs/figure_panels/figure8_scores.yaml ^
  --experiments-root D:\path\to\external\experiments

python comprehensive_analysis/plot_benchmark_distribution.py ^
  --config configs/figure_panels/figure8_ratios.yaml ^
  --experiments-root D:\path\to\external\experiments
```
