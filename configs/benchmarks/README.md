# Benchmark Manifests

Each YAML file in this directory defines one paper benchmark row from Table 1.
Each manifest encodes the exact train config, attack config, and analysis
parameters for that row, making individual runs fully reproducible from a
single command.

## Usage

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
python scripts/run_benchmark.py --benchmark cifar10_baseline --stages analysis
```

## Available benchmarks

| ID | Dataset | Setting |
|----|---------|---------|
| `cifar10_baseline` | CIFAR-10 | Scratch ResNet-18 |
| `cifar10_aof` | CIFAR-10 | Anti-overfitting |
| `cifar10_tl` | CIFAR-10 | Transfer learning |
| `cifar100_baseline` | CIFAR-100 | Scratch WideResNet-28-2 |
| `cifar100_aof` | CIFAR-100 | Anti-overfitting |
| `cifar100_tl` | CIFAR-100 | Transfer learning |
| `gtsrb_baseline` | GTSRB | Scratch ResNet-18 |
| `gtsrb_tl` | GTSRB | Transfer learning |
| `purchase100_baseline` | Purchase-100 | Scratch FCN |
| `purchase100_aof` | Purchase-100 | Anti-overfitting |

## Skip-detection markers

`--skip-existing` checks for the following files before running each stage:

| Stage | Marker |
|-------|--------|
| `train` | `<exp_dir>/train_config.yaml` |
| `attack` | `<exp_dir>/attack_results_leave_one_out_summary.csv` |
| `analysis` | `analysis_results/.../<run>/summary_statistics_two_modes.csv` |
