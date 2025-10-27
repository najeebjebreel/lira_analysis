# Quick Start Guide

This guide will help you run your first LiRA experiment.

## Step 1: Install the Package

```bash
pip install -e .
```

## Step 2: Prepare Configuration

Use one of the provided configuration files in `configs/`:

- `train_image_config.yaml` - For image datasets (CIFAR-10, CIFAR-100, etc.)
- `train_tabular_config.yaml` - For tabular datasets (Purchase, Texas, Location)
- `attack_config.yaml` - For running membership inference attacks

## Step 3: Train Shadow Models

```bash
# Train on CIFAR-10 with default settings
mia-train --config configs/train_image_config.yaml

# Or with custom overrides
mia-train --config configs/train_image_config.yaml \
    --override dataset.name=cifar100 \
    --override training.epochs=50 \
    --override training.num_shadow_models=64
```

Training will create an experiment directory like:
```
experiments/cifar10/resnet18/2024-10-27_1630/
├── model_0/
│   ├── best_model.pth
│   └── ...
├── model_1/
├── ...
├── keep_indices.npy
└── train_config.yaml
```

## Step 4: Run Attack

Update `configs/attack_config.yaml` to point to your experiment:

```yaml
experiment:
  checkpoint_dir: experiments/cifar10/resnet18/2024-10-27_1630
```

Then run the attack:

```bash
mia-attack --config configs/attack_config.yaml
```

## Step 5: View Results

After the attack completes, results are saved in the experiment directory:

- `roc_curve_single.pdf` - ROC curve for single target evaluation
- `attack_results_single.csv` - Metrics table
- `attack_results_leave_one_out_summary.csv` - Leave-one-out metrics
- `membership_labels.npy` - Ground truth membership labels
- `*_scores_leave_one_out.npy` - Attack scores for each method

## Example: Complete Workflow

```bash
# 1. Train 16 shadow models on CIFAR-10
mia-train --config configs/train_image_config.yaml \
    --override training.num_shadow_models=16 \
    --override training.epochs=100

# 2. Run LiRA attack with leave-one-out evaluation
mia-attack --config configs/attack_config.yaml \
    --override attack.evaluation_mode=leave_one_out \
    --override experiment.checkpoint_dir=experiments/cifar10/resnet18/YOUR_EXPERIMENT_DIR
```

## Using as a Python Library

```python
from mia_research.attacks import LiRA
from mia_research.models import get_model
from mia_research.data import load_dataset
from mia_research.utils import setup_logger, set_seed
import yaml

# Load configuration
with open('configs/attack_config.yaml') as f:
    config = yaml.safe_load(f)

# Set up logging and seed
logger = setup_logger('MyExperiment', 'experiment.log')
set_seed(config['seed'])

# Initialize and run attack
lira = LiRA(config, logger)
lira.generate_logits()
lira.compute_scores()
lira.plot(ntest=1, metric='auc')
```

## Configuration Tips

### Quick Training (Testing)
```yaml
training:
  epochs: 10
  num_shadow_models: 4
  batch_size: 128
```

### Production Run
```yaml
training:
  epochs: 100
  num_shadow_models: 64
  batch_size: 256
```

### GPU Memory Saving
```yaml
training:
  batch_size: 64
  use_amp: true  # Mixed precision training
  num_workers: 2
```

## Common Issues

**Issue**: Training is slow
**Solution**: Enable CUDA, increase batch size, or use mixed precision training

**Issue**: Out of memory
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: Poor attack performance
**Solution**: Train more shadow models, increase training epochs, or use stronger augmentation

## Next Steps

- Explore [Configuration Guide](configuration.md) for all options
- Check out analysis notebooks in `notebooks/`
- Read the [API Reference](api.md) for programmatic usage
