# Configuration Guide

This guide explains all configuration options for training and attack experiments.

## Configuration Files

Configuration is managed through YAML files in the `configs/` directory:

- `train_image_config.yaml` - Image dataset training
- `train_tabular_config.yaml` - Tabular dataset training
- `finetune_image_config.yaml` - Fine-tuning pretrained models
- `attack_config.yaml` - Attack evaluation

## Configuration Structure

### Global Settings

```yaml
seed: 42              # Random seed for reproducibility
use_cuda: true        # Use GPU if available
```

### Dataset Configuration

```yaml
dataset:
  name: cifar10                    # Dataset name
  num_classes: 10                  # Number of classes
  input_size: 32                   # Image size (32 for CIFAR, 224 for ImageNet)
  data_dir: data                   # Directory for datasets
  pkeep: 0.5                       # Proportion of data per shadow model
```

**Supported Datasets:**
- Image: `cifar10`, `cifar100`, `cinic10`, `gtsrb`, `imagenet`
- Tabular: `purchase`, `texas`, `location`

### Model Configuration

```yaml
model:
  architecture: resnet18           # Model architecture
  pretrained: false                # Use pretrained weights
  cifar_stem: true                 # Use CIFAR-optimized stem (3x3 conv)
  drop_rate: 0.0                   # Dropout rate
```

**Supported Architectures:**
- ResNet: `resnet18`, `resnet34`, `resnet50`, etc.
- WideResNet: `wrn28-2`, `wrn28-10`
- Vision Transformers: `vit_base_patch16_224`, etc.
- Tabular: `fcn`
- Any model from [timm library](https://github.com/huggingface/pytorch-image-models)

### Training Configuration

```yaml
training:
  epochs: 100                      # Number of training epochs
  batch_size: 128                  # Training batch size
  learning_rate: 0.1               # Initial learning rate
  optimizer: sgd                   # Optimizer: sgd, adam, adamw
  momentum: 0.9                    # SGD momentum
  weight_decay: 5.0e-4            # L2 regularization
  lr_scheduler: cosine             # LR scheduler: cosine, onecycle
  warmup_epochs: 5.0              # Warmup epochs for cosine scheduler
  num_shadow_models: 64            # Number of shadow models to train
  num_workers: 4                   # Data loader workers
  use_amp: true                    # Mixed precision training
  save_models: false               # Save epoch checkpoints
  save_step: 20                    # Save every N epochs
  early_stopping: false            # Enable early stopping
  patience: 10                     # Early stopping patience
  resume: false                    # Resume from checkpoint
```

### Data Augmentation

**Training Augmentations:**
```yaml
train_data_augmentation:
  - random_flip                    # Horizontal flip
  - random_crop                    # Random crop with padding
  - random_rotation                # Small rotations
  - color_jitter                   # Color transformations
  - cutout                         # Cutout/RandomErasing
  - mixup                          # MixUp augmentation
  - cutmix                         # CutMix augmentation
  - normalize                      # Normalize to dataset stats
```

**Inference Augmentations (for LiRA):**
```yaml
inference_data_augmentations:
  type: spatial                    # Type: none, horizontal_flip, spatial
  spatial_shift: 2                 # Pixels to shift
  spatial_stride: 2                # Grid stride
  horizontal_flip: true            # Include horizontal flips
```

### Attack Configuration

```yaml
attack:
  method: [lira]                   # Attack methods to run
  evaluation_mode: both            # single, leave_one_out, or both
  ntest: 1                         # Number of target models (for single mode)
  plot_metric: auc                 # Metric to show: auc or acc
  target_fprs: [0.00001, 0.001]   # Target false positive rates
  prior: 0.5                       # Prior membership probability
  target_model: best               # Use best or epoch N checkpoint
  save_hard_labels: true           # Save ground truth labels
  save_attack_predictions: true    # Save attack scores
```

### Experiment Configuration

```yaml
experiment:
  log_level: info                  # Logging level: debug, info, warning, error
  checkpoint_dir: none             # Path to experiment directory (or none)
  overwrite_logits: false          # Recompute existing logits
  overwrite_scores: false          # Recompute existing scores
```

## Configuration Overrides

Override any config value from command line:

```bash
# Single override
mia-train --config configs/train_image_config.yaml \
    --override training.epochs=50

# Multiple overrides
mia-train --config configs/train_image_config.yaml \
    --override training.epochs=50 \
    --override dataset.name=cifar100 \
    --override model.architecture=wrn28-10

# Nested values
mia-attack --config configs/attack_config.yaml \
    --override attack.evaluation_mode=leave_one_out \
    --override experiment.checkpoint_dir=experiments/cifar10/resnet18/2024-10-27_1630
```

## Configuration Best Practices

### For Quick Testing
```yaml
training:
  epochs: 10
  num_shadow_models: 4
  batch_size: 64
```

### For Research Experiments
```yaml
training:
  epochs: 100
  num_shadow_models: 64
  batch_size: 256
  lr_scheduler: cosine
  warmup_epochs: 5
```

### For Limited GPU Memory
```yaml
training:
  batch_size: 32
  use_amp: true
  num_workers: 2
model:
  architecture: resnet18  # Smaller model
```

### For Best Attack Performance
```yaml
training:
  num_shadow_models: 128
  epochs: 200
inference_data_augmentations:
  type: spatial
  spatial_shift: 4
  spatial_stride: 1
  horizontal_flip: true
```

## Environment Variables

You can also use environment variables:

```bash
export MIA_DATA_DIR=/path/to/data
export MIA_EXPERIMENT_DIR=/path/to/experiments
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPUs
```

## Configuration Validation

The configuration is validated at runtime. Common errors:

- **Missing required fields**: Ensure all required fields are present
- **Invalid values**: Check enum options (e.g., optimizer must be sgd/adam/adamw)
- **Type mismatches**: Ensure numeric values are numbers, not strings
- **Path issues**: Check that checkpoint_dir exists for attacks

## Advanced: Programmatic Configuration

```python
import yaml

# Load base config
with open('configs/train_image_config.yaml') as f:
    config = yaml.safe_load(f)

# Modify programmatically
config['training']['epochs'] = 50
config['dataset']['name'] = 'cifar100'

# Save modified config
with open('my_config.yaml', 'w') as f:
    yaml.dump(config, f)
```
