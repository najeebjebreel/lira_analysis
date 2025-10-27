# Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions

A research implementation of the LiRA (Likelihood Ratio Attack) membership inference attack with improved structure and modern Python packaging.

## Project Structure

```
lira_analysis/
├── README.md                    # This file
├── .gitignore                   # Git ignore patterns
├── pyproject.toml              # Modern Python packaging configuration
├── setup.py                    # Backward compatibility setup script
├── requirements.txt            # Project dependencies
│
├── src/mia_research/           # Main package
│   ├── __init__.py
│   ├── attacks/                # Attack implementations
│   │   ├── __init__.py
│   │   └── lira.py            # LiRA attack implementation
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   └── model_utils.py     # Model definitions and utilities
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_utils.py      # Dataset loaders and transforms
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   └── train_utils.py     # Training loops and helpers
│   └── utils/                  # General utilities
│       ├── __init__.py
│       └── utils.py           # Logging, config, etc.
│
├── scripts/                    # Command-line entry points
│   ├── train.py               # Training script
│   └── attack.py              # Attack evaluation script
│
├── configs/                    # Configuration files
│   ├── attack_config.yaml
│   ├── train_image_config.yaml
│   ├── train_tabular_config.yaml
│   └── finetune_image_config.yaml
│
├── notebooks/                  # Analysis notebooks
│   ├── post_analysis.ipynb
│   ├── agreement.ipynb
│   ├── plot_benchmark_distribution.ipynb
│   ├── loss_ratio_tpr.ipynb
│   └── threshold_dist.py
│
├── experiments/                # Generated outputs (gitignored)
│   └── .gitkeep
│
└── data/                       # Datasets (gitignored)
    └── .gitkeep
```

## Installation

### Using pip (recommended)

Install the package in editable mode with all dependencies:

```bash
# Clone the repository
git clone https://github.com/najeebjebreel/lira_analysis.git
cd lira_analysis

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using requirements.txt

```bash
pip install -r requirements.txt
```

## Usage

### Training Shadow Models

```bash
# Using the installed command
mia-train --config configs/train_image_config.yaml

# Or directly with Python
python scripts/train.py --config configs/train_image_config.yaml

# With config overrides
python scripts/train.py --config configs/train_image_config.yaml \
    --override training.epochs=50 dataset.name=cifar100
```

### Running Attacks

```bash
# Using the installed command
mia-attack --config configs/attack_config.yaml

# Or directly with Python
python scripts/attack.py --config configs/attack_config.yaml

# With config overrides
python scripts/attack.py --config configs/attack_config.yaml \
    --override attack.evaluation_mode=leave_one_out
```

### Using as a Library

```python
from mia_research.attacks import LiRA
from mia_research.data import load_dataset
from mia_research.models import get_model
from mia_research.training import train_model
from mia_research.utils import setup_logger, set_seed

# Your code here
```

## Configuration

Configuration files are in YAML format and support hierarchical structure:

- `configs/train_image_config.yaml` - Image dataset training
- `configs/train_tabular_config.yaml` - Tabular dataset training
- `configs/attack_config.yaml` - Attack configuration
- `configs/finetune_image_config.yaml` - Fine-tuning pretrained models

### Key Configuration Sections

- **dataset**: Dataset name, paths, preprocessing
- **model**: Architecture, hyperparameters
- **training**: Epochs, batch size, optimizer, scheduler
- **attack**: Attack method, evaluation mode, target FPRs
- **experiment**: Logging, checkpointing

## Supported Datasets

### Vision Datasets
- CIFAR-10
- CIFAR-100
- CINIC-10
- GTSRB
- ImageNet

### Tabular Datasets
- Purchase
- Texas
- Location

## Supported Models

### Vision Models (via timm)
- ResNet (18, 34, 50, etc.)
- Wide ResNet (WRN-28-2, WRN-28-10)
- Vision Transformers (ViT)
- EfficientNet
- And many more from the timm library

### Tabular Models
- Fully Connected Networks (FCN)

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Formatting

```bash
# Format code with black
black src/ scripts/

# Sort imports
isort src/ scripts/

# Lint code
flake8 src/ scripts/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lira_revisited,
  title={Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions},
  author={Your Name},
  year={2024}
}
```

## License

MIT License (or specify your license)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
