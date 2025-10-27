# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (optional but recommended)

## Installation Methods

### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/najeebjebreel/lira_analysis.git
cd lira_analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Method 2: Install with Development Dependencies

```bash
# Install with dev dependencies for testing and formatting
pip install -e ".[dev]"
```

### Method 3: Install from requirements.txt

```bash
# Install core dependencies only
pip install -r requirements.txt
```

## Verifying Installation

### Check Package Installation

```python
import mia_research
print(mia_research.__version__)
```

### Check Entry Points

```bash
# These commands should work after installation
mia-train --help
mia-attack --help
```

### Run Tests

```bash
# Run the test suite to verify everything works
pytest tests/ -v
```

## GPU Support

### CUDA Installation

If you have a CUDA-capable GPU, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### ImportError: No module named 'mia_research'

Make sure you installed the package in editable mode:
```bash
pip install -e .
```

### CUDA Out of Memory

Reduce batch size in your configuration file:
```yaml
training:
  batch_size: 32  # Try smaller values like 16 or 8
```

### Package Dependencies Conflict

Create a fresh virtual environment:
```bash
python -m venv fresh_venv
source fresh_venv/bin/activate
pip install -e .
```

## Next Steps

- Read the [Quick Start Guide](quickstart.md)
- Review [Configuration Options](configuration.md)
- Explore example notebooks in `notebooks/`
