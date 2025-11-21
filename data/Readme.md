# Dataset Directory

This directory stores datasets used for LiRA membership inference experiments.

## Automatic Downloads

The following image datasets download automatically when running experiments:
- **CIFAR-10**: 60K 32×32 color images (10 classes)
- **CIFAR-100**: 60K 32×32 color images (100 classes)
- **GTSRB**: ~51K traffic sign images (43 classes)

No manual setup required for these datasets.

## Manual Download Required

### Purchase-100 Dataset

**Download:** [Purchase-100 Dataset](https://drive.proton.me/urls/25C1HJ14S8#3uJjfOAAPblu)

**Setup:**
1. Download the dataset from the link above
2. Create a `purchase` directory inside `data/`:
   ```bash
   mkdir -p data/purchase
   ```
3. Place the downloaded file (`features_labels.npy`) inside:
   ```
   data/purchase/features_labels.npy
   ```

**Format:** The file contains a NumPy array where:
- Column 0: Labels (class IDs)
- Columns 1+: Features (binary indicators)

## Directory Structure After Setup

```
data/
├── Readme.md                    # This file
├── cifar-10-batches-py/         # Auto-downloaded
├── cifar-100-python/            # Auto-downloaded
├── gtsrb/                       # Auto-downloaded
└── purchase/                    # Manual download
    └── features_labels.npy
```

## Usage in Code

Datasets are automatically loaded by specifying the name in config files:

```yaml
dataset:
  name: cifar10  # or cifar100, gtsrb, purchase
  data_dir: data
```

The data loader in `utils/data_utils.py` handles all dataset loading and preprocessing.
