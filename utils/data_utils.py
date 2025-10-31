"""
Optimized data loading utilities with improved memory efficiency and performance.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, GTSRB
from PIL import Image

# Centralized dataset statistics for easier management
DATASET_STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'cifar100': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'gtsrb': ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),  
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}




class TabularDataset(Dataset):
    """
    Memory-efficient dataset class for tabular data with pre-conversion to torch tensors.
    """
    def __init__(self, features, labels, transform=None):
        """
        Initialize a tabular dataset.
        
        Args:
            features: Feature matrix (numpy array or torch tensor)
            labels: Label vector (numpy array or torch tensor)
            transform: Optional transform to apply to features
        """
        self.transform = transform
        
        # Convert to torch tensors once during initialization for efficiency
        if not isinstance(features, torch.Tensor):
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features
            
        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
            
        # No need to convert to tensor again
        return features, label


class TransformSubset(Dataset):
    """
    Efficient dataset wrapper that applies transformations to a subset.
    """
    def __init__(self, full_data, keep_indices, transform=None):
        """
        Args:
            full_data (Dataset): The full dataset to sample from
            keep_indices (ndarray): Boolean array or indices to select samples
            transform (callable): Transform to apply to each sample
        """
        # Efficient index extraction
        if isinstance(keep_indices, np.ndarray) and keep_indices.dtype == bool:
            indices = np.where(keep_indices)[0]
        else:
            indices = keep_indices
            
        self.subset = Subset(full_data, indices)
        self.transform = transform
        # Cache length for faster access
        self._len = len(self.subset)

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            # Handle different input types appropriately
            if isinstance(x, Image.Image):  # PIL images
                x = self.transform(x)
            elif isinstance(x, torch.Tensor):  # Tensors
                x = self.transform(x)
            # Other data types handled by transform
        return x, y

    def __len__(self):
        return self._len


def build_transforms(config, train=True):
    """
    Build optimized transforms with better ordering for efficiency.
    
    Args:
        config (dict): Configuration dictionary containing dataset settings
        train (bool): Whether to build training or evaluation transforms
        
    Returns:
        torchvision.transforms.Compose or None
    """
    ds_cfg = config.get('dataset', {})
    name = ds_cfg.get('name', '').lower()
    
    # Early return for tabular datasets
    if name in ['purchase', 'texas', 'location']:
        return None
    
    input_size = ds_cfg.get('input_size', 32)  # Default to 32 for most datasets

    # Use training-specific augmentations
    aug_list = config.get('train_data_augmentation', []) if train else []
    
    transform_ops = []
    
    transform_ops.append(transforms.Resize((input_size, input_size)))

    # Add training-specific transforms in optimal order
    if train:
        # Spatial transforms first (on PIL images)
        for aug in aug_list:
            if aug == 'random_crop':
                # Scale padding proportionally to input size
                pad = max(1, input_size // 8)  # Default ratio of 4/32 = 1/8
                transform_ops.append(transforms.RandomCrop(input_size, padding=pad, padding_mode='reflect'))
            elif aug == 'random_flip':
                transform_ops.append(transforms.RandomHorizontalFlip())
            elif aug == 'random_rotation':
                # Smaller rotation for traffic signs to maintain recognizability
                angle = 10 if name == 'gtsrb' else 15
                transform_ops.append(transforms.RandomRotation(angle))
            elif aug == 'color_jitter':
                # Reduced color jitter for traffic signs
                if name == 'gtsrb':
                    transform_ops.append(transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ))
                else:
                    transform_ops.append(transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ))
    
    # Convert to tensor (required before tensor-based transforms)
    transform_ops.append(transforms.ToTensor())
    
    # Add tensor-based transforms
    if train and 'cutout' in aug_list:
        # Scale cutout size proportionally to input size
        # Original was 16x16 on 32x32, so maintain the same ratio (1/4 of input_size)
        cutout_size = input_size // 2
        cutout_area = cutout_size * cutout_size
        total_area = input_size * input_size
        scale_ratio = cutout_area / total_area
        
        transform_ops.append(transforms.RandomErasing(
            p=1.0,
            scale=(scale_ratio, scale_ratio),
            ratio=(1.0, 1.0),
            value=0
        ))
    
    # Normalization (always after ToTensor)
    if ('normalize' in aug_list) or (not train):
        # Get dataset normalization stats
        pretrained = config.get('model', {}).get('pretrained', False)
        stats_key = 'imagenet' if (pretrained or name == 'imagenet') else name
        mean, std = DATASET_STATS.get(stats_key, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform_ops.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(transform_ops)


def get_keep_indices(dataset_size, num_shadow_models, pkeep, seed=0):
    """
    Generate optimized boolean mask for sample selection using vectorized operations.

    Args:
        dataset_size (int): Total number of samples in the dataset
        num_shadow_models (int): Number of shadow models to create
        pkeep (float): Proportion of data to keep for each shadow model
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Boolean array of shape (num_shadow_models, dataset_size)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    if num_shadow_models <= 0:
        # Single model case (simpler)
        return np.random.uniform(0, 1, size=dataset_size) <= pkeep
    
    # Vectorized implementation for multiple shadow models
    # Generate uniform random values for all models/samples at once
    keep_scores = np.random.uniform(0, 1, size=(num_shadow_models, dataset_size))
    
    # Compute threshold based on pkeep
    threshold = int(pkeep * num_shadow_models)
    
    # Use numpy's efficient sorting and comparison operations
    order = np.argsort(keep_scores, axis=0)
    keep_indices = order < threshold
    
    return keep_indices


def load_dataset(config, mode='training'):
    """
    Unified dataset loading function for both training and inference.

    Args:
        config (dict): Configuration dictionary containing dataset details
        mode (str): Loading mode - 'training' or 'inference'
            - 'training': Returns datasets without transforms, plus keep_indices and transforms separately
            - 'inference': Returns dataset with transforms applied, plus labels array

    Returns:
        If mode='training': (full_dataset, keep_indices, train_transform, test_transform)
        If mode='inference': (full_dataset, full_labels)
    """
    dataset_name = config['dataset']['name'].lower()
    data_dir = config.get('dataset', {}).get('data_dir', 'data')

    # Prepare transforms based on mode
    if mode == 'training':
        train_transform = build_transforms(config, train=True)
        test_transform = build_transforms(config, train=False)
        apply_transform = None  # Don't apply yet for training
    else:  # inference
        apply_transform = build_transforms(config, train=False)
        train_transform = test_transform = None

    # Load appropriate dataset based on name
    if dataset_name == 'cifar10':
        train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=apply_transform)
        full_dataset = ConcatDataset([train_dataset, test_dataset])

        if mode == 'inference':
            train_labels = np.array(train_dataset.targets)
            test_labels = np.array(test_dataset.targets)
            full_labels = np.concatenate((train_labels, test_labels), axis=0)

    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=apply_transform)
        full_dataset = ConcatDataset([train_dataset, test_dataset])

        if mode == 'inference':
            train_labels = np.array(train_dataset.targets)
            test_labels = np.array(test_dataset.targets)
            full_labels = np.concatenate((train_labels, test_labels), axis=0)

    elif dataset_name == 'gtsrb':
        train_dataset = GTSRB(root=data_dir, split='train', download=True, transform=apply_transform)
        test_dataset = GTSRB(root=data_dir, split='test', download=True, transform=apply_transform)
        full_dataset = ConcatDataset([train_dataset, test_dataset])

        if mode == 'inference':
            train_labels = np.array([sample[1] for sample in train_dataset._samples])
            test_labels = np.array([sample[1] for sample in test_dataset._samples])
            full_labels = np.concatenate((train_labels, test_labels), axis=0)

    elif dataset_name == 'purchase':
        data_path = os.path.join(data_dir, dataset_name, 'features_labels.npy')
        full_data = np.load(data_path)
        full_features = torch.tensor(full_data[:,1:].astype(np.float32))
        full_labels_raw = full_data[:,0].astype(np.int32) - 1

        if mode == 'training':
            full_labels_tensor = torch.tensor(full_labels_raw.astype(np.int64))
            full_dataset = TabularDataset(full_features, full_labels_tensor)
        else:  # inference
            full_dataset = TabularDataset(full_features, full_labels_raw, transform=None)
            full_labels = full_labels_raw

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Return based on mode
    if mode == 'training':
        keep_indices = get_keep_indices(
            len(full_dataset),
            config['training']['num_shadow_models'],
            config['dataset']['pkeep'],
            config['seed']
        )
        return full_dataset, keep_indices, train_transform, test_transform
    else:  # inference
        return full_dataset, full_labels


def create_data_loaders(train_dataset, test_dataset, train_dataset_eval, config):
    """
    Create optimized data loaders with efficient memory usage and augmentations.
    
    Args:
        train_dataset: Training dataset with transforms
        test_dataset: Test dataset with transforms
        train_dataset_eval: Training dataset with evaluation transforms
        config: Configuration dictionary
        
    Returns:
        tuple: (train_loader, test_loader, train_eval_loader)
    """
    t_cfg = config.get('training', {})
    batch_size = t_cfg.get('batch_size', 128)
    
    # Determine optimal number of workers based on CPU count
    workers = t_cfg.get('num_workers', 0)
    # if workers <= 0:
    #     workers = min(8, multiprocessing.cpu_count())
    
    # Check for CUDA availability for pin_memory optimization
    pin_memory = torch.cuda.is_available() and config.get('use_cuda', True) 
    
    # Create optimized data loaders with shared settings
    loader_kwargs = {
        'num_workers': workers,
        'pin_memory': pin_memory,
        'persistent_workers': workers > 0,
        'prefetch_factor': 4 if workers > 0 else None,
    }
    
    # Training loader - needs shuffling and custom collation
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Improves batch norm statistics
        **loader_kwargs
    )
    
    # Evaluation loaders - larger batch size, no shuffling or custom collation
    eval_batch_size = batch_size * 4  # Can use larger batches for evaluation
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    train_eval_loader = DataLoader(
        train_dataset_eval,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, test_loader, train_eval_loader