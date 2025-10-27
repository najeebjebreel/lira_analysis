"""
Tests for data utilities.
"""
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from mia_research.data.data_utils import (
    TabularDataset,
    TransformSubset,
    get_keep_indices,
    build_transforms
)


def test_tabular_dataset():
    """Test TabularDataset creation and access."""
    features = np.random.randn(100, 10)
    labels = np.random.randint(0, 5, 100)

    dataset = TabularDataset(features, labels)

    assert len(dataset) == 100

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (10,)


def test_get_keep_indices():
    """Test keep_indices generation."""
    dataset_size = 1000
    num_shadow_models = 10
    pkeep = 0.5

    keep_indices = get_keep_indices(dataset_size, num_shadow_models, pkeep, seed=42)

    assert keep_indices.shape == (num_shadow_models, dataset_size)
    assert keep_indices.dtype == bool

    # Check that approximately pkeep proportion is True
    mean_keep = keep_indices.mean()
    assert 0.4 < mean_keep < 0.6  # Allow some variance


def test_get_keep_indices_reproducibility():
    """Test that keep_indices generation is reproducible."""
    indices1 = get_keep_indices(100, 5, 0.5, seed=42)
    indices2 = get_keep_indices(100, 5, 0.5, seed=42)

    assert np.array_equal(indices1, indices2)


def test_build_transforms_train():
    """Test building training transforms."""
    config = {
        'dataset': {'name': 'cifar10', 'input_size': 32},
        'train_data_augmentation': ['random_flip', 'random_crop'],
        'model': {'pretrained': False}
    }

    transform = build_transforms(config, train=True)

    assert transform is not None
    # Should have multiple transforms
    assert len(transform.transforms) > 2


def test_build_transforms_test():
    """Test building test transforms."""
    config = {
        'dataset': {'name': 'cifar10', 'input_size': 32},
        'model': {'pretrained': False}
    }

    transform = build_transforms(config, train=False)

    assert transform is not None
    # Should have basic transforms (resize, to_tensor, normalize)
    assert len(transform.transforms) >= 3


def test_build_transforms_tabular():
    """Test that tabular datasets return None for transforms."""
    config = {
        'dataset': {'name': 'purchase'},
    }

    transform = build_transforms(config, train=True)

    assert transform is None
