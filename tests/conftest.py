"""
Pytest configuration and fixtures.
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device('cpu')


@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        'seed': 42,
        'use_cuda': False,
        'dataset': {
            'name': 'cifar10',
            'num_classes': 10,
            'input_size': 32,
            'data_dir': 'data',
            'pkeep': 0.5
        },
        'model': {
            'architecture': 'resnet18',
            'pretrained': False,
            'drop_rate': 0.0,
            'cifar_stem': True
        },
        'training': {
            'batch_size': 32,
            'epochs': 2,
            'learning_rate': 0.01,
            'optimizer': 'sgd',
            'num_shadow_models': 2,
            'num_workers': 0
        }
    }


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
