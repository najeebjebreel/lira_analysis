"""
Tests for utility functions.
"""
import pytest
import torch
import numpy as np
from mia_research.utils.utils import set_seed, parse_overrides, recursive_update


def test_set_seed():
    """Test that set_seed produces reproducible results."""
    set_seed(42)
    x1 = torch.randn(10)
    y1 = np.random.rand(10)

    set_seed(42)
    x2 = torch.randn(10)
    y2 = np.random.rand(10)

    assert torch.allclose(x1, x2), "Torch random values should be identical"
    assert np.allclose(y1, y2), "Numpy random values should be identical"


def test_parse_overrides():
    """Test parsing command-line overrides."""
    overrides_list = [
        'training.epochs=50',
        'dataset.name=cifar100',
        'model.drop_rate=0.1',
        'use_cuda=true'
    ]

    result = parse_overrides(overrides_list)

    assert result['training']['epochs'] == 50
    assert result['dataset']['name'] == 'cifar100'
    assert result['model']['drop_rate'] == 0.1
    assert result['use_cuda'] is True


def test_recursive_update():
    """Test recursive dictionary update."""
    base = {
        'a': 1,
        'b': {'c': 2, 'd': 3},
        'e': 5
    }

    update = {
        'b': {'c': 20},
        'e': 50,
        'f': 6
    }

    recursive_update(base, update)

    assert base['a'] == 1
    assert base['b']['c'] == 20
    assert base['b']['d'] == 3  # Should preserve
    assert base['e'] == 50
    assert base['f'] == 6


def test_recursive_update_none_handling():
    """Test that None values don't overwrite existing values."""
    base = {'key': 'value'}
    update = {'key': None}

    recursive_update(base, update)

    # Should preserve original value
    assert base['key'] == 'value'
