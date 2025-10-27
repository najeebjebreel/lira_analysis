"""
Tests for model architectures.
"""
import pytest
import torch
from mia_research.models.model_utils import get_model, FCN, WideResNet


def test_fcn_creation():
    """Test FCN model creation."""
    model = FCN(input_dim=100, num_classes=10, hidden_dims=[64, 32])

    assert model is not None

    # Test forward pass
    x = torch.randn(4, 100)
    output = model(x)

    assert output.shape == (4, 10)


def test_wideresnet_creation():
    """Test WideResNet model creation."""
    model = WideResNet(depth=28, num_classes=10, width=2)

    assert model is not None

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert output.shape == (2, 10)


def test_get_model_resnet():
    """Test getting ResNet model via timm."""
    model = get_model(
        num_classes=10,
        architecture='resnet18',
        pretrained=False,
        cifar_stem=True
    )

    assert model is not None

    # Test forward pass with CIFAR-sized input
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert output.shape == (2, 10)


def test_get_model_fcn():
    """Test getting FCN model."""
    model = get_model(
        num_classes=10,
        architecture='fcn',
        input_dim=100,
        hidden_dims=[64, 32]
    )

    assert model is not None

    x = torch.randn(4, 100)
    output = model(x)

    assert output.shape == (4, 10)


def test_get_model_wrn():
    """Test getting WideResNet model."""
    model = get_model(
        num_classes=10,
        architecture='wrn28-2'
    )

    assert model is not None

    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert output.shape == (2, 10)
