"""
Numerical stability utilities for LiRA membership inference attack.

This module provides numerically stable implementations of common operations
used in membership inference attacks, particularly for computing probabilities
and losses from model logits.
"""

import numpy as np
from scipy.special import softmax, logsumexp


def stable_softmax(logits, axis=-1):
    """
    Compute numerically stable softmax using scipy.

    Args:
        logits: Input logits (numpy array)
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    return softmax(logits, axis=axis)


def stable_logsumexp(logits, axis=-1, keepdims=False):
    """
    Compute numerically stable log-sum-exp using scipy.

    Args:
        logits: Input logits (numpy array)
        axis: Axis along which to compute logsumexp
        keepdims: Whether to keep dimensions

    Returns:
        Log-sum-exp result
    """
    return logsumexp(logits, axis=axis, keepdims=keepdims)


def compute_cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss with numerical stability.

    Args:
        logits: Model logits, shape (N, C) where N=samples, C=classes
        labels: True labels, shape (N,)

    Returns:
        Per-sample cross-entropy losses, shape (N,)
    """
    N = logits.shape[0]

    # Stable log-sum-exp
    lse = stable_logsumexp(logits, axis=1, keepdims=False)

    # Extract true class logits
    true_class_logits = logits[np.arange(N), labels]

    # Cross-entropy: log-sum-exp - true_class_logit
    losses = lse - true_class_logits

    return losses
