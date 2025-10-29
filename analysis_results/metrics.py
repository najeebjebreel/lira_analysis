"""
Metrics computation utilities for LiRA attack evaluation.

This module provides functions for:
- ROC curve computation
- Confusion matrix statistics
- Precision/recall calculation
- TPR at specific FPR thresholds
"""

import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import roc_curve, auc as sklearn_auc, roc_auc_score


def compute_roc_metrics(scores: np.ndarray,
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Compute ROC curve and derived metrics.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: Ground truth membership labels (boolean)

    Returns:
        Tuple of (fpr, tpr, thresholds, auc, accuracy)
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Compute AUC
    auc_score = sklearn_auc(fpr, tpr)

    # Compute maximum accuracy
    accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)

    return fpr, tpr, thresholds, auc_score, accuracy


def compute_tpr_at_fpr(fpr: np.ndarray,
                       tpr: np.ndarray,
                       thresholds: np.ndarray,
                       target_fpr: float) -> Tuple[float, float, float]:
    """
    Find TPR and threshold at a target FPR.

    Args:
        fpr: False positive rates from ROC curve
        tpr: True positive rates from ROC curve
        thresholds: Thresholds from ROC curve
        target_fpr: Target false positive rate

    Returns:
        Tuple of (tpr_at_target, actual_fpr, threshold)
    """
    # Find largest threshold where FPR <= target_fpr
    idx = np.where(fpr <= target_fpr)[0]

    if len(idx) == 0:
        return 0.0, 0.0, np.inf

    j = idx[-1]
    return float(tpr[j]), float(fpr[j]), float(thresholds[j])


def compute_precision(tpr: float,
                      fpr: float,
                      prior: float = 0.5) -> float:
    """
    Compute precision given TPR, FPR, and membership prior.

    Precision = P(member | predicted_member)
              = (prior * tpr) / (prior * tpr + (1-prior) * fpr)

    Args:
        tpr: True positive rate
        fpr: False positive rate
        prior: Prior probability of membership

    Returns:
        Precision value
    """
    numerator = prior * tpr
    denominator = prior * tpr + (1 - prior) * fpr

    if denominator > 0:
        return numerator / denominator
    else:
        return np.nan


def compute_confusion_matrix_at_threshold(scores: np.ndarray,
                                          labels: np.ndarray,
                                          threshold: float) -> Dict[str, int]:
    """
    Compute confusion matrix at a specific threshold.

    Args:
        scores: Attack scores (higher = more likely member)
        labels: Ground truth labels (boolean)
        threshold: Decision threshold

    Returns:
        Dictionary with keys: tp, fp, tn, fn, tpr, fpr, precision, recall
    """
    predictions = scores >= threshold
    labels_bool = np.asarray(labels, dtype=bool)

    tp = int(np.sum(predictions & labels_bool))
    fp = int(np.sum(predictions & ~labels_bool))
    tn = int(np.sum(~predictions & ~labels_bool))
    fn = int(np.sum(~predictions & labels_bool))

    # Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tpr  # Same as TPR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'recall': recall
    }


def compute_metrics_at_multiple_fprs(scores: np.ndarray,
                                     labels: np.ndarray,
                                     target_fprs: list,
                                     priors: list = [0.5]) -> Dict:
    """
    Compute metrics at multiple target FPR values and priors.

    Args:
        scores: Attack scores
        labels: Ground truth labels
        target_fprs: List of target FPR values
        priors: List of membership priors

    Returns:
        Dictionary with results for each FPR and prior combination
    """
    fpr, tpr, thresholds, auc_score, accuracy = compute_roc_metrics(scores, labels)

    results = {
        'auc': auc_score,
        'accuracy': accuracy,
        'fpr_values': [],
        'tpr_values': [],
        'thresholds': [],
        'precisions': {}
    }

    for target_fpr in target_fprs:
        tpr_at, actual_fpr, thresh = compute_tpr_at_fpr(fpr, tpr, thresholds, target_fpr)

        results['fpr_values'].append((target_fpr, actual_fpr))
        results['tpr_values'].append(tpr_at)
        results['thresholds'].append(thresh)

        # Compute precision for each prior
        for prior in priors:
            prec = compute_precision(tpr_at, actual_fpr, prior)
            key = f'precision_prior_{prior}_fpr_{target_fpr}'
            results['precisions'][key] = prec

    return results


def aggregate_metrics_across_models(scores_list: list,
                                    labels_list: list,
                                    target_fprs: list = [0.001],
                                    priors: list = [0.5]) -> Dict:
    """
    Aggregate metrics across multiple target models (e.g., leave-one-out).

    Args:
        scores_list: List of score arrays (one per target model)
        labels_list: List of label arrays (one per target model)
        target_fprs: List of target FPR values
        priors: List of membership priors

    Returns:
        Dictionary with mean and std of metrics across models
    """
    all_results = []

    for scores, labels in zip(scores_list, labels_list):
        results = compute_metrics_at_multiple_fprs(scores, labels, target_fprs, priors)
        all_results.append(results)

    # Aggregate
    aggregated = {
        'auc_mean': np.mean([r['auc'] for r in all_results]),
        'auc_std': np.std([r['auc'] for r in all_results], ddof=1),
        'accuracy_mean': np.mean([r['accuracy'] for r in all_results]),
        'accuracy_std': np.std([r['accuracy'] for r in all_results], ddof=1),
    }

    # Aggregate TPR at FPRs
    for i, target_fpr in enumerate(target_fprs):
        tprs = [r['tpr_values'][i] for r in all_results]
        aggregated[f'tpr_at_{target_fpr}_mean'] = np.mean(tprs)
        aggregated[f'tpr_at_{target_fpr}_std'] = np.std(tprs, ddof=1)

    # Aggregate precisions
    for prior in priors:
        for target_fpr in target_fprs:
            key = f'precision_prior_{prior}_fpr_{target_fpr}'
            precs = [r['precisions'][key] for r in all_results if key in r['precisions']]
            aggregated[f'{key}_mean'] = np.nanmean(precs)
            aggregated[f'{key}_std'] = np.nanstd(precs, ddof=1)

    return aggregated


def compute_median_and_rmad(values: np.ndarray) -> Tuple[float, float]:
    """
    Compute median and robust MAD (Median Absolute Deviation).

    RMAD = 100 * 1.4826 * MAD / median (as percentage)

    Args:
        values: Array of values

    Returns:
        Tuple of (median, rmad_percentage)
    """
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if median != 0:
        rmad = 100.0 * 1.4826 * mad / median
    else:
        rmad = np.nan

    return median, rmad
