"""
Metrics Computation for LiRA Attack Evaluation

This module provides utilities for computing standard evaluation metrics
for membership inference attacks, including:

1. ROC Analysis: Full ROC curves, AUC, accuracy
2. Threshold Selection: TPR/FPR at specific operating points
3. Precision/Recall: With configurable membership priors
4. Confusion Matrices: At specific thresholds
5. Aggregation: Cross-model statistics for leave-one-out evaluation
6. Robust Statistics: Median and MAD for outlier resistance

Author: N.Jebreel
Date: 2025
"""

import numpy as np
from typing import Tuple, Dict, List
from sklearn.metrics import roc_curve, auc as sklearn_auc, roc_auc_score


# =============================================================================
# ROC CURVE COMPUTATION
# =============================================================================

def compute_roc_metrics(
    scores: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Compute complete ROC curve and derived metrics.
    
    Args:
        scores: Attack scores (higher = more likely member)
        labels: Ground truth membership labels (boolean or 0/1)
        
    Returns:
        Tuple of:
        - fpr: False positive rates
        - tpr: True positive rates  
        - thresholds: Decision thresholds
        - auc: Area under ROC curve
        - accuracy: Maximum balanced accuracy
        
    Example:
        >>> fpr, tpr, thresh, auc, acc = compute_roc_metrics(scores, labels)
        >>> print(f"AUC: {auc:.4f}, Best accuracy: {acc:.4f}")
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Compute AUC
    auc_score = sklearn_auc(fpr, tpr)
    
    # Compute maximum balanced accuracy: 1 - (FPR + FNR)/2 = 1 - (FPR + (1-TPR))/2
    accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)
    
    return fpr, tpr, thresholds, auc_score, accuracy


def compute_tpr_at_fpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float
) -> Tuple[float, float, float]:
    """
    Find TPR and threshold at a target FPR.
    
    Selects the largest threshold where FPR <= target_fpr, which gives the
    highest TPR achievable at that FPR constraint.
    
    Args:
        fpr: False positive rates from ROC curve
        tpr: True positive rates from ROC curve
        thresholds: Thresholds from ROC curve
        target_fpr: Target false positive rate
        
    Returns:
        Tuple of:
        - tpr_at_target: TPR achieved at target FPR
        - actual_fpr: Actual FPR (≤ target_fpr)
        - threshold: Decision threshold
        
    Example:
        >>> fpr, tpr, thresh, _, _ = compute_roc_metrics(scores, labels)
        >>> tpr_001, fpr_actual, tau = compute_tpr_at_fpr(fpr, tpr, thresh, 0.001)
        >>> print(f"At FPR≤0.1%: TPR={tpr_001:.3f}, threshold={tau:.3f}")
    """
    # Find all thresholds where FPR <= target
    idx = np.where(fpr <= target_fpr)[0]
    
    if len(idx) == 0:
        return 0.0, 0.0, np.inf
    
    # Take last index (largest threshold, highest TPR)
    j = idx[-1]
    return float(tpr[j]), float(fpr[j]), float(thresholds[j])


# =============================================================================
# PRECISION AND CONFUSION MATRICES
# =============================================================================

def compute_precision(
    tpr: float,
    fpr: float,
    prior: float = 0.5
) -> float:
    """
    Compute precision given TPR, FPR, and membership prior.
    
    Uses Bayes' theorem to compute positive predictive value:
    
    Precision = P(member | predicted_member)
              = (prior × TPR) / (prior × TPR + (1-prior) × FPR)
    
    This adjusts for the base rate of membership in the population,
    which is critical for interpreting attack effectiveness.
    
    Args:
        tpr: True positive rate
        fpr: False positive rate
        prior: Prior probability of membership (e.g., 0.5 for balanced dataset)
        
    Returns:
        Precision value (NaN if no positive predictions)
        
    Example:
        >>> # With 1% membership prior, even high TPR has low precision
        >>> prec_1pct = compute_precision(tpr=0.9, fpr=0.01, prior=0.01)
        >>> prec_50pct = compute_precision(tpr=0.9, fpr=0.01, prior=0.5)
        >>> print(f"1% prior: {prec_1pct:.3f}, 50% prior: {prec_50pct:.3f}")
        >>> # Output: "1% prior: 0.476, 50% prior: 0.989"
    """
    numerator = prior * tpr
    denominator = prior * tpr + (1 - prior) * fpr
    
    return numerator / denominator if denominator > 0 else np.nan


def compute_confusion_matrix_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute confusion matrix at a specific threshold.
    
    Predictions are made by: score >= threshold → predict member
    
    Args:
        scores: Attack scores (higher = more likely member)
        labels: Ground truth labels (boolean or 0/1)
        threshold: Decision threshold
        
    Returns:
        Dictionary with keys:
        - tp, fp, tn, fn: Confusion matrix counts
        - tpr, fpr: True/false positive rates
        - precision, recall: Precision and recall (recall = TPR)
        
    Example:
        >>> results = compute_confusion_matrix_at_threshold(scores, labels, 0.5)
        >>> print(f"Precision: {results['precision']:.3f}, Recall: {results['recall']:.3f}")
    """
    predictions = scores >= threshold
    labels_bool = np.asarray(labels, dtype=bool)
    
    tp = int(np.sum(predictions & labels_bool))
    fp = int(np.sum(predictions & ~labels_bool))
    tn = int(np.sum(~predictions & ~labels_bool))
    fn = int(np.sum(~predictions & labels_bool))
    
    # Compute rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tpr  # Recall is identical to TPR
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


# =============================================================================
# MULTI-POINT EVALUATION
# =============================================================================

def compute_metrics_at_multiple_fprs(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fprs: List[float],
    priors: List[float] = [0.5]
) -> Dict:
    """
    Compute metrics at multiple target FPR values and priors.
    
    Useful for evaluating attack performance at different operating points
    (e.g., 0.1% FPR and 1% FPR) and under different membership assumptions.
    
    Args:
        scores: Attack scores
        labels: Ground truth labels
        target_fprs: List of target FPR values (e.g., [0.001, 0.01])
        priors: List of membership priors (e.g., [0.01, 0.1, 0.5])
        
    Returns:
        Dictionary with:
        - auc, accuracy: Overall metrics
        - fpr_values: List of (target, actual) FPR pairs
        - tpr_values: List of TPRs at each target FPR
        - thresholds: List of thresholds
        - precisions: Dict of precision values for each (prior, fpr) combo
        
    Example:
        >>> results = compute_metrics_at_multiple_fprs(
        ...     scores, labels,
        ...     target_fprs=[0.001, 0.01],
        ...     priors=[0.01, 0.5]
        ... )
        >>> print(f"AUC: {results['auc']:.4f}")
        >>> for i, tfpr in enumerate([0.001, 0.01]):
        ...     print(f"At FPR={tfpr}: TPR={results['tpr_values'][i]:.3f}")
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


# =============================================================================
# CROSS-MODEL AGGREGATION (for leave-one-out evaluation)
# =============================================================================

def aggregate_metrics_across_models(
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    target_fprs: List[float] = [0.001],
    priors: List[float] = [0.5]
) -> Dict:
    """
    Aggregate metrics across multiple target models (e.g., leave-one-out).
    
    Computes mean and standard deviation of metrics across models to assess
    consistency and variability in attack performance.
    
    Args:
        scores_list: List of score arrays, one per target model [N samples each]
        labels_list: List of label arrays, one per target model [N samples each]
        target_fprs: List of target FPR values
        priors: List of membership priors
        
    Returns:
        Dictionary with mean and std of:
        - AUC and accuracy
        - TPR at each target FPR
        - Precision for each (prior, fpr) combination
        
    Example:
        >>> # Leave-one-out evaluation with M models
        >>> aggregated = aggregate_metrics_across_models(
        ...     [scores[m] for m in range(M)],
        ...     [labels[m] for m in range(M)],
        ...     target_fprs=[0.001],
        ...     priors=[0.5]
        ... )
        >>> print(f"AUC: {aggregated['auc_mean']:.4f} ± {aggregated['auc_std']:.4f}")
    """
    all_results = []
    
    for scores, labels in zip(scores_list, labels_list):
        results = compute_metrics_at_multiple_fprs(scores, labels, target_fprs, priors)
        all_results.append(results)
    
    # Aggregate overall metrics
    aggregated = {
        'auc_mean': np.mean([r['auc'] for r in all_results]),
        'auc_std': np.std([r['auc'] for r in all_results], ddof=1),
        'accuracy_mean': np.mean([r['accuracy'] for r in all_results]),
        'accuracy_std': np.std([r['accuracy'] for r in all_results], ddof=1),
    }
    
    # Aggregate TPR at each FPR
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


# =============================================================================
# ROBUST STATISTICS
# =============================================================================

def compute_median_and_rmad(values: np.ndarray) -> Tuple[float, float]:
    """
    Compute median and robust MAD (Median Absolute Deviation).
    
    MAD is more robust to outliers than standard deviation. RMAD expresses
    MAD as a percentage of the median for interpretability.
    
    RMAD = 100 × 1.4826 × MAD / median (as percentage)
    
    The 1.4826 factor makes MAD comparable to std dev for normal distributions.
    
    Args:
        values: Array of values
        
    Returns:
        Tuple of:
        - median: Median value
        - rmad_percentage: Robust MAD as percentage of median
        
    Example:
        >>> thresholds = np.array([0.45, 0.48, 0.50, 0.51, 2.0])  # One outlier
        >>> median, rmad = compute_median_and_rmad(thresholds)
        >>> print(f"Median: {median:.3f}, RMAD: {rmad:.1f}%")
        >>> # RMAD handles outlier better than std would
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