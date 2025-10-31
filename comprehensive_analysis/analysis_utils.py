"""
Analysis Utilities for LiRA Membership Inference Attacks

This module provides comprehensive utilities for loading, processing, and analyzing
LiRA (Likelihood Ratio Attack) experiment results. It includes:

1. Data Loading: Experiment configs, attack scores, membership labels, datasets
2. Threshold Computation: ROC-based threshold selection for target/shadow models
3. Metrics: Confusion matrices, precision computation with priors
4. Vulnerability Analysis: Per-sample statistics and ranking
5. Visualization: Image grids with vulnerability badges
6. Set Operations: For cross-run agreement analysis

Author: Najeeb Jebreel, optmized by Cloude Sonnet 4.5
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from itertools import combinations

from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100, GTSRB
import torchvision.utils as vutils


# =============================================================================
# CONFIGURATION & FILE MANAGEMENT
# =============================================================================

def load_experiment_config(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load experiment configuration YAML files.
    
    Looks for 'train_config.yaml' and 'attack_config.yaml' in the experiment
    directory and returns their contents.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with keys 'train_config' and/or 'attack_config'
        
    Example:
        >>> configs = load_experiment_config("experiments/cifar10/resnet18")
        >>> num_models = configs['train_config']['training']['num_shadow_models']
    """
    experiment_dir = Path(experiment_dir)
    configs = {}
    
    train_config_path = experiment_dir / 'train_config.yaml'
    attack_config_path = experiment_dir / 'attack_config.yaml'
    
    if train_config_path.exists():
        with open(train_config_path, 'r') as f:
            configs['train_config'] = yaml.safe_load(f)
    
    if attack_config_path.exists():
        with open(attack_config_path, 'r') as f:
            configs['attack_config'] = yaml.safe_load(f)
    
    return configs


def create_output_directory(exp_path: Path) -> Path:
    """
    Create structured output directory from experiment path.
    
    Extracts dataset/model/config from path structure and creates:
    analysis_results/{dataset}/{model}/{config}/
    
    Args:
        exp_path: Path to experiment directory
        
    Returns:
        Created output directory path
        
    Example:
        Path: experiments/cifar10/resnet18/weak_aug
        Output: analysis_results/cifar10/resnet18/weak_aug/
    """
    parts = exp_path.parts
    
    if len(parts) >= 4:
        dataset, model, config = parts[-3], parts[-2], parts[-1]
    else:
        # Fallback for shorter paths
        dataset, model, config = "unknown_dataset", "unknown_model", exp_path.name
    
    out_dir = Path("analysis_results") / dataset / model / config
    out_dir.mkdir(parents=True, exist_ok=True)
    
    return out_dir


def get_experiment_info(experiment_dir: Union[str, Path]) -> Dict:
    """
    Extract comprehensive experiment metadata.
    
    Parses both directory structure and configuration files to gather
    information about the experiment setup.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with experiment metadata including:
        - experiment_dir, dataset, model, config (from path)
        - num_shadow_models, epochs, architecture, dataset_name (from configs)
    """
    experiment_dir = Path(experiment_dir)
    parts = experiment_dir.parts
    
    info = {
        'experiment_dir': str(experiment_dir),
        'dataset': parts[-3] if len(parts) >= 3 else 'unknown',
        'model': parts[-2] if len(parts) >= 2 else 'unknown',
        'config': parts[-1] if len(parts) >= 1 else 'unknown',
    }
    
    # Augment with config details
    configs = load_experiment_config(experiment_dir)
    if 'train_config' in configs:
        tc = configs['train_config']
        info['num_shadow_models'] = tc.get('training', {}).get('num_shadow_models', 'unknown')
        info['epochs'] = tc.get('training', {}).get('epochs', 'unknown')
        info['architecture'] = tc.get('model', {}).get('architecture', 'unknown')
        info['dataset_name'] = tc.get('dataset', {}).get('name', 'unknown')
    
    return info


# =============================================================================
# DATA LOADING: SCORES, LABELS, DATASETS
# =============================================================================

def load_experiment_data(
    exp_path: Path,
    score_files: Dict[str, str],
    labels_file: str
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load membership labels and attack scores from experiment directory.
    
    This is the main data loading function for analysis. It reads ground truth
    membership labels and all attack score arrays, validating dimensions.
    
    Args:
        exp_path: Experiment directory path
        score_files: Mapping of attack names to score filenames
        labels_file: Membership labels filename (usually "membership_labels.npy")
        
    Returns:
        Tuple of:
        - labels: Boolean array [M, N] where M=models, N=samples
        - scores: Dict mapping attack names to score arrays [M, N]
        
    Raises:
        ValueError: If any score array dimensions don't match labels
        
    Example:
        >>> labels, scores = load_experiment_data(
        ...     Path("experiments/cifar10"),
        ...     {"LiRA (online)": "online_scores.npy"},
        ...     "membership_labels.npy"
        ... )
        >>> M, N = labels.shape  # M models, N samples
    """
    # Load ground truth membership labels
    labels = np.load(exp_path / labels_file)
    labels = labels.astype(bool, copy=False)
    M, N = labels.shape
    
    # Load all attack scores and validate dimensions
    scores = {}
    for attack_name, filename in score_files.items():
        arr = np.load(exp_path / filename)
        
        if arr.shape != (M, N):
            raise ValueError(
                f"{attack_name} dimension mismatch: got {arr.shape}, expected {(M, N)}"
            )
        
        scores[attack_name] = arr
    
    return labels, scores


def load_attack_scores(
    experiment_dir: Union[str, Path],
    mode: str = 'leave_one_out'
) -> Dict[str, np.ndarray]:
    """
    Load all attack score files using standard naming convention.
    
    Convenience function that uses predefined filenames for common attack variants.
    
    Args:
        experiment_dir: Path to experiment directory
        mode: 'leave_one_out' or 'single' (only leave_one_out supported currently)
        
    Returns:
        Dictionary mapping attack names to score arrays
        
    Example:
        >>> scores = load_attack_scores("experiments/cifar10/resnet18")
        >>> online_scores = scores["LiRA (online)"]
    """
    experiment_dir = Path(experiment_dir)
    
    if mode == 'leave_one_out':
        score_files = {
            'LiRA (online)': 'online_scores_leave_one_out.npy',
            'LiRA (online, fixed var)': 'online_fixed_scores_leave_one_out.npy',
            'LiRA (offline)': 'offline_scores_leave_one_out.npy',
            'LiRA (offline, fixed var)': 'offline_fixed_scores_leave_one_out.npy',
            'Global threshold': 'global_scores_leave_one_out.npy',
        }
    else:
        raise ValueError(f"Mode '{mode}' not supported yet. Use 'leave_one_out'.")
    
    scores = {}
    for attack_name, filename in score_files.items():
        filepath = experiment_dir / filename
        if filepath.exists():
            scores[attack_name] = np.load(filepath)
        else:
            print(f"Warning: {filename} not found in {experiment_dir}")
    
    return scores


def load_membership_labels(experiment_dir: Union[str, Path]) -> np.ndarray:
    """
    Load ground truth membership labels with fallback.
    
    Tries 'membership_labels.npy' first, then falls back to 'keep_indices.npy'
    for backward compatibility.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Boolean array of shape [M, N] where M=models, N=samples
        
    Raises:
        FileNotFoundError: If neither file exists
    """
    experiment_dir = Path(experiment_dir)
    labels_path = experiment_dir / 'membership_labels.npy'
    
    if not labels_path.exists():
        labels_path = experiment_dir / 'keep_indices.npy'
    
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Neither membership_labels.npy nor keep_indices.npy found in {experiment_dir}"
        )
    
    labels = np.load(labels_path)
    return labels.astype(bool)


def load_dataset(config: Dict, data_dir: str = "D:/mona/mia_research/data") -> Tuple[object, np.ndarray]:
    """
    Load full dataset (train + test) for visualization and analysis.
    
    Combines train and test splits without transforms for analysis purposes.
    Supports CIFAR-10, CIFAR-100, GTSRB, and Purchase-100.
    
    Args:
        config: Configuration dictionary with dataset name
        data_dir: Root directory containing datasets
        
    Returns:
        Tuple of:
        - full_dataset: ConcatDataset or None (for tabular data)
        - full_label: Numpy array of all labels
        
    Raises:
        ValueError: If dataset name is unsupported
        
    Example:
        >>> config = {"dataset": {"name": "CIFAR10"}}
        >>> dataset, labels = load_dataset(config)
        >>> img, label = dataset[0]
    """
    dataset_name = config['dataset']['name'].lower()
    
    if dataset_name == 'cifar10':
        train_dataset = CIFAR10(root=data_dir, train=True, download=False)
        test_dataset = CIFAR10(root=data_dir, train=False, download=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_label = np.array(train_dataset.targets)
        test_label = np.array(test_dataset.targets)
        full_label = np.concatenate((train_label, test_label), axis=0)
    
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100(root=data_dir, train=True, download=False)
        test_dataset = CIFAR100(root=data_dir, train=False, download=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_label = np.array(train_dataset.targets)
        test_label = np.array(test_dataset.targets)
        full_label = np.concatenate((train_label, test_label), axis=0)
    
    elif dataset_name == 'gtsrb':
        train_dataset = GTSRB(root=data_dir, split='train', download=False)
        test_dataset = GTSRB(root=data_dir, split='test', download=False)
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        train_label = np.array([sample[1] for sample in train_dataset._samples])
        test_label = np.array([sample[1] for sample in test_dataset._samples])
        full_label = np.concatenate((train_label, test_label), axis=0)
    
    elif dataset_name == 'purchase':
        data_path = os.path.join(data_dir, dataset_name, 'features_labels.npy')
        full_data = np.load(data_path)
        full_label = full_data[:, 0].astype(np.int32) - 1
        full_dataset = None  # Tabular data has no image dataset object
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return full_dataset, full_label


# =============================================================================
# THRESHOLD COMPUTATION (TARGET & SHADOW)
# =============================================================================

def find_threshold_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find largest threshold where FPR(score >= τ) <= target_fpr.
    
    This implements the standard ROC-based threshold selection. Uses the full
    ROC curve (drop_intermediate=False) for precise threshold placement.
    
    Args:
        scores: Attack scores for one model [N samples]
        labels: Ground truth labels for one model [N samples]
        target_fpr: Target false positive rate (e.g., 0.001 for 0.1% FPR)
        
    Returns:
        Tuple of:
        - threshold: Largest τ meeting FPR constraint (inf if none exists)
        - achieved_fpr: Actual FPR at this threshold (None if no valid τ)
        - tpr_at_threshold: TPR at this threshold (None if no valid τ)
        
    Example:
        >>> tau, fpr, tpr = find_threshold_at_fpr(scores, labels, 0.001)
        >>> print(f"At τ={tau:.3f}: FPR={fpr:.4f}, TPR={tpr:.4f}")
    """
    fpr, tpr, thresholds = roc_curve(
        labels.astype(bool),
        scores,
        drop_intermediate=False  # Keep all points for precision
    )
    
    # Find all thresholds satisfying FPR constraint
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    if valid_indices.size == 0:
        return np.inf, None, None
    
    # Take largest threshold (most conservative choice)
    idx = valid_indices[-1]
    
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def compute_shadow_thresholds(target_thresholds: np.ndarray, exclude_idx: int) -> float:
    """
    Compute shadow threshold: median of other models' thresholds.
    
    This evaluates threshold transferability by using thresholds learned from
    shadow models to attack a target model. The median provides robustness
    against outliers.
    
    Args:
        target_thresholds: Array of per-model thresholds [M models]
        exclude_idx: Index of target model to exclude from computation
        
    Returns:
        Median threshold from remaining models (inf if no valid thresholds)
        
    Example:
        >>> # For each target model, compute shadow threshold from others
        >>> shadow_taus = [compute_shadow_thresholds(all_taus, m) for m in range(M)]
    """
    pool = np.delete(target_thresholds, exclude_idx)
    pool = pool[np.isfinite(pool)]  # Remove inf/nan values
    
    return float(np.median(pool)) if pool.size > 0 else np.inf


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_confusion_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Tuple[int, int, int, int, float, float]:
    """
    Compute confusion matrix and rates at a given threshold.
    
    Predictions are made by comparing scores against threshold (score >= τ → member).
    
    Args:
        scores: Attack scores [N samples]
        labels: Ground truth labels [N samples]
        threshold: Decision threshold
        
    Returns:
        Tuple of:
        - tp: True positives (correctly identified members)
        - fp: False positives (non-members incorrectly flagged)
        - tn: True negatives (correctly identified non-members)
        - fn: False negatives (members missed)
        - tpr: True positive rate = TP / (TP + FN)
        - fpr_achieved: False positive rate = FP / (FP + TN)
        
    Example:
        >>> tp, fp, tn, fn, tpr, fpr = compute_confusion_matrix(scores, labels, 0.5)
        >>> print(f"TPR: {tpr:.3f}, FPR: {fpr:.3f}, Precision: {tp/(tp+fp):.3f}")
    """
    predictions = scores >= threshold
    
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    tn = int(np.sum(~predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    
    # Compute rates (handle division by zero)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_achieved = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return tp, fp, tn, fn, tpr, fpr_achieved


def compute_precision_from_rates(tpr: float, fpr: float, prior: float) -> float:
    """
    Compute precision given TPR, FPR, and membership prior.
    
    Uses Bayes' theorem to compute:
    Precision = P(member | predicted_member)
              = (prior × TPR) / (prior × TPR + (1-prior) × FPR)
    
    This accounts for the base rate of membership in the population.
    
    Args:
        tpr: True positive rate
        fpr: False positive rate (achieved, not target)
        prior: Prior probability of membership (e.g., 0.5 for balanced)
        
    Returns:
        Precision value (NaN if no positive predictions)
        
    Example:
        >>> # Low prior (1% members) dramatically affects precision
        >>> prec_1pct = compute_precision_from_rates(0.8, 0.001, 0.01)
        >>> prec_50pct = compute_precision_from_rates(0.8, 0.001, 0.5)
        >>> print(f"1% prior: {prec_1pct:.3f}, 50% prior: {prec_50pct:.3f}")
    """
    numerator = tpr * prior
    denominator = tpr * prior + fpr * (1 - prior)
    
    return numerator / denominator if denominator > 0 else np.nan


def validate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    expected_fpr: float,
    expected_tpr: float,
    tolerance: float = 1e-12
) -> bool:
    """
    Sanity check: verify threshold produces expected FPR/TPR.
    
    Recomputes metrics at the given threshold and checks they match expected
    values within tolerance. Useful for validating ROC-based threshold extraction.
    
    Args:
        scores: Attack scores
        labels: Ground truth labels
        threshold: Threshold to validate
        expected_fpr: Expected false positive rate
        expected_tpr: Expected true positive rate
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if both FPR and TPR match expected values
        
    Example:
        >>> tau, fpr_exp, tpr_exp = find_threshold_at_fpr(scores, labels, 0.001)
        >>> is_valid = validate_threshold(scores, labels, tau, fpr_exp, tpr_exp)
        >>> assert is_valid, "Threshold extraction failed validation"
    """
    _, _, _, _, tpr_actual, fpr_actual = compute_confusion_matrix(
        scores, labels, threshold
    )
    
    fpr_match = np.isclose(fpr_actual, expected_fpr, atol=tolerance)
    tpr_match = np.isclose(tpr_actual, expected_tpr, atol=tolerance)
    
    return fpr_match and tpr_match


# =============================================================================
# PER-SAMPLE VULNERABILITY ANALYSIS
# =============================================================================

def compute_per_sample_confusion_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Compute confusion matrix statistics per sample across models.
    
    For each sample, counts how many models correctly/incorrectly classify it.
    Useful for identifying samples that are consistently vulnerable or robust.
    
    Args:
        scores: Attack scores [M models, N samples]
        labels: Ground truth labels [M models, N samples]
        threshold: Decision threshold (can be scalar or array [M])
        
    Returns:
        DataFrame with columns: sample_id, tp, fp, tn, fn
        where each count is across M models for that sample
        
    Example:
        >>> sample_stats = compute_per_sample_confusion_matrix(scores, labels, 0.5)
        >>> vulnerable = sample_stats[(sample_stats['fp'] == 0) & (sample_stats['tp'] > 0)]
        >>> print(f"Found {len(vulnerable)} highly vulnerable samples")
    """
    M, N = scores.shape
    
    # Predictions: higher score = member
    predictions = scores >= threshold
    labels_bool = labels.astype(bool)
    
    # Count across models (axis=0) for each sample
    tp = np.sum(predictions & labels_bool, axis=0).astype(int)
    fp = np.sum(predictions & ~labels_bool, axis=0).astype(int)
    tn = np.sum(~predictions & ~labels_bool, axis=0).astype(int)
    fn = np.sum(~predictions & labels_bool, axis=0).astype(int)
    
    return pd.DataFrame({
        'sample_id': np.arange(N),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    })


def rank_samples_by_vulnerability(
    confusion_df: pd.DataFrame,
    sort_by: str = 'low_fp_high_tp'
) -> pd.DataFrame:
    """
    Rank samples by vulnerability to membership inference.
    
    Provides multiple ranking strategies to identify vulnerable samples.
    
    Args:
        confusion_df: DataFrame from compute_per_sample_confusion_matrix
        sort_by: Ranking strategy:
            - 'low_fp_high_tp': Prioritize low FP (stable), then high TP (detectable)
            - 'high_tp_low_fp': Prioritize high TP (detectable), then low FP (stable)
            - 'vulnerability_score': Sort by TP - FP difference
            
    Returns:
        Sorted DataFrame with most vulnerable samples first
        
    Example:
        >>> ranked = rank_samples_by_vulnerability(sample_df, 'low_fp_high_tp')
        >>> top_10 = ranked.head(10)  # Most vulnerable samples
    """
    df = confusion_df.copy()
    
    if sort_by == 'low_fp_high_tp':
        # Samples rarely flagged as non-members but often detected as members
        df = df.sort_values(by=['fp', 'tp'], ascending=[True, False])
    
    elif sort_by == 'high_tp_low_fp':
        # Samples often detected as members, rarely as non-members
        df = df.sort_values(by=['tp', 'fp'], ascending=[False, True])
    
    elif sort_by == 'vulnerability_score':
        # Simple difference score: positive = vulnerable, negative = robust
        df['vulnerability'] = df['tp'] - df['fp']
        df = df.sort_values(by='vulnerability', ascending=False)
    
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")
    
    return df.reset_index(drop=True)


def get_highly_vulnerable_samples(
    confusion_df: pd.DataFrame,
    min_tp: int = 1,
    max_fp: int = 0
) -> pd.DataFrame:
    """
    Get highly vulnerable samples with specific criteria.
    
    Default criteria (min_tp=1, max_fp=0) identifies samples that are:
    - Detected as members at least once (TP >= 1)
    - Never falsely flagged as non-members (FP = 0)
    
    Args:
        confusion_df: DataFrame from compute_per_sample_confusion_matrix
        min_tp: Minimum true positive count
        max_fp: Maximum false positive count
        
    Returns:
        Filtered DataFrame with highly vulnerable samples
        
    Example:
        >>> highly_vuln = get_highly_vulnerable_samples(sample_df, min_tp=5, max_fp=0)
        >>> print(f"Samples detected by ≥5 models with 0 false alarms: {len(highly_vuln)}")
    """
    return confusion_df[
        (confusion_df['tp'] >= min_tp) & (confusion_df['fp'] <= max_fp)
    ]


# =============================================================================
# VISUALIZATION
# =============================================================================

def _to_chw_float_tensor(img):
    """
    Convert image to CHW float tensor [0,1] for visualization.
    
    Handles various input formats: PIL, numpy arrays, torch tensors.
    Ensures 3-channel output (RGB) by replicating grayscale.
    """
    if isinstance(img, torch.Tensor):
        t = img.clone()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)
        t = t.float()
        if t.numel() and t.max() > 1.0:
            t = t / 255.0
    else:
        arr = np.array(img)
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        t = t.float()
        if t.numel() and t.max() > 1.0:
            t = t / 255.0
    
    # Ensure 3 channels
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] > 3:
        t = t[:3]
    
    return t


def display_top_k_vulnerable_samples(
    vulnerable_samples: pd.DataFrame,
    full_dataset,
    k: int = 20,
    nrow: int = 5,
    padding: int = 2,
    normalize: bool = True,
    dpi: int = 300,
    out_dir: Union[Path, str] = ".",
    save_name: str = "vulnerable_samples.png",
    sample_id_col: str = "sample_id",
    font_size: int = 8,
    badge_margin: int = 2,
    overhang_left: int = 6,
    overhang_up: int = 6
):
    """
    Create image grid of top-k vulnerable samples with TP/FP badges.
    
    Displays samples in a grid with vulnerability statistics (TP, FP) overlaid
    in the top-left corner of each image.
    
    Args:
        vulnerable_samples: DataFrame with vulnerability rankings (from rank_samples_by_vulnerability)
        full_dataset: Dataset object supporting indexing (dataset[idx] → (image, label))
        k: Number of samples to display
        nrow: Number of images per row
        padding: Pixel padding between images
        normalize: Whether to normalize image intensities
        dpi: Output image resolution
        out_dir: Output directory for saving
        save_name: Output filename
        sample_id_col: Column name containing sample IDs
        font_size: Font size for badges
        badge_margin: Distance from tile corner to badge
        overhang_left: Extra pixels to shift badge left (can go into padding)
        overhang_up: Extra pixels to shift badge up (can go into padding)
        
    Returns:
        Path to saved image
        
    Example:
        >>> display_top_k_vulnerable_samples(
        ...     vulnerable_samples=ranked_samples,
        ...     full_dataset=cifar_dataset,
        ...     k=20,
        ...     out_dir="analysis_results/cifar10"
        ... )
    """
    # Validate input
    if not isinstance(vulnerable_samples, pd.DataFrame):
        raise TypeError("vulnerable_samples must be a pandas DataFrame")
    
    for col in (sample_id_col, "tp", "fp"):
        if col not in vulnerable_samples.columns:
            raise KeyError(f"Column '{col}' missing from vulnerable_samples")
    
    # Select top-k samples
    vs = vulnerable_samples.head(k).copy()
    ids = vs[sample_id_col].to_numpy()
    
    # Load and convert images
    images = [_to_chw_float_tensor(full_dataset[int(sid)][0]) for sid in ids]
    tensor = torch.stack(images)  # [k, 3, H, W]
    
    # Create grid
    grid = vutils.make_grid(
        tensor, nrow=nrow, padding=padding, normalize=normalize, pad_value=1.0
    )
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Setup figure
    rows = (k + nrow - 1) // nrow
    H, W = tensor.shape[-2], tensor.shape[-1]
    fig_w = max(4, nrow * 1.2)
    fig_h = max(4, rows * 1.2)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(grid_np, aspect="equal")
    ax.axis("off")
    
    # Add badges with TP/FP counts
    stride_x, stride_y = W + padding, H + padding
    base_x = base_y = padding
    
    for i in range(len(ids)):
        r, c = divmod(i, nrow)
        x0 = base_x + c * stride_x  # Tile's left edge
        y0 = base_y + r * stride_y  # Tile's top edge
        
        tp_val = int(vs.iloc[i]["tp"])
        fp_val = int(vs.iloc[i]["fp"])
        text = f"TP:{tp_val}  FP:{fp_val}"
        
        # Position badge in top-left corner with overhang
        x_text = x0 + badge_margin - overhang_left
        y_text = y0 + badge_margin - overhang_up
        
        ax.text(
            x_text, y_text, text,
            ha="left", va="top",
            fontsize=font_size, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9,
                      edgecolor="black", linewidth=0.7),
            color="black",
            clip_on=False,  # Allow overhang beyond axes/tile
        )
    
    plt.tight_layout(pad=0.1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / save_name
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved grid: {out_path}")
    return out_path


# =============================================================================
# SET OPERATIONS FOR CROSS-RUN AGREEMENT ANALYSIS
# =============================================================================

CSV_BASENAME = "samples_vulnerability_ranked_online_shadow_0p001pct.csv"
SAMPLE_ID_COL = "sample_id"


def _csv_path(p: Union[str, Path]) -> Path:
    """
    Resolve CSV path from directory or file path.
    
    Helper function that handles both directory and file inputs with validation.
    
    Args:
        p: Directory containing CSV_BASENAME or direct CSV file path
        
    Returns:
        Validated CSV file path
        
    Raises:
        FileNotFoundError: If path doesn't exist or isn't a CSV
    """
    p = Path(p)
    
    if p.is_dir():
        p = p / CSV_BASENAME
    
    if p.suffix.lower() != ".csv":
        raise FileNotFoundError(f"Expected a directory or CSV; got: {p}")
    
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    
    return p


def to_id_set(
    path_or_dir: Union[str, Path],
    col: str = SAMPLE_ID_COL,
    tp_threshold: int = 1
) -> set:
    """
    Load CSV and return set of sample IDs where TP >= tp_threshold.
    
    This function bridges per-run CSVs to set algebra for agreement analysis.
    Filters samples based on their TP count to ensure sufficient support.
    
    Args:
        path_or_dir: Directory containing CSV_BASENAME, or direct CSV path
        col: Column name storing sample IDs
        tp_threshold: Minimum TP count to include sample (e.g., TP≥2 means
                     at least 2 shadow models detected it)
        
    Returns:
        Set of sample IDs (as strings) meeting the TP threshold
        
    Example:
        >>> # Get samples detected by ≥3 models in each run
        >>> set_a = to_id_set("run_a/", tp_threshold=3)
        >>> set_b = to_id_set("run_b/", tp_threshold=3)
        >>> intersection = set_a & set_b  # Samples vulnerable in both runs
    """
    csvp = _csv_path(path_or_dir)
    df = pd.read_csv(csvp)
    
    # Validate required columns
    if col not in df.columns:
        raise KeyError(f"Expected '{col}' in {csvp}; got {list(df.columns)}")
    if "tp" not in df.columns:
        raise KeyError(f"Expected 'tp' column in {csvp}")
    
    # Filter by TP threshold and convert to set
    filtered = df[df["tp"] >= tp_threshold]
    return set(filtered[col].dropna().astype(str))


def avg_agreement(named_sets: List[Tuple[str, set]], k: int) -> Tuple[float, float, float]:
    """
    Compute average Jaccard, intersection, and union over all C(M,k) combinations.
    
    For each combination of k sets:
    - Intersection: samples vulnerable in ALL k runs (high confidence)
    - Union: samples vulnerable in ANY of k runs (broad capture)
    - Jaccard: |intersection| / |union| (agreement ratio)
    
    Args:
        named_sets: List of (name, set_of_ids) tuples for M runs
        k: Number of runs to combine at a time
        
    Returns:
        Tuple of:
        - avg_jaccard: Mean Jaccard index across all C(M,k) combinations
        - avg_intersection: Mean intersection size
        - avg_union: Mean union size
        
    Example:
        >>> runs = [("run1", set1), ("run2", set2), ("run3", set3)]
        >>> # Average agreement across all pairs (k=2)
        >>> j, inter, union = avg_agreement(runs, k=2)
        >>> print(f"Avg Jaccard: {j:.3f}, Avg overlap: {inter:.1f}/{union:.1f}")
    """
    jaccard_scores, intersections, unions = [], [], []
    
    for combo in combinations(named_sets, k):
        sets = [s for _, s in combo]
        inter_set = set.intersection(*sets)
        union_set = set.union(*sets)
        
        if not union_set:
            continue  # Skip empty unions
        
        jaccard_scores.append(len(inter_set) / len(union_set))
        intersections.append(len(inter_set))
        unions.append(len(union_set))
    
    if not jaccard_scores:
        return np.nan, np.nan, np.nan
    
    return (
        float(np.mean(jaccard_scores)),
        float(np.mean(intersections)),
        float(np.mean(unions))
    )


def compute_scenario(
    label: str,
    named_sets: List[Tuple[str, set]],
    kmin: Optional[int] = None,
    kmax: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute agreement metrics for a scenario across k = kmin..kmax.
    
    Evaluates how agreement changes with the number of runs combined.
    As k increases, intersection typically decreases (stricter requirement)
    while Jaccard index shows agreement stability.
    
    Args:
        label: Scenario description (e.g., 'Identical (2-5 seeds)', '+1 different (Arch)')
        named_sets: List of (name, set) tuples to evaluate
        kmin: Minimum k value (default: 2)
        kmax: Maximum k value (default: M, number of runs)
        
    Returns:
        DataFrame with columns: scenario, M, k, avg_jaccard, avg_intersection, avg_union
        
    Example:
        >>> # Compare identical vs varied training runs
        >>> df_identical = compute_scenario("Identical seeds", identical_runs)
        >>> df_varied = compute_scenario("Different seeds", varied_runs)
        >>> # Plot how agreement degrades with k
    """
    M = len(named_sets)
    lo = 2 if kmin is None else kmin
    hi = M if kmax is None else kmax
    
    rows = []
    for k in range(lo, hi + 1):
        j, inter, union = avg_agreement(named_sets, k)
        rows.append({
            'scenario': label,
            'M': M,
            'k': k,
            'avg_jaccard': j,
            'avg_intersection': inter,
            'avg_union': union
        })
    
    return pd.DataFrame(rows)


def aggregate_over_configs(
    label: str,
    list_of_named_sets: List[List[Tuple[str, set]]]
) -> pd.DataFrame:
    """
    Average metrics across multiple config lists (e.g., all two-change variants).
    
    Useful for aggregating results when testing multiple configurations of the
    same type (e.g., different pairs of hyperparameters changed).
    
    Args:
        label: Label for this aggregated scenario
        list_of_named_sets: List of named_sets lists, one per config variant
            Example: [
                [("base", set_base), ("var_A", set_A), ("var_B", set_B)],
                [("base", set_base), ("var_A", set_A), ("var_C", set_C)],
                ...
            ]
        
    Returns:
        DataFrame with pointwise-averaged metrics across configs
        
    Example:
        >>> # Test multiple architecture pairs
        >>> configs = [
        ...     baseline + [("resnet18", set_r18), ("resnet34", set_r34)],
        ...     baseline + [("resnet18", set_r18), ("vgg16", set_vgg)],
        ... ]
        >>> avg_df = aggregate_over_configs("+1 different (Arch)", configs)
    """
    # Compute metrics for each config variant
    dfs = [compute_scenario("tmp", ns) for ns in list_of_named_sets]
    
    # Average pointwise across k values
    out = dfs[0][["k"]].copy()
    out["avg_jaccard"] = np.mean([d["avg_jaccard"].values for d in dfs], axis=0)
    out["avg_intersection"] = np.mean([d["avg_intersection"].values for d in dfs], axis=0)
    out["avg_union"] = np.mean([d["avg_union"].values for d in dfs], axis=0)
    out["M"] = len(list_of_named_sets[0])
    out["scenario"] = label
    
    return out[["scenario", "M", "k", "avg_jaccard", "avg_intersection", "avg_union"]]


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def _nice_bounds_numeric(ymin: float, ymax: float) -> Tuple[float, float]:
    """
    Compute readable axis bounds for numeric metrics.
    
    Adds headroom and snaps to pleasing tick intervals based on data range.
    """
    span = ymax - ymin
    pad = 0.05 * max(1.0, span)  # 5% headroom, at least ±1
    lo = ymin - pad
    hi = ymax + pad
    rng = hi - lo
    
    # Choose tick step based on range
    if rng <= 20:
        step = 1
    elif rng <= 100:
        step = 5
    elif rng <= 500:
        step = 10
    elif rng <= 2000:
        step = 50
    else:
        step = 100
    
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    
    return lo, hi


def _nice_bounds_percent(ymin: float, ymax: float) -> Tuple[float, float]:
    """
    Compute readable axis bounds for percentage metrics [0, 1].
    
    Clamps to [0,1] and snaps to 5% gridlines for clean labeling.
    """
    lo = max(0.0, ymin - 0.02)
    hi = min(1.0, ymax + 0.02)
    
    # Snap to 5% gridlines
    lo = np.floor(lo * 20) / 20.0
    hi = np.ceil(hi * 20) / 20.0
    
    if np.isclose(lo, hi):
        hi = min(1.0, lo + 0.05)
    
    return lo, hi


def compute_dynamic_ylims(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Compute data-adaptive y-axis limits for metrics.
    
    Ensures plots adapt to actual value ranges while maintaining clean,
    publication-friendly appearance.
    
    Args:
        df: DataFrame with columns: avg_jaccard, avg_intersection, avg_union
        
    Returns:
        Dictionary mapping metric names to (ymin, ymax) tuples
        
    Example:
        >>> ylims = compute_dynamic_ylims(results_df)
        >>> ax1.set_ylim(ylims["avg_jaccard"])
        >>> ax2.set_ylim(ylims["avg_intersection"])
    """
    ylims = {}
    
    # Jaccard: percentage-style [0,1]
    col = df["avg_jaccard"].dropna()
    ylims["avg_jaccard"] = (
        _nice_bounds_percent(col.min(), col.max()) if not col.empty else (0, 1)
    )
    
    # Intersection: numeric (sample counts)
    col = df["avg_intersection"].dropna()
    ylims["avg_intersection"] = (
        _nice_bounds_numeric(col.min(), col.max()) if not col.empty else (0, 1)
    )
    
    # Union: numeric (sample counts)
    col = df["avg_union"].dropna()
    ylims["avg_union"] = (
        _nice_bounds_numeric(col.min(), col.max()) if not col.empty else (0, 1)
    )
    
    return ylims


# =============================================================================
# CSV AND METRICS LOADING
# =============================================================================

def load_metrics_csv(
    experiment_dir: Union[str, Path],
    mode: str = 'leave_one_out'
) -> pd.DataFrame:
    """
    Load attack metrics CSV file.
    
    Args:
        experiment_dir: Path to experiment directory
        mode: 'leave_one_out' or 'single'
        
    Returns:
        DataFrame with attack evaluation metrics
    """
    experiment_dir = Path(experiment_dir)
    
    if mode == 'leave_one_out':
        csv_path = experiment_dir / 'attack_results_leave_one_out_summary.csv'
    else:
        csv_path = experiment_dir / 'attack_results_single.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def load_threshold_info(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load threshold information from leave-one-out evaluation.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        DataFrame with threshold info per target model
    """
    experiment_dir = Path(experiment_dir)
    threshold_path = experiment_dir / 'threshold_info_leave_one_out.csv'
    
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold info not found: {threshold_path}")
    
    return pd.read_csv(threshold_path)


def filter_thresholds(
    threshold_df: pd.DataFrame,
    attack: Optional[str] = None,
    target_fpr: Optional[float] = None,
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """
    Filter threshold dataframe by attack and/or target FPR.
    
    Args:
        threshold_df: Threshold information DataFrame
        attack: Attack name to filter (substring match, case-insensitive)
        target_fpr: Target FPR value to filter
        tolerance: Tolerance for FPR comparison
        
    Returns:
        Filtered DataFrame
        
    Example:
        >>> thresholds = load_threshold_info("experiments/cifar10")
        >>> online_001 = filter_thresholds(thresholds, "online", 0.001)
    """
    df = threshold_df.copy()
    
    if attack is not None:
        mask = df['attack'].astype(str).str.contains(attack, case=False, na=False)
        df = df[mask]
    
    if target_fpr is not None:
        df['target_fpr'] = pd.to_numeric(df['target_fpr'], errors='coerce')
        mask = np.isclose(
            df['target_fpr'].to_numpy(), target_fpr,
            rtol=tolerance, atol=tolerance
        )
        df = df[mask]
    
    return df


# =============================================================================
# LEGACY COMPATIBILITY (for backward compatibility with old notebooks)
# =============================================================================

def load_dataset_for_analysis(
    config: Dict,
    data_dir: str = 'data'
) -> Tuple[object, np.ndarray]:
    """
    Legacy function name for load_dataset. Maintained for compatibility.
    
    Use load_dataset() instead for new code.
    """
    return load_dataset(config, data_dir)