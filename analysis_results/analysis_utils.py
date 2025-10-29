"""
Analysis utilities for loading and processing LiRA attack results.

This module provides reusable functions for:
- Loading experiment results (scores, labels, metrics)
- Dataset loading for analysis
- Common data processing operations
"""

import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100, GTSRB


def load_experiment_config(experiment_dir: Union[str, Path]) -> Dict:
    """
    Load experiment configuration files.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with 'train_config' and 'attack_config'
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


def load_attack_scores(experiment_dir: Union[str, Path],
                       mode: str = 'leave_one_out') -> Dict[str, np.ndarray]:
    """
    Load all attack score files from an experiment.

    Args:
        experiment_dir: Path to experiment directory
        mode: 'leave_one_out' or 'single'

    Returns:
        Dictionary mapping attack names to score arrays
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
            print(f"Warning: {filename} not found")

    return scores


def load_membership_labels(experiment_dir: Union[str, Path]) -> np.ndarray:
    """
    Load ground truth membership labels.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Boolean array of shape [M, N] where M=models, N=samples
    """
    experiment_dir = Path(experiment_dir)
    labels_path = experiment_dir / 'membership_labels.npy'

    if not labels_path.exists():
        # Fallback to keep_indices.npy
        labels_path = experiment_dir / 'keep_indices.npy'

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Neither membership_labels.npy nor keep_indices.npy found in {experiment_dir}"
        )

    labels = np.load(labels_path)
    return labels.astype(bool)


def load_metrics_csv(experiment_dir: Union[str, Path],
                     mode: str = 'leave_one_out') -> pd.DataFrame:
    """
    Load attack metrics CSV file.

    Args:
        experiment_dir: Path to experiment directory
        mode: 'leave_one_out' or 'single'

    Returns:
        Pandas DataFrame with metrics
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


def load_dataset_for_analysis(config: Dict,
                               data_dir: str = 'data') -> Tuple[object, np.ndarray]:
    """
    Load dataset for analysis (without transforms).

    Args:
        config: Configuration dictionary with dataset info
        data_dir: Data directory path

    Returns:
        Tuple of (dataset, labels_array)
    """
    dataset_name = config.get('dataset', {}).get('name', 'unknown').lower()

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
        full_dataset = None  # Tabular data, no image dataset object

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return full_dataset, full_label


def get_experiment_info(experiment_dir: Union[str, Path]) -> Dict:
    """
    Extract experiment information from directory structure and configs.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with experiment metadata
    """
    experiment_dir = Path(experiment_dir)

    # Try to parse from directory structure
    parts = experiment_dir.parts
    info = {
        'experiment_dir': str(experiment_dir),
        'dataset': parts[-3] if len(parts) >= 3 else 'unknown',
        'model': parts[-2] if len(parts) >= 2 else 'unknown',
        'config': parts[-1] if len(parts) >= 1 else 'unknown',
    }

    # Load configs for additional info
    configs = load_experiment_config(experiment_dir)
    if 'train_config' in configs:
        tc = configs['train_config']
        info['num_shadow_models'] = tc.get('training', {}).get('num_shadow_models', 'unknown')
        info['epochs'] = tc.get('training', {}).get('epochs', 'unknown')
        info['architecture'] = tc.get('model', {}).get('architecture', 'unknown')
        info['dataset_name'] = tc.get('dataset', {}).get('name', 'unknown')

    return info


def filter_thresholds(threshold_df: pd.DataFrame,
                      attack: Optional[str] = None,
                      target_fpr: Optional[float] = None,
                      tolerance: float = 1e-6) -> pd.DataFrame:
    """
    Filter threshold dataframe by attack and/or target FPR.

    Args:
        threshold_df: Threshold information DataFrame
        attack: Attack name to filter (substring match)
        target_fpr: Target FPR value to filter
        tolerance: Tolerance for FPR comparison

    Returns:
        Filtered DataFrame
    """
    df = threshold_df.copy()

    if attack is not None:
        mask = df['attack'].astype(str).str.contains(attack, case=False, na=False)
        df = df[mask]

    if target_fpr is not None:
        df['target_fpr'] = pd.to_numeric(df['target_fpr'], errors='coerce')
        mask = np.isclose(df['target_fpr'].to_numpy(), target_fpr,
                         rtol=tolerance, atol=tolerance)
        df = df[mask]

    return df


def compute_per_sample_confusion_matrix(scores: np.ndarray,
                                        labels: np.ndarray,
                                        threshold: float = 0.0) -> pd.DataFrame:
    """
    Compute confusion matrix statistics per sample across models.

    Args:
        scores: Attack scores [M, N]
        labels: Ground truth labels [M, N]
        threshold: Decision threshold

    Returns:
        DataFrame with columns: sample_id, tp, fp, tn, fn
    """
    M, N = scores.shape

    # Predictions: higher score = member
    predictions = scores >= threshold
    labels_bool = labels.astype(bool)

    # Count across models (axis=0)
    tp = np.sum(predictions & labels_bool, axis=0).astype(int)
    fp = np.sum(predictions & ~labels_bool, axis=0).astype(int)
    tn = np.sum(~predictions & ~labels_bool, axis=0).astype(int)
    fn = np.sum(~predictions & labels_bool, axis=0).astype(int)

    df = pd.DataFrame({
        'sample_id': np.arange(N),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    })

    return df


def rank_samples_by_vulnerability(confusion_df: pd.DataFrame,
                                  sort_by: str = 'low_fp_high_tp') -> pd.DataFrame:
    """
    Rank samples by vulnerability to membership inference.

    Args:
        confusion_df: DataFrame from compute_per_sample_confusion_matrix
        sort_by: Ranking strategy:
            - 'low_fp_high_tp': Prioritize low FP, then high TP (default)
            - 'high_tp_low_fp': Prioritize high TP, then low FP
            - 'vulnerability_score': TP - FP difference

    Returns:
        Sorted DataFrame with most vulnerable samples first
    """
    df = confusion_df.copy()

    if sort_by == 'low_fp_high_tp':
        # Samples rarely flagged as non-members but often detected as members
        df = df.sort_values(by=['fp', 'tp'], ascending=[True, False])
    elif sort_by == 'high_tp_low_fp':
        # Samples often detected as members, rarely as non-members
        df = df.sort_values(by=['tp', 'fp'], ascending=[False, True])
    elif sort_by == 'vulnerability_score':
        # Simple difference score
        df['vulnerability'] = df['tp'] - df['fp']
        df = df.sort_values(by='vulnerability', ascending=False)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    return df.reset_index(drop=True)


def get_highly_vulnerable_samples(confusion_df: pd.DataFrame,
                                  min_tp: int = 1,
                                  max_fp: int = 0) -> pd.DataFrame:
    """
    Get highly vulnerable samples with specific criteria.

    Args:
        confusion_df: DataFrame from compute_per_sample_confusion_matrix
        min_tp: Minimum true positive count
        max_fp: Maximum false positive count

    Returns:
        Filtered DataFrame
    """
    return confusion_df[(confusion_df['tp'] >= min_tp) & (confusion_df['fp'] <= max_fp)]
