"""
Path management utilities for experiment organization.

This module provides a centralized way to manage paths for experiments,
models, logits, scores, and results, ensuring consistency across the codebase.
"""

import os
from pathlib import Path


class ExperimentPaths:
    """
    Centralized path management for LiRA experiments.

    This class provides a consistent interface for accessing all paths
    related to an experiment, including model checkpoints, logits, scores,
    and result files.

    Attributes:
        experiment_dir: Root directory for the experiment
    """

    def __init__(self, experiment_dir):
        """
        Initialize experiment paths.

        Args:
            experiment_dir: Root directory for the experiment
        """
        self.experiment_dir = Path(experiment_dir)

    # Root level paths

    @property
    def keep_indices_path(self):
        """Path to keep_indices.npy file."""
        return self.experiment_dir / 'keep_indices.npy'

    @property
    def train_test_stats_path(self):
        """Path to train_test_stats.csv file."""
        return self.experiment_dir / 'train_test_stats.csv'

    @property
    def roc_curve_single_path(self):
        """Path to single-target ROC curve PDF."""
        return self.experiment_dir / 'roc_curve_single.pdf'

    @property
    def attack_results_single_path(self):
        """Path to single-target attack results CSV."""
        return self.experiment_dir / 'attack_results_single.csv'

    @property
    def attack_results_loo_summary_path(self):
        """Path to leave-one-out attack results summary CSV."""
        return self.experiment_dir / 'attack_results_leave_one_out_summary.csv'

    @property
    def membership_labels_path(self):
        """Path to membership labels numpy file."""
        return self.experiment_dir / 'membership_labels.npy'

    @property
    def threshold_info_loo_path(self):
        """Path to leave-one-out threshold info CSV."""
        return self.experiment_dir / 'threshold_info_leave_one_out.csv'

    # Model-specific paths

    def get_model_dir(self, model_idx):
        """
        Get the directory for a specific model.

        Args:
            model_idx: Index of the shadow model

        Returns:
            Path to the model directory
        """
        return self.experiment_dir / f'model_{model_idx}'

    def get_checkpoint_path(self, model_idx, checkpoint_name='best.pth'):
        """
        Get the checkpoint path for a specific model.

        Args:
            model_idx: Index of the shadow model
            checkpoint_name: Name of the checkpoint file (default: 'best.pth')

        Returns:
            Path to the checkpoint file
        """
        return self.get_model_dir(model_idx) / checkpoint_name

    def get_logits_dir(self, model_idx):
        """
        Get the logits directory for a specific model.

        Args:
            model_idx: Index of the shadow model

        Returns:
            Path to the logits directory
        """
        return self.get_model_dir(model_idx) / 'logits'

    def get_logits_path(self, model_idx, filename='logits.npy'):
        """
        Get the logits file path for a specific model.

        Args:
            model_idx: Index of the shadow model
            filename: Name of the logits file (default: 'logits.npy')

        Returns:
            Path to the logits file
        """
        return self.get_logits_dir(model_idx) / filename

    def get_scores_dir(self, model_idx):
        """
        Get the scores directory for a specific model.

        Args:
            model_idx: Index of the shadow model

        Returns:
            Path to the scores directory
        """
        return self.get_model_dir(model_idx) / 'scores'

    def get_scores_path(self, model_idx, filename='scores.npy'):
        """
        Get the scores file path for a specific model.

        Args:
            model_idx: Index of the shadow model
            filename: Name of the scores file (default: 'scores.npy')

        Returns:
            Path to the scores file
        """
        return self.get_scores_dir(model_idx) / filename

    # Leave-one-out specific paths

    def get_loo_scores_path(self, score_filename):
        """
        Get the leave-one-out scores path.

        Args:
            score_filename: Base name for the scores file (without extension)

        Returns:
            Path to the leave-one-out scores file
        """
        return self.experiment_dir / f'{score_filename}_leave_one_out.npy'

    # Utility methods

    def ensure_dir(self, path):
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Path to the directory

        Returns:
            The path as a Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_model_dirs(self, model_idx):
        """
        Ensure all directories for a model exist.

        Args:
            model_idx: Index of the shadow model
        """
        self.ensure_dir(self.get_model_dir(model_idx))
        self.ensure_dir(self.get_logits_dir(model_idx))
        self.ensure_dir(self.get_scores_dir(model_idx))

    def __str__(self):
        """String representation of the experiment paths."""
        return f"ExperimentPaths(experiment_dir={self.experiment_dir})"
