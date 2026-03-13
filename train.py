"""
Main training script for LiRA membership inference experiments.

This module trains multiple shadow models on different subsets of data,
which are later used to perform membership inference attacks.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Subset

from utils.common import (
    aggregate_shadow_metrics,
    cleanup_gpu_memory,
    save_config,
    set_seed,
    setup_logger,
)
from utils.config_utils import add_override_argument, load_config_with_overrides
from utils.data_utils import create_data_loaders, load_dataset
from utils.model_utils import get_model
from utils.path_utils import ExperimentPaths, build_experiment_dir
from utils.train_utils import train_model


REQUIRED_KEYS = [
    "dataset.name",
    "training.num_shadow_models",
    "training.epochs",
    "training.batch_size",
    "model.architecture",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for training."""
    parser = argparse.ArgumentParser(description="Run membership inference attack experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    add_override_argument(parser)
    return parser


def setup_experiment(config):
    """
    Set up the experiment environment.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (experiment_paths, logger, device)
    """
    experiment_paths = ExperimentPaths(build_experiment_dir(config, created_at=datetime.now()))
    experiment_paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("experiment", {})["checkpoint_dir"] = str(experiment_paths.experiment_dir)

    log_level_str = config.get("experiment", {}).get("log_level", "info").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger("Train", str(experiment_paths.experiment_dir / "train_log.log"), level=log_level)

    seed = config.get("seed", 42)
    deterministic = config.get("experiment", {}).get("deterministic", True)
    set_seed(seed, deterministic=deterministic)
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")

    use_cuda = config.get("use_cuda", True)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_config_path = experiment_paths.experiment_dir / "train_config.yaml"
    save_config(config, str(train_config_path))
    logger.info(f"Saved configuration to {train_config_path}")

    return experiment_paths, logger, device


def train_target_model(config, train_dataset, test_dataset, train_dataset_eval, device, shadow_model_dir, logger):
    """
    Train the target model with updated data loader interface.

    Args:
        config (dict): Configuration dictionary
        train_dataset: Training dataset
        test_dataset: Test dataset
        train_dataset_eval: Training dataset for evaluation
        device: Device to train on
        shadow_model_dir: Directory to save results
        logger: Logger

    Returns:
        model: Trained target model
    """
    train_loader, test_loader, train_eval_loader = create_data_loaders(
        train_dataset, test_dataset, train_dataset_eval, config
    )

    num_classes = config.get("dataset", {}).get("num_classes", 10)
    model = get_model(num_classes, **config.get("model", {}))
    model = model.to(device)

    model = train_model(
        model,
        train_loader,
        test_loader,
        train_eval_loader,
        config,
        device,
        save_dir=shadow_model_dir,
        logger=logger,
    )

    del train_loader
    del test_loader
    del train_eval_loader
    cleanup_gpu_memory()

    return model


def main(argv=None):
    """Run experiments from the command line."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config_with_overrides(args.config, args.override, REQUIRED_KEYS)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration validation failed: {exc}")
        raise SystemExit(1) from exc

    experiment_paths, logger, device = setup_experiment(config)

    logger.info("Loading dataset...")
    train_dataset, keep_indices, test_dataset, train_eval_dataset = load_dataset(config)
    keep_indices_path = experiment_paths.keep_indices_path
    if keep_indices_path.exists():
        logger.info("keep_indices.npy already exists, loading from disk...")
        keep_indices = np.load(keep_indices_path)
        expected_shape = (
            config.get("training", {}).get("num_shadow_models", 1),
            len(train_dataset),
        )
        if keep_indices.shape != expected_shape:
            raise ValueError(
                f"keep_indices shape mismatch: found {keep_indices.shape}, expected {expected_shape}"
            )
    else:
        logger.info("Saving keep_indices.npy to disk...")
        keep_indices_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(keep_indices_path, keep_indices)

    logger.info("Dataset loaded...")
    n = len(train_dataset)
    logger.info("train_dataset size: %d", n)
    logger.info("keep_indices shape: %s", keep_indices.shape)
    logger.info("test_dataset size: %d", len(test_dataset))
    logger.info("------------------------------------------")

    start_shadow_model_idx = config.get("training", {}).get("start_shadow_model_idx", 0)
    end_shadow_model_idx = config.get("training", {}).get(
        "end_shadow_model_idx",
        config.get("training", {}).get("num_shadow_models", 1) - 1,
    )
    logger.info(
        "Training shadow models from index %d to %d...",
        start_shadow_model_idx,
        end_shadow_model_idx,
    )

    for model_idx in range(start_shadow_model_idx, end_shadow_model_idx + 1):
        logger.info(f"Training shadow model {model_idx}...")
        shadow_model_dir = experiment_paths.get_model_dir(model_idx)
        shadow_model_dir.mkdir(parents=True, exist_ok=True)
        train_idxs = np.arange(n)
        shadow_train_dataset = Subset(train_dataset, train_idxs[keep_indices[model_idx]])
        shadow_train_dataset_eval = Subset(train_eval_dataset, train_idxs[keep_indices[model_idx]])
        logger.info(f"Shadow train dataset size: {len(shadow_train_dataset)}")
        shadow_model = train_target_model(
            config,
            shadow_train_dataset,
            test_dataset,
            shadow_train_dataset_eval,
            device,
            shadow_model_dir,
            logger,
        )
        logger.info(f"Shadow model {model_idx} trained and saved to {shadow_model_dir}\n")

        del shadow_model
        del shadow_train_dataset
        del shadow_train_dataset_eval
        cleanup_gpu_memory()

    logger.info("All shadow models training completed successfully")
    aggregate_shadow_metrics(
        experiment_paths.experiment_dir,
        end_shadow_model_idx + 1,
        start_idx=start_shadow_model_idx,
        logger=logger,
    )


if __name__ == "__main__":
    main()
