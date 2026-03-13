"""
Main experiment runner for LiRA implementation.

This module provides functions to run experiments with both LiRA and RMIA attacks.
"""

from __future__ import annotations

import argparse
import logging

from attacks.lira import LiRA
from utils.common import save_config, set_seed, setup_logger
from utils.config_utils import add_override_argument, load_attack_runtime_config
from utils.path_utils import ExperimentPaths


REQUIRED_KEYS = ["experiment.checkpoint_dir", "attack.method"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the attack stage."""
    parser = argparse.ArgumentParser(description="Run membership inference attack experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    add_override_argument(parser)
    return parser


def get_attack_methods(config: dict) -> list[str]:
    """Normalize configured attack methods into a list."""
    methods = config.get("attack", {}).get("method", [])
    if isinstance(methods, str):
        return [methods]
    return list(methods)


def main(argv=None):
    """Run attack evaluation from the command line."""
    args = build_parser().parse_args(argv)

    try:
        config = load_attack_runtime_config(args.config, args.override, REQUIRED_KEYS)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration validation failed: {exc}")
        raise SystemExit(1) from exc

    experiment_paths = ExperimentPaths(config["experiment"]["checkpoint_dir"])

    log_level_str = config.get("experiment", {}).get("log_level", "info").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger("Attack", str(experiment_paths.experiment_dir / "attack_log.log"), level=log_level)

    seed = config.get("seed", 42)
    deterministic = config.get("experiment", {}).get("deterministic", True)
    set_seed(seed, deterministic=deterministic)
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")

    attack_config_path = experiment_paths.experiment_dir / "attack_config.yaml"
    save_config(config, str(attack_config_path))
    logger.info(f"Saved configuration to {attack_config_path}")

    compute_logits = config.get("attack", {}).get("compute_logits", True)
    compute_scores = config.get("attack", {}).get("compute_scores", True)
    perform_attack = config.get("attack", {}).get("perform_attack", True)

    for attack_method in get_attack_methods(config):
        logger.info(f"Running {attack_method} attack...")
        if attack_method != "lira":
            logger.warning("Skipping unsupported attack method: %s", attack_method)
            continue

        lira = LiRA(config, logger)
        if compute_logits:
            logger.info("Generating logits for LiRA...")
            lira.generate_logits()
        if compute_scores:
            logger.info("Computing LiRA scores...")
            lira.compute_scores()
        if perform_attack:
            attack_cfg = config.get("attack", {})
            ntest = attack_cfg.get("ntest", 1)
            plot_metric = attack_cfg.get("plot_metric", "auc")
            lira.plot(ntest=ntest, metric=plot_metric)

    logger.info("Attacks completed successfully")


if __name__ == "__main__":
    main()
