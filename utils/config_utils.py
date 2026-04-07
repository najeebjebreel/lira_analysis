"""Shared configuration helpers for CLI entrypoints."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

from utils.common import load_config, parse_overrides, recursive_update, validate_config


OVERRIDE_HELP = "Override config values. Format: key1.subkey=value key2=value"


def add_override_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common override argument used by train/attack entrypoints."""
    parser.add_argument("--override", nargs="*", default=[], help=OVERRIDE_HELP)


def apply_overrides(config: dict, override_args: list[str] | None) -> dict:
    """Apply dotted CLI overrides to a nested config dictionary."""
    overrides = parse_overrides(override_args or [])
    recursive_update(config, overrides)
    return config


def load_config_with_overrides(
    config_path: str | Path,
    override_args: list[str] | None = None,
    required_keys: Iterable[str] | None = None,
) -> dict:
    """Load a YAML config, apply CLI overrides, and validate required keys."""
    config = load_config(config_path)
    apply_overrides(config, override_args)
    if required_keys is not None:
        validate_config(config, list(required_keys))
    return config


def load_attack_runtime_config(
    config_path: str | Path,
    override_args: list[str] | None = None,
    required_keys: Iterable[str] | None = None,
) -> dict:
    """Merge saved train config, attack config, and CLI overrides for attack execution."""
    attack_config = load_config(config_path)
    apply_overrides(attack_config, override_args)
    validate_config(attack_config, ["experiment.checkpoint_dir"])

    checkpoint_dir = attack_config["experiment"]["checkpoint_dir"]
    if checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir is not set. When invoking attack.py directly, you must pass "
            "--override experiment.checkpoint_dir=<path/to/experiment>. "
            "The recommended way to run the pipeline is via scripts/run_benchmark.py, "
            "which sets this automatically."
        )
    experiment_dir = Path(checkpoint_dir)
    train_config_path = experiment_dir / "train_config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Missing training config: {train_config_path}")

    train_config = load_config(train_config_path)
    config = copy.deepcopy(train_config)
    recursive_update(config, attack_config)

    if required_keys is not None:
        validate_config(config, list(required_keys))
    return config
