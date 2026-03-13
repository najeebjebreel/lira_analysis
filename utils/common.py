"""
Logging and utility functions for the LiRA implementation.
"""

import csv
import gc
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import autocast


def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def set_seed(seed, deterministic=True):
    """
    Set random seed across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed
        deterministic (bool): Whether to prefer deterministic execution
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def save_config(config, save_path):
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_results(results, save_path):
    """
    Save results to a JSON file.

    Args:
        results: Results dictionary
        save_path: Path to save the results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(convert_numpy(results), handle, indent=4)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def save_model(model, save_path, optimizer=None, epoch=None, **kwargs):
    """
    Save a model checkpoint.

    Args:
        model: PyTorch model
        save_path: Path to save the model
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        **kwargs: Additional information to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {"model_state_dict": model.state_dict()}

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)


def load_model(model, load_path, optimizer=None, device=None):
    """
    Load a model checkpoint.

    Args:
        model: PyTorch model
        load_path: Path to the checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load the model to

    Returns:
        tuple: (model, optimizer, checkpoint)
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint


def parse_overrides(override_list):
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Use key.subkey=value format.")
        key, value = item.split("=", 1)

        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

        keys = key.split(".")
        current = overrides
        for part in keys[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[keys[-1]] = value
    return overrides


def recursive_update(d, u):
    """
    Recursively update nested dictionary d with values from u.

    Args:
        d: Dictionary to update
        u: Dictionary with updates
    """
    for key, value in u.items():
        if isinstance(value, dict) and isinstance(d.get(key), dict):
            recursive_update(d[key], value)
        elif value in [None, "none"] and d.get(key) not in [None, "none"]:
            continue
        else:
            d[key] = value


def cleanup_gpu_memory():
    """
    Clean up GPU memory by clearing CUDA cache and running garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_checkpoint_safe(checkpoint_path, model, device=None, logger=None):
    """
    Safely load a model checkpoint with fallback for different formats.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: PyTorch model to load weights into
        device: Device to map the checkpoint to
        logger: Optional logger for messages

    Returns:
        tuple: (model, checkpoint_dict)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            checkpoint = {"state_dict": checkpoint}

        if logger:
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return model, checkpoint
    except Exception as exc:
        if logger:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {exc}")
        raise


def validate_config(config, required_keys=None, logger=None):
    """
    Validate configuration dictionary has required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key paths (e.g., ['dataset.name'])
        logger: Optional logger for validation messages

    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    if required_keys is None:
        required_keys = ["dataset.name"]

    for key_path in required_keys:
        current = config
        for key in key_path.split("."):
            if not isinstance(current, dict) or key not in current:
                message = f"Missing required config key: {key_path}"
                if logger:
                    logger.error(message)
                raise ValueError(message)
            current = current[key]

    if logger:
        logger.info("Configuration validation passed")
    return True


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, use_amp=True, logger=None, epoch=None, num_epochs=None, prefix="[Eval ]"):
    """
    Evaluate the model on a dataset.
    """
    del logger, epoch, num_epochs, prefix

    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    use_cuda_amp = use_amp and device.type == "cuda"

    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_cuda_amp:
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = loss_sum / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def aggregate_shadow_metrics(experiment_dir, num_models, start_idx=0, logger=None):
    """
    Aggregate final metrics from model_{i}/metrics.csv and save mean/std to
    <experiment_dir>/shadow_metrics_aggregate.csv.
    """
    expdir = Path(experiment_dir)
    per_model = []

    def to_float(value):
        if value is None:
            return float("nan")
        value = str(value).strip().replace("%", "").replace("(", "").replace(")", "")
        try:
            return float(value)
        except ValueError:
            return float("nan")

    for idx in range(start_idx, num_models):
        metrics_path = expdir / f"model_{idx}" / "metrics.csv"
        if not metrics_path.exists():
            if logger:
                logger.warning(f"[Aggregate] Missing {metrics_path}")
            continue

        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            if logger:
                logger.warning(f"[Aggregate] Empty {metrics_path}")
            continue

        row = rows[-1]
        train_loss = to_float(row.get("Train_loss"))
        test_loss = to_float(row.get("Test_loss"))
        train_acc = to_float(row.get("Train_acc (%)"))
        test_acc = to_float(row.get("Test_acc (%)"))

        loss_ratio = (
            test_loss / train_loss
            if (not math.isnan(train_loss) and train_loss != 0 and not math.isnan(test_loss))
            else float("nan")
        )
        acc_gap = train_acc - test_acc if (not math.isnan(train_acc) and not math.isnan(test_acc)) else float("nan")

        per_model.append(
            {
                "Train_loss": train_loss,
                "Test_loss": test_loss,
                "Loss_ratio": loss_ratio,
                "Train_acc (%)": train_acc,
                "Test_acc (%)": test_acc,
                "Acc_gap (%)": acc_gap,
            }
        )

    if not per_model:
        if logger:
            logger.error("[Aggregate] No metrics found.")
        return

    def mean(values):
        valid = [value for value in values if not math.isnan(value)]
        return sum(valid) / len(valid) if valid else float("nan")

    def std(values):
        valid = [value for value in values if not math.isnan(value)]
        if len(valid) < 2:
            return float("nan")
        avg = sum(valid) / len(valid)
        return math.sqrt(sum((value - avg) ** 2 for value in valid) / (len(valid) - 1))

    metrics = ["Train_loss", "Test_loss", "Loss_ratio", "Train_acc (%)", "Test_acc (%)", "Acc_gap (%)"]
    aggregated = {metric: [row[metric] for row in per_model] for metric in metrics}

    out_path = expdir / "shadow_metrics_aggregate.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Mean", "Std", "Count"])
        for metric in metrics:
            values = aggregated[metric]
            count = len([value for value in values if not math.isnan(value)])
            writer.writerow([metric, f"{mean(values):.4f}", f"{std(values):.4f}", count])

    if logger:
        logger.info(f"[Aggregate] Saved aggregate metrics to {out_path}")
        for metric in metrics:
            logger.info(f"{metric:>15s} | mean={mean(aggregated[metric]):.4f}  std={std(aggregated[metric]):.4f}")
