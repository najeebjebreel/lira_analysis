"""
Logging and utility functions for the unified LiRA and RMIA implementation.
"""

import os
import json
import yaml
import logging
import numpy as np
import torch
import random

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
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_seed(seed):
    """
    Set random seed with a balanced approach for research experiments.
    
    Args:
        seed (int): Random seed
    """
    # Set seeds for standard libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Balanced reproducibility: allows non-determinism but better performance
        torch.backends.cudnn.deterministic = False  # ⚠️ Less reproducible
        torch.backends.cudnn.benchmark = True       # ✅ Enables performance tuning
        torch.backends.cudnn.enabled = True         # ✅ Generally safe to keep True

def save_config(config, save_path):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config




def save_results(results, save_path):
    """
    Save results to a JSON file.
    
    Args:
        results: Results dictionary
        save_path: Path to save the results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_json = convert_numpy(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=4)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Add any additional information
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
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint

def parse_overrides(override_list):
    overrides = {}
    for item in override_list:
        if '=' not in item:
            raise ValueError(f"Invalid override format: {item}. Use key.subkey=value format.")
        key, value = item.split('=', 1)

        # Try to parse value as int, float, or bool
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string

        keys = key.split('.')
        d = overrides
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return overrides

def recursive_update(d, u):
    """
    Recursively update nested dictionary d with values from u.

    Args:
        d: Dictionary to update
        u: Dictionary with updates
    """
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            recursive_update(d[k], v)
        elif v in [None, 'none'] and d.get(k) not in [None, 'none']:
            continue  # Skip overwrite to preserve meaningful value
        else:
            d[k] = v


def cleanup_gpu_memory():
    """
    Clean up GPU memory by clearing CUDA cache and running garbage collection.
    This helps prevent out-of-memory errors in long-running experiments.
    """
    import gc
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
        tuple: (model, checkpoint_dict) where checkpoint_dict contains
               additional info like epoch, optimizer state, etc.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            checkpoint = {'state_dict': checkpoint}

        if logger:
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")

        return model, checkpoint

    except Exception as e:
        if logger:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise


def validate_config(config, required_keys=None, logger=None):
    """
    Validate configuration dictionary has required keys and reasonable values.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key paths (e.g., ['dataset.name', 'training.epochs'])
        logger: Optional logger for validation messages

    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    if required_keys is None:
        required_keys = ['dataset.name']

    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                msg = f"Missing required config key: {key_path}"
                if logger:
                    logger.error(msg)
                raise ValueError(msg)
            current = current[key]

    if logger:
        logger.info("Configuration validation passed")

    return True








