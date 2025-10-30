"""
Main training script for LiRA membership inference experiments.

This module trains multiple shadow models on different subsets of data,
which are later used to perform membership inference attacks.
"""

import os
import sys
import numpy as np
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import torch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.utils import (
    setup_logger, set_seed, save_config, parse_overrides,
    recursive_update, cleanup_gpu_memory, validate_config
)
from utils.data_utils import load_dataset, TransformSubset, create_data_loaders
from utils.model_utils import get_model
from utils.train_utils import train_model

def setup_experiment(config):
    """
    Set up the experiment environment.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (output_dir, logger, device)
    """
    # Create experiment directory
    dataset = config.get('dataset', {}).get('name', 'cifar10')
    model= config.get('model', {}).get('architecture', 'resnet18')
    if config.get('experiment', {}).get('checkpoint_dir', 'none') == 'none':
        experiment_dir = os.path.join('experiments', dataset, model, datetime.now().strftime('%Y-%m-%d_%H%M'))
        os.makedirs(experiment_dir, exist_ok=True)
    else:
        experiment_dir = config.get('experiment', {}).get('checkpoint_dir', 'none')
    
    # Set up logger
    log_level_str = config.get('experiment', {}).get('log_level', 'info').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('Train', os.path.join(experiment_dir, 'train_log.log'), level=log_level)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Get device
    use_cuda = config.get('use_cuda', True)
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Save configuration
    save_config(config, os.path.join(experiment_dir, 'train_config.yaml'))
    logger.info(f"Saved configuration to {experiment_dir}/train_config.yaml")
    
    return experiment_dir, logger, device

def train_target_model(config, train_dataset, test_dataset, train_dataset_eval, device, shadow_model_dir, writer, logger):
    """
    Train the target model with updated data loader interface.
    
    Args:
        config (dict): Configuration dictionary
        train_dataset: Training dataset
        test_dataset: Test dataset
        train_dataset_eval: Training dataset for evaluation
        device: Device to train on
        shadow_model_dir: Directory to save results
        writer: TensorBoard writer
        logger: Logger
        
    Returns:
        model: Trained target model
    """
    
    # Create data loaders with augmentation info
    train_loader, test_loader, train_eval_loader = create_data_loaders(
        train_dataset, test_dataset, train_dataset_eval, config
    )
    
    # Create model
    num_classes = config.get('dataset', {}).get('num_classes', 10)
    
    model = get_model(num_classes, **config.get('model', {}))
    model = model.to(device)
    
    # Train model with augmentation info
    model = train_model(
        model,
        train_loader,
        test_loader,
        train_eval_loader,
        config,
        device,
        save_dir=shadow_model_dir,
        writer=writer,
        logger=logger
    )
    
    del train_loader
    del test_loader
    del train_eval_loader
    cleanup_gpu_memory()
    
    return model
    

def main():
    """
    Main function to run experiments from command line.
    """
    parser = argparse.ArgumentParser(description='Run membership inference attack experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument(
        '--override',
        nargs='*',
        default=[],
        help='Override config values. Format: key1.subkey=value key2=value'
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    overrides = parse_overrides(args.override)
    recursive_update(config, overrides)

    # Validate configuration
    required_keys = [
        'dataset.name',
        'training.num_shadow_models',
        'training.epochs',
        'training.batch_size',
        'model.architecture'
    ]
    try:
        validate_config(config, required_keys)
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Set up experiment
    experiment_dir, logger, device = setup_experiment(config)
    config.get('experiment', {}).update({'checkpoint_dir': experiment_dir})
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset, keep_indices, train_transform, test_transform = load_dataset(config)
    keep_indices_path = os.path.join(experiment_dir, 'keep_indices.npy')
    if os.path.exists(keep_indices_path):
        logger.info("keep_indices.npy already exists, loading from disk...")
        keep_indices = np.load(keep_indices_path)
    else:
        logger.info("Saving keep_indices.npy to disk...")
        np.save(keep_indices_path, keep_indices)

    logger.info("Dataset loaded...")
    logger.info("full_dataset size: %d", len(full_dataset))
    logger.info("keep_indices shape: %s", keep_indices.shape)   
    logger.info("------------------------------------------")

    #writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))
    writer = None  # Initialize TensorBoard writer if needed
    # Train shadow models
    for i in range(config.get('training', {}).get('num_shadow_models', 1)):
        logger.info(f"Training shadow model {i}...")
        shadow_model_dir = os.path.join(experiment_dir, f'model_{i}')
        os.makedirs(shadow_model_dir, exist_ok=True)
        shadow_train_dataset = TransformSubset(full_dataset, keep_indices[i], train_transform)
        shadow_test_dataset = TransformSubset(full_dataset, ~keep_indices[i], test_transform)
        shadow_train_dataset_eval = TransformSubset(full_dataset, keep_indices[i], test_transform)
        logger.info(f"Shadow train dataset size: {len(shadow_train_dataset)}")
        logger.info(f"Shadow test dataset size: {len(shadow_test_dataset)}")
        shadow_model = train_target_model(config, shadow_train_dataset, shadow_test_dataset, shadow_train_dataset_eval, device, shadow_model_dir, writer, logger)
        logger.info(f"Shadow model {i} trained and saved to {shadow_model_dir}")
        # Add explicit cleanup
        del shadow_model
        del shadow_train_dataset
        del shadow_test_dataset
        del shadow_train_dataset_eval
        cleanup_gpu_memory()
    
    logger.info("All shadow models training completed successfully")

if __name__ == '__main__':
    main()
