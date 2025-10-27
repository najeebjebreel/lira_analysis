"""
Main experiment runner for LiRA implementation.

This module provides functions to run experiments with both LiRA and RMIA attacks.
"""

import os
import sys
import logging
import yaml
import argparse
from pathlib import Path

from mia_research.utils.utils import setup_logger, set_seed, save_config, parse_overrides, recursive_update
from mia_research.attacks.lira import LiRA

    
def main():
    parser = argparse.ArgumentParser(description='Run membership inference attack experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument(
        '--override', 
        nargs='*', 
        default=[], 
        help='Override config values. Format: key1.subkey=value key2=value'
    )
    args = parser.parse_args()

    # Load attack configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    overrides = parse_overrides(args.override)
    recursive_update(config, overrides)
    

    # Create experiment directory
    experiment_dir = config.get('experiment', {}).get('checkpoint_dir', 'none')
    train_config_path = os.path.join(experiment_dir, 'train_config.yaml')
    # Load saved train configuration
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # merge train_config with attack config
    recursive_update(config, train_config)
    # print("Configuration after overrides:", config)
    
    # Set up logger
    log_level_str = config.get('experiment', {}).get('log_level', 'info').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('Attack', os.path.join(experiment_dir, 'attack_log.log'), level=log_level)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Save configuration
    save_config(config, os.path.join(experiment_dir, 'attack_config.yaml'))
    logger.info(f"Saved configuration to {experiment_dir}/attack_config.yaml")
    

    # Run attacks
    attack_methods = config.get('attack', {}).get('method', 'none')
    for attack_method in attack_methods:
        logger.info(f"Running {attack_method} attack...")
        if attack_method == 'lira':
            lira = LiRA(config, logger) # Initialize LiRA
            lira.generate_logits() # Generate logits for the shadow models
            lira.compute_scores() # Compute per‐sample LiRA scores (this will load keep_indices.npy automatically)
            
            # Get evaluation parameters from config
            attack_cfg = config.get('attack', {})
            ntest = attack_cfg.get('ntest', 1)
            plot_metric = attack_cfg.get('plot_metric', 'auc')
            
            # Plot ROC curves based on evaluation mode
            lira.plot(
                ntest=ntest,
                metric=plot_metric
            )
    
    logger.info("Attacks completed successfully")

if __name__ == '__main__':
    main()