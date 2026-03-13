"""
Utility modules for LiRA membership inference attack implementation.

This package contains:
- utils: General utilities for logging, seeding, and configuration
- data_utils: Dataset loading and preprocessing utilities
- model_utils: Model architecture definitions and factory functions
- train_utils: Training functions, optimizers, and augmentation utilities
"""

from utils.common import (
    setup_logger,
    set_seed,
    save_config,
    load_config,
    save_results,
    count_parameters,
    save_model,
    load_model,
    parse_overrides,
    recursive_update,
    cleanup_gpu_memory,
    load_checkpoint_safe,
    validate_config,
    evaluate,
    aggregate_shadow_metrics
)
from utils.config_utils import (
    add_override_argument,
    apply_overrides,
    load_attack_runtime_config,
    load_config_with_overrides,
)

from utils.path_utils import (
    ExperimentPaths,
    build_experiment_dir,
)

from utils.data_utils import (
    TabularDataset,
    TransformSubset,
    load_dataset,
    create_data_loaders,
    get_keep_indices,
    build_transforms,
    DATASET_STATS
)

from utils.model_utils import (
    FCN,
    WideResNet,
    get_model
)

from utils.train_utils import (
    train_model,
    mixup_data,
    cutmix_data,
    get_optimizer,
    get_scheduler,
    log_msg
)

__all__ = [
    'setup_logger',
    'set_seed',
    'save_config',
    'load_config',
    'save_results',
    'count_parameters',
    'save_model',
    'load_model',
    'parse_overrides',
    'recursive_update',
    'cleanup_gpu_memory',
    'load_checkpoint_safe',
    'validate_config',
    'aggregate_shadow_metrics',
    'add_override_argument',
    'apply_overrides',
    'load_attack_runtime_config',
    'load_config_with_overrides',
    'evaluate',
    'ExperimentPaths',
    'build_experiment_dir',
    'TabularDataset',
    'TransformSubset',
    'load_dataset',
    'create_data_loaders',
    'get_keep_indices',
    'build_transforms',
    'DATASET_STATS',
    'FCN',
    'WideResNet',
    'get_model',
    'train_model',
    'mixup_data',
    'cutmix_data',
    'get_optimizer',
    'get_scheduler',
    'log_msg',
]
