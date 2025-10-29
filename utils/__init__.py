"""
Utility modules for LiRA membership inference attack implementation.

This package contains:
- utils: General utilities for logging, seeding, and configuration
- data_utils: Dataset loading and preprocessing utilities
- model_utils: Model architecture definitions and factory functions
- train_utils: Training functions, optimizers, and augmentation utilities
"""

from utils.utils import (
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
    stable_softmax,
    stable_logsumexp,
    compute_cross_entropy_loss
)

from utils.data_utils import (
    TabularDataset,
    TransformSubset,
    load_dataset,
    load_dataset_for_mia_inference,
    create_data_loaders,
    get_keep_indices,
    build_transforms,
    build_attack_transform,
    DATASET_STATS
)

from utils.model_utils import (
    FCN,
    WideResNet,
    get_model
)

from utils.train_utils import (
    train_model,
    evaluate_model,
    mixup_data,
    cutmix_data,
    get_optimizer,
    get_scheduler,
    log_msg
)

__all__ = [
    # utils
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
    'stable_softmax',
    'stable_logsumexp',
    'compute_cross_entropy_loss',
    # data_utils
    'TabularDataset',
    'TransformSubset',
    'load_dataset',
    'load_dataset_for_mia_inference',
    'create_data_loaders',
    'get_keep_indices',
    'build_transforms',
    'build_attack_transform',
    'DATASET_STATS',
    # model_utils
    'FCN',
    'WideResNet',
    'get_model',
    # train_utils
    'train_model',
    'evaluate_model',
    'mixup_data',
    'cutmix_data',
    'get_optimizer',
    'get_scheduler',
    'log_msg',
]
