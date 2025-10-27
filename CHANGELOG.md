# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-27

### Added
- Initial project structure with proper Python packaging
- LiRA (Likelihood Ratio Attack) implementation
- Support for multiple datasets (CIFAR-10, CIFAR-100, GTSRB, CINIC-10, Purchase, Texas, Location)
- Support for multiple model architectures via timm
- Custom WideResNet and FCN implementations
- Comprehensive data loading and augmentation utilities
- Training utilities with mixed precision support
- Configuration system via YAML files
- Command-line entry points: `mia-train` and `mia-attack`
- Jupyter notebooks for analysis and visualization
- Unit tests for core functionality
- CI/CD pipeline with GitHub Actions
- Comprehensive documentation

### Project Structure
- Organized code into modular package structure under `src/mia_research/`
- Separated concerns: attacks, models, data, training, utils
- Entry scripts in `src/mia_research/scripts/`
- All notebooks consolidated in `notebooks/`
- Configuration files in `configs/`

### Features
- Leave-one-out and single-target evaluation modes
- Multiple attack variants (online, offline, fixed variance)
- Spatial augmentation support for inference
- Comprehensive ROC curve plotting
- TPR@FPR metrics calculation
- Train/test statistics computation

[0.1.0]: https://github.com/najeebjebreel/lira_analysis/releases/tag/v0.1.0
