"""
Analysis utilities for LiRA membership inference attacks.

This package provides:
- analysis_utils: Functions for loading and processing experiment data
- metrics: ROC curve computation, confusion matrices, and evaluation metrics
- visualization: Publication-quality plotting functions
- latex_utils: LaTeX table generation for research papers

Standalone analysis scripts:
- threshold_dist.py: Analyze threshold distributions across shadow models
- compare_attacks.py: Compare multiple attack variants
- vulnerability_analysis.py: Per-sample vulnerability analysis
"""

from . import analysis_utils
from . import metrics

__all__ = [
    'analysis_utils',
    'metrics',
]
