"""
Membership inference attack implementations.

This package contains implementations of various membership inference attacks:
- LiRA: Likelihood Ratio Attack with online and offline variants
"""

from attacks.lira import LiRA

__all__ = ['LiRA']
