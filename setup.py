"""
Setup script for mia-research package.

For modern installations, use: pip install -e .
This will use pyproject.toml configuration.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="mia-research",
    version="0.1.0",
    author="Research Team",
    description="Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/najeebjebreel/lira_analysis",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "timm>=0.9.0",
        "Pillow>=8.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mia-train=mia_research.scripts.train:main",
            "mia-attack=mia_research.scripts.attack:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
