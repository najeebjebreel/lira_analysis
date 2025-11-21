from setuptools import setup, find_packages

setup(
    name="lira-analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "timm>=0.9.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "seaborn>=0.12.0",
            "ipywidgets>=8.0.0",
        ],
    },
)
