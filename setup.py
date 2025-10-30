from setuptools import setup, find_packages

setup(
    name="lira_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "seaborn",
    ],
)
