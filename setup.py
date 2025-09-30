#!/usr/bin/env python3
"""
Setup script for TIER 1 Cellular Rejuvenation Suite
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="tier1-rejuvenation-suite",
    version="1.0.0",
    description="Biologically validated cellular rejuvenation analysis suite with comprehensive biomarker validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kemal Yaylali",
    author_email="kemal.yaylali@gmail.com",
    url="https://github.com/lynchaos/tier1-rejuvenation-suite",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.csv', '*.tsv', '*.h5', '*.h5ad'],
    },
    install_requires=[
        # Core Data Science Stack
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        
        # Machine Learning & AI
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "shap>=0.41.0",
        "imbalanced-learn>=0.9.0",
        
        # Deep Learning (PyTorch)
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        
        # Bioinformatics & Genomics
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "umap-learn>=0.5.3",
        "leidenalg>=0.8.10",
        "python-igraph>=0.10.0",
        "networkx>=2.8.0",
        "biopython>=1.79",
        
        # Statistical Analysis
        "statsmodels>=0.13.0",
        "pingouin>=0.5.0",
        
        # Data Processing & I/O
        "h5py>=3.7.0",
        "openpyxl>=3.0.10",
        "pyyaml>=6.0",
        "jsonschema>=4.9.0",
        
        # Parallel Processing
        "joblib>=1.1.0",
        "dask>=2022.7.0",
        
        # CLI Framework
        "typer>=0.7.0",
        "rich>=12.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
        ],
        'web': [
            'streamlit>=1.12.0',
            'dash>=2.6.0',
            'bokeh>=2.4.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'ipywidgets>=8.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'tier1=tier1_suite.cli:main',
            'tier1-interactive=tier1_suite.interactive:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    keywords="bioinformatics cellular-rejuvenation aging-research biomarkers genomics",
    project_urls={
        "Homepage": "https://github.com/lynchaos/tier1-rejuvenation-suite",
        "Repository": "https://github.com/lynchaos/tier1-rejuvenation-suite",
        "Documentation": "https://github.com/lynchaos/tier1-rejuvenation-suite#readme",
        "Bug Tracker": "https://github.com/lynchaos/tier1-rejuvenation-suite/issues",
    },
)