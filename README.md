# 🧬 TIER 1 Cellular Rejuvenation Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Biologically validated cellular rejuvenation analysis suite with comprehensive biomarker validation

## 🔬 Overview

The TIER 1 Cellular Rejuvenation Suite is a comprehensive bioinformatics toolkit designed for cellular aging and rejuvenation research. It provides command-line interfaces and interactive tools for:

- **Bulk omics data analysis** with machine learning models
- **Single-cell RNA-seq analysis** including QC, clustering, and trajectory inference
- **Multi-omics integration** and biomarker discovery
- **Biologically validated scoring** using 110+ peer-reviewed aging biomarkers

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/lynchaos/tier1-rejuvenation-suite.git
cd tier1-rejuvenation-suite
pip install -e .

# Or install from PyPI (when available)
pip install tier1-rejuvenation-suite
```

### Basic Usage

```bash
# Show available commands
tier1 --help

# Show suite information
tier1 info

# Launch interactive interface
tier1 interactive
```

## 📋 CLI Commands

### Bulk Data Analysis

Train machine learning models on bulk omics data:

```bash
# Fit models with biomarker validation
tier1 bulk fit data.csv models/ --biomarker-val

# Make predictions
tier1 bulk predict new_data.csv models/ predictions.csv

# Validate predictions
tier1 bulk validate predictions.csv --report
```

### Single-Cell Analysis

Complete single-cell RNA-seq pipeline:

```bash
# Quality control
tier1 sc run-qc raw_data.h5ad qc_data.h5ad --doublets

# Dimensionality reduction
tier1 sc run-embed qc_data.h5ad embedded_data.h5ad --methods umap tsne

# Clustering
tier1 sc cluster embedded_data.h5ad clustered_data.h5ad --method leiden

# Trajectory analysis
tier1 sc paga clustered_data.h5ad final_data.h5ad --pseudotime

# Complete pipeline
tier1 sc pipeline raw_data.h5ad results/
```

### Multi-Omics Integration

Integrate multiple omics datasets:

```bash
# Fit integration models
tier1 multi fit rna.csv protein.csv metabolites.csv integration_models/

# Generate embeddings
tier1 multi embed rna.csv protein.csv metabolites.csv integration_models/ embeddings.csv

# Evaluate integration quality
tier1 multi eval embeddings.csv --biomarker --pathway

# Discover biomarkers
tier1 multi discover-biomarkers rna.csv protein.csv metabolites.csv embeddings.csv biomarkers/

# Complete pipeline
tier1 multi pipeline rna.csv protein.csv metabolites.csv multi_results/
```

## 🔧 Command Reference

### Bulk Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `tier1 bulk fit` | Train ML models | `--target`, `--models`, `--biomarker-val` |
| `tier1 bulk predict` | Make predictions | `--ensemble`, `--ci`, `--biomarker` |
| `tier1 bulk validate` | Validate results | `--truth`, `--report` |

### Single-Cell Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `tier1 sc run-qc` | Quality control | `--min-genes`, `--max-genes`, `--mito-thresh`, `--doublets` |
| `tier1 sc run-embed` | Dimensionality reduction | `--n-pcs`, `--neighbors`, `--methods`, `--batch-correct` |
| `tier1 sc cluster` | Cell clustering | `--method`, `--resolution`, `--annotate` |
| `tier1 sc paga` | Trajectory analysis | `--root`, `--pseudotime`, `--rejuv` |
| `tier1 sc pipeline` | Complete pipeline | `--config`, `--skip-qc`, `--skip-embed` |

### Multi-Omics Commands

| Command | Description | Key Options |
|---------|-------------|-------------|
| `tier1 multi fit` | Integration models | `--method`, `--factors`, `--batch`, `--biomarker` |
| `tier1 multi embed` | Generate embeddings | `--dim`, `--umap`, `--pathway` |
| `tier1 multi eval` | Evaluate integration | `--metrics`, `--biomarker`, `--pathway` |
| `tier1 multi discover-biomarkers` | Find biomarkers | `--method`, `--top-n`, `--pathway-filter` |
| `tier1 multi pipeline` | Complete pipeline | `--method`, `--skip-fit`, `--skip-eval` |

## 📊 Data Formats

### Input Formats
- **CSV/TSV**: Comma or tab-separated values
- **Excel**: `.xlsx`, `.xls` files
- **HDF5**: `.h5` files for large datasets
- **AnnData**: `.h5ad` files for single-cell data

### Data Structure
- **Rows**: Samples/cells
- **Columns**: Features/genes
- **Index**: Sample/cell identifiers
- **Headers**: Feature/gene names

### Example Data Format
```
        Gene1    Gene2    Gene3    ...
Sample1   5.2     3.1      0.8     ...
Sample2   4.8     2.9      1.2     ...
Sample3   6.1     3.5      0.6     ...
```

## 🧪 Biological Features

### Validated Biomarkers
- **110+ peer-reviewed aging biomarkers**
- **12 biological pathway categories**
- **Age-stratified analysis methods**
- **Cross-species validation**

### Pathway Categories
- Cell cycle regulation
- DNA damage response
- Autophagy and proteostasis
- Metabolic pathways
- Inflammation and immunity
- Stem cell markers
- Senescence indicators
- Epigenetic regulators

### Machine Learning Models
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machines
- Neural Networks (Autoencoders)

## 📚 Examples

### Example 1: Bulk RNA-seq Analysis

```bash
# Download example data
wget https://example.com/aging_rnaseq.csv

# Train models with aging biomarkers
tier1 bulk fit aging_rnaseq.csv models/ \
  --target age_group \
  --models rf,xgb \
  --biomarker-val \
  --verbose

# Make predictions on new samples
tier1 bulk predict new_samples.csv models/ predictions.csv \
  --ensemble voting \
  --ci \
  --biomarker
```

### Example 2: Single-Cell Rejuvenation Analysis

```bash
# Process single-cell data through complete pipeline
tier1 sc pipeline raw_pbmc.h5ad results/ \
  --config sc_config.yaml \
  --verbose

# Focus on trajectory analysis
tier1 sc paga results/data_with_clusters.h5ad final_trajectories.h5ad \
  --rejuv \
  --pseudotime \
  --plots
```

### Example 3: Multi-Omics Biomarker Discovery

```bash
# Integrate multi-omics data
tier1 multi fit \
  transcriptomics.csv \
  proteomics.csv \
  metabolomics.csv \
  integration/ \
  --method mofa \
  --factors 15 \
  --biomarker

# Discover cross-omics biomarkers
tier1 multi discover-biomarkers \
  transcriptomics.csv \
  proteomics.csv \
  metabolomics.csv \
  embeddings.csv \
  biomarkers/ \
  --top-n 50 \
  --pathway-filter
```

## 🔧 Configuration

### Configuration Files

Create YAML configuration files for complex analyses:

```yaml
# sc_config.yaml
qc:
  min_genes: 200
  max_genes: 5000
  mito_threshold: 20.0
  doublet_detection: true

embedding:
  n_pcs: 50
  n_neighbors: 15
  methods: ["umap", "tsne"]
  batch_correct: false

clustering:
  method: "leiden"
  resolution: 0.5
  biomarker_annotation: true
```

### Environment Variables

```bash
export TIER1_DATA_DIR=/path/to/data
export TIER1_RESULTS_DIR=/path/to/results
export TIER1_CACHE_DIR=/path/to/cache
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/lynchaos/tier1-rejuvenation-suite.git
cd tier1-rejuvenation-suite

# Create development environment
python -m venv tier1_dev
source tier1_dev/bin/activate  # On Windows: tier1_dev\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black tier1_suite/
flake8 tier1_suite/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Kemal Yaylali
- **Email**: kemal.yaylali@gmail.com
- **GitHub**: [@lynchaos](https://github.com/lynchaos)

## 🏆 Citation

If you use this software in your research, please cite:

```bibtex
@software{yaylali2025tier1,
  author = {Yaylali, Kemal},
  title = {TIER 1 Cellular Rejuvenation Suite: Biologically Validated Analysis Tools},
  year = {2025},
  url = {https://github.com/lynchaos/tier1-rejuvenation-suite}
}
```

## 🔍 Keywords

`bioinformatics` `cellular-rejuvenation` `aging-research` `biomarkers` `genomics` `single-cell` `multi-omics` `machine-learning` `python` `cli`