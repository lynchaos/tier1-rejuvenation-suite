# TIER 1 Rejuvenation Suite

A comprehensive bioinformatics analysis suite for aging research with multi-omics integration, single-cell analysis, and machine learning capabilities.

## Quick Start

### Installation

```bash
# Install directly from source
pip install -e .

# Or using the environment manager
./env_manager.sh create-dev
conda activate tier1-rejuvenation-suite
./env_manager.sh install
```

### CLI Usage

The suite provides three main command groups:

```bash
# Bulk analysis commands
tier1 bulk fit --data data.csv --output results/
tier1 bulk predict --model model.pkl --data test.csv

# Single-cell analysis commands  
tier1 sc run-qc --data sc_data.h5ad --output qc_results/
tier1 sc run-embed --data processed.h5ad --method umap
tier1 sc cluster --data embedded.h5ad --algorithm leiden
tier1 sc paga --data clustered.h5ad --root-cell root_0

# Multi-omics integration
tier1 multi fit --data-dir multi_omics/ --output integration/
tier1 multi embed --data integrated.pkl --method mofa
tier1 multi eval --true-labels labels.csv --predictions pred.csv
```

## Environment Management

### Using the Environment Manager

The `env_manager.sh` script provides comprehensive environment management:

```bash
# Development setup (full environment with all tools)
./env_manager.sh create-dev
conda activate tier1-rejuvenation-suite

# Production setup (minimal dependencies)
./env_manager.sh create-prod
conda activate tier1-rejuvenation-prod

# Install the package
./env_manager.sh install

# Test installation
./env_manager.sh test

# Update dependency locks
./env_manager.sh update-lock

# Build Docker images
./env_manager.sh build-docker
```

### Manual Environment Setup

#### Development Environment

```bash
# Create from environment file
conda env create -f environment.yml
conda activate tier1-rejuvenation-suite

# Install package in development mode
pip install -e .
```

#### Production Environment

```bash
# Minimal environment for production
conda env create -f environment-prod.yml
conda activate tier1-rejuvenation-prod

# Install package
pip install .
```

## Docker Deployment

### Using Docker Compose

```bash
# Start development environment
docker-compose up tier1-dev

# Start production environment  
docker-compose up tier1-prod

# Interactive development session
docker-compose run --rm tier1-dev bash
```

### Manual Docker Build

```bash
# Build production image
docker build --target production -t tier1-suite:latest .

# Build development image
docker build --target development -t tier1-suite:dev .

# Run analysis
docker run -v $(pwd)/data:/app/data tier1-suite:latest \
  tier1 bulk fit --data /app/data/input.csv --output /app/data/results/
```

## Dependencies

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, Windows with WSL
- **Python**: 3.11+
- **Memory**: 8GB+ recommended for large datasets
- **Storage**: 5GB+ for full development environment

### Key Dependencies

- **Core**: pandas, numpy, scikit-learn, scipy
- **Single-cell**: scanpy, anndata, scvelo, paga
- **Machine Learning**: torch, transformers, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn, plotly
- **Bioinformatics**: pysam, pyvcf, biopython
- **Multi-omics**: muon, omicverse

### Environment Files

| File | Purpose | Size | Use Case |
|------|---------|------|----------|
| `environment.yml` | Full development environment | ~100 packages | Development, research |
| `environment-prod.yml` | Minimal production environment | ~30 packages | Production deployment |
| `requirements.in` | pip-tools specification | Core deps | pip-based installs |
| `requirements.txt` | Pinned pip requirements | Locked versions | Reproducible installs |

## CLI Reference

### Bulk Analysis (`tier1 bulk`)

```bash
# Fit models on bulk data
tier1 bulk fit \
  --data data/bulk_expression.csv \
  --output results/bulk_models/ \
  --model-type xgboost \
  --cv-folds 5

# Predict using trained models
tier1 bulk predict \
  --model results/bulk_models/best_model.pkl \
  --data data/test_samples.csv \
  --output results/predictions.csv
```

### Single-Cell Analysis (`tier1 sc`)

```bash
# Quality control pipeline
tier1 sc run-qc \
  --data data/raw_counts.h5ad \
  --output results/qc/ \
  --min-genes 200 \
  --max-genes 5000 \
  --mt-percent 20

# Embedding and dimensionality reduction
tier1 sc run-embed \
  --data data/filtered.h5ad \
  --method umap \
  --n-components 2 \
  --output results/embedded.h5ad

# Clustering analysis
tier1 sc cluster \
  --data results/embedded.h5ad \
  --algorithm leiden \
  --resolution 0.5 \
  --output results/clustered.h5ad

# Trajectory analysis with PAGA
tier1 sc paga \
  --data results/clustered.h5ad \
  --root-cell cell_type_stem \
  --output results/trajectories/
```

### Multi-Omics Integration (`tier1 multi`)

```bash
# Integrate multiple omics datasets
tier1 multi fit \
  --data-dir data/multi_omics/ \
  --output results/integration/ \
  --method mofa \
  --factors 10

# Embed integrated data
tier1 multi embed \
  --data results/integration/integrated_data.pkl \
  --method umap \
  --output results/multi_embedding.pkl

# Evaluate integration quality
tier1 multi eval \
  --true-labels data/cell_types.csv \
  --predictions results/predicted_types.csv \
  --metrics ari silhouette \
  --output results/evaluation_report.json
```

## Development

### Project Structure

```
tier1_suite/
├── cli.py                 # Main CLI entry point
├── bulk/                  # Bulk analysis modules
│   ├── analyzer.py
│   └── bulk_cli.py
├── single_cell/           # Single-cell analysis
│   ├── analyzer.py
│   └── single_cell_cli.py
├── multi_omics/           # Multi-omics integration
│   ├── analyzer.py
│   └── multi_omics_cli.py
└── utils/                 # Shared utilities
    └── __init__.py

environment/               # Environment management
├── environment.yml        # Development environment
├── environment-prod.yml   # Production environment
├── requirements.in        # pip-tools input
├── requirements.txt       # Locked requirements
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Docker services
└── env_manager.sh        # Environment manager script
```

### Contributing

1. **Setup development environment**:
   ```bash
   ./env_manager.sh create-dev
   conda activate tier1-rejuvenation-suite
   ./env_manager.sh install
   ```

2. **Run tests**:
   ```bash
   ./env_manager.sh test
   pytest tests/ -v
   ```

3. **Update dependencies**:
   ```bash
   ./env_manager.sh update-lock
   ```

4. **Build and test Docker images**:
   ```bash
   ./env_manager.sh build-docker
   ```

### Testing

```bash
# Run all tests
./env_manager.sh test

# Run specific test categories
pytest tests/test_bulk.py -v
pytest tests/test_single_cell.py -v  
pytest tests/test_multi_omics.py -v

# Test CLI commands
tier1 --help
tier1 bulk --help
tier1 sc --help
tier1 multi --help
```

## Troubleshooting

### Common Issues

#### Environment Creation Fails
```bash
# Clean conda cache
conda clean --all

# Recreate environment
./env_manager.sh cleanup
./env_manager.sh create-dev
```

#### Package Installation Issues
```bash
# Update conda and pip
conda update conda
pip install --upgrade pip

# Reinstall package
pip uninstall tier1-suite
./env_manager.sh install
```

#### Memory Issues with Large Datasets
```bash
# Use production environment (lighter)
./env_manager.sh create-prod

# Or increase memory limits
export OMP_NUM_THREADS=4
ulimit -m 8388608  # 8GB limit
```

#### Docker Build Issues
```bash
# Clean Docker cache
docker system prune -f

# Rebuild with no cache
docker build --no-cache --target production -t tier1-suite:latest .
```

### Performance Optimization

- Use production environment for large-scale analysis
- Set `OMP_NUM_THREADS` for parallel processing
- Use SSD storage for temporary files
- Consider using Docker for isolated environments
- Monitor memory usage with large single-cell datasets

## License

MIT License - see LICENSE file for details.

## Citation

If you use TIER 1 Rejuvenation Suite in your research, please cite:

```bibtex
@software{tier1_suite,
  title={TIER 1 Rejuvenation Suite: Comprehensive Bioinformatics for Aging Research},
  author={Your Name},
  year={2024},
  url={https://github.com/yourorg/tier1-suite}
}
```

## Support

- **Documentation**: See this README and inline help (`tier1 --help`)
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join our community discussions for usage questions
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