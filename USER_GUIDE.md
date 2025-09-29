# 🧬 TIER 1 Interactive User Guide

## Prerequisites and Installation

### 1. System Requirements
- **Operating System**: Linux (tested on Raspberry Pi OS)
- **Python**: 3.11.2 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 2GB free space for dependencies and generated data

### 2. Dependency Installation

#### Option A: Using Pre-configured Environment (Recommended)
If you already have the `tier1_env` directory:
```bash
cd /home/pi/projects
source tier1_env/bin/activate  # Activate existing environment
```

#### Option B: Fresh Installation
```bash
# Clone the repository
git clone https://github.com/lynchaos/tier1-rejuvenation-suite.git
cd tier1-rejuvenation-suite

# Create virtual environment
python3 -m venv tier1_env
source tier1_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import scanpy, sklearn, torch, pandas; print('✅ All dependencies installed')"
```

#### Key Dependencies Installed:
- **Machine Learning**: scikit-learn 1.7.2, XGBoost 3.0.5
- **Deep Learning**: PyTorch 2.8.0 (CPU version for compatibility)
- **Single-Cell Analysis**: scanpy 1.11.4, anndata, scanpy[leiden]
- **Data Processing**: pandas 2.2.3, numpy 2.1.2
- **Visualization**: matplotlib 3.9.2, seaborn 0.13.2
- **Statistics**: scipy 1.14.1, statsmodels 0.14.4
- **Reporting**: reportlab, fpdf2 (for PDF generation)

### 3. Quick Start

```bash
cd /home/pi/projects
source tier1_env/bin/activate  # Activate virtual environment
python tier1_interactive.py   # Start the interactive suite
```

### 2. Choose Your Workflow

The interactive menu offers several options:

**📊 Demo Data Workflow (Recommended for First Use)**
- Select option `1` to work with generated demo data
- Perfect for learning how each application works
- All three applications run automatically on sample datasets
- **Generated automatically**: No external downloads required

**🧬 Real-World Datasets**
- Select option `2` to work with real-world or generated datasets
- **Multiple data sources available** (see Data Sources section below)

**📖 Application Information**
- Select option `3` to learn about each application's capabilities
- **Includes**: Scientific reporting system details and technical specifications

## Application Overview

### 🧬 RegenOmics Master Pipeline
- **Purpose**: ML-driven bulk RNA-seq analysis and rejuvenation scoring
- **Input**: CSV files with gene expression data (samples × genes)
- **Output**: Rejuvenation scores with confidence intervals and categories
- **Methods**: Ensemble learning (Random Forest, XGBoost, Gradient Boosting)

### 🔬 Single-Cell Rejuvenation Atlas  
- **Purpose**: Interactive single-cell trajectory inference and reprogramming analysis
- **Input**: H5AD files (AnnData format)
- **Output**: UMAP visualizations, clustering, trajectory analysis
- **Methods**: Scanpy ecosystem with PAGA and trajectory inference

### 🧠 Multi-Omics Fusion Intelligence
- **Purpose**: AI-powered integration of multiple omics layers
- **Input**: Directory with RNA-seq, proteomics, and metabolomics CSV files
- **Output**: Integrated latent features and multi-modal analysis
- **Methods**: Deep learning autoencoders with PyTorch

### 📋 Scientific Reporting System
- **Peer-Review Quality**: Publication-ready reports with rigorous statistical analysis
- **Comprehensive Coverage**: Methodology, results, biological interpretation, clinical implications
- **Visual Analytics**: Publication-quality figures with statistical summaries
- **Reproducibility**: Timestamped reports with full technical details

## Data Sources and Formats

### Sample Data (Demo Mode)

#### 1. Generated Demo Datasets
When you select option `1` (Demo Data), the system automatically generates:

**Bulk RNA-seq Data** (`demo_data/bulk_rnaseq.csv`):
- **Size**: 50 samples × 500 genes
- **Format**: CSV with samples as rows, genes as columns
- **Content**: Realistic expression values with aging signatures
- **Features**: 25% of genes show age-related expression changes
- **Age Groups**: Young (20-40 years) and Old (60-80 years) samples

**Single-Cell Data** (`demo_data/single_cell.h5ad`):
- **Size**: 200 cells × 1,000 genes  
- **Format**: AnnData H5AD format (standard for single-cell analysis)
- **Content**: Simulated single-cell expression with aging annotations
- **Metadata**: Age groups, treatment conditions, cell type annotations

**Multi-Omics Data** (`demo_data/multi_omics/`):
- **RNA-seq**: 50 samples × 500 genes (`rnaseq.csv`)
- **Proteomics**: 50 samples × 300 proteins (`proteomics.csv`)
- **Metadata**: Sample annotations (`metadata.csv`)
- **Cross-correlation**: Biologically realistic relationships between omics layers

### Real-World Data Sources (Option 2)

#### 1. Single-Cell RNA-seq Datasets

**PBMC 3K Dataset** (via scanpy):
- **Source**: 10X Genomics public dataset
- **Size**: ~3,000 peripheral blood mononuclear cells
- **Genes**: ~32,000 genes (filtered to highly variable)
- **Format**: Automatically downloaded as H5AD
- **Use Case**: Standard benchmark for single-cell analysis methods
- **Added Annotations**: Age groups and treatment labels for rejuvenation analysis

**PBMC 68K Dataset** (via scanpy):
- **Source**: 10X Genomics public dataset  
- **Size**: ~68,000 cells (subsampled for efficiency)
- **Genes**: Pre-filtered and normalized
- **Format**: H5AD with reduced dimensionality
- **Use Case**: Larger dataset for robust clustering and trajectory analysis

#### 2. Bulk RNA-seq Datasets

**Generated Aging Signatures** (Recommended):
- **Small Dataset**: 50 samples × 500 genes
- **Large Dataset**: 200 samples × 2,000 genes
- **Aging Model**: Based on published aging transcriptomic signatures
- **Gene Categories**: 
  - Age-upregulated genes (senescence, inflammation)
  - Age-downregulated genes (DNA repair, autophagy)
  - Stable genes (housekeeping functions)
- **Biological Realism**: Expression patterns match known aging biology

**Alternative: Upload Your Own Data**:
- **Format Requirements**: CSV file with samples as rows, genes as columns
- **Minimum Size**: 10 samples × 100 genes
- **Gene Names**: Standard gene symbols (HUGO nomenclature preferred)
- **Sample Names**: Unique identifiers for each sample

#### 3. Multi-Omics Datasets

**Generated Integrated Datasets**:
- **RNA-seq Component**: Transcriptomic profiles matching proteomics samples
- **Proteomics Component**: Protein abundance data with realistic dynamic ranges
- **Cross-Omics Correlation**: Maintains biological relationships (mRNA-protein correlations)
- **Sample Matching**: Identical sample IDs across all omics layers
- **Format**: Separate CSV files in a common directory

## File Structure

```
/home/pi/projects/
├── tier1_interactive.py          # Main interactive interface
├── cell_rejuvenation_scoring.py  # RegenOmics implementation
├── single_cell_rejuvenation.py   # Single-Cell Atlas implementation
├── multi_omics_integration.py    # Multi-Omics Intelligence implementation
├── tier1_env/                    # Python virtual environment
├── demo_data/                    # Generated demo datasets
├── real_data/                    # Downloaded real-world datasets
├── models/                       # Trained ML models
└── figures/                      # Generated visualizations
```

## Advanced Usage and Configuration

### Environment Details

- **Python Version**: 3.11.2
- **Virtual Environment**: `tier1_env/` (isolated package management)
- **Key Dependencies**: 
  - **Machine Learning**: scikit-learn 1.7.2, XGBoost 3.0.5, ensemble methods
  - **Single-Cell Analysis**: scanpy 1.11.4, anndata, UMAP, Leiden clustering
  - **Deep Learning**: PyTorch 2.8.0 (CPU-optimized for compatibility)
  - **Statistical Analysis**: scipy 1.14.1, statsmodels 0.14.4
  - **Data Manipulation**: pandas 2.2.3, numpy 2.1.2
  - **Visualization**: matplotlib 3.9.2, seaborn 0.13.2, publication-quality plots

### Performance Optimization

**Memory Management**:
- **Small Datasets** (<1000 samples): Runs on 4GB RAM
- **Medium Datasets** (1000-10000 samples): Requires 8GB+ RAM  
- **Large Datasets** (>10000 samples): Consider cloud computing or workstation

**Processing Time Estimates**:
- **RegenOmics** (50 samples): ~2-3 minutes
- **Single-Cell** (3K cells): ~5-10 minutes
- **Multi-Omics** (50 samples): ~1-2 minutes
- **Report Generation**: Additional 30-60 seconds per application

### Data Format Requirements

#### Bulk RNA-seq (RegenOmics)
```csv
# File: expression_data.csv
Sample_ID,GENE1,GENE2,GENE3,...
Sample_001,12.45,8.92,15.67,...
Sample_002,11.23,9.45,14.12,...
```
- **Samples as Rows**: Each row represents one biological sample
- **Genes as Columns**: Each column represents one gene
- **Values**: Log2-transformed expression values preferred (but not required)
- **Missing Values**: Will be imputed automatically

#### Single-Cell (H5AD format)
- **Standard Format**: AnnData H5AD files from scanpy, Cell Ranger, or Seurat
- **Required Fields**: Expression matrix in `.X`, gene names in `.var`
- **Optional Metadata**: Cell annotations in `.obs` (age, treatment, cell type)
- **Preprocessing**: Raw counts acceptable (normalization performed automatically)

#### Multi-Omics Structure
```
multi_omics_data/
├── rnaseq.csv        # RNA-seq expression (samples × genes)
├── proteomics.csv    # Protein abundance (samples × proteins)  
├── metabolomics.csv  # Metabolite levels (samples × metabolites) [optional]
└── metadata.csv      # Sample annotations [optional]
```

## Tips for Success

1. **Start with Demo Data**: Always try option 1 first to understand the workflows
2. **Verify Data Formats**: Use the format requirements above as templates
3. **Check File Sizes**: Start with smaller datasets to test functionality
4. **Monitor Progress**: Applications provide detailed progress updates
5. **Scientific Reports**: Automatically generated in `reports/` directory with timestamp
6. **Error Recovery**: Most errors provide specific guidance for resolution

## Troubleshooting Guide

### Installation Issues

**Import Errors**:
```bash
# Ensure virtual environment is activated
source tier1_env/bin/activate
# Verify Python version
python --version  # Should be 3.11.2+
# Test key imports
python -c "import scanpy, sklearn, torch"
```

**Missing Dependencies**:
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
# For specific packages:
pip install scanpy[leiden] --upgrade
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Memory Issues**:
- **Solution 1**: Start with demo data (smaller datasets)
- **Solution 2**: Close other applications to free RAM
- **Solution 3**: Use cloud computing platforms (Google Colab, AWS)

### Runtime Issues

**File Format Errors**:
- **CSV Issues**: Ensure samples are rows, features are columns
- **H5AD Issues**: Use scanpy to read/write: `adata.write('file.h5ad')`
- **Missing Files**: Check file paths and permissions

**Single-Cell Specific**:
- **Single Cluster Warning**: Normal for homogeneous cell populations
- **Memory Error**: Reduce cell number or use subsampling
- **Trajectory Issues**: Requires >1 cluster for meaningful trajectories

**Multi-Omics Specific**:
- **Sample Mismatch**: Ensure identical sample IDs across omics files
- **Dimension Mismatch**: Check that all files have same number of samples
- **Missing Files**: Requires at least `rnaseq.csv` and `proteomics.csv`

### Data Quality Issues

**Poor Results**:
- **Check Data Quality**: Look for outliers, missing values, batch effects
- **Verify Biological Signal**: Ensure data contains relevant aging/rejuvenation signatures
- **Sample Size**: Minimum 10-20 samples for reliable statistical analysis
- **Gene Coverage**: Include known aging/rejuvenation markers when possible

**Report Generation Errors**:
- **Permission Issues**: Ensure write access to `reports/` directory
- **Disk Space**: Free up storage space for report files and figures
- **Dependencies**: Install matplotlib, seaborn for figure generation

## Scientific Interpretation Guide

### Understanding Results

**RegenOmics Scores**:
- **Range**: 0.0 (aged) to 1.0 (rejuvenated) 
- **Categories**: Aged < 0.2 < Partially Aged < 0.4 < Intermediate < 0.6 < Partially Rejuvenated < 0.8 < Rejuvenated
- **Confidence Intervals**: Wider intervals indicate higher uncertainty
- **Statistical Significance**: Check p-values in generated reports

**Single-Cell Analysis**:
- **Clustering**: More clusters indicate higher cellular heterogeneity
- **Trajectories**: Show cellular state transitions and developmental paths
- **UMAP Plots**: Spatial organization reflects transcriptional similarity
- **Biomarker Expression**: Identify rejuvenation-associated gene signatures

**Multi-Omics Integration**:
- **Latent Features**: Capture cross-omics relationships and regulatory networks
- **Dimensionality**: 20-50 latent features typically capture key biological signals
- **Cross-Modal Correlations**: Measure consistency across molecular layers

### Publication and Sharing

**Generated Reports**: 
- **Peer-Review Ready**: Include methodology, statistics, biological interpretation
- **Figures**: High-resolution, publication-quality visualizations
- **Reproducibility**: Full technical details for methods replication
- **Citation**: Reference the TIER 1 suite and underlying methods

## Next Steps

### For Researchers
1. **Validate Findings**: Use generated reports to design validation experiments
2. **Expand Analysis**: Apply to larger datasets or different tissues/conditions  
3. **Integrate Results**: Combine with other aging/rejuvenation research
4. **Collaborate**: Share reports with collaborators and domain experts

### For Developers
1. **Extend Applications**: Add new algorithms or visualization methods
2. **Optimize Performance**: Implement GPU acceleration or parallel processing
3. **Add Data Sources**: Integrate additional public datasets or APIs
4. **Enhance Reporting**: Add interactive dashboards or web interfaces

### For Educators
1. **Teaching Material**: Use reports as examples of computational biology analysis
2. **Student Projects**: Assign analysis of different aging datasets
3. **Curriculum Integration**: Incorporate into bioinformatics or systems biology courses

## Support and Resources

### Documentation
- **USER_GUIDE.md**: This comprehensive setup and usage guide
- **Application Code**: Well-documented Python modules with inline comments
- **Example Reports**: Generated automatically during demo runs

### Community
- **GitHub Repository**: https://github.com/lynchaos/tier1-rejuvenation-suite
- **Issue Tracking**: Report bugs or request features via GitHub Issues
- **Contributions**: Submit improvements via pull requests

### Citation
When using TIER 1 in publications, please cite:
```
TIER 1 Core Impact Applications Suite for Cell Rejuvenation Research
https://github.com/lynchaos/tier1-rejuvenation-suite
```

Your TIER 1 suite is ready for cutting-edge cell rejuvenation research! 🚀🧬