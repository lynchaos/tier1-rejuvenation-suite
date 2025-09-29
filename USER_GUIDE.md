# 🧬 TIER 1 Interactive User Guide

## Quick Start

Your TIER 1 Core Impact Applications suite is fully operational! Here's how to get started:

### 1. Run the Interactive Interface

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

**🧬 Real-World Datasets**
- Select option `2` to work with real-world or generated datasets
- Choose from:
  - **Single-Cell RNA-seq**: PBMC datasets via scanpy
  - **Bulk RNA-seq**: Generated datasets with aging signatures  
  - **Multi-Omics**: Generated multi-modal datasets

**📖 Application Information**
- Select option `3` to learn about each application's capabilities

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

## Environment Details

- **Python**: 3.11.2
- **Key Packages**: 
  - scikit-learn 1.7.2 (ML algorithms)
  - scanpy 1.11.4 (single-cell analysis)
  - PyTorch 2.8.0 (deep learning)
  - pandas, numpy, matplotlib (data manipulation)

## Tips for Success

1. **Start with Demo Data**: Always try option 1 first to understand the workflows
2. **Check Data Formats**: 
   - Bulk RNA-seq: CSV with samples as rows, genes as columns
   - Single-cell: H5AD format (AnnData objects)
   - Multi-omics: Directory with multiple CSV files
3. **Monitor Memory Usage**: Large datasets may require more processing time
4. **Generated Files**: Check `demo_data/`, `real_data/`, `models/`, and `figures/` directories for outputs

## Troubleshooting

- **Import Errors**: Ensure virtual environment is activated (`source tier1_env/bin/activate`)
- **Memory Issues**: Start with smaller datasets (demo data works well)
- **File Format Issues**: The interface automatically handles format conversion
- **Single-Cell Clustering**: If only 1 cluster is found, trajectory analysis is skipped (this is normal)

## Next Steps

1. Run the interactive interface to explore all three applications
2. Try different dataset sizes and types
3. Examine generated visualizations in the `figures/` directory
4. Review model performance in the terminal output
5. Use the confidence intervals and categories for biological interpretation

Your TIER 1 suite is ready for cell rejuvenation research! 🚀