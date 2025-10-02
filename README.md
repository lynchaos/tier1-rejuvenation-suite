# TIER 1 Cell Rejuvenation Suite

A comprehensive bioinformatics analysis suite for cellular aging research with machine learning-powered rejuvenation scoring.

## Features

- **Cellular Rejuvenation Analysis**: ML-powered scoring of cellular age reversal
- **Multi-format Reporting**: Professional scientific reports (Markdown, HTML, PDF)
- **Aging Biomarker Validation**: Peer-reviewed aging gene panels
- **Statistical Rigor**: Bootstrap confidence intervals, FDR correction, normalization
- **Real-world Data Support**: Bulk RNA-seq analysis with biological validation

## Advanced Scientific Visualization Suite

Comprehensive mathematical, machine learning, and statistical analysis visualization:
- **5 Analysis Categories**: Statistical distributions, ML performance, mathematical modeling, statistical testing, biological pathways
- **Publication Standards**: 300 DPI resolution, professional matplotlib/seaborn styling
- **Mathematical Analysis**: Correlation matrices, PCA visualization, sigmoid modeling, bootstrap confidence intervals
- **Machine Learning Metrics**: Cross-validation tracking, feature importance, learning curves, ROC analysis
- **Statistical Testing**: t-tests, ANOVA, regression analysis with confidence bands, power analysis
- **Biological Networks**: Pathway enrichment, gene expression heatmaps, regulatory network visualization
- **Enhanced HTML/PDF**: Base64-embedded figures with responsive design and hover effects
- **Automated Integration**: All figures automatically generated and embedded in reports

## Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd tier1-rejuvenation-suite
pip install -e .
```

### Basic Usage

```bash
# Interactive analysis with comprehensive aging data
python tier1_interactive.py --mode real --app bulk --path your_data.csv

# The suite will automatically:
# - Load and normalize your expression data
# - Apply aging biomarker analysis
# - Generate rejuvenation scores
# - Create professional reports (MD/HTML/PDF)
```

### Expected Data Format

CSV file with samples as rows and genes as columns:
```csv
sample_id,GENE1,GENE2,GENE3,...,age,sex,tissue
Sample_001,8.2,12.5,14.1,...,28,F,fibroblast
Sample_002,9.1,14.2,15.8,...,32,M,fibroblast
```

## Output

The analysis generates comprehensive results with supporting visualizations:

### Professional Reports
- `reports/RegenOmics_Report_YYYYMMDD_HHMMSS.md` (Markdown)
- `reports/RegenOmics_Report_YYYYMMDD_HHMMSS.html` (HTML)  
- `reports/RegenOmics_Report_YYYYMMDD_HHMMSS.pdf` (PDF)

### Statistical Analysis
- Rejuvenation scores with confidence intervals
- Age-stratified statistical validation
- Bootstrap analysis and normality testing
- Cross-validation performance metrics

### Publication-Quality Figures
- **Score Distribution Plots**: Histograms and box plots of rejuvenation scores
- **Statistical Summary Charts**: Mean, median, quartiles visualization
- **Category Analysis Graphs**: Rejuvenation classification breakdowns
- **Age Correlation Figures**: Age vs rejuvenation score relationships
- All figures saved as high-resolution PNG files (300 DPI)

## Example Results

```
✅ BIOLOGICALLY VALIDATED ANALYSIS COMPLETE!
📊 Scored 16 samples
📈 Mean rejuvenation score: 0.493
📉 Score range: 0.036 - 0.955
🔬 SCIENTIFIC VALIDATION METRICS:
   📊 Normality: p=0.229, Bootstrap CI: [0.346, 0.640]
   📊 Cross-validation R²: 0.920 ± 0.033

🏆 Top rejuvenated samples:
   Patient_009: 0.955 | age: 25 | Rejuvenated
   Patient_001: 0.885 | age: 28 | Rejuvenated
```

## Requirements

- Python 3.8+
- Required packages: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- Optional: pdfkit + wkhtmltopdf (for PDF reports)
## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{tier1_rejuvenation_suite,
  title={TIER 1 Cell Rejuvenation Suite},
  author={Yaylali, Kemal},
  year={2025},
  url={https://github.com/lynchaos/tier1-rejuvenation-suite}
}
