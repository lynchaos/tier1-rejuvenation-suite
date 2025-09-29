# Single-Cell Rejuvenation Atlas

## Overview
Interactive exploration platform for cellular rejuvenation landscapes, enabling analysis of aging → rejuvenation transitions through advanced single-cell analytics.

### Technology Stack
- **Python**: Scanpy for single-cell analysis
- **R**: Seurat for complementary analysis
- **Frontend**: Streamlit/Dash for interactive visualization
- **Data**: Cross-species aging datasets (human, mouse, primate)

### Core Capabilities
- Advanced trajectory inference: aging → rejuvenation transitions
- Cellular Reprogramming Predictor: optimal condition identification
- Multi-resolution clustering with aging-specific annotations
- Rejuvenation Trajectory Mapping: pseudotime analysis
- Cross-species comparison and analysis

### Specialized Features
- Senescence marker detection and visualization
- Stem cell pluripotency scoring algorithms
- Interactive cell-cell communication network analysis
- Digital Twin Cell Simulator: ODE-based rejuvenation kinetics modeling

## Directory Structure
- `python/`: Scanpy-based analysis scripts
- `r/`: Seurat-based R analysis scripts
- `streamlit/`: Interactive web interface
- `data/`: Sample datasets and references

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Load single-cell data into `data/` directory
3. Run analysis scripts in `python/` or `r/`
4. Launch interactive interface: `streamlit run streamlit/app.py`

---
For detailed analysis workflows, see documentation in each subfolder.