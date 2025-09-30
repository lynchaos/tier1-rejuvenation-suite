"""
TIER 1 Rejuvenation Suite
========================

Biologically validated cellular rejuvenation analysis suite with comprehensive biomarker validation.

This package provides CLI tools for:
- Bulk data processing and ML model fitting/prediction
- Single-cell analysis with QC, embedding, clustering, and trajectory analysis
- Multi-omics integration and evaluation
"""

__version__ = "1.0.0"
__author__ = "Kemal Yaylali"
__email__ = "kemal.yaylali@gmail.com"

try:
    from .bulk import BulkAnalyzer
    from .single_cell import SingleCellAnalyzer
    from .multi_omics import MultiOmicsAnalyzer
    
    __all__ = [
        "BulkAnalyzer",
        "SingleCellAnalyzer", 
        "MultiOmicsAnalyzer",
    ]
except ImportError:
    # Handle import errors gracefully
    __all__ = []