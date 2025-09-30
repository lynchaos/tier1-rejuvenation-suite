"""
Pytest configuration for TIER 1 Rejuvenation Suite tests.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_bulk_data():
    """Generate sample bulk omics data for testing."""
    np.random.seed(42)
    n_samples, n_features = 100, 1000
    
    # Generate expression data with some structure
    data = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_features))
    
    # Add some missing values (reduced to ensure some complete samples)
    missing_mask = np.random.random((n_samples, n_features)) < 0.001
    data[missing_mask] = np.nan
    
    # Create sample and feature names
    sample_names = [f"sample_{i:03d}" for i in range(n_samples)]
    feature_names = [f"gene_{i:04d}" for i in range(n_features)]
    
    df = pd.DataFrame(data, index=sample_names, columns=feature_names)
    
    # Add metadata
    metadata = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'condition': np.random.choice(['young', 'old'], n_samples),
        'batch': np.random.choice(['batch1', 'batch2', 'batch3'], n_samples),
        'tissue': np.random.choice(['liver', 'muscle', 'brain'], n_samples)
    }, index=sample_names)
    
    return df, metadata


@pytest.fixture
def sample_single_cell_data():
    """Generate sample single-cell data for testing."""
    np.random.seed(42)
    n_cells, n_genes = 500, 2000
    
    # Generate count data with dropout
    # Simulate different cell types with different expression patterns
    n_cell_types = 3
    cells_per_type = n_cells // n_cell_types
    
    X = []
    cell_types = []
    
    for i in range(n_cell_types):
        # Each cell type has different expression profile
        base_expression = np.random.gamma(2, 2, n_genes) * (i + 1) * 0.5
        
        for j in range(cells_per_type):
            # Add cell-specific noise and dropout
            cell_expr = np.random.poisson(base_expression)
            
            # Simulate dropout (more dropout in lowly expressed genes)
            dropout_prob = 1 / (1 + np.exp(-(5 - np.log1p(cell_expr))))
            dropout_mask = np.random.random(n_genes) < dropout_prob
            cell_expr[dropout_mask] = 0
            
            X.append(cell_expr)
            cell_types.append(f"celltype_{i}")
    
    # Handle remaining cells
    remaining_cells = n_cells - len(X)
    if remaining_cells > 0:
        for j in range(remaining_cells):
            base_expression = np.random.gamma(2, 2, n_genes) * 0.5
            cell_expr = np.random.poisson(base_expression)
            dropout_prob = 1 / (1 + np.exp(-(5 - np.log1p(cell_expr))))
            dropout_mask = np.random.random(n_genes) < dropout_prob
            cell_expr[dropout_mask] = 0
            X.append(cell_expr)
            cell_types.append(f"celltype_{n_cell_types-1}")
    
    X = np.array(X, dtype=np.float32)
    
    # Create AnnData object
    obs = pd.DataFrame({
        'cell_type': cell_types,
        'n_genes': (X > 0).sum(axis=1),
        'n_counts': X.sum(axis=1),
        'mt_frac': np.random.beta(2, 20, n_cells),  # Mitochondrial fraction
        'batch': np.random.choice(['batch_A', 'batch_B'], n_cells)
    })
    obs.index = [f"cell_{i:04d}" for i in range(n_cells)]
    
    var = pd.DataFrame({
        'gene_name': [f"gene_{i:04d}" for i in range(n_genes)],
        'mt': np.random.random(n_genes) < 0.05,  # 5% mitochondrial genes
        'highly_variable': False
    })
    var.index = [f"ENSG{i:08d}" for i in range(n_genes)]
    
    adata = ad.AnnData(X, obs=obs, var=var)
    
    return adata


@pytest.fixture
def sample_multi_omics_data():
    """Generate sample multi-omics datasets for testing."""
    np.random.seed(42)
    n_samples = 50
    
    # RNA-seq data
    n_genes = 500
    rna_data = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
    rna_df = pd.DataFrame(
        rna_data,
        index=[f"sample_{i:03d}" for i in range(n_samples)],
        columns=[f"gene_{i:04d}" for i in range(n_genes)]
    )
    
    # Proteomics data
    n_proteins = 100
    prot_data = np.random.normal(loc=0, scale=1, size=(n_samples, n_proteins))
    prot_df = pd.DataFrame(
        prot_data,
        index=[f"sample_{i:03d}" for i in range(n_samples)],
        columns=[f"protein_{i:03d}" for i in range(n_proteins)]
    )
    
    # Metabolomics data
    n_metabolites = 200
    metab_data = np.random.gamma(2, 2, size=(n_samples, n_metabolites))
    metab_df = pd.DataFrame(
        metab_data,
        index=[f"sample_{i:03d}" for i in range(n_samples)],
        columns=[f"metabolite_{i:03d}" for i in range(n_metabolites)]
    )
    
    # Shared metadata
    metadata = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'condition': np.random.choice(['control', 'treatment'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples)
    }, index=[f"sample_{i:03d}" for i in range(n_samples)])
    
    return {
        'rna': rna_df,
        'proteomics': prot_df,
        'metabolomics': metab_df,
        'metadata': metadata
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def deterministic_seed():
    """Set deterministic seed for reproducible tests."""
    np.random.seed(12345)
    # Also set random seeds for other libraries if needed
    import random
    random.seed(12345)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = '0'
    
    yield 12345


class TestDataGenerator:
    """Helper class for generating test data with specific properties."""
    
    @staticmethod
    def create_data_with_batch_effect(n_samples=100, n_features=50, n_batches=3):
        """Create data with known batch effects for testing batch correction."""
        np.random.seed(42)
        
        batch_labels = np.repeat(range(n_batches), n_samples // n_batches)
        remaining = n_samples - len(batch_labels)
        if remaining > 0:
            batch_labels = np.concatenate([batch_labels, [n_batches-1] * remaining])
        
        # Base expression
        base_data = np.random.normal(0, 1, (n_samples, n_features))
        
        # Add batch effects
        batch_effects = np.random.normal(0, 2, (n_batches, n_features))
        
        for i in range(n_samples):
            batch_id = batch_labels[i]
            base_data[i] += batch_effects[batch_id]
        
        df = pd.DataFrame(
            base_data,
            index=[f"sample_{i:03d}" for i in range(n_samples)],
            columns=[f"feature_{i:03d}" for i in range(n_features)]
        )
        
        metadata = pd.DataFrame({
            'batch': [f"batch_{b}" for b in batch_labels],
            'condition': np.random.choice(['A', 'B'], n_samples)
        }, index=df.index)
        
        return df, metadata
    
    @staticmethod
    def create_data_with_missing_values(n_samples=100, n_features=50, missing_rate=0.1):
        """Create data with controlled missing values."""
        np.random.seed(42)
        
        data = np.random.normal(0, 1, (n_samples, n_features))
        
        # Add missing values
        missing_mask = np.random.random((n_samples, n_features)) < missing_rate
        data[missing_mask] = np.nan
        
        df = pd.DataFrame(
            data,
            index=[f"sample_{i:03d}" for i in range(n_samples)],
            columns=[f"feature_{i:03d}" for i in range(n_features)]
        )
        
        return df
    
    @staticmethod
    def create_correlated_features(n_samples=100, n_features=50, correlation=0.8):
        """Create data with known feature correlations."""
        np.random.seed(42)
        
        # Create base features
        base_features = np.random.normal(0, 1, (n_samples, n_features // 2))
        
        # Create correlated features
        noise = np.random.normal(0, np.sqrt(1 - correlation**2), (n_samples, n_features // 2))
        corr_features = correlation * base_features + noise
        
        # Combine
        data = np.concatenate([base_features, corr_features], axis=1)
        
        df = pd.DataFrame(
            data,
            index=[f"sample_{i:03d}" for i in range(n_samples)],
            columns=[f"feature_{i:03d}" for i in range(n_features)]
        )
        
        return df