#!/usr/bin/env python3
"""
Test script to verify zero-library sample handling in normalization.
"""
import numpy as np
import pandas as pd
from tier1_interactive import _normalize_bulk_rnaseq

def test_zero_library_samples():
    """Test that zero-library samples are properly handled."""
    # Create test data with some zero-library samples
    data = pd.DataFrame({
        'gene1': [10, 20, 0, 30],
        'gene2': [5, 15, 0, 25], 
        'gene3': [8, 12, 0, 18]
    }, index=['sample1', 'sample2', 'sample3', 'sample4'])
    
    print("Original data:")
    print(data)
    print(f"Library sizes: {data.sum(axis=1).values}")
    
    # Test CPM + log1p normalization (should remove zero-library samples)
    normalized, qc = _normalize_bulk_rnaseq(data, method="cpm_log1p")
    
    print(f"\nAfter CPM + log1p normalization:")
    print(f"Shape: {normalized.shape} (should be 3x3, removing sample3)")
    print(f"Samples removed due to zero library: {qc.get('samples_removed_due_to_zero_library', 0)}")
    print("Normalized data:")
    print(normalized)
    
    # Verify no NaN values remain
    nan_count = normalized.isna().sum().sum()
    print(f"NaN values in normalized data: {nan_count}")
    
    assert normalized.shape[0] == 3, f"Expected 3 samples after removing zero-library sample, got {normalized.shape[0]}"
    assert nan_count == 0, f"Expected no NaN values, found {nan_count}"
    assert 'sample3' not in normalized.index, "Zero-library sample should have been removed"
    
    print("✅ Zero-library sample handling test passed!")

if __name__ == "__main__":
    test_zero_library_samples()