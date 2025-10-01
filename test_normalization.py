#!/usr/bin/env python3
"""
Unit tests for tier1_interactive normalization functions.
Validates zero-library sample handling and different data type detection.
"""
import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tier1_interactive import _normalize_bulk_rnaseq

def test_zero_library_sample_removal():
    """Test that zero-library samples are properly removed during CPM normalization."""
    print("🧪 Testing zero-library sample removal...")
    
    # Create test data with some zero-library samples
    data = pd.DataFrame({
        'gene1': [10, 20, 0, 30, 0],
        'gene2': [5, 15, 0, 25, 0], 
        'gene3': [8, 12, 0, 18, 0],
        'gene4': [2, 8, 0, 12, 0]
    }, index=['sample1', 'sample2', 'zero_lib1', 'sample4', 'zero_lib2'])
    
    print(f"   📊 Original data shape: {data.shape}")
    print(f"   📊 Library sizes: {data.sum(axis=1).to_dict()}")
    
    # Test CPM + log1p normalization (should remove zero-library samples)
    normalized, qc = _normalize_bulk_rnaseq(data, method="cpm_log1p")
    
    print(f"   ✅ Normalized shape: {normalized.shape}")
    print(f"   ✅ Samples removed: {qc.get('samples_removed_due_to_zero_library', 0)}")
    print(f"   ✅ Removed sample IDs: {qc.get('removed_sample_ids', [])}")
    
    # Assertions
    assert normalized.shape[0] == 3, f"Expected 3 samples after removal, got {normalized.shape[0]}"
    assert qc['samples_removed_due_to_zero_library'] == 2, "Should remove 2 zero-library samples"
    assert 'zero_lib1' not in normalized.index, "zero_lib1 should be removed"
    assert 'zero_lib2' not in normalized.index, "zero_lib2 should be removed"
    assert 'sample1' in normalized.index, "sample1 should remain"
    assert normalized.isna().sum().sum() == 0, "No NaN values should remain"
    assert len(qc['removed_sample_ids']) == 2, "Should track removed sample IDs"
    
    print("   ✅ Zero-library sample removal test passed!")

def test_log_transformed_data():
    """Test that already log-transformed data is detected and not double-transformed."""
    print("\n🧪 Testing log-transformed data detection...")
    
    # Create log-transformed-like data (values mostly between 0-15, some decimals)
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.lognormal(mean=1.5, sigma=1.0, size=(20, 100)) + 
        np.random.normal(0, 0.1, size=(20, 100))
    )
    # Clip to reasonable log-space range
    data = data.clip(0, 20)
    
    print(f"   📊 Data shape: {data.shape}")
    print(f"   📊 Value range: {data.min().min():.3f} - {data.max().max():.3f}")
    print(f"   📊 Mean value: {data.mean().mean():.3f}")
    
    normalized, qc = _normalize_bulk_rnaseq(data, method="auto")
    
    print(f"   ✅ Auto-detected method: {qc.get('auto_detection', 'unknown')}")
    print(f"   ✅ Normalization applied: {qc['normalization']}")
    
    # Should detect as already normalized and apply minimal transformation
    assert qc['normalization'] in ['log1p', 'none'], f"Expected minimal transformation, got {qc['normalization']}"
    
    print("   ✅ Log-transformed data detection test passed!")

def test_count_data_detection():
    """Test that raw count data is detected and properly normalized."""
    print("\n🧪 Testing count data detection...")
    
    # Create count-like data (integers, higher values)
    np.random.seed(42)
    counts = np.random.poisson(lam=100, size=(30, 50)).astype(int)
    data = pd.DataFrame(counts, 
                       index=[f'sample_{i:02d}' for i in range(30)],
                       columns=[f'gene_{i:03d}' for i in range(50)])
    
    print(f"   📊 Data shape: {data.shape}")
    print(f"   📊 Value range: {data.min().min()} - {data.max().max()}")
    print(f"   📊 Integer fraction: {((data.round() == data).mean().mean()):.3f}")
    
    normalized, qc = _normalize_bulk_rnaseq(data, method="auto")
    
    print(f"   ✅ Auto-detected method: {qc.get('auto_detection', 'unknown')}")
    print(f"   ✅ Normalization applied: {qc['normalization']}")
    
    # Should detect as counts and apply CPM + log1p
    assert qc.get('auto_detection') == 'detected_counts', f"Expected count detection, got {qc.get('auto_detection')}"
    assert qc['normalization'] == 'CPM_log1p', f"Expected CPM_log1p, got {qc['normalization']}"
    
    print("   ✅ Count data detection test passed!")

def test_qc_metrics_completeness():
    """Test that QC metrics are comprehensive and useful."""
    print("\n🧪 Testing QC metrics completeness...")
    
    # Create test data with mixed characteristics
    data = pd.DataFrame({
        'gene_A': [100, 200, 0, 150],
        'gene_B': [50, 180, 0, 120], 
        'gene_C': [0, 0, 0, 0],  # Zero gene
    }, index=['sample1', 'sample2', 'zero_sample', 'sample4'])
    
    normalized, qc = _normalize_bulk_rnaseq(data, method="cpm_log1p")
    
    print(f"   📊 QC metrics keys: {list(qc.keys())}")
    
    # Check essential QC metrics are present
    required_metrics = [
        'orientation', 'n_samples', 'n_genes', 'original_shape', 'final_shape',
        'zero_genes', 'zero_samples', 'normalization', 'samples_removed_due_to_zero_library'
    ]
    
    for metric in required_metrics:
        assert metric in qc, f"Missing required QC metric: {metric}"
    
    print(f"   ✅ Zero genes detected: {qc['zero_genes']}")
    print(f"   ✅ Samples removed: {qc['samples_removed_due_to_zero_library']}")
    print(f"   ✅ Shape change: {qc['original_shape']} → {qc['final_shape']}")
    
    print("   ✅ QC metrics completeness test passed!")

def run_all_tests():
    """Run all normalization tests."""
    print("🚀 Running TIER 1 Normalization Unit Tests\n" + "="*50)
    
    try:
        test_zero_library_sample_removal()
        test_log_transformed_data() 
        test_count_data_detection()
        test_qc_metrics_completeness()
        
        print(f"\n🎉 All tests passed! Normalization pipeline is robust.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)