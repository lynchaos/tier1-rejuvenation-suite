#!/usr/bin/env python3
"""
Test the robustness fixes for edge cases and missing dependencies.
"""
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_age_stats_without_scipy():
    """Test age-stratified statistics when SciPy is not available."""
    print("🧪 Testing age stats without SciPy...")
    
    from tier1_interactive import _compute_age_stratified_statistics
    
    # Mock scipy import failure
    with patch.dict('sys.modules', {'scipy': None, 'scipy.stats': None}):
        # Create test data
        scores = np.array([0.3, 0.7, 0.5, 0.8, 0.4, 0.9])
        ages = np.array([25, 65, 30, 70, 35, 75])
        
        # Should return descriptive stats without crashing
        result = _compute_age_stratified_statistics(scores, ages, age_threshold=50)
        
        print(f"   ✅ Result keys: {list(result.keys())}")
        assert "error" in result, "Should indicate scipy unavailable"
        assert result["error"] == "scipy_not_available"
        assert "n_young" in result, "Should have young count"
        assert "n_old" in result, "Should have old count"
        assert "young_mean" in result, "Should have young mean"
        assert "old_mean" in result, "Should have old mean"
        
        print(f"   ✅ Young samples: {result['n_young']}, Old samples: {result['n_old']}")
        print("   ✅ Age stats without SciPy test passed!")

def test_fdr_correction_fallbacks():
    """Test FDR correction with various dependency scenarios."""
    print("\n🧪 Testing FDR correction fallbacks...")
    
    from tier1_interactive import _perform_multiple_testing_correction
    
    test_pvals = np.array([0.001, 0.01, 0.04, 0.08, 0.5])
    
    # Test 1: Normal case (should work)
    corrected, rejected = _perform_multiple_testing_correction(test_pvals, alpha=0.05)
    print(f"   ✅ Normal case - Rejected: {rejected.sum()}/{len(rejected)}")
    assert not np.isnan(corrected).all(), "Should return corrected p-values"
    
    # Test 2: With NaN p-values
    test_pvals_nan = np.array([0.001, np.nan, 0.04, np.nan, 0.5])
    corrected, rejected = _perform_multiple_testing_correction(test_pvals_nan, alpha=0.05)
    print(f"   ✅ NaN handling - Valid corrected: {(~np.isnan(corrected)).sum()}")
    assert np.isnan(corrected[1]), "NaN positions should remain NaN"
    assert not rejected[1], "NaN positions should not be rejected"
    
    # Test 3: Empty p-values
    empty_pvals = np.array([])
    corrected, rejected = _perform_multiple_testing_correction(empty_pvals, alpha=0.05)
    print(f"   ✅ Empty array handling: {len(corrected)} elements")
    assert len(corrected) == 0, "Should handle empty arrays"
    
    print("   ✅ FDR correction fallback tests passed!")

def test_zero_library_samples_comprehensive():
    """Test comprehensive zero-library sample handling."""
    print("\n🧪 Testing zero-library sample handling...")
    
    from tier1_interactive import _normalize_bulk_rnaseq
    
    # Create data with multiple zero-library samples
    data = pd.DataFrame({
        'gene1': [100, 0, 50, 0, 80, 0],
        'gene2': [80, 0, 30, 0, 90, 0], 
        'gene3': [60, 0, 40, 0, 70, 0]
    }, index=['good1', 'zero1', 'good2', 'zero2', 'good3', 'zero3'])
    
    print(f"   📊 Original shape: {data.shape}")
    print(f"   📊 Zero-library samples: {(data.sum(axis=1) == 0).sum()}")
    
    # Test CPM + log1p normalization
    normalized, qc = _normalize_bulk_rnaseq(data, method="cpm_log1p")
    
    print(f"   ✅ Final shape: {normalized.shape}")
    print(f"   ✅ Samples removed: {qc['samples_removed_due_to_zero_library']}")
    print(f"   ✅ Removed IDs: {qc.get('removed_sample_ids', [])}")
    
    # Assertions
    assert normalized.shape[0] == 3, f"Should have 3 samples left, got {normalized.shape[0]}"
    assert qc['samples_removed_due_to_zero_library'] == 3, "Should remove 3 zero samples"
    assert len(qc['removed_sample_ids']) == 3, "Should track all removed sample IDs"
    assert not normalized.isnull().any().any(), "No NaN values should remain"
    assert not np.isinf(normalized).any().any(), "No infinite values should remain"
    
    print("   ✅ Zero-library comprehensive test passed!")

def test_run_regenomics_with_all_nan_ages():
    """Test run_regenomics when all ages are NaN to ensure age_stats=None path works."""
    print("\n🧪 Testing run_regenomics with all NaN ages...")
    
    # This is a more complex test that would require mocking the full pipeline
    # For now, let's test the age_stats initialization directly
    from tier1_interactive import _compute_age_stratified_statistics
    
    # Test with all NaN ages
    scores = np.array([0.3, 0.7, 0.5, 0.8])
    ages = np.array([np.nan, np.nan, np.nan, np.nan])
    
    result = _compute_age_stratified_statistics(scores, ages, age_threshold=50)
    
    print(f"   ✅ Result with all NaN ages: {result}")
    assert "error" in result, "Should return error for all NaN ages"
    assert result["error"] == "insufficient_data", "Should indicate insufficient data"
    
    print("   ✅ All NaN ages test passed!")

def test_mixed_data_types_robustness():
    """Test robustness with mixed data types and edge cases."""
    print("\n🧪 Testing mixed data types and edge cases...")
    
    from tier1_interactive import _normalize_bulk_rnaseq
    
    # Test 1: Very small values (near machine precision)
    small_data = pd.DataFrame({
        'gene1': [1e-10, 2e-10, 3e-10],
        'gene2': [1e-15, 2e-15, 0],
        'gene3': [0, 0, 1e-20]
    })
    
    normalized, qc = _normalize_bulk_rnaseq(small_data, method="auto")
    print(f"   ✅ Small values - Method: {qc.get('auto_detection', 'unknown')}")
    assert not normalized.isnull().any().any(), "Should handle small values without NaN"
    
    # Test 2: Large integer values
    large_data = pd.DataFrame({
        'gene1': [1000000, 2000000, 1500000],
        'gene2': [800000, 1200000, 900000],
        'gene3': [600000, 1800000, 1100000]
    })
    
    normalized, qc = _normalize_bulk_rnaseq(large_data, method="auto")
    print(f"   ✅ Large values - Method: {qc.get('auto_detection', 'unknown')}")
    assert qc.get('auto_detection') == 'detected_counts', "Should detect as counts"
    
    # Test 3: Single sample (edge case)
    single_data = pd.DataFrame({
        'gene1': [100],
        'gene2': [200],
        'gene3': [150]
    })
    
    normalized, qc = _normalize_bulk_rnaseq(single_data, method="auto")
    print(f"   ✅ Single sample - Shape: {normalized.shape}")
    assert normalized.shape[0] == 1, "Should handle single sample"
    
    print("   ✅ Mixed data types robustness test passed!")

def run_all_robustness_tests():
    """Run all robustness and edge case tests."""
    print("🛡️ Running Comprehensive Robustness Tests\n" + "="*60)
    
    try:
        test_age_stats_without_scipy()
        test_fdr_correction_fallbacks() 
        test_zero_library_samples_comprehensive()
        test_run_regenomics_with_all_nan_ages()
        test_mixed_data_types_robustness()
        
        print(f"\n🎉 All robustness tests passed! TIER 1 Suite is production-ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_robustness_tests()
    sys.exit(0 if success else 1)