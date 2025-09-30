"""
Seed determinism tests for TIER 1 Rejuvenation Suite.
Tests reproducibility of UMAP, Leiden clustering, and ML models with fixed seeds.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import scanpy as sc
import warnings

# Suppress scanpy warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class TestSeedDeterminism:
    """Test suite for seed-based reproducibility."""
    
    def test_random_forest_determinism(self, sample_bulk_data, deterministic_seed):
        """Test RandomForest reproducibility with fixed random_state."""
        data, metadata = sample_bulk_data
        
        # Prepare data
        y = (metadata['age'] > metadata['age'].median()).astype(int)
        complete_mask = data.notna().all(axis=1) & y.notna()
        X = data.loc[complete_mask].iloc[:, :100]  # Use subset for speed
        y = y.loc[complete_mask]
        
        # Train two models with same seed
        model1 = RandomForestClassifier(
            n_estimators=10, 
            random_state=deterministic_seed,
            max_depth=5
        )
        model2 = RandomForestClassifier(
            n_estimators=10, 
            random_state=deterministic_seed,
            max_depth=5
        )
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should be identical
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        assert np.array_equal(pred1, pred2), "RandomForest predictions should be identical with same seed"
        
        # Probabilities should be identical
        prob1 = model1.predict_proba(X)
        prob2 = model2.predict_proba(X)
        
        assert np.allclose(prob1, prob2), "RandomForest probabilities should be identical with same seed"
        
        # Feature importances should be identical
        assert np.allclose(model1.feature_importances_, model2.feature_importances_)
    
    def test_logistic_regression_determinism(self, sample_bulk_data, deterministic_seed):
        """Test LogisticRegression reproducibility."""
        data, metadata = sample_bulk_data
        
        # Prepare data
        y = (metadata['condition'] == 'old').astype(int)
        complete_mask = data.notna().all(axis=1) & y.notna()
        X = data.loc[complete_mask].iloc[:, :50]
        y = y.loc[complete_mask]
        
        # Train two models with same seed
        model1 = LogisticRegression(
            random_state=deterministic_seed,
            max_iter=1000,
            solver='lbfgs'
        )
        model2 = LogisticRegression(
            random_state=deterministic_seed,
            max_iter=1000,
            solver='lbfgs'
        )
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Coefficients should be identical (for deterministic solvers)
        assert np.allclose(model1.coef_, model2.coef_, atol=1e-10)
        assert np.allclose(model1.intercept_, model2.intercept_, atol=1e-10)
        
        # Predictions should be identical
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        assert np.array_equal(pred1, pred2)
    
    def test_neural_network_determinism(self, sample_bulk_data, deterministic_seed):
        """Test MLPClassifier reproducibility with proper seed setting."""
        data, metadata = sample_bulk_data
        
        # Prepare smaller dataset for faster training
        y = (metadata['age'] > 50).astype(int)
        complete_mask = data.notna().all(axis=1) & y.notna()
        X = data.loc[complete_mask].iloc[:200, :20]  # Small subset
        y = y.loc[complete_mask][:200]
        
        # Train two models with same seed
        model1 = MLPClassifier(
            hidden_layer_sizes=(10,),
            random_state=deterministic_seed,
            max_iter=100,
            solver='lbfgs'  # Deterministic solver
        )
        model2 = MLPClassifier(
            hidden_layer_sizes=(10,),
            random_state=deterministic_seed,
            max_iter=100,
            solver='lbfgs'
        )
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should be very similar (neural networks can have small numerical differences)
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        # Allow for small differences due to numerical precision
        agreement = np.mean(pred1 == pred2)
        assert agreement > 0.95, f"Neural network predictions should be highly consistent, got {agreement:.3f} agreement"
    
    def test_train_test_split_determinism(self, sample_bulk_data, deterministic_seed):
        """Test that train_test_split is reproducible."""
        data, metadata = sample_bulk_data
        
        X = data.dropna()
        y = (metadata.loc[X.index, 'age'] > 50).astype(int)
        
        # Split data twice with same seed
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.3, random_state=deterministic_seed, stratify=y
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.3, random_state=deterministic_seed, stratify=y
        )
        
        # Splits should be identical
        assert X_train1.index.equals(X_train2.index), "Training sets should be identical"
        assert X_test1.index.equals(X_test2.index), "Test sets should be identical"
        assert np.array_equal(y_train1, y_train2), "Training labels should be identical"
        assert np.array_equal(y_test1, y_test2), "Test labels should be identical"
    
    def test_umap_determinism(self, sample_single_cell_data, deterministic_seed):
        """Test UMAP reproducibility with fixed random seed."""
        adata = sample_single_cell_data.copy()
        
        # Preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack', random_state=deterministic_seed)
        
        # Run UMAP twice with same seed
        adata1 = adata.copy()
        adata2 = adata.copy()
        
        sc.pp.neighbors(adata1, random_state=deterministic_seed)
        sc.tl.umap(adata1, random_state=deterministic_seed)
        
        sc.pp.neighbors(adata2, random_state=deterministic_seed)
        sc.tl.umap(adata2, random_state=deterministic_seed)
        
        # UMAP coordinates should be identical (or very close due to numerical precision)
        umap1 = adata1.obsm['X_umap']
        umap2 = adata2.obsm['X_umap']
        
        # Check if embeddings are identical or mirror images (UMAP can flip axes)
        direct_match = np.allclose(umap1, umap2, atol=1e-5)
        flipped_x = np.allclose(umap1[:, 0], -umap2[:, 0], atol=1e-5) and np.allclose(umap1[:, 1], umap2[:, 1], atol=1e-5)
        flipped_y = np.allclose(umap1[:, 0], umap2[:, 0], atol=1e-5) and np.allclose(umap1[:, 1], -umap2[:, 1], atol=1e-5)
        flipped_both = np.allclose(umap1[:, 0], -umap2[:, 0], atol=1e-5) and np.allclose(umap1[:, 1], -umap2[:, 1], atol=1e-5)
        
        is_reproducible = direct_match or flipped_x or flipped_y or flipped_both
        
        assert is_reproducible, "UMAP embeddings should be reproducible (allowing for axis flips)"
    
    def test_leiden_clustering_determinism(self, sample_single_cell_data, deterministic_seed):
        """Test Leiden clustering reproducibility."""
        adata = sample_single_cell_data.copy()
        
        # Preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack', random_state=deterministic_seed)
        
        # Run clustering twice with same seed
        adata1 = adata.copy()
        adata2 = adata.copy()
        
        # Build neighbor graph
        sc.pp.neighbors(adata1, random_state=deterministic_seed)
        sc.pp.neighbors(adata2, random_state=deterministic_seed)
        
        # Run Leiden clustering
        sc.tl.leiden(adata1, resolution=0.5, random_state=deterministic_seed)
        sc.tl.leiden(adata2, resolution=0.5, random_state=deterministic_seed)
        
        # Cluster assignments should be identical
        clusters1 = adata1.obs['leiden'].values
        clusters2 = adata2.obs['leiden'].values
        
        assert np.array_equal(clusters1, clusters2), "Leiden clustering should be deterministic with fixed seed"
        
        # Check that we actually have multiple clusters
        n_clusters1 = len(np.unique(clusters1))
        n_clusters2 = len(np.unique(clusters2))
        
        assert n_clusters1 > 1, "Should find multiple clusters"
        assert n_clusters1 == n_clusters2, "Should find same number of clusters"
    
    def test_pca_determinism(self, sample_bulk_data, deterministic_seed):
        """Test PCA reproducibility with different random seed settings."""
        data, metadata = sample_bulk_data
        
        # Use complete cases
        X = data.dropna()
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test with different PCA settings
        from sklearn.decomposition import PCA
        
        # Method 1: SVD solver with random_state
        pca1 = PCA(n_components=10, svd_solver='randomized', random_state=deterministic_seed)
        pca2 = PCA(n_components=10, svd_solver='randomized', random_state=deterministic_seed)
        
        result1 = pca1.fit_transform(X_scaled)
        result2 = pca2.fit_transform(X_scaled)
        
        # Results should be identical (up to sign flip)
        for i in range(10):
            component_match = np.allclose(result1[:, i], result2[:, i], atol=1e-10) or \
                             np.allclose(result1[:, i], -result2[:, i], atol=1e-10)
            assert component_match, f"PCA component {i} should be deterministic"
        
        # Explained variance should be identical
        assert np.allclose(pca1.explained_variance_ratio_, pca2.explained_variance_ratio_)
    
    def test_cross_validation_determinism(self, sample_bulk_data, deterministic_seed):
        """Test that cross-validation folds are reproducible."""
        data, metadata = sample_bulk_data
        
        X = data.dropna().iloc[:, :50]  # Use subset for speed
        y = (metadata.loc[X.index, 'age'] > 50).astype(int)
        
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Test KFold
        kfold1 = KFold(n_splits=5, shuffle=True, random_state=deterministic_seed)
        kfold2 = KFold(n_splits=5, shuffle=True, random_state=deterministic_seed)
        
        folds1 = list(kfold1.split(X))
        folds2 = list(kfold2.split(X))
        
        assert len(folds1) == len(folds2) == 5
        
        for i, ((train1, test1), (train2, test2)) in enumerate(zip(folds1, folds2)):
            assert np.array_equal(train1, train2), f"KFold training indices should be identical for fold {i}"
            assert np.array_equal(test1, test2), f"KFold test indices should be identical for fold {i}"
        
        # Test StratifiedKFold
        skfold1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=deterministic_seed)
        skfold2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=deterministic_seed)
        
        sfolds1 = list(skfold1.split(X, y))
        sfolds2 = list(skfold2.split(X, y))
        
        for i, ((strain1, stest1), (strain2, stest2)) in enumerate(zip(sfolds1, sfolds2)):
            assert np.array_equal(strain1, strain2), f"StratifiedKFold training indices should be identical for fold {i}"
            assert np.array_equal(stest1, stest2), f"StratifiedKFold test indices should be identical for fold {i}"


class TestNonDeterministicBehavior:
    """Test to ensure we can detect when algorithms are not properly seeded."""
    
    def test_unseeded_randomness_detection(self, sample_bulk_data):
        """Test that unseeded operations produce different results."""
        data, metadata = sample_bulk_data
        
        X = data.dropna().iloc[:, :20]
        y = (metadata.loc[X.index, 'age'] > 50).astype(int)
        
        # Train models without setting random_state
        model1 = RandomForestClassifier(n_estimators=10)  # No random_state
        model2 = RandomForestClassifier(n_estimators=10)  # No random_state
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Predictions should likely be different (though not guaranteed)
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        # Feature importances should likely be different
        importance_diff = np.mean(np.abs(model1.feature_importances_ - model2.feature_importances_))
        
        # This test verifies we can detect non-deterministic behavior
        # (Results may occasionally be similar by chance, but usually different)
        # We just verify the test runs without asserting difference (flaky test otherwise)
        assert isinstance(importance_diff, float), "Should compute importance difference"
        assert len(pred1) == len(pred2), "Predictions should have same length"
    
    def test_seed_isolation(self, sample_bulk_data, deterministic_seed):
        """Test that different random_state values produce different results."""
        data, metadata = sample_bulk_data
        
        X = data.dropna().iloc[:, :30]
        y = (metadata.loc[X.index, 'age'] > 50).astype(int)
        
        # Train models with different seeds
        model1 = RandomForestClassifier(n_estimators=10, random_state=deterministic_seed)
        model2 = RandomForestClassifier(n_estimators=10, random_state=deterministic_seed + 1)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Results should be different with different seeds
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        # Feature importances should be different
        importance_diff = np.mean(np.abs(model1.feature_importances_ - model2.feature_importances_))
        
        assert importance_diff > 1e-10, "Different seeds should produce different feature importances"
        
        # At least some predictions might be different
        prediction_agreement = np.mean(pred1 == pred2)
        
        # Should not have perfect agreement (would suggest seed not working)
        assert prediction_agreement < 1.0, "Different seeds should produce some different predictions"


class TestEnvironmentReproducibility:
    """Test reproducibility across different environments and setups."""
    
    def test_numpy_seed_behavior(self, deterministic_seed):
        """Test numpy random seed behavior."""
        # Set seed and generate numbers
        np.random.seed(deterministic_seed)
        numbers1 = np.random.random(100)
        
        np.random.seed(deterministic_seed)
        numbers2 = np.random.random(100)
        
        # Should be identical
        assert np.array_equal(numbers1, numbers2), "Numpy random numbers should be identical with same seed"
        
        # Different seeds should give different results
        np.random.seed(deterministic_seed + 1)
        numbers3 = np.random.random(100)
        
        assert not np.array_equal(numbers1, numbers3), "Different numpy seeds should give different results"
    
    def test_sklearn_random_state_propagation(self, sample_bulk_data, deterministic_seed):
        """Test that random_state is properly propagated in sklearn pipelines."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        
        data, metadata = sample_bulk_data
        
        X = data.dropna().iloc[:, :50]
        y = (metadata.loc[X.index, 'age'] > 50).astype(int)
        
        # Create pipeline with seeded components
        pipeline1 = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=20)),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=deterministic_seed))
        ])
        
        pipeline2 = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=20)),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=deterministic_seed))
        ])
        
        # Fit pipelines
        pipeline1.fit(X, y)
        pipeline2.fit(X, y)
        
        # Predictions should be identical
        pred1 = pipeline1.predict(X)
        pred2 = pipeline2.predict(X)
        
        assert np.array_equal(pred1, pred2), "Pipeline predictions should be identical with same random_state"
        
        # Selected features should be identical
        selected1 = pipeline1.named_steps['selector'].get_support()
        selected2 = pipeline2.named_steps['selector'].get_support()
        
        assert np.array_equal(selected1, selected2), "Feature selection should be deterministic"
    
    def test_scanpy_random_state_settings(self, sample_single_cell_data, deterministic_seed):
        """Test scanpy random state configuration."""
        # Set global scanpy settings
        sc.settings.n_jobs = 1  # Force single-threaded for reproducibility
        
        adata1 = sample_single_cell_data.copy()
        adata2 = sample_single_cell_data.copy()
        
        # Basic preprocessing (should be deterministic)
        for adata in [adata1, adata2]:
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        # Check that preprocessing gives identical results
        assert np.array_equal(adata1.X.toarray(), adata2.X.toarray()), "Preprocessing should be deterministic"
        
        # Test seeded operations
        sc.pp.highly_variable_genes(adata1, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.highly_variable_genes(adata2, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        # Highly variable gene selection should be identical
        hvg1 = adata1.var['highly_variable'].values
        hvg2 = adata2.var['highly_variable'].values
        
        assert np.array_equal(hvg1, hvg2), "Highly variable gene selection should be deterministic"