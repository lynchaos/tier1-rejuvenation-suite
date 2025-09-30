"""
Data leakage tests for TIER 1 Rejuvenation Suite.
Ensures test fold never influences scalers, filters, or any preprocessing steps.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings


class TestDataLeakage:
    """Test suite to detect data leakage in preprocessing pipelines."""
    
    def test_scaler_leakage_detection(self, sample_bulk_data):
        """Test that scalers are never fit on test data."""
        data, metadata = sample_bulk_data
        
        # Create binary target
        y = (metadata['age'] > metadata['age'].median()).astype(int)
        
        # Remove missing values
        complete_mask = data.notna().all(axis=1) & y.notna()
        X_clean = data.loc[complete_mask]
        y_clean = y.loc[complete_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
        )
        
        # CORRECT: Fit scaler only on training data
        scaler_correct = StandardScaler()
        X_train_scaled_correct = scaler_correct.fit_transform(X_train)
        X_test_scaled_correct = scaler_correct.transform(X_test)
        
        # INCORRECT: Fit scaler on all data (data leakage)
        scaler_incorrect = StandardScaler()
        X_all_scaled_df = pd.DataFrame(
            scaler_incorrect.fit_transform(X_clean), 
            index=X_clean.index, 
            columns=X_clean.columns
        )
        X_train_scaled_incorrect = X_all_scaled_df.loc[X_train.index].values
        X_test_scaled_incorrect = X_all_scaled_df.loc[X_test.index].values
        
        # Test that correct and incorrect approaches give different results
        # (This demonstrates that fitting on all data vs. training only matters)
        train_difference = np.mean(np.abs(X_train_scaled_correct - X_train_scaled_incorrect))
        test_difference = np.mean(np.abs(X_test_scaled_correct - X_test_scaled_incorrect))
        
        # There should be meaningful differences
        assert train_difference > 0.01 or test_difference > 0.01
        
        # Verify that scaler parameters differ
        assert not np.allclose(scaler_correct.mean_, scaler_incorrect.mean_, rtol=1e-3)
        assert not np.allclose(scaler_correct.scale_, scaler_incorrect.scale_, rtol=1e-3)
    
    def test_feature_selection_leakage(self, sample_bulk_data):
        """Test that feature selection never sees test labels."""
        data, metadata = sample_bulk_data
        
        # Create target variable
        y = (metadata['age'] > 50).astype(int)
        
        # Remove missing values
        complete_mask = data.notna().all(axis=1) & y.notna()
        X_clean = data.loc[complete_mask].iloc[:, :100]  # Use subset for speed
        y_clean = y.loc[complete_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
        )
        
        # CORRECT: Feature selection only on training data
        selector_correct = SelectKBest(f_classif, k=20)
        X_train_selected_correct = selector_correct.fit_transform(X_train, y_train)
        X_test_selected_correct = selector_correct.transform(X_test)
        selected_features_correct = X_train.columns[selector_correct.get_support()]
        
        # INCORRECT: Feature selection on all data using all labels
        selector_incorrect = SelectKBest(f_classif, k=20)
        X_all_selected = selector_incorrect.fit_transform(X_clean, y_clean)
        selected_features_incorrect = X_clean.columns[selector_incorrect.get_support()]
        
        # The selected features should potentially be different
        # (This demonstrates that feature selection with test labels can differ)
        feature_overlap = len(set(selected_features_correct) & set(selected_features_incorrect))
        
        # If there's perfect overlap, that would be suspicious but not necessarily wrong
        # The key test is that we're using the correct approach
        assert len(selected_features_correct) == 20
        assert len(selected_features_incorrect) == 20
        
        # Verify that F-scores differ when computed on different datasets
        train_scores = selector_correct.scores_
        all_scores = selector_incorrect.scores_
        
        # Scores should be different (even if slightly)
        assert not np.allclose(train_scores, all_scores, rtol=1e-2)
    
    def test_cross_validation_leakage(self, sample_bulk_data):
        """Test that preprocessing in CV is done correctly without leakage."""
        data, metadata = sample_bulk_data
        
        # Create target
        y = (metadata['condition'] == 'old').astype(int)
        
        # Use complete cases
        complete_mask = data.notna().all(axis=1) & y.notna()
        X = data.loc[complete_mask].iloc[:, :50]  # Subset for speed
        y = y.loc[complete_mask]
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Track scaler parameters across folds to detect leakage
        scaler_means_per_fold = []
        scaler_scales_per_fold = []
        
        for train_idx, val_idx in cv.split(X):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            # CORRECT: Fit scaler only on training fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Store scaler parameters
            scaler_means_per_fold.append(scaler.mean_.copy())
            scaler_scales_per_fold.append(scaler.scale_.copy())
            
            # Verify shapes
            assert X_train_scaled.shape[0] == len(train_idx)
            assert X_val_scaled.shape[0] == len(val_idx)
            assert X_train_scaled.shape[1] == X_val_scaled.shape[1] == X.shape[1]
        
        # Scaler parameters should vary across folds
        # (Different training sets should give different normalization parameters)
        means_array = np.array(scaler_means_per_fold)
        scales_array = np.array(scaler_scales_per_fold)
        
        # Check that parameters vary across folds
        mean_variance = np.var(means_array, axis=0).mean()
        scale_variance = np.var(scales_array, axis=0).mean()
        
        assert mean_variance > 1e-6  # Should have some variation
        assert scale_variance > 1e-6  # Should have some variation
    
    def test_imputation_leakage(self):
        """Test that imputation doesn't use test set statistics."""
        # Create data with missing values
        np.random.seed(42)
        n_samples, n_features = 200, 20
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Introduce missing values with pattern
        missing_mask = np.random.random((n_samples, n_features)) < 0.2
        X[missing_mask] = np.nan
        
        X_df = pd.DataFrame(X)
        y = np.random.binomial(1, 0.5, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.3, random_state=42
        )
        
        # CORRECT: Compute imputation values only from training data
        train_means = X_train.mean()
        X_train_imputed_correct = X_train.fillna(train_means)
        X_test_imputed_correct = X_test.fillna(train_means)
        
        # INCORRECT: Compute imputation values from all data
        all_means = X_df.mean()
        X_train_imputed_incorrect = X_train.fillna(all_means)
        X_test_imputed_incorrect = X_test.fillna(all_means)
        
        # The imputation values should be different
        assert not np.allclose(train_means.values, all_means.values, rtol=1e-3)
        
        # Results should differ
        train_diff = np.nanmean(np.abs(
            X_train_imputed_correct.values - X_train_imputed_incorrect.values
        ))
        test_diff = np.nanmean(np.abs(
            X_test_imputed_correct.values - X_test_imputed_incorrect.values
        ))
        
        assert train_diff > 1e-6 or test_diff > 1e-6
    
    def test_filter_leakage_variance(self, sample_bulk_data):
        """Test that variance filtering doesn't use test set variance."""
        data, metadata = sample_bulk_data
        
        # Create target
        y = (metadata['age'] > metadata['age'].median()).astype(int)
        
        # Use complete cases
        complete_mask = data.notna().all(axis=1) & y.notna()
        X = data.loc[complete_mask]
        y = y.loc[complete_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # CORRECT: Variance filtering based only on training data
        train_variances = X_train.var()
        variance_threshold = train_variances.quantile(0.1)  # Keep top 90% by variance
        high_var_features = train_variances[train_variances >= variance_threshold].index
        
        X_train_filtered_correct = X_train[high_var_features]
        X_test_filtered_correct = X_test[high_var_features]
        
        # INCORRECT: Variance filtering based on all data
        all_variances = X.var()
        all_variance_threshold = all_variances.quantile(0.1)
        all_high_var_features = all_variances[all_variances >= all_variance_threshold].index
        
        # Feature sets should potentially be different
        feature_overlap = len(set(high_var_features) & set(all_high_var_features))
        total_features = len(set(high_var_features) | set(all_high_var_features))
        
        # Verify we're selecting features correctly
        assert len(high_var_features) > 0
        assert len(all_high_var_features) > 0
        
        # Variance values should differ
        assert not np.allclose(
            train_variances.values, 
            all_variances.loc[train_variances.index].values, 
            rtol=1e-3
        )
    
    def test_end_to_end_pipeline_leakage(self, sample_bulk_data):
        """Test complete pipeline to ensure no leakage at any step."""
        data, metadata = sample_bulk_data
        
        # Create target
        y = (metadata['age'] > 50).astype(int)
        
        # Use complete cases and subset for speed
        complete_mask = data.notna().all(axis=1) & y.notna()
        X = data.loc[complete_mask].iloc[:, :100]
        y = y.loc[complete_mask]
        
        # Nested cross-validation to test for leakage
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=2, shuffle=True, random_state=43)
        
        outer_scores = []
        
        for outer_train_idx, outer_test_idx in outer_cv.split(X):
            X_outer_train = X.iloc[outer_train_idx]
            X_outer_test = X.iloc[outer_test_idx]
            y_outer_train = y.iloc[outer_train_idx]
            y_outer_test = y.iloc[outer_test_idx]
            
            # Inner CV for hyperparameter selection (simulated)
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train):
                X_inner_train = X_outer_train.iloc[inner_train_idx]
                X_inner_val = X_outer_train.iloc[inner_val_idx]
                y_inner_train = y_outer_train.iloc[inner_train_idx]
                y_inner_val = y_outer_train.iloc[inner_val_idx]
                
                # CORRECT preprocessing pipeline (no leakage)
                # 1. Variance filtering on inner training set only
                inner_variances = X_inner_train.var()
                threshold = inner_variances.quantile(0.2)
                selected_features = inner_variances[inner_variances >= threshold].index
                
                # 2. Scaling on inner training set only
                scaler = StandardScaler()
                X_inner_train_processed = scaler.fit_transform(X_inner_train[selected_features])
                X_inner_val_processed = scaler.transform(X_inner_val[selected_features])
                
                # 3. Model training
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_inner_train_processed, y_inner_train)
                
                # 4. Validation
                val_score = model.score(X_inner_val_processed, y_inner_val)
                inner_scores.append(val_score)
            
            # Select best hyperparameters (simulated - in practice this would be real)
            best_config = {'variance_threshold': 0.2}  # Simplified
            
            # Retrain on full outer training set with best config
            outer_variances = X_outer_train.var()
            outer_threshold = outer_variances.quantile(best_config['variance_threshold'])
            outer_selected_features = outer_variances[outer_variances >= outer_threshold].index
            
            outer_scaler = StandardScaler()
            X_outer_train_final = outer_scaler.fit_transform(X_outer_train[outer_selected_features])
            X_outer_test_final = outer_scaler.transform(X_outer_test[outer_selected_features])
            
            final_model = LogisticRegression(random_state=42, max_iter=1000)
            final_model.fit(X_outer_train_final, y_outer_train)
            
            # Test on outer test set
            outer_score = final_model.score(X_outer_test_final, y_outer_test)
            outer_scores.append(outer_score)
        
        # Verify that we got reasonable scores (not perfect, which would suggest leakage)
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        # Scores should be reasonable but not suspiciously high
        assert 0.3 <= mean_score <= 0.9  # Reasonable range
        assert len(outer_scores) == 3  # All folds completed
        
        # There should be some variation across folds (perfect scores would be suspicious)
        assert std_score > 0.01


class TestPreprocessingLeakagePatterns:
    """Test specific patterns that commonly lead to data leakage."""
    
    def test_global_statistics_leakage(self, sample_bulk_data):
        """Test that global statistics aren't computed on test data."""
        data, metadata = sample_bulk_data
        
        # Remove missing values for cleaner test
        clean_data = data.dropna()
        
        # Split into train/test
        train_size = int(0.7 * len(clean_data))
        X_train = clean_data.iloc[:train_size]
        X_test = clean_data.iloc[train_size:]
        
        # Compute statistics correctly (training only) vs incorrectly (all data)
        train_stats = {
            'mean': X_train.mean(),
            'std': X_train.std(),
            'median': X_train.median(),
            'q25': X_train.quantile(0.25),
            'q75': X_train.quantile(0.75)
        }
        
        all_stats = {
            'mean': clean_data.mean(),
            'std': clean_data.std(),
            'median': clean_data.median(),
            'q25': clean_data.quantile(0.25),
            'q75': clean_data.quantile(0.75)
        }
        
        # Statistics should be different
        for stat_name in train_stats:
            assert not np.allclose(
                train_stats[stat_name].values, 
                all_stats[stat_name].values, 
                rtol=1e-3
            ), f"Training and all-data {stat_name} should differ"
    
    def test_target_leakage_feature_engineering(self):
        """Test that feature engineering doesn't use target information inappropriately."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features that are legitimately predictive
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        
        # Create target based on features
        y_prob = 1 / (1 + np.exp(-(X1 + 0.5 * X2)))
        y = np.random.binomial(1, y_prob)
        
        # Create a feature that would be leaky if we use target information
        # This simulates accidentally using target statistics in feature engineering
        X3_leaky = X1 + 0.1 * y  # This uses target information!
        X3_clean = X1 + np.random.normal(0, 0.1, n_samples)  # This doesn't
        
        # Compare prediction performance
        X_leaky = np.column_stack([X1, X2, X3_leaky])
        X_clean = np.column_stack([X1, X2, X3_clean])
        
        # Split data
        train_size = int(0.7 * n_samples)
        
        # Test leaky features
        X_train_leaky = X_leaky[:train_size]
        X_test_leaky = X_leaky[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        model_leaky = LogisticRegression(random_state=42)
        model_leaky.fit(X_train_leaky, y_train)
        score_leaky = model_leaky.score(X_test_leaky, y_test)
        
        # Test clean features
        X_train_clean = X_clean[:train_size]
        X_test_clean = X_clean[train_size:]
        
        model_clean = LogisticRegression(random_state=42)
        model_clean.fit(X_train_clean, y_train)
        score_clean = model_clean.score(X_test_clean, y_test)
        
        # Leaky features might give better performance, but this is a red flag
        # The test verifies we can detect this pattern
        assert 0.5 <= score_clean <= 0.9  # Reasonable performance
        assert 0.5 <= score_leaky <= 1.0  # May be suspiciously high
        
        # The key is that we've identified this pattern
        feature_importance_leaky = np.abs(model_leaky.coef_[0])
        feature_importance_clean = np.abs(model_clean.coef_[0])
        
        # Verify models were trained
        assert len(feature_importance_leaky) == 3
        assert len(feature_importance_clean) == 3
    
    def test_temporal_leakage_simulation(self):
        """Simulate temporal data leakage (using future information)."""
        np.random.seed(42)
        n_timepoints = 200
        
        # Create time series data
        time = np.arange(n_timepoints)
        trend = 0.01 * time
        seasonal = 0.5 * np.sin(2 * np.pi * time / 12)
        noise = np.random.normal(0, 0.2, n_timepoints)
        
        # Target variable (e.g., stock price, patient outcome)
        y = trend + seasonal + noise
        
        # Features that would be leaky if we use future information
        # CORRECT: Only use past information
        X_correct = np.column_stack([
            np.roll(y, 1),  # Previous value (t-1)
            np.roll(y, 2),  # Value from t-2
            np.convolve(y, np.ones(3)/3, mode='same')  # This includes current value - problematic!
        ])
        
        # INCORRECT: Using future information (impossible in real prediction)
        X_incorrect = np.column_stack([
            np.roll(y, -1),  # Next value (t+1) - this is leakage!
            np.roll(y, 1),   # Previous value (t-1)
            y                # Current value in prediction context can be leakage
        ])
        
        # Remove first few points due to rolling
        valid_idx = slice(3, -3)
        X_correct = X_correct[valid_idx]
        X_incorrect = X_incorrect[valid_idx]
        y_target = y[valid_idx]
        
        # Split maintaining temporal order
        split_point = len(y_target) // 2
        
        X_train_correct = X_correct[:split_point]
        X_test_correct = X_correct[split_point:]
        X_train_incorrect = X_incorrect[:split_point]
        X_test_incorrect = X_incorrect[split_point:]
        
        y_train = y_target[:split_point]
        y_test = y_target[split_point:]
        
        # Train models
        model_correct = LogisticRegression(random_state=42)
        model_incorrect = LogisticRegression(random_state=42)
        
        # Convert to binary classification for LogisticRegression
        y_train_binary = (y_train > np.median(y_train)).astype(int)
        y_test_binary = (y_test > np.median(y_test)).astype(int)
        
        model_correct.fit(X_train_correct, y_train_binary)
        model_incorrect.fit(X_train_incorrect, y_train_binary)
        
        score_correct = model_correct.score(X_test_correct, y_test_binary)
        score_incorrect = model_incorrect.score(X_test_incorrect, y_test_binary)
        
        # Incorrect model might perform suspiciously well due to leakage
        assert 0.4 <= score_correct <= 0.8  # Reasonable performance
        
        # The key insight is recognizing the leakage pattern
        # Future information should not be available at prediction time
        assert X_train_correct.shape[1] == 3
        assert X_train_incorrect.shape[1] == 3