"""
TIER 1 Multi-Omics Late Fusion Implementation
===========================================

Reference implementation for multi-omics integration using late-fusion stacking
approach with cross-validation to prevent overfitting.

Based on:
- Ritchie et al. (2015) Nature Methods: limma powers differential expression analyses
- Argelaguet et al. (2018) Molecular Systems Biology: Multi-Omics Factor Analysis
- Subramanian et al. (2017) Nature Methods: Multi-omics data integration
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')

class LateFusionMultiOmics(BaseEstimator, RegressorMixin):
    """
    Late fusion multi-omics integration for rejuvenation scoring.
    
    Train separate models for each omics modality, then stack predictions
    using a meta-learner to avoid overfitting.
    """
    
    def __init__(self, 
                 base_model_type='random_forest',
                 meta_learner='elastic_net',
                 cv_folds=5,
                 random_state=42,
                 n_estimators=100):
        """
        Parameters:
        -----------
        base_model_type : str
            Type of base model ('random_forest', 'elastic_net')
        meta_learner : str
            Meta-learner type ('elastic_net', 'random_forest')
        cv_folds : int
            Number of cross-validation folds for stacking
        random_state : int
            Random seed for reproducibility
        n_estimators : int
            Number of trees for random forest models
        """
        self.base_model_type = base_model_type
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_estimators = n_estimators
        
        # Initialize models
        self.base_models = {}
        self.meta_model = None
        self.scalers = {}
        self.modality_names = []
        
    def _create_base_model(self):
        """Create base model based on specified type"""
        if self.base_model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.base_model_type == 'elastic_net':
            return ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown base model type: {self.base_model_type}")
    
    def _create_meta_learner(self):
        """Create meta-learner model"""
        if self.meta_learner == 'elastic_net':
            return ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=self.random_state,
                max_iter=1000
            )
        elif self.meta_learner == 'random_forest':
            return RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner}")
    
    def fit(self, X_dict, y):
        """
        Fit multi-omics late fusion model.
        
        Parameters:
        -----------
        X_dict : dict
            Dictionary with modality names as keys and feature matrices as values
            e.g., {'transcriptomics': X_rna, 'proteomics': X_prot, 'metabolomics': X_metab}
        y : array-like
            Target values (rejuvenation scores)
        """
        self.modality_names = list(X_dict.keys())
        n_samples = len(y)
        
        # Validate input
        for modality, X in X_dict.items():
            if len(X) != n_samples:
                raise ValueError(f"Modality {modality} has {len(X)} samples, expected {n_samples}")
        
        # Initialize base models and scalers
        for modality in self.modality_names:
            self.base_models[modality] = self._create_base_model()
            self.scalers[modality] = StandardScaler()
        
        # Generate out-of-fold predictions for stacking
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Create stratified folds based on binned target values
        y_binned = pd.qcut(y, q=min(5, len(np.unique(y))), duplicates='drop', labels=False)
        
        stacked_predictions = []
        base_model_scores = {}
        
        for modality in self.modality_names:
            X = X_dict[modality]
            
            # Scale features
            X_scaled = self.scalers[modality].fit_transform(X)
            
            # Generate out-of-fold predictions
            oof_preds = cross_val_predict(
                self.base_models[modality], 
                X_scaled, 
                y, 
                cv=cv,
                method='predict'
            )
            
            stacked_predictions.append(oof_preds)
            
            # Calculate base model performance
            base_score = r2_score(y, oof_preds)
            base_model_scores[modality] = base_score
            
            # Fit final base model on full data
            self.base_models[modality].fit(X_scaled, y)
        
        # Stack predictions
        stacked_features = np.column_stack(stacked_predictions)
        
        # Fit meta-learner
        self.meta_model = self._create_meta_learner()
        self.meta_model.fit(stacked_features, y)
        
        # Store performance metrics
        self.base_model_scores_ = base_model_scores
        self.stacked_score_ = r2_score(y, self.meta_model.predict(stacked_features))
        
        return self
    
    def predict(self, X_dict):
        """
        Make predictions using trained multi-omics model.
        
        Parameters:
        -----------
        X_dict : dict
            Dictionary with modality names as keys and feature matrices as values
        
        Returns:
        --------
        predictions : array
            Predicted rejuvenation scores
        """
        if self.meta_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Generate base model predictions
        base_predictions = []
        
        for modality in self.modality_names:
            X = X_dict[modality]
            X_scaled = self.scalers[modality].transform(X)
            pred = self.base_models[modality].predict(X_scaled)
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_features = np.column_stack(base_predictions)
        
        # Meta-learner prediction
        return self.meta_model.predict(stacked_features)
    
    def get_feature_importance(self):
        """
        Get feature importance for each modality.
        
        Returns:
        --------
        importance_dict : dict
            Dictionary with modality importance scores
        """
        if self.meta_model is None:
            raise ValueError("Model must be fitted before getting importance")
        
        # Get meta-learner coefficients or feature importance
        if hasattr(self.meta_model, 'coef_'):
            meta_importance = np.abs(self.meta_model.coef_)
        elif hasattr(self.meta_model, 'feature_importances_'):
            meta_importance = self.meta_model.feature_importances_
        else:
            meta_importance = np.ones(len(self.modality_names)) / len(self.modality_names)
        
        importance_dict = {}
        for i, modality in enumerate(self.modality_names):
            base_score = self.base_model_scores_.get(modality, 0)
            meta_weight = meta_importance[i] if i < len(meta_importance) else 0
            
            # Combined importance score
            importance_dict[modality] = {
                'base_r2': base_score,
                'meta_weight': meta_weight,
                'combined_score': base_score * meta_weight
            }
        
        return importance_dict
    
    def ablation_study(self, X_dict, y, cv=5):
        """
        Perform ablation study by dropping each modality.
        
        Parameters:
        -----------
        X_dict : dict
            Full multi-omics data dictionary
        y : array-like
            Target values
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        ablation_results : dict
            Performance with each modality removed
        """
        results = {}
        
        # Full model performance
        full_cv_scores = []
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        y_binned = pd.qcut(y, q=min(5, len(np.unique(y))), duplicates='drop', labels=False)
        
        for train_idx, test_idx in cv_splitter.split(list(range(len(y))), y_binned):
            # Create train/test splits for all modalities
            X_train = {mod: X[train_idx] for mod, X in X_dict.items()}
            X_test = {mod: X[test_idx] for mod, X in X_dict.items()}
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit and predict
            model = LateFusionMultiOmics(
                base_model_type=self.base_model_type,
                meta_learner=self.meta_learner,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            full_cv_scores.append(r2_score(y_test, pred))
        
        results['full_model'] = {
            'mean_r2': np.mean(full_cv_scores),
            'std_r2': np.std(full_cv_scores),
            'modalities': list(X_dict.keys())
        }
        
        # Ablation: remove each modality
        for remove_modality in self.modality_names:
            ablated_X = {mod: X for mod, X in X_dict.items() if mod != remove_modality}
            
            ablated_scores = []
            for train_idx, test_idx in cv_splitter.split(list(range(len(y))), y_binned):
                X_train = {mod: X[train_idx] for mod, X in ablated_X.items()}
                X_test = {mod: X[test_idx] for mod, X in ablated_X.items()}
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = LateFusionMultiOmics(
                    base_model_type=self.base_model_type,
                    meta_learner=self.meta_learner,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                ablated_scores.append(r2_score(y_test, pred))
            
            results[f'without_{remove_modality}'] = {
                'mean_r2': np.mean(ablated_scores),
                'std_r2': np.std(ablated_scores),
                'modalities': [mod for mod in X_dict.keys() if mod != remove_modality],
                'performance_drop': np.mean(full_cv_scores) - np.mean(ablated_scores)
            }
        
        return results

def create_synthetic_multiomics_data(n_samples=200, n_genes=1000, n_proteins=500, n_metabolites=200, random_state=42):
    """
    Create synthetic multi-omics data for testing.
    
    Returns:
    --------
    X_dict : dict
        Multi-omics data dictionary
    y : array
        Synthetic rejuvenation scores
    age : array
        Age values for validation
    """
    np.random.seed(random_state)
    
    # Generate age (20-80 years)
    age = np.random.uniform(20, 80, n_samples)
    
    # Create aging signal (non-linear)
    aging_signal = 1 / (1 + np.exp(-0.1 * (age - 45)))
    
    # Generate correlated omics data
    # Transcriptomics: some genes correlated with aging
    X_rna = np.random.normal(0, 1, (n_samples, n_genes))
    for i in range(min(50, n_genes)):  # First 50 genes correlated with aging
        X_rna[:, i] += aging_signal * np.random.uniform(0.5, 2.0)
    
    # Proteomics: fewer features, higher correlation with aging
    X_prot = np.random.normal(0, 1, (n_samples, n_proteins))
    for i in range(min(25, n_proteins)):
        X_prot[:, i] += aging_signal * np.random.uniform(0.3, 1.5)
    
    # Metabolomics: metabolites with different aging associations
    X_metab = np.random.normal(0, 1, (n_samples, n_metabolites))
    for i in range(min(15, n_metabolites)):
        X_metab[:, i] += aging_signal * np.random.uniform(0.2, 1.0)
    
    # Create rejuvenation score (inverse of aging + noise)
    y = 1 - aging_signal + np.random.normal(0, 0.1, n_samples)
    y = np.clip(y, 0, 1)  # Ensure [0, 1] range
    
    X_dict = {
        'transcriptomics': X_rna,
        'proteomics': X_prot,
        'metabolomics': X_metab
    }
    
    return X_dict, y, age

# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data
    X_dict, y, age = create_synthetic_multiomics_data()
    
    print("Multi-Omics Late Fusion Testing")
    print("=" * 40)
    print(f"Samples: {len(y)}")
    print(f"Modalities: {list(X_dict.keys())}")
    print(f"Features per modality: {[X.shape[1] for X in X_dict.values()]}")
    
    # Fit model
    model = LateFusionMultiOmics(base_model_type='random_forest', meta_learner='elastic_net')
    model.fit(X_dict, y)
    
    # Make predictions
    predictions = model.predict(X_dict)
    
    print(f"\nModel Performance:")
    print(f"Stacked R²: {model.stacked_score_:.3f}")
    print(f"Overall R²: {r2_score(y, predictions):.3f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nModality Importance:")
    for modality, scores in importance.items():
        print(f"  {modality}:")
        print(f"    Base R²: {scores['base_r2']:.3f}")
        print(f"    Meta weight: {scores['meta_weight']:.3f}")
        print(f"    Combined: {scores['combined_score']:.3f}")
    
    # Ablation study
    print(f"\nAblation Study:")
    ablation = model.ablation_study(X_dict, y)
    for condition, results in ablation.items():
        print(f"  {condition}: R² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
        if 'performance_drop' in results:
            print(f"    Performance drop: {results['performance_drop']:.3f}")