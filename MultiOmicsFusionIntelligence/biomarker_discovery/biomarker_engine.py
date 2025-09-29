"""
Rejuvenation Biomarker Discovery Engine
=====================================
ML-driven identification of aging and rejuvenation biomarkers with uncertainty quantification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RejuvenationBiomarkerEngine:
    """
    Advanced ML-driven biomarker discovery for aging and rejuvenation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.biomarker_scores = {}
        
    def load_multi_omics_data(self, data_dict: Dict[str, pd.DataFrame], 
                             labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and combine multi-omics datasets
        """
        print("Loading multi-omics data for biomarker discovery...")
        
        combined_features = []
        feature_names = []
        
        for omics_type, data in data_dict.items():
            # Ensure consistent sample ordering
            common_samples = data.index.intersection(labels.index)
            data_aligned = data.loc[common_samples]
            
            combined_features.append(data_aligned.values)
            feature_names.extend([f"{omics_type}_{col}" for col in data.columns])
        
        # Combine all features
        X_combined = np.concatenate(combined_features, axis=1)
        combined_df = pd.DataFrame(X_combined, 
                                 index=common_samples, 
                                 columns=feature_names)
        
        # Align labels
        y_aligned = labels.loc[common_samples]
        
        print(f"Combined dataset: {combined_df.shape}, Classes: {y_aligned.value_counts().to_dict()}")
        return combined_df, y_aligned
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Preprocess features for biomarker discovery
        """
        print("Preprocessing features...")
        
        # Handle missing values
        X_filled = X.fillna(X.median())
        
        # Remove constant features
        constant_features = X_filled.columns[X_filled.var() == 0]
        X_cleaned = X_filled.drop(columns=constant_features)
        
        # Remove highly correlated features
        correlation_matrix = X_cleaned.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > 0.95)]
        X_cleaned = X_cleaned.drop(columns=high_corr_features)
        
        print(f"Features after preprocessing: {X_cleaned.shape[1]} (removed {len(constant_features)} constant, {len(high_corr_features)} highly correlated)")
        return X_cleaned
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                         method: str = 'mutual_info', k: int = 1000) -> pd.DataFrame:
        """
        Select top features for biomarker discovery
        """
        print(f"Selecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        X_selected_df = pd.DataFrame(X_selected, index=X.index, columns=selected_features)
        
        # Store feature scores
        feature_scores = pd.Series(selector.scores_, index=X.columns)
        self.feature_selectors[method] = {
            'selector': selector,
            'scores': feature_scores[selected_features],
            'selected_features': selected_features
        }
        
        print(f"Selected {len(selected_features)} features")
        return X_selected_df
    
    def train_biomarker_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train multiple ML models for biomarker identification
        """
        print("Training biomarker identification models...")
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state),
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'svm': SVC(probability=True, random_state=self.random_state)
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        model_performance = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='roc_auc')
            
            # Fit on full data
            model.fit(X_scaled, y_encoded)
            
            # Store results
            model_performance[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
            
            print(f"{name} - CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        self.models = model_performance
        return model_performance
    
    def identify_biomarker_panels(self, X: pd.DataFrame, y: pd.Series, 
                                panel_sizes: List[int] = [5, 10, 15, 20]) -> Dict:
        """
        Identify optimal biomarker panels of different sizes
        """
        print("Identifying optimal biomarker panels...")
        
        biomarker_panels = {}
        
        # Get feature importance from best performing model
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['mean_cv_score'])
        best_model = self.models[best_model_name]['model']
        
        # Calculate feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_scores = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importance_scores = np.abs(best_model.coef_[0])
        else:
            # Use permutation importance for SVM
            from sklearn.inspection import permutation_importance
            X_scaled = self.scalers['main'].transform(X)
            perm_importance = permutation_importance(best_model, X_scaled, y, 
                                                   random_state=self.random_state)
            importance_scores = perm_importance.importances_mean
        
        # Sort features by importance
        feature_importance = pd.Series(importance_scores, index=X.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        # Create panels of different sizes
        for panel_size in panel_sizes:
            panel_features = feature_importance.head(panel_size).index.tolist()
            
            # Evaluate panel performance
            X_panel = X[panel_features]
            X_panel_scaled = self.scalers['main'].transform(X_panel)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            panel_scores = cross_val_score(best_model, X_panel_scaled, 
                                         LabelEncoder().fit_transform(y), 
                                         cv=cv, scoring='roc_auc')
            
            biomarker_panels[f'panel_{panel_size}'] = {
                'features': panel_features,
                'importance_scores': feature_importance.head(panel_size),
                'cv_auc': panel_scores.mean(),
                'cv_auc_std': panel_scores.std()
            }
            
            print(f"Panel size {panel_size}: AUC = {panel_scores.mean():.3f} (+/- {panel_scores.std()*2:.3f})")
        
        self.biomarker_scores = biomarker_panels
        return biomarker_panels
    
    def uncertainty_quantification(self, X: pd.DataFrame, y: pd.Series, 
                                 n_bootstrap: int = 100) -> Dict:
        """
        Quantify uncertainty in biomarker selection using bootstrap sampling
        """
        print("Performing uncertainty quantification...")
        
        feature_stability = {}
        prediction_intervals = {}
        
        # Bootstrap sampling
        n_samples = len(X)
        bootstrap_features = []
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            if i % 20 == 0:
                print(f"Bootstrap iteration {i}/{n_bootstrap}")
            
            # Bootstrap sample
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[boot_indices]
            y_boot = y.iloc[boot_indices]
            
            # Feature selection
            X_boot_selected = self.feature_selection(X_boot, y_boot, k=min(500, X_boot.shape[1]))
            
            # Train model
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot_selected)
            
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            model.fit(X_boot_scaled, LabelEncoder().fit_transform(y_boot))
            
            # Store selected features
            bootstrap_features.append(set(X_boot_selected.columns))
            
            # Get predictions on original data (if features overlap)
            common_features = list(set(X_boot_selected.columns).intersection(set(X.columns)))
            if len(common_features) > 10:  # Minimum features for prediction
                X_pred = X[common_features]
                X_pred_scaled = scaler.transform(X_pred)
                pred_proba = model.predict_proba(X_pred_scaled)
                bootstrap_predictions.append(pred_proba[:, 1])  # Probability of positive class
        
        # Calculate feature stability (how often each feature is selected)
        all_features = set()
        for feature_set in bootstrap_features:
            all_features.update(feature_set)
        
        for feature in all_features:
            stability = sum(1 for feature_set in bootstrap_features if feature in feature_set)
            feature_stability[feature] = stability / n_bootstrap
        
        # Calculate prediction intervals
        if bootstrap_predictions:
            bootstrap_predictions = np.array(bootstrap_predictions)
            prediction_intervals = {
                'mean': np.mean(bootstrap_predictions, axis=0),
                'lower_95': np.percentile(bootstrap_predictions, 2.5, axis=0),
                'upper_95': np.percentile(bootstrap_predictions, 97.5, axis=0),
                'std': np.std(bootstrap_predictions, axis=0)
            }
        
        uncertainty_results = {
            'feature_stability': pd.Series(feature_stability).sort_values(ascending=False),
            'prediction_intervals': prediction_intervals,
            'n_bootstrap': n_bootstrap
        }
        
        print("Uncertainty quantification completed!")
        return uncertainty_results
    
    def generate_biomarker_report(self, X: pd.DataFrame, y: pd.Series, 
                                uncertainty_results: Dict) -> None:
        """
        Generate comprehensive biomarker discovery report
        """
        print("Generating biomarker discovery report...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Rejuvenation Biomarker Discovery Report', fontsize=16, fontweight='bold')
        
        # 1. Model performance comparison
        model_names = list(self.models.keys())
        cv_scores = [self.models[name]['mean_cv_score'] for name in model_names]
        cv_stds = [self.models[name]['std_cv_score'] for name in model_names]
        
        axes[0, 0].bar(model_names, cv_scores, yerr=cv_stds, capsize=5)
        axes[0, 0].set_title('Model Performance (Cross-validation AUC)')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Panel size vs performance
        panel_sizes = []
        panel_aucs = []
        panel_stds = []
        
        for panel_name, panel_data in self.biomarker_scores.items():
            panel_size = int(panel_name.split('_')[1])
            panel_sizes.append(panel_size)
            panel_aucs.append(panel_data['cv_auc'])
            panel_stds.append(panel_data['cv_auc_std'])
        
        axes[0, 1].errorbar(panel_sizes, panel_aucs, yerr=panel_stds, marker='o')
        axes[0, 1].set_title('Biomarker Panel Performance')
        axes[0, 1].set_xlabel('Panel Size')
        axes[0, 1].set_ylabel('AUC Score')
        
        # 3. Top biomarkers importance
        if 'panel_10' in self.biomarker_scores:
            top_biomarkers = self.biomarker_scores['panel_10']['importance_scores']
            top_biomarkers.head(10).plot(kind='barh', ax=axes[0, 2])
            axes[0, 2].set_title('Top 10 Biomarker Importance')
        
        # 4. Feature stability from uncertainty quantification
        feature_stability = uncertainty_results['feature_stability']
        feature_stability.head(15).plot(kind='barh', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Stability (Bootstrap)')
        axes[1, 0].set_xlabel('Selection Frequency')
        
        # 5. Class distribution
        y.value_counts().plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
        axes[1, 1].set_title('Sample Class Distribution')
        
        # 6. Prediction uncertainty
        if uncertainty_results['prediction_intervals']:
            pred_mean = uncertainty_results['prediction_intervals']['mean']
            pred_std = uncertainty_results['prediction_intervals']['std']
            
            axes[1, 2].hist(pred_std, bins=30, alpha=0.7)
            axes[1, 2].set_title('Prediction Uncertainty Distribution')
            axes[1, 2].set_xlabel('Prediction Standard Deviation')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('biomarker_discovery_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate text summary
        print("\n" + "="*60)
        print("BIOMARKER DISCOVERY SUMMARY")
        print("="*60)
        
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {dict(y.value_counts())}")
        
        print(f"\nBest performing model: {max(self.models.keys(), key=lambda x: self.models[x]['mean_cv_score'])}")
        print(f"Best CV AUC: {max(self.models[x]['mean_cv_score'] for x in self.models.keys()):.3f}")
        
        print(f"\nOptimal biomarker panel size: {max(panel_sizes, key=lambda x: panel_aucs[panel_sizes.index(x)])}")
        
        print(f"\nTop 5 most stable biomarkers:")
        for biomarker, stability in feature_stability.head(5).items():
            print(f"  {biomarker}: {stability:.3f}")
        
        print("\n" + "="*60)
    
    def run_biomarker_discovery_pipeline(self, data_dict: Dict[str, pd.DataFrame], 
                                       labels: pd.Series) -> Dict:
        """
        Complete biomarker discovery pipeline
        """
        print("Starting Rejuvenation Biomarker Discovery Pipeline...")
        
        # Load and combine data
        X_combined, y_aligned = self.load_multi_omics_data(data_dict, labels)
        
        # Preprocess
        X_processed = self.preprocess_features(X_combined, y_aligned)
        
        # Feature selection
        X_selected = self.feature_selection(X_processed, y_aligned)
        
        # Train models
        model_performance = self.train_biomarker_models(X_selected, y_aligned)
        
        # Identify biomarker panels
        biomarker_panels = self.identify_biomarker_panels(X_selected, y_aligned)
        
        # Uncertainty quantification
        uncertainty_results = self.uncertainty_quantification(X_selected, y_aligned)
        
        # Generate report
        self.generate_biomarker_report(X_selected, y_aligned, uncertainty_results)
        
        results = {
            'model_performance': model_performance,
            'biomarker_panels': biomarker_panels,
            'uncertainty_results': uncertainty_results,
            'processed_data': X_selected,
            'labels': y_aligned
        }
        
        print("Biomarker discovery pipeline completed!")
        return results

# Example usage with simulated data
def simulate_biomarker_data() -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """
    Simulate multi-omics data for biomarker discovery
    """
    print("Simulating multi-omics data for biomarker discovery...")
    
    n_samples = 300
    np.random.seed(42)
    
    # Create sample labels (young, aged, rejuvenated)
    labels = np.random.choice(['young', 'aged', 'rejuvenated'], size=n_samples, p=[0.4, 0.4, 0.2])
    sample_names = [f'Sample_{i}' for i in range(n_samples)]
    y = pd.Series(labels, index=sample_names)
    
    # Simulate different omics data with some signal
    data_dict = {}
    
    # Transcriptomics (with aging signal)
    transcriptomics_data = np.random.lognormal(mean=1, sigma=1, size=(n_samples, 1000))
    # Add signal for aging vs rejuvenation
    aging_signal = (labels == 'aged').astype(int) * 2 + (labels == 'rejuvenated').astype(int) * -1.5
    transcriptomics_data[:, :50] += aging_signal[:, np.newaxis]  # First 50 genes have signal
    
    data_dict['transcriptomics'] = pd.DataFrame(
        transcriptomics_data, 
        index=sample_names,
        columns=[f'Gene_{i}' for i in range(1000)]
    )
    
    # Proteomics
    proteomics_data = np.random.gamma(2, 2, size=(n_samples, 200))
    proteomics_data[:, :20] += aging_signal[:, np.newaxis] * 0.5  # Protein signal
    
    data_dict['proteomics'] = pd.DataFrame(
        proteomics_data,
        index=sample_names,
        columns=[f'Protein_{i}' for i in range(200)]
    )
    
    # Metabolomics
    metabolomics_data = np.random.exponential(1, size=(n_samples, 150))
    metabolomics_data[:, :15] += aging_signal[:, np.newaxis] * 0.3  # Metabolite signal
    
    data_dict['metabolomics'] = pd.DataFrame(
        metabolomics_data,
        index=sample_names,
        columns=[f'Metabolite_{i}' for i in range(150)]
    )
    
    return data_dict, y

if __name__ == '__main__':
    # Simulate data
    data_dict, labels = simulate_biomarker_data()
    
    # Run biomarker discovery
    engine = RejuvenationBiomarkerEngine()
    results = engine.run_biomarker_discovery_pipeline(data_dict, labels)
    
    print("\nBiomarker discovery completed successfully!")