"""
Cell Rejuvenation Scoring Algorithm - Production Implementation
============================================================
Advanced ML-powered quantification of cellular age reversal using multi-omics signatures.
Implements ensemble models with aging biomarkers, senescence markers, and pathway scores.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CellRejuvenationScorer:
    """
    Advanced cell rejuvenation scoring system using ensemble ML models
    """
    
    def __init__(self, model_dir: str = 'models/', random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Define aging and rejuvenation biomarkers
        self.aging_markers = {
            'senescence': ['CDKN1A', 'CDKN2A', 'TP53', 'RB1', 'GLB1', 'LMNB1'],
            'inflammation': ['TNF', 'IL6', 'IL1B', 'NFKB1', 'CXCL1', 'CCL2'],
            'dna_damage': ['ATM', 'BRCA1', 'H2AFX', 'CHEK2', 'MDC1'],
            'oxidative_stress': ['SOD1', 'SOD2', 'CAT', 'GPX1', 'NQO1'],
            'telomere_dysfunction': ['TERT', 'TERF1', 'TERF2', 'RTEL1']
        }
        
        self.rejuvenation_markers = {
            'longevity': ['SIRT1', 'SIRT3', 'SIRT6', 'FOXO1', 'FOXO3', 'KLOTHO'],
            'metabolism': ['AMPK', 'PGC1A', 'PPARA', 'NRF1', 'TFAM'],
            'autophagy': ['ATG5', 'ATG7', 'BECN1', 'LC3B', 'SQSTM1'],
            'stem_cell': ['POU5F1', 'SOX2', 'NANOG', 'KLF4', 'MYC'],
            'antioxidant': ['NFE2L2', 'HMOX1', 'GCLC', 'GSR', 'PRDX1']
        }
    
    def load_data(self, de_path: str) -> pd.DataFrame:
        """
        Load and validate differential expression data
        """
        logger.info(f"Loading differential expression data from {de_path}")
        
        try:
            df = pd.read_csv(de_path, index_col=0)
            logger.info(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Validate required columns
            if 'sample_id' not in df.columns:
                df['sample_id'] = df.index
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_pathway_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aging and rejuvenation pathway scores
        """
        logger.info("Calculating pathway scores...")
        
        pathway_scores = df.copy()
        
        # Calculate aging pathway scores
        for pathway, markers in self.aging_markers.items():
            available_markers = [m for m in markers if m in df.columns]
            if available_markers:
                pathway_scores[f'aging_{pathway}_score'] = df[available_markers].mean(axis=1)
                logger.info(f"Aging {pathway}: {len(available_markers)} markers found")
        
        # Calculate rejuvenation pathway scores
        for pathway, markers in self.rejuvenation_markers.items():
            available_markers = [m for m in markers if m in df.columns]
            if available_markers:
                pathway_scores[f'rejuv_{pathway}_score'] = df[available_markers].mean(axis=1)
                logger.info(f"Rejuvenation {pathway}: {len(available_markers)} markers found")
        
        return pathway_scores
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for rejuvenation scoring
        """
        logger.info("Engineering features for rejuvenation scoring...")
        
        featured_df = df.copy()
        
        # Calculate composite aging score
        aging_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('aging_')]
        if aging_cols:
            featured_df['composite_aging_score'] = df[aging_cols].mean(axis=1)
        
        # Calculate composite rejuvenation score
        rejuv_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('rejuv_')]
        if rejuv_cols:
            featured_df['composite_rejuv_score'] = df[rejuv_cols].mean(axis=1)
        
        # Calculate aging/rejuvenation ratio
        if 'composite_aging_score' in featured_df.columns and 'composite_rejuv_score' in featured_df.columns:
            featured_df['aging_rejuv_ratio'] = (
                featured_df['composite_aging_score'] / 
                (featured_df['composite_rejuv_score'] + 1e-6)
            )
        
        # Log-transform high-variance features
        numeric_cols = featured_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if featured_df[col].var() > 100:  # High variance features
                featured_df[f'{col}_log'] = np.log1p(featured_df[col].clip(lower=0))
        
        # Calculate feature ratios for key markers
        key_ratios = [
            ('CDKN1A', 'SIRT1'),  # Senescence vs Longevity
            ('TNF', 'FOXO3'),     # Inflammation vs Stress Response
            ('TP53', 'NANOG'),    # DNA damage vs Stemness
        ]
        
        for marker1, marker2 in key_ratios:
            if marker1 in df.columns and marker2 in df.columns:
                featured_df[f'{marker1}_{marker2}_ratio'] = (
                    df[marker1] / (df[marker2] + 1e-6)
                )
        
        logger.info(f"Feature engineering complete: {featured_df.shape[1]} features")
        return featured_df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create rejuvenation target variable from biological indicators
        """
        logger.info("Creating rejuvenation target variable...")
        
        # If explicit labels exist, use them
        if 'rejuvenation_label' in df.columns:
            return df['rejuvenation_label']
        
        # Otherwise, create continuous target from biological markers
        target_components = []
        
        # Senescence component (higher = more aged)
        if 'aging_senescence_score' in df.columns:
            senescence_component = -df['aging_senescence_score']  # Negative because lower is better
            target_components.append(senescence_component)
        
        # Longevity component (higher = more rejuvenated)
        if 'rejuv_longevity_score' in df.columns:
            longevity_component = df['rejuv_longevity_score']
            target_components.append(longevity_component)
        
        # Stem cell component (higher = more rejuvenated)
        if 'rejuv_stem_cell_score' in df.columns:
            stemness_component = df['rejuv_stem_cell_score']
            target_components.append(stemness_component)
        
        # Metabolic health component
        if 'rejuv_metabolism_score' in df.columns:
            metabolism_component = df['rejuv_metabolism_score']
            target_components.append(metabolism_component)
        
        if target_components:
            # Combine components and normalize
            target = np.mean(target_components, axis=0)
            target = (target - target.mean()) / target.std()  # Standardize
            target = (target - target.min()) / (target.max() - target.min())  # Scale to [0,1]
        else:
            # Fallback: create synthetic target
            logger.warning("No biological markers found for target creation. Using synthetic target.")
            np.random.seed(self.random_state)
            target = np.random.beta(2, 2, len(df))  # Beta distribution for realistic scores
        
        return pd.Series(target, index=df.index, name='rejuvenation_score')
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train ensemble of models for robust rejuvenation scoring
        """
        logger.info("Training ensemble models...")
        
        # Prepare data
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.median())
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns, index=X_numeric.index)
        
        self.scalers['robust_scaler'] = scaler
        
        # Define models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=self.random_state,
                max_iter=2000
            )
        }
        
        # Train and evaluate models
        model_performance = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled_df, y, cv=5, scoring='r2')
            
            # Fit on full data
            model.fit(X_scaled_df, y)
            
            # Store model and performance
            model_performance[name] = {
                'model': model,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(model.feature_importances_, index=X_scaled_df.columns)
                self.feature_importance[name] = importance.sort_values(ascending=False)
            elif hasattr(model, 'coef_'):
                importance = pd.Series(np.abs(model.coef_), index=X_scaled_df.columns)
                self.feature_importance[name] = importance.sort_values(ascending=False)
            
            logger.info(f"{name} - CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models = model_performance
        return model_performance
    
    def create_ensemble_prediction(self, X: pd.DataFrame) -> pd.Series:
        """
        Create ensemble prediction from all trained models
        """
        logger.info("Creating ensemble predictions...")
        
        X_numeric = X.select_dtypes(include=[np.number]).fillna(X.median())
        X_scaled = self.scalers['robust_scaler'].transform(X_numeric)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns, index=X_numeric.index)
        
        predictions = {}
        weights = {}
        
        # Get predictions from each model
        for name, model_info in self.models.items():
            model = model_info['model']
            pred = model.predict(X_scaled_df)
            predictions[name] = pred
            
            # Weight by cross-validation performance
            weights[name] = max(0, model_info['cv_r2_mean'])  # Avoid negative weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1/len(weights) for k in weights.keys()}
        
        # Create weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Ensure scores are in [0, 1] range
        ensemble_pred = np.clip(ensemble_pred, 0, 1)
        
        return pd.Series(ensemble_pred, index=X.index, name='rejuvenation_score')
    
    def calculate_confidence_intervals(self, X: pd.DataFrame, n_bootstrap: int = 100) -> Dict:
        """
        Calculate confidence intervals using bootstrap sampling
        """
        logger.info(f"Calculating confidence intervals with {n_bootstrap} bootstrap samples...")
        
        bootstrap_predictions = []
        n_samples = len(X)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            
            # Get prediction
            pred = self.create_ensemble_prediction(X_bootstrap)
            bootstrap_predictions.append(pred.values)
        
        # Calculate statistics
        bootstrap_array = np.array(bootstrap_predictions)
        
        confidence_intervals = {
            'mean': np.mean(bootstrap_array, axis=0),
            'std': np.std(bootstrap_array, axis=0),
            'lower_95': np.percentile(bootstrap_array, 2.5, axis=0),
            'upper_95': np.percentile(bootstrap_array, 97.5, axis=0),
            'lower_68': np.percentile(bootstrap_array, 16, axis=0),
            'upper_68': np.percentile(bootstrap_array, 84, axis=0)
        }
        
        return confidence_intervals
    
    def save_models(self) -> None:
        """
        Save trained models and scalers
        """
        logger.info("Saving models and scalers...")
        
        # Save models
        for name, model_info in self.models.items():
            model_path = self.model_dir / f'{name}_model.joblib'
            joblib.dump(model_info['model'], model_path)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f'{name}.joblib'
            joblib.dump(scaler, scaler_path)
        
        # Save feature importance
        importance_path = self.model_dir / 'feature_importance.csv'
        if self.feature_importance:
            importance_df = pd.DataFrame(self.feature_importance)
            importance_df.to_csv(importance_path)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self) -> None:
        """
        Load pre-trained models and scalers
        """
        logger.info("Loading pre-trained models...")
        
        try:
            # Load models
            for model_file in self.model_dir.glob('*_model.joblib'):
                model_name = model_file.stem.replace('_model', '')
                model = joblib.load(model_file)
                self.models[model_name] = {'model': model}
            
            # Load scalers
            for scaler_file in self.model_dir.glob('*scaler.joblib'):
                scaler_name = scaler_file.stem
                scaler = joblib.load(scaler_file)
                self.scalers[scaler_name] = scaler
                
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def score_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to score cellular rejuvenation
        """
        logger.info("Starting cell rejuvenation scoring...")
        
        # Calculate pathway scores
        pathway_df = self.calculate_pathway_scores(df)
        
        # Engineer features
        featured_df = self.engineer_features(pathway_df)
        
        # Create target for training (if not in inference mode)
        if not self.models:  # Training mode
            target = self.create_target_variable(featured_df)
            
            # Train models
            self.train_ensemble_models(featured_df, target)
            
            # Save models
            self.save_models()
        
        # Get predictions
        predictions = self.create_ensemble_prediction(featured_df)
        
        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(featured_df)
        
        # Prepare output
        result_df = df.copy()
        result_df['rejuvenation_score'] = predictions
        result_df['prediction_std'] = confidence_intervals['std']
        result_df['ci_lower_95'] = confidence_intervals['lower_95']
        result_df['ci_upper_95'] = confidence_intervals['upper_95']
        result_df['ci_lower_68'] = confidence_intervals['lower_68']
        result_df['ci_upper_68'] = confidence_intervals['upper_68']
        
        # Add interpretation
        result_df['rejuvenation_category'] = pd.cut(
            result_df['rejuvenation_score'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Aged', 'Partially Aged', 'Intermediate', 'Partially Rejuvenated', 'Rejuvenated']
        )
        
        logger.info("Cell rejuvenation scoring completed!")
        return result_df

def main():
    """
    Main execution function
    """
    # Configuration
    DE_PATH = '../workflows/de_results/differential_expression.csv'
    OUTPUT_PATH = 'ml_predictions.csv'
    
    # Initialize scorer
    scorer = CellRejuvenationScorer()
    
    # Check if data file exists
    if not Path(DE_PATH).exists():
        logger.warning(f"Data file {DE_PATH} not found. Creating synthetic data for demonstration.")
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        
        # Create sample names
        sample_names = [f'Sample_{i}' for i in range(n_samples)]
        
        # Create synthetic expression data with aging signals
        synthetic_data = {}
        
        # Add aging markers
        for category, markers in scorer.aging_markers.items():
            for marker in markers:
                # Simulate aging effect
                synthetic_data[marker] = np.random.lognormal(0, 0.5, n_samples)
        
        # Add rejuvenation markers  
        for category, markers in scorer.rejuvenation_markers.items():
            for marker in markers:
                # Simulate rejuvenation effect
                synthetic_data[marker] = np.random.lognormal(0, 0.5, n_samples)
        
        # Add some random genes
        for i in range(100):
            synthetic_data[f'Gene_{i}'] = np.random.lognormal(0, 0.3, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(synthetic_data, index=sample_names)
        logger.info(f"Created synthetic data: {df.shape}")
        
    else:
        # Load real data
        df = scorer.load_data(DE_PATH)
    
    # Score cells
    scored_df = scorer.score_cells(df)
    
    # Save results
    scored_df.to_csv(OUTPUT_PATH)
    logger.info(f"Results saved to {OUTPUT_PATH}")
    
    # Print summary
    print("\n" + "="*60)
    print("CELL REJUVENATION SCORING SUMMARY")
    print("="*60)
    print(f"Samples processed: {len(scored_df)}")
    print(f"Mean rejuvenation score: {scored_df['rejuvenation_score'].mean():.3f}")
    print(f"Std rejuvenation score: {scored_df['rejuvenation_score'].std():.3f}")
    print(f"Range: {scored_df['rejuvenation_score'].min():.3f} - {scored_df['rejuvenation_score'].max():.3f}")
    print("\nRejuvenation categories:")
    print(scored_df['rejuvenation_category'].value_counts())
    print("\nTop 10 most rejuvenated samples:")
    top_samples = scored_df.nlargest(10, 'rejuvenation_score')[['rejuvenation_score', 'rejuvenation_category']]
    print(top_samples)
    print("="*60)

if __name__ == '__main__':
    main()
