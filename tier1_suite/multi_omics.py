#!/usr/bin/env python3
"""
Multi-Omics Analyzer
====================

Main analyzer class for multi-omics data integration,
fusion analysis, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import existing components
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MultiOmicsFusionIntelligence.integration.biologically_validated_integrator import BiologicallyValidatedIntegrator

class MultiOmicsAnalyzer:
    """
    Comprehensive multi-omics integration and analysis.
    """
    
    def __init__(
        self,
        integration_method: str = "mofa",
        n_factors: int = 10,
        batch_correction: bool = True,
        biomarker_guided: bool = True,
        verbose: bool = False
    ):
        self.integration_method = integration_method
        self.n_factors = n_factors
        self.batch_correction = batch_correction
        self.biomarker_guided = biomarker_guided
        self.verbose = verbose
        
        # Initialize biological integrator
        self.bio_integrator = BiologicallyValidatedIntegrator()
        
        self.models = {}
        self.is_fitted = False
    
    def load_omics_data(self, data_files: List[str], data_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load multiple omics datasets."""
        
        omics_data = {}
        
        for i, file_path in enumerate(data_files):
            # Determine data type
            if data_types and len(data_types) > i:
                data_type = data_types[i]
            else:
                # Infer from filename
                filename = Path(file_path).stem.lower()
                if 'rna' in filename or 'expression' in filename:
                    data_type = 'rna'
                elif 'protein' in filename or 'proteomics' in filename:
                    data_type = 'protein'
                elif 'metabolite' in filename or 'metabolomics' in filename:
                    data_type = 'metabolite'
                elif 'methylation' in filename or 'dna' in filename:
                    data_type = 'methylation'
                else:
                    data_type = f'omics_{i+1}'
            
            # Load data
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.csv':
                data = pd.read_csv(file_path, index_col=0)
            elif file_path_obj.suffix.lower() in ['.tsv', '.txt']:
                data = pd.read_csv(file_path, sep='\t', index_col=0)
            elif file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, index_col=0)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
            
            omics_data[data_type] = data
            
            if self.verbose:
                print(f"Loaded {data_type}: {data.shape[0]} samples, {data.shape[1]} features")
        
        return omics_data
    
    def fit_integration(
        self,
        data_files: List[str],
        data_types: Optional[List[str]] = None,
        output_dir: str = "integration_models"
    ) -> Dict[str, Any]:
        """Fit multi-omics integration models."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load omics data
        omics_data = self.load_omics_data(data_files, data_types)
        
        # Biological validation and preprocessing
        if self.biomarker_guided:
            omics_data = self.bio_integrator.preprocess_with_biomarkers(omics_data)
        
        # Integration based on method
        if self.integration_method.lower() == "mofa":
            integration_results = self._fit_mofa_integration(omics_data)
        elif self.integration_method.lower() in ["ae", "autoencoder"]:
            integration_results = self._fit_autoencoder_integration(omics_data)
        elif self.integration_method.lower() == "pca":
            integration_results = self._fit_pca_integration(omics_data)
        else:
            raise ValueError(f"Unsupported integration method: {self.integration_method}")
        
        # Save models and metadata
        self._save_integration_models(integration_results, output_path)
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"Integration completed using {self.integration_method}")
            print(f"Models saved to: {output_dir}")
        
        return integration_results
    
    def _fit_mofa_integration(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Fit MOFA-style integration."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Concatenate all omics data (simple approach)
        all_data = []
        feature_ranges = {}
        current_idx = 0
        
        for data_type, data in omics_data.items():
            # Standardize
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.T)  # Transpose: samples as rows
            all_data.append(data_scaled)
            
            # Track feature ranges
            feature_ranges[data_type] = {
                'start': current_idx,
                'end': current_idx + data_scaled.shape[1],
                'scaler': scaler
            }
            current_idx += data_scaled.shape[1]
        
        # Concatenate all features
        X_combined = np.hstack(all_data)
        
        # Apply PCA for factor extraction
        pca_model = PCA(n_components=self.n_factors)
        factors = pca_model.fit_transform(X_combined)
        
        integration_results = {
            'factors': factors,
            'feature_ranges': feature_ranges,
            'pca_model': pca_model,
            'explained_variance': pca_model.explained_variance_ratio_,
            'method': 'mofa_pca'
        }
        
        self.models['integration'] = integration_results
        return integration_results
    
    def _fit_autoencoder_integration(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Fit autoencoder-based integration."""
        # Simplified autoencoder using sklearn for compatibility
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Prepare combined data
        all_data = []
        feature_ranges = {}
        current_idx = 0
        
        for data_type, data in omics_data.items():
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.T)
            all_data.append(data_scaled)
            
            feature_ranges[data_type] = {
                'start': current_idx,
                'end': current_idx + data_scaled.shape[1],
                'scaler': scaler
            }
            current_idx += data_scaled.shape[1]
        
        X_combined = np.hstack(all_data)
        
        # Simple autoencoder architecture
        encoder = MLPRegressor(
            hidden_layer_sizes=(X_combined.shape[1]//2, self.n_factors),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        
        # Train to reconstruct input
        encoder.fit(X_combined, X_combined)
        
        # Get latent representation (approximate)
        factors = encoder.predict(X_combined)[:, :self.n_factors]
        
        integration_results = {
            'factors': factors,
            'feature_ranges': feature_ranges,
            'encoder_model': encoder,
            'method': 'autoencoder'
        }
        
        self.models['integration'] = integration_results
        return integration_results
    
    def _fit_pca_integration(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Fit PCA-based integration."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Simple concatenation approach
        all_data = []
        feature_ranges = {}
        current_idx = 0
        
        for data_type, data in omics_data.items():
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.T)
            all_data.append(data_scaled)
            
            feature_ranges[data_type] = {
                'start': current_idx,
                'end': current_idx + data_scaled.shape[1],
                'scaler': scaler
            }
            current_idx += data_scaled.shape[1]
        
        X_combined = np.hstack(all_data)
        
        pca_model = PCA(n_components=self.n_factors)
        factors = pca_model.fit_transform(X_combined)
        
        integration_results = {
            'factors': factors,
            'feature_ranges': feature_ranges,
            'pca_model': pca_model,
            'explained_variance': pca_model.explained_variance_ratio_,
            'method': 'pca'
        }
        
        self.models['integration'] = integration_results
        return integration_results
    
    def _save_integration_models(self, integration_results: Dict[str, Any], output_path: Path):
        """Save integration models and metadata."""
        
        # Save main results
        with open(output_path / 'integration_results.pkl', 'wb') as f:
            pickle.dump(integration_results, f)
        
        # Save factors as CSV
        factors_df = pd.DataFrame(
            integration_results['factors'],
            columns=[f'Factor_{i+1}' for i in range(integration_results['factors'].shape[1])]
        )
        factors_df.to_csv(output_path / 'integration_factors.csv')
        
        # Save metadata
        metadata = {
            'method': self.integration_method,
            'n_factors': self.n_factors,
            'biomarker_guided': self.biomarker_guided,
            'feature_ranges': integration_results['feature_ranges']
        }
        
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_models(self, model_dir: str):
        """Load trained integration models."""
        model_path = Path(model_dir)
        
        # Load integration results
        with open(model_path / 'integration_results.pkl', 'rb') as f:
            integration_results = pickle.load(f)
        
        # Load metadata
        with open(model_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.models['integration'] = integration_results
        self.integration_method = metadata['method']
        self.n_factors = metadata['n_factors']
        self.is_fitted = True
    
    def generate_embeddings(
        self,
        data_files: List[str],
        embedding_dim: int = 50,
        umap_embed: bool = True,
        pathway_analysis: bool = True,
        generate_plots: bool = True
    ) -> pd.DataFrame:
        """Generate integrated embeddings from multi-omics data."""
        
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit_integration() or load_models() first.")
        
        # Load and transform new data using existing models
        omics_data = self.load_omics_data(data_files)
        integration_results = self.models['integration']
        
        # Transform data using existing scalers and models
        all_data = []
        for data_type, data in omics_data.items():
            if data_type in integration_results['feature_ranges']:
                range_info = integration_results['feature_ranges'][data_type]
                scaler = range_info['scaler']
                data_scaled = scaler.transform(data.T)
                all_data.append(data_scaled)
        
        X_combined = np.hstack(all_data)
        
        # Get factors using trained model
        if integration_results['method'] in ['mofa_pca', 'pca']:
            factors = integration_results['pca_model'].transform(X_combined)
        elif integration_results['method'] == 'autoencoder':
            factors = integration_results['encoder_model'].predict(X_combined)[:, :self.n_factors]
        else:
            factors = integration_results['factors']  # Use stored factors
        
        # Create embeddings DataFrame
        embeddings_df = pd.DataFrame(
            factors,
            columns=[f'Factor_{i+1}' for i in range(factors.shape[1])]
        )
        
        # Add UMAP embedding if requested
        if umap_embed:
            try:
                import umap
                umap_model = umap.UMAP(n_components=2, random_state=42)
                umap_coords = umap_model.fit_transform(factors)
                embeddings_df['UMAP_1'] = umap_coords[:, 0]
                embeddings_df['UMAP_2'] = umap_coords[:, 1]
            except ImportError:
                if self.verbose:
                    print("UMAP not available, skipping UMAP embedding")
        
        # Pathway analysis if requested
        if pathway_analysis:
            pathway_scores = self.bio_integrator.analyze_pathway_enrichment(factors, omics_data)
            for pathway, scores in pathway_scores.items():
                embeddings_df[f'pathway_{pathway}'] = scores
        
        return embeddings_df
    
    def evaluate_integration(
        self,
        embeddings: pd.DataFrame,
        reference_data: Optional[str] = None,
        output_dir: str = "evaluation_results",
        metrics: List[str] = ["silhouette", "ari", "nmi"],
        biomarker_validation: bool = True,
        pathway_enrichment: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """Evaluate integration quality."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        evaluation_results = {
            'summary': {},
            'detailed_metrics': {}
        }
        
        # Basic statistical metrics
        factor_cols = [col for col in embeddings.columns if col.startswith('Factor_')]
        if factor_cols:
            factors = embeddings[factor_cols].values
            
            # Compute basic statistics
            evaluation_results['summary']['n_factors'] = len(factor_cols)
            evaluation_results['summary']['explained_variance'] = np.var(factors, axis=0).sum()
            evaluation_results['summary']['factor_correlations'] = np.corrcoef(factors.T).mean()
        
        # Biological validation if requested
        if biomarker_validation:
            bio_validation = self.bio_integrator.validate_integration_biology(embeddings)
            evaluation_results['summary']['biological_score'] = bio_validation.get('overall_score', 0.0)
        
        # Save evaluation report if requested
        if generate_report:
            report_file = output_path / 'evaluation_report.txt'
            with open(report_file, 'w') as f:
                f.write("Multi-Omics Integration Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                for key, value in evaluation_results['summary'].items():
                    f.write(f"{key}: {value}\n")
        
        return evaluation_results
    
    def discover_biomarkers(
        self,
        data_files: List[str],
        embeddings: pd.DataFrame,
        output_dir: str,
        discovery_method: str = "integrated_shap",
        top_n: int = 100,
        pathway_filter: bool = True,
        validation_split: float = 0.3,
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """Discover novel biomarkers from integration."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load omics data
        omics_data = self.load_omics_data(data_files)
        
        # Use biological integrator for discovery
        discovery_results = self.bio_integrator.discover_cross_omics_biomarkers(
            omics_data=omics_data,
            embeddings=embeddings,
            method=discovery_method,
            top_n=top_n,
            pathway_filter=pathway_filter
        )
        
        # Save results
        biomarkers_df = pd.DataFrame(discovery_results['top_biomarkers'])
        biomarkers_df.to_csv(output_path / 'discovered_biomarkers.csv', index=False)
        
        return discovery_results
    
    def run_complete_pipeline(
        self,
        data_files: List[str],
        output_dir: str,
        config_file: Optional[str] = None,
        integration_method: str = "mofa",
        skip_fitting: bool = False,
        skip_embedding: bool = False,
        skip_evaluation: bool = False,
        skip_discovery: bool = False
    ) -> Dict[str, Any]:
        """Run complete multi-omics pipeline."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pipeline_results = {}
        
        # Integration fitting
        if not skip_fitting:
            if self.verbose:
                print("Fitting integration models...")
            integration_results = self.fit_integration(
                data_files=data_files,
                output_dir=str(output_path / "models")
            )
            pipeline_results['integration'] = integration_results
        
        # Embedding generation
        if not skip_embedding:
            if self.verbose:
                print("Generating embeddings...")
            embeddings = self.generate_embeddings(data_files)
            embeddings.to_csv(output_path / "embeddings.csv", index=False)
            pipeline_results['embeddings'] = embeddings
        else:
            # Load existing embeddings
            embeddings = pd.read_csv(output_path / "embeddings.csv")
        
        # Evaluation
        if not skip_evaluation:
            if self.verbose:
                print("Running evaluation...")
            eval_results = self.evaluate_integration(
                embeddings=embeddings,
                output_dir=str(output_path / "evaluation")
            )
            pipeline_results['evaluation'] = eval_results
        
        # Biomarker discovery
        if not skip_discovery:
            if self.verbose:
                print("Discovering biomarkers...")
            discovery_results = self.discover_biomarkers(
                data_files=data_files,
                embeddings=embeddings,
                output_dir=str(output_path / "biomarkers")
            )
            pipeline_results['discovery'] = discovery_results
        
        # Generate summary report
        summary_file = output_path / 'pipeline_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Multi-Omics Integration Pipeline Summary\n")
            f.write("=" * 45 + "\n\n")
            for stage, results in pipeline_results.items():
                f.write(f"{stage.upper()}:\n")
                if isinstance(results, dict) and 'summary' in results:
                    for key, value in results['summary'].items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        if self.verbose:
            print(f"Complete pipeline finished. Results saved to: {output_dir}")
        
        return pipeline_results