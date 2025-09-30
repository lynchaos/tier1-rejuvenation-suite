#!/usr/bin/env python3
"""
Example script demonstrating the complete data governance and provenance tracking system.
Shows how to use the TIER 1 Suite with full reproducibility and governance features.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tier1_suite.config.settings import GlobalConfig, create_provenance_tracker
from tier1_suite.config.provenance import ProvenanceTracker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    """Demonstrate complete data governance workflow."""
    
    print("🔬 TIER 1 Rejuvenation Suite - Data Governance Demo")
    print("=" * 60)
    
    # 1. Load configuration with YAML support
    print("📋 Loading configuration...")
    config = GlobalConfig(
        project_name="tier1_rejuvenation_suite",
        version="1.0.0",
        random_seed=42,
        numpy_seed=42,
        torch_seed=42
    )
    print(f"   Project: {config.project_name}")
    print(f"   Version: {config.version}")
    print(f"   Random seeds: numpy={config.numpy_seed}, torch={config.torch_seed}")
    
    # 2. Create provenance tracker
    print("\n📊 Initializing provenance tracking...")
    
    # Use the provenance tracker as a context manager
    with create_provenance_tracker("data_governance_demo", config) as provenance:
        
        # 3. Generate synthetic data (in practice, load your data)
        print("   Generating synthetic aging dataset...")
        np.random.seed(config.numpy_seed)
        
        # Simulate multi-omics aging data
        n_samples = 500
        n_features = 100
        
        # Generate age-correlated features (genomics, transcriptomics, etc.)
        ages = np.random.uniform(20, 90, n_samples)
        
        # Some features correlate with age (aging biomarkers)
        aging_features = np.random.normal(0, 1, (n_samples, 20))
        for i in range(20):
            aging_features[:, i] += ages * 0.02 * np.random.normal(0.8, 0.2)
        
        # Other features are noise
        noise_features = np.random.normal(0, 1, (n_samples, 80))
        
        # Combine features
        features = np.hstack([aging_features, noise_features])
        
        # Create DataFrame
        df = pd.DataFrame(
            features, 
            columns=[f"feature_{i:03d}" for i in range(n_features)]
        )
        df['age'] = ages
        df['age_group'] = pd.cut(ages, bins=[0, 40, 60, 100], labels=['young', 'middle', 'old'])
        
        print(f"   Dataset shape: {df.shape}")
        
        # 4. Track input data in provenance
        provenance.add_input_data(df)
        
        # 5. Apply transformations with parameter tracking
        print("\n🔧 Applying data transformations...")
        
        # Configure transformation parameters from config
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Standard scaling
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[feature_cols])
        df_scaled = pd.DataFrame(df_scaled, columns=feature_cols, index=df.index)
        
        # Add back non-feature columns
        for col in ['age', 'age_group']:
            df_scaled[col] = df[col]
        
        print(f"   Applied standard scaling to {len(feature_cols)} features")
        
        # PCA transformation
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        df_pca = pca.fit_transform(df_scaled[feature_cols])
        
        # Convert to DataFrame
        pca_columns = [f"PC_{i+1}" for i in range(df_pca.shape[1])]
        df_pca = pd.DataFrame(df_pca, columns=pca_columns, index=df.index)
        
        # Add back non-feature columns
        for col in ['age', 'age_group']:
            df_pca[col] = df_scaled[col]
        
        print(f"   Applied PCA: {len(feature_cols)} -> {len(pca_columns)} components")
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 6. Track transformations and models in provenance
        provenance.add_model(scaler, "StandardScaler")
        provenance.add_model(pca, "PCA")
        
        # 7. Track output data
        provenance.add_output_data(df_pca)
        
        # 8. Add metrics
        provenance.add_metric("n_samples", len(df))
        provenance.add_metric("n_original_features", len(feature_cols))
        provenance.add_metric("n_pca_components", len(pca_columns))
        provenance.add_metric("pca_explained_variance", float(pca.explained_variance_ratio_.sum()))
        
        # 9. Log hyperparameters
        hyperparams = {
            "scaling_method": "standard", 
            "pca_variance_threshold": 0.95,
            "random_seed": config.numpy_seed
        }
        provenance.hyperparameters = hyperparams
        
        print(f"\n📈 Analysis Results:")
        print(f"   Original features: {len(feature_cols)}")
        print(f"   PCA components: {len(pca_columns)}")
        print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
        print(f"   Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
        
        # 10. Save processed data (tracked as artifact)
        output_dir = config.output_directory
        output_dir.mkdir(exist_ok=True)
        
        processed_data_path = output_dir / "processed_aging_data.csv"
        df_pca.to_csv(processed_data_path, index=False)
        provenance.add_artifact(processed_data_path)
        
        print(f"\n💾 Outputs:")
        print(f"   Processed data: {processed_data_path}")
    
    # Provenance is automatically saved when exiting the context manager
    print("\n✅ Data governance demo completed successfully!")
    print("\n📋 Governance Features Demonstrated:")
    print("   ✓ Centralized YAML configuration")
    print("   ✓ Complete provenance tracking")
    print("   ✓ Git repository information")
    print("   ✓ Environment and package versions")
    print("   ✓ Data fingerprinting (hashes)")
    print("   ✓ Model parameter tracking")
    print("   ✓ Hyperparameter logging")
    print("   ✓ Random seed recording")
    print("   ✓ Execution time tracking")
    print("   ✓ Artifact management")
    
    print("\n📁 Check the outputs/ directory for:")
    print("   - Provenance records (JSON)")
    print("   - Environment lock files")
    print("   - Processed datasets")


if __name__ == "__main__":
    main()