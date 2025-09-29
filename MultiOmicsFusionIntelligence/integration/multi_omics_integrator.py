"""
Multi-Omics Fusion Intelligence: Deep Learning Integration Platform
================================================================
AI-powered integration of genomics, transcriptomics, epigenomics, proteomics, and metabolomics data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MultiOmicsAutoencoder(nn.Module):
    """
    Deep learning autoencoder for multi-omics data integration
    """
    def __init__(self, omics_dims: Dict[str, int], latent_dim: int = 128):
        super(MultiOmicsAutoencoder, self).__init__()
        self.omics_dims = omics_dims
        self.latent_dim = latent_dim
        
        # Encoder networks for each omics type
        self.encoders = nn.ModuleDict()
        for omics_type, input_dim in omics_dims.items():
            self.encoders[omics_type] = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 4, latent_dim // len(omics_dims))
            )
        
        # Shared latent space
        self.latent_fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoder networks for each omics type
        self.decoders = nn.ModuleDict()
        for omics_type, output_dim in omics_dims.items():
            self.decoders[omics_type] = nn.Sequential(
                nn.Linear(latent_dim // len(omics_dims), output_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim // 4, output_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim // 2, output_dim)
            )
    
    def encode(self, omics_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multi-omics data to latent representation"""
        encoded_features = []
        for omics_type, data in omics_data.items():
            encoded = self.encoders[omics_type](data)
            encoded_features.append(encoded)
        
        # Concatenate encoded features
        latent = torch.cat(encoded_features, dim=1)
        latent = self.latent_fusion(latent)
        return latent
    
    def decode(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent representation back to omics data"""
        # Split latent representation for each omics type
        chunk_size = self.latent_dim // len(self.omics_dims)
        latent_chunks = torch.chunk(latent, len(self.omics_dims), dim=1)
        
        decoded_data = {}
        for i, (omics_type, _) in enumerate(self.omics_dims.items()):
            decoded_data[omics_type] = self.decoders[omics_type](latent_chunks[i])
        
        return decoded_data
    
    def forward(self, omics_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass: encode then decode"""
        latent = self.encode(omics_data)
        reconstructed = self.decode(latent)
        return latent, reconstructed

class MultiOmicsIntegrator:
    """
    Main class for multi-omics data integration and analysis
    """
    def __init__(self, latent_dim: int = 128, device: str = 'cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.model = None
        self.scalers = {}
        self.omics_data = {}
        
    def load_omics_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load multi-omics datasets
        
        Args:
            data_paths: Dictionary mapping omics types to file paths
        
        Returns:
            Dictionary of loaded datasets
        """
        print("Loading multi-omics datasets...")
        
        for omics_type, path in data_paths.items():
            try:
                if path.endswith('.csv'):
                    data = pd.read_csv(path, index_col=0)
                elif path.endswith('.h5'):
                    data = pd.read_hdf(path, key='data')
                else:
                    data = pd.read_table(path, index_col=0)
                
                self.omics_data[omics_type] = data
                print(f"Loaded {omics_type}: {data.shape}")
                
            except Exception as e:
                print(f"Error loading {omics_type} data: {e}")
        
        return self.omics_data
    
    def preprocess_data(self) -> Dict[str, np.ndarray]:
        """
        Preprocess and normalize multi-omics data
        """
        print("Preprocessing multi-omics data...")
        
        processed_data = {}
        
        for omics_type, data in self.omics_data.items():
            # Handle missing values
            data_filled = data.fillna(data.median())
            
            # Standard scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_filled)
            
            # Store scaler for later use
            self.scalers[omics_type] = scaler
            processed_data[omics_type] = scaled_data
            
            print(f"Processed {omics_type}: {scaled_data.shape}")
        
        return processed_data
    
    def train_autoencoder(self, processed_data: Dict[str, np.ndarray], 
                         num_epochs: int = 100, batch_size: int = 32, 
                         learning_rate: float = 0.001) -> None:
        """
        Train the multi-omics autoencoder
        """
        print("Training multi-omics autoencoder...")
        
        # Get dimensions for each omics type
        omics_dims = {omics_type: data.shape[1] for omics_type, data in processed_data.items()}
        
        # Initialize model
        self.model = MultiOmicsAutoencoder(omics_dims, self.latent_dim)
        self.model.to(self.device)
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        tensor_data = {}
        for omics_type, data in processed_data.items():
            tensor_data[omics_type] = torch.FloatTensor(data).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            latent, reconstructed = self.model(tensor_data)
            
            # Calculate reconstruction loss for each omics type
            total_loss = 0
            for omics_type in tensor_data.keys():
                loss = criterion(reconstructed[omics_type], tensor_data[omics_type])
                total_loss += loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
        
        print("Training completed!")
    
    def get_integrated_representation(self, processed_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get integrated latent representation of multi-omics data
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the autoencoder first.")
        
        self.model.eval()
        
        # Convert to tensors
        tensor_data = {}
        for omics_type, data in processed_data.items():
            tensor_data[omics_type] = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            latent, _ = self.model(tensor_data)
            integrated_data = latent.cpu().numpy()
        
        return integrated_data
    
    def visualize_integration(self, integrated_data: np.ndarray, 
                            sample_labels: Optional[List[str]] = None) -> None:
        """
        Visualize integrated multi-omics data
        """
        print("Generating visualization...")
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(integrated_data)
        
        # t-SNE for non-linear visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(integrated_data)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA plot
        axes[0].scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
        axes[0].set_title(f'PCA of Integrated Multi-Omics Data\n'
                         f'PC1: {pca.explained_variance_ratio_[0]:.2%}, '
                         f'PC2: {pca.explained_variance_ratio_[1]:.2%}')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        
        # t-SNE plot
        axes[1].scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.7)
        axes[1].set_title('t-SNE of Integrated Multi-Omics Data')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        plt.savefig('multi_omics_integration_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_integration_pipeline(self, data_paths: Dict[str, str], 
                                sample_labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Complete multi-omics integration pipeline
        """
        print("Starting Multi-Omics Integration Pipeline...")
        
        # Load and preprocess data
        self.load_omics_data(data_paths)
        processed_data = self.preprocess_data()
        
        # Train autoencoder
        self.train_autoencoder(processed_data)
        
        # Get integrated representation
        integrated_data = self.get_integrated_representation(processed_data)
        
        # Visualize results
        self.visualize_integration(integrated_data, sample_labels)
        
        print("Integration pipeline completed!")
        return integrated_data

# Example usage and data simulation
def simulate_multi_omics_data() -> Dict[str, str]:
    """
    Simulate multi-omics datasets for demonstration
    """
    print("Simulating multi-omics datasets...")
    
    n_samples = 200
    np.random.seed(42)
    
    # Simulate genomics data (SNP variants)
    genomics_data = np.random.binomial(2, 0.3, size=(n_samples, 1000))
    genomics_df = pd.DataFrame(genomics_data, 
                              index=[f'Sample_{i}' for i in range(n_samples)],
                              columns=[f'SNP_{i}' for i in range(1000)])
    
    # Simulate transcriptomics data (gene expression)
    transcriptomics_data = np.random.lognormal(mean=1, sigma=1, size=(n_samples, 2000))
    transcriptomics_df = pd.DataFrame(transcriptomics_data,
                                    index=[f'Sample_{i}' for i in range(n_samples)],
                                    columns=[f'Gene_{i}' for i in range(2000)])
    
    # Simulate proteomics data (protein abundance)
    proteomics_data = np.random.gamma(2, 2, size=(n_samples, 500))
    proteomics_df = pd.DataFrame(proteomics_data,
                                index=[f'Sample_{i}' for i in range(n_samples)],
                                columns=[f'Protein_{i}' for i in range(500)])
    
    # Simulate metabolomics data (metabolite concentrations)
    metabolomics_data = np.random.exponential(1, size=(n_samples, 300))
    metabolomics_df = pd.DataFrame(metabolomics_data,
                                  index=[f'Sample_{i}' for i in range(n_samples)],
                                  columns=[f'Metabolite_{i}' for i in range(300)])
    
    # Save simulated data
    data_paths = {}
    for omics_type, df in [('genomics', genomics_df), ('transcriptomics', transcriptomics_df),
                          ('proteomics', proteomics_df), ('metabolomics', metabolomics_df)]:
        file_path = f'simulated_{omics_type}_data.csv'
        df.to_csv(file_path)
        data_paths[omics_type] = file_path
        print(f"Saved {omics_type} data: {df.shape}")
    
    return data_paths

# Main execution
if __name__ == '__main__':
    # Simulate data for demonstration
    data_paths = simulate_multi_omics_data()
    
    # Run integration pipeline
    integrator = MultiOmicsIntegrator(latent_dim=128)
    integrated_representation = integrator.run_integration_pipeline(data_paths)
    
    print(f"Final integrated representation shape: {integrated_representation.shape}")
    
    # Save integrated data
    np.save('integrated_multi_omics_data.npy', integrated_representation)
    print("Integrated data saved successfully!")