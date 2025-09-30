"""
Configuration management using Pydantic Settings with YAML support.
Centralized configuration for all hyperparameters and settings.
"""

try:
    from pydantic import Field, validator
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Import provenance tracking
from .provenance import ProvenanceTracker


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class NormalizationMethod(str, Enum):
    """Data normalization methods."""

    LOG1P = "log1p"
    ZSCORE = "zscore"
    QUANTILE = "quantile"
    CPM = "cpm"
    NONE = "none"


class ScalingMethod(str, Enum):
    """Data scaling methods."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class BatchCorrectionMethod(str, Enum):
    """Batch correction methods."""

    COMBAT = "combat"
    HARMONY = "harmony"
    SCANORAMA = "scanorama"
    NONE = "none"


class GlobalConfig(BaseSettings):
    """Global configuration settings."""

    # Project metadata
    project_name: str = "tier1-rejuvenation-suite"
    version: str = "1.0.0"

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_directory: Path = Path("logs")

    # Random seeds for reproducibility
    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42

    # Computational settings
    n_jobs: int = -1  # Use all available cores
    memory_limit_gb: Optional[float] = None
    gpu_enabled: bool = False

    # Output settings
    output_directory: Path = Path("outputs")
    save_intermediate: bool = True
    compression: str = "gzip"

    class Config:
        env_prefix = "TIER1_"
        case_sensitive = False

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "GlobalConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(**data.get("global", {}))

    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.safe_dump(self.dict(), f, default_flow_style=False, indent=2)


class DataConfig(BaseSettings):
    """Data processing configuration."""

    # Quality control thresholds
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 5000
    min_cells_per_gene: int = 3
    mitochondrial_threshold: float = 20.0
    ribosomal_threshold: float = 50.0

    # Normalization settings
    normalization_method: NormalizationMethod = NormalizationMethod.LOG1P
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    target_sum: float = 1e4

    # Feature selection
    highly_variable_genes: bool = True
    n_top_genes: int = 2000
    min_mean: float = 0.0125
    max_mean: float = 3.0
    min_disp: float = 0.5

    # Batch correction
    batch_correction_method: BatchCorrectionMethod = BatchCorrectionMethod.HARMONY
    batch_key: Optional[str] = None

    # Missing value handling
    max_missing_rate: float = 0.1
    imputation_method: str = "knn"

    class Config:
        env_prefix = "TIER1_DATA_"


class AnalysisConfig(BaseSettings):
    """Analysis pipeline configuration."""

    # Dimensionality reduction
    n_pcs: int = 50
    n_neighbors: int = 15
    n_components_umap: int = 2
    umap_min_dist: float = 0.5
    umap_spread: float = 1.0

    # Clustering
    clustering_method: str = "leiden"
    clustering_resolution: float = 0.5
    cluster_key: str = "clusters"

    # Trajectory analysis
    trajectory_method: str = "paga"
    root_cluster: Optional[str] = None
    compute_pseudotime: bool = True

    # Machine learning
    test_size: float = 0.2
    cross_validation_folds: int = 5
    model_selection_metric: str = "roc_auc"

    # Biomarker discovery
    biomarker_method: str = "differential_expression"
    significance_threshold: float = 0.05
    fold_change_threshold: float = 1.5

    class Config:
        env_prefix = "TIER1_ANALYSIS_"


class MultiOmicsConfig(BaseSettings):
    """Multi-omics integration configuration."""

    # Integration method
    integration_method: str = "mofa"
    n_factors: int = 10
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6

    # Data modalities
    rna_weight: float = 1.0
    protein_weight: float = 1.0
    metabolite_weight: float = 1.0
    methylation_weight: float = 0.5

    # Feature selection per modality
    features_per_modality: Dict[str, int] = {
        "rna": 2000,
        "protein": 500,
        "metabolite": 200,
        "methylation": 1000,
    }

    # Pathway analysis
    pathway_databases: List[str] = ["kegg", "reactome", "go_bp"]
    min_pathway_size: int = 10
    max_pathway_size: int = 300

    class Config:
        env_prefix = "TIER1_MULTIOMICS_"


class ProvenanceConfig(BaseSettings):
    """Provenance and reproducibility configuration."""

    # Tracking settings
    track_provenance: bool = True
    save_git_info: bool = True
    save_environment: bool = True
    hash_input_data: bool = True

    # Metadata collection
    collect_system_info: bool = True
    collect_package_versions: bool = True
    collect_command_line_args: bool = True

    # Storage settings
    provenance_file: str = "provenance.json"
    compress_provenance: bool = True

    class Config:
        env_prefix = "TIER1_PROVENANCE_"


class TierConfig(BaseSettings):
    """Master configuration combining all sub-configurations."""

    global_config: GlobalConfig = GlobalConfig()
    data: DataConfig = DataConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    multiomics: MultiOmicsConfig = MultiOmicsConfig()
    provenance: ProvenanceConfig = ProvenanceConfig()

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "TierConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle nested objects
        config_dict = self.dict()

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update_from_dict(self, updates: Dict[str, Any]) -> "TierConfig":
        """Update configuration from dictionary."""
        # Deep merge the updates
        current_dict = self.dict()

        def deep_merge(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        merged_dict = deep_merge(current_dict, updates)
        return TierConfig(**merged_dict)

    def validate_config(self) -> List[str]:
        """Validate configuration and return any warnings or errors."""
        warnings = []

        # Check for potential issues
        if self.data.min_genes_per_cell >= self.data.max_genes_per_cell:
            warnings.append("min_genes_per_cell should be less than max_genes_per_cell")

        if self.analysis.test_size <= 0 or self.analysis.test_size >= 1:
            warnings.append("test_size should be between 0 and 1")

        if self.analysis.n_pcs > self.data.n_top_genes:
            warnings.append("n_pcs should not exceed n_top_genes")

        if self.global_config.n_jobs == 0:
            warnings.append("n_jobs should not be 0")

        # Check file paths
        if not self.global_config.output_directory.parent.exists():
            warnings.append(
                f"Output directory parent does not exist: {self.global_config.output_directory.parent}"
            )

        return warnings


# Default configuration instance
default_config = TierConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> TierConfig:
    """
    Load configuration from file or return default.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to YAML configuration file

    Returns:
    --------
    TierConfig : Loaded configuration
    """
    if config_path is None:
        # Look for default config files
        default_paths = ["config.yaml", "configs/default.yaml", "tier1_config.yaml"]

        for path in default_paths:
            if Path(path).exists():
                config_path = path
                break

    if config_path and Path(config_path).exists():
        return TierConfig.from_yaml(config_path)
    else:
        return TierConfig()


def save_config_template(
    output_path: Union[str, Path] = "config_template.yaml",
) -> None:
    """
    Save a template configuration file with all options documented.

    Parameters:
    -----------
    output_path : str or Path
        Where to save the template
    """
    # Create template config with all default values
    config = GlobalConfig()
    config.save_yaml(output_path)
    print(f"Configuration template saved to: {output_path}")


def create_provenance_tracker(
    pipeline_name: str, config: Optional[GlobalConfig] = None
) -> ProvenanceTracker:
    """
    Create a provenance tracker with current configuration.

    Parameters:
    -----------
    pipeline_name : str
        Name of the pipeline being executed
    config : GlobalConfig, optional
        Configuration object to track

    Returns:
    --------
    ProvenanceTracker
        Configured provenance tracker
    """
    config_dict = config.dict() if config else {}
    return ProvenanceTracker(
        pipeline_name=pipeline_name,
        config=config_dict,
        output_dir=config.output_directory if config else "outputs",
    )
    template = TierConfig()
    template.to_yaml(output_path)
    print(f"Configuration template saved to: {output_path}")
