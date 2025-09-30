"""
Provenance tracking and data governance for TIER 1 Rejuvenation Suite.
Comprehensive tracking of data lineage, model provenance, and reproducibility metadata.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pkg_resources


@dataclass
class GitInfo:
    """Git repository information."""

    commit_sha: Optional[str] = None
    branch: str = ""
    is_dirty: bool = False
    remote_url: Optional[str] = None
    commit_message: Optional[str] = None
    commit_author: Optional[str] = None
    commit_date: Optional[str] = None

    @classmethod
    def collect(cls) -> "GitInfo":
        """Collect current git information."""
        try:
            # Get commit SHA
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

            # Get branch name
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            # Check if repo is dirty
            try:
                subprocess.check_output(
                    ["git", "diff-index", "--quiet", "HEAD", "--"],
                    stderr=subprocess.DEVNULL,
                )
                is_dirty = False
            except subprocess.CalledProcessError:
                is_dirty = True

            # Get remote URL
            try:
                remote_url = (
                    subprocess.check_output(
                        ["git", "config", "--get", "remote.origin.url"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                remote_url = None

            # Get commit info
            try:
                commit_info = (
                    subprocess.check_output(
                        [
                            "git",
                            "show",
                            "-s",
                            "--format=%s|%an|%ad",
                            "--date=iso",
                            "HEAD",
                        ],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                    .split("|")
                )

                commit_message = commit_info[0] if len(commit_info) > 0 else None
                commit_author = commit_info[1] if len(commit_info) > 1 else None
                commit_date = commit_info[2] if len(commit_info) > 2 else None
            except subprocess.CalledProcessError:
                commit_message = commit_author = commit_date = None

            return cls(
                commit_sha=sha,
                branch=branch,
                is_dirty=is_dirty,
                remote_url=remote_url,
                commit_message=commit_message,
                commit_author=commit_author,
                commit_date=commit_date,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return cls()


@dataclass
class SystemInfo:
    """System and environment information."""

    python_version: str = ""
    platform: str = ""
    processor: str = ""
    hostname: str = ""
    username: str = ""
    working_directory: str = ""
    command_line: List[str] = None
    environment_variables: Dict[str, str] = None

    @classmethod
    def collect(cls, include_env_vars: bool = False) -> "SystemInfo":
        """Collect current system information."""
        return cls(
            python_version=sys.version,
            platform=platform.platform(),
            processor=platform.processor() or platform.machine(),
            hostname=platform.node(),
            username=os.getenv("USER", os.getenv("USERNAME", "unknown")),
            working_directory=str(Path.cwd()),
            command_line=sys.argv.copy(),
            environment_variables=dict(os.environ) if include_env_vars else {},
        )


@dataclass
class PackageInfo:
    """Information about installed packages."""

    packages: Dict[str, str] = None

    @classmethod
    def collect(cls) -> "PackageInfo":
        """Collect information about installed packages."""
        packages = {}

        # Get installed packages
        try:
            for dist in pkg_resources.working_set:
                packages[dist.project_name] = dist.version
        except Exception:
            # Fallback method
            try:
                import pip

                installed_packages = pip.get_installed_distributions()
                for package in installed_packages:
                    packages[package.project_name] = package.version
            except Exception:
                pass

        return cls(packages=packages)


@dataclass
class DataInfo:
    """Information about input data."""

    file_path: str = ""
    file_size: int = 0
    file_hash: str = ""
    shape: Optional[tuple] = None
    columns: Optional[List[str]] = None
    dtypes: Optional[Dict[str, str]] = None
    missing_values: Optional[int] = None
    creation_time: Optional[str] = None
    modification_time: Optional[str] = None

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "DataInfo":
        """Create DataInfo from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Basic file info
        stat = file_path.stat()
        file_size = stat.st_size
        creation_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
        modification_time = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Calculate file hash
        file_hash = calculate_file_hash(file_path)

        return cls(
            file_path=str(file_path),
            file_size=file_size,
            file_hash=file_hash,
            creation_time=creation_time,
            modification_time=modification_time,
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, file_path: Optional[str] = None
    ) -> "DataInfo":
        """Create DataInfo from pandas DataFrame."""
        # Calculate hash of dataframe content
        df_hash = calculate_dataframe_hash(df)

        return cls(
            file_path=file_path or "in_memory",
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values=int(df.isnull().sum().sum()),
            file_hash=df_hash,
        )


@dataclass
class ModelInfo:
    """Information about trained models."""

    model_type: str = ""
    parameters: Dict[str, Any] = None
    training_time: Optional[float] = None
    performance_metrics: Dict[str, float] = None
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None

    @classmethod
    def from_sklearn_model(cls, model, model_type: str = "") -> "ModelInfo":
        """Create ModelInfo from scikit-learn model."""
        # Extract parameters
        try:
            parameters = model.get_params()
        except AttributeError:
            parameters = {}

        # Extract feature importance if available
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(model.feature_importances_)
            }
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 1:
                feature_importance = {
                    f"feature_{i}": float(coef[i]) for i in range(len(coef))
                }

        return cls(
            model_type=model_type or model.__class__.__name__,
            parameters=parameters,
            feature_importance=feature_importance,
        )


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a computational run."""

    # Metadata
    timestamp: str = ""
    run_id: str = ""
    pipeline_name: str = ""
    pipeline_version: str = ""

    # Code and environment
    git_info: GitInfo = None
    system_info: SystemInfo = None
    package_info: PackageInfo = None

    # Configuration
    config: Dict[str, Any] = None
    hyperparameters: Dict[str, Any] = None
    random_seeds: Dict[str, int] = None

    # Data
    input_data: List[DataInfo] = None
    output_data: List[DataInfo] = None

    # Models
    models: List[ModelInfo] = None

    # Results
    metrics: Dict[str, Any] = None
    artifacts: List[str] = None

    # Execution info
    execution_time: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None

    @classmethod
    def create(
        cls,
        pipeline_name: str = "",
        pipeline_version: str = "1.0.0",
        config: Optional[Dict] = None,
    ) -> "ProvenanceRecord":
        """Create a new provenance record."""

        timestamp = datetime.now().isoformat()
        run_id = f"{pipeline_name}_{timestamp.replace(':', '-').replace('.', '-')}"

        return cls(
            timestamp=timestamp,
            run_id=run_id,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            git_info=GitInfo.collect(),
            system_info=SystemInfo.collect(),
            package_info=PackageInfo.collect(),
            config=config or {},
            input_data=[],
            output_data=[],
            models=[],
            metrics={},
            artifacts=[],
        )

    def add_input_data(self, data: Union[str, Path, pd.DataFrame]) -> None:
        """Add input data information."""
        if isinstance(data, (str, Path)):
            data_info = DataInfo.from_file(data)
        elif isinstance(data, pd.DataFrame):
            data_info = DataInfo.from_dataframe(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        if self.input_data is None:
            self.input_data = []
        self.input_data.append(data_info)

    def add_output_data(self, data: Union[str, Path, pd.DataFrame]) -> None:
        """Add output data information."""
        if isinstance(data, (str, Path)):
            data_info = DataInfo.from_file(data)
        elif isinstance(data, pd.DataFrame):
            data_info = DataInfo.from_dataframe(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        if self.output_data is None:
            self.output_data = []
        self.output_data.append(data_info)

    def add_model(self, model, model_type: str = "") -> None:
        """Add model information."""
        model_info = ModelInfo.from_sklearn_model(model, model_type)
        if self.models is None:
            self.models = []
        self.models.append(model_info)

    def add_metric(self, name: str, value: Any) -> None:
        """Add a performance metric."""
        if self.metrics is None:
            self.metrics = {}
        self.metrics[name] = value

    def add_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Add an output artifact."""
        if self.artifacts is None:
            self.artifacts = []
        self.artifacts.append(str(artifact_path))

    def save(self, output_path: Union[str, Path]) -> None:
        """Save provenance record to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict (handles dataclasses)
        record_dict = asdict(self)

        with open(output_path, "w") as f:
            json.dump(record_dict, f, indent=2, default=str)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ProvenanceRecord":
        """Load provenance record from JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        # Reconstruct nested objects
        if data.get("git_info"):
            data["git_info"] = GitInfo(**data["git_info"])
        if data.get("system_info"):
            data["system_info"] = SystemInfo(**data["system_info"])
        if data.get("package_info"):
            data["package_info"] = PackageInfo(**data["package_info"])

        # Reconstruct data info lists
        if data.get("input_data"):
            data["input_data"] = [DataInfo(**d) for d in data["input_data"]]
        if data.get("output_data"):
            data["output_data"] = [DataInfo(**d) for d in data["output_data"]]
        if data.get("models"):
            data["models"] = [ModelInfo(**m) for m in data["models"]]

        return cls(**data)


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate hash of file contents."""
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def calculate_dataframe_hash(df: pd.DataFrame, algorithm: str = "sha256") -> str:
    """Calculate hash of DataFrame contents."""
    hash_func = hashlib.new(algorithm)

    # Convert DataFrame to bytes in a deterministic way
    # Sort columns and index for consistency
    df_sorted = df.sort_index(axis=1).sort_index(axis=0)

    # Handle different dtypes appropriately
    for col in df_sorted.columns:
        col_data = df_sorted[col]
        if col_data.dtype == "object":
            # Convert to string and encode
            data_bytes = str(col_data.tolist()).encode("utf-8")
        else:
            # Use pandas to_bytes for numeric data
            try:
                data_bytes = col_data.values.tobytes()
            except Exception:
                data_bytes = str(col_data.tolist()).encode("utf-8")

        hash_func.update(data_bytes)

    return hash_func.hexdigest()


def create_environment_lock(
    output_path: Union[str, Path] = "environment_lock.txt",
) -> None:
    """Create a complete environment lock file."""
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("# Environment Lock File\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

        # Python version
        f.write(f"Python: {sys.version}\n\n")

        # Platform info
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Architecture: {platform.architecture()}\n")
        f.write(f"Machine: {platform.machine()}\n")
        f.write(f"Processor: {platform.processor()}\n\n")

        # Installed packages
        f.write("Installed Packages:\n")
        for dist in sorted(
            pkg_resources.working_set, key=lambda x: x.project_name.lower()
        ):
            f.write(f"{dist.project_name}=={dist.version}\n")


class ProvenanceTracker:
    """Context manager for tracking provenance during pipeline execution."""

    def __init__(
        self,
        pipeline_name: str,
        config: Optional[Dict] = None,
        output_dir: Union[str, Path] = "outputs",
    ):
        self.pipeline_name = pipeline_name
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.record: Optional[ProvenanceRecord] = None
        self.start_time: Optional[float] = None

    def __enter__(self) -> ProvenanceRecord:
        """Start tracking."""
        import time

        self.start_time = time.time()

        self.record = ProvenanceRecord.create(
            pipeline_name=self.pipeline_name, config=self.config
        )

        # Set random seeds if specified in config
        if "random_seeds" in self.config:
            seeds = self.config["random_seeds"]
            if isinstance(seeds, dict):
                # Set numpy seed
                if "numpy" in seeds:
                    np.random.seed(seeds["numpy"])

                # Set Python random seed
                if "python" in seeds:
                    import random

                    random.seed(seeds["python"])

                # Set torch seed if available
                if "torch" in seeds:
                    try:
                        import torch

                        torch.manual_seed(seeds["torch"])
                    except ImportError:
                        pass

                self.record.random_seeds = seeds

        return self.record

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish tracking and save results."""
        import time

        if self.record and self.start_time:
            self.record.execution_time = time.time() - self.start_time

            # Save provenance record
            self.output_dir.mkdir(parents=True, exist_ok=True)
            provenance_path = self.output_dir / f"{self.record.run_id}_provenance.json"
            self.record.save(provenance_path)

            # Create environment lock
            env_lock_path = (
                self.output_dir / f"{self.record.run_id}_environment_lock.txt"
            )
            create_environment_lock(env_lock_path)

            print(f"Provenance saved to: {provenance_path}")
            print(f"Environment lock saved to: {env_lock_path}")


def compare_provenance_records(
    record1: ProvenanceRecord, record2: ProvenanceRecord
) -> Dict[str, Any]:
    """Compare two provenance records to identify differences."""

    differences = {
        "git_differences": {},
        "system_differences": {},
        "package_differences": {},
        "config_differences": {},
        "data_differences": {},
    }

    # Compare git info
    if record1.git_info and record2.git_info:
        if record1.git_info.commit_sha != record2.git_info.commit_sha:
            differences["git_differences"]["commit_sha"] = {
                "record1": record1.git_info.commit_sha,
                "record2": record2.git_info.commit_sha,
            }

        if record1.git_info.is_dirty != record2.git_info.is_dirty:
            differences["git_differences"]["is_dirty"] = {
                "record1": record1.git_info.is_dirty,
                "record2": record2.git_info.is_dirty,
            }

    # Compare packages
    if record1.package_info and record2.package_info:
        packages1 = record1.package_info.packages or {}
        packages2 = record2.package_info.packages or {}

        all_packages = set(packages1.keys()) | set(packages2.keys())

        for pkg in all_packages:
            version1 = packages1.get(pkg, "not_installed")
            version2 = packages2.get(pkg, "not_installed")

            if version1 != version2:
                differences["package_differences"][pkg] = {
                    "record1": version1,
                    "record2": version2,
                }

    # Compare configurations
    config1 = record1.config or {}
    config2 = record2.config or {}

    def find_dict_differences(d1, d2, path=""):
        diffs = {}
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                diffs[current_path] = {"record1": "missing", "record2": d2[key]}
            elif key not in d2:
                diffs[current_path] = {"record1": d1[key], "record2": "missing"}
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested_diffs = find_dict_differences(d1[key], d2[key], current_path)
                diffs.update(nested_diffs)
            elif d1[key] != d2[key]:
                diffs[current_path] = {"record1": d1[key], "record2": d2[key]}

        return diffs

    differences["config_differences"] = find_dict_differences(config1, config2)

    return differences
