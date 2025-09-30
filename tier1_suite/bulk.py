#!/usr/bin/env python3
"""
Bulk Data Analyzer
==================

Main analyzer class for bulk omics data processing, ML model training,
and comprehensive biomarker validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Import existing components
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RegenOmicsMaster.ml.biologically_validated_scorer import (
    BiologicallyValidatedRejuvenationScorer,
)


class BulkAnalyzer:
    """
    Comprehensive bulk omics data analyzer with biological validation.
    """

    def __init__(
        self,
        validation_split: float = 0.2,
        cv_folds: int = 5,
        biomarker_validation: bool = True,
        verbose: bool = False,
    ):
        self.validation_split = validation_split
        self.cv_folds = cv_folds
        self.biomarker_validation = biomarker_validation
        self.verbose = verbose

        # Initialize biological validator
        if biomarker_validation:
            self.bio_validator = BiologicallyValidatedRejuvenationScorer()

        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_data(self, input_file: str) -> pd.DataFrame:
        """Load data from various formats."""
        file_path = Path(input_file)

        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(input_file)
        elif file_path.suffix.lower() in [".tsv", ".txt"]:
            return pd.read_csv(input_file, sep="\t")
        elif file_path.suffix.lower() == ".h5":
            return pd.read_hdf(input_file)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def fit_models(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        output_dir: str = "models",
        models: List[str] = ["rf", "xgb", "lgb"],
    ) -> Dict[str, Any]:
        """Fit ensemble ML models with biological validation."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare target variable
        if target_column is None:
            # Use biological validator to create target
            if self.biomarker_validation:
                if self.verbose:
                    print("Creating biologically validated target variable...")
                y = self.bio_validator.create_ensemble_prediction(data)
            else:
                raise ValueError(
                    "No target column specified and biomarker validation disabled"
                )
        else:
            y = data[target_column]
            data = data.drop(columns=[target_column])

        # Prepare features
        X = data.select_dtypes(include=[np.number])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.validation_split, random_state=42
        )

        results = {
            "cv_scores": {},
            "val_scores": {},
            "feature_names": X.columns.tolist(),
        }

        # Train models
        if "rf" in models:
            if self.verbose:
                print("Training Random Forest...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(rf_model, X_train, y_train, cv=self.cv_folds)
            results["cv_scores"]["rf"] = cv_scores.mean()

            # Validation score
            val_score = rf_model.score(X_val, y_val)
            results["val_scores"]["rf"] = val_score

            self.models["rf"] = rf_model
            joblib.dump(rf_model, output_path / "rf_model.pkl")

        # Save additional components
        joblib.dump(self.scaler, output_path / "scaler.pkl")

        # Save metadata
        metadata = {
            "feature_names": results["feature_names"],
            "model_types": list(self.models.keys()),
            "biomarker_validation": self.biomarker_validation,
        }

        with open(output_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.is_fitted = True
        return results

    def load_models(self, model_dir: str):
        """Load trained models from directory."""
        model_path = Path(model_dir)

        # Load scaler
        self.scaler = joblib.load(model_path / "scaler.pkl")

        # Load metadata
        with open(model_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Load models
        for model_type in metadata["model_types"]:
            model_file = model_path / f"{model_type}_model.pkl"
            if model_file.exists():
                self.models[model_type] = joblib.load(model_file)

        self.is_fitted = True

    def predict(
        self,
        data: pd.DataFrame,
        ensemble_method: str = "voting",
        confidence_intervals: bool = True,
        biomarker_analysis: bool = True,
    ) -> pd.DataFrame:
        """Make predictions with ensemble models."""

        if not self.is_fitted:
            raise ValueError(
                "Models not fitted. Call fit_models() or load_models() first."
            )

        # Prepare features
        X = data.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X)

        predictions = {}

        # Get predictions from each model
        for model_name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[f"{model_name}_prediction"] = pred

        # Ensemble prediction
        if ensemble_method == "voting" or ensemble_method == "mean":
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        else:
            ensemble_pred = np.mean(
                list(predictions.values()), axis=0
            )  # Default to mean

        # Create results DataFrame
        results = pd.DataFrame(predictions)
        results["ensemble_prediction"] = ensemble_pred

        # Add confidence intervals if requested
        if confidence_intervals and len(self.models) > 1:
            pred_std = np.std(list(predictions.values()), axis=0)
            results["prediction_std"] = pred_std
            results["lower_ci"] = ensemble_pred - 1.96 * pred_std
            results["upper_ci"] = ensemble_pred + 1.96 * pred_std

        # Add biomarker analysis if requested
        if biomarker_analysis and self.biomarker_validation:
            biomarker_scores = (
                self.bio_validator.calculate_biological_confidence_intervals(data)
            )
            results["biomarker_confidence"] = biomarker_scores

        return results

    def validate_predictions(
        self,
        predictions: pd.DataFrame,
        ground_truth_file: Optional[str] = None,
        output_dir: str = "validation_results",
        generate_report: bool = True,
    ) -> Dict[str, Any]:
        """Validate predictions using biological knowledge."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        validation_results = {"summary": {}, "detailed_metrics": {}}

        # Basic statistical validation
        pred_col = (
            "ensemble_prediction"
            if "ensemble_prediction" in predictions.columns
            else predictions.columns[0]
        )

        validation_results["summary"]["mean_prediction"] = predictions[pred_col].mean()
        validation_results["summary"]["std_prediction"] = predictions[pred_col].std()
        validation_results["summary"]["n_samples"] = len(predictions)

        # Biological validation if available
        if self.biomarker_validation and "biomarker_confidence" in predictions.columns:
            bio_corr = np.corrcoef(
                predictions[pred_col], predictions["biomarker_confidence"]
            )[0, 1]
            validation_results["summary"]["biomarker_correlation"] = bio_corr

        # Save results
        if generate_report:
            report_file = output_path / "validation_report.txt"
            with open(report_file, "w") as f:
                f.write("Bulk Analysis Validation Report\n")
                f.write("=" * 40 + "\n\n")
                for key, value in validation_results["summary"].items():
                    f.write(f"{key}: {value}\n")

        return validation_results
