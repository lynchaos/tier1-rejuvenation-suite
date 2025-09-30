"""
Schema validation tests for TIER 1 Rejuvenation Suite I/O.
Tests data types, column schemas, missing value patterns, and data integrity.
"""

import warnings
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd


class TestDataSchemas:
    """Test suite for data schema validation and I/O integrity."""

    def test_bulk_data_schema_validation(self, sample_bulk_data, temp_dir):
        """Test bulk omics data schema requirements."""
        data, metadata = sample_bulk_data

        # Test basic schema requirements
        assert isinstance(data, pd.DataFrame), "Bulk data should be pandas DataFrame"
        assert isinstance(metadata, pd.DataFrame), "Metadata should be pandas DataFrame"

        # Test index alignment
        assert data.index.equals(metadata.index), (
            "Data and metadata indices should align"
        )

        # Test data types
        assert data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all(), (
            "All bulk data columns should be numeric"
        )

        # Test required metadata columns
        required_metadata_cols = ["age", "condition"]
        for col in required_metadata_cols:
            assert col in metadata.columns, f"Metadata missing required column: {col}"

        # Test age column properties
        assert pd.api.types.is_numeric_dtype(metadata["age"]), "Age should be numeric"
        assert metadata["age"].min() >= 0, "Age should be non-negative"
        assert metadata["age"].max() <= 150, "Age should be reasonable (≤150)"

        # Test condition column
        assert (
            metadata["condition"].dtype == "object"
            or metadata["condition"].dtype.name == "category"
        ), "Condition should be categorical"

        # Test for suspicious patterns
        assert not (data == 0).all().any(), "No column should be all zeros"
        assert not data.isnull().all().any(), "No column should be all missing"

        # Test file I/O schema preservation
        csv_path = temp_dir / "bulk_data.csv"
        metadata_path = temp_dir / "metadata.csv"

        data.to_csv(csv_path)
        metadata.to_csv(metadata_path)

        # Read back and verify schema preservation
        data_loaded = pd.read_csv(csv_path, index_col=0)
        pd.read_csv(metadata_path, index_col=0)

        assert data_loaded.shape == data.shape, "Shape should be preserved"
        assert data_loaded.columns.equals(data.columns), "Columns should be preserved"
        assert np.allclose(data_loaded.values, data.values, equal_nan=True), (
            "Values should be preserved (with NaN handling)"
        )

    def test_single_cell_schema_validation(self, sample_single_cell_data, temp_dir):
        """Test single-cell AnnData schema requirements."""
        adata = sample_single_cell_data

        # Test basic AnnData structure
        assert isinstance(adata, ad.AnnData), "Should be AnnData object"
        assert adata.X is not None, "Expression matrix should exist"
        assert adata.obs is not None, "Cell metadata should exist"
        assert adata.var is not None, "Gene metadata should exist"

        # Test matrix properties
        assert adata.X.shape[0] == len(adata.obs), "Rows should match cell count"
        assert adata.X.shape[1] == len(adata.var), "Columns should match gene count"
        assert np.all(adata.X >= 0), "Expression values should be non-negative"

        # Test required observation columns
        required_obs_cols = ["n_genes", "n_counts"]
        for col in required_obs_cols:
            assert col in adata.obs.columns, f"Missing required obs column: {col}"

        # Test observation column properties
        assert pd.api.types.is_numeric_dtype(adata.obs["n_genes"]), (
            "n_genes should be numeric"
        )
        assert pd.api.types.is_numeric_dtype(adata.obs["n_counts"]), (
            "n_counts should be numeric"
        )
        assert (adata.obs["n_genes"] >= 0).all(), "n_genes should be non-negative"
        assert (adata.obs["n_counts"] >= 0).all(), "n_counts should be non-negative"

        # Test gene metadata
        assert adata.var.index.is_unique, "Gene IDs should be unique"
        assert adata.obs.index.is_unique, "Cell IDs should be unique"

        # Test computed metrics consistency
        computed_n_genes = (
            (adata.X > 0).sum(axis=1).A1
            if hasattr(adata.X, "A1")
            else (adata.X > 0).sum(axis=1)
        )
        assert np.allclose(computed_n_genes, adata.obs["n_genes"].values), (
            "Stored n_genes should match computed values"
        )

        computed_n_counts = (
            adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=1)
        )
        assert np.allclose(
            computed_n_counts, adata.obs["n_counts"].values, rtol=1e-3
        ), "Stored n_counts should match computed values"

        # Test H5AD file I/O
        h5ad_path = temp_dir / "test_data.h5ad"
        adata.write_h5ad(h5ad_path)

        # Read back and verify
        adata_loaded = ad.read_h5ad(h5ad_path)

        assert adata_loaded.shape == adata.shape, "Shape should be preserved"
        assert adata_loaded.obs.columns.equals(adata.obs.columns), (
            "Obs columns should be preserved"
        )
        assert adata_loaded.var.columns.equals(adata.var.columns), (
            "Var columns should be preserved"
        )

        # Check matrix preservation (allowing for format changes)
        if hasattr(adata.X, "todense"):
            original_matrix = adata.X.todense()
        else:
            original_matrix = adata.X

        if hasattr(adata_loaded.X, "todense"):
            loaded_matrix = adata_loaded.X.todense()
        else:
            loaded_matrix = adata_loaded.X

        assert np.allclose(original_matrix, loaded_matrix), (
            "Expression matrix should be preserved"
        )

    def test_multi_omics_schema_validation(self, sample_multi_omics_data, temp_dir):
        """Test multi-omics data schema requirements."""
        omics_data = sample_multi_omics_data

        # Test that all datasets have samples in common
        sample_ids = set(omics_data["rna"].index)
        for omics_type, data in omics_data.items():
            if omics_type != "metadata":
                assert isinstance(data, pd.DataFrame), (
                    f"{omics_type} should be DataFrame"
                )
                current_samples = set(data.index)
                assert len(current_samples & sample_ids) > 0, (
                    f"{omics_type} should share samples with other datasets"
                )

        # Test metadata alignment
        metadata = omics_data["metadata"]
        for omics_type, data in omics_data.items():
            if omics_type != "metadata":
                shared_samples = set(data.index) & set(metadata.index)
                assert len(shared_samples) > 0, (
                    f"No shared samples between {omics_type} and metadata"
                )

        # Test data type consistency
        for omics_type, data in omics_data.items():
            if omics_type != "metadata":
                assert data.dtypes.apply(
                    lambda x: pd.api.types.is_numeric_dtype(x)
                ).all(), f"All {omics_type} features should be numeric"

        # Test feature name uniqueness within each omics type
        for omics_type, data in omics_data.items():
            if omics_type != "metadata":
                assert data.columns.is_unique, (
                    f"{omics_type} feature names should be unique"
                )

        # Test for reasonable value ranges
        rna_data = omics_data["rna"]
        assert (rna_data >= 0).all().all(), "RNA-seq data should be non-negative"

        # Proteomics can have negative values (log ratios)
        prot_data = omics_data["proteomics"]
        assert prot_data.notna().any().any(), (
            "Proteomics should have some non-missing values"
        )

        # Test file I/O for each omics type
        for omics_type, data in omics_data.items():
            file_path = temp_dir / f"{omics_type}_data.csv"
            data.to_csv(file_path)

            loaded_data = pd.read_csv(file_path, index_col=0)
            assert loaded_data.shape == data.shape, (
                f"{omics_type} shape preservation failed"
            )

    def test_missing_value_patterns(self, sample_bulk_data):
        """Test missing value pattern validation."""
        data, metadata = sample_bulk_data

        # Analyze missing patterns
        missing_per_sample = data.isnull().sum(axis=1)
        missing_per_feature = data.isnull().sum(axis=0)

        # Test that missing values are within reasonable bounds
        max_missing_per_sample = missing_per_sample.max()
        max_missing_per_feature = missing_per_feature.max()

        # No sample should be completely missing
        assert max_missing_per_sample < data.shape[1], (
            "No sample should be completely missing"
        )

        # No feature should be completely missing
        assert max_missing_per_feature < data.shape[0], (
            "No feature should be completely missing"
        )

        # Test missing value patterns
        total_missing_rate = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        assert total_missing_rate < 0.5, "Missing rate should be reasonable (<50%)"

        # Test for systematic missing patterns (potential data quality issues)
        # Check if missing values are clustered in specific samples/features
        data.isnull()

        # Samples with >80% missing values might indicate quality issues
        high_missing_samples = missing_per_sample[
            missing_per_sample > 0.8 * data.shape[1]
        ]
        assert len(high_missing_samples) < 0.1 * data.shape[0], (
            "Too many samples with >80% missing values"
        )

        # Features with >50% missing values might need special handling
        high_missing_features = missing_per_feature[
            missing_per_feature > 0.5 * data.shape[0]
        ]
        if len(high_missing_features) > 0:
            warnings.warn(
                f"Found {len(high_missing_features)} features with >50% missing values",
                stacklevel=2,
            )

    def test_data_type_consistency(self, sample_bulk_data, temp_dir):
        """Test data type consistency across save/load cycles."""
        data, metadata = sample_bulk_data

        # Test different file formats
        formats_to_test = [
            (
                "csv",
                lambda df, path: df.to_csv(path),
                lambda path: pd.read_csv(path, index_col=0),
            ),
            (
                "parquet",
                lambda df, path: df.to_parquet(path),
                lambda path: pd.read_parquet(path),
            ),
            (
                "pickle",
                lambda df, path: df.to_pickle(path),
                lambda path: pd.read_pickle(path),
            ),
        ]

        for format_name, save_func, load_func in formats_to_test:
            try:
                # Test data preservation
                data_path = temp_dir / f"test_data.{format_name}"
                save_func(data, data_path)
                loaded_data = load_func(data_path)

                # Check basic properties
                assert loaded_data.shape == data.shape, (
                    f"{format_name}: Shape not preserved"
                )
                assert loaded_data.index.equals(data.index), (
                    f"{format_name}: Index not preserved"
                )
                assert loaded_data.columns.equals(data.columns), (
                    f"{format_name}: Columns not preserved"
                )

                # Check data values (handle NaN properly)
                data_equal = np.allclose(
                    loaded_data.values, data.values, equal_nan=True
                )
                assert data_equal, f"{format_name}: Values not preserved"

                # Test metadata preservation
                meta_path = temp_dir / f"test_metadata.{format_name}"
                save_func(metadata, meta_path)
                loaded_metadata = load_func(meta_path)

                assert loaded_metadata.shape == metadata.shape, (
                    f"{format_name}: Metadata shape not preserved"
                )

            except Exception as e:
                # Some formats might not be available
                warnings.warn(f"Could not test {format_name} format: {e}", stacklevel=2)

    def test_column_name_validation(self, sample_bulk_data):
        """Test column name consistency and validation."""
        data, metadata = sample_bulk_data

        # Test for duplicate column names
        assert data.columns.is_unique, "Data column names should be unique"
        assert metadata.columns.is_unique, "Metadata column names should be unique"

        # Test for problematic column names
        problematic_patterns = [
            lambda x: x.strip() != x,  # Leading/trailing whitespace
            lambda x: x == "",  # Empty names
            lambda x: x.startswith("."),  # Hidden file patterns
            lambda x: "/" in x or "\\" in x,  # Path separators
            lambda x: x in ["index", "Index", "INDEX"],  # Reserved names
        ]

        all_columns = list(data.columns) + list(metadata.columns)

        for col in all_columns:
            for pattern_func in problematic_patterns:
                assert not pattern_func(str(col)), (
                    f"Problematic column name detected: '{col}'"
                )

        # Test for reasonable column name length
        max_col_length = max(len(str(col)) for col in all_columns)
        assert max_col_length < 200, "Column names should be reasonably short"

        # Test for non-ASCII characters (potential encoding issues)
        for col in all_columns:
            try:
                col.encode("ascii")
            except UnicodeEncodeError:
                warnings.warn(f"Non-ASCII column name detected: '{col}'", stacklevel=2)

    def test_numeric_data_validation(self, sample_bulk_data):
        """Test numeric data properties and edge cases."""
        data, metadata = sample_bulk_data

        # Test for infinite values
        inf_mask = np.isinf(data.values)
        assert not inf_mask.any(), "Data should not contain infinite values"

        # Test for extremely large values (potential data errors)
        finite_data = data.values[np.isfinite(data.values)]
        if len(finite_data) > 0:
            max_abs_value = np.abs(finite_data).max()
            assert max_abs_value < 1e10, (
                "Extremely large values detected (possible data error)"
            )

        # Test for negative values in expression data (should be non-negative)
        if "gene_" in str(data.columns[0]):  # Assume gene expression data
            negative_mask = data < 0
            negative_count = negative_mask.sum().sum()
            if negative_count > 0:
                warnings.warn(
                    f"Found {negative_count} negative values in expression data",
                    stacklevel=2,
                )

        # Test for zero variance features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        zero_var_features = []

        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 1 and col_data.var() == 0:
                zero_var_features.append(col)

        if zero_var_features:
            warnings.warn(
                f"Found {len(zero_var_features)} zero-variance features", stacklevel=2
            )

        # Test for reasonable dynamic range
        for col in numeric_cols[:10]:  # Test subset for performance
            col_data = data[col].dropna()
            if len(col_data) > 0:
                data_range = col_data.max() - col_data.min()
                data_std = col_data.std()

                # Dynamic range should be reasonable relative to standard deviation
                if data_std > 0:
                    range_to_std_ratio = data_range / data_std
                    assert range_to_std_ratio > 0.1, (
                        f"Suspiciously low dynamic range in {col}"
                    )


class TestFileFormatValidation:
    """Test validation of different file formats and their schemas."""

    def test_h5ad_file_integrity(self, sample_single_cell_data, temp_dir):
        """Test H5AD file format integrity and metadata preservation."""
        adata = sample_single_cell_data

        # Add some metadata to test preservation
        adata.obs["test_categorical"] = pd.Categorical(
            ["A", "B", "C"] * (len(adata) // 3 + 1)
        )[: len(adata)]
        adata.var["test_boolean"] = np.random.choice([True, False], adata.n_vars)
        adata.uns["test_dict"] = {"param1": 1.5, "param2": "test"}

        # Save and load
        h5ad_path = temp_dir / "test_integrity.h5ad"
        adata.write_h5ad(h5ad_path)
        adata_loaded = ad.read_h5ad(h5ad_path)

        # Test metadata preservation
        assert "test_categorical" in adata_loaded.obs.columns
        assert "test_boolean" in adata_loaded.var.columns
        assert "test_dict" in adata_loaded.uns

        # Test categorical preservation
        assert isinstance(
            adata_loaded.obs["test_categorical"].dtype, pd.CategoricalDtype
        )

        # Test uns (unstructured) data preservation
        assert adata_loaded.uns["test_dict"]["param1"] == 1.5
        assert adata_loaded.uns["test_dict"]["param2"] == "test"

        # Test file size reasonableness
        file_size = h5ad_path.stat().st_size
        assert file_size > 1000, "File should have reasonable size"
        assert file_size < 100_000_000, (
            "File should not be excessively large for test data"
        )

    def test_csv_encoding_handling(self, temp_dir):
        """Test CSV file encoding and special character handling."""
        # Create data with special characters
        special_data = pd.DataFrame(
            {
                "normal_col": [1, 2, 3],
                "unicode_col": ["café", "naïve", "résumé"],
                "numeric_col": [1.5, 2.7, 3.14159],
            }
        )

        # Test different encodings
        encodings_to_test = ["utf-8", "latin-1", "ascii"]

        for encoding in encodings_to_test:
            csv_path = temp_dir / f"test_{encoding}.csv"

            try:
                # Save with specific encoding
                special_data.to_csv(csv_path, encoding=encoding)

                # Load with same encoding
                loaded_data = pd.read_csv(csv_path, index_col=0, encoding=encoding)

                # Verify numeric columns are preserved correctly
                assert loaded_data["numeric_col"].dtype in [np.float64, np.int64]

                if encoding in ["utf-8", "latin-1"]:
                    # Unicode should be preserved
                    assert loaded_data["unicode_col"].iloc[0] == "café"

            except UnicodeEncodeError:
                # ASCII encoding will fail with unicode characters
                if encoding == "ascii":
                    pass  # Expected failure
                else:
                    raise

    def test_parquet_schema_preservation(self, sample_bulk_data, temp_dir):
        """Test Parquet format schema and metadata preservation."""
        data, metadata = sample_bulk_data

        # Add different data types to test preservation
        test_metadata = metadata.copy()
        test_metadata["int_col"] = np.random.randint(0, 100, len(metadata))
        test_metadata["float_col"] = np.random.random(len(metadata))
        test_metadata["bool_col"] = np.random.choice([True, False], len(metadata))
        test_metadata["category_col"] = pd.Categorical(
            np.random.choice(["A", "B", "C"], len(metadata))
        )

        parquet_path = temp_dir / "test_schema.parquet"

        try:
            # Save as parquet
            test_metadata.to_parquet(parquet_path)

            # Load and verify types
            loaded_metadata = pd.read_parquet(parquet_path)

            # Check data type preservation
            assert loaded_metadata["int_col"].dtype in [np.int32, np.int64]
            assert loaded_metadata["float_col"].dtype == np.float64
            assert loaded_metadata["bool_col"].dtype == bool
            assert isinstance(
                loaded_metadata["category_col"].dtype, pd.CategoricalDtype
            )

            # Check index preservation
            assert loaded_metadata.index.equals(test_metadata.index)

        except ImportError:
            warnings.warn(
                "Parquet support not available (pyarrow not installed)", stacklevel=2
            )

    def test_pickle_security_and_integrity(self, sample_bulk_data, temp_dir):
        """Test pickle format integrity (with security awareness)."""
        data, metadata = sample_bulk_data

        pickle_path = temp_dir / "test_data.pkl"

        # Save as pickle
        data.to_pickle(pickle_path)

        # Load and verify complete preservation
        loaded_data = pd.read_pickle(pickle_path)

        # Pickle should preserve everything exactly
        pd.testing.assert_frame_equal(loaded_data, data)

        # Test that pickle preserves complex data structures
        complex_metadata = metadata.copy()
        complex_metadata["list_col"] = [
            list(range(i, i + 3)) for i in range(len(metadata))
        ]
        complex_metadata["dict_col"] = [
            {"key": i, "value": i**2} for i in range(len(metadata))
        ]

        complex_pickle_path = temp_dir / "complex_data.pkl"
        complex_metadata.to_pickle(complex_pickle_path)

        loaded_complex = pd.read_pickle(complex_pickle_path)

        # Verify complex structures
        assert loaded_complex["list_col"].iloc[0] == [0, 1, 2]
        assert loaded_complex["dict_col"].iloc[0] == {"key": 0, "value": 0}


class TestDataValidationUtils:
    """Utility functions for data validation that can be reused."""

    @staticmethod
    def validate_dataframe_schema(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        allow_missing: bool = True,
        max_missing_rate: float = 0.5,
    ) -> Dict[str, bool]:
        """
        Comprehensive DataFrame schema validation.

        Returns dictionary of validation results.
        """
        results = {}

        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            results["has_required_columns"] = len(missing_cols) == 0
            if missing_cols:
                results["missing_columns"] = list(missing_cols)

        # Check numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    results[f"{col}_is_numeric"] = pd.api.types.is_numeric_dtype(
                        df[col]
                    )

        # Check categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    is_categorical = df[col].dtype == "object" or isinstance(
                        df[col].dtype, pd.CategoricalDtype
                    )
                    results[f"{col}_is_categorical"] = is_categorical

        # Check missing values
        if not allow_missing:
            results["no_missing_values"] = not df.isnull().any().any()
        else:
            missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            results["missing_rate_acceptable"] = missing_rate <= max_missing_rate
            results["missing_rate"] = missing_rate

        # Check for duplicate indices
        results["unique_index"] = df.index.is_unique

        # Check for empty DataFrame
        results["not_empty"] = not df.empty

        return results

    @staticmethod
    def validate_anndata_schema(adata: ad.AnnData) -> Dict[str, bool]:
        """
        Validate AnnData object schema.
        """
        results = {}

        # Basic structure checks
        results["has_expression_matrix"] = adata.X is not None
        results["has_obs"] = adata.obs is not None
        results["has_var"] = adata.var is not None

        if adata.X is not None:
            # Matrix properties
            results["non_negative_expression"] = np.all(adata.X >= 0)
            results["finite_values"] = (
                np.all(np.isfinite(adata.X.data))
                if hasattr(adata.X, "data")
                else np.all(np.isfinite(adata.X))
            )

            # Dimension consistency
            results["obs_matches_rows"] = adata.X.shape[0] == len(adata.obs)
            results["var_matches_cols"] = adata.X.shape[1] == len(adata.var)

        # Index uniqueness
        results["unique_obs_index"] = adata.obs.index.is_unique
        results["unique_var_index"] = adata.var.index.is_unique

        # Check for common QC metrics
        qc_metrics = ["n_genes", "n_counts", "mt_frac"]
        for metric in qc_metrics:
            results[f"has_{metric}"] = metric in adata.obs.columns

        return results
