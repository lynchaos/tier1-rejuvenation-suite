"""
Utility functions for data transformations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Union


def normalize_data(data: pd.DataFrame, method: str = "log1p") -> pd.DataFrame:
    """
    Normalize data using various methods.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to normalize
    method : str
        Normalization method ('log1p', 'zscore', 'quantile')

    Returns:
    --------
    pd.DataFrame : Normalized data
    """
    if method == "log1p":
        return np.log1p(data)
    elif method == "zscore":
        return (data - data.mean()) / data.std()
    elif method == "quantile":
        # Simple quantile normalization implementation
        sorted_data = np.sort(data.values, axis=0)
        ranks = data.rank(method="average")
        quantile_data = data.copy()
        for i, col in enumerate(data.columns):
            quantile_data[col] = sorted_data[
                ranks[col].astype(int) - 1, i % sorted_data.shape[1]
            ]
        return quantile_data
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_features(
    data: pd.DataFrame, method: str = "variance", threshold: float = 0.1
) -> pd.DataFrame:
    """
    Filter features based on various criteria.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    method : str
        Filtering method ('variance', 'missing', 'correlation')
    threshold : float
        Threshold for filtering

    Returns:
    --------
    pd.DataFrame : Filtered data
    """
    if method == "variance":
        variances = data.var()
        return data.loc[:, variances > threshold]
    elif method == "missing":
        missing_rates = data.isnull().mean()
        return data.loc[:, missing_rates <= threshold]
    elif method == "correlation":
        corr_matrix = data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col
            for col in upper_triangle.columns
            if any(upper_triangle[col] > threshold)
        ]
        return data.drop(columns=to_drop)
    else:
        raise ValueError(f"Unknown filtering method: {method}")


def apply_pca(
    data: pd.DataFrame, n_components: int = 10, random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA dimensionality reduction.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    n_components : int
        Number of components to keep
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Tuple[pd.DataFrame, PCA] : Transformed data and fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(
        pca_result, index=data.index, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return pca_df, pca


def scale_data(data: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scale data using various methods.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    method : str
        Scaling method ('standard', 'minmax', 'robust')

    Returns:
    --------
    pd.DataFrame : Scaled data
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)


def remove_batch_effects(
    data: pd.DataFrame, batch_labels: pd.Series, method: str = "combat"
) -> pd.DataFrame:
    """
    Remove batch effects from data.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    batch_labels : pd.Series
        Batch labels for each sample
    method : str
        Batch correction method

    Returns:
    --------
    pd.DataFrame : Batch-corrected data
    """
    # Simple mean centering per batch as basic implementation
    result = data.copy()
    for batch in batch_labels.unique():
        batch_mask = batch_labels == batch
        batch_mean = result.loc[batch_mask].mean()
        result.loc[batch_mask] = result.loc[batch_mask] - batch_mean
    return result


def impute_missing_values(data: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Impute missing values in data.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data with missing values
    method : str
        Imputation method ('mean', 'median', 'zero')

    Returns:
    --------
    pd.DataFrame : Data with imputed values
    """
    if method == "mean":
        return data.fillna(data.mean())
    elif method == "median":
        return data.fillna(data.median())
    elif method == "zero":
        return data.fillna(0)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
