"""Tests for model training module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from src.train import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    load_features,
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_train = np.random.randn(n_samples, n_features)
    X_val = np.random.randn(30, n_features)
    y_train = np.random.binomial(1, 0.1, n_samples)  # 10% positive
    y_val = np.random.binomial(1, 0.1, 30)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X_train, X_val, y_train, y_val, feature_names


def test_train_logistic_regression(sample_data):
    """Test logistic regression training."""
    X_train, X_val, y_train, y_val, feature_names = sample_data
    
    result = train_logistic_regression(X_train, y_train, X_val, y_val, feature_names)
    
    assert "model" in result
    assert "scaler" in result
    assert "feature_names" in result
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert result["pr_auc"] >= 0.0
    assert result["roc_auc"] >= 0.0
    assert result["roc_auc"] <= 1.0
    assert len(result["feature_names"]) == len(feature_names)


def test_train_random_forest(sample_data):
    """Test random forest training."""
    X_train, X_val, y_train, y_val, feature_names = sample_data
    
    result = train_random_forest(X_train, y_train, X_val, y_val, feature_names)
    
    assert "model" in result
    assert "feature_names" in result
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert result["pr_auc"] >= 0.0
    assert result["roc_auc"] >= 0.0
    assert result["roc_auc"] <= 1.0


def test_train_xgboost(sample_data):
    """Test XGBoost training."""
    X_train, X_val, y_train, y_val, feature_names = sample_data
    
    result = train_xgboost(X_train, y_train, X_val, y_val, feature_names)
    
    assert "model" in result
    assert "feature_names" in result
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert result["pr_auc"] >= 0.0
    assert result["roc_auc"] >= 0.0
    assert result["roc_auc"] <= 1.0


def test_train_lightgbm(sample_data):
    """Test LightGBM training."""
    X_train, X_val, y_train, y_val, feature_names = sample_data
    
    result = train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
    
    if result is not None:  # LightGBM may not be available
        assert "model" in result
        assert "feature_names" in result
        assert "pr_auc" in result
        assert "roc_auc" in result
        assert result["pr_auc"] >= 0.0
        assert result["roc_auc"] >= 0.0
        assert result["roc_auc"] <= 1.0


def test_train_with_missing_values(sample_data):
    """Test training handles missing values."""
    X_train, X_val, y_train, y_val, feature_names = sample_data
    
    # Add some NaN values
    X_train[0, 0] = np.nan
    X_train[5, 2] = np.nan
    
    # Should handle missing values (imputation)
    result = train_logistic_regression(X_train, y_train, X_val, y_val, feature_names)
    assert "model" in result
    assert result["pr_auc"] >= 0.0


def test_train_with_all_negative_class(sample_data):
    """Test training with all negative class."""
    X_train, X_val, y_train, y_val, feature_names = sample_data
    
    # All zeros (no positive class)
    y_train_all_neg = np.zeros_like(y_train)
    y_val_all_neg = np.zeros_like(y_val)
    
    # Logistic regression requires at least 2 classes, so this should raise an error
    with pytest.raises(ValueError):
        train_logistic_regression(X_train, y_train_all_neg, X_val, y_val_all_neg, feature_names)


def test_load_features_nonexistent_file():
    """Test load_features with nonexistent file."""
    with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
        load_features(Path("nonexistent_file.parquet"))


def test_train_all_models_integration(tmp_path):
    """Test training all models end-to-end."""
    from src.train import train_all_models
    
    # Create dummy features file
    import polars as pl
    import numpy as np
    
    n_samples = 100
    n_features = 5
    
    data = {
        "split": ["train"] * 60 + ["val"] * 20 + ["test"] * 20,
        "is_all_star": np.random.binomial(1, 0.1, n_samples).tolist(),
    }
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_samples).tolist()
    
    features_df = pl.DataFrame(data)
    features_path = tmp_path / "features.parquet"
    features_df.write_parquet(features_path)
    
    output_dir = tmp_path / "experiments"
    
    # Train models
    results = train_all_models(
        features_path=features_path,
        output_dir=output_dir
    )
    
    assert isinstance(results, dict)
    assert len(results) > 0
    assert output_dir.exists()

