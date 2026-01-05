"""Tests for advanced training with imbalanced data techniques."""

import numpy as np
import pytest

from src.train_advanced import (
    calculate_class_weights,
    apply_smote,
    train_random_forest_advanced,
    train_xgboost_advanced,
)


@pytest.fixture
def imbalanced_data():
    """Create imbalanced training data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X_train = np.random.randn(n_samples, n_features)
    # 5% positive class (highly imbalanced)
    y_train = np.random.binomial(1, 0.05, n_samples)
    
    X_val = np.random.randn(50, n_features)
    y_val = np.random.binomial(1, 0.05, 50)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X_train, X_val, y_train, y_val, feature_names


def test_calculate_class_weights():
    """Test class weight calculation."""
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 10% positive
    
    weights = calculate_class_weights(y_train)
    
    assert isinstance(weights, dict)
    assert 0 in weights
    assert 1 in weights
    assert weights[1] > weights[0]  # Positive class should have higher weight
    assert weights[0] > 0
    assert weights[1] > 0


def test_calculate_class_weights_all_negative():
    """Test class weights with all negative class."""
    y_train = np.zeros(100)
    
    weights = calculate_class_weights(y_train)
    
    assert isinstance(weights, dict)
    assert 0 in weights
    assert 1 in weights


def test_apply_smote(imbalanced_data):
    """Test SMOTE oversampling."""
    X_train, _, y_train, _, _ = imbalanced_data
    
    # Impute missing values first (SMOTE requirement)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    
    X_resampled, y_resampled = apply_smote(X_train_imputed, y_train, k_neighbors=5)
    
    assert X_resampled.shape[0] > X_train.shape[0]  # More samples
    assert X_resampled.shape[1] == X_train.shape[1]  # Same features
    assert y_resampled.sum() > y_train.sum()  # More positive samples
    # Should be roughly balanced
    assert abs(y_resampled.sum() - (len(y_resampled) - y_resampled.sum())) < len(y_resampled) * 0.1


def test_train_random_forest_advanced(imbalanced_data):
    """Test Random Forest with advanced techniques."""
    X_train, X_val, y_train, y_val, feature_names = imbalanced_data
    
    result = train_random_forest_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=True,
        use_smote=True
    )
    
    assert "model" in result
    assert "imputer" in result
    assert "feature_names" in result
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert "techniques" in result
    assert result["techniques"]["class_weights"] is True
    assert result["techniques"]["smote"] is True
    assert result["pr_auc"] >= 0.0
    assert result["roc_auc"] >= 0.0


def test_train_random_forest_advanced_no_smote(imbalanced_data):
    """Test Random Forest with class weights only (no SMOTE)."""
    X_train, X_val, y_train, y_val, feature_names = imbalanced_data
    
    result = train_random_forest_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=True,
        use_smote=False
    )
    
    assert "model" in result
    assert result["techniques"]["smote"] is False
    assert result["techniques"]["class_weights"] is True


def test_train_xgboost_advanced(imbalanced_data):
    """Test XGBoost with advanced techniques."""
    X_train, X_val, y_train, y_val, feature_names = imbalanced_data
    
    result = train_xgboost_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=True,
        use_smote=True
    )
    
    assert "model" in result
    assert "imputer" in result
    assert "feature_names" in result
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert result["techniques"]["class_weights"] is True
    assert result["techniques"]["smote"] is True


def test_smote_handles_missing_values():
    """Test that SMOTE is applied after imputation."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    X_train[0, 0] = np.nan  # Add missing value
    y_train = np.random.binomial(1, 0.1, 100)
    
    # Should impute first, then SMOTE
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Now SMOTE should work
    X_resampled, y_resampled = apply_smote(X_train_imputed, y_train, k_neighbors=5)
    
    assert not np.isnan(X_resampled).any()  # No missing values
    assert X_resampled.shape[0] > X_train.shape[0]

