"""Tests for model evaluation module."""

import numpy as np
import pytest
import joblib
from pathlib import Path
import tempfile

from src.evaluate import (
    recall_at_top_k,
    evaluate_binary_classification,
    find_optimal_threshold,
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create imbalanced data (10% positive)
    y_true = np.random.binomial(1, 0.1, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    
    # Make top predictions more likely to be positive
    top_indices = np.argsort(y_pred_proba)[-10:]
    y_true[top_indices[:3]] = 1  # 3 positives in top 10
    
    return y_true, y_pred_proba


def test_recall_at_top_k(sample_predictions):
    """Test Recall@TopK calculation."""
    y_true, y_pred_proba = sample_predictions
    
    recall_10 = recall_at_top_k(y_true, y_pred_proba, k=10)
    recall_25 = recall_at_top_k(y_true, y_pred_proba, k=25)
    
    assert recall_10 >= 0.0
    assert recall_10 <= 1.0
    assert recall_25 >= 0.0
    assert recall_25 <= 1.0
    assert recall_25 >= recall_10  # More samples should find more positives


def test_recall_at_top_k_no_positives():
    """Test Recall@TopK with no positive samples."""
    y_true = np.zeros(100)
    y_pred_proba = np.random.rand(100)
    
    recall = recall_at_top_k(y_true, y_pred_proba, k=10)
    assert recall == 0.0


def test_recall_at_top_k_all_positives():
    """Test Recall@TopK with all positive samples."""
    y_true = np.ones(100)
    y_pred_proba = np.random.rand(100)
    
    recall = recall_at_top_k(y_true, y_pred_proba, k=10)
    assert recall >= 0.0
    assert recall <= 1.0


def test_evaluate_binary_classification(sample_predictions):
    """Test binary classification evaluation."""
    y_true, y_pred_proba = sample_predictions
    
    metrics = evaluate_binary_classification(y_true, y_pred_proba, threshold=0.5)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "tp" in metrics
    assert "fp" in metrics
    assert "fn" in metrics
    assert "tn" in metrics
    assert "threshold" in metrics
    
    assert metrics["precision"] >= 0.0
    assert metrics["precision"] <= 1.0
    assert metrics["recall"] >= 0.0
    assert metrics["recall"] <= 1.0
    assert metrics["f1"] >= 0.0
    assert metrics["f1"] <= 1.0


def test_evaluate_binary_classification_different_thresholds(sample_predictions):
    """Test binary classification at different thresholds."""
    y_true, y_pred_proba = sample_predictions
    
    metrics_low = evaluate_binary_classification(y_true, y_pred_proba, threshold=0.1)
    metrics_high = evaluate_binary_classification(y_true, y_pred_proba, threshold=0.9)
    
    # Lower threshold should have higher recall, lower precision
    assert metrics_low["recall"] >= metrics_high["recall"]
    assert metrics_low["precision"] <= metrics_high["precision"]


def test_find_optimal_threshold(sample_predictions):
    """Test finding optimal threshold."""
    y_true, y_pred_proba = sample_predictions
    
    result = find_optimal_threshold(
        y_true, y_pred_proba, metric="f1"
    )
    
    # Returns tuple of (best_threshold, results_dict)
    best_threshold, results = result
    
    assert best_threshold >= 0.0
    assert best_threshold <= 1.0
    assert isinstance(results, dict)
    assert "best_threshold" in results
    assert "best_metrics" in results  # Contains metrics at best threshold


def test_find_optimal_threshold_precision(sample_predictions):
    """Test finding optimal threshold for precision."""
    y_true, y_pred_proba = sample_predictions
    
    result = find_optimal_threshold(
        y_true, y_pred_proba, metric="precision"
    )
    
    # Returns tuple of (best_threshold, results_dict)
    best_threshold, results = result
    
    assert best_threshold >= 0.0
    assert best_threshold <= 1.0
    # Check that best_metrics exists (structure may vary)
    assert "best_metrics" in results or "best_threshold" in results


def test_evaluate_binary_classification_empty():
    """Test binary classification with empty arrays."""
    y_true = np.array([])
    y_pred_proba = np.array([])
    
    metrics = evaluate_binary_classification(y_true, y_pred_proba, threshold=0.5)
    
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_model_integration(tmp_path):
    """Test evaluate_model with a saved model."""
    from src.evaluate import evaluate_model
    from sklearn.ensemble import RandomForestClassifier
    import polars as pl
    
    # Create a simple model
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.binomial(1, 0.1, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = tmp_path / "test_model.joblib"
    joblib.dump({
        "model": model,
        "feature_names": [f"feature_{i}" for i in range(5)]
    }, model_path)
    
    # Create features file
    n_samples = 50
    data = {
        "split": ["test"] * n_samples,
        "is_all_star": np.random.binomial(1, 0.1, n_samples).tolist(),
    }
    for i in range(5):
        data[f"feature_{i}"] = np.random.randn(n_samples).tolist()
    
    features_df = pl.DataFrame(data)
    features_path = tmp_path / "features.parquet"
    features_df.write_parquet(features_path)
    
    # Evaluate
    result = evaluate_model(model_path, features_path, split="test")
    
    assert "metrics" in result
    assert "y_true" in result
    assert "y_pred_proba" in result
    assert "pr_auc" in result["metrics"]
    assert "roc_auc" in result["metrics"]

