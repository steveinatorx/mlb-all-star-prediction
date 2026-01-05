"""Tests for MLflow tracking module."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not available")
def test_log_model_training():
    """Test logging model training to MLflow."""
    from src.mlflow_tracking import log_model_training
    
    # Create a simple model
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.binomial(1, 0.1, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    feature_names = [f"feature_{i}" for i in range(5)]
    metrics = {"pr_auc": 0.5, "roc_auc": 0.7}
    params = {"n_estimators": 10, "max_depth": 5}
    tags = {"technique": "baseline"}
    
    # Should not raise exception
    log_model_training(
        model_name="test_model",
        model=model,
        feature_names=feature_names,
        metrics=metrics,
        params=params,
        tags=tags
    )


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not available")
def test_log_hyperparameter_tuning():
    """Test logging hyperparameter tuning results."""
    from src.mlflow_tracking import log_hyperparameter_tuning
    
    best_params = {"n_estimators": 150, "max_depth": 7}
    best_metrics = {"pr_auc": 0.6}
    tags = {"tuning_method": "optuna"}
    
    # Should not raise exception
    log_hyperparameter_tuning(
        model_name="test_model",
        best_params=best_params,
        best_metrics=best_metrics,
        tags=tags
    )


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not available")
def test_log_evaluation_metrics():
    """Test logging evaluation metrics."""
    from src.mlflow_tracking import log_evaluation_metrics
    
    metrics = {"pr_auc": 0.5, "roc_auc": 0.7, "recall_top10": 0.1}
    tags = {"split": "test"}
    
    # Should not raise exception
    log_evaluation_metrics(
        model_name="test_model",
        metrics=metrics,
        split="test",
        tags=tags
    )


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not available")
def test_get_best_run():
    """Test getting best run from MLflow."""
    from src.mlflow_tracking import get_best_run
    
    # May return None if no runs exist
    best_run = get_best_run(metric="pr_auc", ascending=False)
    
    # If runs exist, should have expected structure
    if best_run is not None:
        assert "run_id" in best_run
        assert "metric" in best_run
        assert "value" in best_run


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not available")
def test_compare_runs():
    """Test comparing multiple runs."""
    from src.mlflow_tracking import compare_runs
    
    # Empty list should return empty dict
    comparison = compare_runs([])
    assert isinstance(comparison, dict)
    assert len(comparison) == 0

