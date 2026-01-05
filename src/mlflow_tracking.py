"""MLflow experiment tracking integration."""

import json
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from loguru import logger

from src.config import config

# Set MLflow tracking URI to local directory
MLFLOW_TRACKING_URI = config.project_root / "mlruns"
mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))

# Set experiment name
EXPERIMENT_NAME = "mlb_all_star_prediction"
mlflow.set_experiment(EXPERIMENT_NAME)


def log_model_training(
    model_name: str,
    model,
    feature_names: list[str],
    metrics: dict,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    model_path: Optional[Path] = None,
) -> None:
    """
    Log model training to MLflow.

    Args:
        model_name: Name of the model
        model: Trained model object
        feature_names: List of feature names
        metrics: Dictionary of metrics (e.g., {"pr_auc": 0.95, "roc_auc": 0.87})
        params: Dictionary of hyperparameters
        tags: Dictionary of tags (e.g., {"technique": "smote", "features": "with_interactions"})
        model_path: Path to saved model (optional)
    """
    logger.info(f"Logging {model_name} to MLflow")

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log model
        try:
            if "xgboost" in model_name.lower():
                mlflow.xgboost.log_model(model, "model")
            elif "lightgbm" in model_name.lower() or "lgbm" in model_name.lower():
                try:
                    import mlflow.lightgbm
                    mlflow.lightgbm.log_model(model, "model")
                except (ImportError, AttributeError):
                    # Fallback to sklearn if lightgbm flavor not available
                    mlflow.sklearn.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            logger.warning(f"Could not log model to MLflow: {e}")
            # Continue anyway - model logging is optional

        # Log feature names
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        # Log model path if provided
        if model_path:
            mlflow.log_artifact(str(model_path), "models")

        logger.info(f"Logged {model_name} to MLflow run: {mlflow.active_run().info.run_id}")


def log_hyperparameter_tuning(
    model_name: str,
    best_params: dict,
    best_metrics: dict,
    study_path: Optional[Path] = None,
    tags: Optional[dict] = None,
) -> None:
    """
    Log hyperparameter tuning results to MLflow.

    Args:
        model_name: Name of the model
        best_params: Best hyperparameters found
        best_metrics: Best metrics achieved
        study_path: Path to Optuna study (optional)
        tags: Dictionary of tags
    """
    run_name = f"{model_name}_tuned"
    logger.info(f"Logging hyperparameter tuning for {model_name} to MLflow")

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metrics(best_metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("tuning_method", "optuna")
        mlflow.set_tag("model_type", model_name)

        # Log study if provided
        if study_path and study_path.exists():
            mlflow.log_artifact(str(study_path), "tuning")

        logger.info(f"Logged tuning results for {model_name}")


def log_evaluation_metrics(
    model_name: str,
    metrics: dict,
    split: str = "test",
    tags: Optional[dict] = None,
) -> None:
    """
    Log evaluation metrics to MLflow.

    Args:
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        split: Data split (train/val/test)
        tags: Dictionary of tags
    """
    run_name = f"{model_name}_eval_{split}"
    logger.info(f"Logging evaluation metrics for {model_name} on {split} split")

    with mlflow.start_run(run_name=run_name, nested=True):
        # Log metrics with split prefix
        prefixed_metrics = {f"{split}_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("split", split)
        mlflow.set_tag("model_type", model_name)

        logger.info(f"Logged evaluation metrics for {model_name} on {split}")


def log_feature_engineering(
    feature_count: int,
    interaction_features: bool = False,
    tags: Optional[dict] = None,
) -> None:
    """
    Log feature engineering information.

    Args:
        feature_count: Number of features
        interaction_features: Whether interaction features were used
        tags: Dictionary of tags
    """
    run_name = "feature_engineering"
    logger.info("Logging feature engineering to MLflow")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("feature_count", feature_count)
        mlflow.log_param("interaction_features", interaction_features)

        if tags:
            mlflow.set_tags(tags)

        logger.info("Logged feature engineering information")


def get_best_run(metric: str = "pr_auc", ascending: bool = False) -> Optional[dict]:
    """
    Get the best run based on a metric.

    Args:
        metric: Metric name to optimize
        ascending: If True, lower is better; if False, higher is better

    Returns:
        Dictionary with best run information, or None if no runs found
    """
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            return None

        # Find best run
        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            # Try with split prefix
            metric_col = f"metrics.test_{metric}"
            if metric_col not in runs.columns:
                logger.warning(f"Metric {metric} not found in runs")
                return None

        best_idx = runs[metric_col].idxmax() if not ascending else runs[metric_col].idxmin()
        best_run = runs.iloc[best_idx]

        return {
            "run_id": best_run["run_id"],
            "metric": metric,
            "value": best_run[metric_col],
            "params": {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")},
            "metrics": {k.replace("metrics.", ""): v for k, v in best_run.items() if k.startswith("metrics.")},
        }
    except Exception as e:
        logger.warning(f"Could not get best run: {e}")
        return None


def compare_runs(run_ids: list[str]) -> dict:
    """
    Compare multiple MLflow runs.

    Args:
        run_ids: List of run IDs to compare

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    for run_id in run_ids:
        try:
            run = mlflow.get_run(run_id)
            comparison[run_id] = {
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.warning(f"Could not get run {run_id}: {e}")

    return comparison

