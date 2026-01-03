"""Model evaluation and metric computation."""

import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import config

try:
    import shap
except ImportError:
    shap = None
    logger.warning("SHAP not available, skipping SHAP plots")


def recall_at_top_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
    """
    Compute Recall@TopK: recall among top K predictions.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        k: Top K to consider

    Returns:
        Recall@TopK score
    """
    if len(y_true) == 0:
        return 0.0

    # Get top K indices
    top_k_indices = np.argsort(y_pred_proba)[-k:]
    top_k_true = y_true[top_k_indices]

    # Recall = TP / (TP + FN) = positives in top K / total positives
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0

    return top_k_true.sum() / total_positives


def evaluate_model(
    model_path: Path,
    features_path: Optional[Path] = None,
    split: str = "test",
) -> dict:
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        features_path: Path to features file
        split: Which split to evaluate on

    Returns:
        Dictionary with metrics
    """
    logger.info(f"Evaluating model: {model_path.name} on {split} split")

    # Load model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Load features
    features_path = features_path or config.features_data_dir / "features.parquet"
    df = pl.read_parquet(features_path)

    # Filter by split
    split_df = df.filter(pl.col("split") == split)

    # Extract X and y
    X = split_df.select(feature_names).to_numpy()
    y_true = split_df["is_all_star"].to_numpy().astype(int)

    # Handle scaler if present (for logistic regression)
    if "scaler" in model_data:
        scaler = model_data["scaler"]
        X = scaler.transform(X)

    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Compute metrics
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Recall@TopK
    recall_topk = {
        f"recall_at_top_{k}": recall_at_top_k(y_true, y_pred_proba, k)
        for k in config.top_k_values
    }

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        **recall_topk,
        "n_samples": len(y_true),
        "n_positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }

    logger.info(f"Metrics: {metrics}")

    return {
        "metrics": metrics,
        "y_true": y_true.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_names": feature_names,
    }


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved PR curve to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str, output_path: Path
) -> None:
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (ROC-AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curve to {output_path}")


def plot_shap_importance(
    model_path: Path,
    features_path: Optional[Path] = None,
    split: str = "test",
    output_path: Optional[Path] = None,
    max_samples: int = 100,
) -> None:
    """
    Generate SHAP plots for tree-based models.

    Args:
        model_path: Path to saved model
        features_path: Path to features file
        split: Which split to use for SHAP
        output_path: Output path for plot
        max_samples: Maximum samples for SHAP (can be slow)
    """
    if shap is None:
        logger.warning("SHAP not available, skipping")
        return

    logger.info(f"Generating SHAP plots for {model_path.name}")

    # Load model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Check if model supports SHAP (tree-based)
    if not hasattr(model, "predict_proba"):
        logger.warning("Model does not support SHAP")
        return

    # Load features
    features_path = features_path or config.features_data_dir / "features.parquet"
    df = pl.read_parquet(features_path)
    split_df = df.filter(pl.col("split") == split)

    # Sample if too many
    if len(split_df) > max_samples:
        split_df = split_df.sample(max_samples, seed=config.random_seed)

    X = split_df.select(feature_names).to_numpy()

    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle binary classification (shap_values is a list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Summary plot
        output_path = output_path or config.figures_dir / f"shap_{model_path.stem}.png"
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP plot to {output_path}")

    except Exception as e:
        logger.warning(f"Failed to generate SHAP plot: {e}")


def plot_coefficient_importance(
    coefficients: list[float],
    feature_names: list[str],
    model_name: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot coefficient importance for logistic regression."""
    # Sort by absolute value
    coef_abs = np.abs(coefficients)
    top_indices = np.argsort(coef_abs)[-top_n:]

    top_coefs = [coefficients[i] for i in top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 8))
    colors = ["red" if c < 0 else "blue" for c in top_coefs]
    plt.barh(range(len(top_coefs)), top_coefs, color=colors)
    plt.yticks(range(len(top_coefs)), top_features)
    plt.xlabel("Coefficient Value")
    plt.title(f"Top {top_n} Feature Coefficients: {model_name}")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved coefficient plot to {output_path}")


def evaluate_all_models(
    experiments_dir: Optional[Path] = None,
    features_path: Optional[Path] = None,
    split: str = "test",
) -> dict:
    """
    Evaluate all trained models and generate plots.

    Args:
        experiments_dir: Directory containing trained models
        features_path: Path to features file
        split: Which split to evaluate on

    Returns:
        Dictionary with all evaluation results
    """
    experiments_dir = experiments_dir or config.experiments_dir
    figures_dir = config.figures_dir
    tables_dir = config.tables_dir

    logger.info(f"Evaluating all models on {split} split")

    results = {}

    # Find all model files
    model_files = list(experiments_dir.glob("*.joblib"))
    model_files = [f for f in model_files if f.name != "training_results.json"]

    for model_path in model_files:
        model_name = model_path.stem
        logger.info(f"Evaluating {model_name}")

        # Evaluate
        eval_result = evaluate_model(model_path, features_path, split=split)

        # Generate plots
        plot_precision_recall_curve(
            np.array(eval_result["y_true"]),
            np.array(eval_result["y_pred_proba"]),
            model_name,
            figures_dir / f"pr_curve_{model_name}.png",
        )

        plot_roc_curve(
            np.array(eval_result["y_true"]),
            np.array(eval_result["y_pred_proba"]),
            model_name,
            figures_dir / f"roc_curve_{model_name}.png",
        )

        # SHAP for tree-based models
        if model_name in ["xgboost", "lightgbm", "random_forest"]:
            plot_shap_importance(
                model_path,
                features_path,
                split=split,
                output_path=figures_dir / f"shap_{model_name}.png",
            )

        # Coefficient plot for logistic regression
        if model_name == "logistic_regression":
            model_data = joblib.load(model_path)
            if "coefficients" in model_data:
                plot_coefficient_importance(
                    model_data["coefficients"],
                    model_data["feature_names"],
                    model_name,
                    figures_dir / f"coefficients_{model_name}.png",
                )

        results[model_name] = eval_result["metrics"]

        # Save metrics to CSV
        metrics_df = pl.DataFrame([eval_result["metrics"]])
        metrics_df.write_csv(tables_dir / f"metrics_{model_name}.csv")

    # Save summary
    summary_path = tables_dir / "evaluation_summary.csv"
    summary_df = pl.DataFrame(results).transpose(include_header=True, header_name="model")
    summary_df.write_csv(summary_path)

    logger.info(f"Saved evaluation results to {tables_dir}")

    return results

