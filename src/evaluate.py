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


def evaluate_binary_classification(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Evaluate binary classification at a given threshold.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with precision, recall, F1, and confusion matrix components
    """
    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    # Apply threshold
    y_pred = (y_pred_proba > threshold).astype(int)

    # Calculate confusion matrix components
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[list[float]] = None,
) -> tuple[float, dict]:
    """
    Find optimal threshold for binary classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', or 'pr_auc')
        thresholds: List of thresholds to try (default: 0.05 to 0.95 in 0.05 steps)

    Returns:
        Tuple of (best_threshold, results_dict) where results_dict contains
        metrics at each threshold and the best threshold's metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05).tolist()

    results = []
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        metrics = evaluate_binary_classification(y_true, y_pred_proba, threshold)
        results.append(metrics)

        # Determine score based on metric
        if metric == "f1":
            score = metrics["f1"]
        elif metric == "precision":
            score = metrics["precision"]
        elif metric == "recall":
            score = metrics["recall"]
        elif metric == "pr_auc":
            # For PR-AUC, we'd need to calculate it, but for threshold selection
            # we'll use a combination metric
            score = metrics["f1"]  # Fallback to F1
        else:
            score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold

    # Get metrics at best threshold
    best_metrics = evaluate_binary_classification(y_true, y_pred_proba, best_threshold)

    return best_threshold, {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "all_thresholds": results,
    }


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

    # Encode categorical columns before converting to numpy
    # highest_level_reached: ordinal encoding (A=1, A+=2, AA=3, AAA=4, Unknown=0)
    if "highest_level_reached" in df.columns:
        df = df.with_columns([
            pl.when(pl.col("highest_level_reached") == "A")
            .then(1)
            .when(pl.col("highest_level_reached") == "A+")
            .then(2)
            .when(pl.col("highest_level_reached") == "AA")
            .then(3)
            .when(pl.col("highest_level_reached") == "AAA")
            .then(4)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("highest_level_reached")
        ])

    # Filter by split
    split_df = df.filter(pl.col("split") == split)

    # Extract X and y - convert to float64, handling nulls
    # Convert each column separately to handle nulls properly
    import numpy as np
    
    def df_to_numpy(df_subset: pl.DataFrame) -> np.ndarray:
        """Convert Polars DataFrame to numpy array, handling nulls and missing features."""
        arrays = []
        for col in feature_names:
            if col not in df_subset.columns:
                # Feature missing - fill with zeros
                logger.warning(f"Feature '{col}' not found in DataFrame, filling with zeros")
                arr = np.zeros(len(df_subset), dtype=np.float64)
            else:
                col_data = df_subset[col]
                # Convert to numpy, handling nulls
                if col_data.dtype in [pl.Float64, pl.Float32]:
                    arr = col_data.to_numpy()
                    arr = np.array([float(x) if x is not None else np.nan for x in arr])
                elif col_data.dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64]:
                    arr = col_data.to_numpy()
                    arr = np.array([float(x) if x is not None else np.nan for x in arr])
                else:
                    arr = np.array([float(x) if x is not None else np.nan for x in col_data.to_list()])
            arrays.append(arr)
        
        X = np.column_stack(arrays).astype(np.float64)
        
        # Ensure we have the right number of features expected by the model
        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
            if X.shape[1] != expected_features:
                if X.shape[1] < expected_features:
                    # Pad with zeros if we have fewer features
                    logger.warning(
                        f"X has {X.shape[1]} features, model expects {expected_features}. "
                        f"Padding with zeros."
                    )
                    padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                    X = np.column_stack([X, padding])
                else:
                    # Truncate if we have more features
                    logger.warning(
                        f"X has {X.shape[1]} features, model expects {expected_features}. "
                        f"Using first {expected_features} features."
                    )
                    X = X[:, :expected_features]
        
        return X
    
    # Select features in the exact order expected by the model
    missing_cols = [col for col in feature_names if col not in split_df.columns]
    if missing_cols:
        logger.warning(f"Missing features: {missing_cols}. Adding as null columns.")
        for col in missing_cols:
            split_df = split_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    
    feature_df = split_df.select(feature_names)
    
    # Convert to numpy - ensure we get all features in the right order
    arrays = []
    for col in feature_names:
        if col not in feature_df.columns:
            logger.warning(f"Feature '{col}' not found after select, filling with zeros")
            arr = np.zeros(len(feature_df), dtype=np.float64)
        else:
            col_data = feature_df[col]
            # Convert to numpy, handling nulls
            if col_data.dtype in [pl.Float64, pl.Float32]:
                arr = col_data.to_numpy()
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            elif col_data.dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64]:
                arr = col_data.to_numpy()
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            else:
                arr = np.array([float(x) if x is not None else np.nan for x in col_data.to_list()])
        arrays.append(arr)
    
    X = np.column_stack(arrays).astype(np.float64)
    
    # Ensure we have the right number of features expected by the model
    if hasattr(model, "n_features_in_"):
        expected_features = model.n_features_in_
        if X.shape[1] != expected_features:
            if X.shape[1] < expected_features:
                # Pad with zeros if we have fewer features
                logger.warning(
                    f"X has {X.shape[1]} features, model expects {expected_features}. "
                    f"Padding with zeros."
                )
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.column_stack([X, padding])
            else:
                # Truncate if we have more features
                logger.warning(
                    f"X has {X.shape[1]} features, model expects {expected_features}. "
                    f"Using first {expected_features} features."
                )
                X = X[:, :expected_features]
    
    y_true = split_df["is_all_star"].to_numpy().astype(int)

    # Handle imputer if present (for models that need imputation)
    if "imputer" in model_data:
        imputer = model_data["imputer"]
        # Check if feature count matches (may differ if features changed)
        if X.shape[1] == imputer.n_features_in_:
            X = imputer.transform(X)
        else:
            logger.warning(
                f"Feature count mismatch: X has {X.shape[1]} features, "
                f"imputer expects {imputer.n_features_in_}. Using SimpleImputer as fallback."
            )
            # Fallback: impute with median, ensuring we keep all features
            from sklearn.impute import SimpleImputer
            fallback_imputer = SimpleImputer(strategy="median")
            X = fallback_imputer.fit_transform(X)
            # Ensure we still have the right number of features after imputation
            if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
                if X.shape[1] < model.n_features_in_:
                    padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
                    X = np.column_stack([X, padding])
                else:
                    X = X[:, :model.n_features_in_]
    elif np.isnan(X).any():
        # Impute missing values for models that don't handle NaN (e.g., Logistic Regression)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
        # Ensure we still have the right number of features after imputation
        if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
            if X.shape[1] < model.n_features_in_:
                padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
                X = np.column_stack([X, padding])
            else:
                X = X[:, :model.n_features_in_]
    
    # Handle scaler if present (for logistic regression)
    if "scaler" in model_data:
        scaler = model_data["scaler"]
        # Check if feature count matches (may differ if features changed)
        if X.shape[1] == scaler.n_features_in_:
            X = scaler.transform(X)
        else:
            logger.warning(
                f"Feature count mismatch: X has {X.shape[1]} features, "
                f"scaler expects {scaler.n_features_in_}. Skipping scaling."
            )

    # Predictions
    # Handle models that might not have predict_proba (e.g., GAM)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)
        # Handle binary classification (2D array) vs multi-class
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred_proba.flatten()
    else:
        # Fallback: use decision_function or predict
        if hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X)
            # Normalize to [0, 1] using sigmoid
            from scipy.special import expit
            y_pred_proba = expit(y_pred_proba)
        else:
            # Last resort: use predict and convert to probabilities
            y_pred = model.predict(X)
            y_pred_proba = y_pred.astype(float)
            logger.warning(f"Model {model_path.name} doesn't have predict_proba, using predictions as probabilities")
    
    y_pred = model.predict(X)

    # Compute ranking metrics (primary)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Recall@TopK
    recall_topk = {
        f"recall_at_top_{k}": recall_at_top_k(y_true, y_pred_proba, k)
        for k in config.top_k_values
    }

    # Compute binary classification metrics (comparison)
    optimal_threshold, threshold_results = find_optimal_threshold(
        y_true, y_pred_proba, metric="f1"
    )
    binary_metrics = threshold_results["best_metrics"]

    metrics = {
        # Ranking metrics (primary)
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        **recall_topk,
        # Binary classification metrics (comparison)
        "binary_threshold": optimal_threshold,
        "binary_precision": binary_metrics["precision"],
        "binary_recall": binary_metrics["recall"],
        "binary_f1": binary_metrics["f1"],
        "binary_tp": binary_metrics["tp"],
        "binary_fp": binary_metrics["fp"],
        "binary_fn": binary_metrics["fn"],
        "binary_tn": binary_metrics["tn"],
        # Dataset info
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
    
    # Encode categorical columns before converting to numpy
    if "highest_level_reached" in df.columns:
        df = df.with_columns([
            pl.when(pl.col("highest_level_reached") == "A")
            .then(1)
            .when(pl.col("highest_level_reached") == "A+")
            .then(2)
            .when(pl.col("highest_level_reached") == "AA")
            .then(3)
            .when(pl.col("highest_level_reached") == "AAA")
            .then(4)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("highest_level_reached")
        ])
    
    split_df = df.filter(pl.col("split") == split)

    # Sample if too many
    if len(split_df) > max_samples:
        split_df = split_df.sample(max_samples, seed=config.random_seed)

    # Extract X and y - convert to float64, handling nulls and missing features
    def df_to_numpy(df_subset: pl.DataFrame, expected_features: list[str]) -> np.ndarray:
        """Convert Polars DataFrame to numpy array, handling nulls and missing features."""
        arrays = []
        for col in expected_features:
            if col in df_subset.columns:
                col_data = df_subset[col]
                if col_data.dtype in [pl.Float64, pl.Float32]:
                    arr = col_data.to_numpy()
                    arr = np.array([float(x) if x is not None else np.nan for x in arr])
                elif col_data.dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64]:
                    arr = col_data.to_numpy()
                    arr = np.array([float(x) if x is not None else np.nan for x in arr])
                else:
                    arr = np.array([float(x) if x is not None else np.nan for x in col_data.to_list()])
            else:
                # Feature missing - fill with zeros
                logger.warning(f"Feature '{col}' not found in DataFrame, filling with zeros")
                arr = np.zeros(len(df_subset), dtype=np.float64)
            arrays.append(arr)
        return np.column_stack(arrays).astype(np.float64)
    
    X = df_to_numpy(split_df, feature_names)
    
    # X should now have the correct number of features (matching feature_names length)
    # Verify it matches what model expects
    if hasattr(model, "n_features_in_"):
        expected_features = model.n_features_in_
        if X.shape[1] != expected_features:
            if X.shape[1] < expected_features:
                # Pad with zeros
                logger.warning(
                    f"SHAP: X has {X.shape[1]} features, model expects {expected_features}. "
                    f"Padding with zeros before preprocessing."
                )
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.column_stack([X, padding])
            else:
                # Truncate to expected features
                logger.warning(
                    f"SHAP: X has {X.shape[1]} features, model expects {expected_features}. "
                    f"Truncating to first {expected_features} features before preprocessing."
                )
                X = X[:, :expected_features]
    
    # Handle imputer if present - ensure feature count maintained
    if "imputer" in model_data:
        imputer = model_data["imputer"]
        if X.shape[1] == imputer.n_features_in_:
            X = imputer.transform(X)
        else:
            logger.warning(
                f"SHAP: Imputer expects {imputer.n_features_in_} features, "
                f"but X has {X.shape[1]}. Using fallback imputation."
            )
            from sklearn.impute import SimpleImputer
            fallback_imputer = SimpleImputer(strategy="median")
            X = fallback_imputer.fit_transform(X)
            # Re-align after imputation
            if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
                if X.shape[1] < model.n_features_in_:
                    padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
                    X = np.column_stack([X, padding])
                else:
                    X = X[:, :model.n_features_in_]
    elif np.isnan(X).any():
        # Impute missing values for SHAP
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
        # Re-align after imputation
        if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
            if X.shape[1] < model.n_features_in_:
                padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
                X = np.column_stack([X, padding])
            else:
                X = X[:, :model.n_features_in_]
    
    # Final check before SHAP
    if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
        logger.error(
            f"SHAP: Feature count still mismatched after all preprocessing "
            f"(X has {X.shape[1]} features, model expects {model.n_features_in_}). "
            f"Cannot proceed with SHAP."
        )
        return

    # Create SHAP explainer
    try:
        # Verify feature count matches before creating explainer
        if hasattr(model, "n_features_in_"):
            if X.shape[1] != model.n_features_in_:
                logger.error(
                    f"SHAP: Feature count mismatch after alignment "
                    f"(X has {X.shape[1]} features, model expects {model.n_features_in_}). "
                    f"Skipping SHAP."
                )
                return
        
        # Use auto feature perturbation (will use interventional if background dataset provided)
        explainer = shap.TreeExplainer(model, feature_perturbation="auto")
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

        # Generate waterfall plots for top predictions (highest probability All-Stars)
        if len(X) > 0:
            # Get predictions
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
            
            # Find top predictions (highest probability)
            top_k = min(5, len(X))
            top_indices = np.argsort(y_pred_proba)[-top_k:][::-1]
            
            waterfall_dir = config.figures_dir / "shap_waterfalls"
            waterfall_dir.mkdir(parents=True, exist_ok=True)
            
            # Get base value (expected value)
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[1]  # Positive class
            else:
                base_value = explainer.expected_value
            
            for idx in top_indices:
                try:
                    # Create Explanation object for single instance
                    explanation = shap.Explanation(
                        values=shap_values[idx],
                        base_values=base_value,
                        data=X[idx],
                        feature_names=feature_names[:len(shap_values[idx])]
                    )
                    
                    plt.figure(figsize=(10, 8))
                    shap.plots.waterfall(explanation, show=False, max_display=15)
                    waterfall_path = waterfall_dir / f"{model_path.stem}_top_{idx}.png"
                    plt.tight_layout()
                    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    logger.info(f"Saved waterfall plot to {waterfall_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate waterfall plot for index {idx}: {e}")

        # Generate dependence plots for top 5 features
        if len(shap_values.shape) == 2 and shap_values.shape[1] > 0:
            # Calculate mean absolute SHAP values to find top features
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_feature_indices = np.argsort(mean_abs_shap)[-5:][::-1]
            
            dependence_dir = config.figures_dir / "shap_dependence"
            dependence_dir.mkdir(parents=True, exist_ok=True)
            
            for feat_idx in top_feature_indices:
                try:
                    # Find feature that interacts most with this one
                    # Calculate interaction strength
                    interaction_strengths = []
                    for other_idx in range(len(feature_names)):
                        if other_idx != feat_idx:
                            # Simple interaction: correlation between SHAP values
                            corr = np.corrcoef(shap_values[:, feat_idx], shap_values[:, other_idx])[0, 1]
                            interaction_strengths.append((other_idx, abs(corr)))
                    
                    # Get most interacting feature
                    if interaction_strengths:
                        interaction_strengths.sort(key=lambda x: x[1], reverse=True)
                        interaction_idx = interaction_strengths[0][0]
                        
                        plt.figure(figsize=(10, 6))
                        shap.dependence_plot(
                            feat_idx,
                            shap_values,
                            X,
                            feature_names=feature_names[:X.shape[1]],
                            interaction_index=interaction_idx,
                            show=False
                        )
                        dep_path = dependence_dir / f"{model_path.stem}_{feature_names[feat_idx]}.png"
                        plt.tight_layout()
                        plt.savefig(dep_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        logger.info(f"Saved dependence plot to {dep_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate dependence plot for feature {feat_idx}: {e}")

    except Exception as e:
        logger.warning(f"Failed to generate SHAP plot: {e}")
        # Don't raise - SHAP is optional
        return


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

        # SHAP for tree-based models (non-critical, failures are logged but don't stop evaluation)
        # Check base model name (handle "_advanced" suffix)
        base_model_name = model_name.replace("_advanced", "").replace("_baseline", "")
        if base_model_name in ["xgboost", "lightgbm", "random_forest"]:
            try:
                plot_shap_importance(
                    model_path,
                    features_path,
                    split=split,
                    output_path=figures_dir / f"shap_{model_name}.png",
                )
            except Exception as e:
                logger.warning(f"SHAP plot generation failed for {model_name}: {e}")
                # Continue evaluation - SHAP is optional

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

        # Save metrics to CSV (exclude nested data like y_pred_proba, y_true)
        metrics_dict = {k: v for k, v in eval_result["metrics"].items() 
                       if not isinstance(v, (list, np.ndarray))}
        metrics_df = pl.DataFrame([metrics_dict])
        metrics_df.write_csv(tables_dir / f"metrics_{model_name}.csv")

    # Save summary (exclude nested data from metrics)
    summary_rows = []
    for model_name, metrics_dict in results.items():
        # Flatten metrics, excluding nested arrays/lists
        row = {"model": model_name}
        for k, v in metrics_dict.items():
            if isinstance(v, (list, np.ndarray)):
                continue  # Skip nested data
            # Convert numpy types to Python types
            if isinstance(v, (np.integer, np.floating)):
                row[k] = float(v)
            else:
                row[k] = v
        summary_rows.append(row)
    
    summary_path = tables_dir / "evaluation_summary.csv"
    if summary_rows:
        summary_df = pl.DataFrame(summary_rows)
        summary_df.write_csv(summary_path)
        
        # Log comparison summary
        logger.info("\n" + "=" * 80)
        logger.info("RANKING vs BINARY CLASSIFICATION COMPARISON")
        logger.info("=" * 80)
        for model_name, metrics_dict in results.items():
            if "pr_auc" in metrics_dict and "binary_f1" in metrics_dict:
                logger.info(f"\n{model_name}:")
                logger.info(f"  Ranking (PR-AUC): {metrics_dict['pr_auc']:.4f}")
                logger.info(f"  Binary (F1): {metrics_dict['binary_f1']:.4f} at threshold {metrics_dict.get('binary_threshold', 0.5):.2f}")
                logger.info(f"  Binary Precision: {metrics_dict['binary_precision']:.2%}")
                logger.info(f"  Binary Recall: {metrics_dict['binary_recall']:.2%}")
                logger.info(f"  Binary False Positives: {metrics_dict.get('binary_fp', 0)}")
    else:
        logger.warning("No summary data to save")

    logger.info(f"Saved evaluation results to {tables_dir}")

    return results

