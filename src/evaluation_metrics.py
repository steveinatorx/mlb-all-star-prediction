"""Additional evaluation metrics: learning curves, calibration curves, permutation importance, partial dependence plots."""

import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.utils import resample

from src.config import config

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def plot_learning_curves(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    output_path: Optional[Path] = None,
    train_sizes: Optional[np.ndarray] = None,
    cv: int = 5,
) -> None:
    """
    Plot learning curves to show overfitting/underfitting.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model
        output_path: Path to save the plot
        train_sizes: Array of training set sizes to evaluate
        cv: Number of cross-validation folds
    """
    logger.info(f"Generating learning curves for {model_name}")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    # Combine train and val for learning curve
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    # Calculate learning curves
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_combined,
        y_combined,
        train_sizes=train_sizes,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        random_state=config.random_seed,
    )

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Training Score")
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    ax.plot(train_sizes_abs, val_mean, "o-", color="red", label="Validation Score")
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color="red",
    )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("PR-AUC Score")
    ax.set_title(f"Learning Curves: {model_name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_path or config.figures_dir / f"learning_curves_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved learning curves to {output_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    output_path: Optional[Path] = None,
    n_bins: int = 10,
) -> None:
    """
    Plot calibration curve to check probability calibration.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        output_path: Path to save the plot
        n_bins: Number of bins for calibration curve
    """
    logger.info(f"Generating calibration curve for {model_name}")

    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve: {model_name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_path or config.figures_dir / f"calibration_curve_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved calibration curve to {output_path}")


def calculate_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
) -> dict:
    """
    Calculate permutation importance for features.

    Args:
        model: Trained model
        X: Features
        y: Labels
        feature_names: List of feature names
        model_name: Name of the model
        n_repeats: Number of times to permute each feature
        random_state: Random state for reproducibility

    Returns:
        Dictionary with importance scores and feature names
    """
    logger.info(f"Calculating permutation importance for {model_name}")

    # Calculate permutation importance
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state or config.random_seed,
        scoring="average_precision",
        n_jobs=-1,
    )

    # Sort by importance
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]

    # Create results dictionary
    importance_dict = {
        "feature_names": [feature_names[i] for i in indices],
        "importances": importances[indices].tolist(),
        "importances_std": result.importances_std[indices].tolist(),
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]

    ax.barh(
        range(top_n),
        importances[top_indices],
        xerr=result.importances_std[top_indices],
        align="center",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Permutation Importance: {model_name}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    output_path = config.figures_dir / f"permutation_importance_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved permutation importance to {output_path}")

    # Save to JSON
    json_path = config.tables_dir / f"permutation_importance_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(importance_dict, f, indent=2)

    return importance_dict


def plot_partial_dependence(
    model,
    X: np.ndarray,
    feature_names: list[str],
    model_name: str,
    top_n_features: int = 5,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Plot partial dependence plots for top features.

    Args:
        model: Trained model
        X: Features
        feature_names: List of feature names
        model_name: Name of the model
        top_n_features: Number of top features to plot
        output_dir: Directory to save plots
    """
    logger.info(f"Generating partial dependence plots for {model_name}")

    # Get top features (using feature importance if available)
    if hasattr(model, "feature_importances_"):
        top_indices = np.argsort(model.feature_importances_)[-top_n_features:][::-1]
    else:
        # Fallback: use first N features
        top_indices = list(range(min(top_n_features, len(feature_names))))

    top_features = [feature_names[i] for i in top_indices]
    top_feature_indices = top_indices

    output_dir = output_dir or config.figures_dir / "partial_dependence"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot each top feature
    for feat_idx, feat_name in zip(top_feature_indices, top_features):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model,
                X,
                features=[feat_idx],
                feature_names=feature_names,
                ax=ax,
            )
            ax.set_title(f"Partial Dependence: {feat_name} ({model_name})")
            plt.tight_layout()

            output_path = output_dir / f"pd_{model_name}_{feat_name.replace('/', '_')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved partial dependence plot for {feat_name}")
        except Exception as e:
            logger.warning(f"Failed to generate partial dependence plot for {feat_name}: {e}")

    # Generate combined plot for top features
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=top_feature_indices[:min(4, len(top_feature_indices))],
            feature_names=feature_names,
            ax=ax,
            n_cols=2,
        )
        ax.set_title(f"Partial Dependence: Top Features ({model_name})")
        plt.tight_layout()

        output_path = output_dir / f"pd_combined_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved combined partial dependence plot")
    except Exception as e:
        logger.warning(f"Failed to generate combined partial dependence plot: {e}")


def generate_all_evaluation_metrics(
    model_path: Path,
    features_path: Optional[Path] = None,
    split: str = "test",
) -> None:
    """
    Generate all additional evaluation metrics for a model.

    Args:
        model_path: Path to saved model
        features_path: Path to features file
        split: Which split to use
    """
    logger.info(f"Generating all evaluation metrics for {model_path.name}")

    # Load model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Load features
    features_path = features_path or config.features_data_dir / "features_with_interactions.parquet"
    df = pl.read_parquet(features_path)

    # Encode categorical
    if "highest_level_reached" in df.columns:
        df = df.with_columns([
            pl.when(pl.col("highest_level_reached") == "A").then(1)
            .when(pl.col("highest_level_reached") == "A+").then(2)
            .when(pl.col("highest_level_reached") == "AA").then(3)
            .when(pl.col("highest_level_reached") == "AAA").then(4)
            .otherwise(0)
            .cast(pl.Int64)
            .alias("highest_level_reached")
        ])

    # Get splits
    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")
    test_df = df.filter(pl.col("split") == split)

    # Convert to numpy
    def df_to_numpy(df_subset: pl.DataFrame) -> np.ndarray:
        arrays = []
        for col in feature_names:
            if col in df_subset.columns:
                arr = df_subset[col].to_numpy()
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            else:
                arr = np.zeros(len(df_subset), dtype=np.float64)
            arrays.append(arr)
        return np.column_stack(arrays).astype(np.float64)

    X_train = df_to_numpy(train_df)
    X_val = df_to_numpy(val_df)
    X_test = df_to_numpy(test_df)
    y_train = train_df["is_all_star"].to_numpy().astype(int)
    y_val = val_df["is_all_star"].to_numpy().astype(int)
    y_test = test_df["is_all_star"].to_numpy().astype(int)

    # Impute missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scale if needed (for logistic regression)
    if "scaler" in model_data:
        scaler = model_data["scaler"]
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model_name = model_path.stem

    # Learning curves
    try:
        plot_learning_curves(model, X_train, y_train, X_val, y_val, model_name)
    except Exception as e:
        logger.warning(f"Failed to generate learning curves: {e}")

    # Calibration curve
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        plot_calibration_curve(y_test, y_pred_proba, model_name)
    except Exception as e:
        logger.warning(f"Failed to generate calibration curve: {e}")

    # Permutation importance
    try:
        calculate_permutation_importance(model, X_test, y_test, feature_names, model_name)
    except Exception as e:
        logger.warning(f"Failed to calculate permutation importance: {e}")

    # Partial dependence plots
    try:
        plot_partial_dependence(model, X_test, feature_names, model_name)
    except Exception as e:
        logger.warning(f"Failed to generate partial dependence plots: {e}")

    # Bootstrap confidence intervals
    try:
        bootstrap_results = calculate_bootstrap_confidence_intervals(
            y_test, y_pred_proba, n_bootstrap=1000
        )
        # Save to JSON
        bootstrap_path = config.tables_dir / f"bootstrap_ci_{model_name}.json"
        with open(bootstrap_path, "w") as f:
            json.dump(bootstrap_results, f, indent=2)
        logger.info(f"Saved bootstrap confidence intervals to {bootstrap_path}")
    except Exception as e:
        logger.warning(f"Failed to calculate bootstrap confidence intervals: {e}")

    # Curves with confidence intervals
    try:
        plot_curves_with_confidence_intervals(
            y_test, y_pred_proba, model_name, curve_type="pr", n_bootstrap=1000
        )
        plot_curves_with_confidence_intervals(
            y_test, y_pred_proba, model_name, curve_type="roc", n_bootstrap=1000
        )
    except Exception as e:
        logger.warning(f"Failed to generate curves with confidence intervals: {e}")

    logger.info(f"Completed evaluation metrics for {model_name}")


def calculate_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> dict:
    """
    Calculate bootstrap confidence intervals for metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random state for reproducibility

    Returns:
        Dictionary with metrics and confidence intervals
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    logger.info(f"Calculating bootstrap confidence intervals (n={n_bootstrap})")

    if random_state is None:
        random_state = config.random_seed

    np.random.seed(random_state)

    # Bootstrap samples
    pr_aucs = []
    roc_aucs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(
            np.arange(len(y_true)),
            n_samples=len(y_true),
            random_state=random_state + _,
        )
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]

        # Calculate metrics
        try:
            pr_auc = average_precision_score(y_true_boot, y_pred_proba_boot)
            roc_auc = roc_auc_score(y_true_boot, y_pred_proba_boot)
            pr_aucs.append(pr_auc)
            roc_aucs.append(roc_auc)
        except ValueError:
            # Skip if all labels are same class
            continue

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    pr_auc_ci = (
        np.percentile(pr_aucs, lower_percentile),
        np.percentile(pr_aucs, upper_percentile),
    )
    roc_auc_ci = (
        np.percentile(roc_aucs, lower_percentile),
        np.percentile(roc_aucs, upper_percentile),
    )

    # Calculate mean metrics
    pr_auc_mean = np.mean(pr_aucs)
    roc_auc_mean = np.mean(roc_aucs)

    results = {
        "pr_auc": {
            "mean": float(pr_auc_mean),
            "ci_lower": float(pr_auc_ci[0]),
            "ci_upper": float(pr_auc_ci[1]),
            "std": float(np.std(pr_aucs)),
        },
        "roc_auc": {
            "mean": float(roc_auc_mean),
            "ci_lower": float(roc_auc_ci[0]),
            "ci_upper": float(roc_auc_ci[1]),
            "std": float(np.std(roc_aucs)),
        },
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }

    return results


def plot_curves_with_confidence_intervals(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    curve_type: str = "pr",  # "pr" or "roc"
    n_bootstrap: int = 1000,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot precision-recall or ROC curves with bootstrap confidence intervals.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        curve_type: Type of curve ("pr" or "roc")
        n_bootstrap: Number of bootstrap samples
        output_path: Path to save the plot
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    logger.info(f"Generating {curve_type.upper()} curve with confidence intervals for {model_name}")

    random_state = random_state or config.random_seed
    np.random.seed(random_state)

    # Bootstrap samples
    curves = []
    for i in range(n_bootstrap):
        indices = resample(
            np.arange(len(y_true)),
            n_samples=len(y_true),
            random_state=random_state + i,
        )
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]

        try:
            if curve_type == "pr":
                precision, recall, thresholds = precision_recall_curve(
                    y_true_boot, y_pred_proba_boot
                )
                curves.append((recall, precision))
            else:  # roc
                fpr, tpr, thresholds = roc_curve(y_true_boot, y_pred_proba_boot)
                curves.append((fpr, tpr))
        except ValueError:
            continue

    # Calculate mean and confidence intervals
    if curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        x_axis = recall
        y_axis = precision
        x_label = "Recall"
        y_label = "Precision"
        title = f"Precision-Recall Curve with CI: {model_name}"
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        x_axis = fpr
        y_axis = tpr
        x_label = "False Positive Rate"
        y_label = "True Positive Rate"
        title = f"ROC Curve with CI: {model_name}"

    # Interpolate curves to common x-axis
    x_common = np.linspace(0, 1, 100)
    y_interpolated = []

    for x_boot, y_boot in curves:
        y_interp = np.interp(x_common, x_boot, y_boot)
        y_interpolated.append(y_interp)

    y_interpolated = np.array(y_interpolated)
    y_mean = np.mean(y_interpolated, axis=0)
    y_lower = np.percentile(y_interpolated, 2.5, axis=0)
    y_upper = np.percentile(y_interpolated, 97.5, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_axis, y_axis, "b-", label=f"{model_name}", linewidth=2)
    ax.fill_between(
        x_common, y_lower, y_upper, alpha=0.3, color="blue", label="95% Confidence Interval"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        output_path
        or config.figures_dir / f"{curve_type}_curve_ci_{model_name}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved {curve_type.upper()} curve with CI to {output_path}")

