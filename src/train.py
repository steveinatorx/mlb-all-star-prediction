"""Model training module."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import config

try:
    from src.mlflow_tracking import log_model_training
except ImportError:
    log_model_training = None
    logger.warning("MLflow tracking not available")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    logger.warning("LightGBM not available, skipping LGBM model")

try:
    from pygam import LogisticGAM
except ImportError:
    LogisticGAM = None
    logger.warning("pygam not available, skipping GAM model")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
except ImportError:
    SMOTE = None
    ADASYN = None
    logger.warning("imbalanced-learn not available, SMOTE/ADASYN disabled")


def load_features(
    features_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load features and split into train/val/test.

    Args:
        features_path: Path to features file

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    features_path = features_path or config.features_data_dir / "features.parquet"

    logger.info(f"Loading features from {features_path}")
    df = pl.read_parquet(features_path)

    # Select feature columns (exclude player_id, is_all_star, split)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["player_id", "is_all_star", "split"]
    ]

    # Encode categorical columns before converting to numpy
    # highest_level_reached: ordinal encoding (A=1, A+=2, AA=3, AAA=4, Unknown=0)
    if "highest_level_reached" in feature_cols:
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
    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")
    test_df = df.filter(pl.col("split") == "test")

    # Extract X and y - convert to float64, handling nulls
    # Convert each column separately to handle nulls properly
    def df_to_numpy(df_subset: pl.DataFrame) -> np.ndarray:
        """Convert Polars DataFrame to numpy array, handling nulls."""
        arrays = []
        for col in feature_cols:
            col_data = df_subset[col]
            # Convert to numpy, handling nulls
            if col_data.dtype in [pl.Float64, pl.Float32]:
                arr = col_data.to_numpy()
                # Replace None with np.nan
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            elif col_data.dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64]:
                arr = col_data.to_numpy()
                # Replace None with np.nan, convert to float
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            else:
                # Fallback: convert to float, handling nulls
                arr = np.array([float(x) if x is not None else np.nan for x in col_data.to_list()])
            arrays.append(arr)
        return np.column_stack(arrays).astype(np.float64)
    
    X_train = df_to_numpy(train_df.select(feature_cols))
    y_train = train_df["is_all_star"].to_numpy().astype(int)

    X_val = df_to_numpy(val_df.select(feature_cols))
    y_val = val_df["is_all_star"].to_numpy().astype(int)

    X_test = df_to_numpy(test_df.select(feature_cols))
    y_test = test_df["is_all_star"].to_numpy().astype(int)

    logger.info(
        f"Loaded features: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    logger.info(f"Class balance - train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Train logistic regression with regularization.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names

    Returns:
        Dictionary with model, scaler, and metrics
    """
    logger.info("Training Logistic Regression")

    # Handle missing values: impute with median for numeric features
    from sklearn.impute import SimpleImputer
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Train model with L2 regularization
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        max_iter=1000,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = model.predict(X_val_scaled)

    # Metrics
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    # Coefficients and p-values (approximate)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    logger.info(f"Logistic Regression - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Train Random Forest classifier."""
    logger.info("Training Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Random Forest - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
    }


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Train XGBoost classifier."""
    logger.info("Training XGBoost")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=config.random_seed,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"XGBoost - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
    }


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> Optional[dict]:
    """Train LightGBM classifier."""
    if LGBMClassifier is None:
        return None

    logger.info("Training LightGBM")

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=config.random_seed,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"LightGBM - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
    }


def train_gam(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> Optional[dict]:
    """Train GAM model (optional)."""
    if LogisticGAM is None:
        return None

    logger.info("Training GAM (this may take a while)")

    # GAM can be slow, so limit features or samples if needed
    n_features = min(10, X_train.shape[1])  # Limit to 10 features for GAM
    X_train_subset = X_train[:, :n_features]
    X_val_subset = X_val[:, :n_features]

    model = LogisticGAM()
    model.fit(X_train_subset, y_train)

    y_pred_proba = model.predict_proba(X_val_subset)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"GAM - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "feature_names": feature_names[:n_features],
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
    }


def train_all_models(
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Train all models and save them.

    Args:
        features_path: Path to features file
        output_dir: Output directory for models

    Returns:
        Dictionary mapping model names to saved paths
    """
    output_dir = output_dir or config.experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training all models")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_features(
        features_path
    )

    # Train models
    results = {}

    # Logistic Regression
    lr_result = train_logistic_regression(
        X_train, y_train, X_val, y_val, feature_names
    )
    lr_path = output_dir / "logistic_regression.joblib"
    joblib.dump(
        {
            "model": lr_result["model"],
            "scaler": lr_result["scaler"],
            "feature_names": lr_result["feature_names"],
            "coefficients": lr_result["coefficients"],
            "intercept": lr_result["intercept"],
        },
        lr_path,
    )
    results["logistic_regression"] = {
        "model_path": lr_path,
        "metrics": {
            "pr_auc": lr_result["pr_auc"],
            "roc_auc": lr_result["roc_auc"],
        },
        "coefficients": lr_result["coefficients"],
        "intercept": lr_result["intercept"],
    }

    # Log to MLflow
    if log_model_training:
        try:
            log_model_training(
                model_name="logistic_regression_baseline",
                model=lr_result["model"],
                feature_names=feature_names,
                metrics={"pr_auc": lr_result["pr_auc"], "roc_auc": lr_result["roc_auc"]},
                params={"C": 1.0, "penalty": "l2"},
                tags={"technique": "baseline", "features": "base"},
                model_path=lr_path,
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    # Random Forest
    rf_result = train_random_forest(X_train, y_train, X_val, y_val, feature_names)
    rf_path = output_dir / "random_forest.joblib"
    joblib.dump(
        {
            "model": rf_result["model"],
            "feature_names": rf_result["feature_names"],
        },
        rf_path,
    )
    results["random_forest"] = {
        "model_path": rf_path,
        "metrics": {
            "pr_auc": rf_result["pr_auc"],
            "roc_auc": rf_result["roc_auc"],
        },
    }

    # Log to MLflow
    if log_model_training:
        try:
            log_model_training(
                model_name="random_forest_baseline",
                model=rf_result["model"],
                feature_names=feature_names,
                metrics={"pr_auc": rf_result["pr_auc"], "roc_auc": rf_result["roc_auc"]},
                params={"n_estimators": 100, "max_depth": 10},
                tags={"technique": "baseline", "features": "base"},
                model_path=rf_path,
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    # XGBoost
    xgb_result = train_xgboost(X_train, y_train, X_val, y_val, feature_names)
    xgb_path = output_dir / "xgboost.joblib"
    joblib.dump(
        {
            "model": xgb_result["model"],
            "feature_names": xgb_result["feature_names"],
        },
        xgb_path,
    )
    results["xgboost"] = {
        "model_path": xgb_path,
        "metrics": {
            "pr_auc": xgb_result["pr_auc"],
            "roc_auc": xgb_result["roc_auc"],
        },
    }

    # Log to MLflow
    if log_model_training:
        try:
            log_model_training(
                model_name="xgboost_baseline",
                model=xgb_result["model"],
                feature_names=feature_names,
                metrics={"pr_auc": xgb_result["pr_auc"], "roc_auc": xgb_result["roc_auc"]},
                params={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                tags={"technique": "baseline", "features": "base"},
                model_path=xgb_path,
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    # LightGBM (optional)
    lgbm_result = train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
    if lgbm_result:
        lgbm_path = output_dir / "lightgbm.joblib"
        joblib.dump(
            {
                "model": lgbm_result["model"],
                "feature_names": lgbm_result["feature_names"],
            },
            lgbm_path,
        )
        results["lightgbm"] = {
            "model_path": lgbm_path,
            "metrics": {
                "pr_auc": lgbm_result["pr_auc"],
                "roc_auc": lgbm_result["roc_auc"],
            },
        }

    # GAM (optional)
    gam_result = train_gam(X_train, y_train, X_val, y_val, feature_names)
    if gam_result:
        gam_path = output_dir / "gam.joblib"
        joblib.dump(
            {
                "model": gam_result["model"],
                "feature_names": gam_result["feature_names"],
            },
            gam_path,
        )
        results["gam"] = {
            "model_path": gam_path,
            "metrics": {
                "pr_auc": gam_result["pr_auc"],
                "roc_auc": gam_result["roc_auc"],
            },
        }

    # Save results summary
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved models and results to {output_dir}")

    return results

