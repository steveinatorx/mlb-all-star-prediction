"""Advanced training functions for imbalanced data."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from loguru import logger

from src.config import config
from src.train import load_features

try:
    from imblearn.over_sampling import SMOTE, ADASYN
except ImportError:
    SMOTE = None
    ADASYN = None
    logger.warning("imbalanced-learn not available, SMOTE/ADASYN disabled")


def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Calculate class weights based on class imbalance.

    Args:
        y_train: Training labels

    Returns:
        Dictionary mapping class to weight
    """
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    n_positives = y_train.sum()
    n_negatives = n_samples - n_positives

    # Calculate inverse frequency weights
    weight_negative = n_samples / (n_classes * n_negatives)
    weight_positive = n_samples / (n_classes * n_positives)

    class_weights = {0: weight_negative, 1: weight_positive}

    logger.info(
        f"Class weights - Negative: {weight_negative:.2f}, Positive: {weight_positive:.2f}"
    )
    logger.info(
        f"Class distribution - Negative: {n_negatives}, Positive: {n_positives}"
    )

    return class_weights


def apply_smote(
    X_train: np.ndarray, y_train: np.ndarray, k_neighbors: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to oversample minority class.

    Args:
        X_train: Training features
        y_train: Training labels
        k_neighbors: Number of nearest neighbors for SMOTE

    Returns:
        Resampled X_train, y_train
    """
    if SMOTE is None:
        raise ImportError("imbalanced-learn not installed. Install with: pip install imbalanced-learn")

    n_positives = y_train.sum()
    n_negatives = len(y_train) - n_positives

    # Adjust k_neighbors if we have fewer positive samples than k_neighbors
    k_neighbors = min(k_neighbors, n_positives - 1)
    if k_neighbors < 1:
        logger.warning("Not enough positive samples for SMOTE, skipping")
        return X_train, y_train

    logger.info(f"Applying SMOTE with k_neighbors={k_neighbors}")
    logger.info(f"Before SMOTE - Negative: {n_negatives}, Positive: {n_positives}")

    smote = SMOTE(
        k_neighbors=k_neighbors,
        random_state=config.random_seed,
        sampling_strategy="auto",  # Balance to 50/50
    )

    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        n_positives_after = y_train_resampled.sum()
        n_negatives_after = len(y_train_resampled) - n_positives_after

        logger.info(
            f"After SMOTE - Negative: {n_negatives_after}, Positive: {n_positives_after}"
        )
        logger.info(
            f"SMOTE created {n_positives_after - n_positives} synthetic All-Star samples"
        )

        return X_train_resampled, y_train_resampled
    except Exception as e:
        logger.error(f"SMOTE failed: {e}, using original data")
        return X_train, y_train


def train_logistic_regression_advanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    use_class_weights: bool = True,
    use_smote: bool = False,
) -> dict:
    """
    Train logistic regression with advanced imbalanced data techniques.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        use_class_weights: Whether to use class weights
        use_smote: Whether to use SMOTE oversampling

    Returns:
        Dictionary with model, scaler, and metrics
    """
    logger.info("Training Logistic Regression (Advanced)")

    # IMPORTANT: Impute missing values BEFORE SMOTE (SMOTE doesn't accept NaN)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Handle missing values first
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    # Apply SMOTE if requested (AFTER imputation)
    if use_smote:
        X_train_imputed, y_train = apply_smote(X_train_imputed, y_train, k_neighbors=config.smote_k_neighbors)

    # Calculate class weights if requested
    class_weight = None
    if use_class_weights:
        class_weights_dict = calculate_class_weights(y_train)
        class_weight = class_weights_dict

    # Scale features (AFTER SMOTE, so we scale the resampled data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Train model with class weights
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=config.random_seed,
        solver="lbfgs",
        class_weight=class_weight,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

    # Metrics
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    logger.info(
        f"Logistic Regression (Advanced) - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}"
    )

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": feature_names,
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "techniques": {
            "class_weights": use_class_weights,
            "smote": use_smote,
        },
    }


def train_random_forest_advanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    use_class_weights: bool = True,
    use_smote: bool = False,
) -> dict:
    """Train Random Forest with advanced imbalanced data techniques."""
    logger.info("Training Random Forest (Advanced)")

    # IMPORTANT: Impute missing values BEFORE SMOTE (SMOTE doesn't accept NaN)
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Handle missing values first
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    # Apply SMOTE if requested (AFTER imputation)
    if use_smote:
        X_train_imputed, y_train = apply_smote(X_train_imputed, y_train, k_neighbors=config.smote_k_neighbors)

    # Calculate class weights if requested
    class_weight = None
    if use_class_weights:
        class_weights_dict = calculate_class_weights(y_train)
        class_weight = class_weights_dict

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
        class_weight=class_weight,
    )
    model.fit(X_train_imputed, y_train)

    y_pred_proba = model.predict_proba(X_val_imputed)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Random Forest (Advanced) - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "techniques": {
            "class_weights": use_class_weights,
            "smote": use_smote,
        },
    }


def train_xgboost_advanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    use_class_weights: bool = True,
    use_smote: bool = False,
) -> dict:
    """Train XGBoost with advanced imbalanced data techniques."""
    logger.info("Training XGBoost (Advanced)")

    # IMPORTANT: Impute missing values BEFORE SMOTE (SMOTE doesn't accept NaN)
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier

    # Handle missing values first
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    # Apply SMOTE if requested (AFTER imputation)
    if use_smote:
        X_train_imputed, y_train = apply_smote(X_train_imputed, y_train, k_neighbors=config.smote_k_neighbors)

    # Calculate class weights if requested
    scale_pos_weight = None
    if use_class_weights:
        class_weights_dict = calculate_class_weights(y_train)
        # XGBoost uses scale_pos_weight instead of class_weight
        scale_pos_weight = class_weights_dict[1] / class_weights_dict[0]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=config.random_seed,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train_imputed, y_train)

    y_pred_proba = model.predict_proba(X_val_imputed)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"XGBoost (Advanced) - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "techniques": {
            "class_weights": use_class_weights,
            "smote": use_smote,
        },
    }


def train_lightgbm_advanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    use_class_weights: bool = True,
    use_smote: bool = False,
) -> dict | None:
    """Train LightGBM with advanced imbalanced data techniques."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None

    logger.info("Training LightGBM (Advanced)")

    # IMPORTANT: Impute missing values BEFORE SMOTE (SMOTE doesn't accept NaN)
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Handle missing values first
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    # Apply SMOTE if requested (AFTER imputation)
    if use_smote:
        X_train_imputed, y_train = apply_smote(X_train_imputed, y_train, k_neighbors=config.smote_k_neighbors)

    # Calculate class weights if requested
    class_weight = None
    if use_class_weights:
        class_weights_dict = calculate_class_weights(y_train)
        class_weight = class_weights_dict

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=config.random_seed,
        class_weight=class_weight,
        verbose=-1,
    )
    model.fit(X_train_imputed, y_train)

    y_pred_proba = model.predict_proba(X_val_imputed)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"LightGBM (Advanced) - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "techniques": {
            "class_weights": use_class_weights,
            "smote": use_smote,
        },
    }


def train_all_models_advanced(
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_class_weights: bool = True,
    use_smote: bool = True,
) -> dict[str, Path]:
    """
    Train all models with advanced imbalanced data techniques.

    Args:
        features_path: Path to features file
        output_dir: Output directory for models
        use_class_weights: Whether to use class weights
        use_smote: Whether to use SMOTE oversampling

    Returns:
        Dictionary mapping model names to saved paths
    """
    output_dir = output_dir or config.experiments_dir / "advanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training all models with advanced techniques")
    logger.info(f"  - Class weights: {use_class_weights}")
    logger.info(f"  - SMOTE: {use_smote}")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_features(
        features_path
    )

    # Train models
    results = {}

    # Logistic Regression
    lr_result = train_logistic_regression_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=use_class_weights,
        use_smote=use_smote,
    )
    lr_path = output_dir / "logistic_regression_advanced.joblib"
    joblib.dump(
        {
            "model": lr_result["model"],
            "scaler": lr_result["scaler"],
            "imputer": lr_result["imputer"],
            "feature_names": lr_result["feature_names"],
            "coefficients": lr_result["coefficients"],
            "intercept": lr_result["intercept"],
            "techniques": lr_result["techniques"],
        },
        lr_path,
    )
    results["logistic_regression_advanced"] = {
        "model_path": lr_path,
        "metrics": {
            "pr_auc": lr_result["pr_auc"],
            "roc_auc": lr_result["roc_auc"],
        },
        "coefficients": lr_result["coefficients"],
        "intercept": lr_result["intercept"],
        "techniques": lr_result["techniques"],
    }

    # Random Forest
    rf_result = train_random_forest_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=use_class_weights,
        use_smote=use_smote,
    )
    rf_path = output_dir / "random_forest_advanced.joblib"
    joblib.dump(
        {
            "model": rf_result["model"],
            "feature_names": rf_result["feature_names"],
            "techniques": rf_result["techniques"],
        },
        rf_path,
    )
    results["random_forest_advanced"] = {
        "model_path": rf_path,
        "metrics": {
            "pr_auc": rf_result["pr_auc"],
            "roc_auc": rf_result["roc_auc"],
        },
        "techniques": rf_result["techniques"],
    }

    # XGBoost
    xgb_result = train_xgboost_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=use_class_weights,
        use_smote=use_smote,
    )
    xgb_path = output_dir / "xgboost_advanced.joblib"
    joblib.dump(
        {
            "model": xgb_result["model"],
            "imputer": xgb_result["imputer"],
            "feature_names": xgb_result["feature_names"],
            "techniques": xgb_result["techniques"],
        },
        xgb_path,
    )
    results["xgboost_advanced"] = {
        "model_path": xgb_path,
        "metrics": {
            "pr_auc": xgb_result["pr_auc"],
            "roc_auc": xgb_result["roc_auc"],
        },
        "techniques": xgb_result["techniques"],
    }

    # LightGBM (optional)
    lgbm_result = train_lightgbm_advanced(
        X_train, y_train, X_val, y_val, feature_names,
        use_class_weights=use_class_weights,
        use_smote=use_smote,
    )
    if lgbm_result:
        lgbm_path = output_dir / "lightgbm_advanced.joblib"
        joblib.dump(
            {
                "model": lgbm_result["model"],
                "imputer": lgbm_result["imputer"],
                "feature_names": lgbm_result["feature_names"],
                "techniques": lgbm_result["techniques"],
            },
            lgbm_path,
        )
        results["lightgbm_advanced"] = {
            "model_path": lgbm_path,
            "metrics": {
                "pr_auc": lgbm_result["pr_auc"],
                "roc_auc": lgbm_result["roc_auc"],
            },
            "techniques": lgbm_result["techniques"],
        }

    # Save results summary
    results_path = output_dir / "training_results_advanced.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved advanced models and results to {output_dir}")

    return results
