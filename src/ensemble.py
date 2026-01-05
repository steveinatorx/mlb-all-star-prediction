"""Ensemble methods for combining multiple models."""

import joblib
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import config
from src.train import load_features

try:
    from src.mlflow_tracking import log_model_training
except ImportError:
    log_model_training = None
    logger.warning("MLflow tracking not available")


def load_model(model_path: Path):
    """Load a saved model."""
    model_data = joblib.load(model_path)
    return model_data


def voting_ensemble(
    model_paths: list[Path],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    voting: str = "soft",
) -> dict:
    """
    Create a voting ensemble from multiple models.

    Args:
        model_paths: List of paths to saved models
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        voting: "hard" or "soft" voting

    Returns:
        Dictionary with ensemble model and metrics
    """
    logger.info(f"Creating {voting} voting ensemble from {len(model_paths)} models")

    # Load models
    estimators = []
    for model_path in model_paths:
        model_data = load_model(model_path)
        model = model_data["model"]
        model_name = model_path.stem

        # Create a wrapper that handles preprocessing
        class ModelWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, model, model_data, name):
                self.model = model
                self.model_data = model_data
                self.name = name

            def fit(self, X, y):
                # VotingClassifier will call fit, but we already have trained models
                return self

            def predict_proba(self, X):
                # Handle feature alignment
                expected_features = len(self.model_data.get("feature_names", []))
                if hasattr(self.model, "n_features_in_"):
                    expected_features = self.model.n_features_in_

                # Align features
                if X.shape[1] != expected_features:
                    if X.shape[1] < expected_features:
                        padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                        X = np.column_stack([X, padding])
                    else:
                        X = X[:, :expected_features]

                # Handle imputation if needed
                if "imputer" in self.model_data:
                    imputer = self.model_data["imputer"]
                    if X.shape[1] == imputer.n_features_in_:
                        X = imputer.transform(X)
                    else:
                        from sklearn.impute import SimpleImputer
                        fallback_imputer = SimpleImputer(strategy="median")
                        X = fallback_imputer.fit_transform(X)
                        # Re-align after imputation
                        if X.shape[1] != expected_features:
                            if X.shape[1] < expected_features:
                                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                                X = np.column_stack([X, padding])
                            else:
                                X = X[:, :expected_features]

                # Handle scaling if needed (for logistic regression)
                if "scaler" in self.model_data:
                    scaler = self.model_data["scaler"]
                    if X.shape[1] == scaler.n_features_in_:
                        X = scaler.transform(X)

                return self.model.predict_proba(X)

            def predict(self, X):
                return self.model.predict(X)

        wrapper = ModelWrapper(model, model_data, model_name)
        estimators.append((model_name, wrapper))

    # Create voting ensemble
    ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=config.n_jobs)
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Voting Ensemble ({voting}) - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "model": ensemble,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "ensemble_type": f"voting_{voting}",
        "base_models": [str(p) for p in model_paths],
    }


def stacking_ensemble(
    model_paths: list[Path],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    meta_learner=None,
) -> dict:
    """
    Create a stacking ensemble with a meta-learner.

    Args:
        model_paths: List of paths to saved models
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        meta_learner: Meta-learner model (default: LogisticRegression)

    Returns:
        Dictionary with ensemble model and metrics
    """
    logger.info(f"Creating stacking ensemble from {len(model_paths)} models")

    if meta_learner is None:
        meta_learner = LogisticRegression(random_state=config.random_seed, max_iter=1000)

    # Load models
    base_models = []
    for model_path in model_paths:
        model_data = load_model(model_path)
        base_models.append((model_path.stem, model_data))

    # Generate base model predictions for training meta-learner
    def get_predictions(model_data, X):
        model = model_data["model"]

        # Handle feature alignment
        expected_features = len(model_data.get("feature_names", []))
        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_

        # Align features
        if X.shape[1] != expected_features:
            if X.shape[1] < expected_features:
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.column_stack([X, padding])
            else:
                X = X[:, :expected_features]

        # Handle imputation
        if "imputer" in model_data:
            imputer = model_data["imputer"]
            if X.shape[1] == imputer.n_features_in_:
                X = imputer.transform(X)
            else:
                from sklearn.impute import SimpleImputer
                fallback_imputer = SimpleImputer(strategy="median")
                X = fallback_imputer.fit_transform(X)
                # Re-align after imputation
                if X.shape[1] != expected_features:
                    if X.shape[1] < expected_features:
                        padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                        X = np.column_stack([X, padding])
                    else:
                        X = X[:, :expected_features]

        # Handle scaling
        if "scaler" in model_data:
            scaler = model_data["scaler"]
            if X.shape[1] == scaler.n_features_in_:
                X = scaler.transform(X)

        return model.predict_proba(X)[:, 1]

    # Generate meta-features
    X_meta_train = np.column_stack(
        [get_predictions(model_data, X_train) for _, model_data in base_models]
    )
    X_meta_val = np.column_stack(
        [get_predictions(model_data, X_val) for _, model_data in base_models]
    )

    # Train meta-learner
    meta_learner.fit(X_meta_train, y_train)

    # Evaluate
    y_pred_proba = meta_learner.predict_proba(X_meta_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Stacking Ensemble - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

    return {
        "base_models": base_models,
        "meta_learner": meta_learner,
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "ensemble_type": "stacking",
        "base_model_paths": [str(p) for p in model_paths],
    }


def blending_ensemble(
    model_paths: list[Path],
    weights: Optional[list[float]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Create a blending ensemble (weighted average of predictions).

    Args:
        model_paths: List of paths to saved models
        weights: Weights for each model (if None, use equal weights)
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names

    Returns:
        Dictionary with ensemble predictions and metrics
    """
    logger.info(f"Creating blending ensemble from {len(model_paths)} models")

    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    if len(weights) != len(model_paths):
        raise ValueError("Number of weights must match number of models")

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Load models and get predictions
    predictions = []
    for model_path in model_paths:
        model_data = load_model(model_path)
        model = model_data["model"]

        # Handle feature alignment
        expected_features = len(model_data.get("feature_names", []))
        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_

        X_val_processed = X_val.copy()

        # Align features
        if X_val_processed.shape[1] != expected_features:
            if X_val_processed.shape[1] < expected_features:
                padding = np.zeros((X_val_processed.shape[0], expected_features - X_val_processed.shape[1]))
                X_val_processed = np.column_stack([X_val_processed, padding])
            else:
                X_val_processed = X_val_processed[:, :expected_features]

        # Handle preprocessing
        if "imputer" in model_data:
            imputer = model_data["imputer"]
            if X_val_processed.shape[1] == imputer.n_features_in_:
                X_val_processed = imputer.transform(X_val_processed)
            else:
                from sklearn.impute import SimpleImputer
                fallback_imputer = SimpleImputer(strategy="median")
                X_val_processed = fallback_imputer.fit_transform(X_val_processed)
                # Re-align after imputation
                if X_val_processed.shape[1] != expected_features:
                    if X_val_processed.shape[1] < expected_features:
                        padding = np.zeros((X_val_processed.shape[0], expected_features - X_val_processed.shape[1]))
                        X_val_processed = np.column_stack([X_val_processed, padding])
                    else:
                        X_val_processed = X_val_processed[:, :expected_features]

        if "scaler" in model_data:
            scaler = model_data["scaler"]
            if X_val_processed.shape[1] == scaler.n_features_in_:
                X_val_processed = scaler.transform(X_val_processed)

        pred = model.predict_proba(X_val_processed)[:, 1]
        predictions.append(pred)

    # Weighted average
    predictions = np.array(predictions)
    y_pred_proba = np.average(predictions, axis=0, weights=weights)

    # Evaluate
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    logger.info(f"Blending Ensemble - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Weights: {dict(zip([p.stem for p in model_paths], weights))}")

    return {
        "weights": weights.tolist(),
        "feature_names": feature_names,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "y_pred_proba": y_pred_proba.tolist(),
        "y_val": y_val.tolist(),
        "ensemble_type": "blending",
        "base_model_paths": [str(p) for p in model_paths],
    }


def train_all_ensembles(
    model_paths: Optional[list[Path]] = None,
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Train all ensemble methods.

    Args:
        model_paths: List of paths to base models (if None, use best advanced models)
        features_path: Path to features file
        output_dir: Output directory for ensemble models

    Returns:
        Dictionary with ensemble results
    """
    output_dir = output_dir or config.experiments_dir / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_features(features_path)

    # Use best models if not specified
    if model_paths is None:
        advanced_dir = config.experiments_dir / "advanced"
        model_paths = [
            advanced_dir / "xgboost_advanced.joblib",
            advanced_dir / "random_forest_advanced.joblib",
            advanced_dir / "lightgbm_advanced.joblib",
        ]
        # Filter to existing models
        model_paths = [p for p in model_paths if p.exists()]

    if len(model_paths) == 0:
        raise ValueError("No model paths provided or found")

    logger.info(f"Training ensembles with {len(model_paths)} base models:")
    for p in model_paths:
        logger.info(f"  - {p.name}")

    results = {}

    # Voting Ensemble (Soft)
    try:
        voting_soft = voting_ensemble(
            model_paths, X_train, y_train, X_val, y_val, feature_names, voting="soft"
        )
        voting_soft_path = output_dir / "voting_soft.joblib"
        joblib.dump(voting_soft, voting_soft_path)
        results["voting_soft"] = {
            "model_path": voting_soft_path,
            "metrics": {"pr_auc": voting_soft["pr_auc"], "roc_auc": voting_soft["roc_auc"]},
        }

        # Log to MLflow
        if log_model_training:
            try:
                log_model_training(
                    model_name="voting_ensemble_soft",
                    model=voting_soft["model"],
                    feature_names=feature_names,
                    metrics={"pr_auc": voting_soft["pr_auc"], "roc_auc": voting_soft["roc_auc"]},
                    params={"voting": "soft", "n_models": len(model_paths)},
                    tags={"technique": "ensemble", "ensemble_type": "voting"},
                    model_path=voting_soft_path,
                )
            except Exception as e:
                logger.warning(f"Failed to log voting ensemble to MLflow: {e}")
    except Exception as e:
        logger.error(f"Failed to create voting ensemble: {e}")

    # Stacking Ensemble
    try:
        stacking = stacking_ensemble(
            model_paths, X_train, y_train, X_val, y_val, feature_names
        )
        stacking_path = output_dir / "stacking.joblib"
        joblib.dump(stacking, stacking_path)
        results["stacking"] = {
            "model_path": stacking_path,
            "metrics": {"pr_auc": stacking["pr_auc"], "roc_auc": stacking["roc_auc"]},
        }

        # Log to MLflow
        if log_model_training:
            try:
                log_model_training(
                    model_name="stacking_ensemble",
                    model=stacking["meta_learner"],
                    feature_names=feature_names,
                    metrics={"pr_auc": stacking["pr_auc"], "roc_auc": stacking["roc_auc"]},
                    params={"meta_learner": "LogisticRegression", "n_models": len(model_paths)},
                    tags={"technique": "ensemble", "ensemble_type": "stacking"},
                    model_path=stacking_path,
                )
            except Exception as e:
                logger.warning(f"Failed to log stacking ensemble to MLflow: {e}")
    except Exception as e:
        logger.error(f"Failed to create stacking ensemble: {e}")

    # Blending Ensemble (equal weights)
    try:
        blending = blending_ensemble(model_paths, None, X_train, y_train, X_val, y_val, feature_names)
        blending_path = output_dir / "blending.joblib"
        joblib.dump(blending, blending_path)
        results["blending"] = {
            "model_path": blending_path,
            "metrics": {"pr_auc": blending["pr_auc"], "roc_auc": blending["roc_auc"]},
        }

        # Log to MLflow
        if log_model_training:
            try:
                # Create a dummy model for logging (blending doesn't have a single model)
                from sklearn.base import BaseEstimator

                class BlendingModel(BaseEstimator):
                    def __init__(self, weights, base_model_paths):
                        self.weights = weights
                        self.base_model_paths = base_model_paths

                dummy_model = BlendingModel(blending["weights"], blending["base_model_paths"])
                log_model_training(
                    model_name="blending_ensemble",
                    model=dummy_model,
                    feature_names=feature_names,
                    metrics={"pr_auc": blending["pr_auc"], "roc_auc": blending["roc_auc"]},
                    params={"n_models": len(model_paths), "weights": str(blending["weights"])},
                    tags={"technique": "ensemble", "ensemble_type": "blending"},
                    model_path=blending_path,
                )
            except Exception as e:
                logger.warning(f"Failed to log blending ensemble to MLflow: {e}")
    except Exception as e:
        logger.error(f"Failed to create blending ensemble: {e}")

    logger.info(f"Ensemble training complete. Results saved to {output_dir}")
    return results

