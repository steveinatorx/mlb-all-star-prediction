"""Hyperparameter tuning module using Optuna for Bayesian optimization."""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import config

try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
    )
except ImportError:
    optuna = None
    logger.warning("Optuna not available, hyperparameter tuning disabled")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


def load_features_for_tuning(
    features_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load features for hyperparameter tuning (train + val)."""
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

    # Get feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if c not in ["player_id", "is_all_star", "split"]]

    # Split data
    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "val")

    # Convert to numpy
    X_train = train_df.select(feature_cols).to_numpy().astype(np.float64)
    X_val = val_df.select(feature_cols).to_numpy().astype(np.float64)
    y_train = train_df["is_all_star"].to_numpy().astype(int)
    y_val = val_df["is_all_star"].to_numpy().astype(int)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    logger.info(f"Loaded features for tuning: train={len(X_train)}, val={len(X_val)}")
    logger.info(f"Class balance - train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")

    return X_train, X_val, y_train, y_val, feature_cols


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, k_neighbors: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling."""
    if SMOTE is None:
        return X_train, y_train

    smote = SMOTE(k_neighbors=k_neighbors, random_state=config.random_seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Tune XGBoost hyperparameters using Optuna."""
    if optuna is None:
        raise ImportError("Optuna not available")

    def objective(trial):
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "random_state": config.random_seed,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

        # Train model
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)

        return pr_auc

    study = optuna.create_study(direction="maximize", study_name="xgboost_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best XGBoost PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Tune LightGBM hyperparameters using Optuna."""
    if optuna is None or LGBMClassifier is None:
        raise ImportError("Optuna or LightGBM not available")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "class_weight": "balanced",
            "random_state": config.random_seed,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)

        return pr_auc

    study = optuna.create_study(direction="maximize", study_name="lightgbm_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best LightGBM PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Tune Random Forest hyperparameters using Optuna."""
    if optuna is None:
        raise ImportError("Optuna not available")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "random_state": config.random_seed,
            "n_jobs": config.n_jobs,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)

        return pr_auc

    study = optuna.create_study(direction="maximize", study_name="random_forest_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best Random Forest PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }


def tune_all_models(
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_trials: int = 50,
    use_smote: bool = True,
) -> dict:
    """Tune all models and save results."""
    if optuna is None:
        raise ImportError("Optuna not available. Install with: pipenv install optuna")

    output_dir = output_dir or config.experiments_dir / "tuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting hyperparameter tuning")
    logger.info(f"  Models: XGBoost, LightGBM, Random Forest")
    logger.info(f"  Trials per model: {n_trials}")
    logger.info(f"  SMOTE: {use_smote}")

    # Load data
    X_train, X_val, y_train, y_val, feature_names = load_features_for_tuning(features_path)

    # Apply SMOTE if requested
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
        logger.info(f"After SMOTE - train: {len(X_train)}, positive: {y_train.sum()}")

    results = {}

    # Tune XGBoost
    logger.info("Tuning XGBoost...")
    xgb_results = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=n_trials)
    results["xgboost"] = xgb_results

    # Save XGBoost study
    xgb_study_path = output_dir / "xgboost_study.pkl"
    joblib.dump(xgb_results["study"], xgb_study_path)

    # Tune LightGBM
    if LGBMClassifier is not None:
        logger.info("Tuning LightGBM...")
        lgb_results = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=n_trials)
        results["lightgbm"] = lgb_results

        lgb_study_path = output_dir / "lightgbm_study.pkl"
        joblib.dump(lgb_results["study"], lgb_study_path)

    # Tune Random Forest
    logger.info("Tuning Random Forest...")
    rf_results = tune_random_forest(X_train, y_train, X_val, y_val, n_trials=n_trials)
    results["random_forest"] = rf_results

    rf_study_path = output_dir / "random_forest_study.pkl"
    joblib.dump(rf_results["study"], rf_study_path)

    # Save results summary
    summary = {
        model: {
            "best_params": results[model]["best_params"],
            "best_pr_auc": float(results[model]["best_value"]),
        }
        for model in results
    }

    summary_path = output_dir / "tuning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Tuning complete. Results saved to {output_dir}")

    # Generate visualization plots
    try:
        figures_dir = config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_results in results.items():
            study = model_results["study"]

            # Optimization history
            try:
                fig = plot_optimization_history(study)
                # Try to save as image, fallback to HTML
                try:
                    fig.write_image(figures_dir / f"tuning_history_{model_name}.png")
                except Exception:
                    fig.write_html(figures_dir / f"tuning_history_{model_name}.html")
            except Exception as e:
                logger.warning(f"Could not generate optimization history for {model_name}: {e}")

            # Parameter importance
            try:
                fig = plot_param_importances(study)
                try:
                    fig.write_image(figures_dir / f"tuning_importance_{model_name}.png")
                except Exception:
                    fig.write_html(figures_dir / f"tuning_importance_{model_name}.html")
            except Exception as e:
                logger.warning(f"Could not generate parameter importance for {model_name}: {e}")

        logger.info("Saved tuning visualization plots")
    except Exception as e:
        logger.warning(f"Could not generate tuning plots: {e}")

    return results

