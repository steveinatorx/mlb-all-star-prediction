"""Analyze SHAP values to identify feature interactions and create new features."""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
from loguru import logger

from src.config import config

try:
    import shap
except ImportError:
    shap = None
    logger.warning("SHAP not available, skipping interaction analysis")


def load_shap_values(
    model_path: Path,
    features_path: Optional[Path] = None,
    split: str = "train",
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load SHAP values for a model.
    
    Args:
        model_path: Path to saved model
        features_path: Path to features file
        split: Which split to use
        max_samples: Maximum samples to analyze
        
    Returns:
        Tuple of (shap_values, X, feature_names)
    """
    if shap is None:
        raise ImportError("SHAP not available")
    
    logger.info(f"Loading SHAP values for {model_path.name}")
    
    # Load model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    
    # Align feature_names with model expectations
    if hasattr(model, "n_features_in_"):
        actual_n_features = model.n_features_in_
        if len(feature_names) != actual_n_features:
            logger.warning(
                f"Model feature_names has {len(feature_names)} features but "
                f"n_features_in_ is {actual_n_features}. Using first {actual_n_features} features."
            )
            feature_names = feature_names[:actual_n_features]
    
    # Load features
    features_path = features_path or config.features_data_dir / "features.parquet"
    df = pl.read_parquet(features_path)
    
    # Encode categorical columns
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
    
    # Convert to numpy
    arrays = []
    for col in feature_names:
        if col in split_df.columns:
            col_data = split_df[col]
            if col_data.dtype in [pl.Float64, pl.Float32]:
                arr = col_data.to_numpy()
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            elif col_data.dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64]:
                arr = col_data.to_numpy()
                arr = np.array([float(x) if x is not None else np.nan for x in arr])
            else:
                arr = np.array([float(x) if x is not None else np.nan for x in col_data.to_list()])
        else:
            arr = np.zeros(len(split_df), dtype=np.float64)
        arrays.append(arr)
    
    X = np.column_stack(arrays).astype(np.float64)
    
    # Align with model expectations
    if hasattr(model, "n_features_in_"):
        if X.shape[1] != model.n_features_in_:
            if X.shape[1] < model.n_features_in_:
                padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
                X = np.column_stack([X, padding])
            else:
                X = X[:, :model.n_features_in_]
    
    # Handle imputation
    if "imputer" in model_data:
        imputer = model_data["imputer"]
        if X.shape[1] == imputer.n_features_in_:
            X = imputer.transform(X)
        else:
            from sklearn.impute import SimpleImputer
            fallback_imputer = SimpleImputer(strategy="median")
            X = fallback_imputer.fit_transform(X)
            if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
                if X.shape[1] < model.n_features_in_:
                    padding = np.zeros((X.shape[0], model.n_features_in_ - X.shape[1]))
                    X = np.column_stack([X, padding])
                else:
                    X = X[:, :model.n_features_in_]
    elif np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
    
    # Calculate SHAP values
    # Use interventional perturbation for robustness with feature mismatches
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)
    except Exception as e:
        logger.warning(f"Interventional perturbation failed: {e}, trying auto")
        explainer = shap.TreeExplainer(model, feature_perturbation="auto")
        shap_values = explainer.shap_values(X)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    return shap_values, X, feature_names


def analyze_feature_interactions(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> dict:
    """
    Analyze feature interactions from SHAP values.
    
    Args:
        shap_values: SHAP values (n_samples, n_features)
        X: Feature values (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to analyze
        
    Returns:
        Dictionary with interaction analysis results
    """
    logger.info("Analyzing feature interactions")
    
    # Calculate feature importance (mean absolute SHAP)
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    
    interactions = {}
    
    for i, feat_idx in enumerate(top_indices):
        feat_name = feature_names[feat_idx]
        interactions[feat_name] = {
            "importance": float(feature_importance[feat_idx]),
            "rank": i + 1,
            "interactions": []
        }
        
        # Find features that interact with this one
        for other_idx in top_indices:
            if other_idx == feat_idx:
                continue
            
            other_name = feature_names[other_idx]
            
            # Calculate interaction strength
            # Method 1: Correlation between SHAP values
            shap_corr = np.corrcoef(shap_values[:, feat_idx], shap_values[:, other_idx])[0, 1]
            
            # Method 2: Conditional SHAP (how does feature A's SHAP change with feature B's value?)
            # Bin feature B values and see how feature A's SHAP changes
            other_values = X[:, other_idx]
            feat_shap = shap_values[:, feat_idx]
            
            # Remove NaN
            valid_mask = ~(np.isnan(other_values) | np.isnan(feat_shap))
            if valid_mask.sum() < 10:
                continue
            
            other_vals_clean = other_values[valid_mask]
            feat_shap_clean = feat_shap[valid_mask]
            
            # Split into quartiles
            quartiles = np.percentile(other_vals_clean, [25, 50, 75])
            q1_mask = other_vals_clean <= quartiles[0]
            q4_mask = other_vals_clean > quartiles[2]
            
            if q1_mask.sum() > 0 and q4_mask.sum() > 0:
                q1_mean_shap = feat_shap_clean[q1_mask].mean()
                q4_mean_shap = feat_shap_clean[q4_mask].mean()
                interaction_strength = abs(q4_mean_shap - q1_mean_shap)
            else:
                interaction_strength = 0.0
            
            interactions[feat_name]["interactions"].append({
                "feature": other_name,
                "shap_correlation": float(shap_corr) if not np.isnan(shap_corr) else 0.0,
                "interaction_strength": float(interaction_strength),
                "combined_score": float(abs(shap_corr) + interaction_strength) if not np.isnan(shap_corr) else float(interaction_strength)
            })
        
        # Sort interactions by combined score
        interactions[feat_name]["interactions"].sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )
    
    return interactions


def suggest_interaction_features(
    interactions: dict,
    top_interactions: int = 5,
) -> list[dict]:
    """
    Suggest new interaction features based on SHAP analysis.
    
    Args:
        interactions: Interaction analysis results
        top_interactions: Number of top interactions to suggest
        
    Returns:
        List of suggested feature interactions
    """
    suggestions = []
    
    for feat_name, feat_data in interactions.items():
        for interaction in feat_data["interactions"][:top_interactions]:
            other_name = interaction["feature"]
            
            # Skip if already suggested (reverse order)
            if any(
                s["feature1"] == other_name and s["feature2"] == feat_name
                for s in suggestions
            ):
                continue
            
            suggestions.append({
                "feature1": feat_name,
                "feature2": other_name,
                "interaction_strength": interaction["interaction_strength"],
                "shap_correlation": interaction["shap_correlation"],
                "combined_score": interaction["combined_score"],
                "suggested_operations": [
                    "multiply",  # A * B
                    "ratio",     # A / B (if B != 0)
                    "difference", # A - B
                    "sum",       # A + B
                ]
            })
    
    # Sort by combined score
    suggestions.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return suggestions[:top_interactions * len(interactions)]


def generate_interaction_report(
    interactions: dict,
    suggestions: list[dict],
    output_path: Path,
) -> None:
    """Generate a markdown report of feature interactions."""
    logger.info(f"Generating interaction report to {output_path}")
    
    lines = [
        "# Feature Interaction Analysis",
        "",
        "## Top Features by Importance",
        "",
    ]
    
    # Sort features by importance
    sorted_features = sorted(
        interactions.items(),
        key=lambda x: x[1]["importance"],
        reverse=True
    )
    
    for feat_name, feat_data in sorted_features:
        lines.append(f"### {feat_data['rank']}. {feat_name}")
        lines.append(f"- **Importance**: {feat_data['importance']:.4f}")
        lines.append(f"- **Top Interactions**:")
        
        for interaction in feat_data["interactions"][:3]:
            lines.append(
                f"  - {interaction['feature']}: "
                f"strength={interaction['interaction_strength']:.4f}, "
                f"corr={interaction['shap_correlation']:.4f}"
            )
        lines.append("")
    
    lines.extend([
        "## Suggested Interaction Features",
        "",
        "Based on SHAP interaction analysis, the following feature combinations may improve predictions:",
        "",
    ])
    
    for i, suggestion in enumerate(suggestions, 1):
        lines.append(f"### {i}. {suggestion['feature1']} × {suggestion['feature2']}")
        lines.append(f"- **Interaction Strength**: {suggestion['interaction_strength']:.4f}")
        lines.append(f"- **SHAP Correlation**: {suggestion['shap_correlation']:.4f}")
        lines.append(f"- **Combined Score**: {suggestion['combined_score']:.4f}")
        lines.append(f"- **Suggested Operations**: {', '.join(suggestion['suggested_operations'])}")
        lines.append("")
    
    output_path.write_text("\n".join(lines))
    logger.info(f"Saved interaction report to {output_path}")


def analyze_model_interactions(
    model_path: Path,
    features_path: Optional[Path] = None,
    split: str = "train",
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Analyze feature interactions for a model.
    
    Args:
        model_path: Path to saved model
        features_path: Path to features file
        split: Which split to analyze
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with interaction analysis results
    """
    output_dir = output_dir or config.reports_dir / "interactions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SHAP values
    shap_values, X, feature_names = load_shap_values(
        model_path, features_path, split
    )
    
    # Analyze interactions
    interactions = analyze_feature_interactions(shap_values, X, feature_names)
    
    # Suggest new features
    suggestions = suggest_interaction_features(interactions)
    
    # Generate report
    report_path = output_dir / f"{model_path.stem}_interactions.md"
    generate_interaction_report(interactions, suggestions, report_path)
    
    return {
        "interactions": interactions,
        "suggestions": suggestions,
        "report_path": report_path
    }

