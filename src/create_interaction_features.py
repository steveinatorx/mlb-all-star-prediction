"""Create interaction features based on SHAP analysis insights."""

from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from src.config import config


def create_interaction_features(
    features_df: pl.DataFrame,
    interaction_suggestions: Optional[list[dict]] = None,
) -> pl.DataFrame:
    """
    Create interaction features based on SHAP analysis.
    
    Args:
        features_df: Base features DataFrame
        interaction_suggestions: List of suggested interactions from SHAP analysis
                                If None, creates common baseball stat interactions
        
    Returns:
        DataFrame with added interaction features
    """
    logger.info("Creating interaction features")
    
    df = features_df.clone()
    
    # If no suggestions provided, create common baseball stat interactions
    if interaction_suggestions is None:
        interaction_suggestions = [
            # Strikeout rate interactions
            {"feature1": "career_k_per_9", "feature2": "career_bb_per_9", "operation": "ratio", "name": "k_bb_ratio"},
            {"feature1": "best_k_per_9", "feature2": "best_bb_per_9", "operation": "ratio", "name": "best_k_bb_ratio"},
            
            # ERA/WHIP interactions (efficiency metrics)
            {"feature1": "career_era", "feature2": "career_whip", "operation": "multiply", "name": "era_whip_product"},
            {"feature1": "best_era", "feature2": "best_whip", "operation": "multiply", "name": "best_era_whip_product"},
            
            # Strikeout dominance (K/9 * IP)
            {"feature1": "career_k_per_9", "feature2": "total_milb_ip", "operation": "multiply", "name": "total_strikeouts_estimate"},
            
            # Control ratio (K/BB)
            {"feature1": "career_k_per_9", "feature2": "career_bb_per_9", "operation": "ratio", "name": "career_k_bb_ratio"},
            
            # Best vs Career (consistency)
            {"feature1": "best_era", "feature2": "career_era", "operation": "ratio", "name": "best_career_era_ratio"},
            {"feature1": "best_whip", "feature2": "career_whip", "operation": "ratio", "name": "best_career_whip_ratio"},
            {"feature1": "best_k_per_9", "feature2": "career_k_per_9", "operation": "ratio", "name": "best_career_k_ratio"},
            
            # Level progression (AAA experience)
            {"feature1": "seasons_at_aaa", "feature2": "age_at_debut", "operation": "multiply", "name": "aaa_experience_age"},
            
            # Draft position interactions
            {"feature1": "draft_round", "feature2": "age_at_debut", "operation": "multiply", "name": "draft_age_interaction"},
        ]
    
    new_features = []
    
    for interaction in interaction_suggestions:
        feat1 = interaction["feature1"]
        feat2 = interaction["feature2"]
        operation = interaction.get("operation", "multiply")
        feat_name = interaction.get("name", f"{feat1}_{operation}_{feat2}")
        
        # Check if both features exist
        if feat1 not in df.columns or feat2 not in df.columns:
            logger.debug(f"Skipping {feat_name}: missing features")
            continue
        
        try:
            if operation == "multiply":
                new_feat = (pl.col(feat1) * pl.col(feat2)).alias(feat_name)
            elif operation == "ratio":
                # Avoid division by zero
                new_feat = (
                    pl.when(pl.col(feat2) != 0)
                    .then(pl.col(feat1) / pl.col(feat2))
                    .otherwise(None)
                    .alias(feat_name)
                )
            elif operation == "difference":
                new_feat = (pl.col(feat1) - pl.col(feat2)).alias(feat_name)
            elif operation == "sum":
                new_feat = (pl.col(feat1) + pl.col(feat2)).alias(feat_name)
            else:
                logger.warning(f"Unknown operation: {operation}, skipping {feat_name}")
                continue
            
            df = df.with_columns([new_feat])
            new_features.append(feat_name)
            logger.debug(f"Created interaction feature: {feat_name}")
            
        except Exception as e:
            logger.warning(f"Failed to create {feat_name}: {e}")
            continue
    
    logger.info(f"Created {len(new_features)} interaction features: {new_features}")
    
    return df


def add_interaction_features_to_pipeline(
    features_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    interaction_suggestions: Optional[list[dict]] = None,
) -> Path:
    """
    Add interaction features to existing features file.
    
    Args:
        features_path: Path to existing features file
        output_path: Path to save features with interactions
        interaction_suggestions: Optional list of interaction suggestions
        
    Returns:
        Path to saved features file with interactions
    """
    features_path = features_path or config.features_data_dir / "features.parquet"
    output_path = output_path or config.features_data_dir / "features_with_interactions.parquet"
    
    logger.info(f"Loading features from {features_path}")
    df = pl.read_parquet(features_path)
    
    # Create interaction features
    df_with_interactions = create_interaction_features(df, interaction_suggestions)
    
    # Save
    df_with_interactions.write_parquet(output_path)
    logger.info(f"Saved features with interactions to {output_path}")
    logger.info(f"Feature shape: {df_with_interactions.shape}")
    
    return output_path

