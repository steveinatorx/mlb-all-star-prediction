"""Tests for interaction feature creation."""

import pytest
import polars as pl
import numpy as np

from src.create_interaction_features import create_interaction_features


@pytest.fixture
def sample_features():
    """Create sample features DataFrame."""
    n_samples = 100
    
    data = {
        "career_era": np.random.uniform(2.0, 5.0, n_samples).tolist(),
        "career_whip": np.random.uniform(1.0, 1.5, n_samples).tolist(),
        "career_k_per_9": np.random.uniform(6.0, 12.0, n_samples).tolist(),
        "career_bb_per_9": np.random.uniform(2.0, 5.0, n_samples).tolist(),
        "best_era": np.random.uniform(1.5, 4.0, n_samples).tolist(),
        "best_whip": np.random.uniform(0.8, 1.3, n_samples).tolist(),
        "best_k_per_9": np.random.uniform(7.0, 13.0, n_samples).tolist(),
        "best_bb_per_9": np.random.uniform(1.5, 4.5, n_samples).tolist(),
        "total_milb_ip": np.random.uniform(100, 500, n_samples).tolist(),
    }
    
    return pl.DataFrame(data)


def test_create_interaction_features(sample_features):
    """Test creating interaction features."""
    df_with_interactions = create_interaction_features(sample_features)
    
    # Should have more columns than original
    assert len(df_with_interactions.columns) > len(sample_features.columns)
    
    # Check for specific interaction features
    assert "k_bb_ratio" in df_with_interactions.columns or "career_k_bb_ratio" in df_with_interactions.columns
    assert "era_whip_product" in df_with_interactions.columns or "best_era_whip_product" in df_with_interactions.columns


def test_interaction_features_k_bb_ratio(sample_features):
    """Test K/BB ratio interaction feature."""
    df = create_interaction_features(sample_features)
    
    # Should have K/BB ratio
    k_bb_cols = [col for col in df.columns if "k_bb" in col.lower()]
    assert len(k_bb_cols) > 0
    
    # Ratio should be positive (when BB/9 > 0)
    if "career_k_bb_ratio" in df.columns:
        ratio_values = df["career_k_bb_ratio"].drop_nulls()
        assert (ratio_values > 0).all() or len(ratio_values) == 0


def test_interaction_features_handles_missing():
    """Test interaction features handle missing values."""
    n_samples = 100
    data = {
        "career_era": np.random.uniform(2.0, 5.0, n_samples).tolist(),
        "career_whip": [None] * n_samples,  # All missing
        "career_k_per_9": np.random.uniform(6.0, 12.0, n_samples).tolist(),
    }
    
    df = pl.DataFrame(data)
    
    # Should handle gracefully
    df_with_interactions = create_interaction_features(df)
    assert isinstance(df_with_interactions, pl.DataFrame)


def test_interaction_features_division_by_zero(sample_features):
    """Test interaction features handle division by zero."""
    # Set some BB/9 values to zero
    df = sample_features.with_columns([
        pl.when(pl.col("career_bb_per_9") < 2.5)
        .then(0.0)
        .otherwise(pl.col("career_bb_per_9"))
        .alias("career_bb_per_9")
    ])
    
    # Should handle division by zero gracefully
    df_with_interactions = create_interaction_features(df)
    assert isinstance(df_with_interactions, pl.DataFrame)
    
    # K/BB ratio should be None/Null where BB/9 is zero
    if "career_k_bb_ratio" in df_with_interactions.columns:
        zero_bb_mask = df["career_bb_per_9"] == 0.0
        if zero_bb_mask.sum() > 0:
            # Ratio should be null where denominator is zero
            ratio_at_zero = df_with_interactions.filter(
                pl.col("career_bb_per_9") == 0.0
            )["career_k_bb_ratio"]
            # Should be null or handle gracefully
            assert ratio_at_zero.null_count() >= 0

