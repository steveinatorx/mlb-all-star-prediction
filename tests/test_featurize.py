"""Smoke tests for feature engineering."""

import polars as pl
import pytest

from src.featurize import (
    create_best_season_features,
    create_career_aggregates,
    create_progression_features,
)


def test_create_career_aggregates():
    """Test career aggregate feature creation."""
    milb_df = pl.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "innings_pitched": [50.0, 50.0, 100.0],
        "games": [10, 10, 20],
        "games_started": [8, 8, 15],
        "earned_runs": [20, 20, 30],
        "hits": [40, 40, 80],
        "walks": [15, 15, 25],
        "strikeouts": [60, 60, 120],
    })

    aggregates = create_career_aggregates(milb_df)

    assert len(aggregates) == 2
    assert "total_milb_ip" in aggregates.columns
    assert "career_era" in aggregates.columns
    assert aggregates.filter(pl.col("player_id") == "p1")["total_milb_ip"].item() == 100.0


def test_create_best_season_features():
    """Test best season feature creation."""
    milb_df = pl.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "era": [4.0, 3.0, 5.0],
        "whip": [1.3, 1.2, 1.5],
        "k_per_9": [8.0, 9.0, 7.0],
    })

    best_seasons = create_best_season_features(milb_df)

    assert len(best_seasons) == 2
    assert "best_era" in best_seasons.columns
    p1_best_era = best_seasons.filter(pl.col("player_id") == "p1")["best_era"].item()
    assert p1_best_era == 3.0  # Minimum ERA


def test_create_progression_features():
    """Test progression feature creation."""
    milb_df = pl.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2"],
        "level": ["AA", "AAA", "A", "AA"],
    })

    players_df = pl.DataFrame({
        "player_id": ["p1", "p2"],
        "birth_date": ["1995-01-01", "1996-01-01"],
        "mlb_debut": ["2020-04-01", "2021-04-01"],
    })

    progression = create_progression_features(milb_df, players_df)

    assert len(progression) == 2
    assert "highest_level_reached" in progression.columns
    assert "seasons_at_aaa" in progression.columns
    assert progression.filter(pl.col("player_id") == "p1")["highest_level_reached"].item() == "AAA"

