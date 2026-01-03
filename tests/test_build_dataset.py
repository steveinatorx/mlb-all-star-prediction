"""Smoke tests for dataset building."""

from pathlib import Path

import polars as pl
import pytest

from src.build_dataset import (
    apply_filters,
    create_labels,
    filter_pre_debut_stats,
    load_raw_data,
)


def test_load_raw_data(tmp_path: Path):
    """Test loading raw data files."""
    # Create mock data files
    players_df = pl.DataFrame({
        "player_id": ["p1", "p2"],
        "mlb_debut": ["2020-04-01", None],
    })
    players_df.write_parquet(tmp_path / "players.parquet")

    milb_df = pl.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "season": [2019, 2020, 2019],
        "innings_pitched": [100.0, 50.0, 75.0],
    })
    milb_df.write_parquet(tmp_path / "minor_league_pitching.parquet")

    all_star_df = pl.DataFrame({
        "player_id": ["p1"],
        "season": [2021],
    })
    all_star_df.write_parquet(tmp_path / "all_star_rosters.parquet")

    data = load_raw_data(tmp_path)
    assert "players" in data
    assert "minor_league_pitching" in data
    assert "all_star_rosters" in data


def test_filter_pre_debut_stats():
    """Test filtering to pre-debut stats only."""
    milb_df = pl.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2"],
        "season": [2018, 2020, 2019, 2021],
        "innings_pitched": [50.0, 50.0, 50.0, 50.0],
    })

    players_df = pl.DataFrame({
        "player_id": ["p1", "p2"],
        "mlb_debut": ["2019-04-01", "2020-04-01"],
    })

    filtered = filter_pre_debut_stats(milb_df, players_df)

    # p1: 2018 should be included (before 2019 debut), 2020 should be excluded
    # p2: 2019 should be included (before 2020 debut), 2021 should be excluded
    assert len(filtered) == 2
    assert all(filtered["season"] < 2019) or all(filtered["season"] < 2020)


def test_apply_filters():
    """Test data quality filters."""
    df = pl.DataFrame({
        "player_id": ["p1", "p2", "p3"],
        "innings_pitched": [100.0, 30.0, 60.0],  # p2 below threshold
        "hits": [50, -1, 40],  # p2 has invalid data
        "runs": [20, 15, 18],
    })

    filtered = apply_filters(df)

    # Should filter out p2 (low IP and invalid hits)
    assert len(filtered) <= 2


def test_create_labels():
    """Test label creation."""
    players_df = pl.DataFrame({
        "player_id": ["p1", "p2", "p3"],
    })

    all_star_df = pl.DataFrame({
        "player_id": ["p1", "p2"],
        "season": [2020, 2021],
    })

    labels = create_labels(players_df, all_star_df)

    assert len(labels) == 3
    assert labels.filter(pl.col("player_id") == "p1")["is_all_star"].item() is True
    assert labels.filter(pl.col("player_id") == "p3")["is_all_star"].item() is False

