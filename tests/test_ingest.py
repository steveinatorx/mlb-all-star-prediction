"""Smoke tests for data ingestion."""

from pathlib import Path

import polars as pl
import pytest

from src.ingest import fetch_all_star_rosters, fetch_minor_league_pitching, fetch_player_info


def test_fetch_all_star_rosters():
    """Test All-Star roster fetching returns valid structure."""
    df = fetch_all_star_rosters(2020, 2021)
    assert isinstance(df, pl.DataFrame)
    assert "player_id" in df.columns
    assert "season" in df.columns
    assert len(df) > 0


def test_fetch_minor_league_pitching():
    """Test MiLB pitching data fetching returns valid structure."""
    df = fetch_minor_league_pitching(2020, 2021)
    assert isinstance(df, pl.DataFrame)
    assert "player_id" in df.columns
    assert "season" in df.columns
    assert "innings_pitched" in df.columns
    assert len(df) > 0

    # Check that derived stats are computed
    assert "era" in df.columns
    assert "whip" in df.columns


def test_fetch_player_info():
    """Test player info fetching returns valid structure."""
    df = fetch_player_info()
    assert isinstance(df, pl.DataFrame)
    assert "player_id" in df.columns
    assert len(df) > 0


def test_run_ingestion(tmp_path: Path):
    """Test full ingestion pipeline."""
    from src.ingest import run_ingestion

    results = run_ingestion(start_year=2020, end_year=2021, output_dir=tmp_path)

    assert "all_star_rosters" in results
    assert "minor_league_pitching" in results
    assert "players" in results

    # Check files exist
    for path in results.values():
        assert Path(path).exists()
        df = pl.read_parquet(path)
        assert len(df) > 0

