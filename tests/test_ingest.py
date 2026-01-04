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
    # Use a known player ID to avoid triggering expensive MLB stats fetching
    # Jacob deGrom's MLBAM ID
    df = fetch_player_info(["594798"])
    assert isinstance(df, pl.DataFrame)
    assert "player_id" in df.columns
    assert "name_first" in df.columns
    assert "name_last" in df.columns
    assert "mlb_debut" in df.columns
    assert len(df) > 0


def test_fetch_player_info_with_ids():
    """Test player info fetching with specific player IDs."""
    # Test with a known player ID (Jacob deGrom's MLBAM ID)
    # First get the ID using playerid_lookup
    try:
        from pybaseball import playerid_lookup
        lookup_df = playerid_lookup("degrom", "jacob")
        if len(lookup_df) > 0:
            mlbam_id = str(int(lookup_df["key_mlbam"].iloc[0]))
            df = fetch_player_info([mlbam_id])
            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0
            assert df["player_id"].str.contains(mlbam_id).any()
            # Check that mlb_debut is in YYYY-MM-DD format
            if df["mlb_debut"].is_not_null().any():
                debut_dates = df.filter(pl.col("mlb_debut").is_not_null())["mlb_debut"]
                assert all(len(d) == 10 for d in debut_dates)  # YYYY-MM-DD format
    except ImportError:
        pytest.skip("pybaseball not available for this test")


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

