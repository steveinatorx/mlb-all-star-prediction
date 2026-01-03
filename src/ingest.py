"""Data ingestion module for pulling raw baseball data."""

import time
from pathlib import Path
from typing import Optional

import polars as pl
from diskcache import Cache
from loguru import logger
from pybaseball import cache

from src.config import config

# Configure pybaseball cache
cache.enable()

# Initialize diskcache for our own caching
if config.cache_enabled:
    disk_cache = Cache(str(config.cache_dir))
else:
    disk_cache = None


def fetch_all_star_rosters(start_year: int, end_year: int) -> pl.DataFrame:
    """
    Fetch MLB All-Star rosters for given years.

    Args:
        start_year: Starting year
        end_year: Ending year

    Returns:
        DataFrame with columns: player_id, season, is_all_star
    """
    logger.info(f"Fetching All-Star rosters from {start_year} to {end_year}")

    # TODO: Implement actual All-Star roster fetching
    # pybaseball may not have direct All-Star roster endpoint
    # Alternative sources:
    # - Baseball Reference scraping
    # - MLB Stats API (requires API key)
    # - Retrosheet data

    # For now, return mock data structure
    logger.warning("Using mock All-Star data - implement actual fetching")
    mock_data = {
        "player_id": ["mock_player_1", "mock_player_2"],
        "season": [2020, 2021],
        "is_all_star": [True, True],
    }
    return pl.DataFrame(mock_data)


def fetch_minor_league_pitching(
    start_year: int, end_year: int, cache_key: Optional[str] = None
) -> pl.DataFrame:
    """
    Fetch minor league pitching statistics.

    Args:
        start_year: Starting year
        end_year: Ending year
        cache_key: Optional cache key for diskcache

    Returns:
        DataFrame with minor league pitching stats
    """
    logger.info(f"Fetching MiLB pitching data from {start_year} to {end_year}")

    # Check cache first
    if cache_key and disk_cache:
        cached = disk_cache.get(cache_key)
        if cached is not None:
            logger.info("Loading from cache")
            return pl.read_parquet(cached) if isinstance(cached, str) else cached

    # TODO: Implement actual MiLB data fetching
    # pybaseball has minor league functions but may be limited
    # Options:
    # - pybaseball.minor_leagues (if available)
    # - Baseball Reference scraping
    # - FanGraphs minor league data
    # - Retrosheet minor league data

    logger.warning("Using mock MiLB data - implement actual fetching")

    # Mock data structure matching MinorLeaguePitchingSeasonSchema
    mock_data = {
        "player_id": [f"player_{i}" for i in range(100)],
        "season": [2020 + (i % 4) for i in range(100)],
        "level": ["AAA" if i % 2 == 0 else "AA" for i in range(100)],
        "team": [f"team_{i % 10}" for i in range(100)],
        "league": ["IL" if i % 2 == 0 else "PCL" for i in range(100)],
        "games": [20 + (i % 30) for i in range(100)],
        "games_started": [15 + (i % 20) for i in range(100)],
        "complete_games": [i % 3 for i in range(100)],
        "innings_pitched": [50.0 + (i * 2.5) for i in range(100)],
        "hits": [40 + (i * 2) for i in range(100)],
        "runs": [20 + i for i in range(100)],
        "earned_runs": [18 + i for i in range(100)],
        "walks": [15 + (i % 20) for i in range(100)],
        "strikeouts": [50 + (i * 3) for i in range(100)],
        "home_runs": [5 + (i % 10) for i in range(100)],
        "wins": [5 + (i % 10) for i in range(100)],
        "losses": [3 + (i % 8) for i in range(100)],
        "saves": [i % 5 for i in range(100)],
    }

    df = pl.DataFrame(mock_data)

    # Compute derived stats
    df = df.with_columns(
        [
            ((pl.col("earned_runs") * 9.0) / pl.col("innings_pitched")).alias("era"),
            ((pl.col("hits") + pl.col("walks")) / pl.col("innings_pitched")).alias(
                "whip"
            ),
            ((pl.col("strikeouts") * 9.0) / pl.col("innings_pitched")).alias("k_per_9"),
            ((pl.col("walks") * 9.0) / pl.col("innings_pitched")).alias("bb_per_9"),
        ]
    )

    # Cache if enabled
    if cache_key and disk_cache:
        cache_path = config.cache_dir / f"{cache_key}.parquet"
        df.write_parquet(cache_path)
        disk_cache.set(cache_key, str(cache_path), expire=86400 * 7)  # 7 days

    return df


def fetch_player_info(player_ids: Optional[list[str]] = None) -> pl.DataFrame:
    """
    Fetch player biographical information and MLB debut dates.

    Args:
        player_ids: Optional list of player IDs to fetch. If None, fetch all.

    Returns:
        DataFrame with player info including MLB debut dates
    """
    logger.info("Fetching player information")

    # TODO: Implement actual player info fetching
    # pybaseball.playerid_reverse_lookup or similar
    # Critical: Must get MLB debut date to prevent leakage

    logger.warning("Using mock player data - implement actual fetching")

    mock_data = {
        "player_id": [f"player_{i}" for i in range(100)],
        "name_first": [f"First{i}" for i in range(100)],
        "name_last": [f"Last{i}" for i in range(100)],
        "birth_date": ["1990-01-01" for _ in range(100)],
        "mlb_debut": ["2020-04-01" for _ in range(100)],
    }

    return pl.DataFrame(mock_data)


def run_ingestion(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Run full data ingestion pipeline.

    Args:
        start_year: Starting year (defaults to config)
        end_year: Ending year (defaults to config)
        output_dir: Output directory (defaults to config.raw_data_dir)

    Returns:
        Dictionary mapping dataset names to file paths
    """
    start_year = start_year or config.start_year
    end_year = end_year or config.end_year
    output_dir = output_dir or config.raw_data_dir

    logger.info(f"Starting data ingestion: {start_year}-{end_year}")

    # Fetch All-Star rosters
    logger.info("Step 1: Fetching All-Star rosters")
    all_star_df = fetch_all_star_rosters(start_year, end_year)
    all_star_path = output_dir / "all_star_rosters.parquet"
    all_star_df.write_parquet(all_star_path)
    logger.info(f"Saved All-Star rosters to {all_star_path}")

    # Fetch minor league pitching
    logger.info("Step 2: Fetching minor league pitching stats")
    milb_df = fetch_minor_league_pitching(
        start_year, end_year, cache_key=f"milb_pitching_{start_year}_{end_year}"
    )
    milb_path = output_dir / "minor_league_pitching.parquet"
    milb_df.write_parquet(milb_path)
    logger.info(f"Saved MiLB pitching to {milb_path}")

    # Fetch player info
    logger.info("Step 3: Fetching player information")
    player_df = fetch_player_info()
    player_path = output_dir / "players.parquet"
    player_df.write_parquet(player_path)
    logger.info(f"Saved player info to {player_path}")

    # Add small delay to avoid rate limiting
    time.sleep(1)

    return {
        "all_star_rosters": all_star_path,
        "minor_league_pitching": milb_path,
        "players": player_path,
    }

