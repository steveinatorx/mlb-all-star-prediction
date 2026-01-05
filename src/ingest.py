"""Data ingestion module for pulling raw baseball data."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import polars as pl
from diskcache import Cache
from loguru import logger
from pybaseball import cache

from src.config import config
from src.ingest_tracker import IngestionTracker
from src.incremental_writer import IncrementalWriter

# Try to import scraping dependencies
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    logger.warning(
        "BeautifulSoup4 and requests not installed. "
        "Install with: pipenv install beautifulsoup4 requests lxml"
    )

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

    # Try pybaseball's all_star_full first (Lahman database)
    try:
        from pybaseball import all_star_full
        import pandas as pd

        logger.info("Attempting to fetch from Lahman database via pybaseball")
        all_star_df = all_star_full()

        if len(all_star_df) > 0:
            # Filter by year range
            # Check what columns are available
            logger.info(f"Lahman data columns: {all_star_df.columns.tolist()}")
            
            # Common Lahman columns: yearID, playerID, teamID, etc.
            if "yearID" in all_star_df.columns:
                filtered = all_star_df[
                    (all_star_df["yearID"] >= start_year)
                    & (all_star_df["yearID"] <= end_year)
                ]
                
                # Convert to our format
                # Need to map playerID to MLBAM ID
                result_df = pl.from_pandas(filtered).select([
                    pl.col("playerID").alias("player_id_lahman"),
                    pl.col("yearID").alias("season"),
                ]).with_columns([
                    pl.lit(True).alias("is_all_star")
                ])
                
                # Try to convert Lahman playerID to MLBAM ID
                # This requires a lookup - for now, use Lahman ID as player_id
                # TODO: Add player ID mapping
                logger.warning(
                    "Using Lahman playerID - need to map to MLBAM ID. "
                    "Using Lahman ID as player_id for now."
                )
                result_df = result_df.rename({"player_id_lahman": "player_id"})
                
                logger.info(f"Successfully fetched {len(result_df)} All-Star records")
                return result_df

    except Exception as e:
        logger.warning(f"Could not fetch from Lahman database: {e}")

    # Try Baseball Reference scraping
    try:
        from data_fetch.fetch_all_star import fetch_all_star_rosters_with_mapping
        
        logger.info("Attempting to scrape All-Star rosters from Baseball Reference")
        
        # Load player info for ID mapping if available
        player_info_file = config.raw_data_dir / "players.parquet"
        players_df = None
        if player_info_file.exists():
            try:
                players_df = pl.read_parquet(player_info_file)
            except Exception:
                pass
        
        bref_df = fetch_all_star_rosters_with_mapping(
            start_year, end_year, players_df=players_df
        )
        
        if len(bref_df) > 0:
            logger.info(f"Successfully fetched {len(bref_df)} All-Star records from Baseball Reference")
            return bref_df
        else:
            logger.warning("No All-Star data found from Baseball Reference")
    except ImportError:
        logger.debug("Baseball Reference scraping module not available")
    except Exception as e:
        logger.warning(f"Error scraping All-Star rosters from Baseball Reference: {e}")

    # Last resort: Check for manual dataset file
    manual_file = config.raw_data_dir / "all_star_rosters.csv"
    if manual_file.exists():
        logger.info(f"Found manual All-Star roster file: {manual_file}")
        try:
            manual_df = pl.read_csv(manual_file)
            # Ensure it has the right columns
            if "player_id" in manual_df.columns and "season" in manual_df.columns:
                if "is_all_star" not in manual_df.columns:
                    manual_df = manual_df.with_columns(pl.lit(True).alias("is_all_star"))
                logger.info(f"Loaded {len(manual_df)} All-Star records from manual file")
                return manual_df
        except Exception as e:
            logger.warning(f"Could not read manual All-Star file: {e}")

    # Last resort: Return mock data with warning
    logger.warning(
        "All-Star roster fetching failed. Using mock data. "
        "This will need to be replaced with real data for model training. "
        f"To use real data, create a CSV file at: {manual_file} "
        "with columns: player_id (MLBAM), season, is_all_star"
    )
    
    # Generate mock data for the requested years
    mock_data = {
        "player_id": [f"mock_player_{i}" for i in range((end_year - start_year + 1) * 2)],
        "season": [year for year in range(start_year, end_year + 1) for _ in range(2)],
        "is_all_star": [True] * ((end_year - start_year + 1) * 2),
    }
    return pl.DataFrame(mock_data)


def fetch_minor_league_pitching(
    start_year: int,
    end_year: int,
    cache_key: Optional[str] = None,
    player_info_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    incremental_writer: Optional[IncrementalWriter] = None,
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

    # Try MiLB.com scraping FIRST (has expanded stats: TBF, NP, P/IP, QS, QF, HLD, IBB, WP, BK)
    # This is now the primary source since it provides more comprehensive stats
    try:
        from data_fetch.fetch_milb_mlbcom import (
            fetch_minor_league_pitching_batch_mlbcom,
            calculate_optimal_scraping_range,
        )
        
        logger.info("Attempting to fetch from MiLB.com (league-level scraping - basic stats only)")
        
        # Load player info to calculate optimal scraping range
        player_info_file = player_info_path or (config.raw_data_dir / "players.parquet")
        optimal_start_year = start_year
        optimal_end_year = end_year
        player_ids_set = None
        
        if player_info_file.exists():
            try:
                player_df = pl.read_parquet(player_info_file)
                
                # Calculate optimal scraping range based on player debut dates
                optimal_start_year, optimal_end_year, player_ids_set = (
                    calculate_optimal_scraping_range(player_df, min_years_before_debut=5)
                )
                
                # Use the optimal range, but ensure it overlaps with requested range
                optimal_end_year = min(optimal_end_year, end_year)
                
                logger.info(
                    f"Optimal scraping range: {optimal_start_year} to {optimal_end_year} "
                    f"for {len(player_ids_set) if player_ids_set else 'all'} players"
                )
            except Exception as e:
                logger.warning(f"Could not calculate optimal range: {e}")
        
        # Fetch from MiLB.com with expanded stats enabled (use_expanded=True by default)
        milb_df = fetch_minor_league_pitching_batch_mlbcom(
            player_mlbam_ids=None,  # Not needed for league scraping
            start_year=optimal_start_year,
            end_year=optimal_end_year,
            delay=0.6,  # Faster without Selenium
            use_league_scraping=True,
            league_slugs=None,  # Use default leagues
            player_ids_filter=player_ids_set,
        )
        
        if len(milb_df) > 0:
            # Write incrementally if writer provided
            if incremental_writer:
                incremental_writer.write(milb_df.to_dicts())
                logger.info(f"Wrote {len(milb_df)} records from MiLB.com to incremental writer")
                return pl.DataFrame()  # Empty - data already written
            
            # Cache the results
            if cache_key and disk_cache:
                cache_path = config.cache_dir / f"{cache_key}.parquet"
                milb_df.write_parquet(cache_path)
                disk_cache.set(
                    cache_key, str(cache_path), expire=86400 * 7
                )
            
            logger.info(
                f"Successfully fetched {len(milb_df)} minor league records from MiLB.com (with expanded stats)"
            )
            return milb_df
        else:
            logger.warning("No minor league stats found via MiLB.com scraping")
    except ImportError:
        logger.debug("MiLB.com scraping module not available")
    except Exception as e:
        logger.warning(f"MiLB.com scraping failed: {e}")

    # Fallback: Try FanGraphs API (faster but no expanded stats)
    try:
        from data_fetch.fetch_milb_fangraphs import fetch_minor_league_pitching_batch_fangraphs
        
        logger.info("Attempting to fetch from FanGraphs API")
        
        # Load player info to get FanGraphs IDs
        # Use provided path or default to config.raw_data_dir
        player_info_file = player_info_path or (config.raw_data_dir / "players.parquet")
        if player_info_file.exists():
            try:
                player_df = pl.read_parquet(player_info_file)
                
                # Check if we have FanGraphs IDs
                if "fangraphs_id" in player_df.columns:
                    # Filter out null fangraphs_ids
                    fg_ids = (
                        player_df.filter(pl.col("fangraphs_id").is_not_null())
                        .select("fangraphs_id")
                        .unique()
                        .to_series()
                        .to_list()
                    )
                    
                    if len(fg_ids) > 0:
                        logger.info(
                            f"Found {len(fg_ids)} players with FanGraphs IDs. "
                            f"Fetching minor league stats from FanGraphs API..."
                        )
                        
                        # Fetch minor league stats
                        fg_df = fetch_minor_league_pitching_batch_fangraphs(
                            fg_ids, start_year, end_year, delay=0.6, 
                            progress_callback=progress_callback,
                            incremental_writer=incremental_writer,
                        )
                        
                        # If using incremental writer, data is already written
                        if incremental_writer:
                            logger.info("Data written incrementally to JSONL")
                            return pl.DataFrame()  # Empty - data already written
                        
                        if len(fg_df) > 0:
                            # Map fangraphs_id back to player_id (MLBAM)
                            result_df = fg_df.join(
                                player_df.select(["player_id", "fangraphs_id"]),
                                left_on="player_id",
                                right_on="fangraphs_id",
                                how="left",
                            ).with_columns([
                                # Use MLBAM ID as player_id
                                pl.when(pl.col("player_id").is_not_null())
                                .then(pl.col("player_id"))
                                .otherwise(pl.col("fangraphs_id"))
                                .alias("player_id")
                            ])
                            
                            # Cache the results
                            if cache_key and disk_cache:
                                cache_path = config.cache_dir / f"{cache_key}.parquet"
                                result_df.write_parquet(cache_path)
                                disk_cache.set(
                                    cache_key, str(cache_path), expire=86400 * 7
                                )
                            
                            logger.info(
                                f"Successfully fetched {len(result_df)} minor league records from FanGraphs"
                            )
                            return result_df
                        else:
                            logger.warning("No minor league stats found via FanGraphs API")
                    else:
                        logger.warning("No FanGraphs IDs found in player info")
                else:
                    logger.warning(
                        "Player info doesn't have fangraphs_id column. "
                        "Run fetch_player_info() first to get FanGraphs IDs."
                    )
            except Exception as e:
                logger.warning(f"Could not read player info: {e}")
        else:
            logger.warning("Player info file not found. Run fetch_player_info() first.")
    except ImportError:
        logger.debug("FanGraphs API module not available")

    # MiLB.com scraping is now handled above as primary source
    # This duplicate section removed

    # Try MLB Stats API as alternative
    try:
        from data_fetch.fetch_milb_mlbapi import fetch_player_minor_league_stats_mlbapi
        logger.debug("MLB Stats API module available")
    except ImportError:
        logger.debug("MLB Stats API module not available")

    # Try Baseball Reference scraping as fallback
    if SCRAPING_AVAILABLE:
        try:
            from data_fetch.scrape_milb import scrape_minor_league_pitching_batch

            # Try to get player Baseball Reference IDs and scrape
            logger.info("Attempting to scrape from Baseball Reference")
            
            # Load player info to get Baseball Reference IDs
            player_info_path = config.raw_data_dir / "players.parquet"
            if player_info_path.exists():
                try:
                    player_df = pl.read_parquet(player_info_path)
                    
                    # Check if we have Baseball Reference IDs
                    if "bbref_id" in player_df.columns:
                        # Filter out null bbref_ids
                        bbref_ids = (
                            player_df.filter(pl.col("bbref_id").is_not_null())
                            .select("bbref_id")
                            .unique()
                            .to_series()
                            .to_list()
                        )
                        
                        if len(bbref_ids) > 0:
                            logger.info(
                                f"Found {len(bbref_ids)} players with Baseball Reference IDs. "
                                f"Scraping minor league stats (this may take a while)..."
                            )
                            
                            # Scrape minor league stats
                            scraped_df = scrape_minor_league_pitching_batch(
                                bbref_ids, start_year, end_year, delay=1.5
                            )
                            
                            if len(scraped_df) > 0:
                                # Map bbref_id back to player_id (MLBAM)
                                # Join with player_df to get MLBAM IDs
                                result_df = scraped_df.join(
                                    player_df.select(["player_id", "bbref_id"]),
                                    left_on="player_id",
                                    right_on="bbref_id",
                                    how="left",
                                ).with_columns([
                                    # Use MLBAM ID as player_id, keep bbref_id for reference
                                    pl.when(pl.col("player_id").is_not_null())
                                    .then(pl.col("player_id"))
                                    .otherwise(pl.col("bbref_id"))
                                    .alias("player_id")
                                ])
                                
                                # Cache the results
                                if cache_key and disk_cache:
                                    cache_path = config.cache_dir / f"{cache_key}.parquet"
                                    result_df.write_parquet(cache_path)
                                    disk_cache.set(
                                        cache_key, str(cache_path), expire=86400 * 7
                                    )
                                
                                logger.info(
                                    f"Successfully scraped {len(result_df)} minor league records"
                                )
                                return result_df
                            else:
                                logger.warning("No minor league stats found via scraping")
                        else:
                            logger.warning("No Baseball Reference IDs found in player info")
                    else:
                        logger.warning(
                            "Player info doesn't have bbref_id column. "
                            "Run fetch_player_info() first to get Baseball Reference IDs."
                        )
                except Exception as e:
                    logger.error(f"Error reading player info or scraping: {e}")
            else:
                logger.warning(
                    f"Player info file not found at {player_info_file}. "
                    "Run fetch_player_info() first to get Baseball Reference IDs for scraping."
                )
        except ImportError:
            logger.warning("Could not import scraping module")
    else:
        logger.warning(
            "Scraping dependencies not available. "
            "Install beautifulsoup4, requests, and lxml to enable Baseball Reference scraping."
        )

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


def get_player_ids_from_mlb_stats(start_year: int, end_year: int) -> list[str]:
    """
    Get list of player IDs who played in MLB during the given years.
    This helps bootstrap the player info fetching process.

    Note: pybaseball's batting_stats_range and pitching_stats_range only support
    years 2008 and later. Years before 2008 will be skipped.

    Args:
        start_year: Starting year
        end_year: Ending year

    Returns:
        List of MLBAM player IDs
    """
    # pybaseball stats functions only support 2008+
    MIN_YEAR = 2008
    effective_start = max(start_year, MIN_YEAR)
    
    if start_year < MIN_YEAR:
        logger.info(
            f"MLB stats API only supports {MIN_YEAR}+. "
            f"Adjusting range from {start_year}-{end_year} to {effective_start}-{end_year}"
        )
    
    if effective_start > end_year:
        logger.warning(
            f"Requested range {start_year}-{end_year} is before {MIN_YEAR}. "
            "Cannot fetch from MLB stats API. Consider using Chadwick Register instead."
        )
        return []

    logger.info(f"Getting player IDs from MLB stats ({effective_start}-{end_year})")

    try:
        from pybaseball import batting_stats_range, pitching_stats_range
        import pandas as pd
    except ImportError:
        logger.warning("pybaseball not available for getting player IDs")
        return []

    player_ids = set()

    try:
        # Get batting stats (includes all players who batted)
        for year in range(effective_start, end_year + 1):
            try:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                batting_df = batting_stats_range(start_date, end_date)
                # Try to get MLBAM ID (key_mlbam) or IDfg
                if "key_mlbam" in batting_df.columns:
                    player_ids.update(
                        batting_df["key_mlbam"].dropna().astype(int).astype(str).tolist()
                    )
                elif "IDfg" in batting_df.columns:
                    # If we only have FanGraphs IDs, we'd need to convert them
                    # For now, log a warning
                    logger.warning(
                        f"Only FanGraphs IDs found for {year}, need conversion to MLBAM"
                    )
                logger.info(f"Found {len(batting_df)} batting records for {year}")
            except Exception as e:
                logger.warning(f"Error fetching batting stats for {year}: {e}")

        # Get pitching stats (includes all pitchers)
        for year in range(effective_start, end_year + 1):
            try:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                pitching_df = pitching_stats_range(start_date, end_date)
                # Try to get MLBAM ID (key_mlbam) or IDfg
                if "key_mlbam" in pitching_df.columns:
                    player_ids.update(
                        pitching_df["key_mlbam"]
                        .dropna()
                        .astype(int)
                        .astype(str)
                        .tolist()
                    )
                elif "IDfg" in pitching_df.columns:
                    logger.warning(
                        f"Only FanGraphs IDs found for {year}, need conversion to MLBAM"
                    )
                logger.info(f"Found {len(pitching_df)} pitching records for {year}")
            except Exception as e:
                logger.warning(f"Error fetching pitching stats for {year}: {e}")

        logger.info(f"Found {len(player_ids)} unique player IDs")
        return list(player_ids)

    except Exception as e:
        logger.error(f"Error getting player IDs: {e}")
        return []


def fetch_player_info(player_ids: Optional[list[str]] = None) -> pl.DataFrame:
    """
    Fetch player biographical information and MLB debut dates.

    Args:
        player_ids: Optional list of player IDs (MLBAM IDs) to fetch. If None, will need
                   to get from other sources (e.g., minor league stats).

    Returns:
        DataFrame with player info including MLB debut dates
    """
    logger.info("Fetching player information")

    try:
        from pybaseball import playerid_reverse_lookup
        import pandas as pd
    except ImportError:
        logger.error("pybaseball not available, falling back to mock data")
        return _get_mock_player_info()

    # If no player IDs provided, use Chadwick Register directly (faster and more reliable)
    if player_ids is None or len(player_ids) == 0:
        logger.info("No player IDs provided, using Chadwick Register")
        try:
            from pybaseball import chadwick_register
            
            logger.info("Loading Chadwick Register...")
            register_pd = chadwick_register(save=False)
            register_df = pl.from_pandas(register_pd)
            
            # Filter to players who debuted in our config range
            register_df = register_df.filter(
                (pl.col("mlb_played_first").is_not_null()) &
                (pl.col("mlb_played_first") >= config.start_year) &
                (pl.col("mlb_played_first") <= config.end_year)
            )
            
            if len(register_df) == 0:
                logger.warning("No players found in Chadwick Register for config range. Returning mock data.")
                return _get_mock_player_info()
        except Exception as e:
            logger.warning(f"Could not load Chadwick Register: {e}. Falling back to MLB stats.")
            player_ids = get_player_ids_from_mlb_stats(
                config.start_year, config.end_year
            )
            if len(player_ids) == 0:
                logger.warning(
                    "Could not get player IDs. Returning mock data. "
                    "You may need to provide player_ids."
                )
                return _get_mock_player_info()
            else:
                # Convert to ints for lookup
                player_id_ints = [int(pid) for pid in player_ids if str(pid).isdigit()]
                if len(player_id_ints) == 0:
                    return _get_mock_player_info()
                # Use playerid_reverse_lookup as fallback
                from pybaseball import playerid_reverse_lookup
                player_info_df = playerid_reverse_lookup(player_id_ints, key_type="mlbam")
                if len(player_info_df) == 0:
                    return _get_mock_player_info()
                register_df = pl.from_pandas(player_info_df)

    logger.info(f"Fetching info for {len(player_ids)} players")

    try:
        # Convert player IDs to integers if they're strings (pybaseball expects ints)
        player_id_ints = []
        for pid in player_ids:
            try:
                player_id_ints.append(int(pid))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert player ID to int: {pid}")
                continue

        if len(player_id_ints) == 0:
            logger.warning("No valid player IDs to fetch")
            return _get_mock_player_info()

        # Fetch player info using pybaseball
        # playerid_reverse_lookup expects MLBAM IDs as integers
        player_info_df = playerid_reverse_lookup(player_id_ints, key_type="mlbam")

        if len(player_info_df) == 0:
            logger.warning(
                f"No player info found for {len(player_id_ints)} IDs. "
                "They may not be in pybaseball's lookup table."
            )
            return _get_mock_player_info()

        # Convert to our schema format
        # Note: pybaseball returns mlb_played_first (year), not exact date
        # We'll use year-01-01 as approximate debut date, can refine later
        result_df = pl.from_pandas(player_info_df).select([
            pl.col("key_mlbam").cast(pl.String).alias("player_id"),
            pl.col("name_first").alias("name_first"),
            pl.col("name_last").alias("name_last"),
            # Birth date not directly available, would need additional lookup
            pl.lit(None).alias("birth_date"),
            # Convert debut year to approximate date (YYYY-01-01)
            pl.when(pl.col("mlb_played_first").is_not_null())
            .then(
                pl.col("mlb_played_first")
                .cast(pl.Int64)
                .cast(pl.String)
                + "-01-01"
            )
            .otherwise(None)
            .alias("mlb_debut"),
            # Include Baseball Reference ID for scraping minor league stats
            pl.col("key_bbref").cast(pl.String).alias("bbref_id"),
            # Include FanGraphs ID for API access to minor league stats
            pl.col("key_fangraphs").cast(pl.String).alias("fangraphs_id"),
        ])

        logger.info(f"Successfully fetched info for {len(result_df)} players")
        return result_df

    except Exception as e:
        logger.error(f"Error fetching player info: {e}")
        logger.warning("Falling back to mock data")
        return _get_mock_player_info()


def _get_mock_player_info() -> pl.DataFrame:
    """Return mock player data for testing."""
    mock_data = {
        "player_id": [f"player_{i}" for i in range(100)],
        "name_first": [f"First{i}" for i in range(100)],
        "name_last": [f"Last{i}" for i in range(100)],
        "birth_date": [None] * 100,  # Not available in mock
        "mlb_debut": ["2020-04-01" for _ in range(100)],
    }
    return pl.DataFrame(mock_data)


def run_ingestion(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
    max_players: Optional[int] = None,
) -> dict[str, Path]:
    """
    Run full data ingestion pipeline.

    Args:
        start_year: Starting year (defaults to config)
        end_year: Ending year (defaults to config)
        output_dir: Output directory (defaults to config.raw_data_dir)
        force: If True, re-fetch data even if files already exist
        max_players: Optional limit on number of players to fetch (useful for testing)

    Returns:
        Dictionary mapping dataset names to file paths
    """
    start_year = start_year or config.start_year
    end_year = end_year or config.end_year
    output_dir = output_dir or config.raw_data_dir

    # Initialize progress tracker
    status_file = output_dir / ".ingestion_status.json"
    tracker = IngestionTracker(status_file)
    
    logger.info(f"Starting data ingestion: {start_year}-{end_year}")
    logger.info(f"Progress tracking: {status_file}")

    # Define output file paths
    player_path = output_dir / "players.parquet"
    all_star_path = output_dir / "all_star_rosters.parquet"
    milb_path = output_dir / "minor_league_pitching.parquet"
    
    # Start tracking
    tracker.start_ingestion(start_year, end_year, output_dir, force)

    # Check if files already exist (unless force is True)
    # Also check tracker for resume capability
    if not force:
        existing_files = []
        if player_path.exists() and tracker.is_step_completed("players"):
            existing_files.append("players")
        if all_star_path.exists() and tracker.is_step_completed("all_star_rosters"):
            existing_files.append("all_star_rosters")
        if milb_path.exists() and tracker.is_step_completed("minor_league_pitching"):
            existing_files.append("minor_league_pitching")
        
        if existing_files:
            logger.info(
                f"Found existing completed data files: {', '.join(existing_files)}. "
                "Skipping re-fetch. Use --force to re-fetch."
            )
            # Return existing files, only fetch missing ones
            result = {}
            if player_path.exists() and tracker.is_step_completed("players"):
                result["players"] = player_path
            if all_star_path.exists() and tracker.is_step_completed("all_star_rosters"):
                result["all_star_rosters"] = all_star_path
            if milb_path.exists() and tracker.is_step_completed("minor_league_pitching"):
                result["minor_league_pitching"] = milb_path
            
            # Fetch any missing files
            if not player_path.exists() or not tracker.is_step_completed("players"):
                tracker.mark_step_started("players")
                logger.info("Fetching missing player info...")
                # Fetch player info
                try:
                    from pybaseball import chadwick_register
                    register_pd = chadwick_register(save=False)
                    register_df = pl.from_pandas(register_pd)
                    debut_window_start = max(start_year - 5, 2005)
                    filtered_df = register_df.filter(
                        (pl.col("mlb_played_first").is_not_null()) &
                        (pl.col("mlb_played_first") >= debut_window_start) &
                        (pl.col("mlb_played_first") <= end_year)
                    )
                    if max_players is not None and len(filtered_df) > max_players:
                        filtered_df = filtered_df.head(max_players)
                    player_df = filtered_df.select([
                        pl.col("key_mlbam").cast(pl.String).alias("player_id"),
                        pl.col("name_first").alias("name_first"),
                        pl.col("name_last").alias("name_last"),
                        pl.lit(None).alias("birth_date"),
                        pl.when(pl.col("mlb_played_first").is_not_null())
                        .then(
                            pl.col("mlb_played_first")
                            .cast(pl.Int64)
                            .cast(pl.String)
                            + "-01-01"
                        )
                        .otherwise(None)
                        .alias("mlb_debut"),
                        pl.col("key_bbref").cast(pl.String).alias("bbref_id"),
                        pl.col("key_fangraphs").cast(pl.String).alias("fangraphs_id"),
                    ])
                    player_df.write_parquet(player_path)
                    result["players"] = player_path
                    tracker.mark_step_completed("players", player_path)
                except Exception as e:
                    logger.error(f"Could not fetch player info: {e}")
                    tracker.mark_step_failed("players", str(e))
            
            if not all_star_path.exists() or not tracker.is_step_completed("all_star_rosters"):
                tracker.mark_step_started("all_star_rosters")
                logger.info("Fetching missing All-Star rosters...")
                try:
                    all_star_df = fetch_all_star_rosters(start_year, end_year)
                    all_star_df.write_parquet(all_star_path)
                    result["all_star_rosters"] = all_star_path
                    tracker.mark_step_completed("all_star_rosters", all_star_path)
                except Exception as e:
                    logger.error(f"Could not fetch All-Star rosters: {e}")
                    tracker.mark_step_failed("all_star_rosters", str(e))
            
            if not milb_path.exists() or not tracker.is_step_completed("minor_league_pitching"):
                tracker.mark_step_started("minor_league_pitching")
                logger.info("Fetching missing minor league pitching...")
                try:
                    milb_df = fetch_minor_league_pitching(
                        start_year,
                        end_year,
                        cache_key=f"milb_pitching_{start_year}_{end_year}",
                        player_info_path=player_path if player_path.exists() else None,
                    )
                    milb_df.write_parquet(milb_path)
                    result["minor_league_pitching"] = milb_path
                    tracker.mark_step_completed("minor_league_pitching", milb_path)
                except Exception as e:
                    logger.error(f"Could not fetch minor league pitching: {e}")
                    tracker.mark_step_failed("minor_league_pitching", str(e))
            
            tracker.finish_ingestion()
            return result

    # Fetch player info FIRST (needed for minor league fetching)
    step_name = "players"
    if not tracker.is_step_completed(step_name):
        tracker.mark_step_started(step_name)
        logger.info("Step 1: Fetching player information")
        # Use Chadwick Register directly - it's faster and more reliable than MLB stats API
        # Filter by players who debuted in our year range
        try:
            from pybaseball import chadwick_register
            import pandas as pd
            
            logger.info("Loading Chadwick Register...")
            register_pd = chadwick_register(save=False)
            register_df = pl.from_pandas(register_pd)
            
            # Filter to players who debuted in a reasonable window
            # We want players who:
            # 1. Debuted in MLB (for labeling)
            # 2. Could have minor league stats in our year range
            # Strategy: Include players who debuted between (start_year - 5) and end_year
            # This captures players who had MiLB stats in our range before their debut
            debut_window_start = max(start_year - 5, 2005)  # Don't go before 2005 (FanGraphs limit)
            filtered_df = register_df.filter(
                (pl.col("mlb_played_first").is_not_null()) &
                (pl.col("mlb_played_first") >= debut_window_start) &
                (pl.col("mlb_played_first") <= end_year)
            )
            
            # Limit number of players if specified (useful for testing)
            if max_players is not None and len(filtered_df) > max_players:
                logger.info(f"Limiting to {max_players} players (for testing)")
                filtered_df = filtered_df.head(max_players)
            
            player_df = filtered_df.select([
                pl.col("key_mlbam").cast(pl.String).alias("player_id"),
                pl.col("name_first").alias("name_first"),
                pl.col("name_last").alias("name_last"),
                pl.lit(None).alias("birth_date"),
                pl.when(pl.col("mlb_played_first").is_not_null())
                .then(
                    pl.col("mlb_played_first")
                    .cast(pl.Int64)
                    .cast(pl.String)
                    + "-01-01"
                )
                .otherwise(None)
                .alias("mlb_debut"),
                pl.col("key_bbref").cast(pl.String).alias("bbref_id"),
                pl.col("key_fangraphs").cast(pl.String).alias("fangraphs_id"),
            ])
            
            logger.info(f"Loaded {len(player_df)} players from Chadwick Register")
        except Exception as e:
            logger.warning(f"Could not load Chadwick Register: {e}. Falling back to fetch_player_info()")
            player_df = fetch_player_info()
        
        player_df.write_parquet(player_path)
        logger.info(f"Saved player info to {player_path}")
        tracker.mark_step_completed(step_name, player_path)
    else:
        logger.info(f"Step 1: Skipping (already completed)")
        player_df = pl.read_parquet(player_path)

    # Fetch All-Star rosters
    step_name = "all_star_rosters"
    if not tracker.is_step_completed(step_name):
        tracker.mark_step_started(step_name)
        logger.info("Step 2: Fetching All-Star rosters")
        try:
            all_star_df = fetch_all_star_rosters(start_year, end_year)
            all_star_df.write_parquet(all_star_path)
            logger.info(f"Saved All-Star rosters to {all_star_path}")
            tracker.mark_step_completed(step_name, all_star_path)
        except Exception as e:
            logger.error(f"Failed to fetch All-Star rosters: {e}")
            tracker.mark_step_failed(step_name, str(e))
            raise
    else:
        logger.info(f"Step 2: Skipping (already completed)")

    # Fetch minor league pitching (now that we have player info with FanGraphs IDs)
    step_name = "minor_league_pitching"
    if not tracker.is_step_completed(step_name):
        tracker.mark_step_started(step_name)
        logger.info("Step 3: Fetching minor league pitching stats")
        logger.info("This may take 30-60 minutes depending on number of players...")
        try:
            # Progress callback for tracking
            def progress_callback(current: int, total: int, player_id: str) -> None:
                if current > 0 and current % 50 == 0:
                    pct = (current / total) * 100
                    tracker.update_progress(
                        step_name,
                        {
                            "current": current,
                            "total": total,
                            "percent": pct,
                            "current_player": player_id,
                        },
                    )
            
            # Use incremental writer for resilience
            incremental_writer = IncrementalWriter(milb_path, batch_size=50)
            
            milb_df = fetch_minor_league_pitching(
                start_year,
                end_year,
                cache_key=f"milb_pitching_{start_year}_{end_year}",
                player_info_path=player_path,
                progress_callback=progress_callback,
                incremental_writer=incremental_writer,
            )
            
            # Finalize: convert JSONL to Parquet
            final_path = incremental_writer.finalize()
            record_count = incremental_writer.get_current_count()
            logger.info(f"Saved MiLB pitching to {final_path} ({record_count} records)")
            tracker.mark_step_completed(step_name, final_path)
        except Exception as e:
            logger.error(f"Failed to fetch minor league pitching: {e}")
            tracker.mark_step_failed(step_name, str(e))
            raise
    else:
        logger.info(f"Step 3: Skipping (already completed)")

    # Add small delay to avoid rate limiting
    time.sleep(1)
    
    tracker.finish_ingestion()
    
    # Print summary
    summary = tracker.get_progress_summary()
    logger.info("=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Completed steps: {', '.join(summary['completed_steps'])}")
    logger.info(f"Status file: {status_file}")

    return {
        "all_star_rosters": all_star_path,
        "minor_league_pitching": milb_path,
        "players": player_path,
    }

