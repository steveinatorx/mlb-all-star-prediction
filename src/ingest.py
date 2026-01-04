"""Data ingestion module for pulling raw baseball data."""

import time
from pathlib import Path
from typing import Optional

import polars as pl
from diskcache import Cache
from loguru import logger
from pybaseball import cache

from src.config import config

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

    # Fallback: Try scraping Baseball Reference
    # TODO: Implement Baseball Reference scraping
    # For now, return mock data
    logger.warning(
        "All-Star roster fetching not fully implemented. "
        "Options: Baseball Reference scraping or MLB Stats API. "
        "Returning mock data for now."
    )
    
    # Generate mock data for the requested years
    mock_data = {
        "player_id": [f"mock_player_{i}" for i in range((end_year - start_year + 1) * 2)],
        "season": [year for year in range(start_year, end_year + 1) for _ in range(2)],
        "is_all_star": [True] * ((end_year - start_year + 1) * 2),
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

    # Try FanGraphs API first (works well, has minor league data)
    try:
        from src.fetch_milb_fangraphs import fetch_minor_league_pitching_batch_fangraphs
        
        logger.info("Attempting to fetch from FanGraphs API")
        
        # Load player info to get FanGraphs IDs
        player_info_path = config.raw_data_dir / "players.parquet"
        if player_info_path.exists():
            try:
                player_df = pl.read_parquet(player_info_path)
                
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
                            fg_ids, start_year, end_year, delay=0.6
                        )
                        
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

    # Try MLB Stats API as alternative
    try:
        from src.fetch_milb_mlbapi import fetch_player_minor_league_stats_mlbapi
        logger.debug("MLB Stats API module available")
    except ImportError:
        logger.debug("MLB Stats API module not available")

    # Try Baseball Reference scraping as fallback
    if SCRAPING_AVAILABLE:
        try:
            from src.scrape_milb import scrape_minor_league_pitching_batch

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
                    "Player info file not found. "
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

    # Fetch player info FIRST (needed for minor league fetching)
    logger.info("Step 1: Fetching player information")
    # Use Chadwick Register directly - it's faster and more reliable than MLB stats API
    # Filter by players who debuted in our year range
    try:
        from pybaseball import chadwick_register
        import pandas as pd
        
        logger.info("Loading Chadwick Register...")
        register_pd = chadwick_register(save=False)
        register_df = pl.from_pandas(register_pd)
        
        # Filter to players who debuted in our range (or before, to catch minor league data)
        # We want players who could have minor league stats in our range
        player_df = register_df.filter(
            (pl.col("mlb_played_first").is_not_null()) &
            (pl.col("mlb_played_first") <= end_year)
        ).select([
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
    
    player_path = output_dir / "players.parquet"
    player_df.write_parquet(player_path)
    logger.info(f"Saved player info to {player_path}")

    # Fetch All-Star rosters
    logger.info("Step 2: Fetching All-Star rosters")
    all_star_df = fetch_all_star_rosters(start_year, end_year)
    all_star_path = output_dir / "all_star_rosters.parquet"
    all_star_df.write_parquet(all_star_path)
    logger.info(f"Saved All-Star rosters to {all_star_path}")

    # Fetch minor league pitching (now that we have player info with FanGraphs IDs)
    logger.info("Step 3: Fetching minor league pitching stats")
    milb_df = fetch_minor_league_pitching(
        start_year, end_year, cache_key=f"milb_pitching_{start_year}_{end_year}"
    )
    milb_path = output_dir / "minor_league_pitching.parquet"
    milb_df.write_parquet(milb_path)
    logger.info(f"Saved MiLB pitching to {milb_path}")

    # Add small delay to avoid rate limiting
    time.sleep(1)

    return {
        "all_star_rosters": all_star_path,
        "minor_league_pitching": milb_path,
        "players": player_path,
    }

