"""Fetch minor league pitching stats using FanGraphs API.

FanGraphs has a public API endpoint that returns JSON data.
URL pattern: https://www.fangraphs.com/api/players/stats?playerid={playerid}&position=P&type=0&season={year}
"""

import time
from typing import Optional

import polars as pl
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available for FanGraphs API calls")


def fetch_player_minor_league_stats_fangraphs(
    player_fg_id: str, start_year: int, end_year: int
) -> pl.DataFrame:
    """
    Fetch minor league pitching stats for a player using FanGraphs API.

    Args:
        player_fg_id: FanGraphs player ID (can get from pybaseball playerid_lookup)
        start_year: Start year
        end_year: End year

    Returns:
        DataFrame with minor league pitching stats
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests not available")
        return pl.DataFrame()

    all_stats = []

    for year in range(start_year, end_year + 1):
        try:
            url = (
                f"https://www.fangraphs.com/api/players/stats?"
                f"playerid={player_fg_id}&position=P&type=0&gds=&gde=&season={year}"
            )

            # Use curl-like headers to avoid blocking
            headers = {
                "User-Agent": "curl/7.68.0",
                "Accept": "*/*"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response_data = response.json()

            # FanGraphs returns a dict with 'data' key containing the stats
            if isinstance(response_data, dict) and "data" in response_data:
                data = response_data["data"]
            elif isinstance(response_data, list):
                data = response_data
            else:
                data = []

            # Parse the response - FanGraphs returns stats in a list
            if isinstance(data, list) and len(data) > 0:
                for stat_entry in data:
                    # Check if this is minor league data
                    # Minor league entries have negative type values or specific AbbLevel
                    abb_level = stat_entry.get("AbbLevel", "")
                    team_name = stat_entry.get("ateam", "")
                    
                    # Filter for minor league levels (not MLB)
                    if abb_level and abb_level not in ["MLB", ""]:
                        # Extract pitching stats
                        row = {
                            "player_id": player_fg_id,  # We'll map this to MLBAM ID later
                            "season": year,
                            "level": abb_level,
                            "team": team_name,
                            "league": stat_entry.get("leagueUrl", ""),
                            "games": stat_entry.get("G", 0),
                            "games_started": stat_entry.get("GS", 0),
                            "complete_games": stat_entry.get("CG", 0),
                            "innings_pitched": stat_entry.get("IP", 0.0),
                            "hits": stat_entry.get("H", 0),
                            "runs": stat_entry.get("R", 0),
                            "earned_runs": stat_entry.get("ER", 0),
                            "walks": stat_entry.get("BB", 0),
                            "strikeouts": stat_entry.get("SO", 0),
                            "home_runs": stat_entry.get("HR", 0),
                            "wins": stat_entry.get("W", 0),
                            "losses": stat_entry.get("L", 0),
                            "saves": stat_entry.get("SV", 0),
                            "era": stat_entry.get("ERA", 0.0),
                            "whip": stat_entry.get("WHIP", 0.0),
                            "k_per_9": stat_entry.get("K/9", 0.0),
                            "bb_per_9": stat_entry.get("BB/9", 0.0),
                        }
                        all_stats.append(row)

            time.sleep(0.5)  # Rate limiting - be respectful

        except Exception as e:
            logger.debug(f"Error fetching FanGraphs stats for {player_fg_id} year {year}: {e}")
            continue

    if not all_stats:
        return pl.DataFrame()

    df = pl.DataFrame(all_stats)
    logger.info(f"Fetched {len(df)} minor league records for player {player_fg_id}")
    return df


def fetch_minor_league_pitching_batch_fangraphs(
    player_fg_ids: list[str],
    start_year: int,
    end_year: int,
    delay: float = 0.5,
) -> pl.DataFrame:
    """
    Fetch minor league stats for multiple players using FanGraphs API.

    Args:
        player_fg_ids: List of FanGraphs player IDs
        start_year: Start year
        end_year: End year
        delay: Delay between requests (seconds)

    Returns:
        Combined DataFrame with stats for all players
    """
    if not REQUESTS_AVAILABLE:
        logger.error("requests not available")
        return pl.DataFrame()

    all_dfs = []

    logger.info(f"Fetching FanGraphs minor league stats for {len(player_fg_ids)} players")

    for i, player_id in enumerate(player_fg_ids):
        if i > 0 and i % 10 == 0:
            logger.info(f"Progress: {i}/{len(player_fg_ids)} players")

        df = fetch_player_minor_league_stats_fangraphs(player_id, start_year, end_year)

        if len(df) > 0:
            all_dfs.append(df)

        time.sleep(delay)

    if not all_dfs:
        return pl.DataFrame()

    result = pl.concat(all_dfs)
    logger.info(f"Successfully fetched {len(result)} minor league records")
    return result

