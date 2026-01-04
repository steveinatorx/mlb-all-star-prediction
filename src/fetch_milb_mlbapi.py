"""Fetch minor league pitching stats using MLB Stats API.

Note: MLB Stats API may have limited minor league data.
This module attempts to fetch what's available via the official API.
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
    logger.warning("requests not available for MLB API calls")


def fetch_player_minor_league_stats_mlbapi(
    player_mlbam_id: str, start_year: int, end_year: int
) -> pl.DataFrame:
    """
    Fetch minor league stats for a player using MLB Stats API.

    Args:
        player_mlbam_id: MLB Advanced Media player ID
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
            # Try different endpoints for minor league stats
            # Note: MLB Stats API may not have comprehensive minor league stats
            # This is exploratory - may need to fall back to scraping
            
            # Try getting stats by level/team
            url = (
                f"http://statsapi.mlb.com/api/v1/people/{player_mlbam_id}/stats?"
                f"stats=statSplits&group=pitching&season={year}"
            )

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse the response
            # Structure may vary - need to inspect actual responses
            if "stats" in data and len(data["stats"]) > 0:
                for stat_group in data["stats"]:
                    if "splits" in stat_group:
                        for split in stat_group["splits"]:
                            # Check if this is minor league data
                            # Minor league might be indicated by league/level info
                            if split.get("group", {}).get("displayName") == "pitching":
                                # Extract relevant stats
                                stat_data = split.get("stat", {})
                                if stat_data:
                                    row = {
                                        "player_id": player_mlbam_id,
                                        "season": year,
                                        **stat_data,
                                    }
                                    all_stats.append(row)

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.debug(f"Error fetching stats for {player_mlbam_id} year {year}: {e}")
            continue

    if not all_stats:
        return pl.DataFrame()

    df = pl.DataFrame(all_stats)
    logger.info(f"Fetched {len(df)} records for player {player_mlbam_id}")
    return df


def fetch_minor_league_team_stats(
    team_id: int, season: int, sport_id: int = 11
) -> pl.DataFrame:
    """
    Fetch stats for all players on a minor league team.

    Args:
        team_id: Team ID
        season: Season year
        sport_id: Sport ID (11 = minor leagues)

    Returns:
        DataFrame with player stats
    """
    if not REQUESTS_AVAILABLE:
        return pl.DataFrame()

    try:
        # Get roster
        roster_url = (
            f"http://statsapi.mlb.com/api/v1/teams/{team_id}/roster?"
            f"sportId={sport_id}&season={season}"
        )
        response = requests.get(roster_url, timeout=10)
        response.raise_for_status()
        roster_data = response.json()

        # Extract player IDs and fetch stats for each
        player_ids = []
        if "roster" in roster_data:
            for person in roster_data["roster"]:
                if "person" in person and "id" in person["person"]:
                    player_ids.append(str(person["person"]["id"]))

        logger.info(f"Found {len(player_ids)} players on team {team_id}")

        # Fetch stats for each player
        all_stats = []
        for player_id in player_ids:
            stats = fetch_player_minor_league_stats_mlbapi(
                player_id, season, season
            )
            if len(stats) > 0:
                all_stats.append(stats)
            time.sleep(0.5)

        if all_stats:
            return pl.concat(all_stats)
        return pl.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching team stats: {e}")
        return pl.DataFrame()

