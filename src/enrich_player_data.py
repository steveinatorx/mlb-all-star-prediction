"""Enrich player data with birth dates and draft information."""

import time
from typing import Optional

import polars as pl
from loguru import logger

from src.config import config

# Note: Baseball Reference scraping is currently disabled due to 403 blocking
# Using MLB Stats API instead for birth dates and draft info


def fetch_player_info_from_mlb_api(player_id: str, delay: float = 0.2) -> Optional[dict]:
    """
    Fetch player information (birth date, draft info) from MLB Stats API.
    
    Uses the /api/v1/people/{id} endpoint which provides:
    - birthDate
    - draftYear
    - draftRound
    - draftPick (overall pick number)
    
    Args:
        player_id: MLBAM player ID
        delay: Delay between requests (seconds)
        
    Returns:
        Dictionary with birth_date, draft_year, draft_round, draft_position, or None
    """
    if not player_id or not str(player_id).isdigit():
        return None
        
    try:
        import requests
        
        base_url = "https://statsapi.mlb.com/api/v1"
        url = f"{base_url}/people/{player_id}"
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.debug(f"MLB API returned {response.status_code} for player {player_id}")
            time.sleep(delay)
            return None
        
        data = response.json()
        if 'people' not in data or len(data['people']) == 0:
            time.sleep(delay)
            return None
        
        person = data['people'][0]
        
        result = {}
        
        # Birth date
        birth_date = person.get('birthDate')
        if birth_date:
            result['birth_date'] = birth_date
        
        # Draft information
        # NOTE: MLB API only provides draftYear, not draftRound or draftPick
        # For full draft details, would need to use pybaseball amateur_draft or alternative source
        draft_year = person.get('draftYear')
        if draft_year:
            result['draft_year'] = int(draft_year)
        
        # draftRound and draftPick are not in MLB API person endpoint
        # These would need to come from pybaseball amateur_draft or alternative source
        
        time.sleep(delay)  # Rate limiting
        
        if result:
            return result
        return None
        
    except Exception as e:
        logger.debug(f"Error fetching player info from MLB API for {player_id}: {e}")
        time.sleep(delay)
        return None


def fetch_birth_date_from_bref(bbref_id: str, delay: float = 1.0) -> Optional[str]:
    """
    Fetch birth date from Baseball Reference player page.
    
    Args:
        bbref_id: Baseball Reference player ID
        delay: Delay between requests (seconds)
        
    Returns:
        Birth date as YYYY-MM-DD string, or None if not found
    """
    if not bbref_id or bbref_id == "None":
        return None
        
    # NOTE: Baseball Reference is currently blocking requests (403 Forbidden)
    # This function is disabled until we find an alternative source
    # Options: pybaseball BRefSession (may still be blocked), paid API, or skip
    logger.debug("Birth date fetching from Baseball Reference is currently disabled (403 blocking)")
    return None
    
    # Uncomment below if Baseball Reference access is restored:
    # try:
    #     import requests
    #     from bs4 import BeautifulSoup
    #     url = f"https://www.baseball-reference.com/players/{bbref_id[0]}/{bbref_id}.shtml"
    #     session = requests.Session()
    #     session.headers.update({
    #         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    #         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    #         'Accept-Language': 'en-US,en;q=0.5',
    #     })
    #     response = session.get(url, timeout=10)
    #     response.raise_for_status()
    #     soup = BeautifulSoup(response.content, 'html.parser')
    #     birth_info = soup.find('span', {'data-birth': True})
    #     if birth_info:
    #         birth_date_str = birth_info.get('data-birth')
    #         if birth_date_str:
    #             return birth_date_str
    #     time.sleep(delay)
    #     return None
    # except Exception as e:
    #     logger.debug(f"Error fetching birth date for {bbref_id}: {e}")
    #     time.sleep(delay)
    #     return None


def fetch_draft_data_from_pybaseball(start_year: int = 2000, end_year: int = 2023) -> pl.DataFrame:
    """
    Fetch draft data using pybaseball's amateur_draft function.
    
    NOTE: This may fail if Baseball Reference is blocking requests (similar to other BR scraping).
    The function will return an empty DataFrame if it can't fetch data.
    
    Args:
        start_year: First draft year to fetch
        end_year: Last draft year to fetch
        
    Returns:
        DataFrame with draft information
    """
    try:
        from pybaseball import amateur_draft
        import pandas as pd
        
        logger.info(f"Fetching draft data for {start_year}-{end_year}")
        logger.warning("Note: This may fail if Baseball Reference is blocking (similar to other BR scraping)")
        
        all_drafts = []
        successful_years = 0
        for year in range(start_year, end_year + 1):
            try:
                # amateur_draft requires year and round
                # We'll fetch all rounds (typically 1-40, but can vary)
                year_drafts = []
                for round_num in range(1, 41):  # Typically 40 rounds, but may vary
                    try:
                        draft_df = amateur_draft(year, round_num)
                        if len(draft_df) > 0:
                            draft_df['draft_year'] = year
                            draft_df['draft_round'] = round_num
                            year_drafts.append(draft_df)
                        time.sleep(0.2)  # Rate limiting
                    except (ValueError, Exception) as e:
                        # "No tables found" or other errors - this round may not exist
                        # Continue to next round
                        if round_num == 1:
                            # If round 1 fails, the year probably doesn't have data or is blocked
                            logger.debug(f"Round 1 failed for {year}: {e}. Skipping year.")
                            break
                        continue
                
                if year_drafts:
                    year_combined = pd.concat(year_drafts, ignore_index=True)
                    all_drafts.append(year_combined)
                    successful_years += 1
                    logger.info(f"✅ Fetched {len(year_combined)} picks from {year} draft")
                time.sleep(0.5)  # Rate limiting between years
            except Exception as e:
                logger.debug(f"Error fetching {year} draft: {e}")
                continue
        
        if successful_years == 0:
            logger.warning("⚠️  No draft data fetched - Baseball Reference may be blocking")
            logger.warning("   Consider alternative sources or manual data entry")
        
        if not all_drafts:
            logger.warning("No draft data fetched")
            return pl.DataFrame({
                'key_mlbam': [],
                'draft_round': [],
                'draft_year': [],
                'draft_position': [],
            })
        
        # Combine all drafts
        combined_draft = pd.concat(all_drafts, ignore_index=True)
        draft_pl = pl.from_pandas(combined_draft)
        
        # Map columns to our schema
        # pybaseball returns different column names - need to check
        column_mapping = {}
        if 'mlbamID' in draft_pl.columns:
            column_mapping['mlbamID'] = 'player_id'
        elif 'key_mlbam' in draft_pl.columns:
            column_mapping['key_mlbam'] = 'player_id'
        
        if 'Rd' in draft_pl.columns:
            column_mapping['Rd'] = 'draft_round'
        elif 'Round' in draft_pl.columns:
            column_mapping['Round'] = 'draft_round'
        
        if 'OvPck' in draft_pl.columns:
            column_mapping['OvPck'] = 'draft_position'
        elif 'Overall' in draft_pl.columns:
            column_mapping['Overall'] = 'draft_position'
        
        # Select and rename columns
        select_cols = []
        rename_dict = {}
        
        for old_col, new_col in column_mapping.items():
            if old_col in draft_pl.columns:
                select_cols.append(old_col)
                rename_dict[old_col] = new_col
        
        if 'draft_year' in draft_pl.columns:
            select_cols.append('draft_year')
        
        if select_cols:
            result = draft_pl.select(select_cols).rename(rename_dict)
            # Convert player_id to string
            if 'player_id' in result.columns:
                result = result.with_columns([
                    pl.col('player_id').cast(pl.Utf8).alias('player_id')
                ])
            return result
        else:
            logger.warning("Could not map draft columns")
            return pl.DataFrame({
                'player_id': [],
                'draft_round': [],
                'draft_year': [],
                'draft_position': [],
            })
            
    except ImportError:
        logger.error("pybaseball not available for draft data")
        return pl.DataFrame({
            'player_id': [],
            'draft_round': [],
            'draft_year': [],
            'draft_position': [],
        })
    except Exception as e:
        logger.error(f"Error fetching draft data: {e}")
        return pl.DataFrame({
            'player_id': [],
            'draft_round': [],
            'draft_year': [],
            'draft_position': [],
        })


def enrich_players_with_mlb_api_info(
    players_df: pl.DataFrame,
    batch_size: int = 100,
    delay: float = 0.2,
    max_players: Optional[int] = None,
) -> pl.DataFrame:
    """
    Enrich player DataFrame with birth dates and draft info from MLB Stats API.
    
    Uses /api/v1/people/{id} endpoint which provides:
    - birthDate
    - draftYear, draftRound, draftPick
    
    Args:
        players_df: Player DataFrame with player_id (MLBAM ID) column
        batch_size: Number of players to process before logging
        delay: Delay between requests (seconds) - MLB API is fast, can use shorter delay
        max_players: Maximum number of players to process (for testing)
        
    Returns:
        Enriched DataFrame with birth_date, draft_year, draft_round, draft_position columns
    """
    logger.info("Enriching players with birth dates and draft info from MLB Stats API")
    
    if 'player_id' not in players_df.columns:
        logger.warning("No player_id column found, cannot fetch player info")
        return players_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("birth_date"),
            pl.lit(None).cast(pl.Int64).alias("draft_year"),
            pl.lit(None).cast(pl.Int64).alias("draft_round"),
            pl.lit(None).cast(pl.Int64).alias("draft_position"),
        ])
    
    # Filter to players without complete info
    # Check which columns exist before filtering
    filter_conditions = []
    
    if 'birth_date' in players_df.columns:
        filter_conditions.append(pl.col("birth_date").is_null())
    else:
        # If column doesn't exist, we need to enrich all players
        filter_conditions.append(pl.lit(True))
    
    if 'draft_year' in players_df.columns:
        filter_conditions.append(pl.col("draft_year").is_null())
    else:
        # If column doesn't exist, we need to enrich all players
        filter_conditions.append(pl.lit(True))
    
    # Combine conditions with OR (enrich if either is missing)
    if len(filter_conditions) > 1:
        players_to_enrich = players_df.filter(
            filter_conditions[0] | filter_conditions[1]
        )
    else:
        players_to_enrich = players_df.filter(filter_conditions[0])
    
    if max_players:
        players_to_enrich = players_to_enrich.head(max_players)
    
    logger.info(f"Fetching info for {len(players_to_enrich)} players")
    
    enriched_data = []
    for i, row in enumerate(players_to_enrich.iter_rows(named=True), 1):
        player_id = row['player_id']
        
        if i % batch_size == 0:
            logger.info(f"Processed {i}/{len(players_to_enrich)} players")
        
        player_info = fetch_player_info_from_mlb_api(player_id, delay=delay)
        if player_info:
            player_info['player_id'] = player_id
            enriched_data.append(player_info)
    
    logger.info(f"Successfully fetched info for {len(enriched_data)} players")
    
    # Update DataFrame
    if enriched_data:
        enriched_df = pl.DataFrame(enriched_data)
        
        result_df = players_df.join(
            enriched_df,
            on='player_id',
            how='left'
        )
        
        # Update columns, preferring new data over existing
        # Only update columns that exist in the enriched data
        update_exprs = []
        cols_to_drop = []
        
        # Handle birth_date
        if 'birth_date_right' in result_df.columns:
            update_exprs.append(
                pl.when(pl.col("birth_date_right").is_not_null())
                .then(pl.col("birth_date_right"))
                .otherwise(pl.col("birth_date"))
                .alias("birth_date")
            )
            cols_to_drop.append("birth_date_right")
        
        # Handle draft_year
        if 'draft_year_right' in result_df.columns:
            update_exprs.append(
                pl.when(pl.col("draft_year_right").is_not_null())
                .then(pl.col("draft_year_right"))
                .otherwise(pl.col("draft_year"))
                .alias("draft_year")
            )
            cols_to_drop.append("draft_year_right")
        
        # Handle draft_round (may not exist in enriched data - MLB API doesn't provide it)
        if 'draft_round_right' in result_df.columns:
            update_exprs.append(
                pl.when(pl.col("draft_round_right").is_not_null())
                .then(pl.col("draft_round_right"))
                .otherwise(pl.col("draft_round"))
                .alias("draft_round")
            )
            cols_to_drop.append("draft_round_right")
        
        # Handle draft_position (may not exist in enriched data - MLB API doesn't provide it)
        if 'draft_position_right' in result_df.columns:
            update_exprs.append(
                pl.when(pl.col("draft_position_right").is_not_null())
                .then(pl.col("draft_position_right"))
                .otherwise(pl.col("draft_position"))
                .alias("draft_position")
            )
            cols_to_drop.append("draft_position_right")
        
        if update_exprs:
            result_df = result_df.with_columns(update_exprs)
        
        if cols_to_drop:
            result_df = result_df.drop(cols_to_drop)
    else:
        result_df = players_df
    
    # Ensure columns exist
    if 'birth_date' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Utf8).alias("birth_date")])
    if 'draft_year' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Int64).alias("draft_year")])
    if 'draft_round' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Int64).alias("draft_round")])
    if 'draft_position' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Int64).alias("draft_position")])
    
    return result_df


def enrich_players_with_draft_info_pybaseball(
    players_df: pl.DataFrame,
    start_year: int = 2000,
    end_year: int = 2023,
) -> pl.DataFrame:
    """
    Enrich player DataFrame with draft information using pybaseball.
    
    Args:
        players_df: Player DataFrame with player_id column
        start_year: First draft year to fetch
        end_year: Last draft year to fetch
        
    Returns:
        Enriched DataFrame with draft_round, draft_year, draft_position columns
    """
    logger.info("Enriching players with draft information from pybaseball")
    
    # Fetch draft data
    draft_df = fetch_draft_data_from_pybaseball(start_year, end_year)
    
    if len(draft_df) == 0:
        logger.warning("No draft data fetched, adding null columns")
        return players_df.with_columns([
            pl.lit(None).cast(pl.Int64).alias("draft_round"),
            pl.lit(None).cast(pl.Int64).alias("draft_year"),
            pl.lit(None).cast(pl.Int64).alias("draft_position"),
        ])
    
    logger.info(f"Fetched {len(draft_df)} draft records")
    
    # Join with players
    result_df = players_df.join(
        draft_df,
        on='player_id',
        how='left'
    )
    
    # Handle column name conflicts
    if 'draft_round_right' in result_df.columns:
        result_df = result_df.with_columns([
            pl.when(pl.col("draft_round_right").is_not_null())
            .then(pl.col("draft_round_right"))
            .otherwise(pl.col("draft_round"))
            .alias("draft_round"),
        ]).drop("draft_round_right")
    
    if 'draft_year_right' in result_df.columns:
        result_df = result_df.with_columns([
            pl.when(pl.col("draft_year_right").is_not_null())
            .then(pl.col("draft_year_right"))
            .otherwise(pl.col("draft_year"))
            .alias("draft_year"),
        ]).drop("draft_year_right")
    
    if 'draft_position_right' in result_df.columns:
        result_df = result_df.with_columns([
            pl.when(pl.col("draft_position_right").is_not_null())
            .then(pl.col("draft_position_right"))
            .otherwise(pl.col("draft_position"))
            .alias("draft_position"),
        ]).drop("draft_position_right")
    
    # Ensure columns exist
    if 'draft_round' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Int64).alias("draft_round")])
    if 'draft_year' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Int64).alias("draft_year")])
    if 'draft_position' not in result_df.columns:
        result_df = result_df.with_columns([pl.lit(None).cast(pl.Int64).alias("draft_position")])
    
    matched_count = result_df.filter(pl.col("draft_year").is_not_null()).height
    logger.info(f"Matched {matched_count} players with draft data")
    
    return result_df

