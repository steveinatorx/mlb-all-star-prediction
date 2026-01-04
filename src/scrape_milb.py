"""Baseball Reference scraping for minor league pitching statistics."""

import time
from typing import Optional

import polars as pl
from loguru import logger

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False


def scrape_player_minor_league_stats(
    player_bbref_id: str, start_year: Optional[int] = None, end_year: Optional[int] = None
) -> pl.DataFrame:
    """
    Scrape a single player's minor league pitching stats from Baseball Reference.

    Args:
        player_bbref_id: Baseball Reference player ID (e.g., "degroja01")
        start_year: Optional start year filter
        end_year: Optional end year filter

    Returns:
        DataFrame with minor league pitching stats for the player
    """
    if not SCRAPING_AVAILABLE:
        logger.error("Scraping dependencies not available")
        return pl.DataFrame()

    url = f"https://www.baseball-reference.com/register/player.fcgi?id={player_bbref_id}"

    try:
        # Add comprehensive headers to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }

        # Use a session to maintain cookies
        session = requests.Session()
        session.headers.update(headers)
        
        # First, visit the main page to get cookies
        session.get("https://www.baseball-reference.com/", timeout=10)
        
        # Then request the player page
        response = session.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")

        # Find minor league pitching stats table
        # Baseball Reference uses table id "pitching_minors"
        table = soup.find("table", id="pitching_minors")

        if table is None:
            logger.debug(f"No minor league pitching stats found for {player_bbref_id}")
            return pl.DataFrame()

        # Parse table rows
        rows = []
        tbody = table.find("tbody")
        if tbody:
            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) < 5:  # Skip header rows
                    continue

                row_data = {}
                for cell in cells:
                    # Get column name from header or data-stat attribute
                    col_name = cell.get("data-stat") or cell.get("class", [""])[0]
                    if col_name:
                        row_data[col_name] = cell.get_text(strip=True)

                if row_data:
                    rows.append(row_data)

        if not rows:
            return pl.DataFrame()

        # Convert to DataFrame
        df = pl.DataFrame(rows)

        # Filter by year if specified
        if start_year and "year_id" in df.columns:
            df = df.filter(pl.col("year_id").cast(pl.Int64) >= start_year)
        if end_year and "year_id" in df.columns:
            df = df.filter(pl.col("year_id").cast(pl.Int64) <= end_year)

        # Add player_id column
        df = df.with_columns([pl.lit(player_bbref_id).alias("player_id")])

        # Rate limiting - be respectful
        time.sleep(1)

        return df

    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return pl.DataFrame()
    except Exception as e:
        logger.error(f"Error parsing stats for {player_bbref_id}: {e}")
        return pl.DataFrame()


def scrape_minor_league_pitching_batch(
    player_bbref_ids: list[str],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    delay: float = 1.0,
) -> pl.DataFrame:
    """
    Scrape minor league stats for multiple players.

    Args:
        player_bbref_ids: List of Baseball Reference player IDs
        start_year: Optional start year filter
        end_year: Optional end year filter
        delay: Delay between requests (seconds)

    Returns:
        Combined DataFrame with stats for all players
    """
    if not SCRAPING_AVAILABLE:
        logger.error("Scraping dependencies not available")
        return pl.DataFrame()

    all_dfs = []

    logger.info(f"Scraping minor league stats for {len(player_bbref_ids)} players")

    for i, player_id in enumerate(player_bbref_ids):
        if i > 0 and i % 10 == 0:
            logger.info(f"Progress: {i}/{len(player_bbref_ids)} players")

        df = scrape_player_minor_league_stats(player_id, start_year, end_year)

        if len(df) > 0:
            all_dfs.append(df)

        # Rate limiting
        time.sleep(delay)

    if not all_dfs:
        return pl.DataFrame()

    # Combine all DataFrames
    result = pl.concat(all_dfs)

    logger.info(f"Successfully scraped stats for {len(result)} player-seasons")
    return result

