#!/usr/bin/env python3
"""
Script to enrich player data with birth dates and draft information.

This can be run separately after initial ingestion to add:
- Birth dates (from Baseball Reference - currently blocked)
- Draft information (from pybaseball amateur_draft)

Usage:
    python scripts/enrich_player_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrich_player_data import enrich_players_with_mlb_api_info
from src.config import config
import polars as pl
from loguru import logger

logger.info("Starting player data enrichment")

# Load existing player data
players_path = config.raw_data_dir / "players.parquet"
if not players_path.exists():
    logger.error(f"Player data not found at {players_path}")
    logger.error("Please run 'make ingest' first")
    sys.exit(1)

players_df = pl.read_parquet(players_path)
logger.info(f"Loaded {len(players_df)} players")

# Enrich with birth dates and draft information from MLB Stats API
logger.info("Enriching with birth dates and draft info from MLB Stats API...")
logger.info("This will fetch data for all players (may take a few minutes)")
players_df = enrich_players_with_mlb_api_info(
    players_df,
    delay=0.2,  # MLB API is fast, can use shorter delay
    # max_players=100,  # Uncomment to test with small sample first
)

# Save enriched data
players_df.write_parquet(players_path)
logger.info(f"Saved enriched player data to {players_path}")

# Show summary
draft_count = players_df.filter(pl.col("draft_year").is_not_null()).height
logger.info(f"Summary: {draft_count}/{len(players_df)} players have draft information")

