#!/usr/bin/env python3
"""
Test script for draft data enrichment.

Tests fetching draft data with a small sample before running full enrichment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrich_player_data import fetch_draft_data_from_pybaseball
from loguru import logger

logger.info("Testing draft data fetching with small sample (2010, round 1)")

# Test with just 2010
draft_df = fetch_draft_data_from_pybaseball(start_year=2010, end_year=2010)

print(f"\n=== RESULTS ===")
print(f"Shape: {draft_df.shape}")
print(f"Columns: {list(draft_df.columns)}")

if len(draft_df) > 0:
    print(f"\n✅ Success! Fetched {len(draft_df)} draft records")
    print(f"\nSample data:")
    print(draft_df.head(10))
    
    # Check player ID column
    if 'player_id' in draft_df.columns:
        print(f"\n✅ Player ID column found: {draft_df['player_id'].n_unique()} unique players")
    else:
        print(f"\n⚠️  Player ID column not found. Available columns: {draft_df.columns}")
else:
    print(f"\n⚠️  No draft data fetched. Check:")
    print(f"  1. Is html5lib installed? (pip install html5lib)")
    print(f"  2. Is pybaseball working? (from pybaseball import amateur_draft)")
    print(f"  3. Check error logs above")

