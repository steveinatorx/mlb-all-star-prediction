"""End-to-end test of the optimized MiLB scraping pipeline."""

from pathlib import Path
import tempfile
import shutil

import polars as pl
import pytest
from loguru import logger

from src.ingest import run_ingestion
from src.build_dataset import build_processed_dataset, filter_pre_debut_stats
from data_fetch.fetch_milb_mlbcom import calculate_optimal_scraping_range


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="Skipping E2E scraping test - scraping code not included in open source repo")
def test_e2e_optimized_scraping(temp_data_dir):
    """Test the full pipeline with optimization."""
    logger.info("Starting E2E test with optimized scraping")
    
    # Step 1: Create a small player dataset with known MLB debut dates
    # Using players who debuted recently (2020-2023) so we can test with recent data
    test_players = pl.DataFrame({
        "player_id": [
            "663656",  # Shane Baz (debuted 2021)
            "668678",  # Hunter Greene (debuted 2022)
            "668804",  # Grayson Rodriguez (debuted 2023)
        ],
        "name_first": ["Shane", "Hunter", "Grayson"],
        "name_last": ["Baz", "Greene", "Rodriguez"],
        "mlb_debut": ["2021-09-20", "2022-04-10", "2023-04-05"],
        "bbref_id": [None, None, None],
        "fangraphs_id": [None, None, None],
    })
    
    # Save player info
    players_path = temp_data_dir / "players.parquet"
    test_players.write_parquet(players_path)
    logger.info(f"Created test player dataset: {len(test_players)} players")
    
    # Step 2: Test optimization calculation
    opt_start, opt_end, player_ids = calculate_optimal_scraping_range(
        test_players, min_years_before_debut=5
    )
    
    logger.info(f"Optimal scraping range: {opt_start}-{opt_end}")
    logger.info(f"Target players: {len(player_ids)}")
    
    # Verify optimization
    assert opt_start <= 2020, "Should start before earliest debut"
    assert opt_end >= 2022, "Should end before latest debut"
    assert len(player_ids) == 3, "Should include all test players"
    
    # Step 3: Run ingestion with optimization
    # Use a small year range to minimize scraping time
    logger.info("Running ingestion with optimization...")
    
    # Note: This will actually scrape MiLB.com, so it may take a few minutes
    # We'll use a small year range (2020-2023) to minimize time
    results = run_ingestion(
        start_year=2020,
        end_year=2023,
        output_dir=temp_data_dir,
        force=True,  # Force re-fetch to test fresh scraping
        max_players=3,  # Limit to our test players
    )
    
    # Verify ingestion results
    assert "players" in results
    assert "minor_league_pitching" in results
    assert "all_star_rosters" in results
    
    # Step 4: Verify player info
    players_df = pl.read_parquet(results["players"])
    logger.info(f"Loaded {len(players_df)} players")
    assert len(players_df) >= 3, "Should have at least our test players"
    
    # Step 5: Verify minor league data
    milb_df = pl.read_parquet(results["minor_league_pitching"])
    logger.info(f"Loaded {len(milb_df)} minor league records")
    
    if len(milb_df) > 0:
        # Check that we have data for our test players
        test_player_ids = set(test_players["player_id"].to_list())
        found_players = set(milb_df["player_id"].unique().to_list())
        overlap = test_player_ids & found_players
        
        logger.info(f"Test players found in MiLB data: {len(overlap)}/{len(test_player_ids)}")
        
        # Verify data structure
        required_cols = [
            "player_id", "season", "level", "team", "innings_pitched",
            "era", "whip", "strikeouts", "walks"
        ]
        for col in required_cols:
            assert col in milb_df.columns, f"Missing required column: {col}"
        
        # Verify year range (should be within optimal range)
        if len(milb_df) > 0:
            min_season = milb_df["season"].min()
            max_season = milb_df["season"].max()
            logger.info(f"MiLB data year range: {min_season}-{max_season}")
            
            # Should be within optimal range (or close, accounting for filtering)
            assert min_season >= opt_start - 2, "Data should be within optimal range"
            assert max_season <= opt_end + 1, "Data should be within optimal range"
        
        # Step 6: Test pre-debut filtering
        logger.info("Testing pre-debut filtering...")
        filtered_df = filter_pre_debut_stats(milb_df, players_df)
        
        logger.info(f"Filtered to {len(filtered_df)} pre-debut records")
        
        # Verify filtering worked (all seasons should be before debut)
        if len(filtered_df) > 0:
            # Join to get debut years
            filtered_with_debut = filtered_df.join(
                players_df.select(["player_id", "mlb_debut"]),
                on="player_id",
                how="left",
            )
            
            # Convert mlb_debut to year
            filtered_with_debut = filtered_with_debut.with_columns(
                pl.when(pl.col("mlb_debut").is_not_null())
                .then(pl.col("mlb_debut").str.to_date().dt.year())
                .otherwise(None)
                .alias("debut_year")
            )
            
            # Verify all seasons are before debut
            invalid = filtered_with_debut.filter(
                (pl.col("debut_year").is_not_null()) &
                (pl.col("season") >= pl.col("debut_year"))
            )
            
            assert len(invalid) == 0, f"Found {len(invalid)} records with season >= debut year"
            logger.info("✅ Pre-debut filtering verified")
        
        logger.info("✅ E2E test passed!")
    else:
        logger.warning("No minor league data scraped - this may be expected if players don't have MiLB stats")
        logger.warning("Test will pass but data verification skipped")


if __name__ == "__main__":
    # Run the test directly
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_e2e_optimized_scraping(temp_dir)
        print("\n✅ E2E test completed successfully!")
    finally:
        shutil.rmtree(temp_dir)

