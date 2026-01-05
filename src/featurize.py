"""Feature engineering: create model-ready features from processed data."""

from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from src.config import config


def load_processed_data(data_dir: Optional[Path] = None) -> dict[str, pl.DataFrame]:
    """
    Load processed data files.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Dictionary of DataFrames
    """
    data_dir = data_dir or config.processed_data_dir

    logger.info("Loading processed data files")

    return {
        "minor_league_pitching": pl.read_parquet(
            data_dir / "minor_league_pitching_processed.parquet"
        ),
        "labels": pl.read_parquet(data_dir / "labels.parquet"),
        "players": pl.read_parquet(data_dir / "players_processed.parquet"),
    }


def create_career_aggregates(milb_df: pl.DataFrame) -> pl.DataFrame:
    """
    Create career aggregate statistics per player.

    Args:
        milb_df: Minor league pitching DataFrame

    Returns:
        DataFrame with career aggregates
    """
    logger.info("Creating career aggregate features")

    career_stats = (
        milb_df.group_by("player_id")
        .agg(
            [
                pl.sum("innings_pitched").alias("total_milb_ip"),
                pl.sum("games").alias("total_milb_games"),
                pl.sum("games_started").alias("total_milb_starts"),
                # Weighted averages for rate stats
                (
                    pl.sum("earned_runs") * 9.0 / pl.sum("innings_pitched")
                ).alias("career_era"),
                (
                    (pl.sum("hits") + pl.sum("walks")) / pl.sum("innings_pitched")
                ).alias("career_whip"),
                (
                    pl.sum("strikeouts") * 9.0 / pl.sum("innings_pitched")
                ).alias("career_k_per_9"),
                (
                    pl.sum("walks") * 9.0 / pl.sum("innings_pitched")
                ).alias("career_bb_per_9"),
            ]
        )
    )

    return career_stats


def create_best_season_features(milb_df: pl.DataFrame) -> pl.DataFrame:
    """
    Create best season statistics per player.

    Args:
        milb_df: Minor league pitching DataFrame

    Returns:
        DataFrame with best season stats
    """
    logger.info("Creating best season features")

    # For each player, find best season for each metric
    best_seasons = (
        milb_df.group_by("player_id")
        .agg(
            [
                pl.min("era").alias("best_era"),
                pl.min("whip").alias("best_whip"),
                pl.max("k_per_9").alias("best_k_per_9"),
            ]
        )
    )

    return best_seasons


def create_progression_features(
    milb_df: pl.DataFrame, players_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Create progression and level features.

    Args:
        milb_df: Minor league pitching DataFrame
        players_df: Players DataFrame

    Returns:
        DataFrame with progression features
    """
    logger.info("Creating progression features")

    # Level hierarchy
    level_order = {"R": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}

    # Highest level reached
    highest_level = (
        milb_df.with_columns(
            [
                pl.col("level")
                .map_elements(
                    lambda x: level_order.get(x, -1), return_dtype=pl.Int64
                )
                .alias("level_num")
            ]
        )
        .group_by("player_id")
        .agg([pl.max("level_num").alias("max_level_num")])
        .with_columns(
            [
                pl.col("max_level_num")
                .map_elements(
                    lambda x: [k for k, v in level_order.items() if v == x][0]
                    if x >= 0
                    else "Unknown",
                    return_dtype=pl.String,
                )
                .alias("highest_level_reached")
            ]
        )
        .drop("max_level_num")
    )

    # Seasons at each level
    level_counts = (
        milb_df.group_by(["player_id", "level"])
        .agg([pl.len().alias("seasons")])
        .pivot(
            index="player_id",
            on="level",
            values="seasons",
            aggregate_function="sum",
        )
        .fill_null(0)
    )

    # Ensure AAA and AA columns exist
    if "AAA" not in level_counts.columns:
        level_counts = level_counts.with_columns([pl.lit(0).alias("AAA")])
    if "AA" not in level_counts.columns:
        level_counts = level_counts.with_columns([pl.lit(0).alias("AA")])

    level_counts = level_counts.select(["player_id", "AAA", "AA"]).rename(
        {"AAA": "seasons_at_aaa", "AA": "seasons_at_aa"}
    )

    # Age at debut (if available)
    age_at_debut_df = players_df.select(["player_id", "birth_date", "mlb_debut"])
    
    if age_at_debut_df["birth_date"].is_not_null().any():
        # Calculate age at debut
        # Convert to date types if needed
        age_at_debut = age_at_debut_df.filter(
            pl.col("birth_date").is_not_null() & pl.col("mlb_debut").is_not_null()
        )
        
        # Convert string columns to date
        if age_at_debut["birth_date"].dtype == pl.Utf8:
            age_at_debut = age_at_debut.with_columns(
                pl.col("birth_date").str.to_date().alias("birth_date")
            )
        if age_at_debut["mlb_debut"].dtype == pl.Utf8:
            age_at_debut = age_at_debut.with_columns(
                pl.col("mlb_debut").str.to_date().alias("mlb_debut")
            )
        
        # Calculate age
        age_at_debut = age_at_debut.with_columns([
            ((pl.col("mlb_debut") - pl.col("birth_date")).dt.total_days() / 365.25)
            .alias("age_at_debut")
        ]).select(["player_id", "age_at_debut"])
        all_players = age_at_debut_df.select("player_id")
        age_at_debut = all_players.join(age_at_debut, on="player_id", how="left")
    else:
        # No birth dates available
        age_at_debut = players_df.select("player_id").with_columns([
            pl.lit(None).cast(pl.Float64).alias("age_at_debut")
        ])

    # Join all progression features
    progression = highest_level.join(level_counts, on="player_id", how="left").join(
        age_at_debut, on="player_id", how="left"
    )

    return progression


def create_time_splits(
    features_df: pl.DataFrame, players_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Create time-based train/val/test splits.

    Args:
        features_df: Features DataFrame
        players_df: Players DataFrame with mlb_debut

    Returns:
        Features DataFrame with split column
    """
    logger.info("Creating time-based splits")

    # Use MLB debut year as proxy for "prediction year"
    # Train: debut <= train_end_year
    # Val: train_end_year < debut <= val_end_year
    # Test: debut > val_end_year or no debut

    # Convert mlb_debut to date if it's a string
    splits_df = players_df.select(["player_id", "mlb_debut"])
    if splits_df["mlb_debut"].dtype == pl.String:
        splits_df = splits_df.with_columns(pl.col("mlb_debut").str.to_date().alias("mlb_debut"))

    splits = (
        splits_df.with_columns(
            [
                pl.when(pl.col("mlb_debut").is_null())
                .then(pl.lit("test"))
                .when(pl.col("mlb_debut").dt.year() <= config.train_end_year)
                .then(pl.lit("train"))
                .when(pl.col("mlb_debut").dt.year() <= config.val_end_year)
                .then(pl.lit("val"))
                .otherwise(pl.lit("test"))
                .alias("split")
            ]
        )
        .select(["player_id", "split"])
    )

    features_with_splits = features_df.join(splits, on="player_id", how="left")

    return features_with_splits


def engineer_features(output_dir: Optional[Path] = None) -> Path:
    """
    Engineer all features from processed data.

    Args:
        output_dir: Output directory (defaults to config.features_data_dir)

    Returns:
        Path to saved features file
    """
    output_dir = output_dir or config.features_data_dir

    logger.info("Engineering features")

    # Load processed data
    data = load_processed_data()

    # Map FanGraphs IDs to MLBAM IDs if needed
    # Minor league data may use FanGraphs IDs, but labels use MLBAM IDs
    milb_df = data["minor_league_pitching"]
    players_df = data["players"]
    
    # Check if we need to map IDs
    milb_ids = set(milb_df["player_id"].unique().to_list())
    label_ids = set(data["labels"]["player_id"].unique().to_list())
    
    if len(milb_ids & label_ids) == 0 and "fangraphs_id" in players_df.columns:
        logger.info("Mapping FanGraphs IDs to MLBAM IDs for feature engineering")
        # Create mapping: fangraphs_id -> player_id (MLBAM)
        id_mapping = players_df.select([
            pl.col("player_id").alias("mlbam_id"),
            pl.col("fangraphs_id")
        ]).filter(
            pl.col("fangraphs_id").is_not_null()
        )
        
        # Map minor league data to use MLBAM IDs
        milb_df = milb_df.join(
            id_mapping,
            left_on="player_id",
            right_on="fangraphs_id",
            how="left"
        )
        
        # Check which columns exist before dropping
        cols_to_drop = []
        if "mlbam_id" in milb_df.columns:
            cols_to_drop.append("mlbam_id")
        if "fangraphs_id_right" in milb_df.columns:
            cols_to_drop.append("fangraphs_id_right")
        elif "fangraphs_id" in milb_df.columns and milb_df.columns.count("fangraphs_id") > 1:
            # If there are duplicate fangraphs_id columns, drop the right one
            cols_to_drop.append("fangraphs_id")
        
        milb_df = milb_df.with_columns([
            # Use MLBAM ID if available, otherwise keep FanGraphs ID
            pl.when(pl.col("mlbam_id").is_not_null())
            .then(pl.col("mlbam_id"))
            .otherwise(pl.col("player_id"))
            .alias("player_id")
        ])
        
        if cols_to_drop:
            milb_df = milb_df.drop(cols_to_drop)
        
        logger.info(f"Mapped {milb_df['player_id'].n_unique()} players to MLBAM IDs")

    # Create feature groups
    career_features = create_career_aggregates(milb_df)
    best_season_features = create_best_season_features(milb_df)
    progression_features = create_progression_features(milb_df, players_df)

    # Add draft features from players table (if available)
    # Check which draft columns exist before selecting
    draft_cols = ["player_id"]
    if "draft_round" in players_df.columns:
        draft_cols.append("draft_round")
    if "draft_year" in players_df.columns:
        draft_cols.append("draft_year")
    if "draft_position" in players_df.columns:
        draft_cols.append("draft_position")
    
    draft_features = players_df.select(draft_cols)
    # Only filter if we have draft columns
    if "draft_year" in draft_features.columns or "draft_round" in draft_features.columns:
        draft_features = draft_features.filter(
            pl.col("draft_year").is_not_null() | pl.col("draft_round").is_not_null()
        ).filter(
        pl.col("draft_year").is_not_null() | pl.col("draft_round").is_not_null()
    )

    # Join all features
    features = (
        career_features.join(best_season_features, on="player_id", how="left")
        .join(progression_features, on="player_id", how="left")
        .join(draft_features, on="player_id", how="left")
        .join(data["labels"].select(["player_id", "is_all_star"]), on="player_id", how="left")
    )

    # Add time splits
    features = create_time_splits(features, data["players"])

    # Save features
    features_path = output_dir / "features.parquet"
    features.write_parquet(features_path)

    logger.info(f"Saved features to {features_path}")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Features: {features.columns}")

    return features_path

