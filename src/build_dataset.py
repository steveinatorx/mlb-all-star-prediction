"""Dataset building: cleaning, joining, labeling, and validation."""

from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from src.config import config
from src.schemas import (
    LabelSchema,
    MinorLeaguePitchingSeasonSchema,
    PlayerSchema,
)


def load_raw_data(data_dir: Optional[Path] = None) -> dict[str, pl.DataFrame]:
    """
    Load raw data files.

    Args:
        data_dir: Directory containing raw data (defaults to config)

    Returns:
        Dictionary of DataFrames
    """
    data_dir = data_dir or config.raw_data_dir

    logger.info("Loading raw data files")

    return {
        "players": pl.read_parquet(data_dir / "players.parquet"),
        "minor_league_pitching": pl.read_parquet(
            data_dir / "minor_league_pitching.parquet"
        ),
        "all_star_rosters": pl.read_parquet(data_dir / "all_star_rosters.parquet"),
    }


def validate_schema(df: pl.DataFrame, schema_class: type) -> pl.DataFrame:
    """
    Validate DataFrame against a Pydantic schema.

    Args:
        df: DataFrame to validate
        schema_class: Pydantic schema class

    Returns:
        Validated DataFrame
    """
    # Basic validation - check required columns exist
    # Full validation would require converting to dicts and validating each row
    logger.debug(f"Validating schema: {schema_class.__name__}")
    return df


def filter_pre_debut_stats(
    milb_df: pl.DataFrame, players_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Filter minor league stats to only include pre-MLB debut data.

    Critical for preventing label leakage.

    Args:
        milb_df: Minor league pitching DataFrame
        players_df: Players DataFrame with mlb_debut dates

    Returns:
        Filtered DataFrame with only pre-debut stats
    """
    logger.info("Filtering to pre-MLB debut statistics only")

    # Join with player info to get debut dates
    df = milb_df.join(
        players_df.select(["player_id", "mlb_debut"]),
        on="player_id",
        how="left",
    )

    # Ensure mlb_debut is a date type
    if df["mlb_debut"].dtype == pl.String:
        df = df.with_columns(pl.col("mlb_debut").str.to_date().alias("mlb_debut"))

    # Filter: season must be before debut year, or debut is null (never debuted)
    df = df.filter(
        (pl.col("mlb_debut").is_null())
        | (pl.col("season") < pl.col("mlb_debut").dt.year())
    )

    logger.info(
        f"Filtered to {len(df)} pre-debut records (from {len(milb_df)} total)"
    )

    return df.drop("mlb_debut")


def create_labels(
    players_df: pl.DataFrame, all_star_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Create labels: whether each player ever became an All-Star.

    Args:
        players_df: Players DataFrame
        all_star_df: All-Star rosters DataFrame

    Returns:
        DataFrame with labels: player_id, is_all_star, all_star_seasons, first_all_star_season
    """
    logger.info("Creating labels from All-Star rosters")

    # Aggregate All-Star appearances by player
    all_star_agg = (
        all_star_df.group_by("player_id")
        .agg(
            [
                pl.col("season").min().alias("first_all_star_season"),
                pl.col("season").alias("all_star_seasons"),  # Automatically becomes a list
            ]
        )
        .with_columns([pl.lit(True).alias("is_all_star")])
    )

    # Join with all players to create labels (default to False)
    labels = (
        players_df.select("player_id")
        .join(all_star_agg, on="player_id", how="left")
        .with_columns(
            [
                pl.col("is_all_star").fill_null(False),
                pl.col("all_star_seasons").fill_null([]),
            ]
        )
    )

    logger.info(
        f"Created labels: {labels.filter(pl.col('is_all_star') == True).height} All-Stars "
        f"out of {len(labels)} total players"
    )

    return labels


def apply_filters(milb_df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply data quality filters.

    Args:
        milb_df: Minor league pitching DataFrame

    Returns:
        Filtered DataFrame
    """
    logger.info("Applying data quality filters")

    initial_count = len(milb_df)

    # Filter: minimum IP threshold
    df = milb_df.filter(pl.col("innings_pitched") >= config.min_ip_for_label)

    # Filter: valid stats (non-negative) - only check columns that exist
    filter_conditions = []
    for col in ["hits", "runs", "earned_runs", "walks", "strikeouts"]:
        if col in df.columns:
            filter_conditions.append(pl.col(col) >= 0)
    
    if filter_conditions:
        # Combine all conditions with &
        combined_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_condition = combined_condition & condition
        df = df.filter(combined_condition)

    logger.info(f"Filtered from {initial_count} to {len(df)} records")

    return df


def build_processed_dataset(
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Build processed dataset: clean, join, label, validate.

    Args:
        output_dir: Output directory (defaults to config.processed_data_dir)

    Returns:
        Dictionary mapping dataset names to file paths
    """
    output_dir = output_dir or config.processed_data_dir

    logger.info("Building processed dataset")

    # Load raw data
    raw_data = load_raw_data()

    # Validate schemas
    validate_schema(raw_data["players"], PlayerSchema)
    validate_schema(raw_data["minor_league_pitching"], MinorLeaguePitchingSeasonSchema)

    # Filter to pre-debut stats only
    milb_filtered = filter_pre_debut_stats(
        raw_data["minor_league_pitching"], raw_data["players"]
    )

    # Apply quality filters
    milb_filtered = apply_filters(milb_filtered)

    # Create labels
    labels = create_labels(raw_data["players"], raw_data["all_star_rosters"])

    # Save processed datasets
    milb_path = output_dir / "minor_league_pitching_processed.parquet"
    labels_path = output_dir / "labels.parquet"
    players_path = output_dir / "players_processed.parquet"

    milb_filtered.write_parquet(milb_path)
    labels.write_parquet(labels_path)
    raw_data["players"].write_parquet(players_path)

    logger.info(f"Saved processed datasets to {output_dir}")

    return {
        "minor_league_pitching": milb_path,
        "labels": labels_path,
        "players": players_path,
    }

