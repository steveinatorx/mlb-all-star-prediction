"""Data schemas and validation models."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PlayerSchema(BaseModel):
    """Schema for player information."""

    player_id: str = Field(description="Unique player identifier")
    name_first: Optional[str] = None
    name_last: Optional[str] = None
    birth_date: Optional[date] = None
    mlb_debut: Optional[date] = Field(
        default=None, description="Date of MLB debut (critical for leakage prevention)"
    )
    draft_round: Optional[int] = Field(default=None, description="MLB Draft round")
    draft_year: Optional[int] = Field(default=None, description="MLB Draft year")
    draft_position: Optional[int] = Field(default=None, description="Overall draft position")


class MinorLeaguePitchingSeasonSchema(BaseModel):
    """Schema for minor league pitching season statistics."""

    player_id: str
    season: int = Field(ge=1900, le=2100)
    level: str = Field(description="MiLB level (A, A+, AA, AAA, etc.)")
    team: Optional[str] = None
    league: Optional[str] = None

    # Pitching stats
    games: int = Field(ge=0)
    games_started: int = Field(ge=0)
    complete_games: int = Field(ge=0)
    innings_pitched: float = Field(ge=0.0)
    hits: int = Field(ge=0)
    runs: int = Field(ge=0)
    earned_runs: int = Field(ge=0)
    walks: int = Field(ge=0)
    strikeouts: int = Field(ge=0)
    home_runs: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    saves: int = Field(ge=0)

    # Expanded stats (from MiLB.com expanded view or FanGraphs)
    total_batters_faced: Optional[int] = Field(default=None, ge=0, description="TBF - Total Batters Faced")
    pitches: Optional[int] = Field(default=None, ge=0, description="NP - Number of Pitches")
    pitches_per_ip: Optional[float] = Field(default=None, ge=0.0, description="P/IP - Pitches per Inning Pitched")
    quality_starts: Optional[int] = Field(default=None, ge=0, description="QS - Quality Starts")
    quality_finish: Optional[int] = Field(default=None, ge=0, description="QF - Quality Finish")
    holds: Optional[int] = Field(default=None, ge=0, description="HLD - Holds")
    intentional_walks: Optional[int] = Field(default=None, ge=0, description="IBB - Intentional Walks")
    wild_pitches: Optional[int] = Field(default=None, ge=0, description="WP - Wild Pitches")
    balks: Optional[int] = Field(default=None, ge=0, description="BK - Balks")

    # Derived fields (computed in data processing, not validated here)
    era: Optional[float] = Field(default=None, ge=0.0)
    whip: Optional[float] = Field(default=None, ge=0.0)
    k_per_9: Optional[float] = Field(default=None, ge=0.0)
    bb_per_9: Optional[float] = Field(default=None, ge=0.0)


class LabelSchema(BaseModel):
    """Schema for labels (All-Star status)."""

    player_id: str
    is_all_star: bool = Field(description="Whether player ever became an MLB All-Star")
    all_star_seasons: list[int] = Field(
        default_factory=list, description="List of seasons player was an All-Star"
    )
    first_all_star_season: Optional[int] = Field(
        default=None, description="First season as All-Star"
    )


class FeatureSchema(BaseModel):
    """Schema for engineered features (summary stats)."""

    player_id: str
    # Career aggregates (pre-MLB debut only)
    total_milb_ip: float = Field(ge=0.0)
    total_milb_games: int = Field(ge=0)
    total_milb_starts: int = Field(ge=0)
    career_era: Optional[float] = None
    career_whip: Optional[float] = None
    career_k_per_9: Optional[float] = None
    career_bb_per_9: Optional[float] = None

    # Best season stats
    best_era: Optional[float] = None
    best_whip: Optional[float] = None
    best_k_per_9: Optional[float] = None

    # Progression features
    highest_level_reached: str = Field(description="Highest MiLB level reached")
    seasons_at_aaa: int = Field(ge=0)
    seasons_at_aa: int = Field(ge=0)
    age_at_debut: Optional[float] = None

    # Additional features can be added here
    draft_round: Optional[int] = None
    draft_year: Optional[int] = None

