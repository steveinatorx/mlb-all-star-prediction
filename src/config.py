"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseSettings):
    """Project-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path("data"))
    reports_dir: Path = Field(default=Path("reports"))
    experiments_dir: Path = Field(default=Path("experiments"))

    # Data paths
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    features_data_dir: Path = Field(default=Path("data/features"))

    # Report paths
    figures_dir: Path = Field(default=Path("reports/figures"))
    tables_dir: Path = Field(default=Path("reports/tables"))

    # Data collection
    # Note: FanGraphs API supports minor league stats going back to at least 2005
    # Chadwick Register provides player IDs back to 1871
    # MLB stats API (for player ID discovery) only supports 2008+
    # Project focuses on 2005+ debuts for full data coverage
    start_year: int = Field(
        default=2005,
        description="Start year for data collection. Set to 2005+ for full FanGraphs API coverage. FanGraphs API supports minor league stats from ~2005+, Chadwick Register provides player IDs from 1871+"
    )
    end_year: int = Field(default=2023, description="End year for data collection")
    min_ip_for_label: float = Field(
        default=50.0, description="Minimum IP in MiLB to be included"
    )

    # Imbalanced data techniques
    use_class_weights: bool = Field(
        default=False,
        description="Use class weights to penalize misclassifying All-Stars",
    )
    use_smote: bool = Field(
        default=False,
        description="Use SMOTE for synthetic oversampling of minority class",
    )
    smote_k_neighbors: int = Field(
        default=5,
        description="Number of nearest neighbors for SMOTE",
    )

    # Train/test split
    train_end_year: int = Field(
        default=2018, description="Last year for training set"
    )
    val_end_year: int = Field(
        default=2020, description="Last year for validation set"
    )

    # Model training
    random_seed: int = Field(default=42)
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all)")

    # Evaluation metrics
    top_k_values: list[int] = Field(
        default=[10, 25, 50, 100], description="Top-K values for Recall@TopK"
    )

    # Caching
    cache_dir: Path = Field(default=Path("diskcache"))
    cache_enabled: bool = Field(default=True)

    # Logging
    log_level: str = Field(default="INFO")

    def model_post_init(self, __context) -> None:
        """Resolve paths after initialization."""
        # Resolve all paths relative to project root
        if isinstance(self.raw_data_dir, Path) and not self.raw_data_dir.is_absolute():
            self.raw_data_dir = self.project_root / self.raw_data_dir
        if isinstance(self.processed_data_dir, Path) and not self.processed_data_dir.is_absolute():
            self.processed_data_dir = self.project_root / self.processed_data_dir
        if isinstance(self.features_data_dir, Path) and not self.features_data_dir.is_absolute():
            self.features_data_dir = self.project_root / self.features_data_dir
        if isinstance(self.figures_dir, Path) and not self.figures_dir.is_absolute():
            self.figures_dir = self.project_root / self.figures_dir
        if isinstance(self.tables_dir, Path) and not self.tables_dir.is_absolute():
            self.tables_dir = self.project_root / self.tables_dir
        if isinstance(self.experiments_dir, Path) and not self.experiments_dir.is_absolute():
            self.experiments_dir = self.project_root / self.experiments_dir
        if isinstance(self.cache_dir, Path) and not self.cache_dir.is_absolute():
            self.cache_dir = self.project_root / self.cache_dir

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.features_data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = ProjectConfig()

