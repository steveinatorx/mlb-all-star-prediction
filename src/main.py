"""CLI entrypoint using Typer."""

from pathlib import Path

import typer
from loguru import logger

from src import config
from src.build_dataset import build_processed_dataset
from src.evaluate import evaluate_all_models
from src.featurize import engineer_features
from src.ingest import run_ingestion
from src.report import generate_markdown_report
from src.train import train_all_models

# Configure logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=getattr(config, 'log_level', 'INFO'),
)

app = typer.Typer(help="MLB All-Star Prediction CLI")


@app.command()
def ingest(
    start_year: int = typer.Option(
        None, "--start-year", help="Start year for data collection"
    ),
    end_year: int = typer.Option(
        None, "--end-year", help="End year for data collection"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Output directory for raw data"
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-fetch data even if files already exist"
    ),
) -> None:
    """Download raw data from external sources."""
    logger.info("Starting data ingestion")
    run_ingestion(
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        force=force,
    )
    logger.info("Data ingestion complete")


@app.command("build-dataset")
def build_dataset(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Output directory for processed data"
    ),
) -> None:
    """Build processed dataset: clean, join, label, validate."""
    logger.info("Building processed dataset")
    build_processed_dataset(output_dir=output_dir)
    logger.info("Dataset building complete")


@app.command()
def featurize(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Output directory for features"
    ),
) -> None:
    """Engineer features from processed data."""
    logger.info("Engineering features")
    engineer_features(output_dir=output_dir)
    logger.info("Feature engineering complete")


@app.command()
def train(
    features_path: Path = typer.Option(
        None, "--features-path", help="Path to features file"
    ),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Output directory for models"
    ),
) -> None:
    """Train all models."""
    logger.info("Training models")
    train_all_models(features_path=features_path, output_dir=output_dir)
    logger.info("Model training complete")


@app.command()
def evaluate(
    experiments_dir: Path = typer.Option(
        None, "--experiments-dir", help="Directory containing trained models"
    ),
    features_path: Path = typer.Option(
        None, "--features-path", help="Path to features file"
    ),
    split: str = typer.Option(
        "test", "--split", help="Split to evaluate on (train/val/test)"
    ),
) -> None:
    """Evaluate models and generate metrics and plots."""
    logger.info("Evaluating models")
    evaluate_all_models(
        experiments_dir=experiments_dir,
        features_path=features_path,
        split=split,
    )
    logger.info("Evaluation complete")


@app.command()
def report(
    experiments_dir: Path = typer.Option(
        None, "--experiments-dir", help="Directory containing experiment results"
    ),
    output_path: Path = typer.Option(
        None, "--output-path", help="Output path for report"
    ),
) -> None:
    """Generate markdown report summarizing experiments."""
    logger.info("Generating report")
    generate_markdown_report(experiments_dir=experiments_dir, output_path=output_path)
    logger.info("Report generation complete")


if __name__ == "__main__":
    app()

