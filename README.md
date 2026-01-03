# From MiLB to MLB All-Star: Predicting Pitcher Stardom from Minor League Data

A machine learning project to predict whether minor league pitchers will become MLB All-Stars using historical data and careful feature engineering.

## Project Overview

This project builds a predictive model to identify future MLB All-Star pitchers based solely on their minor league performance data. The key challenge is predicting rare events (All-Stars) while avoiding label leakage by only using pre-MLB debut statistics.

### Key Features

- **Robust Data Pipeline**: Reproducible ingestion, cleaning, and feature engineering
- **Leakage Prevention**: Strict rules to only use pre-debut minor league stats
- **Time-Aware Splits**: Train on earlier years, validate/test on later years
- **Multiple Models**: Logistic Regression (interpretable), Random Forest, XGBoost, LightGBM
- **Model Interpretation**: SHAP values, coefficient analysis, feature importance
- **Evaluation Metrics**: PR-AUC, ROC-AUC, Recall@TopK (scouting-focused)
- **Blog-Ready Outputs**: Charts, tables, and markdown reports

## Technology Stack

- **Python 3.11+**: Modern Python with type hints
- **Polars**: Fast DataFrame library (chosen over pandas for performance with large datasets)
- **pybaseball**: Baseball data retrieval
- **Typer**: CLI framework
- **Pydantic**: Configuration and schema validation
- **Loguru**: Structured logging
- **diskcache**: Caching for expensive operations
- **scikit-learn, XGBoost, LightGBM**: Machine learning models
- **SHAP**: Model interpretation
- **pytest**: Testing framework
- **pipenv**: Dependency management
- **asdf**: Python version management

## Project Structure

```
mlb-all-star-prediction/
├── data/
│   ├── raw/              # Raw data from external sources
│   ├── processed/        # Cleaned and labeled datasets
│   └── features/         # Engineered features ready for modeling
├── src/
│   ├── config.py         # Configuration management
│   ├── schemas.py        # Data schemas and validation
│   ├── ingest.py         # Data ingestion from external sources
│   ├── build_dataset.py  # Dataset building and labeling
│   ├── featurize.py      # Feature engineering
│   ├── train.py          # Model training
│   ├── evaluate.py       # Model evaluation and metrics
│   ├── report.py         # Report generation
│   └── main.py           # CLI entrypoint
├── experiments/          # Trained models and experiment results
├── reports/
│   ├── figures/          # Generated charts and plots
│   └── tables/           # CSV tables with metrics
├── docs/
│   └── blog_series/      # Blog post outlines (5 posts)
├── tests/                # Pytest test suite
├── config/               # Configuration files
├── Pipfile               # pipenv dependencies
├── .tool-versions        # asdf Python version
└── Makefile              # Common tasks
```

## Setup

### Prerequisites

- **asdf**: Version manager for Python
- **pipenv**: Python dependency management

### Installation

1. **Install Python version** (using asdf):
   ```bash
   asdf install
   ```

2. **Install dependencies** (using pipenv):
   ```bash
   make setup
   # or manually:
   pipenv install --dev
   ```

3. **Activate the environment**:
   ```bash
   pipenv shell
   ```

4. **Copy environment file** (optional):
   ```bash
   cp .env.example .env
   # Edit .env if you have API keys
   ```

## Usage

### Quick Start

Run the full pipeline:

```bash
make ingest      # Download raw data
make build       # Build processed dataset
make featurize   # Engineer features
make train       # Train models
make eval        # Evaluate models
make report      # Generate markdown report
```

### CLI Commands

All commands can also be run directly:

```bash
pipenv run python -m src.main ingest
pipenv run python -m src.main build-dataset
pipenv run python -m src.main featurize
pipenv run python -m src.main train
pipenv run python -m src.main evaluate
pipenv run python -m src.main report
```

### Configuration

Configuration is managed in `src/config.py` using Pydantic. Key settings:

- **Data Collection**: `start_year`, `end_year` (default: 2000-2023)
- **Train/Test Splits**: `train_end_year`, `val_end_year` (default: 2018, 2020)
- **Minimum IP**: `min_ip_for_label` (default: 50.0)
- **Top-K Values**: `top_k_values` (default: [10, 25, 50, 100])

Override via environment variables or edit `src/config.py`.

## Data Pipeline

### 1. Ingestion (`ingest`)

Downloads raw data from external sources:
- MLB All-Star rosters
- Minor league pitching statistics
- Player biographical data (including MLB debut dates)

**Output**: `data/raw/*.parquet`

**Note**: Currently uses mock data. Implement actual data fetching in `src/ingest.py`:
- `fetch_all_star_rosters()`: TODO - implement All-Star roster fetching
- `fetch_minor_league_pitching()`: TODO - implement MiLB stats fetching
- `fetch_player_info()`: TODO - implement player info fetching

### 2. Dataset Building (`build-dataset`)

Cleans, validates, and labels the data:
- Validates schemas
- Filters to pre-MLB debut stats only (leakage prevention)
- Applies quality filters (minimum IP, valid stats)
- Creates labels (All-Star status)

**Output**: `data/processed/*.parquet`

### 3. Feature Engineering (`featurize`)

Creates model-ready features:
- Career aggregates (total IP, career ERA, WHIP, K/9, BB/9)
- Best season stats (best ERA, best WHIP, best K/9)
- Progression features (highest level, seasons at AAA/AA, age at debut)
- Time-based train/val/test splits

**Output**: `data/features/features.parquet`

### 4. Model Training (`train`)

Trains multiple models:
- Logistic Regression (with L2 regularization)
- Random Forest
- XGBoost
- LightGBM (if available)
- GAM (optional, if available)

**Output**: `experiments/*.joblib` (models), `experiments/training_results.json`

### 5. Evaluation (`evaluate`)

Evaluates models and generates:
- Metrics: PR-AUC, ROC-AUC, Recall@TopK
- Plots: Precision-Recall curves, ROC curves
- SHAP plots (for tree-based models)
- Coefficient plots (for logistic regression)

**Output**: `reports/figures/*.png`, `reports/tables/*.csv`

### 6. Report Generation (`report`)

Compiles a markdown report with:
- Model performance summary
- Links to figures and tables
- Key findings

**Output**: `reports/experiment_report.md`

## Leakage Prevention Rules

Critical rules to prevent label leakage:

1. **Pre-Debut Only**: Only use minor league stats from BEFORE MLB debut
2. **No Rehab Stats**: Exclude post-debut minor league appearances (rehab assignments)
3. **Debut Date Cutoff**: Use MLB debut date as strict cutoff for all features
4. **No Post-Debut Info**: No information from after debut in any feature

These rules are enforced in `src/build_dataset.py` via `filter_pre_debut_stats()`.

## Evaluation Metrics

- **PR-AUC**: Precision-Recall Area Under Curve (better for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Recall@TopK**: Recall among top K predictions (scouting perspective)
  - How many All-Stars found in top 10/25/50/100 predictions?

## Model Interpretation

- **Logistic Regression**: Coefficients, p-values, confidence intervals
- **Tree Models**: SHAP values for feature importance and individual predictions
- **All Models**: Feature importance plots, permutation importance

## Testing

Run tests:

```bash
make test
# or
pipenv run pytest tests/ -v
```

Current tests are smoke tests to verify pipeline steps work end-to-end.

## Development

### Code Quality

```bash
make lint      # Run linters (ruff, mypy)
make format    # Format code (black, ruff --fix)
```

### Adding New Features

1. Add feature engineering in `src/featurize.py`
2. Update `src/schemas.py` if needed
3. Add tests in `tests/`
4. Update documentation

## Blog Series

Outlines for 5 Medium posts are in `docs/blog_series/`:

1. **Framing + Data Sources + Leakage Rules**
2. **Data Engineering Pipeline + Schema + Validation**
3. **Feature Engineering + EDA Findings**
4. **Statistical Significance + Interpretable Modeling**
5. **XGBoost + Ranking + SHAP + Takeaways**

## Data Sources

- **pybaseball**: Primary library for baseball data
- **Baseball Reference**: Alternative source (may require scraping)
- **MLB Stats API**: Official source (requires API key)
- **Retrosheet**: Historical data

See `src/ingest.py` for implementation details and TODOs.

## Why Polars?

Polars is chosen over pandas for:
- **Performance**: Faster on large datasets (lazy evaluation, parallel processing)
- **Memory Efficiency**: Better memory usage
- **API**: Modern, expressive API
- **Type Safety**: Better integration with type hints

## Output Locations

- **Raw Data**: `data/raw/*.parquet`
- **Processed Data**: `data/processed/*.parquet`
- **Features**: `data/features/*.parquet`
- **Models**: `experiments/*.joblib`
- **Figures**: `reports/figures/*.png`
- **Tables**: `reports/tables/*.csv`
- **Reports**: `reports/experiment_report.md`

## Limitations

- Currently uses mock data (implement actual data fetching)
- Limited feature set (can add velocity, spin rate, pitch mix if available)
- Class imbalance (All-Stars are rare)
- Incomplete historical data (especially older years)

## Future Work

- Implement actual data fetching from pybaseball/other sources
- Add velocity and spin rate features
- Include pitch mix information
- Model time-to-All-Star (survival analysis)
- Include organizational factors
- Add more sophisticated feature engineering

## License

MIT

## Author

Steven Brezina

## Acknowledgments

- pybaseball library
- Baseball Reference
- MLB Stats API

