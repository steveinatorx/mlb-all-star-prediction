# From MiLB to MLB All-Star: Predicting Pitcher Stardom from Minor League Data

A comprehensive machine learning project that predicts whether minor league pitchers will become MLB All-Stars using historical data, advanced feature engineering, and model interpretability techniques.

## Project Overview

This project builds predictive models to identify future MLB All-Star pitchers based solely on their pre-debut minor league performance data. The project demonstrates expertise in handling imbalanced data, feature engineering, model interpretability, and evaluation of both ranking and binary classification approaches.

**Key Challenge**: Predicting rare events (All-Stars represent ~2% of players) while avoiding label leakage by only using pre-MLB debut statistics.

### Key Features

- **Robust Data Pipeline**: Reproducible ingestion, cleaning, and feature engineering with real baseball data
- **Leakage Prevention**: Strict rules to only use pre-debut minor league stats (enforced via schema validation)
- **Time-Aware Splits**: Train on earlier years (2005-2018), validate (2019-2020), test (2021-2023)
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, GAM
- **Advanced Techniques**: SMOTE oversampling, class weights for imbalanced data
- **Feature Engineering**: Career aggregates, best-season stats, progression features, interaction features (30 total)
- **Model Interpretation**: SHAP values (summary, waterfall, dependence plots), coefficient analysis, feature importance
- **Comprehensive Evaluation**: Ranking metrics (PR-AUC, ROC-AUC, Recall@TopK) and binary classification comparison
- **Feature Interactions**: SHAP-based interaction analysis and engineered interaction features
- **Production-Ready**: CLI interface, configuration management, comprehensive documentation

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
make ingest      # Download raw data (skips if files exist, use --force to re-fetch)
make build       # Build processed dataset
make featurize   # Engineer features
make train       # Train models
make eval        # Evaluate models
make report      # Generate markdown report
```

**Note**: The `ingest` command checks for existing data files and skips re-fetching by default. Use `--force` to re-fetch:
```bash
pipenv run python -m src.main ingest --force
```

### CLI Commands

All commands can also be run directly:

```bash
pipenv run python -m src.main ingest
pipenv run python -m src.main build-dataset
pipenv run python -m src.main featurize
pipenv run python -m src.main train              # Baseline models
pipenv run python -m src.main train-advanced     # Advanced (SMOTE + class weights)
pipenv run python -m src.main evaluate            # Evaluate models
pipenv run python -m src.main analyze-interactions --model-path experiments/advanced/random_forest_advanced.joblib
pipenv run python -m src.main add-interactions    # Add interaction features
pipenv run python -m src.main report              # Generate report
```

### Configuration

Configuration is managed in `src/config.py` using Pydantic. Key settings:

- **Data Collection**: `start_year`, `end_year` (default: 2005-2023)
  - FanGraphs API supports minor league stats from ~2005+
  - Chadwick Register provides player IDs from 1871+
- **Train/Test Splits**: `train_end_year`, `val_end_year` (default: 2018, 2020)
- **Minimum IP**: `min_ip_for_label` (default: 50.0)
- **Top-K Values**: `top_k_values` (default: [10, 25, 50, 100])

Override via environment variables or edit `src/config.py`.

## Data Pipeline

### 1. Ingestion (`ingest`)

Downloads raw data from external sources:
- **Player Info**: Chadwick Register via pybaseball (player IDs, MLB debut dates)
- **Minor League Pitching**: FanGraphs API (stats from ~2005+)
- **All-Star Rosters**: Lahman database via pybaseball (falls back to mock if unavailable)

**Output**: `data/raw/*.parquet`

**Data Sources**:
- **Chadwick Register**: Comprehensive player ID mapping (1871+)
- **FanGraphs API**: Minor league pitching statistics (~2005+)
- **MLB Stats API**: Player ID discovery (2008+, optional if using Chadwick Register)

**Note**: All-Star roster fetching may fall back to mock data if Lahman database is unavailable. See `docs/DATA_SOURCES.md` for details.
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
- **Career aggregates**: Total IP, career ERA, WHIP, K/9, BB/9
- **Best season stats**: Best ERA, best WHIP, best K/9
- **Progression features**: Highest level reached, seasons at AAA/AA, age at debut
- **Draft information**: Draft year, round, position (when available)
- **Time-based splits**: Train/val/test splits based on MLB debut year

**Output**: `data/features/features.parquet`

**Additional**: Interaction features can be added via `add-interactions` command:
- K/BB ratios, ERA×WHIP products, consistency metrics
- See `src/create_interaction_features.py` for details

### 4. Model Training (`train`)

Trains multiple baseline models:
- Logistic Regression (with L2 regularization, median imputation)
- Random Forest
- XGBoost
- LightGBM
- GAM (Generalized Additive Model)

**Advanced Training** (`train-advanced`):
- SMOTE oversampling for imbalanced data
- Class weights (inverse frequency)
- Proper preprocessing order: Impute → SMOTE → Scale → Train

**Output**: `experiments/*.joblib` (models), `experiments/training_results.json`

### 5. Evaluation (`evaluate`)

Evaluates models and generates comprehensive metrics and visualizations:

**Metrics**:
- Ranking: PR-AUC, ROC-AUC, Recall@TopK (Top 10, 25, 50, 100)
- Binary Classification: Precision, Recall, F1 at optimal threshold (for comparison)

**Plots**:
- Precision-Recall curves, ROC curves
- SHAP summary plots (feature importance)
- SHAP waterfall plots (individual prediction explanations)
- SHAP dependence plots (feature interactions)
- Coefficient plots (logistic regression)

**Output**: `reports/figures/*.png`, `reports/tables/*.csv`

**Additional Analysis** (`analyze-interactions`):
- SHAP-based feature interaction analysis
- Generates markdown reports with interaction suggestions

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

### Ranking Metrics (Primary Approach)
- **PR-AUC**: Precision-Recall Area Under Curve (better for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Recall@TopK**: Recall among top K predictions (scouting perspective)
  - How many All-Stars found in top 10/25/50/100 predictions?

### Binary Classification Metrics (Comparison)
- **Precision/Recall/F1**: Binary classification metrics at optimal threshold
- Included for comparison to demonstrate evaluation of both approaches
- See [Ranking vs Binary Classification](docs/RANKING_VS_BINARY_CLASSIFICATION.md) for detailed comparison

**Why Ranking?** Ranking avoids false positives and threshold selection complexity, making it better suited for this imbalanced data problem. See the comparison document for details.

## Model Interpretation

Comprehensive interpretability tools:

- **SHAP Values**: 
  - Summary plots (feature importance ranking)
  - Waterfall plots (individual prediction breakdowns)
  - Dependence plots (feature interactions)
  - See [SHAP Analysis Guide](docs/SHAP_ANALYSIS.md) for details

- **Logistic Regression**: Coefficients, coefficient plots

- **Tree Models**: SHAP values, tree-based feature importance

- **Feature Interactions**: SHAP-based interaction analysis to identify and create interaction features

- **Ranking vs Binary Classification**: Comprehensive comparison of evaluation approaches (see [docs/RANKING_VS_BINARY_CLASSIFICATION.md](docs/RANKING_VS_BINARY_CLASSIFICATION.md))

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

## Documentation

Comprehensive documentation available in `docs/`:

- **[SHAP Analysis Guide](docs/SHAP_ANALYSIS.md)**: Understanding SHAP values, interpretation, and implementation
- **[Ranking vs Binary Classification](docs/RANKING_VS_BINARY_CLASSIFICATION.md)**: Comparison of evaluation approaches
- **[Evaluation Metrics Explanation](docs/EVALUATION_METRICS_EXPLANATION.md)**: Why PR-AUC/ROC-AUC over F1 score
- **[Model Training Plan](docs/MODEL_TRAINING_PLAN.md)**: Training strategy and roadmap
- **[Roadmap](docs/ROADMAP.md)**: Project roadmap and next steps

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

## Current Dataset

- **2,471 players** with minor league pitching data (2005+ debuts)
- **50 All-Stars** (2.02% positive rate)
- **30 features** (20 base + 10 interaction features)
- **Train/Val/Test splits**: 2005-2018 / 2019-2020 / 2021-2023

## Model Performance

**Baseline Models**:
- Random Forest: PR-AUC 0.0366, ROC-AUC 0.6996
- XGBoost: PR-AUC 0.0453, ROC-AUC 0.7193
- LightGBM: PR-AUC 0.0549, ROC-AUC 0.6855

**Advanced Models** (with SMOTE + class weights):
- Random Forest: PR-AUC 0.0929 (+154% improvement)
- XGBoost: PR-AUC 0.0946 (+109% improvement)
- LightGBM: PR-AUC 0.0629 (+14.5% improvement)

## Limitations

- **Class imbalance**: All-Stars are rare (~2% of players)
- **Limited scope**: Only US minor league pitchers (2005+ debuts)
- **Missing data**: Some All-Stars don't have minor league data (foreign leagues, international signings)
- **Feature limitations**: No velocity, spin rate, or pitch mix data (would require Statcast)

## Future Work

- **Hyperparameter tuning**: Bayesian optimization with Optuna
- **Ensemble methods**: Voting, stacking, blending
- **Additional features**: Velocity, spin rate, pitch mix (if Statcast data available)
- **Data expansion**: Foreign leagues (NPB, KBO), college baseball
- **Time-to-All-Star**: Survival analysis for time-to-event prediction
- **Organizational factors**: Team, coaching, development system features

## License

MIT

## Author

Steven Brezina

## Acknowledgments

- pybaseball library
- Baseball Reference
- MLB Stats API

