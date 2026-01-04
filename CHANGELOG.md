# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-04

### Added
- **FanGraphs API integration** for minor league pitching statistics
  - `src/fetch_milb_fangraphs.py`: Module for fetching MiLB stats from FanGraphs API
  - Supports minor league stats from ~2005+ (tested back to 2005)
  - Automatic integration into `fetch_minor_league_pitching()`
- **Chadwick Register integration** for comprehensive player ID mapping
  - Uses `chadwick_register()` from pybaseball for player info
  - Provides MLBAM, FanGraphs, and Baseball Reference IDs
  - Supports players from 1871+ (much better than MLB stats API's 2008+ limit)
- **Baseball Reference scraping module** (`src/scrape_milb.py`) - ready when unblocked
- **MLB Stats API module** (`src/fetch_milb_mlbapi.py`) - for future exploration
- Enhanced `fetch_player_info()` to include FanGraphs and Baseball Reference IDs
- Updated `run_ingestion()` to fetch player info first (needed for minor league fetching)
- Documentation updates in `docs/DATA_SOURCES.md` and `README.md`

### Fixed
- Fixed `get_player_ids_from_mlb_stats()` to respect 2008+ API limitation
- Fixed `test_fetch_player_info()` hanging by providing specific player ID
- Fixed `test_run_ingestion()` hanging by using Chadwick Register directly
- Improved error handling and fallback logic

### Changed
- `fetch_player_info()` now uses Chadwick Register by default (faster, more reliable)
- `run_ingestion()` reordered to fetch player info before minor league stats
- Updated config documentation to clarify year limitations

### Added
- Initial project scaffold with complete data pipeline structure
- Data ingestion module (`src/ingest.py`) with mock data support
- Dataset building module (`src/build_dataset.py`) with leakage prevention
- Feature engineering module (`src/featurize.py`) with career aggregates and progression features
- Model training module (`src/train.py`) supporting Logistic Regression, Random Forest, XGBoost, LightGBM, and GAM
- Model evaluation module (`src/evaluate.py`) with PR-AUC, ROC-AUC, and Recall@TopK metrics
- Report generation module (`src/report.py`) for markdown reports
- CLI entrypoint (`src/main.py`) using Typer with commands: ingest, build-dataset, featurize, train, evaluate, report
- Configuration management using Pydantic (`src/config.py`)
- Data schemas with validation (`src/schemas.py`)
- Comprehensive test suite with smoke tests
- Blog post outlines (5 posts) in `docs/blog_series/`
- MIT License
- README with full project documentation
- Makefile with common tasks
- Pipfile for dependency management with pipenv
- asdf configuration (`.tool-versions`)
- Ruff, mypy, pytest, and black configuration files

### Changed
- N/A

### Fixed
- Polars API compatibility issues (pivot function, list aggregation)
- Date type handling in filters and feature engineering
- Missing column handling in data quality filters

## [0.1.0] - 2024-01-03

### Added
- Real player info fetching using pybaseball's `playerid_reverse_lookup()`
- `get_player_ids_from_mlb_stats()` helper function to bootstrap player ID collection
- Enhanced tests for player info fetching with real player ID lookup
- Support for fetching MLB debut dates (approximate from year)

### Changed
- `fetch_player_info()` now uses real pybaseball API instead of mock data
- Improved error handling and fallback to mock data when pybaseball unavailable

### Fixed
- Player ID type conversion (handles both integer and string IDs)
- Date string formatting for MLB debut dates

## [0.0.1] - 2024-01-03

### Added
- Initial project setup
- MIT License

