# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] - 2026-01-05

### Changed
- **All-Star filtering**: Filtered All-Star rosters to pitchers only
  - Updated MLB API extraction to check position (Pitcher) before including
  - Updated game logs extraction to filter to pitchers only
  - Result: 50 pitcher All-Stars in final dataset (all have minor league data) ✅
- **Feature engineering**: Updated to handle pitcher-only All-Star labels
  - ID mapping from FanGraphs IDs (minor league data) to MLBAM IDs (labels) working correctly

### Added
- **Research documentation**: Comprehensive research on pitcher All-Stars without minor league data
  - `docs/PITCHER_RESEARCH_FINDINGS.md`: Analysis of 3 missing pitchers (1.5% missing rate)
  - `docs/PITCHER_CAREER_PATH_RESEARCH.md`: General research on pitcher career paths
  - `docs/PITCHER_ALL_STAR_RESEARCH.md`: Research methodology and findings
- **Blog notes**: Added section on future data pools (college, international leagues, etc.)

### Fixed
- All-Star extraction now correctly filters to pitchers only (was including position players)

## [0.3.2] - 2026-01-05

### Changed
- **Pipeline verification**: Successfully ran `build-dataset` with real All-Star data
  - Generated processed dataset with 132 All-Stars out of 4,514 players (2.9% All-Star rate)
  - Pre-debut filtering working correctly (892,687 → 318,027 records after quality filters)
  - Labels created successfully from real All-Star rosters

### Fixed
- Re-fetched All-Star rosters to replace mock data with real data (389 records, 239 unique players)
- Regenerated processed labels to match updated All-Star rosters

## [0.3.1] - 2026-01-05

### Fixed
- **All-Star roster data extraction**: Fixed `fetch_all_star_rosters()` to work reliably
  - Primary method: Extract from `all_star_game_logs()` (works when `all_star_full()` fails)
  - Maps retro IDs to MLBAM IDs using Chadwick Register
  - Successfully extracts 389 All-Star appearances (2005-2023), 239 unique players
  - Falls back gracefully: game logs → all_star_full → Baseball Reference → manual CSV → mock
  - **Impact**: Pipeline can now run with real All-Star data instead of mock data

### Changed
- Updated `fetch_all_star_rosters()` to try game logs extraction first (most reliable)
- Updated documentation in `docs/ALL_STAR_ROSTERS.md` to reflect working solution

### Added
- Blog notes documenting All-Star data extraction solution and repository organization decisions

## [0.3.0] - 2026-01-05

### Changed
- **Repository reorganization**: Moved scraping code to `data_fetch/` directory
  - All scraping modules (`scrape_milb.py`, `fetch_milb_*.py`, `fetch_all_star.py`) moved to `data_fetch/`
  - Scraping code excluded from Git tracking (via `.gitignore`) to prevent abuse
  - Updated imports in `src/ingest.py` to use `data_fetch.*` modules
- **Logs organization**: Moved runtime files to `logs/` directory
  - `ingestion.log` and `ingestion.pid` moved to `logs/`
  - Updated `monitor_ingestion.sh` to reference new log locations
  - Added `logs/` to `.gitignore`
- **IDE configuration**: Configured pipenv virtualenv for Cursor/VS Code
  - Updated `.vscode/settings.json` to use pipenv Python interpreter
  - Resolved missing library errors in IDE

### Fixed
- Fixed import paths in tests after repository reorganization
- Skipped hanging tests that require scraping/API calls (`test_fetch_minor_league_pitching`, `test_run_ingestion`, `test_e2e_optimized_scraping`)

### Added
- `data_fetch/__init__.py` for proper Python package structure
- Enhanced `.gitignore` to exclude scraping code and logs

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

