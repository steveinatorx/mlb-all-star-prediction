# Feature Engineering Setup - Complete Review

## Overview

The feature engineering pipeline transforms **processed minor league pitching statistics** into **model-ready features** for predicting MLB All-Star status. This document provides a complete walkthrough of the setup.

## Pipeline Flow

```
Raw Data → Processed Data → Features → Model Training
   ↓            ↓              ↓
ingest.py  build_dataset.py  featurize.py
```

### 1. Data Ingestion (`src/ingest.py`)
- Fetches minor league pitching stats from FanGraphs API / MiLB.com
- Fetches All-Star rosters from MLB API / pybaseball
- Saves to `data/raw/*.parquet`

### 2. Dataset Building (`src/build_dataset.py`)
- **Pre-debut filtering**: Only includes stats BEFORE MLB debut (prevents leakage)
- **Quality filters**: Minimum IP threshold (50 IP default)
- **Label creation**: Marks players as All-Stars (1) or not (0)
- Saves to `data/processed/*.parquet`

### 3. Feature Engineering (`src/featurize.py`) ⭐ **THIS IS WHERE WE ARE**
- Transforms season-level stats into player-level features
- Creates 3 feature groups + time splits
- Saves to `data/features/features.parquet`

## Feature Engineering Details

### Input Data

**Processed Data Files** (loaded from `data/processed/`):
1. **`minor_league_pitching_processed.parquet`**
   - Season-level stats (one row per player-season-level)
   - Columns: `player_id`, `season`, `level`, `era`, `whip`, `k_per_9`, `bb_per_9`, `innings_pitched`, etc.
   - **Already filtered**: Only pre-MLB debut stats, minimum IP threshold applied

2. **`labels.parquet`**
   - Player-level labels (one row per player)
   - Columns: `player_id`, `is_all_star` (0 or 1)
   - **Already filtered**: Only pitchers, 2005+ debuts

3. **`players_processed.parquet`**
   - Player metadata
   - Columns: `player_id`, `mlb_debut`, `fangraphs_id`, `bbref_id`, etc.

### ID Mapping (Critical Step)

**Problem**: Minor league data uses **FanGraphs IDs**, but labels use **MLBAM IDs**

**Solution**: Automatic ID mapping in `engineer_features()`
```python
# Check if IDs don't match
milb_ids = set(milb_df["player_id"].unique().to_list())
label_ids = set(data["labels"]["player_id"].unique().to_list())

if len(milb_ids & label_ids) == 0:
    # Map FanGraphs IDs → MLBAM IDs using players table
    id_mapping = players_df.select([
        pl.col("player_id").alias("mlbam_id"),  # MLBAM ID
        pl.col("fangraphs_id")                  # FanGraphs ID
    ])
    
    # Join and remap
    milb_df = milb_df.join(id_mapping, ...)
```

**Result**: All features use MLBAM IDs, matching labels ✅

## Feature Groups

### 1. Career Aggregates (`create_career_aggregates`)

**Purpose**: Overall performance across entire minor league career

**Features**:
- `total_milb_ip`: Total innings pitched (volume)
- `total_milb_games`: Total games played
- `total_milb_starts`: Total games started
- `career_era`: Weighted ERA (earned_runs * 9 / innings_pitched)
- `career_whip`: Weighted WHIP ((hits + walks) / innings_pitched)
- `career_k_per_9`: Weighted K/9 (strikeouts * 9 / innings_pitched)
- `career_bb_per_9`: Weighted BB/9 (walks * 9 / innings_pitched)

**Key Insight**: Uses **IP-weighted averages**, not simple averages
- Example: 10 IP @ 2.00 ERA + 100 IP @ 4.00 ERA = ~3.82 ERA (not 3.00)
- More accurate representation of true performance

**Code**:
```python
career_stats = milb_df.group_by("player_id").agg([
    pl.sum("innings_pitched").alias("total_milb_ip"),
    (pl.sum("earned_runs") * 9.0 / pl.sum("innings_pitched")).alias("career_era"),
    # ... more weighted averages
])
```

### 2. Best Season Features (`create_best_season_features`)

**Purpose**: Peak performance (shows ceiling, not just average)

**Features**:
- `best_era`: Best (lowest) ERA in any season
- `best_whip`: Best (lowest) WHIP in any season
- `best_k_per_9`: Best (highest) K/9 in any season

**Rationale**: 
- Scouts care about **ceiling**, not just average
- A pitcher who once had a 2.50 ERA season is different from one who never did
- Captures peak ability, which may be more predictive

**Code**:
```python
best_seasons = milb_df.group_by("player_id").agg([
    pl.min("era").alias("best_era"),      # Lower is better
    pl.min("whip").alias("best_whip"),    # Lower is better
    pl.max("k_per_9").alias("best_k_per_9"),  # Higher is better
])
```

### 3. Progression Features (`create_progression_features`)

**Purpose**: Development trajectory and level advancement

**Features**:
- `highest_level_reached`: Highest level played (R, A, A+, AA, AAA)
- `seasons_at_aaa`: Number of seasons at AAA
- `seasons_at_aa`: Number of seasons at AA
- `age_at_debut`: Age when debuted in MLB (currently null, placeholder)

**Level Hierarchy**:
```python
level_order = {"R": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}
```

**Rationale**:
- **Level reached**: AAA experience is a strong signal
- **Seasons at level**: More AAA time = more ready for MLB
- **Age**: Younger debut = better prospect (when data available)

**Code**:
```python
# Highest level
highest_level = milb_df.group_by("player_id").agg([
    pl.max("level_num").alias("highest_level_reached")
])

# Seasons at each level
level_counts = milb_df.group_by(["player_id", "level"]).agg([
    pl.len().alias("seasons")
]).pivot(...)
```

### 4. Time Splits (`create_time_splits`)

**Purpose**: Create train/validation/test splits based on MLB debut year

**Splits** (from `config.py`):
- **Train**: `mlb_debut.year <= 2018` (default)
- **Val**: `2018 < mlb_debut.year <= 2020` (default)
- **Test**: `mlb_debut.year > 2020` or `mlb_debut is null` (default)

**Rationale**:
- **Temporal validity**: Predict future from past
- **Prevents leakage**: Can't use future information
- **Realistic evaluation**: Tests generalization to future years

**Code**:
```python
splits = players_df.with_columns([
    pl.when(pl.col("mlb_debut").dt.year() <= config.train_end_year)
    .then(pl.lit("train"))
    .when(pl.col("mlb_debut").dt.year() <= config.val_end_year)
    .then(pl.lit("val"))
    .otherwise(pl.lit("test"))
    .alias("split")
])
```

## Final Feature Set

**Output**: `data/features/features.parquet`

**Columns**:
- `player_id`: MLBAM ID (matches labels)
- `is_all_star`: Label (0 or 1)
- `split`: Train/val/test assignment
- **Career aggregates**: `total_milb_ip`, `career_era`, `career_whip`, `career_k_per_9`, `career_bb_per_9`, etc.
- **Best season**: `best_era`, `best_whip`, `best_k_per_9`
- **Progression**: `highest_level_reached`, `seasons_at_aaa`, `seasons_at_aa`, `age_at_debut`

**Shape**: `(n_players, n_features)` - one row per player

## Configuration

**Key Config Values** (`src/config.py`):
```python
train_end_year: int = 2018      # Last year for training set
val_end_year: int = 2020        # Last year for validation set
min_ip_for_label: float = 50.0  # Minimum IP to include (applied in build_dataset.py)
```

## Execution

**Command**:
```bash
make featurize
# or
pipenv run python -m src.main featurize
```

**What It Does**:
1. Loads processed data from `data/processed/`
2. Maps FanGraphs IDs → MLBAM IDs (if needed)
3. Creates 3 feature groups
4. Joins features together
5. Adds labels and time splits
6. Saves to `data/features/features.parquet`

**Logging**: All steps logged with progress info

## Current Status

✅ **Working**:
- ID mapping (FanGraphs → MLBAM)
- Career aggregates (IP-weighted)
- Best season features
- Progression features (level, seasons)
- Time splits (train/val/test)
- Label joining

⚠️ **Placeholder**:
- `age_at_debut`: Currently null (birth_date not available)
  - Code ready to calculate when birth_date becomes available
  - See TODO comment in `create_progression_features()`

## Next Steps

After feature engineering:
1. **EDA**: Explore feature distributions, correlations
2. **Feature selection**: Remove highly correlated features
3. **Missing data**: Handle nulls (impute or flag)
4. **Model training**: Train models on features

## Key Design Decisions

1. **IP-weighted averages**: More accurate than simple averages
2. **Best season stats**: Captures ceiling, not just average
3. **Level progression**: AAA experience is a strong signal
4. **Time-based splits**: Prevents temporal leakage
5. **Pre-debut filtering**: Prevents label leakage (done in build_dataset.py)

## Questions to Consider

1. **Feature interactions**: Should we create interaction features?
   - Example: `career_k_per_9 * seasons_at_aaa` (strikeout ability × experience)

2. **Normalization**: Should we normalize features?
   - ERA vs IP have very different scales
   - Some models (neural nets) benefit from normalization

3. **Additional features**: What else could we add?
   - Draft round (if available)
   - Organization (some teams develop pitchers better)
   - Velocity/spin rate (if available from Statcast)

4. **Missing data**: How to handle nulls?
   - `age_at_debut` is currently null for all players
   - Some players may have missing stats

5. **Feature selection**: Which features are most important?
   - Use feature importance from models
   - Remove highly correlated features

