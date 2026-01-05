# Data Enrichment Setup: Draft Information & Birth Dates

## Overview

We've implemented functionality to enrich player data with:
1. **Draft Information** (draft_round, draft_year, draft_position) - ✅ Implemented
2. **Birth Dates** (for age_at_debut calculation) - ⚠️ Blocked (Baseball Reference 403)

## Implementation Status

### ⚠️ Draft Information (Implemented, but may be blocked)

**Source**: pybaseball `amateur_draft()` function
**Status**: Code implemented, but Baseball Reference may be blocking (similar to other BR scraping)

**How It Works**:
- Fetches draft data year-by-year, round-by-round (2000-2023)
- Matches players by MLBAM ID
- Adds `draft_round`, `draft_year`, `draft_position` columns

**Known Issues**:
- Baseball Reference may return "No tables found" (similar to 403 blocking)
- May need alternative source or manual data entry

**Files**:
- `src/enrich_player_data.py`: Core enrichment functions
- `scripts/enrich_player_data.py`: Standalone script to run enrichment

**Usage**:
```bash
# After initial ingestion, run enrichment
python scripts/enrich_player_data.py

# This will:
# - Fetch birth dates for all players (enables age_at_debut calculation)
# - Fetch draft years for all players
# - Takes ~5-10 minutes for ~2000 players (with 0.2s delay)
```

**Note**: This will make ~960 API calls (24 years × 40 rounds), so it will take time (~10-20 minutes with rate limiting)

### ✅ Birth Dates (Working via MLB Stats API)

**Source**: MLB Stats API `/api/v1/people/{id}` endpoint
**Status**: ✅ Working

**Impact**: `age_at_debut` feature can now be calculated!

## Integration Points

### 1. Schema Updates (`src/schemas.py`)
- Added `draft_round`, `draft_year`, `draft_position` to `PlayerSchema`
- `FeatureSchema` already has these fields

### 2. Ingestion (`src/ingest.py`)
- Added draft columns (null initially) to player DataFrame
- Enrichment code is commented out (can be enabled when ready)

### 3. Feature Engineering (`src/featurize.py`)
- Updated `create_progression_features()` to calculate `age_at_debut` when birth_date is available
- Added draft features join in `engineer_features()`

## Next Steps

### Immediate (Testing)
1. ✅ Install `html5lib` dependency (`pipenv install html5lib`)
2. ✅ Test draft data fetching with small sample (2010, round 1)
3. ✅ Verify column mapping works correctly
4. ✅ Run full enrichment script

### Short-Term
1. Run enrichment script to add draft data to existing players
2. Re-run feature engineering to include draft features
3. Evaluate impact on model performance

### Long-Term
1. Find alternative source for birth dates (or accept null)
2. Consider caching draft data to avoid re-fetching
3. Add draft data to CI/CD pipeline

## Expected Impact

### Draft Information
- **Coverage**: ~60-70% of players (some are international signings, undrafted)
- **Features Added**: 3 (draft_round, draft_year, draft_position)
- **Model Impact**: Moderate (draft round correlates with All-Star status)

### Birth Dates
- **Coverage**: 0% (currently blocked)
- **Features Added**: 1 (age_at_debut)
- **Model Impact**: Low-Moderate (age at debut is predictive but not critical)

## Testing

```bash
# Test draft fetching
python3 << 'PYTHON'
from src.enrich_player_data import fetch_draft_data_from_pybaseball
draft_df = fetch_draft_data_from_pybaseball(start_year=2010, end_year=2010)
print(draft_df.head())
PYTHON

# Run full enrichment (will take ~10-20 minutes)
python scripts/enrich_player_data.py
```

## Notes

- Draft data fetching is slow (~960 API calls)
- Consider running overnight or in background
- Can be interrupted and resumed (data saved incrementally)
- Birth date fetching is disabled until we find alternative source

