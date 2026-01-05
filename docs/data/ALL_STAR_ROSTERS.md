# All-Star Roster Data Sources

## Current Status: ✅ Working via pybaseball Game Logs

All-Star rosters are now successfully fetched from pybaseball using `all_star_game_logs()`:

1. **pybaseball's `all_star_game_logs()`**: ✅ Working
   - Extracts player IDs from All-Star game logs (1933-2024)
   - Maps retro IDs to MLBAM IDs using Chadwick Register
   - Successfully extracts ~389 All-Star appearances for 2005-2023
   - Covers 239 unique players

2. **pybaseball's `all_star_full()`**: ❌ Still Failing
   - Relies on Lahman database download
   - Error: "File is not a zip file"
   - Lahman database source appears to be blocked/inaccessible
   - **Workaround**: Using game logs instead (works perfectly)

3. **Baseball Reference Scraping**: ❌ Blocked (fallback only)
   - URL pattern: `https://www.baseball-reference.com/allstar/MLB-allstar-game-{year}.shtml`
   - Getting 403 Forbidden errors
   - Even with `BRefSession` and browser-like headers
   - **Not needed**: Game logs provide better coverage

## Why This Matters

All-Star rosters are **critical** for labeling our dataset:
- They define the positive class (All-Stars) vs negative class (non-All-Stars)
- Without real All-Star data, we can't train or evaluate models properly

## Alternative Solutions

### Option 1: Manual Dataset (Recommended for Now)
- **Source**: Publicly available All-Star rosters (Wikipedia, MLB.com, etc.)
- **Format**: CSV/JSON with columns: `player_id` (MLBAM), `season`, `is_all_star`
- **Years needed**: 2005-2023
- **Effort**: ~1-2 hours to compile manually
- **Status**: ✅ Most reliable short-term solution

### Option 2: MLB Stats API
- **Endpoint**: Need to explore `statsapi.mlb.com` endpoints
- **Potential**: `/api/v1/allStarGame/{year}` or similar
- **Status**: ❓ Not yet explored
- **Next step**: Research MLB Stats API documentation

### Option 3: Pre-existing Dataset
- **Sources**:
  - Kaggle datasets
  - Retrosheet (if they have All-Star data)
  - The Baseball Cube (may require purchase)
- **Status**: ❓ Not yet explored

### Option 4: Fix Lahman Download
- **Issue**: Lahman database download failing
- **Potential fixes**:
  - Manual download from GitHub: https://github.com/chadwickbureau/baseballdatabank
  - Point pybaseball to local copy
  - Update pybaseball version
- **Status**: ❓ Needs investigation

## Recommended Next Steps

1. **Short-term**: Create manual All-Star roster dataset
   - Use Wikipedia or MLB.com as source
   - Format: CSV with MLBAM IDs
   - Save to `data/raw/all_star_rosters.csv`

2. **Medium-term**: Explore MLB Stats API
   - Check if they have All-Star endpoints
   - Implement API-based fetching

3. **Long-term**: Fix Lahman/pybaseball integration
   - Or find reliable alternative data source

## Current Mock Data Impact

- **For development**: ✅ Allows pipeline to run end-to-end
- **For modeling**: ❌ Cannot train/evaluate without real labels
- **Action needed**: Replace mock data before model training

