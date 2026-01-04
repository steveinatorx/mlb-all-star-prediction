# Data Sources and Implementation Status

## Current Status

### ✅ Implemented
- **Player Info** (`fetch_player_info()`): ✅ Working
  - Uses pybaseball `chadwick_register()` for comprehensive player ID mapping
  - Gets MLBAM IDs, Baseball Reference IDs, FanGraphs IDs, MLB debut dates
  - Supports players from 1871+ (via Chadwick Register)
  - **Status**: Fully functional

- **Minor League Pitching** (`fetch_minor_league_pitching()`): ✅ Working
  - Uses FanGraphs API (`src/fetch_milb_fangraphs.py`)
  - Supports minor league stats from ~2005+ (tested back to 2005)
  - Automatically fetches when player info with FanGraphs IDs is available
  - **Status**: Fully functional and integrated

### ⚠️ Partially Implemented
- **All-Star Rosters** (`fetch_all_star_rosters()`): ⚠️ Partial
  - Attempts Lahman database via pybaseball (may fail)
  - Falls back to mock data
  - **Options**: Baseball Reference scraping, MLB Stats API, manual data entry

## Data Source Options

### Option 1: Baseball Reference Scraping (Blocked)
- **Status**: ❌ Currently blocked (403 Forbidden)
- **Module**: `src/scrape_milb.py`
- **Workarounds**:
  - Use rotating proxies
  - Use Selenium/Playwright for browser automation
  - Respect rate limits more aggressively
  - Contact Baseball Reference for API access

### Option 2: MLB Stats API (Limited)
- **Status**: ⚠️ API accessible but doesn't return minor league stats
- **Module**: `src/fetch_milb_mlbapi.py`
- **What works**: Team rosters, game schedules
- **What doesn't**: Historical minor league player statistics
- **Note**: API may have minor league data in different endpoints (needs exploration)

### Option 3: FanGraphs ✅ IMPLEMENTED
- **Status**: ✅ Working and integrated
- **Module**: `src/fetch_milb_fangraphs.py`
- **What works**: 
  - Minor league pitching stats via public API
  - Supports years ~2005+ (tested back to 2005)
  - Returns comprehensive stats (ERA, WHIP, K/9, BB/9, etc.)
- **API Endpoint**: `https://www.fangraphs.com/api/players/stats?playerid={id}&position=P&type=0&season={year}`
- **Limitations**: Requires FanGraphs player IDs (available via Chadwick Register)

### Option 4: The Baseball Cube
- **Status**: ❓ Not explored yet
- **Potential**: Offers minor league stats (may require purchase)
- **Link**: https://www.thebaseballcube.com/

### Option 5: Retrosheet
- **Status**: ❓ Limited historical coverage
- **Potential**: Historical minor league data
- **Approach**: Check Retrosheet minor league files

### Option 6: Manual Data Entry / Pre-existing Dataset
- **Status**: ✅ Most reliable short-term solution
- **Approach**: 
  - Use existing datasets if available
  - Manual entry for key players
  - Focus on recent years (2020-2023) first

## Year Limitations

- **FanGraphs API**: Supports minor league stats from ~2005+ (tested)
- **Chadwick Register**: Player IDs available from 1871+
- **MLB Stats API** (for player ID discovery): Only supports 2008+ (not required if using Chadwick Register)

**Current Setup**: Uses Chadwick Register for player IDs (all years) + FanGraphs API for minor league stats (2005+)

## Recommended Next Steps

1. **Short-term**: ✅ Data pipeline working with FanGraphs API
2. **Medium-term**: 
   - Implement All-Star roster fetching (currently using mock data)
   - Test FanGraphs API with earlier years (< 2005) if needed
   - Consider Baseball Reference scraping for pre-2005 minor league stats if needed
3. **Long-term**: 
   - Consider purchasing dataset from The Baseball Cube for comprehensive historical coverage
   - Build relationships for API access
   - Use combination of sources

## Testing the Pipeline

Even with mock data, you can:
- ✅ Test the entire pipeline end-to-end
- ✅ Validate leakage prevention logic
- ✅ Test feature engineering
- ✅ Train and evaluate models
- ✅ Generate reports

The pipeline is designed to work with mock data, so you can develop and test everything while working on getting real data sources.

## Notes

- The scraping infrastructure is ready - just needs a working data source
- All modules handle missing data gracefully (fall back to mock)
- Once you have real data, just replace the mock data functions

