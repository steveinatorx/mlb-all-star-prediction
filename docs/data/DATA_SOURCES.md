# Data Sources and Implementation Status

## Current Status

### ✅ Implemented
- **Player Info** (`fetch_player_info()`): ✅ Working
  - Uses pybaseball `chadwick_register()` for comprehensive player ID mapping
  - Gets MLBAM IDs, Baseball Reference IDs, FanGraphs IDs, MLB debut dates
  - Supports players from 1871+ (via Chadwick Register)
  - **Status**: Fully functional

- **Minor League Pitching** (`fetch_minor_league_pitching()`): ✅ Working
  - Primary: MiLB.com scraping (`src/fetch_milb_mlbcom.py`) - ✅ Fully functional
    - Uses league-level scraping (more efficient than player-by-player)
    - Handles pagination automatically
    - Supports multiple leagues and year ranges
    - Uses MLBAM IDs directly (no ID mapping needed)
  - Fallback: FanGraphs API (`src/fetch_milb_fangraphs.py`) - ✅ Fully functional
    - Supports minor league stats from ~2005+ (tested back to 2005)
  - **Status**: Fully functional with both sources, MiLB.com prioritized

### ⚠️ Partially Implemented
- **All-Star Rosters** (`fetch_all_star_rosters()`): ⚠️ Partial
  - Attempts Lahman database via pybaseball (may fail)
  - Falls back to mock data
  - **Options**: Baseball Reference scraping, MLB Stats API, manual data entry

## Data Source Options

### Option 1: Baseball Reference Scraping (Blocked)
- **Status**: ❌ Currently blocked (403 Forbidden)
- **Module**: `src/scrape_milb.py`
- **Attempted Solutions**:
  - ✅ Integrated `pybaseball.BRefSession` for rate limiting (10 req/min)
  - ✅ Added browser-like headers (User-Agent, Accept, etc.)
  - ✅ Visited main page first to get cookies
  - ❌ Still getting 403 errors - likely advanced bot detection
- **Workarounds** (not implemented):
  - Use rotating proxies
  - Use Selenium/Playwright for browser automation with JavaScript rendering
  - Contact Baseball Reference for API access
- **Recommendation**: Focus on FanGraphs API and MiLB.com scraping (both working)

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

### Option 4: MiLB.com ✅ IMPLEMENTED (Primary Source)
- **Status**: ✅ Fully functional and tested
- **Module**: `src/fetch_milb_mlbcom.py`
- **What works**: 
  - Official Minor League Baseball website
  - Uses MLBAM IDs directly (no ID mapping needed)
  - League-level scraping (more efficient than player-by-player)
  - Handles pagination automatically (`?page=1`, `?page=2`, etc.)
  - Supports year filtering (`?season=YYYY`)
  - URL pattern: `https://www.milb.com/{league}/stats/pitching?season=YYYY&page=N`
  - Scrapes multiple leagues across year ranges efficiently
- **Advantages**:
  - More efficient than player-by-player scraping
  - Handles pagination, league switching, and year filtering
  - Official source with comprehensive coverage
  - No API rate limits (just respectful delays)
- **Limitations**: 
  - Requires web scraping (HTML structure may change over time)
  - Slower than API calls but more reliable than Baseball Reference
- **Priority**: Primary source (before FanGraphs) due to efficiency and reliability

### Option 5: The Baseball Cube
- **Status**: ❓ Not explored yet
- **Potential**: Offers minor league stats (may require purchase)
- **Link**: https://www.thebaseballcube.com/

### Option 6: Retrosheet
- **Status**: ❓ Limited historical coverage
- **Potential**: Historical minor league data
- **Approach**: Check Retrosheet minor league files

### Option 7: Manual Data Entry / Pre-existing Dataset
- **Status**: ✅ Most reliable short-term solution
- **Approach**: 
  - Use existing datasets if available
  - Manual entry for key players
  - Focus on recent years (2020-2023) first

## Year Limitations

- **FanGraphs API**: Supports minor league stats from ~2005+ (tested)
- **Chadwick Register**: Player IDs available from 1871+
- **MLB Stats API** (for player ID discovery): Only supports 2008+ (not required if using Chadwick Register)

**Project Scope**: 
- **Focus**: Players who debuted 2005-2023 (ensures full data coverage)
- **Rationale**: FanGraphs API provides complete minor league stats from 2005+, eliminating data gaps

**Current Setup**: 
- Chadwick Register for player IDs (all years)
- FanGraphs API for minor league stats (2005+) - primary source
- MiLB.com scraping as fallback/supplement (2005+)

## Recommended Next Steps

1. **Short-term**: ✅ Data pipeline working with FanGraphs API
2. **Medium-term**: 
   - Implement All-Star roster fetching (currently using mock data)
   - Test full ingestion pipeline with 2005+ data
   - Verify data completeness and quality
3. **Long-term**: 
   - Expand to earlier years if needed (would require alternative data sources)
   - Build relationships for API access
   - Use combination of sources for redundancy

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

