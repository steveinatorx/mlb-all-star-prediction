# MiLB.com Scraping Optimization Strategy

## Current Problem
We're scraping ALL leagues for ALL years, which includes:
- Players who never made MLB (we don't need)
- Years outside our target range (wasteful)
- Leagues/years that don't contain our target players

## Solution: Calculate Optimal Scraping Range

### Strategy
1. **Load player info** with MLB debut dates
2. **Calculate optimal year range**:
   - For each player: minor league years = `[debut_year - 5, debut_year - 1]`
   - Overall range = `min(debut_year - 5)` to `max(debut_year - 1)`
3. **Scrape only optimal range** + filter immediately to our players
4. **Cache aggressively** to avoid re-scraping

### Implementation
- Add `calculate_optimal_scraping_range()` function ✅
- Update `fetch_minor_league_pitching()` to use optimal range
- Filter results immediately to player set
- Cache by player set + year range

### Benefits
- Reduces scraping by ~80-90% (only relevant years)
- Filters out irrelevant players immediately
- Still comprehensive (covers all needed years)
- One-time scrape with caching

