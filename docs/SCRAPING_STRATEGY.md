# MiLB.com Scraping Strategy

## Problem
We need minor league stats for players who debuted in MLB, but only BEFORE their debut (to prevent label leakage). Currently, we're scraping ALL leagues for ALL years, which is wasteful.

## Solution: Targeted Scraping

### Step 1: Calculate Optimal Range
1. Load player info with MLB debut dates
2. For each player, calculate their minor league years: `[debut_year - 5, debut_year - 1]`
3. Find the overall range: `min(debut_year - 5)` to `max(debut_year - 1)`
4. This is typically much smaller than `start_year` to `end_year`

### Step 2: Scrape Only What We Need
1. Scrape league pages only for the optimal year range
2. Filter results immediately to our player set
3. Cache aggressively to avoid re-scraping

### Step 3: Filter Post-Scrape
1. In `build_dataset.py`, we already filter to pre-debut stats
2. This provides a second layer of safety

## Example
- **Config**: `start_year=2015, end_year=2023`
- **Players**: 1000 players who debuted 2015-2023
- **Old approach**: Scrape 2015-2023 (9 years) × 15 leagues = 135 league-years
- **New approach**: 
  - Players debuted 2015-2023
  - Minor league years: 2010-2022 (debut - 5 to debut - 1)
  - Scrape 2010-2022 (13 years) × 15 leagues = 195 league-years
  - BUT: Filter immediately to our 1000 players
  - Result: Only ~10-20% of scraped data is relevant

## Optimization: Player-Specific Year Ranges
Even better: For each league/year, only scrape if at least one of our players had MiLB stats that year.

This requires:
1. For each player, determine their MiLB years
2. Group by league/year combinations
3. Only scrape those combinations

## Implementation Plan
1. Add `calculate_optimal_scraping_range()` function
2. Update `fetch_minor_league_pitching()` to use optimal range
3. Filter results immediately to our player set
4. Cache by player set + year range

