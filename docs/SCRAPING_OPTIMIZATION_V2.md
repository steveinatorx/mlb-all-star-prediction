# MiLB.com Scraping Optimization Strategy V2

## Current Scope Analysis

For players who debuted 2015-2023:
- **Optimal range**: 2010-2022 (13 years)
- **Leagues**: 13 (AAA, AA, A+, A, Complex)
- **Total league-years**: 169
- **Estimated requests**: ~8,450 (assuming 50 pages/league-year)
- **Time estimate**: ~1.9 hours at 0.8s/request

## Problems
1. **Too many requests** - 8,450+ HTTP requests is excessive
2. **Rate limiting risk** - MiLB.com may block us
3. **Most data irrelevant** - Only ~10-20% of scraped players are in our dataset
4. **Redundant with FanGraphs** - FanGraphs API already works and is faster

## Proposed Strategy: Hybrid Approach

### Phase 1: Use FanGraphs API First (Fast, Reliable)
- ✅ Already implemented and working
- ✅ Faster than scraping (API vs HTML parsing)
- ✅ Covers ~2005+ (sufficient for our needs)
- **Time**: ~10-30 minutes for 1000 players

### Phase 2: Fill Gaps with MiLB.com (Targeted)
Only scrape MiLB.com for:
1. **Players missing from FanGraphs** (no FanGraphs ID or no data)
2. **Years before 2005** (if needed)
3. **Specific leagues/years** where we know players were active

### Phase 3: Smart Caching
- Cache by player set + year range
- Skip re-scraping if cache exists
- Cache individual league/year combinations

## Implementation Plan

1. **Prioritize FanGraphs**: Use it as primary source
2. **MiLB.com as supplement**: Only scrape missing data
3. **Progressive enhancement**: Start with FanGraphs, add MiLB.com incrementally
4. **Resume capability**: Save progress, can resume if interrupted

## Estimated Time Savings

- **FanGraphs only**: ~30 minutes (1000 players × 0.5s delay)
- **MiLB.com gaps**: ~10-20% of players × ~5 minutes = ~10-20 minutes
- **Total**: ~40-50 minutes vs 1.9 hours

## Rate Limiting Considerations

- **FanGraphs API**: No known limits, but we use 0.5s delay
- **MiLB.com**: Unknown limits, using 0.8s delay
- **Risk mitigation**: 
  - Use FanGraphs first (less risky)
  - Cache aggressively
  - Handle 429/503 errors gracefully
  - Can resume from cache if blocked

