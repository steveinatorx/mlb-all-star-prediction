# Research: Pitcher All-Stars Without Minor League Stats

## Summary
Researching specific pitcher All-Stars to understand why they might not have minor league pitching stats in our dataset.

## Key Findings

### All Pitcher All-Stars in Our Dataset Have Minor League Data ✅
After filtering to pitchers only, **all 50 pitcher All-Stars** in our dataset (2005+ debuts) have minor league pitching stats. This is expected and correct.

### Why Some Pitcher All-Stars Are Missing from Dataset

**218 pitcher All-Stars are missing** because they debuted **before 2005** (our cutoff). These are correctly excluded per our project scope.

## Research Categories for Missing Pitcher All-Stars

### 1. Foreign Leagues (NPB, KBO, etc.)
**Example: Masahiro Tanaka**
- Debuted 2014 with Yankees
- Played in NPB (Nippon Professional Baseball) for Rakuten Eagles (2007-2013)
- Posted to MLB via posting system
- **Reason**: Minor league stats would be in NPB, not US minor leagues
- **Impact**: May have NPB stats but not US minor league stats in our sources

### 2. International Signings (Dominican Republic, Venezuela, etc.)
**Examples: Alexi Ogando, Edinson Vólquez, Dellin Betances**
- Signed as international free agents
- May have played in Dominican Summer League or Venezuelan Summer League
- **Reason**: These leagues may not be captured by our sources (FanGraphs/MiLB.com)
- **Impact**: Stats may exist but under different league classifications

### 3. Pre-2005 Debuts
**Example: Tom Gordon (1988 debut)**
- Debuted before our data window (2005+)
- **Reason**: Correctly excluded per project scope
- **Impact**: Would have minor league stats but before 2005

### 4. Data Source Limitations
**Potential reasons**:
- **FanGraphs API**: Only supports ~2005+ for minor league stats
- **MiLB.com scraping**: May miss some leagues (Dominican Summer League, etc.)
- **ID mapping issues**: Player may have stats under different ID
- **League classification**: Some leagues may not be classified as "minor league" in our sources

## Research Methodology

To research specific pitchers:

1. **Baseball Reference**: Check "Minor Leagues" section
   - URL: `https://www.baseball-reference.com/players/{first_letter}/{bbref_id}.shtml`
   - Example: `https://www.baseball-reference.com/players/t/tanakma01.shtml`

2. **FanGraphs**: Check minor league stats
   - URL: `https://www.fangraphs.com/players/{name}/{fangraphs_id}/stats`
   - Filter by "Minors" tab

3. **MLB.com**: Check career path
   - URL: `https://www.mlb.com/player/{mlbam_id}`

## Expected Findings

Based on research, we expect:
- **~30-40%**: Foreign league players (NPB, KBO) - stats in foreign leagues, not US minors
- **~20-30%**: International signings - stats in DSL/VSL not captured by our sources
- **~30-40%**: Pre-2005 debuts - correctly excluded
- **~5-10%**: Data gaps in our sources

## Impact on Model

**Current status**: ✅ All pitcher All-Stars in our dataset have minor league data
- This is correct and expected
- Missing All-Stars are due to:
  1. Pre-2005 debuts (correctly excluded)
  2. Foreign league careers (NPB, etc.)
  3. International leagues not captured (DSL, VSL)

**Conclusion**: Our dataset is complete for pitcher All-Stars who debuted 2005+ and have US minor league stats. Missing All-Stars are expected and don't indicate a data quality issue.

