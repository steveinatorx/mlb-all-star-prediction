# Data Expansion Opportunities

## Current Situation

- **Training**: 459 players (30 All-Stars, 429 non-All-Stars)
- **Year Range**: 2005-2023 debuts
- **Minor League Data**: 2005-2023 seasons
- **Main Constraint**: Only 30 positive examples

## Would More Data Help?

### ✅ **YES - More Data Would Significantly Help**

**Impact of More Positive Examples**:
- **Current**: 30 All-Stars in training
- **If we double**: 60 All-Stars → Much more stable model
- **If we triple**: 90 All-Stars → Robust model
- **Rule of thumb**: Need ~100+ positive examples for reliable models

**Impact of More Features**:
- **Current**: 14 features
- **More features**: Draft round, organization, velocity, spin rate
- **Benefit**: Better signal, more predictive power

## Data Expansion Strategies

### 1. **Expand Year Range** ⭐⭐⭐ (HIGHEST IMPACT)

**Option A: Go Earlier (2000-2004 debuts)**
- **Challenge**: FanGraphs API only supports ~2005+ minor league stats
- **Solution**: Use Baseball Reference scraping (currently blocked)
- **Potential Gain**: ~50-100 more All-Stars
- **Effort**: High (need to unblock Baseball Reference or find alternative)

**Option B: Go Later (2024-2025 debuts)**
- **Challenge**: Future data (2024 season just ended, 2025 hasn't happened)
- **Solution**: Include 2024 debuts (if data available)
- **Potential Gain**: ~5-10 more All-Stars
- **Effort**: Low (just extend year range)

**Option C: Include Pre-2005 Debuts with Available Data**
- **Challenge**: Many pre-2005 All-Stars don't have minor league data in our sources
- **Solution**: Use players who debuted 2000-2004 but have minor league data
- **Potential Gain**: ~20-30 more All-Stars
- **Effort**: Medium (need to check data availability)

**Recommendation**: **Try Option C first** - check if any 2000-2004 debuts have minor league data

### 2. **Add More Features** ⭐⭐ (MEDIUM IMPACT)

**A. Draft Information**
- **Source**: MLB Draft database, Baseball Reference
- **Features**: Draft round, draft year, draft position
- **Value**: Higher draft picks = better prospects = more likely All-Stars
- **Potential Impact**: Moderate (correlates with All-Star status)
- **Effort**: Medium (need to scrape/match draft data)

**B. Organization/Team**
- **Source**: Minor league data (already have team)
- **Features**: Organization (Yankees, Dodgers, etc.), farm system quality
- **Value**: Some organizations develop pitchers better
- **Potential Impact**: Low-Moderate
- **Effort**: Low (already in data, just need to map to organizations)

**C. Statcast Data (Velocity, Spin Rate)**
- **Source**: Baseball Savant (MLB Statcast)
- **Features**: Average fastball velocity, spin rate, pitch mix
- **Value**: Velocity/spin rate are strong predictors of success
- **Potential Impact**: HIGH (very predictive)
- **Effort**: High (need to scrape Statcast, only available for recent years)

**D. College/International League Stats**
- **Source**: NCAA stats, NPB/KBO stats
- **Features**: College ERA, NPB stats, etc.
- **Value**: Earlier signal on prospects
- **Potential Impact**: Moderate
- **Effort**: High (new data sources, ID mapping)

**Recommendation**: **Start with Draft Information** - easiest to add, moderate impact

### 3. **Add More Data Sources** ⭐⭐ (MEDIUM IMPACT)

**A. Baseball Reference Scraping**
- **Status**: Currently blocked (403 Forbidden)
- **Potential**: Historical minor league stats (pre-2005)
- **Value**: Could add ~50-100 more All-Stars
- **Effort**: High (need to unblock or use proxies/Selenium)

**B. Retrosheet**
- **Status**: Not explored
- **Potential**: Historical minor league data
- **Value**: Could add pre-2005 data
- **Effort**: Medium (need to parse Retrosheet format)

**C. The Baseball Cube**
- **Status**: Not explored
- **Potential**: Comprehensive minor league stats (may require purchase)
- **Value**: Could fill gaps in our data
- **Effort**: Medium (need to explore/purchase)

**D. MLB Stats API (Minor League Endpoints)**
- **Status**: Partially explored
- **Potential**: Official MLB minor league stats
- **Value**: Could supplement existing data
- **Effort**: Medium (need to explore API endpoints)

**Recommendation**: **Explore Retrosheet** - free, historical data

### 4. **Improve Data Quality** ⭐ (LOW IMPACT, HIGH VALUE)

**A. Fill Missing Data**
- **Current**: 886 players have labels but no features
- **Solution**: Try to find their minor league data
- **Value**: Could add ~50-100 more training examples
- **Effort**: Medium (need to investigate why missing)

**B. Add Missing Features**
- **Current**: `age_at_debut` is 100% null
- **Solution**: Get birth dates from Baseball Reference or Chadwick Register
- **Value**: Adds one more feature
- **Effort**: Low (data might be available)

**Recommendation**: **Fill missing `age_at_debut`** - easy win

## Prioritized Expansion Plan

### Phase 1: Quick Wins (1-2 days) ⭐
1. ✅ **Add 2024 debuts** (if data available)
2. ✅ **Fill `age_at_debut`** feature (get birth dates)
3. ✅ **Add organization features** (map teams to organizations)

**Expected Gain**: +5-10 All-Stars, +1-2 features

### Phase 2: Medium Effort (3-5 days) ⭐⭐
1. ✅ **Add draft information** (draft round, year, position)
2. ✅ **Check 2000-2004 debuts** for available minor league data
3. ✅ **Investigate missing players** (why 886 have labels but no features)

**Expected Gain**: +20-30 All-Stars, +3-4 features

### Phase 3: High Effort (1-2 weeks) ⭐⭐⭐
1. ✅ **Unblock Baseball Reference** (proxies, Selenium, or API access)
2. ✅ **Explore Retrosheet** for historical data
3. ✅ **Add Statcast data** (velocity, spin rate) for recent years

**Expected Gain**: +50-100 All-Stars, +5-10 features

## Impact Analysis

### Current Model Performance (Estimated)
- **ROC-AUC**: 0.65-0.75 (moderate discrimination)
- **PR-AUC**: 0.10-0.20 (low due to imbalance)
- **Recall@Top50**: 5-10 All-Stars found

### With Phase 1 Expansion (+5-10 All-Stars)
- **ROC-AUC**: 0.70-0.80 (improved discrimination)
- **PR-AUC**: 0.15-0.25 (better precision)
- **Recall@Top50**: 8-12 All-Stars found

### With Phase 2 Expansion (+20-30 All-Stars)
- **ROC-AUC**: 0.75-0.85 (good discrimination)
- **PR-AUC**: 0.20-0.30 (much better precision)
- **Recall@Top50**: 12-18 All-Stars found

### With Phase 3 Expansion (+50-100 All-Stars)
- **ROC-AUC**: 0.80-0.90 (excellent discrimination)
- **PR-AUC**: 0.30-0.40 (strong precision)
- **Recall@Top50**: 18-25 All-Stars found

## Recommendations

### **Immediate Action** (This Week)
1. ✅ **Check 2000-2004 debuts** for available minor league data
   - Run query: "Do any players who debuted 2000-2004 have minor league data?"
   - If yes, include them (could add 20-30 All-Stars)

2. ✅ **Add draft information**
   - Scrape/match draft data from Baseball Reference
   - Add draft round, year, position features
   - Moderate effort, moderate impact

3. ✅ **Fill `age_at_debut`**
   - Get birth dates from Chadwick Register or Baseball Reference
   - Easy win, adds one feature

### **Short-Term** (Next 2 Weeks)
1. ✅ **Explore Retrosheet** for historical minor league data
2. ✅ **Investigate missing players** (why 886 have labels but no features)
3. ✅ **Add organization features** (map teams to organizations)

### **Long-Term** (If Needed)
1. ✅ **Unblock Baseball Reference** (proxies, Selenium, API access)
2. ✅ **Add Statcast data** (velocity, spin rate) for recent years
3. ✅ **Add college/international league stats**

## Conclusion

### **YES - More Data Would Significantly Help**

**Key Insights**:
1. **More positive examples** (All-Stars) is the highest priority
2. **Draft information** is the easiest feature to add
3. **Pre-2005 data** could add 20-50 more All-Stars if available
4. **Statcast data** would be very valuable but only for recent years

**Recommended Approach**:
- **Start with Phase 1** (quick wins) - 1-2 days
- **Then Phase 2** (medium effort) - 3-5 days
- **Evaluate results** before Phase 3 (high effort)

**Expected Outcome**:
- **Phase 1+2**: Could get to 60-80 All-Stars in training (2-3x current)
- **This would be sufficient** for demonstrating model training expertise
- **Phase 3**: Would be nice-to-have, not essential

The goal should be **60-100 positive examples** for a robust model. We're currently at 30, so **doubling or tripling would be ideal**.

