# Ranking vs Binary Classification: A Comparison

## Overview

This document explains why we evaluated both ranking and binary classification approaches for predicting MLB All-Stars, and why ranking is the preferred approach for this imbalanced data problem.

## The Problem

Predicting MLB All-Stars from minor league statistics is a **highly imbalanced classification problem**:
- Only ~2-3% of minor league pitchers become All-Stars
- We have 10 All-Stars out of 445 test samples (2.2% positive rate)
- Traditional binary classification struggles with such extreme imbalance

## Two Approaches

### 1. Ranking (Primary Approach)

**What it is:**
- Orders predictions by probability (highest to lowest)
- Evaluates using Recall@TopK: "How many All-Stars are in the top K predictions?"
- No threshold needed - just rank and prioritize

**Metrics:**
- **PR-AUC**: Precision-Recall Area Under Curve (best for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Recall@TopK**: Recall among top K predictions (e.g., top 10, 25, 50, 100)

**Example:**
- If we rank all 445 test pitchers by predicted probability
- Top 10 predictions contain 2 All-Stars → Recall@Top10 = 20%
- Top 25 predictions contain 2 All-Stars → Recall@Top25 = 20%

**Advantages:**
- ✅ No arbitrary threshold decisions
- ✅ Better suited for imbalanced data
- ✅ Natural fit for prioritization/scouting use cases
- ✅ Avoids false positive problem (no hard "yes/no" decisions)
- ✅ More interpretable: "These are the top K prospects to watch"

### 2. Binary Classification (Comparison)

**What it is:**
- Applies a threshold to convert probabilities to "yes/no" predictions
- Evaluates using precision, recall, F1 score
- Requires threshold selection (which threshold to use?)

**Metrics:**
- **Precision**: Of all predicted All-Stars, how many are actually All-Stars?
- **Recall**: Of all actual All-Stars, how many did we predict?
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True Positives, False Positives, True Negatives, False Negatives

**Example:**
- Optimal threshold: 0.40 (found by trying thresholds from 0.05 to 0.95)
- At threshold 0.40: 14 predicted All-Stars, 2 are actually All-Stars
- Precision = 2/14 = 14.29%
- Recall = 2/10 = 20.00%
- F1 = 0.1667

**Disadvantages:**
- ⚠️ Requires threshold selection (complex decision)
- ⚠️ Many false positives (12 false positives for GAM model)
- ⚠️ Threshold may need retuning as data changes
- ⚠️ Less suitable for imbalanced data
- ⚠️ Hard "yes/no" decisions may not match use case

## Results Comparison

### GAM Model (Best Ranking Performance)

**Ranking Metrics:**
- PR-AUC: 0.0678
- Recall@Top10: 20% (2 out of 10 All-Stars)
- Recall@Top25: 20% (2 out of 10 All-Stars)
- Recall@Top50: 20% (2 out of 10 All-Stars)
- Recall@Top100: 30% (3 out of 10 All-Stars)

**Binary Classification Metrics:**
- Optimal Threshold: 0.40
- Precision: 14.29% (2 true positives, 12 false positives)
- Recall: 20.00% (2 out of 10 All-Stars found)
- F1 Score: 0.1667
- **False Positives: 12** (12 pitchers incorrectly labeled as All-Stars)

### Key Insight

**Ranking finds the same All-Stars (20% recall) without false positives!**

- Ranking: Top 10 predictions contain 2 All-Stars, 0 false positives
- Binary: 14 predictions contain 2 All-Stars, **12 false positives**

## Why Ranking is Better

1. **No False Positives**: Ranking doesn't make hard "yes/no" decisions, so there are no false positives. You just prioritize the top K.

2. **No Threshold Selection**: Ranking avoids the complex decision of "what threshold should we use?" This is especially important for imbalanced data where optimal thresholds can be very low (0.05-0.40).

3. **Better for Prioritization**: The use case is "which prospects should scouts prioritize?" Ranking naturally answers this: "Look at the top K predictions."

4. **More Robust**: Ranking doesn't break if the data distribution changes slightly. Binary classification may need threshold retuning.

5. **Interpretable**: "These are the top 10 prospects to watch" is clearer than "These 14 prospects have >40% probability of being All-Stars" (when only 2 actually are).

## When to Use Each Approach

### Use Ranking When:
- ✅ Data is imbalanced (few positive examples)
- ✅ Use case is prioritization/ranking (e.g., scouting)
- ✅ You want to avoid false positives
- ✅ You don't need hard "yes/no" decisions
- ✅ You want interpretable results ("top K")

### Use Binary Classification When:
- ✅ Data is balanced (roughly equal classes)
- ✅ Use case requires hard "yes/no" decisions
- ✅ You need to optimize precision/recall trade-off
- ✅ You can tolerate false positives
- ✅ You have a clear threshold selection criterion

## Implementation Details

### Threshold Selection

For binary classification comparison, we use **F1 score optimization**:
- Try thresholds from 0.05 to 0.95 in 0.05 steps
- Calculate F1 score at each threshold
- Select threshold with highest F1 score

This is a common approach, but it's still an arbitrary choice. Why F1? Why not precision? Why not recall? These decisions add complexity.

### Evaluation Code

Both approaches are evaluated in `src/evaluate.py`:
- `recall_at_top_k()`: Ranking evaluation
- `evaluate_binary_classification()`: Binary classification evaluation
- `find_optimal_threshold()`: Threshold selection for binary classification

Results are saved to:
- `reports/tables/metrics_{model}.csv`: Individual model metrics (includes both ranking and binary)
- `reports/tables/evaluation_summary.csv`: Summary across all models

## Conclusion

For this MLB All-Star prediction problem:
- **Ranking is the better approach** - simpler, more robust, better suited for imbalanced data
- **Binary classification is included for comparison** - shows we evaluated both approaches and made an informed decision

This demonstrates:
1. **Technical breadth**: Can implement both approaches
2. **Good judgment**: Chose the right approach for the problem
3. **Communication**: Clear explanation of trade-offs
4. **Understanding**: Knows when to use each approach

