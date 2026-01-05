# Evaluation Metrics: Why PR-AUC and ROC-AUC Instead of F1 Score

## The Problem with F1 Score for Imbalanced Data

### Our Dataset Characteristics
- **Positive class (All-Stars)**: 30 in training (6.5%), 10 in validation (3.6%)
- **Negative class (non-All-Stars)**: 429 in training (93.5%), 269 in validation (96.4%)
- **Class imbalance ratio**: ~15:1 (non-All-Stars to All-Stars)

### Why F1 Score Fails Here

**F1 Score Formula**: `F1 = 2 * (precision * recall) / (precision + recall)`

**Problem 1: Threshold-Dependent**
- F1 score requires choosing a classification threshold (typically 0.5)
- With severe class imbalance, threshold=0.5 is almost always wrong
- A naive model that predicts "no All-Star" for everyone gets:
  - Precision: undefined (0/0)
  - Recall: 0
  - F1: 0
- But a model that predicts "All-Star" for top 10% gets:
  - Precision: ~0.065 (if 6.5% are actually All-Stars)
  - Recall: ~1.0 (if we predict all All-Stars)
  - F1: ~0.12
- **Issue**: F1 doesn't tell us if 0.12 is good or bad without context

**Problem 2: Doesn't Account for Probability Calibration**
- F1 treats predictions as binary (0 or 1)
- Doesn't use the probability scores that models output
- We lose information about model confidence
- **Example**: Two models might have same F1 but very different probability distributions

**Problem 3: Misleading for Imbalanced Data**
- With 93.5% negative class, a model can achieve high F1 by:
  - Predicting "no All-Star" for most players (high precision)
  - Missing most actual All-Stars (low recall)
- **Example**: 
  - Model predicts 5 All-Stars, 3 are correct → Precision=0.6, Recall=0.1, F1=0.17
  - This seems "good" but we're missing 90% of All-Stars!

**Problem 4: Not Optimized for Ranking**
- F1 doesn't measure how well we rank players
- For scouting, we care about: "Are All-Stars in the top predictions?"
- F1 doesn't answer: "If we evaluate top 25 players, how many All-Stars do we find?"

## Why PR-AUC (Precision-Recall AUC) is Better

### What PR-AUC Measures
- **Area Under the Precision-Recall Curve**
- Plots precision vs recall at different probability thresholds
- Higher is better (range: 0 to 1)
- **Interpretation**: Average precision across all recall levels

### Advantages for Imbalanced Data

**1. Focuses on Positive Class**
- PR-AUC emphasizes the minority class (All-Stars)
- Less influenced by the large negative class
- **Example**: 
  - Model A: PR-AUC=0.07, finds 7% of All-Stars on average
  - Model B: PR-AUC=0.15, finds 15% of All-Stars on average
  - Clear that Model B is better, even if both have low absolute scores

**2. Threshold-Independent**
- Evaluates performance across all possible thresholds
- No need to choose a single threshold
- Shows model performance at different precision/recall trade-offs
- **Use case**: Scouts can choose threshold based on resources
  - High precision (fewer false positives) → evaluate top 10
  - High recall (find more All-Stars) → evaluate top 50

**3. Better for Small Positive Class**
- PR-AUC is more informative when positive class is rare
- ROC-AUC can be misleading (see below)
- **Our case**: 30 All-Stars in training → PR-AUC is more reliable

**4. Aligns with Scouting Use Case**
- Scouts have limited resources (can't evaluate everyone)
- Care about precision: "Of the players we evaluate, how many are All-Stars?"
- Care about recall: "How many All-Stars did we find?"
- PR-AUC captures both concerns

### PR-AUC Interpretation

**Our Results**:
- Logistic Regression: PR-AUC = 0.0717
- **Meaning**: At optimal threshold, model achieves ~7.2% average precision
- **Context**: Random baseline = 0.065 (6.5% positive rate)
- **Improvement**: Model is slightly better than random (0.0717 vs 0.065)

**What "Good" PR-AUC Looks Like**:
- **Random baseline**: PR-AUC ≈ positive class rate (0.065 for us)
- **Good model**: PR-AUC > 0.2-0.3 (3-5x better than random)
- **Excellent model**: PR-AUC > 0.5 (10x better than random)
- **Our models**: PR-AUC 0.036-0.072 (slightly better than random, expected with small dataset)

## Why ROC-AUC (Receiver Operating Characteristic AUC)

### What ROC-AUC Measures
- **Area Under the ROC Curve**
- Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at different thresholds
- Higher is better (range: 0 to 1)
- **Interpretation**: Probability that model ranks a random All-Star higher than a random non-All-Star

### Advantages

**1. Threshold-Independent**
- Like PR-AUC, evaluates across all thresholds
- No need to choose a single threshold

**2. Interpretable**
- ROC-AUC = 0.72 means: "72% chance model ranks All-Star higher than non-All-Star"
- Easy to understand and communicate

**3. Standard Metric**
- Widely used in ML literature
- Easy to compare with other models/datasets

### Limitations for Imbalanced Data

**1. Can Be Misleading**
- ROC-AUC can be high even when model performs poorly on minority class
- **Example**: 
  - Model predicts "All-Star" for top 20% of players
  - Gets 80% of All-Stars (high TPR)
  - Gets 20% false positives (moderate FPR)
  - ROC-AUC = 0.80 (seems good!)
  - But precision = 0.065/0.20 = 0.325 (only 32.5% of predictions are correct)
  - **Problem**: We're evaluating 5x more players than necessary

**2. Less Sensitive to Minority Class**
- With 93.5% negative class, ROC-AUC is dominated by negative class performance
- Small improvements in All-Star detection don't move ROC-AUC much
- **Our case**: ROC-AUC 0.46-0.72, but PR-AUC 0.036-0.072 (much lower)

**3. Doesn't Reflect Scouting Constraints**
- ROC-AUC doesn't account for resource limitations
- Scouts can't evaluate 20% of all players
- Need to focus on top predictions (precision matters more)

### Why We Include Both

**PR-AUC**: Primary metric (better for imbalanced data, aligns with use case)
**ROC-AUC**: Secondary metric (standard, interpretable, for comparison)

**Our Results**:
- Logistic Regression: PR-AUC=0.0717, ROC-AUC=0.7234
- **Interpretation**: 
  - PR-AUC: Model is slightly better than random (0.0717 vs 0.065)
  - ROC-AUC: Model ranks All-Stars reasonably well (72% chance All-Star ranked higher)
  - **Gap**: ROC-AUC seems "good" but PR-AUC shows reality (small improvement over random)

## Why Not F1 Score?

### Summary of F1 Score Problems

1. **Threshold-dependent**: Requires choosing a threshold (0.5 is usually wrong)
2. **Binary only**: Doesn't use probability scores
3. **Misleading for imbalance**: Can seem "good" while missing most All-Stars
4. **Not ranking-focused**: Doesn't measure "Are All-Stars in top predictions?"
5. **Context-dependent**: F1=0.12 doesn't tell us if that's good without baseline

### When F1 Score IS Appropriate

- **Balanced classes**: ~50/50 split
- **Clear threshold**: Single decision point (e.g., spam/not spam)
- **Binary decisions**: No need for ranking or probability scores
- **Sufficient data**: Large enough positive class to make F1 meaningful

### Our Use Case: F1 Score Doesn't Fit

- **Imbalanced**: 6.5% positive rate
- **Ranking task**: Scouts evaluate top N players
- **Probability scores**: Models output probabilities, not binary predictions
- **Small positive class**: 30 All-Stars in training (F1 would be unstable)

## Additional Metrics We Should Consider

### Recall@TopK (Scouting Perspective)

**What it measures**: How many All-Stars found in top K predictions

**Why it's useful**:
- Directly answers: "If we evaluate top 25 players, how many All-Stars do we find?"
- Aligns with scouting resource constraints
- More actionable than F1 or PR-AUC

**Example**:
- Recall@Top10: Found 2 All-Stars in top 10 predictions (20% recall)
- Recall@Top25: Found 5 All-Stars in top 25 predictions (50% recall)
- Recall@Top50: Found 8 All-Stars in top 50 predictions (80% recall)

**Implementation**: Should add this to evaluation!

### Calibration

**What it measures**: Are predicted probabilities well-calibrated?

**Why it's useful**:
- If model says "80% chance All-Star", is it actually 80%?
- Important for decision-making
- Can be measured with calibration curves

## Summary

| Metric | Best For | Why We Use It |
|--------|----------|---------------|
| **PR-AUC** | Imbalanced data, ranking tasks | ✅ Primary metric - focuses on All-Stars, threshold-independent |
| **ROC-AUC** | Standard comparison, interpretability | ✅ Secondary metric - widely used, easy to understand |
| **F1 Score** | Balanced data, binary decisions | ❌ Not suitable - threshold-dependent, misleading for imbalance |
| **Recall@TopK** | Scouting/ranking use cases | ⚠️ Should add - directly answers scouting question |

## Key Takeaway

**For imbalanced classification with small positive class:**
- ✅ **PR-AUC**: Best metric (focuses on minority class)
- ✅ **ROC-AUC**: Good secondary metric (standard, interpretable)
- ❌ **F1 Score**: Not suitable (threshold-dependent, misleading)
- ✅ **Recall@TopK**: Should add (directly answers scouting question)

**Our Results Context**:
- PR-AUC 0.036-0.072: Slightly better than random (0.065)
- ROC-AUC 0.46-0.72: Model ranks All-Stars reasonably well
- **Expected**: With only 30 All-Stars in training, models can't learn much signal
- **Next steps**: Add Recall@TopK, implement advanced techniques (SMOTE, class weights)

