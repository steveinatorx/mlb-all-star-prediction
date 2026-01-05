# SHAP Analysis: Model Interpretability

## Overview

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance by explaining the output of any machine learning model. This document explains what SHAP values are, how they're calculated, and what insights they reveal about our MLB All-Star prediction models.

## What are SHAP Values?

SHAP values are based on **cooperative game theory** (Shapley values). They answer the question: **"How much does each feature contribute to a specific prediction?"**

### Key Properties

1. **Additivity**: SHAP values sum to the difference between the model's prediction and the baseline (expected value)
   - `prediction = baseline + sum(SHAP values)`
   - For binary classification: `logit(prediction) = logit(baseline) + sum(SHAP values)`

2. **Efficiency**: The sum of SHAP values equals the difference between prediction and baseline

3. **Symmetry**: Features with equal marginal contributions have equal SHAP values

4. **Dummy**: Features that don't affect the prediction have SHAP value of 0

## How SHAP Values Work

### For Tree-Based Models (Random Forest, XGBoost, LightGBM)

We use `TreeExplainer`, which is optimized for tree models:

1. **Traverses the tree**: For each prediction, SHAP traces the path through the decision tree
2. **Calculates contributions**: At each split, calculates how much each feature contributes
3. **Aggregates**: Combines contributions across all trees in the ensemble

**Advantages**:
- Fast (exact calculation, not sampling)
- Handles feature interactions automatically
- Works with missing values

### Calculation Process

For a single prediction:
1. Start with baseline (average prediction across training data)
2. For each feature, calculate its marginal contribution:
   - What's the prediction with this feature?
   - What's the prediction without this feature?
   - Difference = feature's contribution
3. Account for feature interactions (order matters)
4. Average contributions across all possible feature orderings

## SHAP Plot Types

### 1. Summary Plot (What We Generate)

**What it shows**:
- **Y-axis**: Features ranked by importance (most important at top)
- **X-axis**: SHAP value (impact on prediction)
- **Color**: Feature value (red = high, blue = low)
- **Dots**: Individual predictions

**How to read**:
- Features at the top are most important
- Points to the right push prediction toward All-Star (positive SHAP)
- Points to the left push prediction away from All-Star (negative SHAP)
- Red dots (high feature values) on the right = high values increase All-Star probability
- Blue dots (low feature values) on the left = low values decrease All-Star probability

**Example interpretation**:
- If `career_k_per_9` is at the top with red dots on the right:
  - High strikeout rates increase All-Star probability
  - This feature is important for predictions

### 2. Waterfall Plot (Not Yet Implemented)

Shows how each feature moves the prediction from baseline to final prediction for a single instance.

### 3. Dependence Plot (Not Yet Implemented)

Shows how a feature's SHAP value changes as the feature value changes, revealing interactions.

## Implementation Details

### Code Location

`src/evaluate.py` → `plot_shap_importance()`

### Key Steps

1. **Load model and features**
   ```python
   model_data = joblib.load(model_path)
   model = model_data["model"]
   feature_names = model_data["feature_names"]
   ```

2. **Align features with model expectations**
   - Models may have fewer features than `feature_names` (if features were dropped during training)
   - We use `n_features_in_` to determine actual feature count
   - Truncate `feature_names` to match model's expectations

3. **Prepare data**
   - Load test split (or specified split)
   - Sample if too large (SHAP can be slow)
   - Encode categorical features (e.g., `highest_level_reached`)
   - Handle missing values (impute if needed)

4. **Create SHAP explainer**
   ```python
   explainer = shap.TreeExplainer(model, feature_perturbation="auto")
   shap_values = explainer.shap_values(X)
   ```

5. **Handle binary classification**
   - For binary classification, SHAP returns a list `[negative_class, positive_class]`
   - We use `shap_values[1]` (positive class = All-Star)

6. **Generate plot**
   ```python
   shap.summary_plot(shap_values, X, feature_names=feature_names)
   ```

### Feature Alignment Fix

**Problem**: Models trained with 17 features but `n_features_in_` was 15
- Some features were dropped during training (likely due to all-null values)
- `feature_names` still had all 17 features
- SHAP failed because feature count didn't match

**Solution**: 
- Check `n_features_in_` to determine actual feature count
- Use first `n_features_in_` features from `feature_names`
- Align data features with model expectations before SHAP calculation

## Interpreting SHAP Results

### Feature Importance Ranking

Features are ranked by mean absolute SHAP value:
- **Top features**: Most impact on predictions
- **Bottom features**: Least impact (may be noise)

### Direction of Impact

- **Positive SHAP values**: Increase All-Star probability
- **Negative SHAP values**: Decrease All-Star probability

### Feature Value Patterns

- **Red dots (high values) on right**: High feature values increase All-Star probability
- **Blue dots (low values) on left**: Low feature values decrease All-Star probability
- **Mixed patterns**: Feature has complex relationship (may indicate interactions)

### Example Interpretations

**Scenario 1: Clear positive relationship**
- Feature: `career_k_per_9`
- Pattern: Red dots clustered on right, blue dots on left
- Interpretation: High strikeout rates strongly predict All-Star status

**Scenario 2: Complex relationship**
- Feature: `age_at_debut`
- Pattern: Red and blue dots mixed across SHAP values
- Interpretation: Age has non-linear relationship (may be optimal age range)

**Scenario 3: Low importance**
- Feature: `total_milb_games`
- Pattern: SHAP values clustered near zero
- Interpretation: Number of games played doesn't strongly predict All-Star status

## Current SHAP Results

### Models Analyzed

1. **LightGBM** (`shap_lightgbm.png`)
2. **Random Forest** (`shap_random_forest.png`)
3. **XGBoost** (`shap_xgboost.png`)

### Files Generated

- `reports/figures/shap_lightgbm.png` (363KB)
- `reports/figures/shap_random_forest.png` (97KB)
- `reports/figures/shap_xgboost.png` (358KB)

### How to View Results

1. Open the PNG files in `reports/figures/`
2. Look at feature ranking (top = most important)
3. Examine SHAP value distribution (spread indicates impact)
4. Check color patterns (red/blue = feature value, position = impact direction)

## Limitations and Considerations

### 1. Computational Cost

- SHAP can be slow for large datasets
- We sample test data (max 100 samples by default)
- TreeExplainer is fast for tree models, but still takes time

### 2. Feature Interactions

- SHAP values show marginal contributions
- Complex interactions may not be fully captured
- Dependence plots can reveal interactions (not yet implemented)

### 3. Model-Specific

- TreeExplainer only works for tree-based models
- Logistic Regression uses coefficients (different interpretation)
- GAM doesn't have SHAP support (uses partial dependence plots)

### 4. Baseline Dependency

- SHAP values are relative to baseline (average prediction)
- Baseline depends on training data distribution
- Different baselines = different SHAP values

## Future Enhancements

### Planned Additions

1. **Waterfall Plots**: Show prediction breakdown for individual players
   - "Why did we predict Player X as All-Star?"
   - Feature-by-feature contribution

2. **Dependence Plots**: Reveal feature interactions
   - How does Feature A's impact change with Feature B?
   - Identify non-linear relationships

3. **Feature Interaction Values**: Quantify interaction strength
   - Which feature pairs interact most?
   - How do interactions affect predictions?

4. **Permutation Importance Comparison**: Compare SHAP vs permutation importance
   - SHAP: Marginal contribution
   - Permutation: Overall importance
   - Differences reveal interactions

## Key Takeaways

1. **SHAP provides feature importance**: Rank features by their impact on predictions

2. **SHAP shows direction**: Positive values increase All-Star probability, negative decrease it

3. **SHAP reveals patterns**: Color patterns show how feature values relate to predictions

4. **SHAP is model-agnostic**: Works for any model, but TreeExplainer is optimized for trees

5. **SHAP complements other methods**: Use with permutation importance, coefficients, etc.

## References

- **SHAP Paper**: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- **SHAP Documentation**: https://shap.readthedocs.io/
- **TreeExplainer**: Optimized for tree-based models (fast, exact)

## Next Steps

1. **Review generated SHAP plots**: Examine feature importance rankings
2. **Compare across models**: Do different models agree on important features?
3. **Identify key predictors**: Which minor league stats best predict All-Star status?
4. **Plan feature engineering**: Use insights to create better features
5. **Add waterfall/dependence plots**: Deeper analysis of individual predictions

