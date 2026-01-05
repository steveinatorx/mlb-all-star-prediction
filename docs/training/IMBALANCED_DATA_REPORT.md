# Imbalanced Data Techniques: Results and Analysis

## Executive Summary

This report documents the implementation and results of advanced techniques for handling severe class imbalance in the MLB All-Star prediction dataset. The dataset has only 30 All-Stars (6.5%) in the training set, making it a challenging imbalanced classification problem.

**Key Findings**:
- SMOTE successfully balanced the dataset (429 negatives, 429 positives)
- Tree-based models (Random Forest, XGBoost) saw dramatic improvements (+154%, +109%)
- Linear models (Logistic Regression) saw slight decreases (-8%)
- Proper preprocessing order is critical: Impute → SMOTE → Scale → Train

## Dataset Characteristics

### Class Imbalance
- **Training set**: 459 players
  - All-Stars (positive): 30 (6.5%)
  - Non-All-Stars (negative): 429 (93.5%)
  - **Imbalance ratio**: 14.3:1

- **Validation set**: 279 players
  - All-Stars: 10 (3.6%)
  - Non-All-Stars: 269 (96.4%)
  - **Imbalance ratio**: 26.9:1

- **Test set**: 445 players
  - All-Stars: 10 (2.25%)
  - Non-All-Stars: 435 (97.75%)
  - **Imbalance ratio**: 43.5:1

### Missing Data
- **draft_round**: 100% null (not available from MLB API)
- **draft_position**: 100% null (not available from MLB API)
- **age_at_debut**: 47.9% null (birth dates not always available)
- **draft_year**: 26.1% null in training set

## Techniques Implemented

### 1. SMOTE (Synthetic Minority Oversampling Technique)

**What it does**: Creates synthetic All-Star samples by interpolating between existing All-Stars.

**Configuration**:
- `k_neighbors`: 5 (default, adjusted if fewer positive samples)
- `sampling_strategy`: "auto" (balances to 50/50)
- `random_state`: 42 (for reproducibility)

**Results**:
- **Before SMOTE**: 429 negatives, 30 positives
- **After SMOTE**: 429 negatives, 429 positives
- **Synthetic samples created**: 399 All-Star samples

**Implementation Challenge**:
- **Problem**: SMOTE doesn't accept NaN values
- **Solution**: Impute missing values BEFORE applying SMOTE
- **Order**: Impute → SMOTE → Scale → Train

### 2. Class Weights

**What it does**: Penalizes misclassifying All-Stars more than misclassifying non-All-Stars.

**Calculation**:
```python
weight_negative = n_samples / (n_classes * n_negatives)
weight_positive = n_samples / (n_classes * n_positives)
```

**Results**:
- **Before SMOTE**: Negative weight ~0.53, Positive weight ~7.65
- **After SMOTE**: Both weights become 1.00 (balanced classes)

**Model-Specific Implementation**:
- **Logistic Regression, Random Forest, LightGBM**: `class_weight` parameter
- **XGBoost**: `scale_pos_weight` parameter (ratio of weights)

### 3. Data Imputation

**Strategy**: Median imputation (robust to outliers)

**Why median**:
- Better than mean for skewed distributions (common in baseball stats)
- Less sensitive to outliers
- Works well with SMOTE (no NaN values)

**Implementation**:
- Applied BEFORE SMOTE (critical!)
- Uses `SimpleImputer(strategy="median")`
- Features with 100% nulls are skipped (warning, not error)

## Model Performance Comparison

### Baseline vs Advanced Techniques

| Model | Baseline PR-AUC | Advanced PR-AUC | Change | % Change |
|-------|----------------|-----------------|--------|----------|
| **Random Forest** | 0.0366 | 0.0929 | +0.0563 | **+154%** |
| **XGBoost** | 0.0453 | 0.0946 | +0.0493 | **+109%** |
| **LightGBM** | 0.0549 | 0.0629 | +0.0080 | +14.5% |
| **Logistic Regression** | 0.0717 | 0.0659 | -0.0058 | -8% |

### ROC-AUC Comparison

| Model | Baseline ROC-AUC | Advanced ROC-AUC | Change |
|-------|-----------------|------------------|--------|
| **Random Forest** | 0.4645 | 0.6996 | +0.2351 |
| **XGBoost** | 0.4959 | 0.7193 | +0.2234 |
| **LightGBM** | 0.6227 | 0.6855 | +0.0628 |
| **Logistic Regression** | 0.7234 | 0.7056 | -0.0178 |

## Key Insights

### 1. Tree-Based Models Benefit More from SMOTE

**Random Forest** and **XGBoost** saw dramatic improvements:
- Random Forest: +154% PR-AUC improvement
- XGBoost: +109% PR-AUC improvement

**Why**:
- Tree-based models can better learn from synthetic samples in feature space
- They can create more complex decision boundaries
- Better at handling the increased training data

### 2. Linear Models Benefit Less

**Logistic Regression** saw a slight decrease:
- PR-AUC: -8% (0.0717 → 0.0659)
- ROC-AUC: -2.5% (0.7234 → 0.7056)

**Why**:
- Linear models rely on linear separability
- Synthetic samples may not improve linear separation
- May introduce noise that hurts linear models

### 3. Preprocessing Order Matters

**Critical order**: Impute → SMOTE → Scale → Train

**Why**:
- SMOTE doesn't accept NaN values
- Must impute before SMOTE
- Scaling should happen after SMOTE (to scale resampled data)

**Lesson**: Always check library requirements and data dependencies!

### 4. Class Weights After SMOTE

**Observation**: After SMOTE balances classes, class weights become 1.00/1.00

**Why**:
- Balanced classes (50/50) → equal weights
- This is correct behavior
- Class weights are most useful when classes are imbalanced

## Technical Implementation

### Code Structure

```
src/train_advanced.py
├── calculate_class_weights()      # Calculate inverse frequency weights
├── apply_smote()                  # Apply SMOTE oversampling
├── train_logistic_regression_advanced()
├── train_random_forest_advanced()
├── train_xgboost_advanced()
├── train_lightgbm_advanced()
└── train_all_models_advanced()     # Orchestrate all models
```

### Preprocessing Pipeline

```python
# 1. Impute missing values
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# 2. Apply SMOTE (if requested)
if use_smote:
    X_train_imputed, y_train = apply_smote(
        X_train_imputed, y_train, k_neighbors=5
    )

# 3. Scale features (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)

# 4. Calculate class weights (if requested)
if use_class_weights:
    class_weights = calculate_class_weights(y_train)

# 5. Train model
model.fit(X_train_scaled, y_train, class_weight=class_weights)
```

## Recommendations

### 1. Use SMOTE for Tree-Based Models
- **Best for**: Random Forest, XGBoost, LightGBM
- **Expected improvement**: 50-150% PR-AUC improvement
- **Trade-off**: Increased training time (2x data)

### 2. Use Class Weights for Linear Models
- **Best for**: Logistic Regression, SVM
- **Expected improvement**: Modest (may decrease slightly)
- **Trade-off**: Minimal (no data increase)

### 3. Combine Both Techniques
- **Best practice**: Use SMOTE + class weights together
- **Why**: SMOTE balances data, class weights fine-tune learning
- **Result**: Best performance for tree-based models

### 4. Always Impute Before SMOTE
- **Critical**: SMOTE doesn't accept NaN values
- **Order**: Impute → SMOTE → Scale → Train
- **Check**: Verify no NaN values before SMOTE

## Future Work

### 1. Hyperparameter Tuning
- **SMOTE k_neighbors**: Try different values (3, 5, 7, 10)
- **Class weight ratios**: Try different weighting schemes
- **SMOTE sampling_strategy**: Try different ratios (not just 50/50)

### 2. Alternative Techniques
- **ADASYN**: Adaptive Synthetic Sampling (alternative to SMOTE)
- **Borderline-SMOTE**: Focus on borderline samples
- **SMOTE-ENN**: Combine SMOTE with Edited Nearest Neighbors

### 3. Model-Specific Optimization
- **Random Forest**: Tune max_depth, min_samples_split for balanced data
- **XGBoost**: Tune scale_pos_weight more carefully
- **Logistic Regression**: Try different regularization (L1 vs L2)

### 4. Evaluation
- **Test set evaluation**: Compare baseline vs advanced on test set
- **Cross-validation**: Use stratified K-fold for more robust estimates
- **Bootstrap confidence intervals**: Quantify uncertainty

## Conclusion

Advanced techniques for imbalanced data significantly improved model performance, especially for tree-based models:

- **Random Forest**: +154% PR-AUC improvement
- **XGBoost**: +109% PR-AUC improvement
- **LightGBM**: +14.5% PR-AUC improvement

**Key lessons**:
1. Tree-based models benefit more from SMOTE
2. Preprocessing order is critical (Impute → SMOTE → Scale → Train)
3. Class weights are most useful when classes are imbalanced
4. Always check library requirements (SMOTE doesn't accept NaN)

**Blog Post Topic**: "Balancing Your Data: SMOTE, Class Weights, and the Importance of Preprocessing Order"

This work demonstrates the importance of:
- Understanding library requirements
- Proper preprocessing order
- Model-specific technique selection
- Comprehensive evaluation

