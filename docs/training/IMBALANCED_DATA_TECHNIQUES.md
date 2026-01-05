# Advanced Techniques for Imbalanced Data

## Overview

This document describes the advanced techniques implemented to handle the severe class imbalance in our dataset (6.5% All-Stars, 93.5% non-All-Stars).

## Techniques Implemented

### 1. Class Weights

**What it does**: Penalizes misclassifying All-Stars more than misclassifying non-All-Stars.

**How it works**:
- Calculates inverse frequency weights: `weight = n_samples / (n_classes * n_class_samples)`
- For our dataset: Negative class weight ≈ 0.5, Positive class weight ≈ 7.1
- Models learn to prioritize correctly identifying All-Stars

**Implementation**:
```python
class_weights = {
    0: n_samples / (n_classes * n_negatives),  # ~0.5
    1: n_samples / (n_classes * n_positives),   # ~7.1
}
```

**Supported models**:
- Logistic Regression (`class_weight` parameter)
- Random Forest (`class_weight` parameter)
- XGBoost (`scale_pos_weight` parameter)
- LightGBM (`class_weight` parameter)

### 2. SMOTE (Synthetic Minority Oversampling Technique)

**What it does**: Creates synthetic All-Star samples by interpolating between existing All-Stars.

**How it works**:
1. For each All-Star sample, finds k nearest neighbors (also All-Stars)
2. Creates new synthetic samples along the line segments between the sample and its neighbors
3. Balances the dataset to 50/50 (or specified ratio)

**Benefits**:
- Increases training data for minority class
- Helps models learn All-Star patterns
- Reduces overfitting to the few positive examples

**Configuration**:
- `k_neighbors`: Number of nearest neighbors (default: 5)
- `sampling_strategy`: Target ratio (default: "auto" = 50/50)

**Example**:
```
Before SMOTE: 429 negatives, 30 positives
After SMOTE:  429 negatives, 429 positives (399 synthetic All-Stars created)
```

**Limitations**:
- Requires at least k+1 positive samples (we have 30, so k ≤ 29)
- May create unrealistic samples if feature space is sparse
- Can increase training time

### 3. Combined Approach

**Best practice**: Use both class weights AND SMOTE together.

**Rationale**:
- SMOTE: Provides more training data for All-Stars
- Class weights: Ensures models prioritize All-Stars even after SMOTE

## Usage

### Command Line

**Baseline training** (no advanced techniques):
```bash
make train
# or
pipenv run python -m src.main train
```

**Advanced training** (with SMOTE and class weights):
```bash
make train-advanced
# or
pipenv run python -m src.main train-advanced
```

**Custom options**:
```bash
# Only class weights (no SMOTE)
pipenv run python -m src.main train-advanced --use-class-weights --no-smote

# Only SMOTE (no class weights)
pipenv run python -m src.main train-advanced --no-class-weights --use-smote

# Both (default)
pipenv run python -m src.main train-advanced --use-class-weights --use-smote
```

### Python API

```python
from src.train_advanced import train_all_models_advanced

# Train with both techniques
results = train_all_models_advanced(
    features_path="data/features/features.parquet",
    output_dir="experiments/advanced",
    use_class_weights=True,
    use_smote=True,
)
```

## Model Outputs

### Baseline Models
- Saved to: `experiments/`
- Files: `logistic_regression.joblib`, `random_forest.joblib`, etc.
- Results: `experiments/training_results.json`

### Advanced Models
- Saved to: `experiments/advanced/`
- Files: `logistic_regression_advanced.joblib`, `random_forest_advanced.joblib`, etc.
- Results: `experiments/advanced/training_results_advanced.json`

### Model Metadata

Advanced models include technique metadata:
```json
{
  "techniques": {
    "class_weights": true,
    "smote": true
  }
}
```

## Expected Improvements

### Baseline Performance (from training)
- Logistic Regression: PR-AUC=0.0717, ROC-AUC=0.7234
- Random Forest: PR-AUC=0.0366, ROC-AUC=0.4645
- XGBoost: PR-AUC=0.0453, ROC-AUC=0.4959
- LightGBM: PR-AUC=0.0549, ROC-AUC=0.6227

### Expected with Advanced Techniques

**Class Weights Only**:
- Should improve PR-AUC (better at finding All-Stars)
- May slightly decrease ROC-AUC (more false positives)
- Better precision at high recall

**SMOTE Only**:
- Should improve both PR-AUC and ROC-AUC
- More training data helps models learn patterns
- Risk of overfitting to synthetic samples

**Both Combined**:
- Best of both worlds
- Expected PR-AUC improvement: 0.08-0.15 (vs 0.07 baseline)
- Expected ROC-AUC: Similar or slightly better

## Comparison

To compare baseline vs advanced:

```python
import json

# Load baseline results
with open("experiments/training_results.json") as f:
    baseline = json.load(f)

# Load advanced results
with open("experiments/advanced/training_results_advanced.json") as f:
    advanced = json.load(f)

# Compare
for model_name in baseline:
    baseline_pr = baseline[model_name]["metrics"]["pr_auc"]
    advanced_name = f"{model_name}_advanced"
    if advanced_name in advanced:
        advanced_pr = advanced[advanced_name]["metrics"]["pr_auc"]
        improvement = advanced_pr - baseline_pr
        print(f"{model_name}: {baseline_pr:.4f} → {advanced_pr:.4f} ({improvement:+.4f})")
```

## Technical Details

### Class Weight Calculation

```python
def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    n_positives = y_train.sum()
    n_negatives = n_samples - n_positives

    weight_negative = n_samples / (n_classes * n_negatives)
    weight_positive = n_samples / (n_classes * n_positives)

    return {0: weight_negative, 1: weight_positive}
```

### SMOTE Application

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    k_neighbors=5,
    random_state=42,
    sampling_strategy="auto",  # Balance to 50/50
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Model-Specific Implementations

**Logistic Regression**:
```python
model = LogisticRegression(
    class_weight=class_weights_dict,
    ...
)
```

**XGBoost**:
```python
# XGBoost uses scale_pos_weight instead
scale_pos_weight = class_weights_dict[1] / class_weights_dict[0]
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    ...
)
```

## Limitations and Considerations

### SMOTE Limitations

1. **Small positive class**: With only 30 All-Stars, SMOTE can only create so many synthetic samples
2. **Feature space**: If All-Stars are very different from each other, synthetic samples may be unrealistic
3. **Training time**: SMOTE increases training data size, slowing training

### Class Weight Limitations

1. **Over-penalization**: Too high weights can cause overfitting to All-Stars
2. **False positives**: May increase false positive rate
3. **Calibration**: May affect probability calibration

### Best Practices

1. **Always compare**: Run both baseline and advanced, compare results
2. **Cross-validation**: Use stratified K-fold to ensure All-Stars in each fold
3. **Monitor overfitting**: Check if advanced models overfit to training data
4. **Threshold tuning**: After training, tune threshold based on precision/recall trade-off

## Next Steps

1. **Hyperparameter tuning**: Optimize SMOTE k_neighbors, class weight ratios
2. **ADASYN**: Try ADASYN (Adaptive Synthetic Sampling) as alternative to SMOTE
3. **Ensemble methods**: Combine baseline and advanced models
4. **Threshold optimization**: Find optimal threshold for precision/recall trade-off
5. **Evaluation**: Compare baseline vs advanced on test set

## References

- [SMOTE Paper](https://arxiv.org/abs/1106.1813): Chawla et al., 2002
- [Class Weights in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)

