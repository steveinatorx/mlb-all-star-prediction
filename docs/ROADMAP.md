# Project Roadmap

## Current Status: v0.3.5+ ✅

**Completed**:
- ✅ Baseline model training (5 models)
- ✅ Advanced imbalanced data techniques (SMOTE + class weights)
- ✅ Feature engineering pipeline
- ✅ Data enrichment (birth dates, draft years)
- ✅ Evaluation metrics (PR-AUC, ROC-AUC, Recall@TopK)
- ✅ Ranking vs Binary Classification comparison

**Performance**:
- Random Forest: PR-AUC 0.0929 (+154% improvement)
- XGBoost: PR-AUC 0.0946 (+109% improvement)
- LightGBM: PR-AUC 0.0629 (+14.5% improvement)

---

## Phase 1: Baseline Models ✅ COMPLETE

**Status**: ✅ Done

**What was done**:
- Trained 5 baseline models (Logistic Regression, Random Forest, XGBoost, LightGBM, GAM)
- Established baseline performance metrics
- Created training pipeline with `make train`

**Results**:
- Baseline PR-AUC: 0.036-0.072
- Baseline ROC-AUC: 0.46-0.72

---

## Phase 2: Advanced Techniques ✅ COMPLETE

**Status**: ✅ Done

**What was done**:
- Implemented SMOTE oversampling (399 synthetic samples)
- Implemented class weights (inverse frequency)
- Fixed preprocessing order (Impute → SMOTE → Scale → Train)
- Created `make train-advanced` command

**Results**:
- Random Forest: +154% PR-AUC improvement
- XGBoost: +109% PR-AUC improvement
- LightGBM: +14.5% PR-AUC improvement

---

## Phase 3: Evaluation & Testing ✅ COMPLETE

**Status**: ✅ Complete

### 3.1 Test Set Evaluation ✅ COMPLETE
**Goal**: Evaluate baseline vs advanced models on test set

**Tasks**:
- [x] Run `make eval` on baseline models
- [x] Run `make eval` on advanced models
- [x] Compare test set performance
- [x] Generate evaluation report
- [x] Add ranking vs binary classification comparison

**Expected Output**:
- Test set metrics (PR-AUC, ROC-AUC, Recall@TopK)
- Performance comparison (baseline vs advanced)
- Evaluation plots (PR curves, ROC curves)
- Binary classification metrics for comparison

**Results**:
- Baseline models evaluated on test set
- Advanced models evaluated on test set
- Ranking vs binary classification comparison implemented
- Documentation created: `docs/RANKING_VS_BINARY_CLASSIFICATION.md`

**Commands**:
```bash
# Evaluate all models (baseline + advanced)
make eval
```

### 3.2 Comprehensive Evaluation Metrics ✅ COMPLETE
**Goal**: Add more evaluation metrics beyond PR-AUC/ROC-AUC

**Tasks**:
- [x] Add Recall@TopK metrics (Top 10, 25, 50, 100) ✅
- [x] Add binary classification metrics for comparison ✅
- [x] Add learning curves (overfitting/underfitting analysis) ✅
- [x] Add calibration curves (probability calibration) ✅
- [x] Add bootstrap confidence intervals (uncertainty quantification) ✅
- [x] Add precision-recall curves with confidence intervals ✅
- [x] Add ROC curves with confidence intervals ✅

**Implementation**: Enhanced `src/evaluate.py`

**Completed Output**:
- `reports/tables/metrics_{model}.csv` (includes ranking + binary metrics)
- `reports/tables/evaluation_summary.csv` (comparison across models)
- `docs/RANKING_VS_BINARY_CLASSIFICATION.md` (comprehensive comparison)

**Completed Output**:
- `reports/figures/learning_curves_{model}.png`
- `reports/figures/calibration_curve_{model}.png`
- `reports/tables/bootstrap_ci_{model}.json`
- `reports/figures/pr_curve_ci_{model}.png`
- `reports/figures/roc_curve_ci_{model}.png`

---

## Phase 4: Hyperparameter Tuning ✅ COMPLETE

**Status**: ✅ Done

**Goal**: Optimize model performance through hyperparameter tuning

### 4.1 Bayesian Optimization (Optuna)
**Models**: XGBoost, LightGBM, Random Forest

**Hyperparameters to Tune**:
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`
- **LightGBM**: `n_estimators`, `max_depth`, `learning_rate`, `class_weight`
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `class_weight`

**Implementation**: Created `src/tune.py` module ✅

**Expected Output**:
- Best hyperparameters for each model
- Tuned model performance (expected +10-20% improvement)
- Hyperparameter importance plots

**Commands**:
```bash
# Tune models
pipenv run python -m src.main tune

# Train with tuned hyperparameters
pipenv run python -m src.main train --use-tuned-hyperparameters
```

### 4.2 Grid Search (Logistic Regression)
**Hyperparameters**: `C` (regularization strength), `penalty` (L1 vs L2)

**Implementation**: Add to `src/tune.py`

### 4.3 SMOTE Hyperparameter Tuning
**Hyperparameters**: `k_neighbors`, `sampling_strategy`

**Goal**: Optimize SMOTE parameters for best performance

---

## Phase 5: Model Interpretability ✅ COMPLETE

**Status**: ✅ Done

**Goal**: Extract insights and demonstrate business value

### 5.1 SHAP Values ✅ COMPLETE
**Models**: Random Forest, XGBoost, LightGBM

**Tasks**:
- [x] Generate SHAP values for tree-based models ✅
- [x] Create SHAP summary plots ✅
- [x] Create SHAP waterfall plots (individual predictions) ✅
- [x] Create SHAP dependence plots (feature interactions) ✅

**Implementation**: Enhance `src/evaluate.py` with SHAP integration

**Expected Output**:
- `reports/figures/shap_summary.png`
- `reports/figures/shap_waterfall_*.png` (for top predictions)
- `reports/figures/shap_dependence_*.png`

### 5.2 Feature Importance ✅ COMPLETE
**Tasks**:
- [x] Permutation importance (all models) ✅
- [x] Tree-based feature importance (tree models) ✅
- [x] Coefficient analysis (Logistic Regression) ✅

**Expected Output**:
- `reports/figures/feature_importance.png`
- `reports/tables/feature_importance.csv`

### 5.3 Partial Dependence Plots ✅ COMPLETE
**Tasks**:
- [x] Plot PDPs for top features ✅
- [x] Analyze feature interactions ✅

**Expected Output**:
- `reports/figures/pdp_*.png`

### 5.4 Counterfactual Analysis
**Tasks**:
- [ ] "What would make a player All-Star?"
- [ ] Analyze feature changes needed for predictions

---

## Phase 6: Experiment Tracking 📋 PLANNED

**Status**: 📋 Planned

**Goal**: Track experiments and compare results

### 6.1 MLflow Integration
**Decision**: Use MLflow (not Dagster) for experiment tracking

**Why MLflow**:
- Purpose-built for ML experiment tracking
- Lighter weight than Dagster (no database/webserver)
- Easy to use, minimal setup
- Better fit for ML iteration

**Tasks**:
- [ ] Install MLflow: `pipenv install mlflow`
- [ ] Wrap training functions with MLflow tracking
- [ ] Track experiments, metrics, parameters, models
- [ ] Set up MLflow UI: `mlflow ui`

**Implementation**: Create `src/mlflow_tracking.py` module

**Expected Output**:
- MLflow experiment runs
- Model registry
- Experiment comparison UI

**Commands**:
```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

### 6.2 Experiment Comparison
**Tasks**:
- [ ] Compare baseline vs advanced vs tuned models
- [ ] Track hyperparameter experiments
- [ ] Generate experiment comparison reports

---

## Phase 7: Ensemble Methods 📋 PLANNED

**Status**: 📋 Planned

**Goal**: Combine models for better performance

### 7.1 Voting Ensemble
**Tasks**:
- [ ] Combine top 3 models (Random Forest, XGBoost, LightGBM)
- [ ] Test hard voting vs soft voting
- [ ] Evaluate ensemble performance

### 7.2 Stacking
**Tasks**:
- [ ] Stack models with meta-learner
- [ ] Use Logistic Regression as meta-learner
- [ ] Evaluate stacked model performance

### 7.3 Blending
**Tasks**:
- [ ] Weighted average of model predictions
- [ ] Optimize weights for best performance

---

## Phase 8: Data Expansion 📋 PLANNED (Optional)

**Status**: 📋 Optional (if needed)

**Goal**: Expand dataset to improve model performance

### 8.1 Quick Wins (1-2 days)
- [ ] Add 2024 debuts (if data available)
- [ ] Fill missing `age_at_debut` (get birth dates)
- [ ] Add organization features (map teams to organizations)

**Expected Gain**: +5-10 All-Stars, +1-2 features

### 8.2 Medium Effort (3-5 days)
- [ ] Check 2000-2004 debuts for available minor league data
- [ ] Investigate missing players (why 886 have labels but no features)
- [ ] Add more draft information (draft round, position)

**Expected Gain**: +20-30 All-Stars, +3-4 features

### 8.3 High Effort (1-2 weeks)
- [ ] Unblock Baseball Reference (proxies, Selenium, API access)
- [ ] Explore Retrosheet for historical data
- [ ] Add Statcast data (velocity, spin rate) for recent years

**Expected Gain**: +50-100 All-Stars, +5-10 features

---

## Phase 9: Production Readiness 📋 PLANNED

**Status**: 📋 Planned

**Goal**: Make code production-ready

### 9.1 Code Quality
- [ ] Comprehensive unit tests
- [ ] Integration tests
- [ ] Code coverage > 80%
- [ ] Type hints throughout
- [ ] Documentation (docstrings, README)

### 9.2 Model Deployment
- [ ] Model serving API (FastAPI/Flask)
- [ ] Model versioning (MLflow model registry)
- [ ] Prediction endpoint
- [ ] Batch prediction script

### 9.3 Monitoring
- [ ] Model performance monitoring
- [ ] Data drift detection
- [ ] Prediction logging

---

## Immediate Next Steps (This Week)

### Priority 1: Model Interpretability 🔥 NEXT
1. **Add SHAP values**
   - Tree-based models (Random Forest, XGBoost, LightGBM)
   - Generate summary plots
   - Analyze feature importance
   - Fix current SHAP issues (feature mismatch errors)

2. **Feature importance analysis**
   - Which stats matter most?
   - Permutation importance (all models)
   - Coefficient analysis (Logistic Regression)
   - Compare feature importance across models

3. **Partial Dependence Plots**
   - Plot PDPs for top features
   - Analyze feature interactions
   - Show how features affect predictions

### Priority 2: Additional Evaluation Metrics
1. **Add learning curves**
   - Show overfitting/underfitting
   - Analyze training vs validation performance
   - Identify optimal stopping points

2. **Add calibration curves**
   - Check probability calibration
   - Are predicted probabilities well-calibrated?
   - Compare calibration across models

3. **Add bootstrap confidence intervals**
   - Quantify uncertainty in metrics
   - PR-AUC confidence intervals
   - ROC-AUC confidence intervals

### Priority 3: Hyperparameter Tuning (After Interpretability)
1. **Bayesian Optimization (Optuna)**
   - Tune XGBoost, LightGBM, Random Forest
   - Optimize for PR-AUC
   - Expected +10-20% improvement

2. **SMOTE Hyperparameter Tuning**
   - Optimize `k_neighbors` and `sampling_strategy`
   - Find best SMOTE configuration

---

## Decision Points

### When to Add MLflow?
**Recommendation**: Add MLflow when starting hyperparameter tuning (Phase 4)

**Why**:
- Need experiment tracking for multiple tuning runs
- MLflow makes comparison easier
- Not needed for current simple experiments

### When to Add Hyperparameter Tuning?
**Recommendation**: After comprehensive evaluation (Phase 3)

**Why**:
- Need to understand current performance first
- Tuning without evaluation is blind optimization
- Evaluation reveals what needs tuning

### When to Expand Data?
**Recommendation**: Only if model performance is insufficient

**Why**:
- Current performance is reasonable for dataset size
- More data = more time investment
- Focus on techniques first, data second

---

## Success Metrics

### Current Performance (v0.3.5)
- **Random Forest**: PR-AUC 0.0929, ROC-AUC 0.6996
- **XGBoost**: PR-AUC 0.0946, ROC-AUC 0.7193
- **LightGBM**: PR-AUC 0.0629, ROC-AUC 0.6855

### Target Performance (After Phase 4)
- **PR-AUC**: > 0.15 (2x improvement from baseline)
- **ROC-AUC**: > 0.75 (consistent across models)
- **Recall@Top25**: > 0.50 (find 50% of All-Stars in top 25)

### Stretch Goals
- **PR-AUC**: > 0.20 (3x improvement from baseline)
- **ROC-AUC**: > 0.80 (excellent discrimination)
- **Recall@Top25**: > 0.70 (find 70% of All-Stars in top 25)

---

## Timeline Estimate

### This Week
- ✅ Phase 1: Baseline models (DONE)
- ✅ Phase 2: Advanced techniques (DONE)
- ✅ Phase 3.1: Test set evaluation (DONE)
- ✅ Phase 3.2: Ranking vs Binary comparison (DONE)

### Next Week
- Phase 5.1: SHAP values (interpretability) - Fix current issues
- Phase 5.2: Feature importance analysis
- Phase 3.2: Additional evaluation metrics (learning curves, calibration)

### Week 3-4
- Phase 4: Hyperparameter tuning
- Phase 6: MLflow integration

### Week 5+
- Phase 7: Ensemble methods
- Phase 9: Production readiness

---

## Key Files

### Current Implementation
- `src/train.py`: Baseline training
- `src/train_advanced.py`: Advanced training (SMOTE + class weights)
- `src/evaluate.py`: Evaluation (needs enhancement)
- `src/main.py`: CLI entrypoint

### Planned Implementation
- `src/tune.py`: Hyperparameter tuning (Optuna)
- `src/mlflow_tracking.py`: MLflow experiment tracking
- `src/interpret.py`: Model interpretability (SHAP, PDPs)
- `src/ensemble.py`: Ensemble methods

---

## Questions to Answer

1. **Is current performance sufficient?**
   - If yes → Focus on interpretability and production
   - If no → Focus on hyperparameter tuning and data expansion

2. **What's the priority: performance or interpretability?**
   - Performance → Hyperparameter tuning, ensemble methods
   - Interpretability → SHAP values, feature importance, PDPs

3. **Do we need more data?**
   - Current: 30 All-Stars in training
   - Target: 50-100 All-Stars for robust models
   - Decision: Expand data if performance plateaus

---

## Summary

**Completed**: ✅ Baseline models, ✅ Advanced techniques

**Next**: 🔄 Test set evaluation, 📋 Comprehensive metrics, 📋 Interpretability

**Future**: 📋 Hyperparameter tuning, 📋 MLflow, 📋 Ensemble methods

**Focus**: Demonstrate ML expertise through technique, not just performance

