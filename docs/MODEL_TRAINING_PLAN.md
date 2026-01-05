# Model Training Plan

## Current Status

**Dataset**:
- 2,471 players with 20 features
- 50 All-Stars (2.02% positive rate)
- Train/Val/Test splits: Based on debut year (2005-2018 / 2019-2020 / 2021-2023)

**Current Implementation**:
- Simple `make train` command
- Trains 5 models sequentially (Logistic Regression, Random Forest, XGBoost, LightGBM, GAM)
- Saves models to `experiments/` directory
- Basic metrics: PR-AUC, ROC-AUC
- No experiment tracking, hyperparameter tuning, or advanced techniques

## Proposed Training Plan

### Phase 1: Baseline Models (Current) ✅
**Goal**: Establish baseline performance

**Models**:
1. Logistic Regression (L2 regularization)
2. Random Forest (default hyperparameters)
3. XGBoost (default hyperparameters)
4. LightGBM (if available)
5. GAM (if available)

**Metrics**:
- PR-AUC (primary)
- ROC-AUC (secondary)
- Basic evaluation plots

**Status**: ✅ Already implemented

### Phase 2: Advanced Techniques for Small/Imbalanced Data
**Goal**: Demonstrate expertise with imbalanced data techniques

**Techniques to Add**:
1. **Class Weights**: Penalize misclassifying All-Stars
   ```python
   class_weight = {0: 1.0, 1: 49.4}  # Inverse of imbalance ratio
   ```

2. **SMOTE/ADASYN**: Synthetic oversampling of minority class
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **Stratified K-Fold CV**: Ensure All-Stars in each fold
   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   ```

4. **Bootstrap Confidence Intervals**: Quantify uncertainty
   ```python
   from sklearn.utils import resample
   n_bootstraps = 1000
   bootstrap_scores = []
   ```

**Implementation**: Add to `src/train.py` with flags to enable/disable techniques

### Phase 3: Hyperparameter Tuning
**Goal**: Optimize model performance

**Approach**:
1. **Bayesian Optimization** (Optuna): For XGBoost, LightGBM
2. **Grid Search**: For Logistic Regression (regularization strength)
3. **Random Search**: For Random Forest (if time permits)

**Hyperparameters to Tune**:
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, `class_weight`
- **Logistic Regression**: `C` (regularization strength)
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`

**Implementation**: Create `src/tune.py` module

### Phase 4: Comprehensive Evaluation
**Goal**: Demonstrate thorough evaluation beyond basic metrics

**Additional Metrics**:
1. **Recall@TopK**: Top 10, 25, 50, 100 (scouting perspective)
2. **Learning Curves**: Show overfitting/underfitting
3. **Calibration Curves**: Probability calibration
4. **Precision-Recall Curves**: With confidence intervals
5. **ROC Curves**: With confidence intervals

**Implementation**: Enhance `src/evaluate.py`

### Phase 5: Model Interpretability
**Goal**: Extract insights and demonstrate business value

**Techniques**:
1. **SHAP Values**: Explain individual predictions (tree-based models)
2. **Feature Importance**: Which stats matter most?
3. **Partial Dependence Plots**: How features affect predictions
4. **Coefficient Analysis**: For Logistic Regression (p-values, confidence intervals)
5. **Counterfactual Analysis**: What would make a player All-Star?

**Implementation**: Enhance `src/evaluate.py` with interpretability module

## Should We Introduce Dagster Now?

### Arguments FOR Dagster ⭐⭐⭐

**1. Experiment Tracking**
- Track multiple experiments (baseline → SMOTE → tuned → ensemble)
- Compare results across experiments
- Reproducibility: Track which features → which model → which results
- **Value**: High - Critical for ML iteration

**2. Hyperparameter Tuning Orchestration**
- Run multiple tuning jobs in parallel
- Track which hyperparameters → which performance
- Automatically select best model
- **Value**: High - Makes tuning manageable

**3. Dependency Management**
- Training depends on features (which depend on processed data)
- Evaluation depends on training
- Reporting depends on evaluation
- Handle failures gracefully (retry training if features change)
- **Value**: Medium - Current Makefile handles this, but Dagster is more robust

**4. Model Versioning**
- Track model versions over time
- Compare model performance across versions
- Rollback to previous models if needed
- **Value**: Medium - Important for production, less critical for research

**5. Team Collaboration** (if applicable)
- Multiple people running experiments
- Share experiment results
- Avoid conflicts
- **Value**: Low - Solo project currently

### Arguments AGAINST Dagster ⭐

**1. Infrastructure Overhead**
- Need to set up Dagster (database, webserver)
- Learning curve for Dagster API
- Additional dependencies
- **Cost**: Medium - Time investment upfront

**2. Current Pipeline is Simple**
- Linear dependencies: ingest → build → featurize → train → evaluate
- Makefile handles dependencies adequately
- No complex branching or conditional logic
- **Cost**: Low - Current approach works fine

**3. Research/Prototyping Stage**
- Still experimenting with models and techniques
- Need flexibility to iterate quickly
- May change approach based on results
- **Cost**: Medium - Dagster adds structure that may slow iteration

**4. Small Dataset**
- Only 2,471 players, 50 All-Stars
- Training is fast (seconds, not hours)
- No need for distributed processing
- **Cost**: Low - Current approach is sufficient

### Recommendation: **Hybrid Approach** 🎯

**Phase 1-2: Keep Simple (Current)**
- Use Makefile for training pipeline
- Add experiment tracking manually (simple JSON/CSV)
- Focus on implementing advanced techniques
- **Rationale**: Fast iteration, no infrastructure overhead

**Phase 3-4: Introduce Dagster (When Needed)**
- When hyperparameter tuning becomes complex
- When running many experiments
- When need for reproducibility becomes critical
- **Rationale**: Dagster shines with multiple experiments and tuning

**Alternative: Use MLflow Instead** ⭐⭐⭐
- **Lighter weight**: No database/webserver needed (file-based)
- **Experiment tracking**: Track runs, metrics, parameters, artifacts
- **Model registry**: Version models
- **Less overhead**: Easier to set up than Dagster
- **Better fit**: Designed specifically for ML experiment tracking

**Recommendation**: **Start with MLflow, not Dagster**
- MLflow is purpose-built for ML experiment tracking
- Less infrastructure overhead than Dagster
- Can add Dagster later if needed for complex orchestration
- MLflow handles experiment tracking, Dagster handles orchestration (different use cases)

## Implementation Plan

### Option A: Keep Simple (Recommended for Now)
1. ✅ Keep current `make train` approach
2. Add experiment tracking with simple JSON/CSV
3. Implement advanced techniques (SMOTE, class weights)
4. Add hyperparameter tuning with Optuna
5. Enhance evaluation with more metrics

**Pros**: Fast iteration, no infrastructure overhead
**Cons**: Manual experiment tracking, less reproducibility

### Option B: Add MLflow (Recommended for Phase 3)
1. Install MLflow: `pipenv install mlflow`
2. Wrap training functions with MLflow tracking
3. Track experiments, metrics, parameters, models
4. View results in MLflow UI (`mlflow ui`)

**Pros**: Professional experiment tracking, easy to use, no heavy infrastructure
**Cons**: Additional dependency, some setup time

### Option C: Add Dagster (For Later)
1. Install Dagster: `pipenv install dagster dagster-webserver`
2. Create DAG for training pipeline
3. Define assets (features, models, evaluations)
4. Run via Dagster UI

**Pros**: Full orchestration, dependency management, monitoring
**Cons**: Infrastructure overhead, learning curve, may be overkill

## Decision Matrix

| Factor | Makefile | MLflow | Dagster |
|--------|----------|--------|---------|
| **Setup Time** | ✅ None | ⚠️ Low | ❌ Medium |
| **Experiment Tracking** | ❌ Manual | ✅ Excellent | ✅ Good |
| **Hyperparameter Tuning** | ⚠️ Manual | ✅ Excellent | ✅ Good |
| **Dependency Management** | ⚠️ Basic | ❌ None | ✅ Excellent |
| **Infrastructure** | ✅ None | ✅ Minimal | ❌ Database + Web |
| **Learning Curve** | ✅ None | ⚠️ Low | ❌ Medium |
| **Best For** | Simple pipelines | ML experiments | Complex orchestration |

## Final Recommendation

**Start with Option A (Keep Simple) + Add MLflow for Experiment Tracking**

1. **Now**: Keep `make train`, add manual experiment tracking
2. **Phase 2**: Add MLflow for experiment tracking (when tuning starts)
3. **Later**: Consider Dagster if orchestration becomes complex

**Rationale**:
- MLflow is the right tool for ML experiment tracking
- Dagster is better for complex orchestration (we don't need that yet)
- Can always add Dagster later if needed
- Focus on model quality first, infrastructure second

