# ML Training Pipeline Guide

Complete guide for using the ML training pipeline with data preparation, model training, registration, and validation.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Quick Start](#quick-start)
- [Individual Scripts](#individual-scripts)
- [Databricks Workflow](#databricks-workflow)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The ML training pipeline automates the complete model lifecycle from data preparation to deployment:

1. **Data Preparation**: Load, split, and version training/test datasets
2. **Model Training**: Train model with MLFlow tracking and baseline comparison
3. **Model Registration**: Register to Unity Catalog if model improves baseline
4. **Model Validation**: Validate registered model before deployment
5. **Deployment**: Optionally deploy to serving (prod environment only)

### Key Features

- ✅ Complete MLFlow experiment tracking
- ✅ Delta table versioning for data lineage
- ✅ Baseline F1 score comparison for model promotion
- ✅ Unity Catalog integration with versioning and aliases
- ✅ Comprehensive model validation before deployment
- ✅ Conditional execution (only register if model improves)
- ✅ Production-ready error handling and logging

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. Data Preparation                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Load source data or generate synthetic data           │   │
│  │ • Split into train/test (80/20 default)                 │   │
│  │ • Save to Delta tables with versioning                  │   │
│  │ • Log datasets to MLFlow with versions                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. Model Training                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Load training/test data with version tracking         │   │
│  │ • Train LightGBM model with hyperparameters            │   │
│  │ • Evaluate: F1, Accuracy, Precision, Recall, ROC-AUC   │   │
│  │ • Compare F1 with baseline threshold                    │   │
│  │ • Log model, params, metrics, artifacts to MLFlow      │   │
│  │ • Return: Exit 0 if improves, Exit 1 if not            │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ (Only if F1 > baseline)
┌─────────────────────────────────────────────────────────────────┐
│                  3. Model Registration                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Register model to Unity Catalog                       │   │
│  │ • Set version tags (git_sha, environment, timestamp)    │   │
│  │ • Set alias (latest-model, champion, etc.)              │   │
│  │ • Generate description with metrics                     │   │
│  │ • Output registered version for downstream tasks        │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. Model Validation                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Load model from Unity Catalog                         │   │
│  │ • Verify model signature (MLFlow)                       │   │
│  │ • Test inference on sample data                         │   │
│  │ • Retrieve and verify metadata                          │   │
│  │ • Return: Pass if 3/4 or 4/4 checks succeed             │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ (Only in prod environment)
┌─────────────────────────────────────────────────────────────────┐
│                  5. Deploy to Serving                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Deploy model to serving endpoint                      │   │
│  │ • Enable inference table logging                        │   │
│  │ • Setup lakehouse monitoring                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
# Install the package
pip install -e .

# Or with uv
uv sync
```

### Environment Variables

```bash
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Run Complete Pipeline

```bash
# 1. Prepare data
prepare_data \
  --catalog mlops_dev \
  --schema my_model \
  --environment dev \
  --test-size 0.2

# 2. Train model
train_model \
  --catalog mlops_dev \
  --schema my_model \
  --environment dev \
  --model-name my_classifier \
  --baseline-f1 0.75 \
  --max-depth 5 \
  --n-estimators 100 \
  --learning-rate 0.1

# 3. Register model (if training succeeded)
register_model \
  --catalog mlops_dev \
  --schema my_model \
  --model-name my_classifier \
  --alias latest-model \
  --environment dev \
  --git-sha $(git rev-parse HEAD)

# 4. Validate model
validate_model \
  --catalog mlops_dev \
  --schema my_model \
  --model-name my_classifier \
  --alias latest-model \
  --sample-size 10
```

---

## Individual Scripts

### 1. Data Preparation Script

**Purpose**: Prepare and version training/test datasets

The script supports three data sources:
1. **Delta Tables** - Load from existing Unity Catalog tables
2. **Feature Store** - Load features from Databricks Feature Store
3. **Synthetic Data** - Generate sample data for testing

#### Option A: Load from Delta Table

```bash
prepare_data \
  --catalog mlops_dev \
  --schema fraud_detection \
  --environment dev \
  --source-table raw_data \
  --train-table train_set \
  --test-table test_set \
  --test-size 0.2 \
  --random-seed 42
```

#### Option B: Load from Feature Store

```bash
prepare_data \
  --catalog mlops_dev \
  --schema fraud_detection \
  --environment dev \
  --feature-store-table mlops_dev.features.user_features \
  --lookup-key user_id \
  --label-table mlops_dev.labels.fraud_labels \
  --label-column is_fraud \
  --train-table train_set \
  --test-table test_set \
  --test-size 0.2 \
  --random-seed 42
```

#### Option C: Load Specific Features from Feature Store

```bash
prepare_data \
  --catalog mlops_dev \
  --schema fraud_detection \
  --environment dev \
  --feature-store-table mlops_dev.features.user_features \
  --lookup-key user_id \
  --feature-columns transaction_amount avg_transaction_amount transaction_count \
  --label-table mlops_dev.labels.fraud_labels \
  --label-column is_fraud \
  --train-table train_set \
  --test-table test_set \
  --test-size 0.2
```

**Parameters**:

**Required**:
- `--catalog` (required): Unity Catalog name
- `--schema` (required): Schema name
- `--environment` (required): Environment (dev/aut/prod)

**Delta Table Source**:
- `--source-table`: Source Delta table name (if not provided, uses synthetic data)

**Feature Store Source** (cannot be used with `--source-table`):
- `--feature-store-table`: Feature Store table name (format: catalog.schema.table)
- `--lookup-key`: Primary key column for feature lookup (required with Feature Store)
- `--feature-columns`: Specific features to load (optional, loads all if not specified)
- `--label-table`: Label table to join with features (format: catalog.schema.table)
- `--label-column`: Target column name in label table (default: target)

**Output Options**:
- `--train-table`: Training table name (default: train_set)
- `--test-table`: Test table name (default: test_set)
- `--test-size`: Test set proportion 0.0-1.0 (default: 0.2)
- `--random-seed`: Random seed for reproducibility (default: 42)

**Outputs**:
- Delta tables: `{catalog}.{schema}.train_set`, `{catalog}.{schema}.test_set`
- MLFlow datasets logged with versions and source metadata
- Exit code 0 on success

**Feature Store Benefits**:
- ✅ Centralized feature management
- ✅ Feature versioning and lineage tracking
- ✅ Automatic feature freshness
- ✅ Consistent features across training and serving
- ✅ Feature monitoring and drift detection

---

### 2. Model Training Script

**Purpose**: Train model with MLFlow tracking and baseline comparison

```bash
train_model \
  --catalog mlops_dev \
  --schema fraud_detection \
  --environment dev \
  --model-name fraud_detector \
  --train-table train_set \
  --test-table test_set \
  --target-col is_fraud \
  --baseline-f1 0.82 \
  --max-depth 7 \
  --n-estimators 200 \
  --learning-rate 0.05 \
  --random-seed 42
```

**Parameters**:
- `--catalog` (required): Unity Catalog name
- `--schema` (required): Schema name
- `--environment` (required): Environment (dev/aut/prod)
- `--model-name` (required): Model name for registration
- `--train-table`: Training table name (default: train_set)
- `--test-table`: Test table name (default: test_set)
- `--target-col`: Target column name (default: target)
- `--baseline-f1`: Baseline F1 score to beat (default: 0.0)
- `--max-depth`: LightGBM max_depth (default: 5)
- `--n-estimators`: LightGBM n_estimators (default: 100)
- `--learning-rate`: LightGBM learning_rate (default: 0.1)
- `--random-seed`: Random seed (default: 42)

**Outputs**:
- MLFlow run with logged model, params, metrics, artifacts
- Exit code 0 if F1 > baseline (ready for registration)
- Exit code 1 if F1 ≤ baseline (skip registration)

**Logged Metrics**:
- `test_f1`: F1 score
- `test_accuracy`: Accuracy
- `test_precision`: Precision
- `test_recall`: Recall
- `test_roc_auc`: ROC-AUC score

---

### 3. Model Registration Script

**Purpose**: Register model to Unity Catalog with versioning and tagging

```bash
register_model \
  --catalog mlops_dev \
  --schema fraud_detection \
  --model-name fraud_detector \
  --run-id abc123def456 \
  --alias latest-model \
  --description "Improved fraud detection with 85% F1" \
  --git-sha $(git rev-parse HEAD) \
  --environment dev \
  --experiment-name "/Shared/mlops_dev/fraud_detection/training"
```

**Parameters**:
- `--catalog` (required): Unity Catalog name
- `--schema` (required): Schema name
- `--model-name` (required): Model name in Unity Catalog
- `--run-id`: Specific MLFlow run ID (if not provided, uses latest successful run)
- `--alias`: Model alias to set (default: latest-model)
- `--description`: Model description
- `--git-sha`: Git commit SHA for tagging
- `--environment`: Environment (dev/aut/prod)
- `--experiment-name`: MLFlow experiment to search

**Outputs**:
- Registered model: `{catalog}.{schema}.{model_name}`
- Model version number
- Console output: `REGISTERED_VERSION=<version>`
- Exit code 0 on success

**Model Tags**:
- `environment`: dev/aut/prod
- `git_sha`: Git commit SHA
- `registered_at`: ISO timestamp

---

### 4. Model Validation Script

**Purpose**: Validate registered model before deployment

```bash
validate_model \
  --catalog mlops_dev \
  --schema fraud_detection \
  --model-name fraud_detector \
  --version 5 \
  --alias latest-model \
  --test-table test_set \
  --sample-size 10
```

**Parameters**:
- `--catalog` (required): Unity Catalog name
- `--schema` (required): Schema name
- `--model-name` (required): Model name
- `--version`: Specific version to validate (optional)
- `--alias`: Model alias (default: latest-model)
- `--test-table`: Test data table (default: test_set)
- `--sample-size`: Number of test samples (default: 5)

**Validation Checks**:
1. ✓ Model loads successfully from Unity Catalog
2. ✓ Model signature is valid (MLFlow signature)
3. ✓ Inference test passes (predictions work)
4. ✓ Metadata retrieval succeeds

**Exit Codes**:
- 0: All checks passed (or 3/4 with warnings)
- 1: Validation failed (< 3 checks passed)

---

## Databricks Workflow

### Deploy Workflow

```bash
# Deploy workflow to dev
databricks bundle deploy --target dev

# Run training workflow
databricks bundle run ml_training_pipeline --target dev

# Monitor workflow
databricks jobs list-runs --job-id <job-id>
```

### Workflow Variables

The workflow uses the following variables from `databricks.yml`:

```yaml
variables:
  catalog: mlops_dev
  schema: my_model
  environment: dev
  model_name: my_classifier
  baseline_f1_score: "0.75"
  git_sha: "${git.commit}"
  alert_email: "ml-team@company.com"
  service_principal_name: "sp-databricks-dev"
```

### Conditional Execution

The workflow implements smart conditional execution:

1. **Model Registration**: Only runs if training exit code = 0 (F1 > baseline)
2. **Model Validation**: Only runs if registration succeeds
3. **Deploy to Serving**: Only runs in prod environment after validation

---

## Configuration

### Environment-Specific Settings

Edit `config/monitoring_config.yml` for environment-specific settings:

```yaml
environments:
  dev:
    catalog: mlops_dev
    schema: fraud_detection
    baseline_f1: 0.70  # Lower threshold for dev

  aut:
    catalog: mlops_aut
    schema: fraud_detection
    baseline_f1: 0.80  # Higher threshold for acceptance

  prod:
    catalog: mlops_prod
    schema: fraud_detection
    baseline_f1: 0.85  # Highest threshold for production
```

### Hyperparameter Tuning

Recommended hyperparameters by use case:

**Binary Classification (Balanced)**:
```bash
train_model \
  --max-depth 5 \
  --n-estimators 100 \
  --learning-rate 0.1
```

**Binary Classification (Imbalanced)**:
```bash
train_model \
  --max-depth 7 \
  --n-estimators 200 \
  --learning-rate 0.05
```

**Multiclass Classification**:
```bash
train_model \
  --max-depth 6 \
  --n-estimators 150 \
  --learning-rate 0.08
```

---

## Examples

### Example 1: Complete Training for New Model

```bash
#!/bin/bash
# Complete training pipeline for fraud detection model

CATALOG="mlops_dev"
SCHEMA="fraud_detection"
ENV="dev"
MODEL_NAME="fraud_detector_v2"
BASELINE_F1="0.82"

echo "Step 1: Prepare data..."
prepare_data \
  --catalog $CATALOG \
  --schema $SCHEMA \
  --environment $ENV \
  --source-table transactions \
  --test-size 0.2

echo "Step 2: Train model..."
train_model \
  --catalog $CATALOG \
  --schema $SCHEMA \
  --environment $ENV \
  --model-name $MODEL_NAME \
  --baseline-f1 $BASELINE_F1 \
  --max-depth 7 \
  --n-estimators 200 \
  --learning-rate 0.05

# Check if training succeeded (exit code 0)
if [ $? -eq 0 ]; then
    echo "Step 3: Register model..."
    register_model \
      --catalog $CATALOG \
      --schema $SCHEMA \
      --model-name $MODEL_NAME \
      --alias latest-model \
      --git-sha $(git rev-parse HEAD) \
      --environment $ENV

    echo "Step 4: Validate model..."
    validate_model \
      --catalog $CATALOG \
      --schema $SCHEMA \
      --model-name $MODEL_NAME \
      --alias latest-model

    if [ $? -eq 0 ]; then
        echo "✓ Training pipeline completed successfully!"
    else
        echo "✗ Model validation failed"
        exit 1
    fi
else
    echo "✗ Model did not improve baseline - skipping registration"
    exit 1
fi
```

### Example 2: Retrain Existing Model

```bash
# Retrain existing model with same configuration

MODEL_NAME="fraud_detector"

# Get current production model F1 score as baseline
BASELINE_F1=$(databricks model get $MODEL_NAME --version latest | jq -r '.tags.f1_score')

echo "Current baseline F1: $BASELINE_F1"

# Train with current baseline
train_model \
  --catalog mlops_prod \
  --schema fraud_detection \
  --environment prod \
  --model-name $MODEL_NAME \
  --baseline-f1 $BASELINE_F1 \
  --max-depth 7 \
  --n-estimators 200
```

### Example 3: Training with Feature Store

```bash
#!/bin/bash
# Complete training pipeline using Databricks Feature Store

CATALOG="mlops_dev"
SCHEMA="fraud_detection"
ENV="dev"
MODEL_NAME="fraud_detector_fs"

# Feature Store tables
FEATURE_TABLE="mlops_dev.features.user_transaction_features"
LABEL_TABLE="mlops_dev.labels.fraud_labels"

echo "Step 1: Prepare data from Feature Store..."
prepare_data \
  --catalog $CATALOG \
  --schema $SCHEMA \
  --environment $ENV \
  --feature-store-table $FEATURE_TABLE \
  --lookup-key user_id \
  --label-table $LABEL_TABLE \
  --label-column is_fraud \
  --test-size 0.2

echo "Step 2: Train model..."
train_model \
  --catalog $CATALOG \
  --schema $SCHEMA \
  --environment $ENV \
  --model-name $MODEL_NAME \
  --target-col is_fraud \
  --baseline-f1 0.75 \
  --max-depth 7 \
  --n-estimators 200 \
  --learning-rate 0.05

if [ $? -eq 0 ]; then
    echo "Step 3: Register model..."
    register_model \
      --catalog $CATALOG \
      --schema $SCHEMA \
      --model-name $MODEL_NAME \
      --alias latest-model \
      --git-sha $(git rev-parse HEAD) \
      --environment $ENV

    echo "Step 4: Validate model..."
    validate_model \
      --catalog $CATALOG \
      --schema $SCHEMA \
      --model-name $MODEL_NAME \
      --alias latest-model

    echo "✓ Feature Store training pipeline completed!"
else
    echo "✗ Model did not improve baseline"
    exit 1
fi
```

**Feature Store Setup**:

Before running this pipeline, ensure your Feature Store tables exist:

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create feature table
fe.create_table(
    name="mlops_dev.features.user_transaction_features",
    primary_keys=["user_id"],
    df=user_features_df,
    description="User transaction aggregated features"
)

# Create label table (separate from features)
label_df.write.saveAsTable("mlops_dev.labels.fraud_labels")
```

### Example 4: A/B Testing with Champion/Challenger

```bash
# Register champion model
register_model \
  --catalog mlops_prod \
  --schema recommender \
  --model-name product_recommender \
  --alias champion \
  --environment prod

# Register challenger model
register_model \
  --catalog mlops_prod \
  --schema recommender \
  --model-name product_recommender_v2 \
  --alias challenger \
  --environment prod

# Validate both
validate_model --model-name product_recommender --alias champion
validate_model --model-name product_recommender_v2 --alias challenger
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Training fails with "No module named 'databricks_monitoring'"

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Or reinstall
pip uninstall databricks-lakehouse-monitoring
pip install -e .
```

#### Issue 2: Model doesn't improve baseline

**Symptom**: Training exits with code 1, registration skipped

**Solution**:
- Lower the baseline threshold
- Tune hyperparameters
- Check data quality
- Verify feature engineering

```bash
# Check current model performance
train_model \
  --baseline-f1 0.0 \  # Set baseline to 0 temporarily
  # ... other params
```

#### Issue 3: Unity Catalog permission denied

**Symptom**: `PermissionDenied: User does not have USAGE on catalog`

**Solution**:
```sql
-- Grant necessary permissions
GRANT USAGE ON CATALOG mlops_dev TO `your-user@company.com`;
GRANT CREATE ON SCHEMA mlops_dev.my_model TO `your-user@company.com`;
GRANT SELECT, MODIFY ON SCHEMA mlops_dev.my_model TO `your-user@company.com`;
```

#### Issue 4: MLFlow experiment not found

**Symptom**: `Experiment '/Shared/catalog/schema/training' not found`

**Solution**:
```python
# Create experiment manually
import mlflow
mlflow.create_experiment(
    name="/Shared/mlops_dev/my_model/training",
    tags={"purpose": "training", "environment": "dev"}
)
```

#### Issue 5: Feature Store table not found

**Symptom**: `Table 'catalog.schema.feature_table' not found`

**Solution**:
```python
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()

# Verify feature table exists
try:
    feature_df = spark.table("mlops_dev.features.user_features")
    print(f"Feature table found: {feature_df.count()} rows")
except Exception as e:
    print(f"Feature table not found: {e}")
    # Create the feature table
    # fe.create_table(...)
```

#### Issue 6: Feature Store join fails - no matching keys

**Symptom**: `Lost X rows in join. Ensure label table has matching lookup_key values`

**Solution**:
```python
# Check for matching keys between feature and label tables
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

feature_df = spark.table("mlops_dev.features.user_features")
label_df = spark.table("mlops_dev.labels.fraud_labels")

# Count unique keys in each table
feature_keys = feature_df.select("user_id").distinct().count()
label_keys = label_df.select("user_id").distinct().count()

print(f"Feature table has {feature_keys} unique user_ids")
print(f"Label table has {label_keys} unique user_ids")

# Find keys in features but not in labels
missing_in_labels = feature_df.select("user_id").subtract(label_df.select("user_id"))
print(f"Keys in features missing from labels: {missing_in_labels.count()}")
```

#### Issue 7: Feature Engineering client not available

**Symptom**: `No module named 'databricks.feature_engineering'`

**Solution**:
```bash
# Install feature engineering client
pip install databricks-feature-engineering

# Or add to pyproject.toml
[project]
dependencies = [
    "databricks-feature-engineering>=0.2.0",
    # ... other dependencies
]
```

#### Issue 5: Validation fails on signature check

**Symptom**: Model signature validation fails (check 2/4 fails)

**Solution**:
- Ensure model was logged with signature in training
- Check MLFlow model artifacts
- Verify input_example was provided

---

## Best Practices

### 1. Version Control

Always tag models with git commit SHA:
```bash
register_model --git-sha $(git rev-parse HEAD)
```

### 2. Baseline Management

Update baseline incrementally:
```bash
# Get current champion F1
CURRENT_F1=0.85

# Set slightly higher baseline for next training
NEW_BASELINE=$(echo "$CURRENT_F1 + 0.01" | bc)

train_model --baseline-f1 $NEW_BASELINE
```

### 3. Experiment Tracking

Use descriptive experiment names:
```bash
# Good
/Shared/mlops_prod/fraud_detection/training

# Bad
/my_experiment
```

### 4. Testing Strategy

Always test in dev before prod:
```bash
# Test in dev first
prepare_data --environment dev
train_model --environment dev

# If successful, promote to prod
prepare_data --environment prod
train_model --environment prod --baseline-f1 ${PROD_BASELINE}
```

---

## Next Steps

1. **Set up scheduled training**: Configure workflow to run weekly
2. **Monitor model performance**: Track F1 scores over time
3. **Implement A/B testing**: Use champion/challenger aliases
4. **Add custom metrics**: Extend evaluation with domain-specific metrics
5. **Automate retraining**: Trigger training on data drift detection

## Support

For issues:
- Check [Troubleshooting](#troubleshooting) section
- Review MLFlow experiment logs
- Check Databricks job logs
- Contact: ml-ops-team@company.com
