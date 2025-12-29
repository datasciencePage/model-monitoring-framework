# Databricks Lakehouse Monitoring Framework - Setup Guide

## Overview

This document provides comprehensive instructions for setting up and using the Databricks Lakehouse monitoring framework for ML models. The framework supports end-to-end MLOps with model serving, inference logging, drift detection, and alerting.

## Architecture

```
Training → Model Registry (Unity Catalog) → Model Serving (with Inference Tables) → Lakehouse Monitoring → Dashboards & Alerts
```

### Key Components

1. **MLFlow Tracking** - Comprehensive experiment logging
2. **Model Registry** - Unity Catalog model management
3. **Model Serving** - Databricks serving endpoints with inference tables
4. **Lakehouse Monitoring** - Drift detection and data quality monitoring
5. **Generic Framework** - Reusable monitoring for any model type

---

## Prerequisites

### Azure Setup

1. **Service Principal**: Create in Azure AD for Databricks access
   ```bash
   az ad sp create-for-rbac --name "databricks-cicd" \
     --role Contributor \
     --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group}
   ```

2. **Databricks Workspaces**: Three separate workspaces for dev/aut/prod

3. **Unity Catalog**: Set up with catalogs:
   - `mlops_dev`
   - `mlops_aut`
   - `mlops_prod`

### Azure DevOps Setup

1. **Service Connection**:
   - Go to Project Settings → Service connections
   - Create "Azure Resource Manager" connection named `Azure-ServiceConnection`
   - Use Service Principal authentication

2. **Variable Groups**:
   - Create `databricks-secrets-dev`:
     - `DATABRICKS_HOST_DEV`: https://adb-xxxxx.azuredatabricks.net
   - Create `databricks-secrets-aut`:
     - `DATABRICKS_HOST_AUT`: https://adb-xxxxx.azuredatabricks.net
   - Create `databricks-secrets-prod`:
     - `DATABRICKS_HOST_PRD`: https://adb-xxxxx.azuredatabricks.net

3. **Environments**:
   - Create environments: `dev`, `aut`, `prod`
   - Add approval gates for `aut` and `prod`

### Databricks Setup

1. **Add Service Principal to Workspaces**:
   ```sql
   -- In each workspace
   GRANT CREATE, USAGE ON CATALOG mlops_{env} TO SERVICE_PRINCIPAL `{app-id}`;
   GRANT CREATE, USAGE ON SCHEMA mlops_{env}.databricks_monitoring TO SERVICE_PRINCIPAL `{app-id}`;
   ```

2. **Enable Serverless Compute** (if using serverless endpoints)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/ml-model-characters.git
cd ml-model-characters
```

### 2. Install Dependencies

```bash
# Install UV package manager
pip install uv

# Install project dependencies
uv sync --extra dev
```

### 3. Build Package

```bash
uv build
```

---

## Configuration

### Monitoring Configuration

Edit [`config/monitoring_config.yml`](config/monitoring_config.yml):

```yaml
environments:
  dev:
    catalog: mlops_dev
    schema: databricks_monitoring
    serving:
      endpoint_name: ml-model-model-serving-dev
      workload_size: Small
      scale_to_zero: true
    inference_table:
      name: inference_logs
      retention_days: 90
    monitoring:
      output_schema: monitoring
      granularities: ["1 day"]
      refresh_schedule: "0 0 * * *"
      alerts:
        - metric: prediction_drift
          threshold: 0.1
          notification_type: email
          recipients: ["your-team@example.com"]
```

### Databricks Bundle Configuration

Edit [`databricks.yml`](databricks.yml):

```yaml
targets:
  dev:
    workspace:
      host: https://your-workspace-url.azuredatabricks.net
      profile: your-profile

  # Update aut and prod similarly
```

---

## Deployment

### Local Testing (Dev Environment)

```bash
# Deploy to dev
databricks bundle deploy --target dev

# Run serving deployment workflow
databricks bundle run serving_and_monitoring_deployment --target dev

# Run monitoring refresh
databricks bundle run monitoring_refresh --target dev
```

### CI/CD Pipeline

The Azure DevOps pipeline ([`azure-pipelines.yml`](azure-pipelines.yml)) automatically:

1. **On push to `develop` branch**:
   - Builds wheel package
   - Runs tests
   - Deploys to `dev` environment
   - Deploys to `aut` environment (with approval)

2. **On push to `main` branch**:
   - Builds wheel package
   - Runs tests
   - Deploys to `prod` environment (with approval)
   - Creates git tag for release

---

## Usage

### Using the Generic Monitoring Framework

```python
from databricks_monitoring.monitoring.framework import MonitoringFramework

# Initialize framework
framework = MonitoringFramework(
    catalog="mlops_prod",
    schema="databricks_monitoring",
    model_name="ml_character_model_basic"
)

# Deploy with monitoring
deployment = framework.deploy_with_monitoring(
    model_version="1",
    serving_config={
        "endpoint_name": "ml-model-model-serving-prod",
        "workload_size": "Medium",
        "scale_to_zero": False,
    },
    inference_table_config={
        "name": "inference_logs",
        "enabled": True,
    },
    monitoring_config={
        "output_schema": "monitoring",
        "granularities": ["1 day"],
        "baseline_table": f"mlops_prod.databricks_monitoring.train_set",
        "problem_type": "classification",
        "alerts": [
            {
                "metric": "prediction_drift",
                "threshold": 0.05,
                "notification_type": "email",
                "recipients": ["ml-ops@example.com"],
            }
        ],
    },
    tags={
        "environment": "prod",
        "model_type": "classification",
    },
)

print(f"Endpoint: {deployment['endpoint_name']}")
print(f"Monitor: {deployment['monitor_name']}")
print(f"Dashboard: {deployment['dashboard_url']}")
```

### Using Individual Components

#### 1. MLFlow Tracking

```python
from databricks_monitoring.mlflow_tracking import MLFlowTracker

tracker = MLFlowTracker(experiment_name="/Shared/ml-model-characters-basic")

with tracker.start_run(run_name="training-run-001"):
    # Log training data
    tracker.log_training_data(
        train_df=train_df,
        test_df=test_df,
        train_table_name="mlops_prod.databricks_monitoring.train_set",
        test_table_name="mlops_prod.databricks_monitoring.test_set",
    )

    # Log hyperparameters
    tracker.log_model_params({
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "max_depth": 6,
    })

    # Train model...
    model.fit(X_train, y_train)

    # Log metrics
    tracker.log_model_metrics({
        "f1_score": 0.85,
        "accuracy": 0.82,
        "precision": 0.87,
        "recall": 0.83,
    })

    # Log model
    model_info = tracker.log_model_artifacts(
        model=model,
        X_sample=X_train.head(100),
        registered_model_name="mlops_prod.databricks_monitoring.ml_character_model_basic",
    )
```

#### 2. Model Registry

```python
from databricks_monitoring.model_registry import ModelRegistry

registry = ModelRegistry(catalog="mlops_prod", schema="databricks_monitoring")

# Register model
model_version = registry.register_model(
    model_uri="runs:/{run_id}/model",
    model_name="ml_character_model_basic",
    tags={"git_sha": "abc123", "environment": "prod"},
    description="LightGBM classifier for Generic character survival prediction",
)

# Set alias
registry.set_model_alias(
    model_name="ml_character_model_basic",
    version=model_version.version,
    alias="latest-model",
)

# Load model for inference
model = registry.load_model_for_inference(
    model_name="ml_character_model_basic",
    alias="latest-model",
)
```

#### 3. Model Serving

```python
from databricks_monitoring.serving.model_serving_setup import ServingSetup

serving = ServingSetup()

# Create endpoint with inference table
endpoint_name = serving.create_endpoint(
    endpoint_name="ml-model-model-serving-prod",
    model_name="mlops_prod.databricks_monitoring.ml_character_model_basic",
    model_version="1",
    workload_size="Medium",
    scale_to_zero=False,
    inference_table_config={
        "catalog": "mlops_prod",
        "schema": "databricks_monitoring",
        "table_name": "inference_logs",
        "enabled": True,
    },
)

# Wait for endpoint to be ready
serving.wait_for_endpoint_ready(endpoint_name, timeout=600)

# Invoke endpoint
response = serving.invoke_endpoint(
    endpoint_name=endpoint_name,
    payload={"dataframe_records": [{"Height": 1.75, "Weight": 70, ...}]},
)
```

#### 4. Lakehouse Monitoring

```python
from databricks_monitoring.monitoring.lakehouse_monitor import LakehouseMonitor

monitor = LakehouseMonitor(catalog="mlops_prod", schema="databricks_monitoring")

# Create monitor
monitor_name = monitor.create_monitor(
    inference_table="mlops_prod.databricks_monitoring.inference_logs",
    output_schema="mlops_prod.monitoring",
    profile_type="InferenceLog",
    granularities=["1 day"],
    baseline_table="mlops_prod.databricks_monitoring.train_set",
    problem_type="classification",
    prediction_col="prediction",
    timestamp_col="timestamp",
)

# Refresh monitor
monitor.refresh_monitor(monitor_name)

# Get metrics
metrics = monitor.get_monitor_metrics(monitor_name)
```

---

## Workflows

### Serving & Monitoring Deployment Workflow

**File**: [`resources/workflows/serving_deployment.yml`](resources/workflows/serving_deployment.yml)

**Tasks**:
1. Deploy serving endpoint with inference table
2. Validate inference table configuration
3. Setup Lakehouse monitoring

**Trigger**: On-demand or after successful training

### Monitoring Refresh Workflow

**File**: [`resources/workflows/monitoring_refresh.yml`](resources/workflows/monitoring_refresh.yml)

**Tasks**:
1. Refresh all monitors
2. Analyze drift
3. Send alerts if thresholds exceeded

**Schedule**: Daily (configurable per environment)

---

## Scripts

All scripts are available as entry points after installation:

### Deployment Scripts

```bash
# Deploy serving endpoint
deploy_serving --catalog mlops_prod --schema databricks_monitoring --environment prod --git-sha abc123

# Validate inference table
validate_inference_table --catalog mlops_prod --schema databricks_monitoring

# Setup monitoring
setup_monitoring --catalog mlops_prod --schema databricks_monitoring --environment prod
```

### Monitoring Scripts

```bash
# Refresh all monitors
refresh_all_monitors --catalog mlops_prod --schema databricks_monitoring --environment prod

# Analyze drift
analyze_drift --catalog mlops_prod --schema databricks_monitoring --days 7

# Send alerts
send_alerts --catalog mlops_prod --schema databricks_monitoring --environment prod
```

---

## Monitoring Dashboard

After setting up Lakehouse monitoring, dashboards are automatically created at:

```
/Workspace/Shared/lakehouse_monitoring/{catalog}_{schema}_{table_name}/
```

**Dashboard Includes**:
- Prediction distribution over time
- Feature drift scores
- Data quality metrics
- Null rates and type mismatches
- Comparison with baseline (training data)

---

## Alerting

### Email Alerts

Configure email recipients in `config/monitoring_config.yml`:

```yaml
alerts:
  - metric: prediction_drift
    threshold: 0.05
    severity: critical
    notification_type: email
    recipients:
      - ml-ops-team@example.com
      - data-science-lead@example.com
```

### Custom Alerts

Extend [`src/databricks_monitoring/scripts/send_alerts.py`](src/databricks_monitoring/scripts/send_alerts.py) to add:
- Slack notifications
- PagerDuty integration
- Custom webhooks
- SMS alerts

---

## Adapting for New Models

### 1. Use Configuration Templates

Choose a template from `config_templates/`:
- [`classification_model.yml`](config_templates/classification_model.yml)
- [`regression_model.yml`](config_templates/regression_model.yml)
- [`forecasting_model.yml`](config_templates/forecasting_model.yml)

### 2. Update Configuration

```yaml
# config/my_new_model_config.yml
model:
  name: my_new_model
  type: classification

serving:
  endpoint_name: my-new-model-serving
  # ... other settings
```

### 3. Use Generic Framework

```python
from databricks_monitoring.monitoring.framework import MonitoringFramework

framework = MonitoringFramework(
    catalog="mlops_prod",
    schema="my_schema",
    model_name="my_new_model"
)

framework.deploy_with_monitoring(
    model_version="1",
    serving_config={...},
    inference_table_config={...},
    monitoring_config={...},
)
```

---

## Troubleshooting

### Issue: Inference table not populating

**Solution**: Send test requests to the endpoint
```python
serving.invoke_endpoint(endpoint_name, payload={...})
```

### Issue: Monitor refresh fails

**Solution**: Check monitor status and retry
```python
monitor.refresh_monitor(table_name)
```

### Issue: Alerts not sending

**Solution**: Verify alert configuration and implement email sending (see `send_alerts.py`)

### Issue: Databricks bundle deploy fails

**Solution**:
1. Check authentication: `databricks auth profiles`
2. Verify workspace URL in `databricks.yml`
3. Ensure Service Principal has permissions

---

## Best Practices

1. **Always use model aliases** (`latest-model`, `champion`) instead of hard-coded versions
2. **Enable inference tables at endpoint creation** - cannot be added later
3. **Set up baseline tables** for meaningful drift detection
4. **Monitor daily** in production, less frequently in dev/aut
5. **Use 90-day retention** for inference logs (balance cost vs. analysis needs)
6. **Tag all deployments** with git_sha and environment for traceability
7. **Test in dev** before promoting to aut/prod
8. **Set up approval gates** in Azure DevOps for production deployments

---

## References

- [Databricks Model Serving Documentation](https://docs.databricks.com/en/machine-learning/model-serving.html)
- [Databricks Inference Tables](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html)
- [Databricks Lakehouse Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Databricks documentation
3. Open an issue in the repository
4. Contact the ML Engineering team

---

**Last Updated**: December 26, 2024
**Version**: 1.0.0
