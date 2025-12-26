# Databricks Lakehouse Monitoring Framework with Azure DevOps CI/CD

## Project Goal

Design and implement a scalable, reusable Databricks Lakehouse monitoring framework for ML models that can be integrated into Azure DevOps CI/CD pipelines using Databricks Asset Bundles.

## Requirements

1. **MLFlow Integration**: Log model parameters, metrics, artifacts, and datasets (training & inference)
2. **Model Registry**: Register models in Unity Catalog
3. **Model Management**: Load models and metadata when needed
4. **Model Serving**: Deploy models with Databricks Model Serving
5. **Inference Tables**: Enable inference table logging for monitoring
6. **Lakehouse Monitoring**: Create comprehensive monitoring dashboards
7. **Scalability**: Framework must be easily customizable for future ML models
8. **CI/CD**: Deploy via Azure DevOps pipelines (not GitHub Actions)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Azure DevOps CI/CD Pipeline                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Build Stage │→ │Deploy to DEV │→ │Deploy to PRD │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Databricks Asset Bundle Deployment               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML Training Workflow                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Prep    │→ │Model Training│→ │   MLFlow     │      │
│  │              │  │   + Eval     │  │   Logging    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                            ↓                                 │
│                    ┌──────────────┐                         │
│                    │Unity Catalog │                         │
│                    │   Registry   │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Model Serving + Inference Tables                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Model Serving │→ │Inference Log │→ │Inference Tbl │      │
│  │  Endpoint    │  │  (Requests)  │  │   (Delta)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             Lakehouse Monitoring Framework                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Monitor Config│→ │Monitor Tables│→ │  Dashboards  │      │
│  │   Setup      │  │   Creation   │  │  & Alerts    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Framework Components

#### 1.1 MLFlow Tracking Module
**File**: `src/marvel_characters/mlflow_tracking.py` (new)

**Purpose**: Centralized MLFlow logging utilities

**Key Functions**:
- `log_training_data(train_df, test_df, version)`: Log input datasets with versions
- `log_model_params(params_dict)`: Log hyperparameters
- `log_model_metrics(metrics_dict)`: Log performance metrics
- `log_model_artifacts(model, signature, input_example)`: Log model with metadata
- `log_inference_data(inference_df, version)`: Log inference datasets

**Integration Points**:
- Use existing `BasicModel` class but enhance with comprehensive logging
- Leverage Delta table versions from `DataProcessor`
- Track lineage from data → training → inference

---

#### 1.2 Model Registry Module
**File**: `src/marvel_characters/model_registry.py` (new)

**Purpose**: Unity Catalog registration and version management

**Key Functions**:
- `register_model(model_uri, model_name, tags, description)`: Register to UC
- `get_latest_model_version(model_name, stage)`: Retrieve latest version
- `get_model_metadata(model_name, version)`: Load model info
- `load_model_for_inference(model_name, version)`: Load model object
- `set_model_alias(model_name, version, alias)`: Set aliases (e.g., "production")
- `transition_model_stage(model_name, version, stage)`: Stage transitions

**Best Practices**:
- Use model aliases for deployment (`latest-model`, `champion`, `challenger`)
- Tag models with git_sha, branch, environment
- Store model descriptions with performance metrics

---

#### 1.3 Model Serving Setup Module
**File**: `src/marvel_characters/serving/model_serving_setup.py` (enhanced)

**Purpose**: Create and configure model serving endpoints with inference tables

**Key Functions**:
- `create_serving_endpoint(config)`: Create endpoint with inference table enabled
- `update_serving_endpoint(endpoint_name, model_version)`: Update model version
- `enable_inference_table(endpoint_name, catalog, schema, table_name)`: Configure inference logging
- `get_endpoint_status(endpoint_name)`: Check endpoint health
- `invoke_endpoint(endpoint_name, payload)`: Test endpoint

**Configuration Structure**:
```python
ServingConfig:
  - endpoint_name: str
  - model_name: str (UC path)
  - model_version: str
  - workload_size: str (Small, Medium, Large)
  - scale_to_zero: bool
  - inference_table_enabled: bool
  - inference_table_catalog: str
  - inference_table_schema: str
  - inference_table_name: str
```

**Key Feature**: Enable inference tables at endpoint creation time

Reference: https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables

---

#### 1.4 Lakehouse Monitoring Module
**File**: `src/marvel_characters/monitoring/lakehouse_monitor.py` (new)

**Purpose**: Create and manage Lakehouse monitors for model drift detection

**Key Functions**:
- `create_monitor(config)`: Create Lakehouse monitor from inference table
- `refresh_monitor(monitor_name)`: Manually refresh monitor metrics
- `get_monitor_metrics(monitor_name, time_range)`: Retrieve metrics
- `create_alert(monitor_name, alert_config)`: Set up alerting
- `delete_monitor(monitor_name)`: Cleanup

**Monitor Configuration**:
```python
MonitorConfig:
  - table_name: str (inference table)
  - profile_type: str (InferenceLog, TimeSeries)
  - output_catalog: str
  - output_schema: str
  - granularities: List[str] (e.g., ["30 minutes", "1 day"])
  - baseline_table: str (optional, for drift comparison)
  - slicing_exprs: List[str] (optional, segment analysis)
  - linked_entities: List[str] (optional, related tables)
```

**Monitoring Features**:
- Prediction drift detection
- Feature drift detection (input distribution changes)
- Data quality metrics (null rates, type mismatches)
- Model performance tracking (if labels available)

Reference: https://docs.databricks.com/aws/en/lakehouse-monitoring/index.html

---

#### 1.5 Configuration Management
**File**: `src/marvel_characters/monitoring/config.py` (new)

**Purpose**: Centralized configuration for monitoring framework

**Configuration Structure**:
```yaml
# monitoring_config.yml
environments:
  dev:
    catalog: mlops_dev
    schema: marvel_characters
    serving:
      endpoint_name: marvel-model-serving-dev
      workload_size: Small
      scale_to_zero: true
    inference_table:
      name: inference_logs
      retention_days: 90
    monitoring:
      output_schema: monitoring
      granularities: ["1 day"]
      refresh_schedule: "0 0 * * *"  # Daily at midnight
      alerts:
        - metric: prediction_drift
          threshold: 0.1
          notification_type: email
          recipients: ["dev-team@example.com"]

  aut:
    catalog: mlops_aut
    schema: marvel_characters
    serving:
      endpoint_name: marvel-model-serving-aut
      workload_size: Small
      scale_to_zero: true
    inference_table:
      name: inference_logs
      retention_days: 90
    monitoring:
      output_schema: monitoring
      granularities: ["1 day"]
      refresh_schedule: "0 0 * * *"  # Daily at midnight
      alerts:
        - metric: prediction_drift
          threshold: 0.1
          notification_type: email
          recipients: ["qa-team@example.com"]

  prod:
    catalog: mlops_prod
    schema: marvel_characters
    serving:
      endpoint_name: marvel-model-serving-prod
      workload_size: Medium
      scale_to_zero: false
    inference_table:
      name: inference_logs
      retention_days: 90
    monitoring:
      output_schema: monitoring
      granularities: ["1 day"]
      refresh_schedule: "0 2 * * *"  # Daily at 2 AM
      alerts:
        - metric: prediction_drift
          threshold: 0.05
          notification_type: email
          recipients: ["ml-ops-team@example.com", "data-science-lead@example.com"]
```

**Pydantic Models**:
- `MonitoringEnvironmentConfig`: Environment-specific settings
- `ServingConfig`: Model serving configuration
- `InferenceTableConfig`: Inference table settings
- `LakehouseMonitorConfig`: Monitor configuration
- `AlertConfig`: Alerting rules

---

### Phase 2: Databricks Workflow Integration

#### 2.1 Training + Registration Workflow
**File**: `resources/workflows/ml_training.yml` (new)

**Purpose**: End-to-end training workflow with MLFlow logging and UC registration

**Workflow Tasks**:

1. **data_preparation**
   - Script: `scripts/prepare_data.py`
   - Actions:
     - Load and preprocess data
     - Split train/test
     - Save to Delta tables with versions
     - Log dataset versions to MLFlow

2. **model_training**
   - Script: `scripts/train_model.py`
   - Actions:
     - Load training data with version tracking
     - Train model (LightGBM)
     - Log parameters, metrics, artifacts to MLFlow
     - Log training datasets with versions
     - Evaluate on test set
     - Compare with baseline model

3. **model_registration**
   - Script: `scripts/register_model.py`
   - Actions:
     - Register model to Unity Catalog if improved
     - Set model tags (git_sha, environment, metrics)
     - Set model alias ("latest-model")
     - Output: model version number

4. **model_validation** (conditional)
   - Script: `scripts/validate_model.py`
   - Actions:
     - Load registered model
     - Run validation tests
     - Verify model signature
     - Test inference

---

#### 2.2 Serving + Monitoring Deployment Workflow
**File**: `resources/workflows/serving_deployment.yml` (new)

**Purpose**: Deploy model serving with inference tables and monitoring

**Workflow Tasks**:

1. **create_serving_endpoint**
   - Script: `scripts/deploy_serving.py`
   - Actions:
     - Get latest model version from UC
     - Create/update serving endpoint
     - **Enable inference table at creation**
     - Configure autoscaling and workload size
     - Wait for endpoint to be ready

2. **validate_inference_table**
   - Script: `scripts/validate_inference_table.py`
   - Actions:
     - Send test requests to endpoint
     - Verify inference table is populated
     - Check schema and data quality
     - Validate logging frequency

3. **create_lakehouse_monitor**
   - Script: `scripts/setup_monitoring.py`
   - Actions:
     - Create Lakehouse monitor on inference table
     - Configure drift detection (prediction + features)
     - Set granularity (30 min, 1 day)
     - Create baseline from training data (optional)
     - Set up alerting rules

4. **refresh_monitor_tables**
   - Script: `scripts/refresh_monitoring.py`
   - Actions:
     - Refresh monitor metrics
     - Validate dashboard creation
     - Send test alerts

---

#### 2.3 Monitoring Refresh Workflow (Scheduled)
**File**: `resources/workflows/monitoring_refresh.yml` (new)

**Purpose**: Periodic monitoring refresh and alerting

**Schedule**: Daily (configurable per environment)

**Workflow Tasks**:

1. **refresh_all_monitors**
   - Script: `scripts/refresh_all_monitors.py`
   - Actions:
     - List all active monitors
     - Refresh each monitor
     - Collect metrics
     - Check alert conditions

2. **drift_detection_analysis**
   - Script: `scripts/analyze_drift.py`
   - Actions:
     - Analyze prediction drift
     - Analyze feature drift
     - Compare against baseline
     - Generate drift report

3. **alert_on_anomalies**
   - Script: `scripts/send_alerts.py`
   - Actions:
     - Check drift thresholds
     - Check data quality issues
     - Send notifications (email, PagerDuty, Slack)

---

### Phase 3: Azure DevOps CI/CD Pipeline

#### 3.1 Azure Pipelines Configuration
**File**: `azure-pipelines.yml` (new)

**Purpose**: Multi-stage CI/CD pipeline for Databricks deployment

**Pipeline Structure**:

```yaml
trigger:
  branches:
    include:
      - main
      - develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  - group: databricks-secrets-dev
  - group: databricks-secrets-aut
  - group: databricks-secrets-prod

stages:
  - stage: Build
    jobs:
      - job: BuildPackage
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '3.12'
          - script: pip install uv
            displayName: 'Install UV'
          - script: uv build
            displayName: 'Build wheel package'
          - publish: dist/
            artifact: wheel-package

  - stage: Test
    jobs:
      - job: RunTests
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '3.12'
          - script: |
              pip install uv
              uv sync --extra dev
            displayName: 'Install dependencies'
          - script: |
              uv run pytest tests/ -v --cov=src
            displayName: 'Run tests'
          - script: |
              uv run ruff check src/
            displayName: 'Run linting'

  - stage: DeployDev
    dependsOn: Test
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
    jobs:
      - deployment: DeployToDev
        environment: 'dev'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: UsePythonVersion@0
                  inputs:
                    versionSpec: '3.12'
                - task: AzureCLI@2
                  displayName: 'Authenticate with Azure AD'
                  inputs:
                    azureSubscription: 'Azure-ServiceConnection'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      # Get Azure AD token for Databricks
                      export DATABRICKS_TOKEN=$(az account get-access-token --resource 2ff814a6-3304-4ab8-85cb-cd0e6f879c1d --query accessToken -o tsv)
                - script: |
                    pip install databricks-cli
                  displayName: 'Install Databricks CLI'
                - script: |
                    databricks bundle deploy --target dev \
                      --var="git_sha=$(Build.SourceVersion)" \
                      --var="branch=$(Build.SourceBranchName)"
                  displayName: 'Deploy to Dev'
                  env:
                    DATABRICKS_HOST: $(DATABRICKS_HOST_DEV)
                    DATABRICKS_TOKEN: $(DATABRICKS_TOKEN)

  - stage: DeployAut
    dependsOn: DeployDev
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
    jobs:
      - deployment: DeployToAut
        environment: 'aut'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: UsePythonVersion@0
                  inputs:
                    versionSpec: '3.12'
                - task: AzureCLI@2
                  displayName: 'Authenticate with Azure AD'
                  inputs:
                    azureSubscription: 'Azure-ServiceConnection'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      # Get Azure AD token for Databricks
                      export DATABRICKS_TOKEN=$(az account get-access-token --resource 2ff814a6-3304-4ab8-85cb-cd0e6f879c1d --query accessToken -o tsv)
                - script: |
                    pip install databricks-cli
                  displayName: 'Install Databricks CLI'
                - script: |
                    databricks bundle deploy --target aut \
                      --var="git_sha=$(Build.SourceVersion)" \
                      --var="branch=$(Build.SourceBranchName)"
                  displayName: 'Deploy to Aut'
                  env:
                    DATABRICKS_HOST: $(DATABRICKS_HOST_AUT)
                    DATABRICKS_TOKEN: $(DATABRICKS_TOKEN)

  - stage: DeployPrd
    dependsOn: Test
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - deployment: DeployToPrd
        environment: 'prod'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: UsePythonVersion@0
                  inputs:
                    versionSpec: '3.12'
                - task: AzureCLI@2
                  displayName: 'Authenticate with Azure AD'
                  inputs:
                    azureSubscription: 'Azure-ServiceConnection'
                    scriptType: 'bash'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      # Get Azure AD token for Databricks
                      export DATABRICKS_TOKEN=$(az account get-access-token --resource 2ff814a6-3304-4ab8-85cb-cd0e6f879c1d --query accessToken -o tsv)
                - script: |
                    pip install databricks-cli
                  displayName: 'Install Databricks CLI'
                - script: |
                    databricks bundle deploy --target prod \
                      --var="git_sha=$(Build.SourceVersion)" \
                      --var="branch=$(Build.SourceBranchName)"
                  displayName: 'Deploy to Prd'
                  env:
                    DATABRICKS_HOST: $(DATABRICKS_HOST_PRD)
                    DATABRICKS_TOKEN: $(DATABRICKS_TOKEN)
                - script: |
                    git tag -a "v$(cat version.txt)" -m "Production release"
                    git push origin "v$(cat version.txt)"
                  displayName: 'Create Git Tag'
```

**Key Features**:
- Multi-stage pipeline (Build → Test → Deploy Dev → Deploy Prd)
- Branch-based deployment (develop → dev, main → prod)
- Environment-specific secrets from Azure DevOps variable groups
- Databricks CLI for bundle deployment
- Git tagging for production releases

---

#### 3.2 Azure DevOps Variable Groups

**Create in Azure DevOps**:

**Variable Group: databricks-secrets-dev**
- `DATABRICKS_HOST_DEV`: Databricks workspace URL (e.g., https://adb-123456.azuredatabricks.net)

**Variable Group: databricks-secrets-aut**
- `DATABRICKS_HOST_AUT`: Databricks workspace URL

**Variable Group: databricks-secrets-prod**
- `DATABRICKS_HOST_PRD`: Databricks workspace URL

**Service Connection**:
- Create Azure Resource Manager service connection named `Azure-ServiceConnection`
- This provides Service Principal authentication for all environments
- The Service Principal must have "Contributor" role on Databricks workspaces

---

### Phase 4: Databricks Asset Bundle Structure

#### 4.1 Main Bundle Configuration
**File**: `databricks.yml` (update existing)

**Key Additions**:

```yaml
bundle:
  name: marvel-characters

include:
  - resources/workflows/ml_training.yml
  - resources/workflows/serving_deployment.yml
  - resources/workflows/monitoring_refresh.yml

variables:
  git_sha:
    description: Git commit SHA
  branch:
    description: Git branch name

targets:
  dev:
    mode: development
    workspace:
      host: ${var.databricks_host_dev}
    variables:
      catalog: mlops_dev
      schema: marvel_characters
      environment: dev
    run_as:
      service_principal_name: ${var.sp_name_dev}

  aut:
    mode: development
    workspace:
      host: ${var.databricks_host_aut}
    variables:
      catalog: mlops_aut
      schema: marvel_characters
      environment: aut
    run_as:
      service_principal_name: ${var.sp_name_aut}

  prod:
    mode: production
    workspace:
      host: ${var.databricks_host_prod}
    variables:
      catalog: mlops_prod
      schema: marvel_characters
      environment: prod
    run_as:
      service_principal_name: ${var.sp_name_prod}

artifacts:
  default:
    type: whl
    build: uv build
    path: dist/*.whl
```

---

#### 4.2 Monitoring-Specific Workflow Definition
**File**: `resources/workflows/serving_deployment.yml` (new)

```yaml
resources:
  jobs:
    serving_and_monitoring_deployment:
      name: "Marvel Characters - Serving & Monitoring"

      tasks:
        - task_key: deploy_serving_endpoint
          job_cluster_key: main_cluster
          python_wheel_task:
            package_name: marvel_characters
            entry_point: deploy_serving
            parameters:
              - "--catalog=${var.catalog}"
              - "--schema=${var.schema}"
              - "--environment=${var.environment}"
              - "--git-sha=${var.git_sha}"
          libraries:
            - whl: ../dist/*.whl

        - task_key: validate_inference_table
          depends_on:
            - task_key: deploy_serving_endpoint
          job_cluster_key: main_cluster
          python_wheel_task:
            package_name: marvel_characters
            entry_point: validate_inference_table
            parameters:
              - "--catalog=${var.catalog}"
              - "--schema=${var.schema}"
          libraries:
            - whl: ../dist/*.whl

        - task_key: setup_lakehouse_monitoring
          depends_on:
            - task_key: validate_inference_table
          job_cluster_key: main_cluster
          python_wheel_task:
            package_name: marvel_characters
            entry_point: setup_monitoring
            parameters:
              - "--catalog=${var.catalog}"
              - "--schema=${var.schema}"
              - "--environment=${var.environment}"
          libraries:
            - whl: ../dist/*.whl

      job_clusters:
        - job_cluster_key: main_cluster
          new_cluster:
            spark_version: "15.4.x-scala2.12"
            node_type_id: "Standard_DS3_v2"
            num_workers: 2
            spark_conf:
              spark.databricks.delta.preview.enabled: "true"
```

---

### Phase 5: Implementation Scripts

#### 5.1 Serving Deployment Script
**File**: `scripts/deploy_serving.py` (new)

**Purpose**: Deploy model serving endpoint with inference table enabled

**Key Logic**:
```python
from marvel_characters.serving.model_serving_setup import ServingSetup
from marvel_characters.model_registry import ModelRegistry
from marvel_characters.monitoring.config import load_config

def main(catalog, schema, environment, git_sha):
    # Load configuration
    config = load_config(environment)

    # Get latest model version
    registry = ModelRegistry(catalog, schema)
    model_version = registry.get_latest_model_version(
        model_name="marvel_character_model_basic",
        alias="latest-model"
    )

    # Create serving endpoint with inference table
    serving_setup = ServingSetup(config.serving)
    endpoint_name = serving_setup.create_endpoint(
        model_name=f"{catalog}.{schema}.marvel_character_model_basic",
        model_version=model_version,
        inference_table_config={
            "catalog": catalog,
            "schema": schema,
            "table_name": config.inference_table.name,
            "enabled": True
        },
        tags={
            "git_sha": git_sha,
            "environment": environment
        }
    )

    # Wait for endpoint to be ready
    serving_setup.wait_for_endpoint_ready(endpoint_name, timeout=600)

    print(f"Serving endpoint deployed: {endpoint_name}")
    print(f"Model version: {model_version}")
    print(f"Inference table: {catalog}.{schema}.{config.inference_table.name}")
```

---

#### 5.2 Monitoring Setup Script
**File**: `scripts/setup_monitoring.py` (new)

**Purpose**: Create Lakehouse monitor on inference table

**Key Logic**:
```python
from marvel_characters.monitoring.lakehouse_monitor import LakehouseMonitor
from marvel_characters.monitoring.config import load_config

def main(catalog, schema, environment):
    # Load configuration
    config = load_config(environment)

    # Create monitor
    monitor = LakehouseMonitor(catalog, schema)

    monitor_name = monitor.create_monitor(
        inference_table=f"{catalog}.{schema}.{config.inference_table.name}",
        output_schema=f"{catalog}.{config.monitoring.output_schema}",
        profile_type="InferenceLog",
        granularities=config.monitoring.granularities,
        baseline_table=f"{catalog}.{schema}.train_set",  # Optional
        problem_type="classification",
        prediction_col="prediction",
        label_col=None,  # If labels available later
        timestamp_col="timestamp"
    )

    # Refresh monitor to generate initial metrics
    monitor.refresh_monitor(monitor_name)

    # Set up alerts
    for alert_config in config.monitoring.alerts:
        monitor.create_alert(
            monitor_name=monitor_name,
            metric=alert_config.metric,
            threshold=alert_config.threshold,
            notification=alert_config.notification
        )

    print(f"Lakehouse monitor created: {monitor_name}")
    print(f"Dashboard available at: {monitor.get_dashboard_url(monitor_name)}")
```

---

### Phase 6: Scalability and Reusability

#### 6.1 Generic Monitoring Framework
**File**: `src/marvel_characters/monitoring/framework.py` (new)

**Purpose**: Abstract monitoring framework for any model

**Key Classes**:

```python
class MonitoringFramework:
    """Generic monitoring framework for ML models"""

    def __init__(self, catalog, schema, model_name):
        self.catalog = catalog
        self.schema = schema
        self.model_name = model_name
        self.registry = ModelRegistry(catalog, schema)
        self.serving_setup = ServingSetup()
        self.monitor = LakehouseMonitor(catalog, schema)

    def deploy_with_monitoring(
        self,
        model_version,
        serving_config,
        inference_table_config,
        monitoring_config
    ):
        """End-to-end deployment with monitoring"""

        # 1. Deploy serving endpoint
        endpoint_name = self.serving_setup.create_endpoint(
            model_name=f"{self.catalog}.{self.schema}.{self.model_name}",
            model_version=model_version,
            inference_table_config=inference_table_config,
            **serving_config
        )

        # 2. Wait for endpoint ready
        self.serving_setup.wait_for_endpoint_ready(endpoint_name)

        # 3. Validate inference table
        self._validate_inference_table(inference_table_config["table_name"])

        # 4. Create Lakehouse monitor
        monitor_name = self.monitor.create_monitor(
            inference_table=f"{self.catalog}.{self.schema}.{inference_table_config['table_name']}",
            **monitoring_config
        )

        # 5. Set up alerts
        self._setup_alerts(monitor_name, monitoring_config.get("alerts", []))

        return {
            "endpoint_name": endpoint_name,
            "monitor_name": monitor_name,
            "inference_table": f"{self.catalog}.{self.schema}.{inference_table_config['table_name']}"
        }
```

**Usage for Future Models**:
```python
# Easy to adapt for new models
framework = MonitoringFramework(
    catalog="mlops_prod",
    schema="new_model_schema",
    model_name="new_classification_model"
)

deployment = framework.deploy_with_monitoring(
    model_version="1",
    serving_config={...},
    inference_table_config={...},
    monitoring_config={...}
)
```

---

#### 6.2 Configuration Templates
**Directory**: `config_templates/`

**Purpose**: Template configurations for different model types

**Templates**:
- `classification_model.yml`: Binary/multiclass classification
- `regression_model.yml`: Regression models
- `forecasting_model.yml`: Time series forecasting
- `ranking_model.yml`: Ranking/recommendation models

**Example** - `config_templates/classification_model.yml`:
```yaml
model:
  type: classification
  problem_type: binary  # or multiclass

serving:
  workload_size: Small
  scale_to_zero: true
  min_replicas: 1
  max_replicas: 5

inference_table:
  enabled: true
  retention_days: 90

monitoring:
  profile_type: InferenceLog
  granularities:
    - "1 day"
  refresh_schedule: "daily"
  drift_detection:
    enabled: true
    baseline: training_data
  data_quality:
    enabled: true
    rules:
      - null_check
      - type_check
      - range_check
  alerts:
    - metric: prediction_drift
      threshold: 0.1
      severity: warning
      notification_type: email
    - metric: data_quality_score
      threshold: 0.8
      severity: critical
      notification_type: email
```

---

## File Structure Summary

### New Files to Create

```
marvel-characters/
├── src/marvel_characters/
│   ├── mlflow_tracking.py              # MLFlow logging utilities (NEW)
│   ├── model_registry.py               # Unity Catalog management (NEW)
│   ├── serving/
│   │   └── model_serving_setup.py      # Serving + inference table (NEW)
│   └── monitoring/
│       ├── __init__.py                 (NEW)
│       ├── config.py                   # Monitoring configuration (NEW)
│       ├── lakehouse_monitor.py        # Monitor creation/management (NEW)
│       └── framework.py                # Generic monitoring framework (NEW)
├── scripts/
│   ├── deploy_serving.py               # Deploy serving endpoint (NEW)
│   ├── validate_inference_table.py     # Validate inference logging (NEW)
│   ├── setup_monitoring.py             # Create Lakehouse monitor (NEW)
│   ├── refresh_all_monitors.py         # Refresh monitoring (NEW)
│   ├── analyze_drift.py                # Drift analysis (NEW)
│   └── send_alerts.py                  # Alerting (NEW)
├── resources/workflows/
│   ├── ml_training.yml                 # Training workflow (UPDATE)
│   ├── serving_deployment.yml          # Serving deployment workflow (NEW)
│   └── monitoring_refresh.yml          # Scheduled monitoring refresh (NEW)
├── config_templates/
│   ├── classification_model.yml        # Classification template (NEW)
│   ├── regression_model.yml            # Regression template (NEW)
│   └── forecasting_model.yml           # Time series template (NEW)
├── config/
│   └── monitoring_config.yml           # Monitoring configuration (NEW)
├── azure-pipelines.yml                 # Azure DevOps pipeline (NEW)
└── databricks.yml                      # Asset bundle config (UPDATE)
```

### Files to Update

- `databricks.yml`: Add new workflows
- `resources/model_deployment.yml`: Rename to `ml_training.yml`, update structure
- `src/marvel_characters/models/basic_model.py`: Enhance MLFlow logging
- `pyproject.toml`: Add new entry points for scripts

---

## Key Implementation Details

### 1. Inference Table Enablement

**Critical**: Inference tables must be enabled **at endpoint creation time**

```python
# When creating serving endpoint
endpoint_config = {
    "name": endpoint_name,
    "config": {
        "served_models": [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }],
        # CRITICAL: Enable inference table here
        "inference_table_config": {
            "catalog_name": catalog,
            "schema_name": schema,
            "table_name_prefix": "inference_logs",
            "enabled": True
        }
    }
}

# Use Databricks SDK
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.serving_endpoints.create(**endpoint_config)
```

**Schema**: Inference table automatically includes:
- `request_id`: Unique identifier
- `timestamp`: Request time
- `request`: JSON payload (inputs)
- `response`: JSON payload (predictions)
- `status_code`: HTTP response code
- `execution_duration_ms`: Latency

---

### 2. Lakehouse Monitor Creation

**Use Databricks SDK**:
```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog

w = WorkspaceClient()

# Create monitor
monitor = w.quality_monitors.create(
    table_name=f"{catalog}.{schema}.inference_logs",
    assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{model_name}",
    output_schema_name=f"{catalog}.monitoring",
    inference_log=MonitorInferenceLog(
        problem_type="classification",
        prediction_col="response",
        timestamp_col="timestamp",
        granularities=["30 minutes", "1 day"],
        model_id_col="request_id"
    )
)

# Refresh monitor
w.quality_monitors.run_refresh(
    table_name=f"{catalog}.{schema}.inference_logs"
)
```

---

### 3. Azure DevOps Integration

**Key Steps**:
1. **Create Service Principal** in Azure AD for Databricks access
   - Go to Azure Portal → Azure Active Directory → App registrations
   - Create new registration for Databricks CI/CD
   - Note: Application (client) ID, Directory (tenant) ID
   - Create client secret

2. **Grant Service Principal access to Databricks workspaces**
   - Add Service Principal as admin in each Databricks workspace
   - Grant Unity Catalog privileges (CREATE, USAGE on catalogs/schemas)

3. **Create Azure DevOps Service Connection**
   - Project Settings → Service connections → New service connection
   - Type: Azure Resource Manager
   - Authentication: Service Principal
   - Scope: Subscription level
   - Name: `Azure-ServiceConnection`

4. **Create Variable Groups** in Azure DevOps
   - Library → Variable groups
   - Create: databricks-secrets-dev, databricks-secrets-aut, databricks-secrets-prod
   - Add DATABRICKS_HOST_* variables for each environment

5. **Configure Environments** in Azure DevOps
   - Pipelines → Environments → Create: dev, aut, prod
   - Add approvals for aut and prod environments

6. **Create Pipeline** using `azure-pipelines.yml`

**Authentication Method (Service Principal)**:
- Uses Azure AD token obtained via AzureCLI@2 task
- Token resource ID for Databricks: `2ff814a6-3304-4ab8-85cb-cd0e6f879c1d`
- More secure than PATs, supports automatic token rotation

---

## Success Criteria

✅ **MLFlow Tracking**: All training runs log params, metrics, artifacts, and datasets

✅ **Model Registry**: Models registered in Unity Catalog with versions and aliases

✅ **Model Loading**: Models can be loaded by version or alias for inference

✅ **Model Serving**: Endpoints deployed with configured workload and scaling

✅ **Inference Tables**: Inference logs captured in Delta tables automatically

✅ **Lakehouse Monitoring**: Monitors created with drift detection and dashboards

✅ **Azure DevOps CI/CD**: Pipeline deploys to dev/prod based on branch

✅ **Scalability**: Framework easily adaptable for new models with templates

✅ **Automation**: End-to-end deployment with single pipeline run

---

## Next Steps After Approval

1. Create new monitoring module files
2. Implement MLFlow tracking utilities
3. Implement model registry management
4. Implement serving setup with inference tables
5. Implement Lakehouse monitor creation
6. Create generic monitoring framework
7. Create Azure DevOps pipeline configuration
8. Update Databricks Asset Bundle configuration
9. Create monitoring configuration templates
10. Update existing scripts with enhanced logging
11. Create documentation and usage examples
12. Test end-to-end in dev environment
13. Deploy to production

---

## Configuration Decisions (Confirmed)

1. **Authentication**: Service Principal authentication for Azure DevOps ✅
   - Will use Azure AD Service Principal with OAuth tokens
   - Requires setup of Service Principal in Azure AD
   - More secure than Personal Access Tokens

2. **Alerting**: Email notifications ✅
   - Will configure SMTP or Databricks email notifications
   - Alert on drift threshold breaches and data quality issues

3. **Baseline**: Use training data as baseline for drift detection ✅
   - Training dataset will serve as reference for drift comparison
   - Monitor will compare inference data distributions against training data

4. **Monitor Granularity**: 1-day granularity ✅
   - Daily aggregation for monitoring metrics
   - Balances freshness with computational cost

5. **Model Comparison**: Keep existing F1-score comparison logic for model promotion ✅
   - Only register and deploy models that improve F1-score
   - Maintains continuous improvement pattern

6. **Environments**: Three environments - dev, aut (acceptance/UAT), prod (production) ✅
   - **dev**: Development environment for testing
   - **aut**: Acceptance/UAT environment for validation
   - **prod**: Production environment for live models

7. **Monitoring Refresh**: Daily refresh ✅
   - Scheduled daily monitoring refresh workflow
   - Runs after business hours to minimize impact

8. **Data Retention**: 90 days for all environments ✅
   - **dev**: 90 days
   - **aut**: 90 days
   - **prod**: 90 days
   - Balances storage costs with analysis needs
   - Can be adjusted per environment if needed in the future
