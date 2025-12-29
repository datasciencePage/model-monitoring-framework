# Generic Databricks Lakehouse Monitoring Framework

## Complete Directory Structure

```
databricks-lakehouse-monitoring/
├── src/                                # NOTE: Update package name in pyproject.toml
│   ├── __init__.py
│   ├── mlflow_tracking.py              # ✅ Generic MLFlow utilities
│   ├── model_registry.py               # ✅ Generic Unity Catalog management
│   │
│   ├── monitoring/                     # ✅ Generic monitoring framework
│   │   ├── __init__.py
│   │   ├── config.py                   # Pydantic configuration models
│   │   ├── framework.py                # End-to-end monitoring orchestration
│   │   └── lakehouse_monitor.py        # Lakehouse monitor creation/management
│   │
│   ├── serving/                        # ✅ Generic serving setup
│   │   └── model_serving_setup.py      # Model serving with inference tables
│   │
│   └── scripts/                        # ✅ Production deployment scripts
│       ├── __init__.py
│       ├── deploy_serving.py           # CLI: deploy_serving
│       ├── validate_inference_table.py # CLI: validate_inference_table
│       ├── setup_monitoring.py         # CLI: setup_monitoring
│       ├── refresh_all_monitors.py     # CLI: refresh_all_monitors
│       ├── analyze_drift.py            # CLI: analyze_drift
│       └── send_alerts.py              # CLI: send_alerts
│
├── resources/
│   └── workflows/                      # ✅ Databricks workflows
│       ├── serving_deployment.yml      # Deploy serving + monitoring
│       └── monitoring_refresh.yml      # Scheduled monitoring refresh
│
├── config/
│   └── monitoring_config.yml           # ✅ Environment configs (dev/aut/prod)
│
├── config_templates/                   # ✅ Model type templates
│   ├── classification_model.yml        # Classification model config
│   ├── regression_model.yml            # Regression model config
│   └── forecasting_model.yml           # Time series model config
│
├── tests/
│   ├── __init__.py
│   └── conftest.py                     # Test fixtures
│
├── azure-pipelines.yml                 # ✅ Azure DevOps CI/CD pipeline
├── databricks.yml                      # ✅ Databricks Asset Bundle config
├── pyproject.toml                      # ✅ Package configuration
├── .pre-commit-config.yaml             # Code quality checks
│
└── Documentation/
    ├── README.md                       # Main documentation
    ├── MONITORING_README.md            # Quick start guide
    ├── MONITORING_DESIGN.md            # Design & architecture
    ├── MONITORING_SETUP.md             # Setup instructions
    ├── CLEANUP_SUMMARY.md              # Cleanup history
    └── FRAMEWORK_STRUCTURE.md          # This file
```

---

## Core Components

### 1. MLFlow Integration (`mlflow_tracking.py`)
- Log training datasets with versions
- Log model parameters and metrics
- Log model artifacts with signatures
- Track experiment lineage

### 2. Model Registry (`model_registry.py`)
- Register models to Unity Catalog
- Get latest model versions
- Load models for inference
- Manage model aliases (latest-model, champion, etc.)
- Model metadata management

### 3. Monitoring Framework (`monitoring/`)

#### `config.py`
- Pydantic models for type-safe configuration
- Environment-specific settings (dev/aut/prod)
- Serving, inference table, and monitoring configs

#### `lakehouse_monitor.py`
- Create Lakehouse monitors on inference tables
- Configure drift detection (prediction + features)
- Set up data quality checks
- Manage monitor refresh schedules
- Create alert rules

#### `framework.py`
- **End-to-end orchestration** of deployment + monitoring
- Single entry point: `MonitoringFramework.deploy_with_monitoring()`
- Coordinates serving, inference tables, and monitoring
- Generic for any model type

### 4. Model Serving (`serving/model_serving_setup.py`)
- Create serving endpoints
- **Enable inference tables at creation time** (critical!)
- Configure autoscaling and workload size
- Update model versions
- Health checks

### 5. Production Scripts (`scripts/`)

All scripts are CLI-enabled via `pyproject.toml` entry points:

| Script | CLI Command | Purpose |
|--------|-------------|---------|
| `deploy_serving.py` | `deploy_serving` | Deploy model serving endpoint |
| `validate_inference_table.py` | `validate_inference_table` | Validate inference logging |
| `setup_monitoring.py` | `setup_monitoring` | Create Lakehouse monitor |
| `refresh_all_monitors.py` | `refresh_all_monitors` | Refresh monitoring metrics |
| `analyze_drift.py` | `analyze_drift` | Analyze prediction/feature drift |
| `send_alerts.py` | `send_alerts` | Send email alerts |

---

## Workflows

### Serving Deployment Workflow (`serving_deployment.yml`)

**Trigger**: On-demand or triggered by training workflow

**Tasks**:
1. **deploy_serving_endpoint**
   - Get latest model from Unity Catalog
   - Create/update serving endpoint
   - Enable inference table
   - Configure workload size

2. **validate_inference_table**
   - Send test requests
   - Verify logging is working
   - Check data quality

3. **setup_lakehouse_monitoring**
   - Create Lakehouse monitor
   - Configure drift detection
   - Set up alerts

### Monitoring Refresh Workflow (`monitoring_refresh.yml`)

**Trigger**: Scheduled daily (configurable per environment)

**Tasks**:
1. **refresh_all_monitors**
   - List active monitors
   - Refresh each monitor
   - Collect metrics

2. **analyze_drift**
   - Analyze prediction drift
   - Analyze feature drift
   - Generate drift report

3. **send_alerts**
   - Check drift thresholds
   - Check data quality
   - Send notifications

---

## Configuration

### Environment Configuration (`config/monitoring_config.yml`)

```yaml
environments:
  dev:
    catalog: mlops_dev           # UPDATE: Your dev catalog
    schema: your_model_schema    # UPDATE: Your schema
    serving:
      endpoint_name: your-model-serving-dev
      workload_size: Small
      scale_to_zero: true
    inference_table:
      name: inference_logs
      retention_days: 90
    monitoring:
      output_schema: monitoring
      granularities: ["1 day"]
      alerts:
        - metric: prediction_drift
          threshold: 0.1
          recipients: [your-team@example.com]
  
  aut: # ... similar structure
  prod: # ... similar structure
```

### Databricks Bundle (`databricks.yml`)

```yaml
bundle:
  name: your-project-name        # UPDATE: Your project name

targets:
  dev:
    workspace:
      host: <your-workspace-url>  # UPDATE: Your workspace URL
      profile: your-profile        # UPDATE: Your profile
    variables:
      catalog: mlops_dev           # UPDATE: Your catalog
      schema: your_model_schema    # UPDATE: Your schema
```

### Azure DevOps Pipeline (`azure-pipelines.yml`)

Multi-stage pipeline:
- **Build**: Build wheel package with UV
- **Test**: Run pytest and ruff linting
- **DeployDev**: Deploy to dev (develop branch)
- **DeployAut**: Deploy to aut (develop branch)
- **DeployProd**: Deploy to prod (main branch)

Uses Service Principal authentication via Azure AD.

---

## Usage Examples

### 1. Deploy with Framework (Python)

```python
from your_package.monitoring.framework import MonitoringFramework

framework = MonitoringFramework(
    catalog="mlops_prod",
    schema="my_model",
    model_name="my_classification_model"
)

deployment = framework.deploy_with_monitoring(
    model_version="1",
    serving_config={
        "endpoint_name": "my-model-serving",
        "workload_size": "Medium",
        "scale_to_zero": False
    },
    inference_table_config={
        "name": "inference_logs",
        "enabled": True,
        "retention_days": 90
    },
    monitoring_config={
        "granularities": ["1 day"],
        "baseline_table": "training_data",
        "alerts": [...]
    }
)

print(f"Dashboard: {deployment['dashboard_url']}")
```

### 2. Deploy with CLI

```bash
# Deploy serving endpoint
deploy_serving --catalog mlops_prod --schema my_model --environment prod --git-sha abc123

# Setup monitoring
setup_monitoring --catalog mlops_prod --schema my_model --environment prod

# Analyze drift
analyze_drift --catalog mlops_prod --schema my_model --days 7
```

### 3. Deploy with Databricks Bundle

```bash
# Deploy to dev
databricks bundle deploy --target dev

# Run serving deployment workflow
databricks bundle run serving_and_monitoring_deployment --target dev

# Deploy to production
databricks bundle deploy --target prod
```

---

## Customization Guide

### To Adapt for Your ML Model:

1. **Update package name in `pyproject.toml`**:
   ```toml
   [project]
   name = "your-package-name"  # Change from "databricks_monitoring"
   ```

2. **Update configuration files**:
   - Search for `# UPDATE:` comments
   - Replace placeholders with your values

3. **Configure Azure DevOps**:
   - Create Service Principal
   - Set up variable groups
   - Configure environments

4. **Implement your model training**:
   - Use `mlflow_tracking.py` for logging
   - Use `model_registry.py` for registration
   - Follow MLflow best practices

5. **Deploy with framework**:
   - Use generic monitoring framework
   - No code changes needed!

---

## Key Features

✅ **Service Principal Auth** - Secure Azure AD authentication
✅ **Inference Tables** - Automatic request/response logging
✅ **Lakehouse Monitoring** - Drift detection & data quality
✅ **Email Alerting** - Configurable thresholds
✅ **Multi-Environment** - dev/aut/prod support
✅ **Generic Framework** - Works with any model type
✅ **CI/CD Ready** - Azure DevOps pipeline included
✅ **Fully Documented** - Comprehensive guides

---

## Requirements

- Python 3.12+
- Databricks workspace with Unity Catalog
- Azure subscription (for Azure DevOps CI/CD)
- UV package manager
- Databricks CLI

---

## Getting Started

1. Read [MONITORING_DESIGN.md](MONITORING_DESIGN.md) for architecture
2. Follow [MONITORING_SETUP.md](MONITORING_SETUP.md) for setup
3. Use [MONITORING_README.md](MONITORING_README.md) for quick start
4. Customize configuration files (search for `# UPDATE:`)
5. Deploy to dev and test
6. Deploy to production via CI/CD

---

**Framework Version**: 1.0.0
**Status**: ✅ Production-Ready & Fully Generic
**Last Updated**: December 26, 2024
