# Databricks Lakehouse Monitoring Framework

A production-ready, generic monitoring framework for ML models deployed on Databricks Lakehouse Platform with Unity Catalog integration.

## Overview

This framework provides end-to-end MLOps capabilities for deploying, monitoring, and maintaining machine learning models in production. It automates the entire lifecycle from model serving deployment to drift detection and alerting.

### Key Features

- **MLFlow Integration**: Automatic experiment tracking, model registration, and lineage
- **Unity Catalog Support**: Native integration with Databricks Unity Catalog for model governance
- **Model Serving**: Automated deployment to Databricks serving endpoints with autoscaling
- **Inference Logging**: Automatic capture of predictions and features to Delta tables
- **Drift Detection**: Real-time monitoring of prediction drift and feature drift
- **Data Quality Monitoring**: Automated checks for data quality issues
- **Alert System**: Configurable email alerts when metrics exceed thresholds
- **Multi-Environment**: Separate configurations for dev, acceptance, and production
- **CI/CD Ready**: Complete Azure DevOps pipeline for automated deployments

## Quick Start

### Prerequisites

- Python 3.12+
- Databricks workspace with Unity Catalog enabled
- Azure subscription (for CI/CD)
- Access to a Databricks catalog and schema

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd model-monitoring-framework

# Install dependencies using uv
pip install uv
uv sync

# Or install the package
pip install -e .
```

### Basic Usage

#### 1. Deploy Model with Monitoring (Python API)

```python
from databricks_monitoring.monitoring.framework import MonitoringFramework

# Initialize framework
framework = MonitoringFramework(
    catalog="mlops_prod",
    schema="fraud_detection",
    model_name="fraud_model"
)

# Deploy model with full monitoring
deployment = framework.deploy_with_monitoring(
    model_version="3",
    serving_config={
        "endpoint_name": "fraud-detection-serving",
        "workload_size": "Medium",
        "scale_to_zero": False
    },
    inference_table_config={
        "name": "inference_logs",
        "enabled": True
    },
    monitoring_config={
        "output_schema": "monitoring",
        "granularities": ["1 day"],
        "alerts": [{
            "metric": "prediction_drift",
            "threshold": 0.1,
            "severity": "critical",
            "recipients": ["ml-team@company.com"]
        }]
    }
)

print(f"Endpoint: {deployment['endpoint_name']}")
print(f"Monitor: {deployment['monitor_name']}")
print(f"Dashboard: {deployment['dashboard_url']}")
```

#### 2. Deploy Using CLI Commands

```bash
# Set environment variables
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"

# Deploy serving endpoint
deploy_serving --catalog mlops_prod --schema fraud_detection --environment prod

# Validate inference table is receiving data
validate_inference_table --catalog mlops_prod --schema fraud_detection

# Setup lakehouse monitoring
setup_monitoring --catalog mlops_prod --schema fraud_detection --environment prod

# Analyze drift metrics
analyze_drift --catalog mlops_prod --schema fraud_detection --days 7

# Send alerts for threshold violations
send_alerts --catalog mlops_prod --schema fraud_detection --environment prod
```

#### 3. Deploy Using Databricks Bundle

```bash
# Configure Databricks CLI
databricks configure --host https://your-workspace.databricks.com --token

# Deploy to development
databricks bundle deploy --target dev

# Run the deployment workflow
databricks bundle run serving_and_monitoring_deployment --target dev

# Deploy to production
databricks bundle deploy --target prod
```

## Configuration

### Environment Configuration

Edit [config/monitoring_config.yml](config/monitoring_config.yml) to configure environments:

```yaml
environments:
  prod:
    catalog: mlops_prod
    schema: fraud_detection

    serving:
      endpoint_name: fraud-detection-prod
      workload_size: Medium
      scale_to_zero: false
      min_replicas: 2
      max_replicas: 10

    inference_table:
      name: inference_logs
      retention_days: 90

    monitoring:
      output_schema: monitoring
      granularities:
        - "1 day"
      refresh_schedule: "0 0 * * *"  # Daily at midnight

      alerts:
        - metric: prediction_drift
          threshold: 0.05
          severity: critical
          recipients:
            - ml-team@company.com
```

### Model Templates

The framework includes templates for common model types:

- [classification_model.yml](config_templates/classification_model.yml) - Binary/multiclass classification
- [regression_model.yml](config_templates/regression_model.yml) - Regression models
- [forecasting_model.yml](config_templates/forecasting_model.yml) - Time series forecasting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Scripts Layer                     │
│  (deploy_serving, setup_monitoring, analyze_drift...)   │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│              Orchestration Layer                         │
│         MonitoringFramework (framework.py)               │
│    Coordinates end-to-end deployment + monitoring        │
└────────┬────────────┬────────────┬──────────────────────┘
         │            │            │
    ┌────▼───┐   ┌───▼────┐  ┌────▼─────────┐
    │ Model  │   │Serving │  │ Lakehouse    │
    │Registry│   │ Setup  │  │  Monitor     │
    └────┬───┘   └───┬────┘  └────┬─────────┘
         │           │            │
    ┌────▼───────────▼────────────▼─────────┐
    │       Databricks SDK / MLFlow         │
    └───────────────────────────────────────┘
```

## Available Commands

### Deployment Commands
- `deploy_serving` - Deploy model to serving endpoint with inference tables
- `validate_inference_table` - Validate inference logging is working
- `setup_monitoring` - Create lakehouse monitor with drift detection

### Monitoring Commands
- `refresh_all_monitors` - Refresh monitoring metrics for all monitors
- `analyze_drift` - Analyze prediction and feature drift
- `send_alerts` - Check thresholds and send email alerts

## CI/CD Integration

The framework includes a complete Azure DevOps pipeline with 5 stages:

1. **Build**: Create wheel package
2. **Test**: Run tests and linting
3. **DeployDev**: Deploy to development (automatic on develop branch)
4. **DeployAut**: Deploy to acceptance (automatic after dev)
5. **DeployProd**: Deploy to production (automatic on main branch)

See [azure-pipelines.yml](azure-pipelines.yml) for configuration.

## Project Structure

```
model-monitoring-framework/
├── src/databricks_monitoring/
│   ├── mlflow_tracking.py          # MLFlow experiment tracking
│   ├── model_registry.py           # Unity Catalog model registry
│   ├── monitoring/
│   │   ├── config.py               # Configuration management
│   │   ├── lakehouse_monitor.py    # Lakehouse monitoring
│   │   └── framework.py            # End-to-end orchestration
│   ├── serving/
│   │   └── model_serving_setup.py  # Model serving deployment
│   └── scripts/                    # CLI scripts
├── config/
│   ├── monitoring_config.yml       # Environment configurations
│   └── config_templates/           # Model type templates
├── resources/workflows/            # Databricks workflows
├── tests/                          # Test suite (to be implemented)
└── docs/                           # Additional documentation
```

## Documentation

- [Framework Structure](FRAMEWORK_STRUCTURE.md) - Detailed code organization
- [Monitoring Design](MONITORING_DESIGN.md) - Architecture and design decisions
- [Setup Guide](MONITORING_SETUP.md) - Step-by-step deployment instructions
- [Monitoring README](MONITORING_README.md) - Monitoring features and examples

## Requirements

- Python 3.12+
- Databricks workspace with Unity Catalog
- MLFlow 3.1.1+
- Databricks SDK 0.55.0+
- Pydantic 2.11.7+

## Development

### Running Tests

```bash
# Install test dependencies
uv sync --extra test

# Run tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=term

# Run linting
uv run ruff check src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Known Issues

⚠️ **Note**: This framework is under active development. Known issues:

1. Inference table auto-capture requires SDK update (line 128-136 in model_serving_setup.py)
2. Alert creation API integration pending (lakehouse_monitor.py)
3. Test suite in progress - current coverage: 0%

See the [review plan](.claude/plans/vectorized-tumbling-moore.md) for full details.

## License

[Add your license information here]

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: See docs/ folder
- Team Contact: [your-team-email]
