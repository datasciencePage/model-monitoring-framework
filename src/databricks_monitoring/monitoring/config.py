"""Configuration management for monitoring framework."""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class AlertConfig(BaseModel):
    """Alert configuration."""

    metric: str = Field(..., description="Metric to monitor (e.g., prediction_drift)")
    threshold: float = Field(..., description="Threshold value for alerting")
    severity: str = Field(default="warning", description="Alert severity (warning, critical)")
    notification_type: str = Field(default="email", description="Notification channel")
    recipients: List[str] = Field(default_factory=list, description="List of recipients")


class Inference

TableConfig(BaseModel):
    """Inference table configuration."""

    name: str = Field(..., description="Inference table name")
    retention_days: int = Field(default=90, description="Data retention period in days")


class ServingConfig(BaseModel):
    """Model serving configuration."""

    endpoint_name: str = Field(..., description="Serving endpoint name")
    workload_size: str = Field(default="Small", description="Workload size (Small, Medium, Large)")
    scale_to_zero: bool = Field(default=True, description="Enable scale to zero")
    min_replicas: int = Field(default=1, description="Minimum number of replicas")
    max_replicas: int = Field(default=5, description="Maximum number of replicas")


class LakehouseMonitorConfig(BaseModel):
    """Lakehouse monitoring configuration."""

    output_schema: str = Field(..., description="Output schema for monitoring tables")
    granularities: List[str] = Field(default_factory=lambda: ["1 day"], description="Monitor granularities")
    refresh_schedule: str = Field(default="0 0 * * *", description="Cron schedule for refresh")
    alerts: List[AlertConfig] = Field(default_factory=list, description="Alert configurations")


class MonitoringEnvironmentConfig(BaseModel):
    """Environment-specific monitoring configuration."""

    catalog: str = Field(..., description="Catalog name")
    schema: str = Field(..., description="Schema name")
    serving: ServingConfig = Field(..., description="Serving configuration")
    inference_table: InferenceTableConfig = Field(..., description="Inference table configuration")
    monitoring: LakehouseMonitorConfig = Field(..., description="Lakehouse monitoring configuration")


class MonitoringConfig(BaseModel):
    """Root monitoring configuration."""

    environments: dict[str, MonitoringEnvironmentConfig] = Field(..., description="Environment configurations")


def load_config(environment: str, config_path: Optional[str] = None) -> MonitoringEnvironmentConfig:
    """Load monitoring configuration for a specific environment.

    Args:
        environment: Environment name (dev, aut, prod)
        config_path: Optional path to configuration file. If not specified,
                    uses config/monitoring_config.yml in project root.

    Returns:
        MonitoringEnvironmentConfig: Environment-specific configuration

    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If environment not found in configuration
    """
    if config_path is None:
        # Default to config/monitoring_config.yml in project root
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config" / "monitoring_config.yml"

    config_file = Path(config_path)
    if not config_file.exists():
        msg = f"Configuration file not found: {config_file}"
        raise FileNotFoundError(msg)

    with config_file.open() as f:
        config_data = yaml.safe_load(f)

    # Parse configuration
    monitoring_config = MonitoringConfig(**config_data)

    # Get environment-specific configuration
    if environment not in monitoring_config.environments:
        msg = f"Environment '{environment}' not found in configuration"
        raise ValueError(msg)

    return monitoring_config.environments[environment]


def save_config(config: MonitoringConfig, config_path: Optional[str] = None) -> None:
    """Save monitoring configuration to file.

    Args:
        config: Monitoring configuration to save
        config_path: Optional path to configuration file. If not specified,
                    uses config/monitoring_config.yml in project root.
    """
    if config_path is None:
        # Default to config/monitoring_config.yml in project root
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config" / "monitoring_config.yml"

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with config_file.open("w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
