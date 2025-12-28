"""Tests for configuration management module."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from databricks_monitoring.monitoring.config import (
    AlertConfig,
    InferenceTableConfig,
    ServingConfig,
    LakehouseMonitorConfig,
    MonitoringEnvironmentConfig,
    MonitoringConfig,
    load_config,
)


class TestAlertConfig:
    """Tests for AlertConfig model."""

    def test_alert_config_required_fields(self):
        """Test that AlertConfig requires metric and threshold."""
        with pytest.raises(ValidationError):
            AlertConfig()

    def test_alert_config_defaults(self):
        """Test that AlertConfig applies default values correctly."""
        config = AlertConfig(metric="drift", threshold=0.1)
        assert config.severity == "warning"
        assert config.notification_type == "email"
        assert config.recipients == []

    def test_alert_config_custom_values(self):
        """Test AlertConfig with custom values."""
        config = AlertConfig(
            metric="prediction_drift",
            threshold=0.05,
            severity="critical",
            notification_type="slack",
            recipients=["team@example.com"],
        )
        assert config.metric == "prediction_drift"
        assert config.threshold == 0.05
        assert config.severity == "critical"
        assert config.notification_type == "slack"
        assert config.recipients == ["team@example.com"]


class TestInferenceTableConfig:
    """Tests for InferenceTableConfig model."""

    def test_inference_table_config_required_fields(self):
        """Test that InferenceTableConfig requires name."""
        with pytest.raises(ValidationError):
            InferenceTableConfig()

    def test_inference_table_config_defaults(self):
        """Test that InferenceTableConfig applies default retention."""
        config = InferenceTableConfig(name="inference_logs")
        assert config.name == "inference_logs"
        assert config.retention_days == 90

    def test_inference_table_config_custom_retention(self):
        """Test InferenceTableConfig with custom retention."""
        config = InferenceTableConfig(name="inference_logs", retention_days=30)
        assert config.retention_days == 30


class TestServingConfig:
    """Tests for ServingConfig model."""

    def test_serving_config_required_fields(self):
        """Test that ServingConfig requires endpoint_name."""
        with pytest.raises(ValidationError):
            ServingConfig()

    def test_serving_config_defaults(self):
        """Test that ServingConfig applies default values."""
        config = ServingConfig(endpoint_name="test-endpoint")
        assert config.endpoint_name == "test-endpoint"
        assert config.workload_size == "Small"
        assert config.scale_to_zero is True
        assert config.min_replicas == 1
        assert config.max_replicas == 5

    def test_serving_config_production_values(self):
        """Test ServingConfig with production values."""
        config = ServingConfig(
            endpoint_name="prod-endpoint",
            workload_size="Medium",
            scale_to_zero=False,
            min_replicas=2,
            max_replicas=10,
        )
        assert config.workload_size == "Medium"
        assert config.scale_to_zero is False
        assert config.min_replicas == 2
        assert config.max_replicas == 10


class TestLakehouseMonitorConfig:
    """Tests for LakehouseMonitorConfig model."""

    def test_monitor_config_required_fields(self):
        """Test that LakehouseMonitorConfig requires output_schema."""
        with pytest.raises(ValidationError):
            LakehouseMonitorConfig()

    def test_monitor_config_defaults(self):
        """Test that LakehouseMonitorConfig applies defaults."""
        config = LakehouseMonitorConfig(output_schema="monitoring")
        assert config.output_schema == "monitoring"
        assert config.granularities == ["1 day"]
        assert config.refresh_schedule == "0 0 * * *"
        assert config.alerts == []

    def test_monitor_config_with_alerts(self):
        """Test LakehouseMonitorConfig with alert configurations."""
        alert = AlertConfig(metric="drift", threshold=0.1)
        config = LakehouseMonitorConfig(
            output_schema="monitoring",
            granularities=["1 hour", "1 day"],
            alerts=[alert],
        )
        assert len(config.alerts) == 1
        assert config.alerts[0].metric == "drift"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_dev_environment(self, sample_config_path):
        """Test loading dev environment configuration."""
        if not sample_config_path.exists():
            pytest.skip("Config file not found")

        config = load_config("dev", str(sample_config_path))
        assert isinstance(config, MonitoringEnvironmentConfig)
        assert config.catalog is not None
        assert config.schema is not None

    def test_load_config_invalid_environment(self, sample_config_path):
        """Test that load_config raises ValueError for invalid environment."""
        if not sample_config_path.exists():
            pytest.skip("Config file not found")

        with pytest.raises(ValueError, match="Environment 'invalid' not found"):
            load_config("invalid", str(sample_config_path))

    def test_load_config_missing_file(self):
        """Test that load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("dev", "/nonexistent/config.yml")


class TestMonitoringEnvironmentConfig:
    """Tests for MonitoringEnvironmentConfig model."""

    def test_complete_environment_config(self):
        """Test creating a complete environment configuration."""
        config = MonitoringEnvironmentConfig(
            catalog="test_catalog",
            schema="test_schema",
            serving=ServingConfig(endpoint_name="test-endpoint"),
            inference_table=InferenceTableConfig(name="inference_logs"),
            monitoring=LakehouseMonitorConfig(output_schema="monitoring"),
        )
        assert config.catalog == "test_catalog"
        assert config.schema == "test_schema"
        assert config.serving.endpoint_name == "test-endpoint"
        assert config.inference_table.name == "inference_logs"
        assert config.monitoring.output_schema == "monitoring"
