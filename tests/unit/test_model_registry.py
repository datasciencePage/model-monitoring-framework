"""Tests for model registry module."""

import pytest
from unittest.mock import Mock, patch

from databricks_monitoring.model_registry import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def model_registry(self, sample_catalog, sample_schema):
        """Create ModelRegistry instance for testing."""
        return ModelRegistry(sample_catalog, sample_schema)

    def test_initialization(self, model_registry, sample_catalog, sample_schema):
        """Test ModelRegistry initialization."""
        assert model_registry.catalog == sample_catalog
        assert model_registry.schema == sample_schema

    def test_get_full_model_name(self, model_registry, sample_catalog, sample_schema):
        """Test full model name construction."""
        model_name = "fraud_detection"
        full_name = model_registry._get_full_model_name(model_name)
        assert full_name == f"{sample_catalog}.{sample_schema}.{model_name}"

    @patch("databricks_monitoring.model_registry.MlflowClient")
    def test_get_latest_model_version_with_alias(
        self, mock_mlflow_client, model_registry
    ):
        """Test retrieving model version by alias."""
        # Setup mock
        mock_version = Mock()
        mock_version.version = "3"
        mock_mlflow_client.return_value.get_model_version_by_alias.return_value = (
            mock_version
        )

        # Test
        version = model_registry.get_latest_model_version(
            "test_model", alias="champion"
        )

        # Verify
        assert version == "3"
        mock_mlflow_client.return_value.get_model_version_by_alias.assert_called_once()

    @patch("databricks_monitoring.model_registry.MlflowClient")
    def test_get_latest_model_version_without_alias(
        self, mock_mlflow_client, model_registry
    ):
        """Test retrieving latest model version without alias."""
        # Setup mock
        mock_versions = [Mock(version="1"), Mock(version="2"), Mock(version="3")]
        mock_mlflow_client.return_value.search_model_versions.return_value = (
            mock_versions
        )

        # Test
        version = model_registry.get_latest_model_version("test_model")

        # Verify
        assert version == "3"

    @patch("databricks_monitoring.model_registry.MlflowClient")
    def test_set_model_alias(self, mock_mlflow_client, model_registry):
        """Test setting model alias."""
        # Test
        model_registry.set_model_alias("test_model", "2", "champion")

        # Verify
        mock_mlflow_client.return_value.set_registered_model_alias.assert_called_once()

    @patch("databricks_monitoring.model_registry.MlflowClient")
    def test_register_model(self, mock_mlflow_client, model_registry):
        """Test registering a model."""
        # Setup mock
        mock_version = Mock()
        mock_version.version = "1"
        mock_mlflow_client.return_value.create_model_version.return_value = (
            mock_version
        )

        # Test
        version = model_registry.register_model(
            model_uri="runs:/abc123/model",
            model_name="test_model",
            description="Test model",
        )

        # Verify
        assert version == mock_version
        mock_mlflow_client.return_value.create_model_version.assert_called_once()


class TestModelRegistryErrorHandling:
    """Tests for error handling in ModelRegistry."""

    @pytest.fixture
    def model_registry(self, sample_catalog, sample_schema):
        """Create ModelRegistry instance for testing."""
        return ModelRegistry(sample_catalog, sample_schema)

    @patch("databricks_monitoring.model_registry.MlflowClient")
    def test_get_latest_model_version_not_found(
        self, mock_mlflow_client, model_registry
    ):
        """Test handling of model not found."""
        # Setup mock to raise exception
        mock_mlflow_client.return_value.get_model_version_by_alias.side_effect = (
            Exception("Model not found")
        )

        # Test - should return None on error
        version = model_registry.get_latest_model_version(
            "nonexistent_model", alias="champion"
        )

        # Verify
        assert version is None
