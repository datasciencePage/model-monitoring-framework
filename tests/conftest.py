"""Shared pytest fixtures for model monitoring framework tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def sample_config_path():
    """Path to sample configuration file."""
    return Path(__file__).parent.parent / "config" / "monitoring_config.yml"


@pytest.fixture
def mock_workspace_client():
    """Mock Databricks WorkspaceClient."""
    client = Mock()
    client.quality_monitors = Mock()
    client.serving_endpoints = Mock()
    client.model_registry = Mock()
    return client


@pytest.fixture
def sample_catalog():
    """Sample catalog name for testing."""
    return "test_catalog"


@pytest.fixture
def sample_schema():
    """Sample schema name for testing."""
    return "test_schema"


@pytest.fixture
def sample_model_name():
    """Sample model name for testing."""
    return "test_model"


@pytest.fixture
def sample_environment():
    """Sample environment name for testing."""
    return "dev"
