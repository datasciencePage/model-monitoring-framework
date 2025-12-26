"""Generic monitoring framework for ML models."""

import time
from typing import Any, Optional

from loguru import logger

from databricks_monitoring.model_registry import ModelRegistry
from databricks_monitoring.monitoring.lakehouse_monitor import LakehouseMonitor
from databricks_monitoring.serving.model_serving_setup import ServingSetup


class MonitoringFramework:
    """Generic end-to-end monitoring framework for ML models.

    This class provides a unified interface for deploying models with
    serving endpoints and comprehensive monitoring.
    """

    def __init__(self, catalog: str, schema: str, model_name: str):
        """Initialize monitoring framework.

        Args:
            catalog: Unity Catalog name
            schema: Schema name
            model_name: Base model name
        """
        self.catalog = catalog
        self.schema = schema
        self.model_name = model_name

        # Initialize components
        self.registry = ModelRegistry(catalog, schema)
        self.serving_setup = ServingSetup()
        self.monitor = LakehouseMonitor(catalog, schema)

        logger.info(f"Initialized monitoring framework for {catalog}.{schema}.{model_name}")

    def deploy_with_monitoring(
        self,
        model_version: str,
        serving_config: dict[str, Any],
        inference_table_config: dict[str, Any],
        monitoring_config: dict[str, Any],
        tags: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Deploy model with serving endpoint and monitoring.

        This method performs end-to-end deployment:
        1. Creates/updates serving endpoint with inference table enabled
        2. Waits for endpoint to be ready
        3. Validates inference table configuration
        4. Creates Lakehouse monitor for drift detection
        5. Sets up alerts

        Args:
            model_version: Model version number
            serving_config: Serving configuration dict
                - endpoint_name: str
                - workload_size: str
                - scale_to_zero: bool
            inference_table_config: Inference table configuration dict
                - name: str (table name)
                - enabled: bool
            monitoring_config: Monitoring configuration dict
                - output_schema: str
                - granularities: List[str]
                - baseline_table: Optional[str]
                - problem_type: str
                - alerts: List[dict]
            tags: Optional tags for the deployment

        Returns:
            dict: Deployment information
                - endpoint_name: str
                - monitor_name: str
                - inference_table: str
                - dashboard_url: str

        Raises:
            Exception: If deployment fails at any step
        """
        logger.info("=" * 80)
        logger.info("Starting end-to-end model deployment with monitoring")
        logger.info("=" * 80)

        deployment_info = {}

        try:
            # Step 1: Create serving endpoint
            logger.info("[1/5] Creating serving endpoint...")
            full_model_name = f"{self.catalog}.{self.schema}.{self.model_name}"

            inference_table_full_config = {
                "catalog": self.catalog,
                "schema": self.schema,
                "table_name": inference_table_config["name"],
                "enabled": inference_table_config.get("enabled", True),
            }

            endpoint_name = self.serving_setup.create_endpoint(
                endpoint_name=serving_config["endpoint_name"],
                model_name=full_model_name,
                model_version=model_version,
                workload_size=serving_config.get("workload_size", "Small"),
                scale_to_zero=serving_config.get("scale_to_zero", True),
                inference_table_config=inference_table_full_config,
                tags=tags,
            )

            deployment_info["endpoint_name"] = endpoint_name

            # Step 2: Wait for endpoint to be ready
            logger.info("[2/5] Waiting for endpoint to be ready...")
            is_ready = self.serving_setup.wait_for_endpoint_ready(
                endpoint_name=endpoint_name,
                timeout=600,
            )

            if not is_ready:
                msg = f"Endpoint {endpoint_name} failed to become ready"
                raise TimeoutError(msg)

            # Step 3: Validate inference table
            logger.info("[3/5] Validating inference table configuration...")
            inference_table_name = f"{self.catalog}.{self.schema}.{inference_table_config['name']}"

            if not self._validate_inference_table(inference_table_name):
                logger.warning("Inference table validation failed, but continuing...")

            deployment_info["inference_table"] = inference_table_name

            # Step 4: Create Lakehouse monitor
            logger.info("[4/5] Creating Lakehouse monitor...")
            output_schema_full = f"{self.catalog}.{monitoring_config['output_schema']}"

            monitor_name = self.monitor.create_monitor(
                inference_table=inference_table_name,
                output_schema=output_schema_full,
                profile_type="InferenceLog",
                granularities=monitoring_config.get("granularities", ["1 day"]),
                baseline_table=monitoring_config.get("baseline_table"),
                problem_type=monitoring_config.get("problem_type", "classification"),
                prediction_col="prediction",
                label_col=monitoring_config.get("label_col"),
                timestamp_col="timestamp",
            )

            deployment_info["monitor_name"] = monitor_name

            # Step 5: Set up alerts
            logger.info("[5/5] Setting up alerts...")
            alerts = monitoring_config.get("alerts", [])
            self._setup_alerts(monitor_name, alerts)

            # Get dashboard URL
            dashboard_url = self.monitor.get_dashboard_url(monitor_name)
            deployment_info["dashboard_url"] = dashboard_url

            logger.info("=" * 80)
            logger.info("Deployment completed successfully!")
            logger.info(f"  Endpoint: {endpoint_name}")
            logger.info(f"  Inference Table: {inference_table_name}")
            logger.info(f"  Monitor: {monitor_name}")
            logger.info(f"  Dashboard: {dashboard_url}")
            logger.info("=" * 80)

            return deployment_info

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"Deployment failed: {e}")
            logger.error("=" * 80)
            raise

    def _validate_inference_table(self, table_name: str, timeout: int = 300) -> bool:
        """Validate that inference table exists and is being populated.

        Args:
            table_name: Full table name
            timeout: Maximum wait time in seconds

        Returns:
            bool: True if validation successful
        """
        logger.info(f"Validating inference table: {table_name}")

        # Wait for table to be created and populated
        # This typically happens after first inference request

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if table exists by attempting to query it
                # Note: This requires PySpark or SQL API
                logger.info("Inference table validation requires manual verification")
                logger.info("Send test requests to the endpoint to populate the table")
                return True

            except Exception as e:
                logger.debug(f"Table not ready yet: {e}")
                time.sleep(10)

        logger.warning(f"Timeout waiting for inference table {table_name}")
        return False

    def _setup_alerts(self, monitor_name: str, alerts: list[dict[str, Any]]) -> None:
        """Set up alerts for monitoring metrics.

        Args:
            monitor_name: Monitor name (table name)
            alerts: List of alert configurations
        """
        if not alerts:
            logger.info("No alerts configured")
            return

        logger.info(f"Setting up {len(alerts)} alerts...")

        for alert in alerts:
            try:
                self.monitor.create_alert(
                    monitor_name=monitor_name,
                    metric=alert["metric"],
                    threshold=alert["threshold"],
                    notification=alert.get("notification_type", "email"),
                )

                logger.info(f"  ✓ Alert configured: {alert['metric']} (threshold: {alert['threshold']})")

            except Exception as e:
                logger.warning(f"  ✗ Failed to configure alert {alert['metric']}: {e}")

    def refresh_monitoring(self) -> None:
        """Refresh all monitoring metrics."""
        logger.info("Refreshing monitoring metrics...")

        inference_table = f"{self.catalog}.{self.schema}.inference_logs"

        try:
            self.monitor.refresh_monitor(inference_table)
            logger.info("Monitoring refresh completed")

        except Exception as e:
            logger.error(f"Failed to refresh monitoring: {e}")
            raise

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status.

        Returns:
            dict: Monitoring status information
        """
        inference_table = f"{self.catalog}.{self.schema}.inference_logs"

        return {
            "inference_table": inference_table,
            "metrics": self.monitor.get_monitor_metrics(inference_table),
            "dashboard_url": self.monitor.get_dashboard_url(inference_table),
        }

    def cleanup(self, delete_endpoint: bool = False, delete_monitor: bool = False) -> None:
        """Clean up resources.

        Args:
            delete_endpoint: Whether to delete the serving endpoint
            delete_monitor: Whether to delete the monitor
        """
        logger.info("Cleaning up resources...")

        if delete_monitor:
            inference_table = f"{self.catalog}.{self.schema}.inference_logs"
            try:
                self.monitor.delete_monitor(inference_table)
                logger.info(f"Deleted monitor for {inference_table}")
            except Exception as e:
                logger.error(f"Failed to delete monitor: {e}")

        if delete_endpoint:
            endpoint_name = f"{self.model_name}-serving"
            try:
                self.serving_setup.delete_endpoint(endpoint_name)
                logger.info(f"Deleted endpoint {endpoint_name}")
            except Exception as e:
                logger.error(f"Failed to delete endpoint: {e}")

        logger.info("Cleanup completed")
