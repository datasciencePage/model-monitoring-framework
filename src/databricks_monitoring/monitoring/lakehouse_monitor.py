"""Lakehouse monitoring for model drift detection and data quality."""

from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorTimeSeriesProfile
from loguru import logger


class LakehouseMonitor:
    """Lakehouse monitoring for ML models."""

    def __init__(self, catalog: str, schema: str):
        """Initialize Lakehouse monitor.

        Args:
            catalog: Unity Catalog name
            schema: Schema name
        """
        self.catalog = catalog
        self.schema = schema
        self.workspace_client = WorkspaceClient()

    def create_monitor(
        self,
        inference_table: str,
        output_schema: str,
        profile_type: str = "InferenceLog",
        granularities: Optional[list[str]] = None,
        baseline_table: Optional[str] = None,
        problem_type: str = "classification",
        prediction_col: str = "prediction",
        label_col: Optional[str] = None,
        timestamp_col: str = "timestamp",
        model_id_col: Optional[str] = None,
        slicing_exprs: Optional[list[str]] = None,
    ) -> str:
        """Create Lakehouse monitor on inference table.

        Args:
            inference_table: Full name of inference table (catalog.schema.table)
            output_schema: Schema for monitoring output tables (catalog.schema)
            profile_type: Profile type ("InferenceLog" or "TimeSeries")
            granularities: List of granularities (e.g., ["1 day", "1 hour"])
            baseline_table: Optional baseline table for drift detection
            problem_type: Problem type ("classification", "regression")
            prediction_col: Column containing predictions
            label_col: Optional column containing ground truth labels
            timestamp_col: Column containing timestamps
            model_id_col: Optional column containing model ID
            slicing_exprs: Optional list of slicing expressions

        Returns:
            str: Monitor name (same as table name)

        Raises:
            Exception: If monitor creation fails
        """
        logger.info(f"Creating Lakehouse monitor for table: {inference_table}")

        if granularities is None:
            granularities = ["1 day"]

        try:
            # Determine assets directory
            assets_dir = f"/Workspace/Shared/lakehouse_monitoring/{inference_table.replace('.', '_')}"

            if profile_type == "InferenceLog":
                # Create inference log monitor
                inference_log_config = MonitorInferenceLog(
                    problem_type=problem_type,
                    prediction_col=prediction_col,
                    timestamp_col=timestamp_col,
                    granularities=granularities,
                    model_id_col=model_id_col or "request_id",
                    label_col=label_col,
                )

                monitor = self.workspace_client.quality_monitors.create(
                    table_name=inference_table,
                    assets_dir=assets_dir,
                    output_schema_name=output_schema,
                    inference_log=inference_log_config,
                    baseline_table_name=baseline_table,
                    slicing_exprs=slicing_exprs,
                )

            elif profile_type == "TimeSeries":
                # Create time series monitor
                ts_profile_config = MonitorTimeSeriesProfile(
                    timestamp_col=timestamp_col,
                    granularities=granularities,
                )

                monitor = self.workspace_client.quality_monitors.create(
                    table_name=inference_table,
                    assets_dir=assets_dir,
                    output_schema_name=output_schema,
                    time_series=ts_profile_config,
                    baseline_table_name=baseline_table,
                    slicing_exprs=slicing_exprs,
                )

            else:
                msg = f"Unsupported profile type: {profile_type}"
                raise ValueError(msg)

            logger.info(f"Monitor created for {inference_table}")
            logger.info(f"Dashboard available at: {assets_dir}")

            return inference_table

        except Exception as e:
            logger.error(f"Failed to create monitor for {inference_table}: {e}")
            raise

    def refresh_monitor(self, table_name: str) -> None:
        """Manually refresh monitor metrics.

        Args:
            table_name: Full table name (catalog.schema.table)

        Raises:
            Exception: If refresh fails
        """
        logger.info(f"Refreshing monitor for table: {table_name}")

        try:
            self.workspace_client.quality_monitors.run_refresh(
                table_name=table_name,
            )

            logger.info(f"Monitor refresh initiated for {table_name}")

        except Exception as e:
            logger.error(f"Failed to refresh monitor for {table_name}: {e}")
            raise

    def get_monitor_metrics(
        self,
        table_name: str,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
    ) -> Optional[dict]:
        """Retrieve monitor metrics.

        Args:
            table_name: Full table name
            from_timestamp: Optional start timestamp (ISO format)
            to_timestamp: Optional end timestamp (ISO format)

        Returns:
            Optional[dict]: Monitor metrics or None if not available
        """
        try:
            monitor = self.workspace_client.quality_monitors.get(
                table_name=table_name,
            )

            # Extract metrics from monitor object
            # Note: The exact structure depends on Databricks SDK version
            return {
                "table_name": monitor.table_name,
                "status": monitor.status if hasattr(monitor, "status") else None,
                "drift_metrics": monitor.drift_metrics if hasattr(monitor, "drift_metrics") else None,
                "profile_metrics": monitor.profile_metrics if hasattr(monitor, "profile_metrics") else None,
            }

        except Exception as e:
            logger.error(f"Failed to get metrics for {table_name}: {e}")
            return None

    def create_alert(
        self,
        monitor_name: str,
        metric: str,
        threshold: float,
        notification: str,
    ) -> None:
        """Create alert for monitor metric.

        Args:
            monitor_name: Monitor name (table name)
            metric: Metric to monitor
            threshold: Threshold value
            notification: Notification channel (email address or webhook)

        Note:
            Alert creation may require additional Databricks SQL Alert API calls.
            This is a placeholder implementation.
        """
        logger.info(f"Creating alert for {monitor_name}: {metric} > {threshold}")
        logger.warning("Alert creation requires Databricks SQL Alerts API integration")

        # TODO: Implement alert creation using Databricks SQL Alerts API
        # This typically involves:
        # 1. Creating a SQL query that checks the metric
        # 2. Creating an alert based on the query
        # 3. Configuring notification destinations

    def delete_monitor(self, table_name: str) -> None:
        """Delete monitor for a table.

        Args:
            table_name: Full table name (catalog.schema.table)

        Raises:
            Exception: If deletion fails
        """
        logger.info(f"Deleting monitor for table: {table_name}")

        try:
            self.workspace_client.quality_monitors.delete(
                table_name=table_name,
            )

            logger.info(f"Monitor deleted for {table_name}")

        except Exception as e:
            logger.error(f"Failed to delete monitor for {table_name}: {e}")
            raise

    def get_dashboard_url(self, table_name: str) -> str:
        """Get dashboard URL for monitor.

        Args:
            table_name: Full table name

        Returns:
            str: Dashboard URL or assets directory path
        """
        assets_dir = f"/Workspace/Shared/lakehouse_monitoring/{table_name.replace('.', '_')}"
        return assets_dir

    def list_monitors(self) -> list[dict[str, Any]]:
        """List all monitors in the catalog/schema.

        Returns:
            list[dict]: List of monitor information
        """
        try:
            # Note: The SDK may not have a direct list method
            # This is a placeholder that could be implemented by:
            # 1. Listing tables in the schema
            # 2. Checking which tables have monitors

            logger.info(f"Listing monitors in {self.catalog}.{self.schema}")
            logger.warning("List monitors requires additional implementation")

            return []

        except Exception as e:
            logger.error(f"Failed to list monitors: {e}")
            return []

    def update_monitor(
        self,
        table_name: str,
        output_schema: Optional[str] = None,
        baseline_table: Optional[str] = None,
        slicing_exprs: Optional[list[str]] = None,
    ) -> None:
        """Update existing monitor configuration.

        Args:
            table_name: Full table name
            output_schema: Optional new output schema
            baseline_table: Optional new baseline table
            slicing_exprs: Optional new slicing expressions

        Raises:
            Exception: If update fails
        """
        logger.info(f"Updating monitor for table: {table_name}")

        try:
            # Update monitor
            # Note: The exact update method depends on Databricks SDK version
            self.workspace_client.quality_monitors.update(
                table_name=table_name,
                output_schema_name=output_schema,
                baseline_table_name=baseline_table,
                slicing_exprs=slicing_exprs,
            )

            logger.info(f"Monitor updated for {table_name}")

        except Exception as e:
            logger.error(f"Failed to update monitor for {table_name}: {e}")
            raise
