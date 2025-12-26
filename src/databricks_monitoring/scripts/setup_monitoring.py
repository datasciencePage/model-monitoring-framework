"""Setup Lakehouse monitoring for inference table."""

import argparse
import sys

from loguru import logger

from databricks_monitoring.monitoring.config import load_config
from databricks_monitoring.monitoring.lakehouse_monitor import LakehouseMonitor


def main():
    """Setup Lakehouse monitoring on inference table."""
    parser = argparse.ArgumentParser(description="Setup Lakehouse monitoring")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--environment", required=True, help="Environment (dev, aut, prod)")
    parser.add_argument("--baseline-table", default="train_set", help="Baseline table for drift detection")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Lakehouse Monitoring Setup")
    logger.info("=" * 80)
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.environment)

        # Initialize Lakehouse monitor
        logger.info("Initializing Lakehouse monitor...")
        monitor = LakehouseMonitor(args.catalog, args.schema)

        # Prepare table names
        inference_table = f"{args.catalog}.{args.schema}.{config.inference_table.name}"
        output_schema = f"{args.catalog}.{config.monitoring.output_schema}"
        baseline_table = f"{args.catalog}.{args.schema}.{args.baseline_table}"

        logger.info(f"Inference Table: {inference_table}")
        logger.info(f"Output Schema: {output_schema}")
        logger.info(f"Baseline Table: {baseline_table}")

        # Create monitor
        logger.info("Creating Lakehouse monitor...")
        monitor_name = monitor.create_monitor(
            inference_table=inference_table,
            output_schema=output_schema,
            profile_type="InferenceLog",
            granularities=config.monitoring.granularities,
            baseline_table=baseline_table,
            problem_type="classification",
            prediction_col="prediction",
            label_col=None,  # No labels available in inference
            timestamp_col="timestamp",
            model_id_col="request_id",
        )

        logger.info(f"✓ Monitor created: {monitor_name}")

        # Refresh monitor to generate initial metrics
        logger.info("Refreshing monitor to generate initial metrics...")
        monitor.refresh_monitor(monitor_name)
        logger.info("✓ Monitor refreshed")

        # Set up alerts
        logger.info("Setting up alerts...")
        for alert_config in config.monitoring.alerts:
            try:
                monitor.create_alert(
                    monitor_name=monitor_name,
                    metric=alert_config.metric,
                    threshold=alert_config.threshold,
                    notification=alert_config.notification_type,
                )
                logger.info(f"✓ Alert configured: {alert_config.metric} (threshold: {alert_config.threshold})")
            except Exception as e:
                logger.warning(f"⚠ Failed to configure alert {alert_config.metric}: {e}")

        # Get dashboard URL
        dashboard_url = monitor.get_dashboard_url(monitor_name)

        logger.info("=" * 80)
        logger.info("Monitoring Setup Successful!")
        logger.info("=" * 80)
        logger.info(f"Monitor Name: {monitor_name}")
        logger.info(f"Dashboard URL: {dashboard_url}")
        logger.info(f"Granularities: {', '.join(config.monitoring.granularities)}")
        logger.info(f"Alerts Configured: {len(config.monitoring.alerts)}")
        logger.info("=" * 80)

        # Output for Databricks task values
        print(f"monitor_name={monitor_name}")
        print(f"dashboard_url={dashboard_url}")

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Monitoring setup failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
