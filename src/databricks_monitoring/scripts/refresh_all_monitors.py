"""Refresh all Lakehouse monitors."""

import argparse
import sys

from loguru import logger

from databricks_monitoring.monitoring.config import load_config
from databricks_monitoring.monitoring.lakehouse_monitor import LakehouseMonitor


def main():
    """Refresh all monitors for the environment."""
    parser = argparse.ArgumentParser(description="Refresh all Lakehouse monitors")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--environment", required=True, help="Environment (dev, aut, prod)")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Lakehouse Monitoring Refresh")
    logger.info("=" * 80)
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")

    try:
        # Load configuration
        config = load_config(args.environment)

        # Initialize monitor
        monitor = LakehouseMonitor(args.catalog, args.schema)

        # Refresh inference table monitor
        inference_table = f"{args.catalog}.{args.schema}.{config.inference_table.name}"

        logger.info(f"Refreshing monitor for: {inference_table}")

        monitor.refresh_monitor(inference_table)

        logger.info("✓ Monitor refresh completed")

        # Get monitor metrics
        logger.info("Retrieving monitor metrics...")
        metrics = monitor.get_monitor_metrics(inference_table)

        if metrics:
            logger.info("Monitor Metrics:")
            logger.info(f"  Status: {metrics.get('status', 'Unknown')}")
            logger.info(f"  Drift Metrics: {metrics.get('drift_metrics', 'N/A')}")
        else:
            logger.warning("Could not retrieve monitor metrics")

        logger.info("=" * 80)
        logger.info("Refresh Successful!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Monitor refresh failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
