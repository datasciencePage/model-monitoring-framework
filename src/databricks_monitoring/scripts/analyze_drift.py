"""Analyze model drift from monitoring tables."""

import argparse
import sys

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    """Analyze drift from monitoring tables."""
    parser = argparse.ArgumentParser(description="Analyze model drift")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--monitoring-schema", default="monitoring", help="Monitoring schema name")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Drift Analysis")
    logger.info("=" * 80)

    try:
        # Initialize Spark
        spark = SparkSession.builder.getOrCreate()

        # Construct monitoring table name
        # Note: The actual table name depends on how Databricks creates monitoring tables
        profile_table = f"{args.catalog}.{args.monitoring_schema}.profile_metrics"
        drift_table = f"{args.catalog}.{args.monitoring_schema}.drift_metrics"

        logger.info(f"Analyzing drift over last {args.days} days...")

        # Try to read drift metrics
        try:
            logger.info(f"Reading from: {drift_table}")
            drift_df = spark.table(drift_table)

            # Filter to recent data
            recent_drift = drift_df.filter(
                F.col("window_end") >= F.date_sub(F.current_date(), args.days)
            )

            count = recent_drift.count()

            if count > 0:
                logger.info(f"Found {count} drift records in the last {args.days} days")

                # Show drift summary
                logger.info("\nDrift Summary:")
                recent_drift.select(
                    "column_name",
                    "drift_type",
                    "drift_score",
                    "window_start",
                    "window_end",
                ).show(20, truncate=False)

                # Identify high drift columns
                high_drift = recent_drift.filter(F.col("drift_score") > 0.1)
                high_drift_count = high_drift.count()

                if high_drift_count > 0:
                    logger.warning(f"\n⚠ Found {high_drift_count} columns with high drift (>0.1)")
                    high_drift.select(
                        "column_name",
                        "drift_score",
                        "window_end",
                    ).orderBy(F.desc("drift_score")).show(10, truncate=False)
                else:
                    logger.info("\n✓ No significant drift detected")

            else:
                logger.warning("No drift records found in the specified time period")

        except Exception as e:
            logger.warning(f"Could not read drift table: {e}")
            logger.info("Drift metrics may not be available yet")

        # Try to read profile metrics
        try:
            logger.info(f"\nReading from: {profile_table}")
            profile_df = spark.table(profile_table)

            # Filter to recent data
            recent_profile = profile_df.filter(
                F.col("window_end") >= F.date_sub(F.current_date(), args.days)
            )

            count = recent_profile.count()

            if count > 0:
                logger.info(f"Found {count} profile records in the last {args.days} days")

                # Show profile summary
                logger.info("\nProfile Metrics Summary:")
                recent_profile.select(
                    "column_name",
                    "metric_name",
                    "metric_value",
                    "window_end",
                ).show(20, truncate=False)

            else:
                logger.warning("No profile records found in the specified time period")

        except Exception as e:
            logger.warning(f"Could not read profile table: {e}")
            logger.info("Profile metrics may not be available yet")

        logger.info("=" * 80)
        logger.info("Drift Analysis Complete")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Drift analysis failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
