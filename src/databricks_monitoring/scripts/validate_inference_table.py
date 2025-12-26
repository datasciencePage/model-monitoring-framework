"""Validate inference table configuration and logging."""

import argparse
import sys
import time

from loguru import logger
from pyspark.sql import SparkSession


def main():
    """Validate inference table is configured and receiving data."""
    parser = argparse.ArgumentParser(description="Validate inference table")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--table-name", default="inference_logs", help="Inference table name")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Inference Table Validation")
    logger.info("=" * 80)

    table_name = f"{args.catalog}.{args.schema}.{args.table_name}"
    logger.info(f"Table: {table_name}")

    try:
        # Initialize Spark session
        spark = SparkSession.builder.getOrCreate()

        # Wait for table to exist and be populated
        start_time = time.time()
        table_exists = False
        has_data = False

        while time.time() - start_time < args.timeout:
            try:
                # Check if table exists
                df = spark.table(table_name)
                table_exists = True
                logger.info("✓ Inference table exists")

                # Check if table has data
                count = df.count()
                if count > 0:
                    has_data = True
                    logger.info(f"✓ Inference table has {count} records")

                    # Show schema
                    logger.info("\nTable Schema:")
                    df.printSchema()

                    # Show sample data
                    logger.info("\nSample Records:")
                    df.show(5, truncate=False)

                    break
                else:
                    logger.info("Waiting for data to be logged...")
                    time.sleep(10)

            except Exception as e:
                logger.debug(f"Table not ready: {e}")
                time.sleep(10)

        if not table_exists:
            logger.error("✗ Inference table does not exist")
            logger.error("Make sure the serving endpoint has inference table enabled")
            sys.exit(1)

        if not has_data:
            logger.warning("✗ Inference table exists but has no data")
            logger.warning("Send test requests to the endpoint to populate the table")
            logger.warning("This is not a critical error - the table will be populated on first inference")

        logger.info("=" * 80)
        logger.info("Validation Summary")
        logger.info("=" * 80)
        logger.info(f"Table Exists: {'✓' if table_exists else '✗'}")
        logger.info(f"Has Data: {'✓' if has_data else '⚠'}")
        logger.info("=" * 80)

        # Exit successfully even if no data yet
        sys.exit(0)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Validation failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
