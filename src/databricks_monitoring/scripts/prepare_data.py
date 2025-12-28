"""Data preparation script for ML training pipeline.

This script prepares training and test datasets, saves them to Delta tables with versioning,
and logs dataset information to MLFlow for tracking and lineage.

Usage:
    python prepare_data.py --catalog <catalog> --schema <schema> --environment <env>

    Or as CLI entry point:
    prepare_data --catalog mlops_dev --schema my_model --environment dev
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from databricks_monitoring.mlflow_tracking import MLFlowTracker


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare training and test datasets for ML training"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="Unity Catalog name (e.g., mlops_dev, mlops_prod)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Schema name within the catalog",
    )
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        choices=["dev", "aut", "prod"],
        help="Environment (dev, aut, prod)",
    )
    parser.add_argument(
        "--source-table",
        type=str,
        help="Source table name for data. If not provided, uses synthetic data.",
    )
    parser.add_argument(
        "--train-table",
        type=str,
        default="train_set",
        help="Output table name for training data (default: train_set)",
    )
    parser.add_argument(
        "--test-table",
        type=str,
        default="test_set",
        help="Output table name for test data (default: test_set)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def load_or_generate_data(
    spark: SparkSession, catalog: str, schema: str, source_table: str = None
) -> DataFrame:
    """Load data from source table or generate synthetic data.

    Args:
        spark: SparkSession instance
        catalog: Unity Catalog name
        schema: Schema name
        source_table: Optional source table name. If None, generates synthetic data.

    Returns:
        DataFrame: Raw data for training

    Raises:
        Exception: If source table doesn't exist or data loading fails
    """
    if source_table:
        table_path = f"{catalog}.{schema}.{source_table}"
        logger.info(f"Loading data from {table_path}")
        try:
            df = spark.table(table_path)
            logger.info(f"Loaded {df.count()} rows from {table_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {table_path}: {e}")
            raise
    else:
        logger.info("Generating synthetic data for training")
        # Generate synthetic data for demonstration
        # Replace this with actual data loading logic
        data = [
            (1, 25, 50000, 1),
            (2, 35, 60000, 0),
            (3, 45, 70000, 1),
            (4, 22, 45000, 0),
            (5, 50, 80000, 1),
        ]
        df = spark.createDataFrame(data, ["id", "age", "income", "target"])
        logger.info(f"Generated {df.count()} synthetic rows")
        return df


def split_train_test(
    df: DataFrame, test_size: float, random_seed: int
) -> Tuple[DataFrame, DataFrame]:
    """Split data into training and test sets.

    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set (0.0 to 1.0)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data with test_size={test_size}, seed={random_seed}")

    train_ratio = 1.0 - test_size
    train_df, test_df = df.randomSplit([train_ratio, test_size], seed=random_seed)

    train_count = train_df.count()
    test_count = test_df.count()

    logger.info(f"Train set: {train_count} rows ({train_ratio*100:.1f}%)")
    logger.info(f"Test set: {test_count} rows ({test_size*100:.1f}%)")

    return train_df, test_df


def save_to_delta(
    df: DataFrame, catalog: str, schema: str, table_name: str, mode: str = "overwrite"
) -> str:
    """Save DataFrame to Delta table and return table version.

    Args:
        df: DataFrame to save
        catalog: Unity Catalog name
        schema: Schema name
        table_name: Table name
        mode: Write mode (default: overwrite)

    Returns:
        str: Delta table version
    """
    table_path = f"{catalog}.{schema}.{table_name}"
    logger.info(f"Saving {df.count()} rows to {table_path}")

    try:
        df.write.format("delta").mode(mode).saveAsTable(table_path)
        logger.info(f"Successfully saved to {table_path}")

        # Get table version
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forName(df.sparkSession, table_path)
        version = delta_table.history(1).select("version").collect()[0]["version"]
        logger.info(f"Table version: {version}")

        return str(version)
    except Exception as e:
        logger.error(f"Failed to save to {table_path}: {e}")
        raise


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Data Preparation Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random seed: {args.random_seed}")

    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("DataPreparation").getOrCreate()

        # Initialize MLFlow tracker
        tracker = MLFlowTracker(
            experiment_name=f"/Shared/{args.catalog}/{args.schema}/training"
        )

        with tracker.start_run(run_name=f"data_prep_{args.environment}"):
            # Log parameters
            tracker.log_params(
                {
                    "catalog": args.catalog,
                    "schema": args.schema,
                    "environment": args.environment,
                    "test_size": args.test_size,
                    "random_seed": args.random_seed,
                    "train_table": args.train_table,
                    "test_table": args.test_table,
                    "preparation_date": datetime.now().isoformat(),
                }
            )

            # Load or generate data
            raw_df = load_or_generate_data(
                spark, args.catalog, args.schema, args.source_table
            )

            # Split into train and test
            train_df, test_df = split_train_test(
                raw_df, args.test_size, args.random_seed
            )

            # Save to Delta tables
            train_version = save_to_delta(
                train_df, args.catalog, args.schema, args.train_table
            )
            test_version = save_to_delta(
                test_df, args.catalog, args.schema, args.test_table
            )

            # Log datasets to MLFlow
            train_table_path = f"{args.catalog}.{args.schema}.{args.train_table}"
            test_table_path = f"{args.catalog}.{args.schema}.{args.test_table}"

            tracker.log_training_data(
                train_df=train_df,
                test_df=test_df,
                train_table=train_table_path,
                test_table=test_table_path,
                train_version=train_version,
                test_version=test_version,
            )

            # Log metrics about the data
            tracker.log_metrics(
                {
                    "train_count": train_df.count(),
                    "test_count": test_df.count(),
                    "total_count": raw_df.count(),
                    "feature_count": len(raw_df.columns) - 1,  # Excluding target
                }
            )

            logger.info("=" * 60)
            logger.info("Data Preparation Completed Successfully")
            logger.info("=" * 60)
            logger.info(f"Train table: {train_table_path} (version: {train_version})")
            logger.info(f"Test table: {test_table_path} (version: {test_version})")
            logger.info("=" * 60)

            return 0

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
