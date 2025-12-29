"""Data preparation script for ML training pipeline.

This script prepares training and test datasets, saves them to Delta tables with versioning,
and logs dataset information to MLFlow for tracking and lineage.

Supports multiple data sources:
- Delta tables (Unity Catalog)
- Databricks Feature Store tables
- Synthetic data generation

Usage:
    # From Delta table
    python prepare_data.py --catalog <catalog> --schema <schema> --environment <env> --source-table my_table

    # From Feature Store
    python prepare_data.py --catalog <catalog> --schema <schema> --environment <env> --feature-store-table <table> --lookup-key <key>

    Or as CLI entry point:
    prepare_data --catalog mlops_dev --schema my_model --environment dev --feature-store-table my_features --lookup-key user_id
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
        help="Source Delta table name for data. If not provided and no Feature Store options given, uses synthetic data.",
    )

    # Feature Store options
    parser.add_argument(
        "--feature-store-table",
        type=str,
        help="Feature Store table name (format: catalog.schema.table). Cannot be used with --source-table.",
    )
    parser.add_argument(
        "--lookup-key",
        type=str,
        help="Primary key column name for Feature Store lookup (required if using --feature-store-table).",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="+",
        help="Specific feature columns to load from Feature Store. If not provided, loads all features.",
    )
    parser.add_argument(
        "--label-table",
        type=str,
        help="Optional label table to join with Feature Store features (format: catalog.schema.table).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="target",
        help="Target column name in label table (default: target)",
    )

    # Output options
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

    args = parser.parse_args()

    # Validation: Cannot use both source-table and feature-store-table
    if args.source_table and args.feature_store_table:
        parser.error("Cannot use both --source-table and --feature-store-table. Choose one data source.")

    # Validation: Feature Store requires lookup key
    if args.feature_store_table and not args.lookup_key:
        parser.error("--lookup-key is required when using --feature-store-table")

    return args


def load_from_feature_store(
    spark: SparkSession,
    feature_table: str,
    lookup_key: str,
    feature_columns: Optional[List[str]] = None,
    label_table: Optional[str] = None,
    label_column: str = "target",
) -> DataFrame:
    """Load data from Databricks Feature Store.

    Args:
        spark: SparkSession instance
        feature_table: Feature Store table name (catalog.schema.table)
        lookup_key: Primary key column name for feature lookup
        feature_columns: Optional list of specific features to load. If None, loads all.
        label_table: Optional label table to join with features (catalog.schema.table)
        label_column: Target column name in label table

    Returns:
        DataFrame: Training data with features and optional labels

    Raises:
        Exception: If Feature Store table doesn't exist or loading fails
    """
    logger.info("=" * 60)
    logger.info("Loading Data from Feature Store")
    logger.info("=" * 60)
    logger.info(f"Feature table: {feature_table}")
    logger.info(f"Lookup key: {lookup_key}")

    try:
        from databricks.feature_engineering import FeatureEngineeringClient

        fe = FeatureEngineeringClient()

        # Load feature table
        logger.info(f"Reading feature table: {feature_table}")
        feature_df = spark.table(feature_table)

        # Filter to specific columns if requested
        if feature_columns:
            # Always include lookup key
            columns_to_select = [lookup_key] + [col for col in feature_columns if col != lookup_key]
            feature_df = feature_df.select(*columns_to_select)
            logger.info(f"Selected {len(feature_columns)} specific features")

        feature_count = feature_df.count()
        logger.info(f"Loaded {feature_count} rows from feature table")
        logger.info(f"Feature columns: {feature_df.columns}")

        # Join with labels if label table provided
        if label_table:
            logger.info(f"Joining with label table: {label_table}")
            label_df = spark.table(label_table)

            # Select only lookup key and label column
            label_df = label_df.select(lookup_key, label_column)

            # Join features with labels
            training_df = feature_df.join(label_df, on=lookup_key, how="inner")

            final_count = training_df.count()
            logger.info(f"Joined {final_count} rows (features + labels)")

            if final_count < feature_count:
                logger.warning(
                    f"Lost {feature_count - final_count} rows in join. "
                    f"Ensure label table has matching {lookup_key} values."
                )
        else:
            training_df = feature_df
            logger.warning(
                f"No label table provided. Ensure feature table contains '{label_column}' column, "
                "or use --label-table to join with labels."
            )

        logger.info(f"Final dataset: {training_df.count()} rows, {len(training_df.columns)} columns")
        logger.info("=" * 60)

        return training_df

    except ImportError:
        logger.error(
            "Databricks Feature Engineering client not available. "
            "Ensure you're running on Databricks or have databricks-feature-engineering installed."
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load from Feature Store: {e}")
        raise


def load_or_generate_data(
    spark: SparkSession, catalog: str, schema: str, source_table: Optional[str] = None
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
        logger.info(f"Loading data from Delta table: {table_path}")
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
            # Determine data source type
            data_source = "synthetic"
            if args.feature_store_table:
                data_source = "feature_store"
            elif args.source_table:
                data_source = "delta_table"

            # Log parameters
            params = {
                "catalog": args.catalog,
                "schema": args.schema,
                "environment": args.environment,
                "test_size": args.test_size,
                "random_seed": args.random_seed,
                "train_table": args.train_table,
                "test_table": args.test_table,
                "data_source": data_source,
                "preparation_date": datetime.now().isoformat(),
            }

            # Add Feature Store specific parameters
            if args.feature_store_table:
                params.update({
                    "feature_store_table": args.feature_store_table,
                    "lookup_key": args.lookup_key,
                    "label_table": args.label_table or "none",
                    "label_column": args.label_column,
                    "feature_columns": ",".join(args.feature_columns) if args.feature_columns else "all",
                })
            elif args.source_table:
                params["source_table"] = args.source_table

            tracker.log_params(params)

            # Load data based on source type
            if args.feature_store_table:
                logger.info("Loading data from Feature Store")
                raw_df = load_from_feature_store(
                    spark=spark,
                    feature_table=args.feature_store_table,
                    lookup_key=args.lookup_key,
                    feature_columns=args.feature_columns,
                    label_table=args.label_table,
                    label_column=args.label_column,
                )
            else:
                # Load from Delta table or generate synthetic data
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
