"""Model training script with MLFlow tracking.

This script trains a machine learning model, logs parameters and metrics to MLFlow,
evaluates performance, and compares against baseline models.

Usage:
    python train_model.py --catalog <catalog> --schema <schema> --environment <env>

    Or as CLI entry point:
    train_model --catalog mlops_dev --schema my_model --environment dev
"""

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from databricks_monitoring.mlflow_tracking import MLFlowTracker


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML model with MLFlow tracking")
    parser.add_argument(
        "--catalog",
        type=str,
        required=True,
        help="Unity Catalog name",
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
        "--train-table",
        type=str,
        default="train_set",
        help="Training data table name (default: train_set)",
    )
    parser.add_argument(
        "--test-table",
        type=str,
        default="test_set",
        help="Test data table name (default: test_set)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Target column name (default: target)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ml_model",
        help="Model name for MLFlow registration (default: ml_model)",
    )
    parser.add_argument(
        "--baseline-f1",
        type=float,
        default=0.0,
        help="Baseline F1 score to beat (default: 0.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="LightGBM max_depth parameter (default: 5)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="LightGBM n_estimators parameter (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="LightGBM learning_rate parameter (default: 0.1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def load_data(
    spark: SparkSession, catalog: str, schema: str, table_name: str
) -> Tuple[pd.DataFrame, str]:
    """Load data from Delta table and return as pandas DataFrame with version.

    Args:
        spark: SparkSession instance
        catalog: Unity Catalog name
        schema: Schema name
        table_name: Table name

    Returns:
        Tuple of (pandas DataFrame, table version)
    """
    table_path = f"{catalog}.{schema}.{table_name}"
    logger.info(f"Loading data from {table_path}")

    try:
        df = spark.table(table_path).toPandas()
        logger.info(f"Loaded {len(df)} rows from {table_path}")

        # Get table version
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forName(spark, table_path)
        version = delta_table.history(1).select("version").collect()[0]["version"]

        return df, str(version)
    except Exception as e:
        logger.error(f"Failed to load data from {table_path}: {e}")
        raise


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare feature matrices and target vectors.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    logger.info("Preparing features and target")

    # Separate features and target
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logger.info(f"Features: {list(feature_cols)}")
    logger.info(f"Target: {target_col}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    return X_train, y_train, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    random_seed: int,
) -> LGBMClassifier:
    """Train LightGBM classifier.

    Args:
        X_train: Training features
        y_train: Training target
        params: Model hyperparameters
        random_seed: Random seed

    Returns:
        Trained model
    """
    logger.info("Training LightGBM classifier")
    logger.info(f"Hyperparameters: {params}")

    model = LGBMClassifier(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        random_state=random_seed,
        verbose=-1,  # Suppress training output
    )

    model.fit(X_train, y_train)
    logger.info("Training completed")

    return model


def evaluate_model(
    model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate model and return metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, average="binary"),
        "test_recall": recall_score(y_test, y_pred, average="binary"),
        "test_f1": f1_score(y_test, y_pred, average="binary"),
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    logger.info("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    return metrics


def compare_with_baseline(f1_score: float, baseline_f1: float) -> bool:
    """Compare model performance with baseline.

    Args:
        f1_score: Current model F1 score
        baseline_f1: Baseline F1 score to beat

    Returns:
        True if model improves on baseline, False otherwise
    """
    improvement = f1_score - baseline_f1
    improvement_pct = (improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0

    logger.info("=" * 60)
    logger.info("Baseline Comparison")
    logger.info("=" * 60)
    logger.info(f"Current F1: {f1_score:.4f}")
    logger.info(f"Baseline F1: {baseline_f1:.4f}")
    logger.info(f"Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

    if f1_score > baseline_f1:
        logger.info("✓ Model IMPROVES on baseline - ready for registration")
        return True
    else:
        logger.warning("✗ Model DOES NOT improve on baseline - skipping registration")
        return False


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Model Training Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

        # Initialize MLFlow tracker
        tracker = MLFlowTracker(
            experiment_name=f"/Shared/{args.catalog}/{args.schema}/training"
        )

        with tracker.start_run(run_name=f"training_{args.environment}"):
            # Hyperparameters
            params = {
                "max_depth": args.max_depth,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "random_seed": args.random_seed,
            }

            # Log parameters
            tracker.log_params(
                {
                    **params,
                    "catalog": args.catalog,
                    "schema": args.schema,
                    "environment": args.environment,
                    "model_name": args.model_name,
                    "target_col": args.target_col,
                    "train_table": args.train_table,
                    "test_table": args.test_table,
                    "training_date": datetime.now().isoformat(),
                }
            )

            # Load training and test data
            train_df, train_version = load_data(
                spark, args.catalog, args.schema, args.train_table
            )
            test_df, test_version = load_data(
                spark, args.catalog, args.schema, args.test_table
            )

            # Log dataset versions
            tracker.log_params(
                {
                    "train_version": train_version,
                    "test_version": test_version,
                }
            )

            # Prepare features
            X_train, y_train, X_test, y_test = prepare_features(
                train_df, test_df, args.target_col
            )

            # Train model
            model = train_model(X_train, y_train, params, args.random_seed)

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics to MLFlow
            tracker.log_metrics(metrics)

            # Compare with baseline
            improves_baseline = compare_with_baseline(
                metrics["test_f1"], args.baseline_f1
            )

            # Log model artifacts
            logger.info("Logging model to MLFlow")

            # Create model signature
            from mlflow.models.signature import infer_signature

            signature = infer_signature(X_train, model.predict(X_train))

            # Log model with artifacts
            tracker.log_model(
                model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(1),
            )

            # Log feature importance
            feature_importance = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            tracker.log_artifact_dataframe(
                feature_importance, "feature_importance.csv"
            )

            # Log decision: register or not
            tracker.log_params(
                {
                    "improves_baseline": improves_baseline,
                    "baseline_f1": args.baseline_f1,
                    "should_register": improves_baseline,
                }
            )

            logger.info("=" * 60)
            logger.info("Model Training Completed Successfully")
            logger.info("=" * 60)
            logger.info(f"Test F1 Score: {metrics['test_f1']:.4f}")
            logger.info(f"Improves baseline: {improves_baseline}")
            logger.info("=" * 60)

            # Return 0 if improves baseline (for pipeline conditional logic)
            return 0 if improves_baseline else 1

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
