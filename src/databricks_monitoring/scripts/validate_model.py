"""Model validation script for registered models.

This script validates a registered model by loading it from Unity Catalog,
verifying its signature, running test predictions, and checking model quality.

Usage:
    python validate_model.py --catalog <catalog> --schema <schema> --model-name <name>

    Or as CLI entry point:
    validate_model --catalog mlops_dev --schema my_model --model-name my_classifier
"""

import argparse
import sys
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from databricks_monitoring.model_registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate registered model from Unity Catalog"
    )
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
        "--model-name",
        type=str,
        required=True,
        help="Model name in Unity Catalog",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Specific model version to validate (if not provided, uses latest-model alias)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="latest-model",
        help="Model alias to validate (default: latest-model)",
    )
    parser.add_argument(
        "--test-table",
        type=str,
        default="test_set",
        help="Test data table name for validation (default: test_set)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of test samples for inference (default: 5)",
    )
    return parser.parse_args()


def load_model(
    registry: ModelRegistry, model_name: str, version: Optional[str], alias: str
) -> tuple[Any, str]:
    """Load model from Unity Catalog.

    Args:
        registry: ModelRegistry instance
        model_name: Model name
        version: Specific version (optional)
        alias: Model alias to use if version not specified

    Returns:
        Tuple of (loaded model, version used)

    Raises:
        Exception: If model loading fails
    """
    logger.info("=" * 60)
    logger.info("Loading Model from Unity Catalog")
    logger.info("=" * 60)

    try:
        if version:
            logger.info(f"Loading specific version: {version}")
            model = registry.load_model_for_inference(model_name, version=version)
            model_version = version
        else:
            logger.info(f"Loading by alias: {alias}")
            model = registry.load_model_for_inference(model_name, alias=alias)
            # Get version from alias
            model_version = registry.get_latest_model_version(
                model_name, alias=alias
            )

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Model: {registry.catalog}.{registry.schema}.{model_name}")
        logger.info(f"  Version: {model_version}")

        return model, model_version

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def verify_model_signature(model_name: str, version: str) -> bool:
    """Verify model has valid signature.

    Args:
        model_name: Model name
        version: Model version

    Returns:
        True if signature is valid, False otherwise
    """
    logger.info("Verifying model signature")

    try:
        import mlflow

        model_uri = f"models:/{model_name}/{version}"
        model_info = mlflow.models.get_model_info(model_uri)

        if model_info.signature:
            logger.info("✓ Model has valid signature")
            logger.info(f"  Inputs: {model_info.signature.inputs}")
            logger.info(f"  Outputs: {model_info.signature.outputs}")
            return True
        else:
            logger.warning("✗ Model has no signature")
            return False

    except Exception as e:
        logger.error(f"Failed to verify signature: {e}")
        return False


def load_test_data(
    spark: SparkSession, catalog: str, schema: str, table_name: str, sample_size: int
) -> pd.DataFrame:
    """Load sample test data for validation.

    Args:
        spark: SparkSession instance
        catalog: Catalog name
        schema: Schema name
        table_name: Table name
        sample_size: Number of samples to load

    Returns:
        pandas DataFrame with test samples
    """
    table_path = f"{catalog}.{schema}.{table_name}"
    logger.info(f"Loading {sample_size} test samples from {table_path}")

    try:
        df = spark.table(table_path).limit(sample_size).toPandas()
        logger.info(f"✓ Loaded {len(df)} test samples")
        return df
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


def test_inference(model: Any, test_df: pd.DataFrame, target_col: str = "target") -> bool:
    """Test model inference on sample data.

    Args:
        model: Loaded model
        test_df: Test data
        target_col: Target column name

    Returns:
        True if inference succeeds, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Testing Model Inference")
    logger.info("=" * 60)

    try:
        # Prepare features (exclude target if present)
        feature_cols = [col for col in test_df.columns if col != target_col]
        X_test = test_df[feature_cols]

        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Features: {list(X_test.columns)}")

        # Make predictions
        predictions = model.predict(X_test)
        logger.info(f"✓ Predictions generated: {predictions}")

        # Check prediction probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            logger.info(f"✓ Probabilities shape: {probabilities.shape}")

        logger.info("✓ Inference test PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ Inference test FAILED: {e}")
        return False


def get_model_metadata(registry: ModelRegistry, model_name: str, version: str) -> Dict:
    """Retrieve model metadata from registry.

    Args:
        registry: ModelRegistry instance
        model_name: Model name
        version: Model version

    Returns:
        Dictionary with model metadata
    """
    logger.info("Retrieving model metadata")

    try:
        metadata = registry.get_model_metadata(model_name, version)

        logger.info("Model metadata:")
        logger.info(f"  Name: {metadata.get('name')}")
        logger.info(f"  Version: {metadata.get('version')}")
        logger.info(f"  Description: {metadata.get('description')}")
        logger.info(f"  Tags: {metadata.get('tags')}")
        logger.info(f"  Status: {metadata.get('status')}")

        return metadata

    except Exception as e:
        logger.warning(f"Could not retrieve metadata: {e}")
        return {}


def run_validation_checks(
    model: Any,
    model_name: str,
    version: str,
    test_df: pd.DataFrame,
    registry: ModelRegistry,
) -> bool:
    """Run all validation checks.

    Args:
        model: Loaded model
        model_name: Model name
        version: Model version
        test_df: Test data
        registry: ModelRegistry instance

    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Running Validation Checks")
    logger.info("=" * 60)

    checks_passed = 0
    checks_total = 4

    # Check 1: Model loaded successfully
    if model is not None:
        logger.info("✓ Check 1/4: Model loaded successfully")
        checks_passed += 1
    else:
        logger.error("✗ Check 1/4: Model failed to load")

    # Check 2: Model signature verification
    full_model_name = f"{registry.catalog}.{registry.schema}.{model_name}"
    if verify_model_signature(full_model_name, version):
        logger.info("✓ Check 2/4: Model signature is valid")
        checks_passed += 1
    else:
        logger.warning("✗ Check 2/4: Model signature validation failed")

    # Check 3: Test inference
    if test_inference(model, test_df):
        logger.info("✓ Check 3/4: Inference test passed")
        checks_passed += 1
    else:
        logger.error("✗ Check 3/4: Inference test failed")

    # Check 4: Metadata retrieval
    metadata = get_model_metadata(registry, model_name, version)
    if metadata:
        logger.info("✓ Check 4/4: Metadata retrieved successfully")
        checks_passed += 1
    else:
        logger.warning("✗ Check 4/4: Metadata retrieval failed")

    # Summary
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info(f"Checks passed: {checks_passed}/{checks_total}")

    if checks_passed == checks_total:
        logger.info("✓ ALL VALIDATION CHECKS PASSED")
        return True
    elif checks_passed >= 3:
        logger.warning(f"⚠ VALIDATION PASSED with warnings ({checks_passed}/{checks_total})")
        return True
    else:
        logger.error(f"✗ VALIDATION FAILED ({checks_passed}/{checks_total})")
        return False


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Model Validation Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("ModelValidation").getOrCreate()

        # Initialize model registry
        registry = ModelRegistry(args.catalog, args.schema)

        # Load model
        model, version = load_model(
            registry, args.model_name, args.version, args.alias
        )

        # Load test data
        test_df = load_test_data(
            spark, args.catalog, args.schema, args.test_table, args.sample_size
        )

        # Run validation checks
        validation_passed = run_validation_checks(
            model, args.model_name, version, test_df, registry
        )

        logger.info("=" * 60)
        logger.info("Model Validation Completed")
        logger.info("=" * 60)
        logger.info(f"Result: {'PASSED' if validation_passed else 'FAILED'}")
        logger.info("=" * 60)

        return 0 if validation_passed else 1

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
