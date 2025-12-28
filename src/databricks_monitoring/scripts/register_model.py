"""Model registration script for Unity Catalog.

This script registers trained models to Unity Catalog with versioning, tagging,
and alias management. It reads the latest MLFlow run and registers the model
if it meets quality criteria.

Usage:
    python register_model.py --catalog <catalog> --schema <schema> --model-name <name>

    Or as CLI entry point:
    register_model --catalog mlops_dev --schema my_model --model-name my_classifier
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import mlflow
from loguru import logger

from databricks_monitoring.model_registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Register model to Unity Catalog with versioning"
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
        help="Model name for Unity Catalog registration",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="MLFlow run ID to register (if not provided, uses latest successful run)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="latest-model",
        help="Model alias to set (default: latest-model)",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Model description",
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        help="Git commit SHA for tagging",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["dev", "aut", "prod"],
        help="Environment (dev, aut, prod)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLFlow experiment name to search for runs",
    )
    return parser.parse_args()


def get_latest_successful_run(experiment_name: str) -> Optional[Dict]:
    """Get the latest successful MLFlow run from experiment.

    Args:
        experiment_name: MLFlow experiment name

    Returns:
        Dictionary with run information or None if no suitable run found
    """
    logger.info(f"Searching for latest successful run in: {experiment_name}")

    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error(f"Experiment not found: {experiment_name}")
            return None

        # Search for successful runs with models
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED' and tags.should_register = 'True'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            logger.warning("No successful runs found with should_register=True")
            return None

        run = runs.iloc[0]
        run_id = run["run_id"]
        f1_score = run.get("metrics.test_f1", 0.0)

        logger.info(f"Found latest run: {run_id}")
        logger.info(f"F1 Score: {f1_score:.4f}")

        return {
            "run_id": run_id,
            "f1_score": f1_score,
            "start_time": run.get("start_time"),
        }

    except Exception as e:
        logger.error(f"Failed to search for runs: {e}")
        return None


def register_model_to_uc(
    registry: ModelRegistry,
    model_uri: str,
    model_name: str,
    tags: Dict[str, str],
    description: str,
) -> str:
    """Register model to Unity Catalog.

    Args:
        registry: ModelRegistry instance
        model_uri: MLFlow model URI (e.g., runs:/run_id/model)
        model_name: Model name in Unity Catalog
        tags: Dictionary of tags to add to model
        description: Model description

    Returns:
        str: Registered model version

    Raises:
        Exception: If registration fails
    """
    logger.info("=" * 60)
    logger.info("Registering Model to Unity Catalog")
    logger.info("=" * 60)
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Tags: {tags}")

    try:
        model_version = registry.register_model(
            model_uri=model_uri,
            model_name=model_name,
            tags=tags,
            description=description,
        )

        version_number = model_version.version
        logger.info(f"✓ Model registered successfully")
        logger.info(f"  Version: {version_number}")
        logger.info(f"  Full name: {registry.catalog}.{registry.schema}.{model_name}")

        return version_number

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise


def set_model_alias(
    registry: ModelRegistry, model_name: str, version: str, alias: str
) -> None:
    """Set alias for registered model version.

    Args:
        registry: ModelRegistry instance
        model_name: Model name
        version: Model version
        alias: Alias to set (e.g., "latest-model", "champion")

    Raises:
        Exception: If setting alias fails
    """
    logger.info(f"Setting alias '{alias}' for version {version}")

    try:
        registry.set_model_alias(model_name, version, alias)
        logger.info(f"✓ Alias '{alias}' set successfully")
    except Exception as e:
        logger.error(f"Failed to set alias: {e}")
        raise


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Model Registration Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")
    logger.info(f"Model name: {args.model_name}")

    try:
        # Initialize model registry
        registry = ModelRegistry(args.catalog, args.schema)

        # Get run ID
        run_id = args.run_id
        if not run_id:
            # Search for latest successful run
            experiment_name = (
                args.experiment_name
                or f"/Shared/{args.catalog}/{args.schema}/training"
            )
            run_info = get_latest_successful_run(experiment_name)

            if not run_info:
                logger.error("No suitable run found for registration")
                return 1

            run_id = run_info["run_id"]

        logger.info(f"Using run ID: {run_id}")

        # Build model URI
        model_uri = f"runs:/{run_id}/model"

        # Build tags
        tags = {
            "environment": args.environment or "unknown",
            "registered_at": datetime.now().isoformat(),
        }

        if args.git_sha:
            tags["git_sha"] = args.git_sha

        # Get metrics from run for description
        run = mlflow.get_run(run_id)
        f1_score = run.data.metrics.get("test_f1", 0.0)
        accuracy = run.data.metrics.get("test_accuracy", 0.0)

        # Build description
        description = args.description or (
            f"Model trained on {datetime.now().strftime('%Y-%m-%d')} "
            f"with F1={f1_score:.4f}, Accuracy={accuracy:.4f}"
        )

        # Register model
        version = register_model_to_uc(
            registry=registry,
            model_uri=model_uri,
            model_name=args.model_name,
            tags=tags,
            description=description,
        )

        # Set alias
        if args.alias:
            set_model_alias(registry, args.model_name, version, args.alias)

        # Get model metadata
        logger.info("=" * 60)
        logger.info("Registration Summary")
        logger.info("=" * 60)
        logger.info(f"Model: {registry.catalog}.{registry.schema}.{args.model_name}")
        logger.info(f"Version: {version}")
        logger.info(f"Alias: {args.alias}")
        logger.info(f"F1 Score: {f1_score:.4f}")
        logger.info(f"Run ID: {run_id}")
        logger.info("=" * 60)

        # Output version for downstream tasks
        print(f"REGISTERED_VERSION={version}")

        return 0

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
