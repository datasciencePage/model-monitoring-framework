"""Deploy model serving endpoint with inference table enabled."""

import argparse
import sys

from loguru import logger

from databricks_monitoring.model_registry import ModelRegistry
from databricks_monitoring.monitoring.config import load_config
from databricks_monitoring.serving.model_serving_setup import ServingSetup


def main():
    """Deploy model serving endpoint with inference table."""
    parser = argparse.ArgumentParser(description="Deploy model serving endpoint")
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--environment", required=True, help="Environment (dev, aut, prod)")
    parser.add_argument("--git-sha", required=False, help="Git commit SHA for tagging")
    parser.add_argument("--model-name", default="marvel_character_model_basic", help="Model name")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Model Serving Deployment")
    logger.info("=" * 80)
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Schema: {args.schema}")
    logger.info(f"Model: {args.model_name}")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.environment)

        # Initialize model registry
        logger.info("Initializing model registry...")
        registry = ModelRegistry(args.catalog, args.schema)

        # Get latest model version
        logger.info("Retrieving latest model version...")
        model_version = registry.get_latest_model_version(
            model_name=args.model_name,
            alias="latest-model",
        )

        if model_version is None:
            logger.error(f"No model found with alias 'latest-model' for {args.model_name}")
            sys.exit(1)

        logger.info(f"Latest model version: {model_version}")

        # Initialize serving setup
        logger.info("Initializing serving setup...")
        serving_setup = ServingSetup()

        # Prepare inference table configuration
        inference_table_config = {
            "catalog": args.catalog,
            "schema": args.schema,
            "table_name": config.inference_table.name,
            "enabled": True,
        }

        # Prepare tags
        tags = {
            "environment": args.environment,
            "model_version": model_version,
        }

        if args.git_sha:
            tags["git_sha"] = args.git_sha

        # Create serving endpoint
        logger.info("Creating/updating serving endpoint...")
        full_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"

        endpoint_name = serving_setup.create_endpoint(
            endpoint_name=config.serving.endpoint_name,
            model_name=full_model_name,
            model_version=model_version,
            workload_size=config.serving.workload_size,
            scale_to_zero=config.serving.scale_to_zero,
            inference_table_config=inference_table_config,
            tags=tags,
        )

        # Wait for endpoint to be ready
        logger.info("Waiting for endpoint to be ready...")
        is_ready = serving_setup.wait_for_endpoint_ready(
            endpoint_name=endpoint_name,
            timeout=600,
        )

        if not is_ready:
            logger.error("Endpoint failed to become ready")
            sys.exit(1)

        # Get endpoint status
        status = serving_setup.get_endpoint_status(endpoint_name)

        logger.info("=" * 80)
        logger.info("Deployment Successful!")
        logger.info("=" * 80)
        logger.info(f"Endpoint Name: {endpoint_name}")
        logger.info(f"Model: {full_model_name}")
        logger.info(f"Version: {model_version}")
        logger.info(f"Inference Table: {args.catalog}.{args.schema}.{config.inference_table.name}")
        logger.info(f"Status: {status['state'] if status else 'Unknown'}")
        logger.info("=" * 80)

        # Output for Databricks task values
        print(f"endpoint_name={endpoint_name}")
        print(f"model_version={model_version}")
        print(f"inference_table={args.catalog}.{args.schema}.{config.inference_table.name}")

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Deployment failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
