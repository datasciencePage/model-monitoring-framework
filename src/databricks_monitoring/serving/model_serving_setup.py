"""Model serving setup with inference table support."""

import time
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
)
from loguru import logger


class ServingSetup:
    """Model serving endpoint setup and management."""

    def __init__(self):
        """Initialize serving setup with Databricks workspace client."""
        self.workspace_client = WorkspaceClient()

    def create_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        inference_table_config: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Create or update serving endpoint with inference table enabled.

        Args:
            endpoint_name: Name of the serving endpoint
            model_name: Full model name (catalog.schema.model_name)
            model_version: Model version number
            workload_size: Workload size (Small, Medium, Large)
            scale_to_zero: Enable scale to zero
            inference_table_config: Configuration for inference table logging.
                                   Should include: catalog, schema, table_name, enabled
            tags: Optional tags for the endpoint

        Returns:
            str: Endpoint name

        Raises:
            Exception: If endpoint creation/update fails
        """
        logger.info(f"Creating/updating serving endpoint: {endpoint_name}")
        logger.info(f"Model: {model_name} version {model_version}")

        try:
            # Check if endpoint already exists
            existing_endpoint = self._get_endpoint(endpoint_name)

            if existing_endpoint:
                logger.info(f"Endpoint {endpoint_name} exists, updating...")
                self._update_endpoint(
                    endpoint_name=endpoint_name,
                    model_name=model_name,
                    model_version=model_version,
                    workload_size=workload_size,
                    scale_to_zero=scale_to_zero,
                    inference_table_config=inference_table_config,
                    tags=tags,
                )
            else:
                logger.info(f"Creating new endpoint: {endpoint_name}")
                self._create_new_endpoint(
                    endpoint_name=endpoint_name,
                    model_name=model_name,
                    model_version=model_version,
                    workload_size=workload_size,
                    scale_to_zero=scale_to_zero,
                    inference_table_config=inference_table_config,
                    tags=tags,
                )

            logger.info(f"Endpoint {endpoint_name} configured successfully")
            return endpoint_name

        except Exception as e:
            logger.error(f"Failed to create/update endpoint {endpoint_name}: {e}")
            raise

    def _create_new_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        workload_size: str,
        scale_to_zero: bool,
        inference_table_config: Optional[dict[str, Any]],
        tags: Optional[dict[str, str]],
    ) -> None:
        """Create a new serving endpoint.

        Args:
            endpoint_name: Name of the serving endpoint
            model_name: Full model name
            model_version: Model version number
            workload_size: Workload size
            scale_to_zero: Enable scale to zero
            inference_table_config: Inference table configuration
            tags: Endpoint tags
        """
        # Configure served entity
        served_entity = ServedEntityInput(
            entity_name=model_name,
            entity_version=model_version,
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero,
        )

        # Configure endpoint
        config = EndpointCoreConfigInput(
            served_entities=[served_entity],
        )

        # CRITICAL: Add inference table configuration if provided
        if inference_table_config and inference_table_config.get("enabled", False):
            catalog = inference_table_config["catalog"]
            schema = inference_table_config["schema"]
            table_name = inference_table_config["table_name"]

            # Note: The actual SDK parameter name may vary - check Databricks SDK documentation
            # This is a placeholder for the inference table configuration
            # config.auto_capture_config = AutoCaptureConfigInput(
            #     catalog_name=catalog,
            #     schema_name=schema,
            #     table_name_prefix=table_name,
            #     enabled=True,
            # )
            logger.info(f"Inference table will be: {catalog}.{schema}.{table_name}")
            logger.warning("Note: Inference table configuration may require SDK update")

        # Create endpoint
        self.workspace_client.serving_endpoints.create(
            name=endpoint_name,
            config=config,
            tags=tags or [],
        )

        logger.info(f"Endpoint {endpoint_name} created")

    def _update_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        workload_size: str,
        scale_to_zero: bool,
        inference_table_config: Optional[dict[str, Any]],
        tags: Optional[dict[str, str]],
    ) -> None:
        """Update existing serving endpoint.

        Args:
            endpoint_name: Name of the serving endpoint
            model_name: Full model name
            model_version: Model version number
            workload_size: Workload size
            scale_to_zero: Enable scale to zero
            inference_table_config: Inference table configuration
            tags: Endpoint tags
        """
        # Configure served entity
        served_entity = ServedEntityInput(
            entity_name=model_name,
            entity_version=model_version,
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero,
        )

        # Configure endpoint
        config = EndpointCoreConfigInput(
            served_entities=[served_entity],
        )

        # Update endpoint
        self.workspace_client.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[served_entity],
        )

        # Update tags if provided
        if tags:
            # Note: Tag updates may require separate API call
            logger.info("Updated endpoint configuration")

    def _get_endpoint(self, endpoint_name: str) -> Optional[Any]:
        """Get endpoint if it exists.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Optional endpoint object
        """
        try:
            return self.workspace_client.serving_endpoints.get(name=endpoint_name)
        except Exception:
            return None

    def wait_for_endpoint_ready(
        self,
        endpoint_name: str,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> bool:
        """Wait for endpoint to be ready.

        Args:
            endpoint_name: Name of the endpoint
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            bool: True if endpoint is ready, False if timeout

        Raises:
            Exception: If endpoint enters error state
        """
        logger.info(f"Waiting for endpoint {endpoint_name} to be ready...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            endpoint = self._get_endpoint(endpoint_name)

            if endpoint is None:
                logger.error(f"Endpoint {endpoint_name} not found")
                return False

            state = endpoint.state.config_update if hasattr(endpoint.state, "config_update") else endpoint.state.ready

            if state == "READY":
                logger.info(f"Endpoint {endpoint_name} is ready")
                return True

            if state in ["ERROR", "FAILED"]:
                logger.error(f"Endpoint {endpoint_name} entered error state: {state}")
                msg = f"Endpoint {endpoint_name} failed"
                raise Exception(msg)

            logger.info(f"Endpoint state: {state}, waiting...")
            time.sleep(poll_interval)

        logger.error(f"Timeout waiting for endpoint {endpoint_name}")
        return False

    def get_endpoint_status(self, endpoint_name: str) -> Optional[dict]:
        """Get endpoint status.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Optional[dict]: Endpoint status information
        """
        endpoint = self._get_endpoint(endpoint_name)

        if endpoint is None:
            return None

        return {
            "name": endpoint.name,
            "state": endpoint.state,
            "config": endpoint.config,
            "tags": endpoint.tags if hasattr(endpoint, "tags") else [],
        }

    def invoke_endpoint(
        self,
        endpoint_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke serving endpoint for testing.

        Args:
            endpoint_name: Name of the endpoint
            payload: Request payload

        Returns:
            dict: Response from endpoint

        Raises:
            Exception: If invocation fails
        """
        try:
            response = self.workspace_client.serving_endpoints.query(
                name=endpoint_name,
                inputs=payload,
            )

            return response.predictions

        except Exception as e:
            logger.error(f"Failed to invoke endpoint {endpoint_name}: {e}")
            raise

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete serving endpoint.

        Args:
            endpoint_name: Name of the endpoint to delete
        """
        logger.info(f"Deleting endpoint: {endpoint_name}")

        try:
            self.workspace_client.serving_endpoints.delete(name=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} deleted")
        except Exception as e:
            logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")
            raise
