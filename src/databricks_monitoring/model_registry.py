"""Model registry management for Unity Catalog."""

from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistry:
    """Unity Catalog model registry management."""

    def __init__(self, catalog: str, schema: str):
        """Initialize model registry.

        Args:
            catalog: Unity Catalog name
            schema: Schema name within catalog
        """
        self.catalog = catalog
        self.schema = schema
        self.client = MlflowClient()

    def _get_full_model_name(self, model_name: str) -> str:
        """Get full model name with catalog and schema.

        Args:
            model_name: Base model name

        Returns:
            str: Full model name (catalog.schema.model_name)
        """
        return f"{self.catalog}.{self.schema}.{model_name}"

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> mlflow.entities.model_registry.ModelVersion:
        """Register model to Unity Catalog.

        Args:
            model_uri: URI of the model to register (e.g., runs:/<run_id>/model)
            model_name: Base model name (without catalog/schema prefix)
            tags: Optional dictionary of tags for the model version
            description: Optional description for the model version

        Returns:
            ModelVersion: Registered model version object
        """
        full_model_name = self._get_full_model_name(model_name)

        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=full_model_name,
        )

        # Set tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=full_model_name,
                    version=model_version.version,
                    key=key,
                    value=str(value),
                )

        # Set description if provided
        if description:
            self.client.update_model_version(
                name=full_model_name,
                version=model_version.version,
                description=description,
            )

        return model_version

    def get_latest_model_version(
        self,
        model_name: str,
        alias: Optional[str] = None,
    ) -> Optional[str]:
        """Get latest model version by alias or stage.

        Args:
            model_name: Base model name
            alias: Optional alias to filter by (e.g., "latest-model", "champion")

        Returns:
            Optional[str]: Model version number or None if not found
        """
        full_model_name = self._get_full_model_name(model_name)

        try:
            if alias:
                # Get version by alias
                model_version = self.client.get_model_version_by_alias(
                    name=full_model_name,
                    alias=alias,
                )
                return model_version.version
            # Get latest version
            versions = self.client.search_model_versions(f"name='{full_model_name}'")
            if versions:
                # Sort by version number (descending) and return latest
                latest = max(versions, key=lambda v: int(v.version))
                return latest.version
            return None
        except Exception:
            return None

    def get_model_metadata(
        self,
        model_name: str,
        version: str,
    ) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Get model metadata for a specific version.

        Args:
            model_name: Base model name
            version: Model version number

        Returns:
            Optional[ModelVersion]: Model version object or None if not found
        """
        full_model_name = self._get_full_model_name(model_name)

        try:
            return self.client.get_model_version(
                name=full_model_name,
                version=version,
            )
        except Exception:
            return None

    def load_model_for_inference(
        self,
        model_name: str,
        version: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Any:
        """Load model for inference.

        Args:
            model_name: Base model name
            version: Optional model version number
            alias: Optional alias (e.g., "latest-model")

        Returns:
            Any: Loaded model object

        Raises:
            ValueError: If neither version nor alias is provided
        """
        if version is None and alias is None:
            msg = "Either version or alias must be provided"
            raise ValueError(msg)

        full_model_name = self._get_full_model_name(model_name)

        if alias:
            model_uri = f"models:/{full_model_name}@{alias}"
        else:
            model_uri = f"models:/{full_model_name}/{version}"

        return mlflow.sklearn.load_model(model_uri)

    def set_model_alias(
        self,
        model_name: str,
        version: str,
        alias: str,
    ) -> None:
        """Set or update alias for a model version.

        Args:
            model_name: Base model name
            version: Model version number
            alias: Alias to set (e.g., "latest-model", "champion", "challenger")
        """
        full_model_name = self._get_full_model_name(model_name)

        self.client.set_registered_model_alias(
            name=full_model_name,
            alias=alias,
            version=version,
        )

    def delete_model_alias(
        self,
        model_name: str,
        alias: str,
    ) -> None:
        """Delete alias from a model.

        Args:
            model_name: Base model name
            alias: Alias to delete
        """
        full_model_name = self._get_full_model_name(model_name)

        self.client.delete_registered_model_alias(
            name=full_model_name,
            alias=alias,
        )

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ) -> mlflow.entities.model_registry.ModelVersion:
        """Transition model to a different stage (deprecated in favor of aliases).

        Args:
            model_name: Base model name
            version: Model version number
            stage: Target stage (e.g., "Staging", "Production", "Archived")

        Returns:
            ModelVersion: Updated model version object

        Note:
            This method is deprecated in favor of model aliases.
            Use set_model_alias instead for Unity Catalog.
        """
        full_model_name = self._get_full_model_name(model_name)

        return self.client.transition_model_version_stage(
            name=full_model_name,
            version=version,
            stage=stage,
        )

    def get_model_uri(
        self,
        model_name: str,
        version: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        """Get model URI for loading.

        Args:
            model_name: Base model name
            version: Optional model version number
            alias: Optional alias

        Returns:
            str: Model URI

        Raises:
            ValueError: If neither version nor alias is provided
        """
        if version is None and alias is None:
            msg = "Either version or alias must be provided"
            raise ValueError(msg)

        full_model_name = self._get_full_model_name(model_name)

        if alias:
            return f"models:/{full_model_name}@{alias}"
        return f"models:/{full_model_name}/{version}"

    def list_model_versions(
        self,
        model_name: str,
        max_results: int = 100,
    ) -> list[mlflow.entities.model_registry.ModelVersion]:
        """List all versions of a model.

        Args:
            model_name: Base model name
            max_results: Maximum number of versions to return

        Returns:
            list[ModelVersion]: List of model versions
        """
        full_model_name = self._get_full_model_name(model_name)

        versions = self.client.search_model_versions(
            filter_string=f"name='{full_model_name}'",
            max_results=max_results,
        )

        # Sort by version number (descending)
        return sorted(versions, key=lambda v: int(v.version), reverse=True)
