"""MLFlow tracking utilities for comprehensive experiment logging."""

from typing import Any, Optional

import mlflow
import pandas as pd
from mlflow.models import infer_signature


class MLFlowTracker:
    """Centralized MLFlow tracking for model training and inference."""

    def __init__(self, experiment_name: str):
        """Initialize MLFlow tracker.

        Args:
            experiment_name: Name of the MLFlow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_training_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_table_name: str,
        test_table_name: str,
        train_version: Optional[str] = None,
        test_version: Optional[str] = None,
    ) -> None:
        """Log training and test datasets to MLFlow.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            train_table_name: Full table name for training data (catalog.schema.table)
            test_table_name: Full table name for test data (catalog.schema.table)
            train_version: Optional version/timestamp for training data
            test_version: Optional version/timestamp for test data
        """
        # Log training dataset
        train_dataset = mlflow.data.from_pandas(
            train_df,
            source=train_table_name,
            name="training_data",
        )
        mlflow.log_input(train_dataset, context="training")

        # Log test dataset
        test_dataset = mlflow.data.from_pandas(
            test_df,
            source=test_table_name,
            name="test_data",
        )
        mlflow.log_input(test_dataset, context="validation")

        # Log dataset metadata
        mlflow.log_param("train_table", train_table_name)
        mlflow.log_param("test_table", test_table_name)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))

        if train_version:
            mlflow.log_param("train_version", train_version)
        if test_version:
            mlflow.log_param("test_version", test_version)

    def log_model_params(self, params: dict[str, Any]) -> None:
        """Log model hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        mlflow.log_params(params)

    def log_model_metrics(self, metrics: dict[str, float]) -> None:
        """Log model performance metrics.

        Args:
            metrics: Dictionary of metrics (e.g., {'f1_score': 0.85, 'accuracy': 0.82})
        """
        mlflow.log_metrics(metrics)

    def log_model_artifacts(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        y_sample: Optional[pd.Series] = None,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> mlflow.models.model.ModelInfo:
        """Log model with signature and input example.

        Args:
            model: Trained model object
            X_sample: Sample input features for signature inference
            y_sample: Optional sample predictions for signature inference
            artifact_path: Path within the run to log the model
            registered_model_name: Optional name to register model in Model Registry

        Returns:
            ModelInfo: Information about the logged model
        """
        # Infer model signature
        if y_sample is not None:
            signature = infer_signature(X_sample, y_sample)
        else:
            # Predict on sample to infer output signature
            y_pred = model.predict(X_sample)
            signature = infer_signature(X_sample, y_pred)

        # Create input example (first 5 rows)
        input_example = X_sample.head(5)

        # Log model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        return model_info

    def log_inference_data(
        self,
        inference_df: pd.DataFrame,
        inference_table_name: str,
        inference_version: Optional[str] = None,
    ) -> None:
        """Log inference dataset to MLFlow.

        Args:
            inference_df: Inference dataframe
            inference_table_name: Full table name for inference data
            inference_version: Optional version/timestamp for inference data
        """
        # Log inference dataset
        inference_dataset = mlflow.data.from_pandas(
            inference_df,
            source=inference_table_name,
            name="inference_data",
        )
        mlflow.log_input(inference_dataset, context="inference")

        # Log inference metadata
        mlflow.log_param("inference_table", inference_table_name)
        mlflow.log_param("inference_size", len(inference_df))

        if inference_version:
            mlflow.log_param("inference_version", inference_version)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact.

        Args:
            local_path: Path to the file or directory to log
            artifact_path: Optional path within the run to log the artifact
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log text content as an artifact.

        Args:
            text: Text content to log
            artifact_file: Name of the artifact file
        """
        mlflow.log_text(text, artifact_file)

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        """Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file (should end with .json)
        """
        mlflow.log_dict(dictionary, artifact_file)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set tags for the current run.

        Args:
            tags: Dictionary of tags (e.g., {'git_sha': 'abc123', 'environment': 'prod'})
        """
        mlflow.set_tags(tags)

    @staticmethod
    def start_run(run_name: Optional[str] = None, nested: bool = False) -> mlflow.ActiveRun:
        """Start a new MLFlow run.

        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run

        Returns:
            ActiveRun: Active MLFlow run context
        """
        return mlflow.start_run(run_name=run_name, nested=nested)

    @staticmethod
    def end_run() -> None:
        """End the current MLFlow run."""
        mlflow.end_run()

    @staticmethod
    def get_experiment_id(experiment_name: str) -> str:
        """Get experiment ID by name.

        Args:
            experiment_name: Name of the experiment

        Returns:
            str: Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            msg = f"Experiment '{experiment_name}' not found"
            raise ValueError(msg)
        return experiment.experiment_id

    @staticmethod
    def get_run_id() -> Optional[str]:
        """Get the current run ID.

        Returns:
            Optional[str]: Current run ID or None if no active run
        """
        run = mlflow.active_run()
        return run.info.run_id if run else None
