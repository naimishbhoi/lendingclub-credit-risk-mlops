"""
Module: src.common.exceptions
Purpose: Custom exception hierarchy for the ML system.
"""

from typing import Any, Mapping, Optional


class MLSystemError(Exception):
    """Base class for all exception and domain-specific errors in the ML system."""

    def __init__(
            self,
            message: str,
            *,
            metadata: Optional[Mapping[str, Any]] = None,
            cause: Optional[Exception] = None,
    ):
        """
        Initialize the MLSystemError.
        args:
            message: A human-readable message describing the error.
            metadata: Optional directory of additional information about the error, i.e. error codes, relevant parameters, ertc.
        """
        super().__init__(message)
        self.metadata = metadata or {}
        self.__cause__ = cause


### ----------------------------------------
# Configuration Errors
### ----------------------------------------
class ConfigError(MLSystemError):
    """Raised when configuration loading or validation fails."""


### ----------------------------------------
# Data Errors
### ----------------------------------------
class DataError(MLSystemError):
    """Base class for all data-related failures."""

class DataIngestionError(DataError):
    """Raised when raw data ingestion fails."""

class DataValidationError(DataError):
    """Raised when dataset violates expected schema or quality checks."""

class FeatureGenerationError(DataError):
    """Raised during feature engineering failures."""


### ----------------------------------------
# Training  & Evaluation
### ----------------------------------------
class TrainingError(MLSystemError):
    """Raised when model training fails."""

class EvaluationError(MLSystemError):
    """Raised when model evaluation or checks fails."""


### ----------------------------------------
# Registry & Inference
### ----------------------------------------
class RegistryError(MLSystemError):
    """Raised when model registry operations fail."""

class InferenceError(MLSystemError):
    """Raised during batch or online inference failures."""


### ----------------------------------------
# Monitoring & Pipelines
### ----------------------------------------
class MonitoringError(MLSystemError):
    """Raised during monitoring or alerting failures."""

class PipelineError(MLSystemError):
    """Raised when a pipeline orchestration step fails."""