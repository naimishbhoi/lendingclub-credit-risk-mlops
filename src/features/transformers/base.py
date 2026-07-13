"""
Module: src.features.transformers.base
Purpose: Define the base interface for all feature transformers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.common.exceptions import PipelineError
from src.common.logging import get_logger
from src.features.artifacts import ArtifactManager
from src.features.context import TransformationContext


class BaseTransformer(ABC):
    """
    Base class for all feature transformers.
    Defines the common lifecycle and utilities shared across
    all feature engineering transformers.

    Provides:
    - fit / transform / fit_transform lifecycle
    - fitted-state tracking
    - input / output column tracking
    - validation utilities
    - metadata persistence
    - transformer state persistence
    - debug artifact persistence
    - transformation summaries
    """

    requires_fit: bool = True
    state_version: int = 1

    ALLOWED_PARAMS = {
        "columns",
        "debug",
    }

    def __init__(
        self,
        name: str,
        context: TransformationContext,
        columns: Optional[list[str]] = None,
        debug: bool = False,
        persist_debug_sample_size: int = 10_000,
    ) -> None:
        # Validate transformer name
        if name is None or not str(name).strip():
            raise ValueError("Transformer name cannot be empty.")
        self.name = str(name).strip()

        # Validate runtime context
        if not isinstance(context, TransformationContext):
            raise TypeError(
                f"context must be a TransformationContext, got {type(context)}"
            )
        self.context = context

        # Validate columns
        if columns is None:
            normalized_columns: list[str] = []
        else:
            if not isinstance(columns, list) or not all(
                isinstance(col, str) for col in columns
            ):
                raise TypeError("columns must be a list of strings.")
            normalized_columns = [str(col).strip() for col in columns]

            if len(normalized_columns) != len(set(normalized_columns)):
                raise ValueError(
                    f"Duplicate column names are not allowed in {self.name}."
                )

            if not normalized_columns and columns:
                raise ValueError("columns list cannot be empty strings.")

        # Validate debug flag
        if not isinstance(debug, bool):
            raise TypeError(f"debug must be a boolean, got {type(debug)}")
        self.debug = debug

        if (
            not isinstance(persist_debug_sample_size, int)
            or persist_debug_sample_size <= 0
        ):
            raise ValueError("persist_debug_sample_size must be a positive integer.")
        self.persist_debug_sample_size = persist_debug_sample_size

        self.columns = normalized_columns
        self.logger = get_logger(self.name)

        self._is_fitted = False

        self.input_columns_: list[str] = []
        self.output_columns_: list[str] = []

        self.metadata_: Dict[str, Any] = {}

        self.last_input_shape_: Optional[Tuple[int, int]] = None
        self.last_output_shape_: Optional[Tuple[int, int]] = None

        self.input_schema_: Dict[str, str] = {}
        self.output_schema_: Dict[str, str] = {}

        self.created_columns_: list[str] = []
        self.dropped_columns_: list[str] = []

        self.artifact_manager = ArtifactManager(
            artifacts_root=self.context.artifacts_root,
            run_id=self.context.run_id,
            logger=self.logger,
        )

    # --------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """
        Learn the parameters of the transformation from the training data.
        Stateless transformers override 'requires_fit' to False.
        """
        start = perf_counter()

        self._validate_input_dataframe(df)
        self._validate_required_columns(df)

        self.input_columns_ = self.columns or df.columns.tolist()
        self.last_input_shape_ = df.shape

        self.input_schema_ = self._capture_schema(df)

        if self.requires_fit:
            try:
                self._fit(df.copy())

            except Exception as e:
                raise PipelineError(f"[{self.name}] Fitting failed: {e}") from e

        self._is_fitted = True

        self._generate_metadata(stage="fit")

        elapsed = perf_counter() - start
        self.metadata_["fit_time_seconds"] = elapsed

        self.persist_metadata_to_context()

        self.logger.info(f"[{self.name}] Fit completed in {elapsed:.2f} seconds.")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to a DataFrame.
        """
        start = perf_counter()

        self._validate_input_dataframe(df)
        self._validate_required_columns(df)

        if self.requires_fit:
            self._validate_fitted()

        self.input_columns_ = self.columns or df.columns.tolist()

        before_shape = df.shape
        self.last_input_shape_ = before_shape

        self.input_schema_ = self._capture_schema(df)

        try:
            transformed_df = self._transform(df.copy())

        except Exception as e:
            raise PipelineError(f"[{self.name}] Transformation failed: {e}") from e

        if not isinstance(transformed_df, pd.DataFrame):
            raise PipelineError(
                f"[{self.name}] _transform must return a pandas DataFrame, got {type(transformed_df)}"
            )

        after_shape = transformed_df.shape
        self.last_output_shape_ = after_shape

        self.output_columns_ = transformed_df.columns.tolist()

        self.output_schema_ = self._capture_schema(transformed_df)

        self._track_columns_lineage(
            before_cols=self.input_columns_,
            after_cols=self.output_columns_,
        )

        self._generate_metadata(stage="transform")

        elapsed = perf_counter() - start
        self.metadata_["transform_time_seconds"] = elapsed

        self.persist_metadata_to_context()

        self._log_transformation_summary(
            before_shape=before_shape, after_shape=after_shape
        )

        self.logger.info(f"[{self.name}] Transform completed in {elapsed:.2f} seconds.")

        if self.debug:
            self._persist_debug_artifact(transformed_df)

        return transformed_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the transformer and return the transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)

    # --------------------------------------------------------------------------------------
    # Abstract Internal Methods
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> None:
        """
        Internal fit implementation. Override for stateful transformers.
        """
        return None

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal transform implementation.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------------------
    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise PipelineError(
                f"[{self.name}] Input must be a pandas DataFrame, got {type(df)}"
            )

        if df.empty:
            raise PipelineError(
                f"[{self.name}] Input DataFrame is empty. Cannot fit/transform on empty data."
            )

        duplicate_columns = df.columns[df.columns.duplicated()].tolist()

        if duplicate_columns:
            raise PipelineError(
                f"[{self.name}] Input DataFrame contains duplicate columns: {duplicate_columns}. "
            )

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Ensure that all required columns are present in the DataFrame.
        """
        if not self.columns:
            return

        available = set(df.columns)

        missing_columns = [column for column in self.columns if column not in available]

        if missing_columns:
            raise PipelineError(
                f"[{self.name}] Missing required columns: {missing_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )

    def _validate_fitted(self) -> None:
        """
        Ensure that the transformer has been fitted before transformation.
        """
        if not self._is_fitted:
            raise PipelineError(
                f"[{self.name}] Transformer has not been fitted yet. Call fit() before transform()."
            )

    def _validate_loaded_state(self) -> None:
        """
        Optional validation hook for loaded state.
        Override in stateful transformers to validate
        state integrity after _set_state is called.
        """
        return None

    # --------------------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------------------
    def _log_transformation_summary(
        self,
        before_shape: Tuple[int, int],
        after_shape: Tuple[int, int],
    ) -> None:
        """
        Log transformation summary.
        """
        self.logger.info(
            f"[{self.name}] Transformation summary | "
            f"Input shape: {before_shape} | "
            f"Output shape: {after_shape} | "
            f"Created columns: {len(self.created_columns_)} | "
            f"Dropped columns: {len(self.dropped_columns_)}"
        )

    # --------------------------------------------------------------------------------------
    # Schema Helpers
    # --------------------------------------------------------------------------------------
    def _capture_schema(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Capture dataframe schema snapshot.
        """
        return {column: str(dtype) for column, dtype in df.dtypes.items()}

    # --------------------------------------------------------------------------------------
    # Column Lineage Tracking
    # --------------------------------------------------------------------------------------
    def _track_columns_lineage(
        self,
        before_cols: list[str],
        after_cols: list[str],
    ) -> None:
        """
        Track columns created and removed by the transformer.
        Useful for
        - debugging
        - explainability
        - pipeline auditing
        """
        before = set(before_cols)
        after = set(after_cols)

        self.created_columns_ = sorted(list(after - before))
        self.dropped_columns_ = sorted(list(before - after))

    # --------------------------------------------------------------------------------------
    # Metadata Persistence
    # --------------------------------------------------------------------------------------
    def _generate_metadata(self, stage: str) -> None:
        """
        Generate transformer metadata snapshot.
        """
        metadata = {
            "transformer_name": self.name,
            "transformer_class": self.__class__.__name__,
            "stage": stage,
            "state_version": self.state_version,
            "requires_fit": self.requires_fit,
            "is_fitted": self._is_fitted,
            "run_id": self.context.run_id,
            "mode": self.context.mode,
            "required_columns": list(self.columns),
            "input_columns": list(self.input_columns_),
            "input_schema": dict(self.input_schema_),
            "last_input_shape": self.last_input_shape_,
            "debug": self.debug,
        }

        if stage != "fit":
            metadata.update(
                {
                    "output_columns": list(self.output_columns_),
                    "output_schema": dict(self.output_schema_),
                    "created_columns": list(self.created_columns_),
                    "dropped_columns": list(self.dropped_columns_),
                    "last_output_shape": self.last_output_shape_,
                }
            )

        self.metadata_ = metadata

    # --------------------------------------------------------------------------------------
    # Metadata Persistence
    # --------------------------------------------------------------------------------------
    def persist_metadata_to_context(self) -> None:
        """
        Persist transformer metadata into the runtime context.
        """
        stage = self.metadata_.get("stage", "unknown")

        transformer_bucket = self.context.metadata.setdefault(stage, {})
        transformer_bucket = transformer_bucket.setdefault(self.name, {})
        transformer_bucket[stage] = dict(self.metadata_)

    def save_metadata(self) -> Optional[Path]:
        """
        Persist transformer metadata as a JSON artifact.
        """
        if not self.metadata_:
            self._generate_metadata(stage="manual_save")

        stage = self.metadata_.get("stage", "unknown")

        filename = (
            f"{self.__class__.__name__}_"
            f"{self.name}_"
            f"{stage}_"
            f"{self.context.run_id}_metadata.json"
        )

        return self.artifact_manager.save_metadata(
            metadata=self.metadata_,
            filename=filename,
        )

    # --------------------------------------------------------------------------------------
    # State Persistence
    # --------------------------------------------------------------------------------------
    def save_state(self, relative_path: Optional[str] = None) -> Path:
        """
        Persist transformer state to joblib artifact.
        Only applicable to stateful transformers.
        """
        if not self.requires_fit:
            raise PipelineError(
                f"[{self.name}] Save state is not applicable to stateful transformers."
            )

        self._validate_fitted()

        if relative_path is None:
            relative_path = (
                f"states/"
                f"{self.__class__.__name__}_{self.name}_state_v{self.state_version}.joblib"
            )

        state = self._get_state()

        save_path = self.artifact_manager.save_joblib(
            obj=state,
            relative_path=relative_path,
        )

        self.context.log(f"[{self.name}] State saved to {save_path}")

        return save_path

    def load_state(self, relative_path: Optional[str] = None) -> None:
        """
        Load transformer state from a persisted joblib artifact.
        """
        if not self.requires_fit:
            raise PipelineError(
                f"[{self.name}] Load state is not applicable to stateless transformers."
            )

        if relative_path is None:
            relative_path = (
                f"states/"
                f"{self.__class__.__name__}_{self.name}_state_v{self.state_version}.joblib"
            )

        state = self.artifact_manager.load_joblib(
            relative_path=relative_path,
        )

        if not isinstance(state, dict):
            raise PipelineError(
                f"Invalid state format {self.name}. "
                f"Expected a dictionary, got {type(state)}"
            )

        version = state.get("version")

        if version != self.state_version:
            raise PipelineError(
                f"Unsupported state version {version} for transformer {self.name}. "
                f"Expected version '{self.state_version}'."
            )

        expected_class = self.__class__.__name__
        loaded_class = state.get("transformer_class")

        if loaded_class != expected_class:
            raise PipelineError(
                f"[{self.name}] State belongs to '{loaded_class}', expected '{expected_class}'. "
            )

        loaded_name = state.get("transformer_name")

        if loaded_name != self.name:
            raise PipelineError(
                f"[{self.name}] State belongs to transformer '{loaded_name}', expected '{self.name}'. "
            )

        self._set_state(state)
        self._validate_loaded_state()

        self._is_fitted = True

        self._generate_metadata(stage="load_state")
        self.persist_metadata_to_context()

        self.context.log(f"[{self.name}] State loaded successfully.")

    def _get_state(self) -> Dict[str, Any]:
        """
        Return serializable transformer state.
        Override in stateful transformers.
        """
        return {
            "version": self.state_version,
            "transformer_class": self.__class__.__name__,
            "transformer_name": self.name,
            "columns": list(self.columns),
            "debug": self.debug,
            "requires_fit": self.requires_fit,
            "input_columns": list(self.input_columns_),
            "output_columns": list(self.output_columns_),
            "input_schema": dict(self.input_schema_),
            "output_schema": dict(self.output_schema_),
            "created_columns": list(self.created_columns_),
            "dropped_columns": list(self.dropped_columns_),
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore transformer state from loaded state dictionary.
        Override in stateful transformers.
        """
        self.columns = list(state.get("columns", self.columns))
        self.debug = bool(state.get("debug", self.debug))
        self.input_columns_ = list(state.get("input_columns", []))
        self.output_columns_ = list(state.get("output_columns", []))
        self.input_schema_ = dict(state.get("input_schema", {}))
        self.output_schema_ = dict(state.get("output_schema", {}))
        self.created_columns_ = list(state.get("created_columns", []))
        self.dropped_columns_ = list(state.get("dropped_columns", []))

    # --------------------------------------------------------------------------------------
    # Debug Artifacts
    # --------------------------------------------------------------------------------------
    def _persist_debug_artifact(self, df: pd.DataFrame) -> None:
        """
        Persist transformed output for debugging purposes.
        """
        try:
            if len(df) > self.persist_debug_sample_size:
                debug_df = df.sample(
                    n=self.persist_debug_sample_size,
                    random_state=self.context.random_seed,
                )

            else:
                debug_df = df

            path = self.artifact_manager.save_parquet(
                df=debug_df,
                relative_path=(
                    f"debug/" f"{self.__class__.__name__}_{self.name}.parquet"
                ),
            )

            self.logger.info(f"[{self.name}] Debug artifact saved to {path}")

        except Exception as e:
            self.logger.warning(f"[{self.name}] Failed to save debug artifact: {e}")

    # --------------------------------------------------------------------------------------
    # sklearn-like API
    # --------------------------------------------------------------------------------------
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """
        Return minimal sklearn-compatible parameter snapshot.
        """
        return {
            "name": self.name,
            "columns": list(self.columns),
            "debug": self.debug,
            "persist_debug_sample_size": self.persist_debug_sample_size,
        }

    def set_params(self, **params: Any) -> BaseTransformer:
        """
        Minimal sklearn-compatible parameter setter.
        """
        reset_required = False
        for key, value in params.items():

            if key not in self.ALLOWED_PARAMS:
                raise ValueError(
                    f"Parameter '{key}' cannot be modified."
                    f" Allowed parameters are: {self.ALLOWED_PARAMS}."
                )

            if key == "columns":
                if value is None:
                    value = []

                elif not isinstance(value, list) or not all(
                    isinstance(column, str) for column in value
                ):
                    raise TypeError("columns must be a list of strings.")
                value = [col.strip() for col in value if str(col).strip()]
                if len(value) != len(set(value)):
                    raise ValueError("Duplicate column names are not allowed.")
                reset_required = True

            elif key == "debug":
                if not isinstance(value, bool):
                    raise TypeError("debug must be a boolean.")

            setattr(self, key, value)

        if reset_required:
            self._is_fitted = False
            self.metadata_.clear()
            self.input_columns_.clear()
            self.output_columns_.clear()
            self.input_schema_.clear()
            self.output_schema_.clear()
            self.created_columns_.clear()
            self.dropped_columns_.clear()
            self.last_input_shape_ = None
            self.last_output_shape_ = None

        return self

    # --------------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """
        Check if the transformer has been fitted.
        """
        return self._is_fitted

    # --------------------------------------------------------------------------------------
    # Representation
    # --------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"is_fitted={self._is_fitted}, "
            f"requires_fit={self.requires_fit}, "
            f"debug={self.debug})"
        )
