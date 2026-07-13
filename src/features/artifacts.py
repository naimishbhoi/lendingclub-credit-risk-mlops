"""
Module: src.features.artifacts
Purpose: Manage persistence of feature engineering artifacts.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.common.exceptions import PipelineError
from src.common.logging import get_logger


class ArtifactManager:
    """
    Centralized artifact persistence layer for feature engineering.

    Handles:
    - joblib persistence
    - JSON persistence
    - parquet persistence
    - schema persistence
    - metadata persistence
    - feature order persistence
    """

    def __init__(
        self,
        artifacts_root: Path,
        run_id: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if artifacts_root is None or not str(artifacts_root).strip():
            raise ValueError("Artifacts root cannot be None or empty.")

        self.artifacts_root = Path(artifacts_root).expanduser().resolve()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        if run_id is None or not str(run_id).strip():
            raise ValueError("Run ID cannot be None or empty.")

        self.run_id = str(run_id).strip()
        self.logger = logger or get_logger(__name__)
        self.run_root = self.artifacts_root / self.run_id

        self._initialize_directories()

    # ------------------------------------------------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------------------------------------------------
    def _initialize_directories(self) -> None:
        """
        Create required artifact directories if they don't exist.
        """
        required_dirs = [
            self.run_root / "encoders",
            self.run_root / "imputers",
            self.run_root / "metadata",
            self.run_root / "schemas",
            self.run_root / "statistics",
            self.run_root / "states",
            self.run_root / "debug",
            self.run_root / "manifests",
        ]

        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured artifact directory exists: {directory}")

    # ---------------------------------------------------------------------------------------------
    # Relative Path Resolution
    # ---------------------------------------------------------------------------------------------
    def _resolve_path(
        self,
        relative_path: str | Path,
    ) -> Path:
        """
        Resolve a relative artifact path.
        """
        path = Path(relative_path)

        if path.is_absolute():
            raise ValueError(
                f"Relative path expected, got absolute path: {relative_path}"
            )

        resolved = (self.run_root / path).resolve()
        root_resolved = self.run_root.resolve()

        try:
            is_relative = resolved.is_relative_to(root_resolved)

        except AttributeError:
            # Python < 3.9 fallback
            is_relative = root_resolved in resolved.parents or resolved == root_resolved

        if not is_relative:
            raise PipelineError(
                f"Invalid relative path: {relative_path}. "
                f"Relative path must remain inside the artifacts root: {self.run_root}"
            )

        return resolved

    # ---------------------------------------------------------------------------------------------
    # JSON Serialization Helper
    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def _json_default(value: Any) -> Any:
        """
        Custom JSON serializer for non-serializable objects.
        """
        if isinstance(value, Path):
            return str(value)

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, (np.integer, np.floating)):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, set):
            return sorted(list(value))

        if isinstance(value, tuple):
            return list(value)

        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    # --------------------------------------------------------------------------------------
    # Joblib Persistence
    # --------------------------------------------------------------------------------------
    def save_joblib(
        self,
        obj: Any,
        relative_path: str | Path,
    ) -> Path:
        """
        Save an object using joblib to the specified relative path under the artifacts root.
        """
        save_path = self._resolve_path(relative_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            joblib.dump(obj, save_path)

        except Exception as e:
            raise PipelineError(f"Failed to save artifact to {save_path}: {e}") from e

        self.logger.info(f"Saved artifact to: {save_path}")

        return save_path

    def load_joblib(
        self,
        relative_path: str | Path,
    ) -> Any:
        """
        Load an object using joblib from the specified relative path under the artifacts root.
        """
        load_path = self._resolve_path(relative_path)

        if not load_path.exists():
            raise PipelineError(f"Artifact not found at path: {load_path}")

        try:
            return joblib.load(load_path)

        except Exception as e:
            raise PipelineError(f"Failed to load artifact from {load_path}: {e}") from e

    # --------------------------------------------------------------------------------------
    # JSON Persistence
    # --------------------------------------------------------------------------------------
    def save_json(
        self,
        data: Any,
        relative_path: str | Path,
    ) -> Path:
        """
        Save a JSON-serializable object to the specified relative path.
        """
        save_path = self._resolve_path(relative_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=self._json_default,
                )

        except Exception as e:
            raise PipelineError(
                f"Failed to save JSON artifact to {save_path}: {e}"
            ) from e

        self.logger.info(f"Saved JSON artifact to: {save_path}")

        return save_path

    def load_json(
        self,
        relative_path: str | Path,
    ) -> Any:
        """
        Load a JSON object from the specified relative path.
        """
        load_path = self._resolve_path(relative_path)

        if not load_path.exists():
            raise PipelineError(f"Artifact not found at path: {load_path}")

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            raise PipelineError(
                f"Failed to load JSON artifact from {load_path}: {e}"
            ) from e

    # --------------------------------------------------------------------------------------
    # Parquet Persistence
    # --------------------------------------------------------------------------------------
    def save_parquet(
        self,
        df: pd.DataFrame,
        relative_path: str | Path,
    ) -> Path:
        """
        Save a pandas DataFrame to a parquet file at the specified relative path.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"save_parquet expects a pandas DataFrame, got {type(df)}")

        save_path = self._resolve_path(relative_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(save_path, index=False)

        except Exception as e:
            raise PipelineError(
                f"Failed to save DataFrame to parquet at {save_path}: {e}"
            ) from e

        self.logger.info(f"Successfully saved DataFrame to: {save_path}")

        return save_path

    def load_parquet(
        self,
        relative_path: str | Path,
    ) -> pd.DataFrame:
        """
        Load a pandas DataFrame from a parquet file at the specified relative path.
        """
        load_path = self._resolve_path(relative_path)

        if not load_path.exists():
            raise PipelineError(f"Artifact not found at path: {load_path}")

        try:
            return pd.read_parquet(load_path)

        except Exception as e:
            raise PipelineError(
                f"Failed to load DataFrame from parquet at {load_path}: {e}"
            ) from e

    # --------------------------------------------------------------------------------------
    # Feature Order Persistence
    # --------------------------------------------------------------------------------------
    def save_feature_order(
        self,
        feature_order: list[str],
    ) -> Path:
        """
        Persist final feature ordering.
        Critical for inference parity.
        """
        if not feature_order:
            raise ValueError("Feature order cannot be empty.")

        if not all(isinstance(feature, str) for feature in feature_order):
            raise ValueError("Feature order must be a list of strings.")

        if len(feature_order) != len(set(feature_order)):
            raise ValueError("Feature order contains duplicate feature names.")

        return self.save_json(
            data=feature_order,
            relative_path="metadata/feature_order.json",
        )

    def load_feature_order(self) -> list[str]:
        """
        Load persisted feature ordering.
        """
        data = self.load_json("metadata/feature_order.json")

        if not isinstance(data, list):
            raise PipelineError(
                f"Invalid feature order format in metadata/feature_order.json. "
                f"Expected a list of feature names, got {type(data)}"
            )

        if not all(isinstance(feature, str) for feature in data):
            raise PipelineError("Feature order must contain only strings.")

        return data

    # --------------------------------------------------------------------------------------
    # Metadata Persistence
    # --------------------------------------------------------------------------------------
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "metadata.json",
    ) -> Path:
        """
        Save pipeline or transformer metadata.
        """
        if not isinstance(metadata, dict):
            raise TypeError(f"Metadata must be a dictionary, got {type(metadata)}")

        return self.save_json(
            data=metadata,
            relative_path=f"metadata/{filename}",
        )

    def load_metadata(
        self,
        filename: str = "metadata.json",
    ) -> Dict[str, Any]:
        """
        Load pipeline or transformer metadata.
        """
        data = self.load_json(f"metadata/{filename}")

        if not isinstance(data, dict):
            raise PipelineError(
                f"Expected metadata to be a dictionary, got {type(data)}"
            )

        return data

    # --------------------------------------------------------------------------------------
    # Schema Persistence
    # --------------------------------------------------------------------------------------
    def save_schema(
        self,
        schema: Dict[str, Any],
        filename: str = "schema.json",
    ) -> Path:
        """
        Save feature schema snapshot.
        """
        if not isinstance(schema, dict):
            raise TypeError(f"Schema must be a dictionary, got {type(schema)}")

        return self.save_json(
            data=schema,
            relative_path=f"schemas/{filename}",
        )

    def load_schema(
        self,
        filename: str = "schema.json",
    ) -> Dict[str, Any]:
        """
        Load feature schema snapshot.
        """
        data = self.load_json(f"schemas/{filename}")

        if not isinstance(data, dict):
            raise PipelineError(f"Expected schema to be a dictionary, got {type(data)}")

        return data

    # --------------------------------------------------------------------------------------
    # Manifest Persistence
    # --------------------------------------------------------------------------------------
    def save_manifest(
        self,
        manifest: Dict[str, Any],
        filename: str = "manifest.json",
    ) -> Path:
        """
        Save feature manifest snapshot.
        """
        if not isinstance(manifest, dict):
            raise TypeError(f"Manifest must be a dictionary, got {type(manifest)}")

        return self.save_json(
            data=manifest,
            relative_path=f"manifests/{filename}",
        )

    def load_manifest(
        self,
        filename: str = "manifest.json",
    ) -> Dict[str, Any]:
        """
        Load feature manifest snapshot.
        """
        data = self.load_json(f"manifests/{filename}")

        if not isinstance(data, dict):
            raise PipelineError(
                f"Expected manifest to be a dictionary, got {type(data)}"
            )

        return data
