"""
Module: src.features.context
Purpose: Define the runtime context for feature engineering.
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from src.common.logging import get_logger

VALID_MODES = frozenset({"train", "test", "inference"})
VALID_METADATA_NAMESPACES = frozenset(
    {
        "pipeline",
        "transformations",
        "monitoring",
    }
)


@dataclass(slots=True)
class TransformationContext:
    """
    Shared runtime context for feature engineering pipelines.

    This object should remain lightweight:
    - runtime mode
    - artifacts root
    - run metadata
    - configs / contracts snapshots
    - shared logger
    """

    mode: str
    artifacts_root: Path

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    random_seed: int = 42

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    logger: logging.Logger = field(init=False, repr=False)

    # Runtime Config Snapshots
    feature_config: Mapping[str, Any] = field(default_factory=dict)
    feature_contract: Mapping[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            namespace: {} for namespace in VALID_METADATA_NAMESPACES
        }
    )

    # ------------------------------------------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.logger = get_logger(type(self).__name__)

        if self.mode is None or not str(self.mode).strip():
            raise ValueError("Mode cannot be None or empty.")

        self.mode = str(self.mode).strip().lower()

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. "
                f"Valid modes are: {sorted(VALID_MODES)}."
            )

        if self.artifacts_root is None or not str(self.artifacts_root).strip():
            raise ValueError("Artifacts root cannot be None or empty.")

        self.artifacts_root = Path(self.artifacts_root).expanduser().resolve()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        self.feature_config = MappingProxyType(deepcopy(dict(self.feature_config)))
        self.feature_contract = MappingProxyType(deepcopy(dict(self.feature_contract)))

        for namespace in VALID_METADATA_NAMESPACES:
            self.metadata.setdefault(namespace, {})

        self.logger.info(
            "Initialized TransformationContext | "
            f"mode: {self.mode} | "
            f"run_id: {self.run_id} | "
            f"artifacts_root: {self.artifacts_root}"
        )

    # ------------------------------------------------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------------------------------------------------
    @property
    def is_train(self) -> bool:
        return self.mode == "train"

    @property
    def is_test(self) -> bool:
        return self.mode == "test"

    @property
    def is_inference(self) -> bool:
        return self.mode == "inference"

    @property
    def artifact_root(self) -> Path:
        """Backward-compatible alias for artifacts_root."""
        return self.artifacts_root

    # ------------------------------------------------------------------------------------------------------------
    # Metadata Management
    # ------------------------------------------------------------------------------------------------------------
    def add_metadata(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Store metadata generated during transformations and pipeline execution.
        """
        self._validate_namespace(namespace)

        self.metadata[namespace][key] = deepcopy(value)

    def get_metadata(
        self,
        namespace: str,
        key: str,
        default: Optional[Any] = None,
    ) -> Any:
        """
        Retrieve metadata by key, with optional default if not found.
        """
        self._validate_namespace(namespace)

        return self.metadata[namespace].get(key, default)

    def has_metadata(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """
        Check if metadata exists for a given key.
        """
        self._validate_namespace(namespace)

        return key in self.metadata[namespace]

    def get_namespace_metadata(
        self,
        namespace: str,
    ) -> dict[str, Any]:
        """
        Retrieve all metadata for a given namespace.
        """
        self._validate_namespace(namespace)

        return self.metadata[namespace].copy()

    def clear_metadata(
        self,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Clear metadata for a specific namespace.
        """
        self._validate_namespace(namespace)

        self.metadata[namespace].clear()

    # ------------------------------------------------------------------------------------------------------------
    # Logging Helpers
    # ------------------------------------------------------------------------------------------------------------
    def log(self, message: str) -> None:
        """
        Safe logging helper.
        """
        if self.logger:
            self.logger.info(message)

    # ------------------------------------------------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if namespace not in VALID_METADATA_NAMESPACES:
            raise ValueError(
                f"Invalid metadata namespace '{namespace}'. "
                f"Valid namespaces are: {sorted(VALID_METADATA_NAMESPACES)}."
            )

    # ------------------------------------------------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', "
            f"run_id='{self.run_id}', "
            f"artifacts_root='{self.artifacts_root}')"
        )
