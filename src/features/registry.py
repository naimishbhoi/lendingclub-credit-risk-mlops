"""
Module: src.features.registry
Purpose: Central registry for feature engineering transformers.
"""

from __future__ import annotations

from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Type

from src.common.exceptions import PipelineError
from src.features.context import TransformationContext
from src.features.transformers.base import BaseTransformer


class TransformerRegistry:
    """
    Registry for feature engineering transformers.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseTransformer]] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------------
    # Internal Helper
    # ------------------------------------------------------------------------
    @staticmethod
    def _normalize_name(transformer_type: str) -> str:
        """
        Normalize the transformer type name.
        """
        if transformer_type is None or not str(transformer_type).strip():
            raise ValueError("Transformer type cannot be None or empty.")

        return str(transformer_type).strip().lower()

    # ------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------
    def register(
        self, transformer_type: str, transformer_class: Type[BaseTransformer]
    ) -> None:
        """
        Register a transformer class.
        """
        transformer_type = self._normalize_name(transformer_type)

        if not isinstance(transformer_class, type):
            raise TypeError(
                "Transformer class must be a class type, "
                f"got {type(transformer_class)}"
            )

        if not issubclass(transformer_class, BaseTransformer):
            raise TypeError(
                "Transformer class must inherit from BaseTransformer, "
                f"got {transformer_class}"
            )

        with self._lock:
            if transformer_type in self._registry:
                registered = self._registry[transformer_type]

                raise PipelineError(
                    f"Transformer type '{transformer_type}' is already registered "
                    f"with class '{registered.__name__}'."
                )

            self._registry[transformer_type] = transformer_class

    # ------------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------------
    def get_transformer_class(
        self,
        transformer_type: str,
    ) -> Type[BaseTransformer]:
        """
        Retrieve a registered transformer class.
        """
        transformer_type = self._normalize_name(transformer_type)

        try:
            return self._registry[transformer_type]

        except KeyError as e:
            available = sorted(self._registry.keys())
            available_text = ", ".join(available) if available else "None"

            raise PipelineError(
                f"Unknown transformer type '{transformer_type}'. "
                f"Available transformer types: {available_text}"
            ) from e

    # ------------------------------------------------------------------------
    # Factory Method
    # ------------------------------------------------------------------------
    def create_transformer(
        self,
        transformer_type: str,
        *,
        name: str,
        context: TransformationContext,
        **kwargs: Any,
    ) -> BaseTransformer:
        """
        Create an instance for a registered transformer.
        """
        transformer_class = self.get_transformer_class(transformer_type)

        try:
            return transformer_class(
                name=name,
                context=context,
                **kwargs,
            )

        except (ValueError, TypeError) as e:
            raise PipelineError(
                f"Failed to instantiate transformer '{transformer_type}': {e}"
            ) from e

    # ------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------
    def available_transformers(self) -> list[str]:
        """
        Return all registered transformer types.
        """
        return sorted(self._registry.keys())

    def is_registered(
        self,
        transformer_type: str,
    ) -> bool:
        """
        Check if a transformer type is registered.
        """
        if transformer_type is None:
            return False

        return self._normalize_name(transformer_type) in self._registry

    def clear_registry(self) -> None:
        """
        Clear all registered transformers.
        """
        with self._lock:
            self._registry.clear()

    @property
    def registry(self):
        """
        Return a read-only view of the registry.
        """
        return MappingProxyType(self._registry)

    # ------------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------------
    def __contains__(self, transformer_type: str) -> bool:
        return self.is_registered(transformer_type)

    def __getitem__(self, transformer_type: str) -> type[BaseTransformer]:
        return self.get_transformer_class(transformer_type)

    def __iter__(self):
        return iter(sorted(self._registry))

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        registered = ", ".join(sorted(self._registry))

        return (
            f"{self.__class__.__name__}("
            f"count={len(self)}, "
            f"registered=[{registered}])"
        )
