"""
Module: src.ingestion.validate
Purpose: Dataset validation for ML ingestion pipelines.
"""

import time
from typing import Dict, List

import pandas as pd

from src.common.exceptions import DataValidationError
from src.common.logging import get_logger

# ---------------------------------------------------------------------
# Supported contract data types
# ---------------------------------------------------------------------
SUPPORTED_TYPES = {
    "string",
    "float",
    "integer",
    "datetime",
    "categorical",
}


# ---------------------------------------------------------------------
# Column-level validation helpers
# ---------------------------------------------------------------------
def _assert_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Ensure all required columns are present in the DataFrame."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}",
            metadata={"missing_columns": missing},
        )


def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Check if a column exists in the DataFrame."""
    if column not in df.columns:
        raise DataValidationError(f"Expected column '{column}' not found in dataset")


def _validate_required_non_null(df: pd.DataFrame, column: str) -> None:
    if df[column].isna().any():
        raise DataValidationError(f"Required column '{column}' contains null values.")


# ---------------------------------------------------------------------
# Casting  helpers
# ---------------------------------------------------------------------
def _cast_column(df: pd.DataFrame, column: str, spec: Dict) -> None:
    """Cast a single column deterministically before constraint checks."""
    col_type = spec["type"]

    try:
        if col_type == "datetime":
            fmt = spec.get("format")
            if fmt:
                df[column] = pd.to_datetime(df[column], format=fmt, errors="raise")
            else:
                df[column] = pd.to_datetime(df[column], errors="raise")

        elif col_type == "float":
            df[column] = pd.to_numeric(df[column], errors="raise").astype("Float64")

        elif col_type == "integer":
            df[column] = pd.to_numeric(df[column], errors="raise").astype("Int64")

        elif col_type in {"string", "categorical"}:
            df[column] = df[column].astype("string")

    except Exception as e:
        raise DataValidationError(
            f"Type casting failed for column '{column}'",
        ) from e


# ---------------------------------------------------------------------
# Constraint validation helper
# ---------------------------------------------------------------------
def _validate_column_constraints(df: pd.DataFrame, column: str, spec: Dict) -> None:
    """Validate column constraints as specified in the dataset contract."""
    col_type = spec["type"]

    if spec.get("required", False):
        _validate_required_non_null(df, column)

    if col_type in {"float", "integer"}:
        if "min" in spec:
            mask = df[column].notna() & (df[column] < spec["min"])
            if mask.any():
                raise DataValidationError(
                    f"Column '{column}' violates minimum value constraint",
                    metadata={"min": spec["min"]},
                )

        if "max" in spec:
            mask = df[column].notna() & (df[column] > spec["max"])
            if mask.any():
                raise DataValidationError(
                    f"Column '{column}' violates maximum value constraint",
                    metadata={"max": spec["max"]},
                )

    if col_type == "string":
        lengths = df[column].str.len()

        if "min_length" in spec:
            mask = df[column].notna() & lengths.lt(spec["min_length"])
            if mask.any():
                raise DataValidationError(
                    f"Column '{column}' violates minimum length constraint",
                )

        if "max_length" in spec:
            mask = df[column].notna() & lengths.gt(spec["max_length"])
            if mask.any():
                raise DataValidationError(
                    f"Column '{column}' violates maximum length constraint",
                )

    if col_type == "categorical":
        allowed_values = spec.get("allowed_values")
        if allowed_values:
            mask = df[column].notna() & ~df[column].isin(allowed_values)
            if mask.any():
                raise DataValidationError(
                    f"Column '{column}' contains invalid categorical values",
                    metadata={"allowed_values": allowed_values},
                )


# ---------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------
def validate_dataframe(
    df: pd.DataFrame,
    contract: Dict,
    logger_name: str = "data_validation",
) -> pd.DataFrame:
    """
    Validate a DataFrame against the dataset contract.
    Returns validated DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")

    if df.empty:
        raise DataValidationError("Input DataFrame is empty.")

    df = df.copy(deep=True)

    logger = get_logger(logger_name)
    start_time = time.time()

    logger.info(f"Using dataset contract version {contract['version']}")

    columns_spec = contract["columns"]
    constraints = contract.get("constraints", {})

    primary_key_spec = contract["primary_key"]
    primary_key_col = primary_key_spec["column"]

    event_time_spec = contract["event_time"]
    event_time_col = event_time_spec["column"]

    if event_time_col not in columns_spec:
        raise DataValidationError(
            f"Event time column '{event_time_col}' missing from column specification."
        )

    required_cols = [
        column for column, spec in columns_spec.items() if spec.get("required", False)
    ]

    # Required column check
    _assert_required_columns(df, required_cols)

    # Existence checks
    _validate_column_exists(df, primary_key_col)
    _validate_column_exists(df, event_time_col)

    # Log unexpected columns
    unexpected = set(df.columns) - set(columns_spec.keys())
    if unexpected:
        logger.warning(
            "Unexpected columns detected in dataset.",
            extra={"unexpected_columns": list(unexpected)},
        )

    # Primary key checks
    if primary_key_spec.get("enforce_non_null", False):
        if df[primary_key_col].isna().any():
            raise DataValidationError(
                f"Primary key '{primary_key_col}' contains null values.",
            )

    if primary_key_spec.get("enforce_uniqueness", True):
        if df[primary_key_col].duplicated().any():
            raise DataValidationError(
                f"Primary keys '{primary_key_col}' contains duplicate values.",
            )

    # Casting
    for column, spec in columns_spec.items():
        if column not in df.columns:
            continue  # Optional column missing is allowed

        if spec["type"] not in SUPPORTED_TYPES:
            raise DataValidationError(
                f"Unsupported type '{spec['type']}' for column '{column}'"
            )

        _cast_column(df, column, spec)

    # Constraint validation
    for column, spec in columns_spec.items():
        if column not in df.columns:
            continue

        _validate_column_constraints(df, column, spec)

        if spec.get("unique", False):
            if df[column].duplicated().any():
                raise DataValidationError(
                    f"Column '{column}' is unique but contains duplicate values."
                )

    # Event time non-null check
    if event_time_spec.get("enforce_non_null", False):
        if df[event_time_col].isna().any():
            raise DataValidationError(
                f"Event time column '{event_time_col}' contains null values"
            )

    # Minimum row constraint
    min_rows = constraints.get("min_rows")
    if min_rows is not None and len(df) < min_rows:
        raise DataValidationError(
            "Dataset does not meet minimum row requirement",
            metadata={"min_rows": min_rows},
        )

    # Maximum missing ratio constraint
    max_missing_ratio = constraints.get("max_missing_ratio")

    if max_missing_ratio is not None and required_cols:
        missing_ratios = df[required_cols].isna().mean()

        if missing_ratios.max() > max_missing_ratio:
            raise DataValidationError(
                "Dataset exceeds maximum missing ratio constraint",
                metadata={
                    "column": missing_ratios.idxmax(),
                    "missing_ratio": round(missing_ratios.max(), 4),
                    "max_allowed": max_missing_ratio,
                },
            )

    duration = round(time.time() - start_time, 3)

    logger.info(
        "Dataset validation successful",
        extra={
            "rows": len(df),
            "columns": list(df.columns),
            "max_missing_ratio": round(df.isna().mean().max(), 4),
            "duration_seconds": duration,
        },
    )

    return df
