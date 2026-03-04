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
# Schema Alignment Diagnostics
# ---------------------------------------------------------------------
def _log_schema_alignment(
    df: pd.DataFrame,
    columns_spec: Dict,
    logger,
    enforce_strict: bool = False,
) -> None:
    """Log schema alignment diagnostics and optionally fail on unexpected columns."""
    contract_cols = set(columns_spec.keys())
    data_cols = set(df.columns)

    missing_in_data = sorted(contract_cols - data_cols)
    unexpected_in_data = sorted(data_cols - contract_cols)

    logger.info(
        "Schema alignment check",
        extra={
            "contract_column_count": len(contract_cols),
            "data_column_count": len(data_cols),
            "missing_in_data": missing_in_data,
            "unexpected_in_data": unexpected_in_data,
        },
    )

    if enforce_strict and unexpected_in_data:
        raise DataValidationError(
            "Unexpected columns detected in dataset.",
            metadata={"unexpected_columns": unexpected_in_data},
        )


# ---------------------------------------------------------------------
# Canonicalization Layer
# ---------------------------------------------------------------------
def _canonicalize_numeric_domains(df: pd.DataFrame, logger) -> None:
    """
    Clean known invalid numeric values before constraint checks.
    Dataset-specific canonicalization logic.
    """
    # ---- DTI cleaning ----
    if "dti" in df.columns:
        dti = df["dti"]

        invalid_mask = dti.notna() & ((dti < 0) | (dti == 999) | (dti > 200))

        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            logger.info(
                "Canonicalizing invalid dti values to NA.",
                extra={
                    "column": "dti",
                    "invalid_count": invalid_count,
                },
            )

            df.loc[invalid_mask, "dti"] = pd.NA


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
            df[column] = df[column].astype("string").str.strip()

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

    if spec.get("enforce_non_null", False):
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

    label_spec = contract.get("label")
    label_col = label_spec["column"] if label_spec else None

    if primary_key_col not in columns_spec:
        raise DataValidationError(
            f"Primary key column '{primary_key_col}' missing from column specification."
        )

    if event_time_col not in columns_spec:
        raise DataValidationError(
            f"Event time column '{event_time_col}' missing from column specification."
        )

    if label_spec and label_col not in columns_spec:
        raise DataValidationError(
            f"Label column '{label_col}' missing from column specification."
        )

    # ------------------------------
    # Schema alignment check
    # ------------------------------
    _log_schema_alignment(
        df=df,
        columns_spec=columns_spec,
        logger=logger,
        enforce_strict=constraints.get("enforce_strict_column_set", False),
    )

    # ------------------------------
    # Required column check
    # ------------------------------
    required_cols = [
        column for column, spec in columns_spec.items() if spec.get("required", False)
    ]

    _assert_required_columns(df, required_cols)

    # ------------------------------
    # Existence checks
    # ------------------------------
    _validate_column_exists(df, primary_key_col)
    _validate_column_exists(df, event_time_col)

    if label_spec:
        _validate_column_exists(df, label_col)

    # ------------------------------
    # Primary key checks
    # ------------------------------
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

    # ------------------------------
    # Casting
    # ------------------------------
    for column, spec in columns_spec.items():
        if column not in df.columns:
            continue  # Optional column missing is allowed

        if spec["type"] not in SUPPORTED_TYPES:
            raise DataValidationError(
                f"Unsupported type '{spec['type']}' for column '{column}'"
            )

        _cast_column(df, column, spec)

    # Canonicalization check
    _canonicalize_numeric_domains(df, logger)

    # ------------------------------
    # Column constraint validation
    # ------------------------------
    for column, spec in columns_spec.items():
        if column not in df.columns:
            continue

        _validate_column_constraints(df, column, spec)

        if spec.get("unique", False):
            if df[column].duplicated().any():
                raise DataValidationError(
                    f"Column '{column}' is unique but contains duplicate values."
                )

    # ------------------------------
    # Label validation
    # ------------------------------
    if label_spec:
        if label_spec.get("required", False):
            _validate_required_non_null(df, label_col)

        allowed_values = label_spec.get("allowed_values")
        if allowed_values:
            mask = df[label_col].notna() & ~df[label_col].isin(allowed_values)

            if mask.any():
                raise DataValidationError(
                    f"Label column '{label_col}' contains invalid values.",
                    metadata={"allowed_values": allowed_values},
                )

        if df[label_col].nunique(dropna=True) < 2:
            raise DataValidationError(
                f"Label column '{label_col}' insufficient unique values."
            )

    # ------------------------------
    # Event time non-null check
    # ------------------------------
    if event_time_spec.get("enforce_non_null", False):
        if df[event_time_col].isna().any():
            raise DataValidationError(
                f"Event time column '{event_time_col}' contains null values"
            )

    # ------------------------------
    # Event time granularity check
    # ------------------------------
    granularity = event_time_spec.get("granularity")
    if granularity:
        if not pd.api.types.is_datetime64_dtype(df[event_time_col]):
            raise DataValidationError(
                f"Event time column '{event_time_col}' must be datetime type."
            )

        if granularity == "month":
            if not (df[event_time_col].dt.day == 1).all():
                raise DataValidationError(
                    f"Event time column '{event_time_col}' is not monthly granularity."
                )

    # ------------------------------
    # Minimum row constraint
    # ------------------------------
    min_rows = constraints.get("min_rows")
    if min_rows is not None and len(df) < min_rows:
        raise DataValidationError(
            "Dataset does not meet minimum row requirement",
            metadata={"min_rows": min_rows},
        )

    # -----------------------------------
    # Maximum missing ratio constraint
    # -----------------------------------
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
