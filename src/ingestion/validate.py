"""
Module: src.ingestion.validate
Purpose: Validate ingested LendingClub data for quality and consistency.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from src.common.exceptions import DataValidationError
from src.common.logging import get_logger


datatypes = {
    "string": str,
    "float": float,
    "integer": int,
    "datetime": pd.Timestamp,
    "categorical": str,
}


def load_dataset_contract(contract_path: Path) -> Dict:
    """
    Load the dataset contract from the dataset_contract.yaml file.
    """
    if not contract_path.exists():
        raise DataValidationError(
            f"Dataset contract file not found at {contract_path}"
        )
    
    with open(contract_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def _assert_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Assert dataset contains all required columns."""

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}",
            metadata={"missing_columns": missing},
        )
    

def _parse_datetime(series: pd.Series, fmt: str, column: str) -> pd.Series:
    """Parse the datetime column using specified format."""

    try:
        return pd.to_datetime(series, format=fmt, errors="raise")
    
    except Exception as e:
        raise DataValidationError(
            f"Failed to parse datetime column '{column}' with format '{fmt}'"
        )
    

def _validate_column_type(df: pd.DataFrame, column: str, spec: Dict) -> None:
    """Validate a single column's data type and format."""

    expected_type = spec.get("type")
    
    if expected_type not in datatypes:
        raise DataValidationError(
            f"Unsupported data type '{expected_type}' for column '{column}'"
        )
    
    if expected_type == "datetime":
        fmt = spec.get("format")
        
        if not fmt:
            raise DataValidationError(
                f"Missing 'format' for datetime column '{column}'"
            )
        df[column] = _parse_datetime(df[column], fmt, column)

    elif expected_type == "float":
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except Exception as e:
            raise DataValidationError(
                f"Failed to cast column '{column}' to float",
                cause=e,
            )
        
    elif expected_type == "integer":
        try:
            df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
        except Exception as e:
            raise DataValidationError(
                f"Failed to cast column '{column}' to integer",
                cause=e,
            )
        
    # For string and categorical types, we can just ensure they are treated as strings
        

def _validate_column_constraints(df: pd.DataFrame, column: str, spec: Dict) -> None:
    """Validate column constraints as spcified in the dataset contract."""

    if "min" in spec:
        if (df[column] < spec["min"]).any():
            raise DataValidationError(
                f"Column '{column}' violates minimum value constraint",
                metadata={"min": str(spec["min"])},
            )
        
    if "max" in spec:
        if (df[column] > spec["max"]).any():
            raise DataValidationError(
                f"Column '{column}' violates maximum value constraint",
                metadata={"max": str(spec["max"])},
            )
        

def validate_dataset(
        csv_paths: List[Path],
        contract_path: Path,
        logger_name: str = "data_validation",
) -> pd.DataFrame:
    """
    Validate one or more raw csv files against the dataset contract.
    Returns a concatenated, validated Dataframe.
    """

    logger = get_logger(logger_name)

    contract = load_dataset_contract(contract_path)

    columns_spec = contract["columns"]
    constraints = contract.get("constraints", {})
    primary_key = contract["primary_key"]
    event_time = contract["event_time"]["column"]

    required_cols = [
        c for c, s in columns_spec.items() if s.get("required", False)
    ]

    dfs: List[pd.DataFrame] = []

    for path in csv_paths:
        logger.info(f"Loading raw data file: {path}")

        df = pd.read_csv(path)

        _assert_required_columns(df, required_cols)

        for column, spec in columns_spec.items():
            if column not in df.columns:
                continue  # Optional column missing is allowed

            _validate_column_type(df, column, spec)
            _validate_column_constraints(df, column, spec)

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Primary key uniqueness check
    if full_df[primary_key].duplicated().any():
        raise DataValidationError(
            f"Primary key '{primary_key}' contains duplicate values"
        )
    
    # Dataset-level constraints
    min_rows = constraints.get("min_rows")
    if min_rows and len(full_df) < min_rows:
        raise DataValidationError(
            f"Dataset does not meet minimum row requirement",
            metadata={"min_rows": str(min_rows)},
        )
    
    # Event time must exist and be parsable
    if event_time not in full_df.columns:
        raise DataValidationError(
            f"Event time column '{event_time}' is missing from dataset"
        )
    
    logger.info(
        "Dataset validation successful",
        extra={
            "rows": str(len(full_df)),
            "Columns": str(list(full_df.columns)),
        }
    )

    return full_df