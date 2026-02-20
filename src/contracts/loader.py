"""
Module: src/contracts/loader.py
purpose: Contract loading and governance utilities.
"""

from pathlib import Path
from typing import Dict

import yaml

from src.common.exceptions import DataValidationError

supported_types = {"string", "float", "integer", "datetime", "categorical"}


def _validate_contract_schema(contract: Dict) -> None:
    """Validate dataset contract structure and semantics."""
    required_top_keys = ["version", "columns", "primary_key", "event_time"]
    for key in required_top_keys:
        if key not in contract:
            raise DataValidationError(f"Contract missing required key: '{key}'")

    columns = contract["columns"]
    if not isinstance(columns, dict) or not columns:
        raise DataValidationError("Contract 'columns' must be a non-empty dictionary.")

    for col_name, spec in columns.items():
        if "type" not in spec:
            raise DataValidationError(
                f"Column '{col_name}' missing required 'type' filed."
            )

        if spec["type"] not in supported_types:
            raise DataValidationError(
                f"Unsupported type '{spec['type']}' for column '{col_name}'."
            )

    primary_key_col = contract["primary_key"].get("column")
    if primary_key_col not in columns:
        raise DataValidationError(
            f"Primary key column '{primary_key_col}' not defined in columns."
        )

    event_time_col = contract["event_time"].get("column")
    if event_time_col not in columns:
        raise DataValidationError(
            f"Event time column '{event_time_col}' not defined in columns."
        )


def load_dataset_contract(contract_path: Path) -> Dict:
    """
    Load and validate dataset contract for YAML file.
    """
    if not contract_path.exists():
        raise DataValidationError(f"Dataset contract file not fount at {contract_path}")

    try:
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)

    except Exception as e:
        raise DataValidationError(
            "Failed to parse dataset contract YAML.",
            metadata={"contract_path": str(contract_path)},
        ) from e

    if contract.get("version") != 1:
        raise DataValidationError(
            f"Unsupported contract version: {contract.get('version')}"
        )

    _validate_contract_schema(contract)

    return contract
