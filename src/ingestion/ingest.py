"""
Module: src.ingestion.ingest
Purpose: Load raw LendingClub data into the system for further processing.
"""

from pathlib import Path
from typing import List

import pandas as pd

from src.common.exceptions import PipelineError


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def discover_raw_csvs(raw_dir: Path) -> List[Path]:
    """Discover CSV files in raw data directory."""
    if not raw_dir.exists():
        raise PipelineError(
            "Raw data directory does not exist.",
            metadata={"raw_dir": str(raw_dir)},
        )

    if not raw_dir.is_dir():
        raise PipelineError(
            "Raw data path is not a directory.",
            metadata={"raw_dir": str(raw_dir)},
        )

    csv_files = sorted(p for p in raw_dir.rglob("*.csv") if p.is_file())

    if not csv_files:
        raise PipelineError(
            "No CSV files found in raw directory.",
            metadata={"raw_dir": str(raw_dir)},
        )

    return csv_files


# ----------------------------------------------------------------------
# Raw CSV Loading
# ----------------------------------------------------------------------
def load_raw_csv_files(csv_paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate raw CSV files into a single DataFrame."""
    if not csv_paths:
        raise PipelineError("No CSV files provided for ingestion.")

    dfs = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path, compression="infer", low_memory=False)
            dfs.append(df)
        except Exception as e:
            raise PipelineError(
                "Failed to read raw CSV file.", metadata={"file_path": str(path)}
            ) from e

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df
