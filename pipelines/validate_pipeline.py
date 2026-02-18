"""
Module: pipelines.validate_pipeline
Purpose: Orchestrate the dataset and schema validation pipeline.
"""

import sys
import time
import hashlib
import shutil
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd

from src.common.cli import build_base_parser
from src.common.config import load_app_config, save_config_snapshot
from src.common.logging import get_logger, set_run_id
from src.common.exceptions import MLSystemError, PipelineError
from src.ingestion.validate import validate_dataset, load_dataset_contract


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def _discover_raw_csvs(raw_dir: Path) -> List[Path]:
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
    
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise PipelineError(
            "No CSV files found in raw directory.",
            metadata={"raw_dir": str(raw_dir)},
        )
    
    return csv_files
    

# ----------------------------------------------------------------------
# Persist Interim Data
# ----------------------------------------------------------------------
def _persist_interim_data(
    df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    primary_key:str,
    logger,
) -> Path:
    """Persist validated dataset to parquet."""
    if not isinstance(df, pd.DataFrame):
        raise PipelineError("Validated output is not a DataFrame.")

    if df.empty:
        raise PipelineError(
            "Validated dataframe is empty.",
            metadata={"row_count": 0},
        )
    
    if primary_key not in df.columns:
        raise PipelineError(
            "Primary key missing in dataframe.",
            metadata={
                "primary_key": primary_key,
                "columns": list(df.columns),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    df_sorted = df.sort_values(by=primary_key).reset_index(drop=True)
    output_path = output_dir / f"validated_dataset_{run_id}.parquet"
    tmp_path = output_path.with_suffix(".tmp")
    
    if output_path.exists():
        raise PipelineError(
            "Output artifact already exists.",
            metadata={"output_path": str(output_path)},
        )
    
    try:
        df_sorted.to_parquet(tmp_path, index=False, engine="pyarrow")
        tmp_path.replace(output_path)

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

        raise PipelineError(
            "Failed to persist parquet artifact.",
            metadata={"output_path": str(output_path)},
        ) from e
    
    artifact_size_mb = round(output_path.stat().st_size / (1024 ** 2), 3)

    logger.info(
        "Interim artifact written successfully.",
        extra={
            "artifact_path": str(output_path),
            "artifact_size_mb": artifact_size_mb,
        },
    )
    
    return output_path


# ----------------------------------------------------------------------
# Core Orchestration
# ----------------------------------------------------------------------
def run_validation_pipeline(config, run_id: str) -> Path:
    """Execute dataset validation pipeline and returns the persisted artifact path."""
    logger = get_logger(__name__)
    pipeline_start = time.time()

    # ---------------------------------------
    # Load Configuration
    # ---------------------------------------
    raw_dir = Path(config.data.raw_data_path).resolve()
    contract_path = Path(config.data.dataset_contract_path).resolve()
    artifacts_root = Path(config.paths.artifacts_root).resolve()
    
    interim_root = Path(config.data.interim_data_path).resolve()
    interim_dir = interim_root / run_id
    interim_dir.mkdir(parents=True, exist_ok=True)

    if not contract_path.is_file():
        raise PipelineError(
            "Dataset contract file does not exist.",
            metadata={"contract_path": str(contract_path)},
        )
    
    contract = load_dataset_contract(contract_path)

    contract_version = contract.get("version")
    primary_key_spec = contract.get("primary_key", {})
    primary_key = primary_key_spec.get("column")

    if contract_version is None or primary_key is None:
        raise PipelineError(
            "Dataset contract missing required keys.",
            metadata={
                "contract_path": str(contract_path),
                "contract_keys": list(contract.keys()),
            },
        )
        
    contract_bytes = contract_path.read_bytes()
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()

    logger.info(
        "Starting dataset validation pipeline.",
        extra={
            "run_id": run_id,
            "raw_dir": str(raw_dir),
            "contract_path": str(contract_path),
            "contract_version": contract_version,
            "contract_hash": contract_hash,
        },
    )

    # ---------------------------------------
    # Discover Files
    # ---------------------------------------
    discover_start = time.time()
    csv_files = _discover_raw_csvs(raw_dir)
    discover_duration = round(time.time() - discover_start, 3)

    total_input_size_mb = round(
        sum(p.stat().st_size for p in csv_files) / (1024 ** 2), 3
    )

    logger.info(
        "Discovered raw CSV files",
        extra={
            "file_count": len(csv_files),
            "total_size_mb": total_input_size_mb,
            "duration_seconds": discover_duration,
        },
    )

    # ---------------------------------------
    # Validate Dataset
    # ---------------------------------------
    validation_start = time.time()
    validated_df = validate_dataset(
        csv_paths=csv_files,
        contract_path=contract_path,
        logger_name="data_validation",
    )
    validation_duration = round(time.time() - validation_start, 3)

    if not isinstance(validated_df, pd.DataFrame):
        raise PipelineError(
            "Validated output is not a DataFrame.",
            metadata={"output_type": type(validated_df)},
        )

    logger.info(
        "Dataset validation complete.",
        extra={
            "row_count": len(validated_df),
            "column_count": len(validated_df.columns),
            "duration_seconds": validation_duration,
        },
    )

    # ---------------------------------------
    # Persist Interim Artifact
    # ---------------------------------------
    persistence_start = time.time()
    artifact_path = _persist_interim_data(
        df=validated_df,
        output_dir=interim_dir,
        run_id=run_id,
        primary_key=primary_key,
        logger=logger,
    )
    persistence_duration = round(time.time() - persistence_start, 3)

    logger.info(
        "Persistence complete.",
        extra={
            "interim_path": str(artifact_path),
            "duration_seconds": persistence_duration
        }
    )

    # ---------------------------------------
    # Save Config & Contract Snapshot
    # ---------------------------------------
    metadata_dir = artifacts_root / "metadata" / run_id
    metadata_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(config, metadata_dir)
    shutil.copy2(contract_path, metadata_dir / "dataset_contract.yaml")

    total_duration = round(time.time() - pipeline_start, 3)

    logger.info(
        "Validation pipeline completed successfully.",
        extra={
            "run_id": run_id,
            "total_duration_seconds": total_duration,
        }
    )

    return artifact_path


# ----------------------------------------------------------------------
# Main Orchestration
# ----------------------------------------------------------------------
def main() -> None:
    """CLI entrypoint for dataset validation pipeline."""
    parser = build_base_parser("Dataset & Schema Validation Pipeline")
    args = parser.parse_args()
    
    run_id = set_run_id(str(uuid4()))
    logger = get_logger(__name__)

    try:
        config = load_app_config(args.config_dir)
        run_validation_pipeline(config=config, run_id=run_id)
        sys.exit(0)

    except (MLSystemError, PipelineError) as e:
        logger.error(
            "Pipeline failed with controlled system exception.",
            extra={
                "error_type": e.__class__.__name__,
                "message": str(e),
                "metadata": getattr(e, "metadata", {})
            },
            exc_info=True,
        )
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected failure in validation pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()