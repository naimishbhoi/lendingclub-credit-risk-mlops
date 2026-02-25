"""
Module: src.schemas.config_schemas
Purpose: Pydantic schemas for validating configuration sections.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetSourceConfig(BaseModel):
    """Schema for dataset source configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["kaggle"]
    dataset_slug: str = Field(..., min_length=1)
    dataset_version: Optional[str] = None


class DataConfig(BaseModel):
    """Schema for data configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_data_path: Path
    processed_data_path: Path
    interim_data_path: Path
    dataset_contract_path: Path

    label_window_days: int = Field(
        ..., gt=0, description="Number of days to look back for labeling"
    )

    dataset_source: DatasetSourceConfig


class TrainingConfig(BaseModel):
    """Schema for training configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_type: str
    random_state: int = 42


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifacts_root: Path
    models_dir: Path
    metrics_dir: Path


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    enable_file: bool = False
    log_dir: Path = Path("logs")
