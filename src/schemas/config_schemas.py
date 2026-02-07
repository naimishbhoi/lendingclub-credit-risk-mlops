"""
Module: src.schemas.config_schemas
Purpose: Pydantic schemas for validating configuration sections.
"""

from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Schema for data configuration."""

    raw_data_path: str
    processed_data_path: str
    label_window_days: int = Field(
        ..., gt=0, description="Number of days to look back for labeling"
    )

    class Config:
        extra = "forbid"
        frozen = True


class TrainingConfig(BaseModel):
    """Schema for training configuration."""

    model_type: str
    random_state: int = 42

    class Config:
        extra = "forbid"
        frozen = True


class PathsConfig(BaseModel):
    artifacts_root: str
    models_dir: str
    metrics_dir: str

    class Config:
        extra = "forbid"
        frozen = True


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    enable_file: bool = False
    log_dir: str = "logs"

    class Config:
        extra = "forbid"
        frozen = True
