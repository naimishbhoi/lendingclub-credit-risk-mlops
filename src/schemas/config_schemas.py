"""
Module: src.schemas.config_schemas
Purpose: Pydantic schemas for validating configuration sections.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DataConfig(BaseModel):
    """Schema for data configuration."""
    
    raw_data_path: str
    processed_data_path: str
    label_window_days: int = Field(..., gt=0, description="Number of days to look back for labeling")

    class Config:
        extra = "forbid"
        frozen = True

class TrainingConfig(BaseModel):
    """Schema for training configuration."""
    
    model_type: str
    radom_state: int = 42

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