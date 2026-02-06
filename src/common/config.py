"""
Modlue: src.common.config
Purpose: Centralized configuration settings for the application. Loading and validation utilities.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ValidationError

from src.schemas.config_schemas import (
    DataConfig,
    PathsConfig,
    TrainingConfig,
    LoggingConfig
)


class ConfigLoadError(Exception):
    """Custom exception for configuration loading errors."""


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""

    if not file_path.exists():
        raise ConfigLoadError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        try:
            return yaml.safe_load(file) or {}

        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Error parsing YAML file: {file_path}") from e


def load_config_dir(config_dir: str) -> Dict[str, Any]:
    """Load all YAML configuration files from a directory and merge them into a single dictionary."""

    base_path = Path(config_dir)

    if not base_path.exists() or not base_path.is_dir():
        raise ConfigLoadError(f"Configuration directory not found: {config_dir}")

    expected_files = [
        "data.yaml",
        "features.yaml",
        "training.yaml",
        "tuning.yaml",
        "evaluation.yaml",
        "inference.yaml",
        "logging.yaml",
        "paths.yaml",
    ]

    configs: Dict[str, Any] = {}

    for file_name in expected_files:
        file_path = base_path / file_name
        key = file_name.replace(".yaml", "")
        configs[key] = _load_yaml(file_path)

    return configs


class AppConfig(BaseModel):
    """Centralized application configuration model."""

    data: DataConfig
    training: TrainingConfig
    paths: PathsConfig
    logging: LoggingConfig

    class Config:
        extra = "forbid"
        frozen = True


def load_app_config(config_dir: str) -> AppConfig:
    """Load and validate the application configuration from a directory."""

    raw_configs = load_config_dir(config_dir)

    required_sections = {"data", "training", "paths", "logging"}
    missing = required_sections - raw_configs.keys()

    if missing:
        raise ConfigLoadError(f"Missing required config sections: {sorted(missing)}")

    try:
        app_config = AppConfig(
            data = raw_configs["data"],
            training = raw_configs["training"],
            paths = raw_configs["paths"],
            logging = raw_configs["logging"]
        )
        return app_config

    except ValidationError as e:
        raise ConfigLoadError("Configuration validation error") from e


def save_config_snapshot(config: AppConfig, output_dir: str) -> None:
    """Save a snapshot of the current configuration to a YAML file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents = True, exist_ok = True)

    config_dict = config.model_dump()

    snapshot_file = output_path / "config_snapshot.yaml"
    with open(snapshot_file, "w", encoding = "utf-8") as file:
        yaml.safe_dump(config_dict, file, sort_keys=True)

    # Additionally, save a hash of the configuration for integrity checks
    config_json = json.dumps(
        config_dict, sort_keys = True, separators = (",", ": ")
    ).encode("utf-8")
    config_hash = hashlib.sha256(config_json).hexdigest()

    hash_file = output_path / "config_hash.txt"
    hash_file.write_text(config_hash, encoding = "utf-8")