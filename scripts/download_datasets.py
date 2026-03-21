"""
Module: scripts.download_data
Purpose: Download LendingClub dataset via Kaggle API into data/raw.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.common.config import load_app_config
from src.common.logging import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Configuration Model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DownloadConfig:
    dataset_slug: str
    raw_dir: Path
    kaggle_binary: str = "kaggle"
    dataset_version: Optional[str] = None


# ----------------------------------------------------------------------
# Domain Exception
# ----------------------------------------------------------------------
class DownloadError(RuntimeError):
    """Domain-specific exception for dataset download failures."""

    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


# ----------------------------------------------------------------------
# Environment Validation
# ----------------------------------------------------------------------
def _has_env_auth() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))


def _has_file_auth() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def validate_environment(config: DownloadConfig) -> None:
    """
    Validate:
    - Kaggle CLI availability
    - Authentication configuration (env or file)
    """

    # CLI availability
    if shutil.which(config.kaggle_binary) is None:
        raise DownloadError(
            "Kaggle CLI is not available in the PATH.",
            metadata={"binary": config.kaggle_binary},
        )

    # Authentication presence
    env_auth = _has_env_auth()
    file_auth = _has_file_auth()

    if not env_auth and not file_auth:
        raise DownloadError(
            "Kaggle authentication configuration not found. "
            "Set environment variables (KAGGLE_USERNAME and KAGGLE_KEY) "
            "or provide a kaggle.json file.",
            metadata={
                "env_auth": env_auth,
                "file_auth": file_auth,
            },
        )


# ----------------------------------------------------------------------
# Raw Directory Immutability
# ----------------------------------------------------------------------
def validate_raw_dir_immutability(raw_dir: Path) -> None:
    """Ensure raw data directory exists and is empty."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    if any(raw_dir.iterdir()):
        raise DownloadError(
            f"Raw directory '{raw_dir}' is not empty. "
            "Abort to preserve data immutability.",
            metadata={"raw_dir": str(raw_dir)},
        )


# ----------------------------------------------------------------------
# Download Utilities
# ----------------------------------------------------------------------
def build_download_command(config: DownloadConfig) -> List[str]:
    command = [
        config.kaggle_binary,
        "datasets",
        "download",
        "-d",
        config.dataset_slug,
        "-p",
        str(config.raw_dir),
        "--unzip",
    ]

    if config.dataset_version:
        command.extend(["--version", config.dataset_version])

    return command


def download_dataset(config: DownloadConfig) -> None:
    command = build_download_command(config)

    logger.info(
        "Starting dataset download.",
        extra={
            "dataset_slug": config.dataset_slug,
            "raw_dir": str(config.raw_dir),
            "dataset_version": config.dataset_version,
        },
    )

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
        logger.info("Kaggle CLI output:\n%s", result.stdout)

    except subprocess.CalledProcessError as e:
        raise DownloadError(
            "Dataset download via Kaggle CLI failed.",
            metadata={
                "returncode": e.returncode,
                "output": e.stdout,
            },
        ) from e

    logger.info("Dataset successfully downloaded to %s", config.raw_dir)


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def main() -> None:
    try:
        # Load project configuration
        app_config = load_app_config("configs")

        dataset_source = app_config.data.dataset_source

        config = DownloadConfig(
            dataset_slug=dataset_source.dataset_slug,
            raw_dir=Path(app_config.data.raw_data_path),
            dataset_version=dataset_source.dataset_version,
        )

        validate_environment(config)
        validate_raw_dir_immutability(config.raw_dir)
        download_dataset(config)

    except DownloadError as e:
        logger.error(
            "Download failed.",
            extra={"metadata": getattr(e, "metadata", {})},
            exc_info=True,
        )
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected failure in download script.")
        sys.exit(1)

    logger.info("Download completed successfully.")


if __name__ == "__main__":
    main()
