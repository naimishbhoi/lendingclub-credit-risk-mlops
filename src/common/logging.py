"""
Module: src.common.logging
Purpose: Project-wide structured logging and configuration.
"""

import logging
from logging import Logger
from pathlib import Path
from typing import Optional
from uuid import uuid4


_LOGGERS = {}
_RUN_ID: Optional[str] = None

def set_run_id(run_id: Optional[str] = None) -> str:
    """Set a run_id for the contextual logging."""

    global _RUN_ID
    _RUN_ID = run_id or str(uuid4())

    return _RUN_ID


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID or "_"
        return True
    

def _build_formatter() -> logging.Formatter:
    fmt = (
        "[%(asctime)s] [%(levelname)s]"
        "[%(name)s] [%(run_id)s] %(message)s"
    )

    return logging.Formatter(fmt = fmt, datefmt = "%Y-%m-%d %H:%M:%S")


def _config_root_logger(
        level: str, enable_file: bool, log_dir: str, logger_name: str,
) -> Logger:
    """Configure the root logger with the specified settings."""

    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    formatter = _build_formatter()
    context_filter = ContextFilter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)

    # File Handler
    if enable_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(Path(log_dir) / f"{logger_name}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(
        name: str,
        *,
        level:str = "INFO",
        enable_file: bool = False,
        log_dir: Optional[str] = None,
) -> Logger:
    """Get a configured logger instance."""

    if name not in _LOGGERS:
        _LOGGERS[name] = _config_root_logger(
            level = level,
            enable_file = enable_file,
            log_dir = log_dir,
            logger_name = name
        )

    return _LOGGERS[name]