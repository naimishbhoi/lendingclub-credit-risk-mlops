"""
Module: pipelines.validate_pipeline
Purpose: Orchestrate the dataset and schema validation pipeline.
"""

import sys
from src.common.cli import build_base_parser
from src.common.config import load_app_config, ConfigLoadError
from src.common.logging import get_logger, set_run_id

def main() -> None:
    """Main function to execute the validation pipeline."""
    parser = build_base_parser("Dataset & Schema Validation Pipeline")
    args = parser.parse_args()

    try:
        config = load_app_config(args.config_dir)
        run_id = set_run_id()
        logger = get_logger(__name__)

        logger = get_logger(
            __name__,
            level = config.logging.level,
            enable_file = config.logging.enable_file,
            log_dir = config.logging.log_dir,
        )

        logger.info("Validation pipeline config loaded successfully")
        logger.info(f"Run ID: {run_id}")
        logger.info("Run completed successfully (placeholder)")

    
    except ConfigLoadError as e:
        logger.exception("Failed to load configuration")
        sys.exit(1)


if main == "__main__":
    main()