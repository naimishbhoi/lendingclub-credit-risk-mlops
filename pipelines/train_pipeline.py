"""
Module: pipelines.train_pipeline
Purpose: End-to-end training pipeline for the LendingClub credit risk model.
"""

import sys
from src.common.cli import build_base_parser
from src.common.config import load_app_config, ConfigLoadError
from src.common.logging import get_logger, set_run_id

def main() -> None:
    """Main function to execute the training pipeline."""
    parser = build_base_parser("LendingClub Credit Risk Model Training Pipeline")
    args = parser.parse_args()
    logger = get_logger(__name__)

    try:
        config = load_app_config(args.config_dir)
        run_id = set_run_id()

        logger = get_logger(
            __name__,
            level = config.logging.level,
            enable_file = config.logging.enable_file,
            log_dir = config.logging.log_dir,
        )

        logger.info("Training pipeline config loaded successfully")
        logger.info(f"Run ID: {run_id}")
        logger.info("Run completed successfully (placeholder)")
    
    
    except ConfigLoadError as e:
        logger.exception("Failed to load configuration")
        sys.exit(1)

 
if __name__ == "__main__":
    main()