"""
Module: pipelines.train_pipeline
Purpose: End-to-end training pipeline for the LendingClub credit risk model.
"""

import sys

from src.common.cli import build_base_parser
from src.common.config import load_app_config
from src.common.logging import get_logger, set_run_id
from src.common.exceptions import MLSystemError, PipelineError

def main() -> None:
    """Main function to execute the training pipeline."""
    parser = build_base_parser("LendingClub Credit Risk Model Training Pipeline")
    args = parser.parse_args()

    run_id = set_run_id()
    logger = get_logger(__name__)

    try:
        config = load_app_config(args.config_dir)

        logger = get_logger(
            __name__,
            level = config.logging.level,
            enable_file = config.logging.enable_file,
            log_dir = config.logging.log_dir,
        )

        logger.info("Training pipeline config loaded successfully")

        logger.info("Run completed successfully (placeholder)")
    
    
    except MLSystemError as e:
        logger.error(
            f"Pipeline failed: {e.__class__.__name__} - {str(e)}",
            extra={"metadata": e.metadata},
            exc_info=False,
        )
        sys.exit(1)

        # raise PipelineError(
        #     "Training pipeline execution failed",
        #     cause=e
        # ) from e

 
if __name__ == "__main__":
    main()