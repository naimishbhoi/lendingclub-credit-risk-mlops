"""
Module: pipelines.retrain_pipeline
Purpose: Orchestrate monitoring-triggered model retraining pipeline.
"""

import sys

from src.common.cli import build_base_parser
from src.common.config import load_app_config
from src.common.exceptions import MLSystemError
from src.common.logging import get_logger, set_run_id


def main() -> None:
    """Main function to execute the retraining pipeline."""
    parser = build_base_parser("Retraining pipeline")
    args = parser.parse_args()

    try:
        config = load_app_config(args.config_dir)
        run_id = set_run_id()

        logger = get_logger(
            __name__,
            level=config.logging.level,
            enable_file=config.logging.enable_file,
            log_dir=config.logging.log_dir,
        )

        logger.info(f"Retraining pipeline started | run_id: {run_id}")
        logger.info("Retraining pipeline config loaded successfully")

        logger.info("Run completed successfully (placeholder)")

    except MLSystemError as e:
        logger = get_logger(__name__)
        logger.error(
            f"pipeline failed: {e.__class__.__name__} - {str(e)}",
            extra={"metadata": e.metadata},
            exc_info=False,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
