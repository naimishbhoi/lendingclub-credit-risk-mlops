"""
Module: pipelines.train_pipeline
Purpose: End-to-end training pipeline for the LendingClub credit risk model.
"""

import sys
from src.common.cli import build_base_parser
from src.common.config import load_config_dir, ConfigLoadError

def main() -> None:
    """Main function to execute the training pipeline."""
    parser = build_base_parser("LendingClub Credit Risk Model Training Pipeline")
    args = parser.parse_args()

    try:
        config = load_config_dir(args.config_dir)
        print("Configuration loaded successfully.")
    
    except ConfigLoadError as e:
        print(f"[ERROR] Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Placeholder: actual training logic comes later
    print("Training pipeline config loaded successfully")

if __name__ == "__main__":
    main()