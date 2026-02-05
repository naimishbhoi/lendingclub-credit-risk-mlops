"""
Module: pipelines.retrain_pipeline
Purpose: Orchestrate monitoring-triggered model retraining pipeline.
"""

import sys
from src.common.cli import build_base_parser
from src.common.config import load_config_dir, ConfigLoadError

def main() -> None:
    """Main function to execute the retraining pipeline."""
    parser = build_base_parser("Retraining pipeline")
    args = parser.parse_args()

    try:
        config = load_config_dir(args.config_dir)
        print("Configuration loaded successfully.")

    except ConfigLoadError as e:
        print(f"[ERROR] failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Placeholder
    print("Retraining pipeline config loaded successfully")

if __name__ == "__main__":
    main()