"""
Module: pipelines.validate_pipeline
Purpose: Orchestrate the dataset and schema validation pipeline.
"""

import sys
from src.common.cli import build_base_parser
from src.common.config import load_config_dir, ConfigLoadError

def main() -> None:
    """Main function to execute the validation pipeline."""
    parser = build_base_parser("Dataset & Schema Validation Pipeline")
    args = parser.parse_args()

    try:
        config = load_config_dir(args.config_dir)
        print("Configuration loaded successfully")

    except ConfigLoadError as e:
        print(f"[ERROR] failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Placeholder
    print("Schema validattion pipeline config loaded successfully.")

if main == "__main__":
    main()