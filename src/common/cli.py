"""
Module: src.common.cli
Purpose: Command-line interface utilities for the application.
"""

import argparse


def build_base_parser(description: str) -> argparse.ArgumentParser:
    """Builds the base argument parser with common CLI options."""

    parser = argparse.ArgumentParser(
        description="Application Command-Line Interface",
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to the configuration directory containing YAML files.",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    return parser
