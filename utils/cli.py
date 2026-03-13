"""Unified CLI for the LiRA workflow."""

from __future__ import annotations

import argparse
import importlib


COMMAND_TARGETS = {
    "train": "train:main",
    "attack": "attack:main",
    "analyze": "comprehensive_analysis.run_analysis:main",
    "benchmark": "utils.benchmark_cli:main",
}

COMMAND_HELP = {
    "train": "Run shadow-model training",
    "attack": "Run LiRA attack evaluation",
    "analyze": "Run comprehensive post analysis for one experiment",
    "benchmark": "Run a named paper benchmark manifest",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level passthrough CLI parser."""
    parser = argparse.ArgumentParser(description="Unified CLI for LiRA experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in COMMAND_HELP.items():
        subparser = subparsers.add_parser(command, help=help_text, description=help_text, add_help=False)
        subparser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def dispatch_command(command: str, argv: list[str] | None = None):
    """Import and dispatch to the selected stage entrypoint."""
    module_name, function_name = COMMAND_TARGETS[command].split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(argv or [])


def main(argv: list[str] | None = None):
    """Run the unified CLI."""
    args = build_parser().parse_args(argv)
    return dispatch_command(args.command, args.args)
