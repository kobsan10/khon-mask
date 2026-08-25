"""Shared argument parsing for the numbered stage scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config, load_config, parse_set_overrides
from .io_utils import setup_logging


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=None,
        help="YAML config; omit to use built-in defaults",
    )
    parser.add_argument(
        "-s", "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="override a config value, e.g. -s mesh.poisson_depth=9 (repeatable)",
    )
    parser.add_argument("--overwrite", action="store_true", help="redo work already on disk")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def resolve(args: argparse.Namespace) -> Config:
    """Build the config and start logging into the run directory."""
    cfg = load_config(args.config, parse_set_overrides(args.overrides))
    cfg.make_run_dirs()
    setup_logging(args.verbose, log_file=cfg.run_dir / "pipeline.log")
    return cfg
