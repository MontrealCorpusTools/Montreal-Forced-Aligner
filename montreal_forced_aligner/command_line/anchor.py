"""Command line functions for launching anchor annotation"""
from __future__ import annotations

import sys

import click

__all__ = ["anchor_cli"]


@click.command(name="anchor", short_help="Launch Anchor")
@click.help_option("-h", "--help")
def anchor_cli(*args, **kwargs) -> None:  # pragma: no cover
    """
    Launch Anchor Annotator (if installed)
    """
    try:
        from anchor.command_line import main
    except ImportError:
        raise
        print(
            "Anchor annotator utility is not installed, please install it via pip install anchor-annotator."
        )
        sys.exit(1)
    main()
