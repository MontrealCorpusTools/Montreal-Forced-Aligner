"""Command line functions for launching anchor annotation"""
from __future__ import annotations

import logging

import requests
import rich_click as click

from montreal_forced_aligner import config

__all__ = ["anchor_cli"]

logger = logging.getLogger("mfa")


@click.command(name="anchor", short_help="Launch Anchor")
@click.help_option("-h", "--help")
def anchor_cli(*args, **kwargs) -> None:  # pragma: no cover
    """
    Launch Anchor Annotator (if installed)
    """
    from anchor.command_line import main  # noqa

    if config.VERBOSE:
        try:
            from anchor._version import version

            response = requests.get(
                "https://api.github.com/repos/MontrealCorpusTools/Anchor-annotator/releases/latest"
            )
            latest_version = response.json()["tag_name"].replace("v", "")
            if version < latest_version:
                click.echo(
                    f"You are currently running an older version of Anchor annotator ({version}) than the latest available ({latest_version}). "
                    f"To update, please run mfa_update."
                )
        except ImportError:
            pass
    main()
