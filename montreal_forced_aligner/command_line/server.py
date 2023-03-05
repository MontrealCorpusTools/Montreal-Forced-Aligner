"""Command line functionality for managing servers"""
import os

import rich_click as click

from montreal_forced_aligner.command_line.utils import (
    delete_server,
    initialize_server,
    start_server,
    stop_server,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, MFA_PROFILE_VARIABLE


@click.group(name="server", short_help="Start, stop, and delete MFA database servers")
@click.help_option("-h", "--help")
def server_cli():
    pass


@server_cli.command(name="init", short_help="Initialize the MFA database server")
@click.option(
    "-p",
    "--profile",
    "profile",
    help='Configuration profile to use, defaults to "global"',
    type=str,
    default=None,
)
@click.help_option("-h", "--help")
@click.pass_context
def init_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    initialize_server()


@server_cli.command(name="start", short_help="Start the MFA database server")
@click.option(
    "-p",
    "--profile",
    help='Configuration profile to use, defaults to "global"',
    type=str,
    default=None,
)
@click.help_option("-h", "--help")
@click.pass_context
def start_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    start_server()


@server_cli.command(name="stop", short_help="Stop the MFA database server")
@click.option(
    "-p",
    "--profile",
    help='Configuration profile to use, defaults to "global"',
    type=str,
    default=None,
)
@click.option(
    "-m",
    "--mode",
    help="Mode flag to be passed to pg_ctl",
    type=click.Choice(["fast", "immediate", "smart"], case_sensitive=False),
    default="fast",
)
@click.help_option("-h", "--help")
@click.pass_context
def stop_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    stop_server(mode=kwargs.get("mode", "fast"))


@server_cli.command(name="delete", short_help="Delete the MFA database server")
@click.option(
    "-p",
    "--profile",
    help='Configuration profile to use, defaults to "global"',
    type=str,
    default=None,
)
@click.help_option("-h", "--help")
@click.pass_context
def delete_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        os.environ[MFA_PROFILE_VARIABLE] = kwargs.pop("profile")
    GLOBAL_CONFIG.current_profile.update(kwargs)
    GLOBAL_CONFIG.save()
    delete_server()
