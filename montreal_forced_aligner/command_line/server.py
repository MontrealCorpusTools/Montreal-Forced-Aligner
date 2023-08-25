"""Command line functionality for managing servers"""
import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.utils import (
    common_options,
    delete_server,
    initialize_server,
    start_server,
    stop_server,
)


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
@common_options
@click.pass_context
def init_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        config.CURRENT_PROFILE_NAME = kwargs.pop("profile")
    config.update_configuration(kwargs)
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
@common_options
@click.pass_context
def start_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        config.CURRENT_PROFILE_NAME = kwargs.pop("profile")
    config.update_configuration(kwargs)
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
@common_options
@click.pass_context
def stop_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        config.CURRENT_PROFILE_NAME = kwargs.pop("profile")
    config.update_configuration(kwargs)
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
@common_options
@click.pass_context
def delete_cli(context, **kwargs):
    if kwargs.get("profile", None) is not None:
        config.CURRENT_PROFILE_NAME = kwargs.pop("profile")
    config.update_configuration(kwargs)
    delete_server()
