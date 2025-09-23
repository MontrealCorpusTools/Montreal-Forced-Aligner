"""Command line functionality for managing servers"""
import rich_click as click

from montreal_forced_aligner.command_line.utils import (
    common_options,
    delete_server,
    initialize_configuration,
    initialize_server,
    start_server,
    stop_server,
)


@click.group(name="server", short_help="Start, stop, and delete MFA database servers")
@click.help_option("-h", "--help")
def server_cli():
    pass


@server_cli.command(name="init", short_help="Initialize the MFA database server")
@click.help_option("-h", "--help")
@common_options
@click.pass_context
def init_cli(context, **kwargs):
    initialize_configuration(context)
    initialize_server()


@server_cli.command(name="start", short_help="Start the MFA database server")
@click.help_option("-h", "--help")
@common_options
@click.pass_context
def start_cli(context, **kwargs):
    initialize_configuration(context)
    start_server()


@server_cli.command(name="stop", short_help="Stop the MFA database server")
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
    initialize_configuration(context)
    stop_server(mode=kwargs.get("mode", "fast"))


@server_cli.command(name="delete", short_help="Delete the MFA database server")
@click.help_option("-h", "--help")
@common_options
@click.pass_context
def delete_cli(context, **kwargs):
    initialize_configuration(context)
    delete_server()
