import logging
import time

import rich_click as click

from montreal_forced_aligner import config

__all__ = ["history_cli"]

logger = logging.getLogger("mfa")


@click.command(
    "history",
    help="Show previously run mfa commands",
)
@click.option("--depth", help="Number of commands to list, defaults to 10", type=int, default=10)
@click.option(
    "--verbose/--no_verbose",
    "-v/-nv",
    "verbose",
    help=f"Output debug messages, default is {config.VERBOSE}",
    default=config.VERBOSE,
)
@click.help_option("-h", "--help")
def history_cli(depth: int, verbose: bool) -> None:
    """
    List previous MFA commands
    """
    history = config.load_command_history()[-depth:]
    if verbose:
        logger.info("command\tDate\tExecution time\tVersion\tExit code\tException")
        for h in history:
            execution_time = time.strftime("%H:%M:%S", time.gmtime(h["execution_time"]))
            d = h["date"].isoformat()
            logger.info(
                f"{h['command']}\t{d}\t{execution_time}\t{h.get('version', 'unknown')}\t{h['exit_code']}\t{h['exception']}"
            )
        pass
    else:
        for h in history:
            logger.info(h["command"])
