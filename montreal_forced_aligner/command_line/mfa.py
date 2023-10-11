"""Command line functions for calling the root mfa command"""
from __future__ import annotations

import atexit
import logging
import re
import sys
import time
import warnings
from datetime import datetime

import rich_click as click

from montreal_forced_aligner import config
from montreal_forced_aligner.command_line.adapt import adapt_model_cli
from montreal_forced_aligner.command_line.align import align_corpus_cli
from montreal_forced_aligner.command_line.align_one import align_one_cli
from montreal_forced_aligner.command_line.anchor import anchor_cli
from montreal_forced_aligner.command_line.configure import configure_cli
from montreal_forced_aligner.command_line.create_segments import (
    create_segments_cli,
    create_segments_vad_cli,
)
from montreal_forced_aligner.command_line.diarize_speakers import diarize_speakers_cli
from montreal_forced_aligner.command_line.g2p import g2p_cli
from montreal_forced_aligner.command_line.history import history_cli
from montreal_forced_aligner.command_line.model import model_cli
from montreal_forced_aligner.command_line.server import server_cli
from montreal_forced_aligner.command_line.tokenize import tokenize_cli
from montreal_forced_aligner.command_line.train_acoustic_model import train_acoustic_model_cli
from montreal_forced_aligner.command_line.train_dictionary import train_dictionary_cli
from montreal_forced_aligner.command_line.train_g2p import train_g2p_cli
from montreal_forced_aligner.command_line.train_ivector_extractor import train_ivector_cli
from montreal_forced_aligner.command_line.train_lm import train_lm_cli
from montreal_forced_aligner.command_line.train_tokenizer import train_tokenizer_cli
from montreal_forced_aligner.command_line.transcribe import transcribe_corpus_cli
from montreal_forced_aligner.command_line.validate import (
    validate_corpus_cli,
    validate_dictionary_cli,
)
from montreal_forced_aligner.utils import check_third_party

BEGIN = time.time()
BEGIN_DATE = datetime.now()


__all__ = ["ExitHooks", "mfa_cli"]


class ExitHooks(object):
    """
    Class for capturing exit information for MFA commands
    """

    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self) -> None:
        """Hook for capturing information about exit code and exceptions"""
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0) -> None:
        """Actual exit for the program"""
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args) -> None:
        """Handle and save exceptions"""
        self.exception = exc
        logger = logging.getLogger("mfa")
        import traceback

        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_text = "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.debug(error_text)
        self.exit_code = 1

    def history_save_handler(self) -> None:
        """
        Handler for saving history on exit.  In addition to the command run, also saves exit code, whether
        an exception was encountered, when the command was executed, and how long it took to run
        """
        from montreal_forced_aligner.utils import get_mfa_version

        history_data = {
            "command": " ".join(sys.argv),
            "execution_time": time.time() - BEGIN,
            "date": BEGIN_DATE,
            "version": get_mfa_version(),
        }
        if "github_token" in history_data["command"]:
            return
        if self.exit_code is not None:
            history_data["exit_code"] = self.exit_code
            history_data["exception"] = ""
        elif self.exception is not None:
            history_data["exit_code"] = 1
            history_data["exception"] = str(self.exception)
        else:
            history_data["exception"] = ""
            history_data["exit_code"] = 0
        config.update_command_history(history_data)
        if self.exception:
            raise self.exception


@click.group(
    name="mfa",
    help="Montreal Forced Aligner is a command line utility for aligning speech and text.",
)
@click.pass_context
def mfa_cli(ctx: click.Context) -> None:
    """
    Main function for the MFA command line interface
    """
    from montreal_forced_aligner.command_line.utils import check_server, start_server, stop_server

    try:
        from montreal_forced_aligner._version import version

        if re.search(r"\d+\.\d+\.\d+a", version) is not None:
            print(
                "Please be aware that you are running an alpha version of MFA. If you would like to install a more "
                "stable version, please visit https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html#installing-older-versions-of-mfa",
                file=sys.stderr,
            )
    except ImportError:
        pass
    config.load_configuration()
    auto_server = False
    run_check = True
    if ctx.invoked_subcommand == "anchor":
        config.CLEAN = False
        config.USE_POSTGRES = True
    if "--help" in sys.argv or ctx.invoked_subcommand in [
        "configure",
        "version",
        "history",
        "server",
        "align_one",
    ]:
        auto_server = False
        run_check = False
    elif ctx.invoked_subcommand in ["model", "models"]:
        if "add_words" in sys.argv or "inspect" in sys.argv:
            config.CLEAN = True
            config.USE_POSTGRES = False
        else:
            run_check = False
    elif ctx.invoked_subcommand == "g2p":
        if len(sys.argv) > 2 and sys.argv[2] == "-":
            run_check = False
            auto_server = False
    else:
        auto_server = config.AUTO_SERVER
    if "--no_use_postgres" in sys.argv or not config.USE_POSTGRES:
        run_check = False
        auto_server = False
    if auto_server:
        start_server()
    elif run_check:
        check_server()
    warnings.simplefilter("ignore")
    check_third_party()
    if ctx.invoked_subcommand != "anchor":
        hooks = ExitHooks()
        hooks.hook()
        atexit.register(hooks.history_save_handler)
        if auto_server:
            atexit.register(stop_server)


@click.command(
    name="version",
    short_help="Show version of MFA",
)
def version_cli():
    try:
        from montreal_forced_aligner._version import version
    except ImportError:
        version = None
    print(version)


mfa_cli.add_command(adapt_model_cli)
mfa_cli.add_command(align_corpus_cli)
mfa_cli.add_command(align_one_cli)
mfa_cli.add_command(anchor_cli)
mfa_cli.add_command(diarize_speakers_cli)
mfa_cli.add_command(create_segments_cli)
mfa_cli.add_command(create_segments_vad_cli)
mfa_cli.add_command(configure_cli)
mfa_cli.add_command(history_cli)
mfa_cli.add_command(g2p_cli)
mfa_cli.add_command(model_cli, name="model")
mfa_cli.add_command(model_cli, name="models")
mfa_cli.add_command(server_cli)
mfa_cli.add_command(tokenize_cli)
mfa_cli.add_command(train_acoustic_model_cli)
mfa_cli.add_command(train_dictionary_cli)
mfa_cli.add_command(train_g2p_cli)
mfa_cli.add_command(train_ivector_cli)
mfa_cli.add_command(train_lm_cli)
mfa_cli.add_command(train_tokenizer_cli)
mfa_cli.add_command(transcribe_corpus_cli)
mfa_cli.add_command(validate_corpus_cli)
mfa_cli.add_command(validate_dictionary_cli)
mfa_cli.add_command(version_cli)

if __name__ == "__main__":
    mfa_cli()
