"""Command line functions for calling the root mfa command"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import sys
import time
import warnings
from datetime import datetime

import click

from montreal_forced_aligner.command_line.adapt import adapt_model_cli
from montreal_forced_aligner.command_line.align import align_corpus_cli
from montreal_forced_aligner.command_line.anchor import anchor_cli
from montreal_forced_aligner.command_line.configure import configure_cli
from montreal_forced_aligner.command_line.create_segments import create_segments_cli
from montreal_forced_aligner.command_line.diarize_speakers import diarize_speakers_cli
from montreal_forced_aligner.command_line.g2p import g2p_cli
from montreal_forced_aligner.command_line.history import history_cli
from montreal_forced_aligner.command_line.model import model_cli
from montreal_forced_aligner.command_line.train_acoustic_model import train_acoustic_model_cli
from montreal_forced_aligner.command_line.train_dictionary import train_dictionary_cli
from montreal_forced_aligner.command_line.train_g2p import train_g2p_cli
from montreal_forced_aligner.command_line.train_ivector_extractor import train_ivector_cli
from montreal_forced_aligner.command_line.train_lm import train_lm_cli
from montreal_forced_aligner.command_line.transcribe import transcribe_corpus_cli
from montreal_forced_aligner.command_line.validate import (
    validate_corpus_cli,
    validate_dictionary_cli,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG, update_command_history
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
        update_command_history(history_data)
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
    GLOBAL_CONFIG.load()
    from montreal_forced_aligner.helper import configure_logger

    if not GLOBAL_CONFIG.current_profile.debug:
        warnings.simplefilter("ignore")
    configure_logger("mfa")
    check_third_party()
    if ctx.invoked_subcommand != "anchor":
        hooks = ExitHooks()
        hooks.hook()
        atexit.register(hooks.history_save_handler)
    from colorama import init

    init()
    mp.freeze_support()


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
mfa_cli.add_command(anchor_cli)
mfa_cli.add_command(diarize_speakers_cli)
mfa_cli.add_command(create_segments_cli)
mfa_cli.add_command(configure_cli)
mfa_cli.add_command(history_cli)
mfa_cli.add_command(g2p_cli)
mfa_cli.add_command(model_cli, name="model")
mfa_cli.add_command(model_cli, name="models")
mfa_cli.add_command(train_acoustic_model_cli)
mfa_cli.add_command(train_dictionary_cli)
mfa_cli.add_command(train_g2p_cli)
mfa_cli.add_command(train_ivector_cli)
mfa_cli.add_command(train_lm_cli)
mfa_cli.add_command(transcribe_corpus_cli)
mfa_cli.add_command(validate_corpus_cli)
mfa_cli.add_command(validate_dictionary_cli)
mfa_cli.add_command(version_cli)

if __name__ == "__main__":
    mfa_cli()
