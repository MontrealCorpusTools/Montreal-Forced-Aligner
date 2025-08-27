"""Command line functions for calling the root mfa command"""
from __future__ import annotations

import rich_click as click
from click import Command, Context

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
from montreal_forced_aligner.command_line.find_oovs import find_oovs_cli
from montreal_forced_aligner.command_line.g2p import g2p_cli
from montreal_forced_aligner.command_line.history import history_cli
from montreal_forced_aligner.command_line.model import model_cli
from montreal_forced_aligner.command_line.remap import remap_cli
from montreal_forced_aligner.command_line.server import server_cli
from montreal_forced_aligner.command_line.tokenize import tokenize_cli
from montreal_forced_aligner.command_line.train_acoustic_model import train_acoustic_model_cli
from montreal_forced_aligner.command_line.train_dictionary import train_dictionary_cli
from montreal_forced_aligner.command_line.train_g2p import train_g2p_cli
from montreal_forced_aligner.command_line.train_ivector_extractor import train_ivector_cli
from montreal_forced_aligner.command_line.train_lm import train_lm_cli
from montreal_forced_aligner.command_line.train_tokenizer import train_tokenizer_cli
from montreal_forced_aligner.command_line.transcribe import (
    transcribe_corpus_cli,
    transcribe_speechbrain_cli,
    transcribe_whisper_cli,
)
from montreal_forced_aligner.command_line.validate import (
    validate_corpus_cli,
    validate_dictionary_cli,
)

__all__ = ["mfa_cli"]


class MfaGroup(click.RichGroup):
    def resolve_command(
        self, ctx: Context, args: list[str]
    ) -> tuple[str | None, Command | None, list[str]]:
        if args[0] in {"remap_dictionary", "remap_alignments"}:
            t = args[0].split("_")[-1]
            args[0] = "remap"
            args.insert(1, t)
        if args[0] == "models":
            args[0] = "model"
        return super().resolve_command(ctx, args)


@click.group(
    name="mfa",
    help="Montreal Forced Aligner is a command line utility for aligning speech and text.",
    cls=MfaGroup,
)
def mfa_cli() -> None:
    """
    Main function for the MFA command line interface
    """
    pass


@click.command(
    name="version",
    short_help="Show version of MFA",
)
def version_cli():
    try:
        from montreal_forced_aligner._version import version
    except ImportError:
        version = None
    click.echo(version)


_commands = [
    adapt_model_cli,
    align_corpus_cli,
    align_one_cli,
    anchor_cli,
    diarize_speakers_cli,
    create_segments_cli,
    create_segments_vad_cli,
    configure_cli,
    find_oovs_cli,
    history_cli,
    g2p_cli,
    model_cli,
    remap_cli,
    server_cli,
    tokenize_cli,
    train_acoustic_model_cli,
    train_dictionary_cli,
    train_g2p_cli,
    train_ivector_cli,
    train_lm_cli,
    train_tokenizer_cli,
    transcribe_corpus_cli,
    transcribe_speechbrain_cli,
    transcribe_whisper_cli,
    validate_corpus_cli,
    validate_dictionary_cli,
    version_cli,
]

for c in _commands:
    mfa_cli.add_command(c)

if __name__ == "__main__":
    mfa_cli()
