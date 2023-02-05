"""
Command line functionality
==========================

"""

from montreal_forced_aligner.command_line.adapt import adapt_model_cli
from montreal_forced_aligner.command_line.align import align_corpus_cli
from montreal_forced_aligner.command_line.anchor import anchor_cli
from montreal_forced_aligner.command_line.configure import configure_cli
from montreal_forced_aligner.command_line.create_segments import create_segments_cli
from montreal_forced_aligner.command_line.diarize_speakers import diarize_speakers_cli
from montreal_forced_aligner.command_line.g2p import g2p_cli
from montreal_forced_aligner.command_line.history import history_cli
from montreal_forced_aligner.command_line.mfa import mfa_cli
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

__all__ = [
    "adapt",
    "align",
    "anchor",
    "diarize_speakers.py",
    "create_segments",
    "g2p",
    "mfa",
    "model",
    "configure",
    "history",
    "train_acoustic_model",
    "train_dictionary",
    "train_g2p",
    "train_ivector_extractor",
    "train_lm",
    "transcribe",
    "utils",
    "validate",
    "adapt_model_cli",
    "align_corpus_cli",
    "diarize_speakers_cli",
    "create_segments_cli",
    "g2p_cli",
    "mfa_cli",
    "configure_cli",
    "history_cli",
    "anchor_cli",
    "model_cli",
    "train_acoustic_model_cli",
    "train_dictionary_cli",
    "train_g2p_cli",
    "train_ivector_cli",
    "train_lm_cli",
    "transcribe_corpus_cli",
    "validate_dictionary_cli",
    "validate_corpus_cli",
]
