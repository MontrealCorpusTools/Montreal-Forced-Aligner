"""
Command line functionality
==========================

"""

from montreal_forced_aligner.command_line.adapt import run_adapt_model
from montreal_forced_aligner.command_line.align import run_align_corpus
from montreal_forced_aligner.command_line.anchor import run_anchor
from montreal_forced_aligner.command_line.classify_speakers import run_classify_speakers
from montreal_forced_aligner.command_line.create_segments import run_create_segments
from montreal_forced_aligner.command_line.g2p import run_g2p
from montreal_forced_aligner.command_line.mfa import create_parser, main
from montreal_forced_aligner.command_line.model import (
    download_model,
    inspect_model,
    list_model,
    run_model,
    save_model,
)
from montreal_forced_aligner.command_line.train_acoustic_model import run_train_acoustic_model
from montreal_forced_aligner.command_line.train_dictionary import run_train_dictionary
from montreal_forced_aligner.command_line.train_g2p import run_train_g2p
from montreal_forced_aligner.command_line.train_ivector_extractor import (
    run_train_ivector_extractor,
)
from montreal_forced_aligner.command_line.train_lm import run_train_lm
from montreal_forced_aligner.command_line.transcribe import run_transcribe_corpus
from montreal_forced_aligner.command_line.utils import validate_model_arg
from montreal_forced_aligner.command_line.validate import run_validate_corpus

__all__ = [
    "adapt",
    "align",
    "anchor",
    "classify_speakers",
    "create_segments",
    "g2p",
    "mfa",
    "model",
    "train_acoustic_model",
    "train_dictionary",
    "train_g2p",
    "train_ivector_extractor",
    "train_lm",
    "transcribe",
    "utils",
    "validate",
    "run_transcribe_corpus",
    "run_validate_corpus",
    "run_train_lm",
    "run_train_g2p",
    "run_align_corpus",
    "run_train_dictionary",
    "run_anchor",
    "run_model",
    "run_adapt_model",
    "run_train_acoustic_model",
    "run_train_ivector_extractor",
    "run_g2p",
    "run_create_segments",
    "run_classify_speakers",
    "create_parser",
    "validate_model_arg",
    "main",
    "list_model",
    "save_model",
    "inspect_model",
    "download_model",
]
