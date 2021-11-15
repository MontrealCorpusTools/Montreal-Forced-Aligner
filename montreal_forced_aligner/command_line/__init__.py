"""
Command line functionality
==========================


"""

from .adapt import run_adapt_model  # noqa
from .align import run_align_corpus  # noqa
from .anchor import run_anchor  # noqa
from .classify_speakers import run_classify_speakers  # noqa
from .create_segments import run_create_segments  # noqa
from .g2p import run_g2p  # noqa
from .mfa import create_parser, main  # noqa
from .model import download_model, inspect_model, list_model, run_model, save_model  # noqa
from .train_acoustic_model import run_train_acoustic_model  # noqa
from .train_dictionary import run_train_dictionary  # noqa
from .train_g2p import run_train_g2p  # noqa
from .train_ivector_extractor import run_train_ivector_extractor  # noqa
from .train_lm import run_train_lm  # noqa
from .transcribe import run_transcribe_corpus  # noqa
from .utils import validate_model_arg  # noqa
from .validate import run_validate_corpus  # noqa

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
