"""
Multiprocessing functions
=========================

"""
from .alignment import calc_fmllr  # noqa
from .alignment import calc_lda_mllt  # noqa
from .alignment import compile_information  # noqa
from .alignment import compile_train_graphs  # noqa
from .alignment import compute_alignment_improvement  # noqa
from .alignment import convert_ali_to_textgrids  # noqa
from .alignment import convert_alignments  # noqa
from .alignment import create_align_model  # noqa
from .alignment import lda_acc_stats  # noqa
from .alignment import mono_align_equal  # noqa
from .alignment import train_map  # noqa
from .alignment import tree_stats  # noqa
from .alignment import (  # noqa
    CleanupWordCtmProcessWorker,
    CombineProcessWorker,
    ExportPreparationProcessWorker,
    ExportTextGridProcessWorker,
    NoCleanupWordCtmProcessWorker,
    PhoneCtmProcessWorker,
    acc_stats,
    acc_stats_func,
    align,
    align_func,
)
from .classes import Job  # noqa
from .corpus import CorpusProcessWorker  # noqa
from .helper import Counter, ProcessWorker, Stopped, run_mp, run_non_mp  # noqa
from .ivector import acc_global_stats  # noqa
from .ivector import acc_ivector_stats  # noqa
from .ivector import extract_ivectors  # noqa
from .ivector import gauss_to_post  # noqa
from .ivector import gmm_gselect  # noqa
from .ivector import segment_vad  # noqa
from .pronunciations import generate_pronunciations  # noqa
from .transcription import transcribe, transcribe_fmllr  # noqa

__all__ = [
    "alignment",
    "classes",
    "corpus",
    "features",
    "helper",
    "ivector",
    "pronunciations",
    "transcription",
]

Job.__module__ = "montreal_forced_aligner.multiprocessing"
CleanupWordCtmProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
CombineProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
PhoneCtmProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
ExportPreparationProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
ExportTextGridProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
NoCleanupWordCtmProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"

CorpusProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
Counter.__module__ = "montreal_forced_aligner.multiprocessing"
Stopped.__module__ = "montreal_forced_aligner.multiprocessing"
ProcessWorker.__module__ = "montreal_forced_aligner.multiprocessing"
