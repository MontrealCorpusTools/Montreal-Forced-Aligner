"""Montreal Forced Aligner is a package for aligning speech corpora through the use of acoustic models and
            dictionaries using Kaldi functionality."""
import montreal_forced_aligner.aligner as aligner  # noqa
import montreal_forced_aligner.command_line as command_line  # noqa
import montreal_forced_aligner.config as config  # noqa
import montreal_forced_aligner.corpus as corpus  # noqa
import montreal_forced_aligner.dictionary as dictionary  # noqa
import montreal_forced_aligner.exceptions as exceptions  # noqa
import montreal_forced_aligner.g2p as g2p  # noqa
import montreal_forced_aligner.helper as helper  # noqa
import montreal_forced_aligner.models as models  # noqa
import montreal_forced_aligner.multiprocessing as multiprocessing  # noqa
import montreal_forced_aligner.textgrid as textgrid  # noqa
import montreal_forced_aligner.utils as utils  # noqa

__all__ = [
    "abc",
    "data",
    "aligner",
    "command_line",
    "config",
    "corpus",
    "dictionary",
    "exceptions",
    "g2p",
    "helper",
    "models",
    "multiprocessing",
    "textgrid",
    "utils",
]
