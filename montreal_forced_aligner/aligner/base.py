"""Class definitions for base aligner"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Optional

from ..abc import Aligner
from ..config import TEMP_DIR
from ..exceptions import KaldiProcessingError
from ..multiprocessing import (
    align,
    calc_fmllr,
    compile_information,
    compile_train_graphs,
    convert_ali_to_textgrids,
)
from ..utils import log_kaldi_errors

if TYPE_CHECKING:
    from logging import Logger

    import montreal_forced_aligner

    from ..config import AlignConfig
    from ..corpus.base import Corpus
    from ..dictionary import MultispeakerDictionary
    from ..models import AcousticModel

__all__ = ["BaseAligner"]


class BaseAligner(Aligner):
    """
    Base aligner class for common aligner functions

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`
        Dictionary object for the pronunciation dictionary
    align_config : :class:`~montreal_forced_aligner.config.AlignConfig`
        Configuration for alignment
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    debug : bool
        Flag for running in debug mode, defaults to false
    verbose : bool
        Flag for running in verbose mode, defaults to false
    logger : :class:`~logging.Logger`
        Logger to use
    """

    def __init__(
        self,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        align_config: AlignConfig,
        temp_directory: Optional[str] = None,
        debug: bool = False,
        verbose: bool = False,
        logger: Optional[Logger] = None,
        acoustic_model: Optional[AcousticModel] = None,
    ):
        super().__init__(corpus, dictionary, align_config)
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        os.makedirs(self.temp_directory, exist_ok=True)
        self.log_file = os.path.join(self.temp_directory, "aligner.log")
        if logger is None:
            self.logger = logging.getLogger("corpus_setup")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, "w", "utf-8")
            handler.setFormatter = logging.Formatter("%(name)s %(message)s")
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        self.acoustic_model = None
        self.verbose = verbose
        self.debug = debug
        self.speaker_independent = True
        self.uses_cmvn = True
        self.uses_splices = False
        self.uses_voiced = False
        self.iteration = None
        self.acoustic_model = acoustic_model
        self.setup()

    def setup(self) -> None:
        """
        Set up dictionary, corpus and configurations
        """
        self.dictionary.set_word_set(self.corpus.word_set)
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary, self.align_config.feature_config)
        self.align_config.silence_csl = self.dictionary.config.silence_csl
        self.data_directory = self.corpus.split_directory
        self.feature_config = self.align_config.feature_config

    @property
    def use_mp(self) -> bool:
        """Flag for using multiprocessing"""
        return self.align_config.use_mp

    @property
    def meta(self) -> montreal_forced_aligner.abc.MetaDict:
        """Metadata for the trained model"""
        from ..utils import get_mfa_version

        data = {
            "phones": sorted(self.dictionary.config.non_silence_phones),
            "version": get_mfa_version(),
            "architecture": "gmm-hmm",
            "features": "mfcc+deltas",
        }
        return data

    @property
    def align_options(self):
        """Options for alignment"""
        options = self.align_config.align_options
        options["optional_silence_csl"] = self.dictionary.config.optional_silence_csl
        return options

    @property
    def fmllr_options(self):
        """Options for fMLLR"""
        options = self.align_config.fmllr_options
        options["silence_csl"] = self.dictionary.config.silence_csl
        return options

    @property
    def align_directory(self) -> str:
        """Align directory"""
        return os.path.join(self.temp_directory, "align")

    @property
    def working_directory(self) -> str:
        """Current working directory"""
        return self.align_directory

    @property
    def model_path(self) -> str:
        """Current acoustic model path"""
        return self.current_model_path

    @property
    def current_model_path(self) -> str:
        """Current acoustic model path"""
        return os.path.join(self.align_directory, "final.mdl")

    @property
    def alignment_model_path(self):
        """Alignment acoustic model path"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if self.speaker_independent and os.path.exists(path):
            return path
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def working_log_directory(self) -> str:
        """Current log directory"""
        return os.path.join(self.align_directory, "log")

    @property
    def backup_output_directory(self) -> Optional[str]:
        """Backup output directory"""
        if self.align_config.overwrite:
            return None
        return os.path.join(self.align_directory, "textgrids")

    def compile_information(self, output_directory: str) -> None:
        """
        Compile information about the quality of alignment

        Parameters
        ----------
        output_directory: str
            Directory to save information to
        """
        issues, average_log_like = compile_information(self)
        errors_path = os.path.join(output_directory, "output_errors.txt")
        if os.path.exists(errors_path):
            self.logger.warning(
                "There were errors when generating the textgrids. See the output_errors.txt in the "
                "output directory for more details."
            )
        if issues:
            issue_path = os.path.join(output_directory, "unaligned.txt")
            with open(issue_path, "w", encoding="utf8") as f:
                for u, r in sorted(issues.items()):
                    f.write(f"{u}\t{r}\n")
            self.logger.warning(
                f"There were {len(issues)} segments/files not aligned.  Please see {issue_path} for more details on why "
                "alignment failed for these files."
            )
        if (
            self.backup_output_directory is not None
            and os.path.exists(self.backup_output_directory)
            and os.listdir(self.backup_output_directory)
        ):
            self.logger.info(
                f"Some TextGrids were not output in the output directory to avoid overwriting existing files. "
                f"You can find them in {self.backup_output_directory}, and if you would like to disable this "
                f"behavior, you can rerun with the --overwrite flag or run `mfa configure --always_overwrite`."
            )

    def export_textgrids(self, output_directory: str) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        """
        begin = time.time()
        self.textgrid_output = output_directory
        if self.backup_output_directory is not None and os.path.exists(
            self.backup_output_directory
        ):
            shutil.rmtree(self.backup_output_directory, ignore_errors=True)
        convert_ali_to_textgrids(self)
        self.compile_information(output_directory)
        self.logger.debug(f"Exported TextGrids in a total of {time.time() - begin} seconds")

    def align(self, subset: Optional[int] = None) -> None:
        """
        Perform alignment

        Parameters
        ----------
        subset: int, optional
            Number of utterances to align
        """
        done_path = os.path.join(self.align_directory, "done")
        dirty_path = os.path.join(self.align_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.info("Alignment already done, skipping.")
            return
        try:
            compile_train_graphs(self)
            log_dir = os.path.join(self.align_directory, "log")
            os.makedirs(log_dir, exist_ok=True)

            self.logger.info("Performing first-pass alignment...")
            align(self)
            _, average_log_like = compile_information(self)
            self.logger.debug(
                f"Prior to SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}"
            )
            if (
                not self.align_config.disable_sat
                and self.acoustic_model.feature_config.fmllr
                and not os.path.exists(os.path.join(self.align_directory, "trans.0"))
            ):
                self.logger.info("Calculating fMLLR for speaker adaptation...")
                calc_fmllr(self)
                self.logger.info("Performing second-pass alignment...")
                align(self)

                _, average_log_like = compile_information(self)
                self.logger.debug(
                    f"Following SAT, average per frame likelihood (this might not actually mean anything): {average_log_like}"
                )

        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
        with open(done_path, "w"):
            pass
