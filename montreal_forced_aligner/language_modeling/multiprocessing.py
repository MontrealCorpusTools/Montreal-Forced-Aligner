"""Multiprocessing functions for training language models"""
from __future__ import annotations

import os
import subprocess
import typing
from typing import TYPE_CHECKING, Dict, List

from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import KaldiFunction, thirdparty_binary

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from dataclassy import dataclass


@dataclass
class TrainSpeakerLmArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.language_modeling.multiprocessing.TrainSpeakerLmFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    model_directory: str
        Path to model directory
    word_symbols_paths: dict[int, str]
        Per dictionary words symbol table paths
    speaker_mapping: dict[int, str]
        Mapping of dictionaries to speakers
    speaker_paths: dict[int, str]
        Per speaker output LM paths
    oov_word: str
        OOV word
    order: int
        Ngram order of the language models
    method: str
        Ngram smoothing method
    target_num_ngrams: int
        Target number of ngrams
    """

    model_directory: str
    word_symbols_paths: Dict[int, str]
    speaker_mapping: Dict[int, List[str]]
    speaker_paths: Dict[int, str]
    oov_word: str
    order: int
    method: str
    target_num_ngrams: int


class TrainSpeakerLmFunction(KaldiFunction):
    """
    Multiprocessing function to training small language models for each speaker

    See Also
    --------
    :openfst_src:`farcompilestrings`
        Relevant OpenFst binary
    :ngram_src:`ngramcount`
        Relevant OpenGrm-Ngram binary
    :ngram_src:`ngrammake`
        Relevant OpenGrm-Ngram binary
    :ngram_src:`ngramshrink`
        Relevant OpenGrm-Ngram binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.language_modeling.multiprocessing.TrainSpeakerLmArguments`
        Arguments for the function
    """

    def __init__(self, args: TrainSpeakerLmArguments):
        super().__init__(args)
        self.model_directory = args.model_directory
        self.word_symbols_paths = args.word_symbols_paths
        self.speaker_mapping = args.speaker_mapping
        self.speaker_paths = args.speaker_paths
        self.oov_word = args.oov_word
        self.order = args.order
        self.method = args.method
        self.target_num_ngrams = args.target_num_ngrams

    def _run(self) -> typing.Generator[bool]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            print(str(self.speaker_mapping), file=log_file)
            for dict_id, speakers in self.speaker_mapping.items():
                word_symbols_path = self.word_symbols_paths[dict_id]
                for speaker in speakers:
                    training_path = os.path.join(self.model_directory, f"{speaker}.txt")
                    base_path = os.path.splitext(training_path)[0]
                    mod_path = base_path + ".mod"
                    far_proc = subprocess.Popen(
                        [
                            thirdparty_binary("farcompilestrings"),
                            "--fst_type=compact",
                            f"--unknown_symbol={self.oov_word}",
                            f"--symbols={word_symbols_path}",
                            "--keep_symbols",
                            training_path,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        env=os.environ,
                    )
                    count_proc = subprocess.Popen(
                        [thirdparty_binary("ngramcount"), f"--order={self.order}"],
                        stdin=far_proc.stdout,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                    )
                    make_proc = subprocess.Popen(
                        [thirdparty_binary("ngrammake"), "--method=kneser_ney"],
                        stdin=count_proc.stdout,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        env=os.environ,
                    )
                    shrink_proc = subprocess.Popen(
                        [
                            thirdparty_binary("ngramshrink"),
                            "--method=relative_entropy",
                            f"--target_number_of_ngrams={self.target_num_ngrams}",
                            "--shrink_opt=2",
                            "--theta=0.001",
                            "-",
                            mod_path,
                        ],
                        stdin=make_proc.stdout,
                        stderr=log_file,
                        env=os.environ,
                    )
                    shrink_proc.communicate()
                    self.check_call(shrink_proc)
                    assert os.path.exists(mod_path)
                    os.remove(training_path)
                    yield os.path.exists(mod_path)
