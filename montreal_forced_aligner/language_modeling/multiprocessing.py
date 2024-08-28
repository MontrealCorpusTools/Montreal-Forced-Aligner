"""Multiprocessing functions for training language models"""
from __future__ import annotations

import os
import subprocess
import typing
from pathlib import Path

import sqlalchemy
from _kalpy.fstext import VectorFst
from kalpy.decoder.decode_graph import DecodeGraphCompiler
from kalpy.fstext.lexicon import LexiconCompiler
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import MfaArguments, WordType
from montreal_forced_aligner.db import Job, Phone, PhoneInterval, Speaker, Utterance, Word
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import thirdparty_binary

if typing.TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.abc import MetaDict
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
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    model_path: :class:`~pathlib.Path`
        Path to model
    order: int
        Ngram order of the language models
    method: str
        Ngram smoothing method
    target_num_ngrams: int
        Target number of ngrams
    hclg_options: dict[str, Any]
        HCLG creation options
    """

    model_path: Path
    tree_path: Path
    lexicon_compilers: typing.Dict[int, LexiconCompiler]
    order: int
    method: str
    target_num_ngrams: int
    hclg_options: MetaDict


@dataclass
class TrainLmArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.language_modeling.multiprocessing.TrainSpeakerLmFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    session: :class:`sqlalchemy.orm.scoped_session` or str
        SqlAlchemy scoped session or string for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Working directory
    symbols_path: :class:`~pathlib.Path`
        Words symbol table paths
    order: int
        Ngram order of the language models
    oov_word: str
        OOV word
    """

    working_directory: Path
    symbols_path: Path
    order: int
    oov_word: str


class TrainLmFunction(KaldiFunction):
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

    def __init__(self, args: TrainLmArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.symbols_path = args.symbols_path
        self.order = args.order
        self.oov_word = args.oov_word

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, mfa_open(self.log_path, "w") as log_file:
            word_query = session.query(Word.word).filter(
                Word.word_type.in_(WordType.speech_types())
            )
            included_words = set(x[0] for x in word_query)
            utterance_query = session.query(Utterance.normalized_text, Utterance.text).filter(
                Utterance.job_id == self.job_name
            )

            farcompile_proc = subprocess.Popen(
                [
                    thirdparty_binary("farcompilestrings"),
                    "--fst_type=compact",
                    "--token_type=symbol",
                    "--generate_keys=16",
                    "--keep_symbols",
                    f"--symbols={self.symbols_path}",
                ],
                stderr=log_file,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ngramcount_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramcount"),
                    "--round_to_int",
                    f"--order={self.order}",
                    "-",
                    self.working_directory.joinpath(f"{self.job_name}.cnts"),
                ],
                stderr=log_file,
                stdin=farcompile_proc.stdout,
                env=os.environ,
            )
            for normalized_text, text in utterance_query:
                if not normalized_text:
                    normalized_text = text
                text = " ".join(
                    x if x in included_words else self.oov_word for x in normalized_text.split()
                )
                farcompile_proc.stdin.write(f"{text}\n".encode("utf8"))
                farcompile_proc.stdin.flush()
                self.callback(1)
            farcompile_proc.stdin.close()
            self.check_call(ngramcount_proc)


class TrainPhoneLmFunction(KaldiFunction):
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

    def __init__(self, args: TrainLmArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.symbols_path = args.symbols_path
        self.order = args.order

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, mfa_open(self.log_path, "w") as log_file:
            if config.USE_POSTGRES:
                string_agg_function = sqlalchemy.func.string_agg
            else:
                string_agg_function = sqlalchemy.func.group_concat
            pronunciation_query = (
                sqlalchemy.select(Utterance.id, string_agg_function(Phone.kaldi_label, " "))
                .select_from(Utterance)
                .join(Utterance.phone_intervals)
                .join(PhoneInterval.phone)
                .where(Utterance.job_id == self.job_name)
                .group_by(Utterance.id)
            )
            farcompile_proc = subprocess.Popen(
                [
                    thirdparty_binary("farcompilestrings"),
                    "--fst_type=compact",
                    "--token_type=symbol",
                    "--generate_keys=16",
                    f"--symbols={self.symbols_path}",
                ],
                stderr=log_file,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            ngramcount_proc = subprocess.Popen(
                [
                    thirdparty_binary("ngramcount"),
                    "--require_symbols=false",
                    "--round_to_int",
                    f"--order={self.order}",
                    "-",
                    self.working_directory.joinpath(f"{self.job_name}.cnts"),
                ],
                stderr=log_file,
                stdin=farcompile_proc.stdout,
                env=os.environ,
            )
            for utt_id, phones in session.execute(pronunciation_query):
                farcompile_proc.stdin.write(f"{phones}\n".encode("utf8"))
                farcompile_proc.stdin.flush()
                self.callback((utt_id, phones))
            farcompile_proc.stdin.close()
            self.check_call(ngramcount_proc)


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
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.lexicon_compilers = args.lexicon_compilers
        self.order = args.order
        self.method = args.method
        self.target_num_ngrams = args.target_num_ngrams
        self.hclg_options = args.hclg_options

    def _run(self) -> None:
        """Run the function"""
        with self.session() as session, mfa_open(self.log_path, "w") as log_file:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )

            for d in job.dictionaries:
                dict_id = d.id
                word_symbols_path = d.words_symbol_path
                speakers = (
                    session.query(Speaker.id)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == job.id)
                    .filter(Speaker.dictionary_id == dict_id)
                    .distinct()
                )
                for (speaker_id,) in speakers:
                    hclg_path = d.temp_directory.joinpath(f"{speaker_id}.fst")
                    if os.path.exists(hclg_path):
                        continue
                    utterances = (
                        session.query(Utterance.normalized_text)
                        .filter(Utterance.speaker_id == speaker_id)
                        .order_by(Utterance.kaldi_id)
                    )
                    mod_path = d.temp_directory.joinpath(f"g.{speaker_id}.fst")
                    farcompile_proc = subprocess.Popen(
                        [
                            thirdparty_binary("farcompilestrings"),
                            "--fst_type=compact",
                            f"--unknown_symbol={d.oov_word}",
                            f"--symbols={word_symbols_path}",
                            "--keep_symbols",
                            "--generate_keys=16",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        env=os.environ,
                    )
                    count_proc = subprocess.Popen(
                        [thirdparty_binary("ngramcount"), f"--order={self.order}"],
                        stdin=farcompile_proc.stdout,
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
                    for (text,) in utterances:
                        farcompile_proc.stdin.write(f"{text}\n".encode("utf8"))
                        farcompile_proc.stdin.flush()
                    farcompile_proc.stdin.close()
                    shrink_proc.wait()
                    compiler = DecodeGraphCompiler(
                        self.model_path,
                        self.tree_path,
                        self.lexicon_compilers[dict_id],
                        **self.hclg_options,
                    )
                    compiler.g_fst = VectorFst.Read(str(mod_path))
                    compiler.export_hclg("", hclg_path)
                    self.callback(os.path.exists(hclg_path))
