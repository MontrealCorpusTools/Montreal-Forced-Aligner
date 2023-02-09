"""Multiprocessing functions for training language models"""
from __future__ import annotations

import os
import subprocess
import typing

import sqlalchemy
from sqlalchemy.orm import Session, joinedload, subqueryload

from montreal_forced_aligner.data import MfaArguments, WordType
from montreal_forced_aligner.db import Job, Phone, PhoneInterval, Speaker, Utterance, Word
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.transcription.multiprocessing import (
    compose_clg,
    compose_hclg,
    compose_lg,
)
from montreal_forced_aligner.utils import KaldiFunction, thirdparty_binary

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

    model_path: str
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

    working_directory: str
    symbols_path: str
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

    def _run(self) -> typing.Generator[bool]:
        """Run the function"""
        with Session(self.db_engine()) as session, mfa_open(self.log_path, "w") as log_file:
            word_query = session.query(Word.word).filter(Word.word_type == WordType.speech)
            included_words = set(x[0] for x in word_query)
            utterance_query = session.query(Utterance.normalized_text, Utterance.text).filter(
                Utterance.job_id == self.job_name
            )

            farcompile_proc = subprocess.Popen(
                [
                    thirdparty_binary("farcompilestrings"),
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
                    os.path.join(self.working_directory, f"{self.job_name}.cnts"),
                ],
                stderr=log_file,
                stdin=farcompile_proc.stdout,
                env=os.environ,
            )
            for (normalized_text, text) in utterance_query:
                if not normalized_text:
                    normalized_text = text
                text = " ".join(
                    x if x in included_words else self.oov_word for x in normalized_text.split()
                )
                farcompile_proc.stdin.write(f"{text}\n".encode("utf8"))
                farcompile_proc.stdin.flush()
                yield 1
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

    def _run(self) -> typing.Generator[bool]:
        """Run the function"""
        with Session(self.db_engine()) as session, mfa_open(self.log_path, "w") as log_file:
            pronunciation_query = (
                sqlalchemy.select(Utterance.id, sqlalchemy.func.string_agg(Phone.kaldi_label, " "))
                .select_from(Utterance)
                .join(Utterance.phone_intervals)
                .join(PhoneInterval.phone)
                .where(Utterance.job_id == self.job_name)
                .group_by(Utterance.id)
            )
            farcompile_proc = subprocess.Popen(
                [
                    thirdparty_binary("farcompilestrings"),
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
                    os.path.join(self.working_directory, f"{self.job_name}.cnts"),
                ],
                stderr=log_file,
                stdin=farcompile_proc.stdout,
                env=os.environ,
            )
            for utt_id, phones in session.execute(pronunciation_query):
                farcompile_proc.stdin.write(f"{phones}\n".encode("utf8"))
                farcompile_proc.stdin.flush()
                yield utt_id, phones
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
        self.order = args.order
        self.method = args.method
        self.target_num_ngrams = args.target_num_ngrams
        self.hclg_options = args.hclg_options

    def _run(self) -> typing.Generator[bool]:
        """Run the function"""
        with Session(self.db_engine()) as session, mfa_open(self.log_path, "w") as log_file:

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

                    hclg_path = os.path.join(d.temp_directory, f"{speaker_id}.fst")
                    if os.path.exists(hclg_path):
                        continue
                    utterances = (
                        session.query(Utterance.normalized_text)
                        .filter(Utterance.speaker_id == speaker_id)
                        .order_by(Utterance.kaldi_id)
                    )
                    mod_path = os.path.join(d.temp_directory, f"g.{speaker_id}.fst")
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
                    context_width = self.hclg_options["context_width"]
                    central_pos = self.hclg_options["central_pos"]
                    path_template = os.path.join(
                        d.temp_directory, f"{{file_name}}.{speaker_id}.fst"
                    )
                    lg_path = path_template.format(file_name="LG")
                    hclga_path = path_template.format(file_name="HCLGa")
                    clg_path = path_template.format(file_name=f"CLG_{context_width}_{central_pos}")
                    ilabels_temp = path_template.format(
                        file_name=f"ilabels_{context_width}_{central_pos}"
                    ).replace(".fst", "")
                    out_disambig = path_template.format(
                        file_name=f"disambig_ilabels_{context_width}_{central_pos}"
                    ).replace(".fst", ".int")
                    log_file.write("Generating LG.fst...")
                    compose_lg(d.lexicon_disambig_fst_path, mod_path, lg_path, log_file)
                    log_file.write("Generating CLG.fst...")
                    compose_clg(
                        d.disambiguation_symbols_int_path,
                        out_disambig,
                        context_width,
                        central_pos,
                        ilabels_temp,
                        lg_path,
                        clg_path,
                        log_file,
                    )
                    log_file.write("Generating HCLGa.fst...")
                    compose_hclg(
                        self.model_path,
                        ilabels_temp,
                        self.hclg_options["transition_scale"],
                        clg_path,
                        hclga_path,
                        log_file,
                    )
                    log_file.write("Generating HCLG.fst...")
                    self_loop_proc = subprocess.Popen(
                        [
                            thirdparty_binary("add-self-loops"),
                            f"--self-loop-scale={self.hclg_options['self_loop_scale']}",
                            "--reorder=true",
                            self.model_path,
                            hclga_path,
                        ],
                        stderr=log_file,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    convert_proc = subprocess.Popen(
                        [
                            thirdparty_binary("fstconvert"),
                            "--v=100",
                            "--fst_type=const",
                            "-",
                            hclg_path,
                        ],
                        stdin=self_loop_proc.stdout,
                        stderr=log_file,
                        env=os.environ,
                    )
                    convert_proc.communicate()
                    self.check_call(convert_proc)
                    os.remove(mod_path)
                    os.remove(lg_path)
                    os.remove(clg_path)
                    os.remove(hclga_path)
                    os.remove(ilabels_temp)
                    os.remove(out_disambig)
                    yield os.path.exists(hclg_path)
