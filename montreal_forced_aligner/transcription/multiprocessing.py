"""
Transcription functions
-----------------------

"""
from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from _kalpy.fstext import ConstFst, VectorFst
from _kalpy.lat import CompactLatticeWriter
from _kalpy.lm import ConstArpaLm
from _kalpy.util import BaseFloatMatrixWriter, Int32VectorWriter, ReadKaldiObject
from kalpy.data import KaldiMapping, MatrixArchive
from kalpy.decoder.decode_graph import DecodeGraphCompiler
from kalpy.feat.data import FeatureArchive
from kalpy.feat.fmllr import FmllrComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.data import LatticeArchive
from kalpy.gmm.decode import GmmDecoder, GmmRescorer
from kalpy.lm.rescore import LmRescorer
from kalpy.utils import generate_write_specifier
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.data import MfaArguments, PhoneType
from montreal_forced_aligner.db import Job, Phone, Utterance
from montreal_forced_aligner.utils import thread_logger

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclassy import dataclass


__all__ = [
    "FmllrRescoreFunction",
    "FinalFmllrFunction",
    "InitialFmllrFunction",
    "CarpaLmRescoreFunction",
    "DecodeFunction",
    "LmRescoreFunction",
    "CreateHclgFunction",
]


@dataclass
class CreateHclgArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Current working directory
    words_path: :class:`~pathlib.Path`
        Path to words symbol table
    carpa_path: :class:`~pathlib.Path`
        Path to .carpa file
    small_arpa_path: :class:`~pathlib.Path`
        Path to small ARPA file
    medium_arpa_path: :class:`~pathlib.Path`
        Path to medium ARPA file
    big_arpa_path: :class:`~pathlib.Path`
        Path to big ARPA file
    model_path: :class:`~pathlib.Path`
        Acoustic model path
    disambig_L_path: :class:`~pathlib.Path`
        Path to disambiguated lexicon file
    disambig_int_path: :class:`~pathlib.Path`
        Path to disambiguation symbol integer file
    hclg_options: dict[str, Any]
        HCLG options
    words_mapping: dict[str, int]
        Words mapping
    """

    lexicon_compiler: LexiconCompiler
    working_directory: Path
    small_arpa_path: Path
    medium_arpa_path: Path
    big_arpa_path: Path
    model_path: Path
    tree_path: Path
    hclg_options: MetaDict


@dataclass
class DecodeArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    decode_options: dict[str, Any]
        Decoding options
    model_path: :class:`~pathlib.Path`
        Path to model file
    lat_paths: dict[int, Path]
        Per dictionary lattice paths
    word_symbol_paths: dict[int, Path]
        Per dictionary word symbol table paths
    hclg_paths: dict[int, Path]
        Per dictionary HCLG.fst paths
    """

    working_directory: Path
    model_path: Path
    decode_options: MetaDict
    hclg_paths: Dict[int, Path]


@dataclass
class DecodePhoneArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.validation.corpus_validator.DecodePhoneFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    decode_options: dict[str, Any]
        Decoding options
    model_path: :class:`~pathlib.Path`
        Path to model file
    lat_paths: dict[int, Path]
        Per dictionary lattice paths
    phone_symbol_path: :class:`~pathlib.Path`
        Phone symbol table paths
    hclg_path: :class:`~pathlib.Path`
        HCLG.fst paths
    """

    working_directory: Path
    model_path: Path
    hclg_path: Path
    decode_options: MetaDict


@dataclass
class LmRescoreArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    lm_rescore_options: dict[str, Any]
        Rescoring options
    lat_paths: dict[int, Path]
        Per dictionary lattice paths
    rescored_lat_paths: dict[int, Path]
        Per dictionary rescored lattice paths
    old_g_paths: dict[int, Path]
        Mapping of dictionaries to small G.fst paths
    new_g_paths: dict[int, Path]
        Mapping of dictionaries to medium G.fst paths
    """

    working_directory: Path
    lm_rescore_options: MetaDict
    old_g_paths: Dict[int, Path]
    new_g_paths: Dict[int, Path]


@dataclass
class CarpaLmRescoreArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    lat_paths: dict[int, Path]
        Per dictionary lattice paths
    rescored_lat_paths: dict[int, Path]
        Per dictionary rescored lattice paths
    old_g_paths: dict[int, Path]
        Mapping of dictionaries to medium G.fst paths
    new_g_paths: dict[int, Path]
        Mapping of dictionaries to G.carpa paths
    """

    working_directory: Path
    lm_rescore_options: MetaDict
    old_g_paths: Dict[int, Path]
    new_g_paths: Dict[int, Path]


@dataclass
class InitialFmllrArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: :class:`~pathlib.Path`
        Path to model file
    fmllr_options: dict[str, Any]
        fMLLR options
    pre_trans_paths: dict[int, Path]
        Per dictionary pre-fMLLR lattice paths
    lat_paths: dict[int, Path]
        Per dictionary lattice paths
    spk2utt_paths: dict[int, Path]
        Per dictionary speaker to utterance mapping paths
    """

    working_directory: Path
    model_path: Path
    fmllr_options: MetaDict


@dataclass
class FinalFmllrArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    working_directory: :class:`~pathlib.Path`
        Working directory
    model_path: :class:`~pathlib.Path`
        Path to model file
    fmllr_options: dict[str, Any]
        fMLLR options
    """

    working_directory: Path
    model_path: Path
    fmllr_options: MetaDict


@dataclass
class FmllrRescoreArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: :class:`~pathlib.Path`
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: :class:`~pathlib.Path`
        Path to model file
    fmllr_options: dict[str, Any]
        fMLLR options
    tmp_lat_paths: dict[int, Path]
        Per dictionary temporary lattice paths
    final_lat_paths: dict[int, Path]
        Per dictionary lattice paths
    """

    working_directory: Path
    model_path: Path
    rescore_options: MetaDict


class CreateHclgFunction(KaldiFunction):
    """
    Create HCLG.fst file

    See Also
    --------
    :meth:`.Transcriber.create_hclgs`
        Main function that calls this function in parallel
    :meth:`.Transcriber.create_hclgs_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`add-self-loops`
        Relevant Kaldi binary
    :openfst_src:`fstconvert`
        Relevant OpenFst binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgArguments`
        Arguments for the function
    """

    def __init__(self, args: CreateHclgArguments):
        super().__init__(args)
        self.lexicon_compiler = args.lexicon_compiler
        self.working_directory = args.working_directory
        self.small_arpa_path = args.small_arpa_path
        self.medium_arpa_path = args.medium_arpa_path
        self.big_arpa_path = args.big_arpa_path
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.hclg_options = args.hclg_options

    def _run(self) -> typing.Generator[typing.Tuple[bool, str]]:
        """Run the function"""
        with thread_logger("kalpy.decode_graph", self.log_path, job_name=self.job_name):
            hclg_path = self.working_directory.joinpath(f"HCLG.{self.job_name}.fst")
            small_g_path = self.working_directory.joinpath(f"G_small.{self.job_name}.fst")
            medium_g_path = self.working_directory.joinpath(f"G_med.{self.job_name}.fst")
            carpa_path = self.working_directory.joinpath(f"G.{self.job_name}.carpa")
            small_compiler = DecodeGraphCompiler(
                self.model_path, self.tree_path, self.lexicon_compiler, **self.hclg_options
            )
            small_compiler.export_hclg(self.small_arpa_path, hclg_path)
            small_compiler.export_g(small_g_path)
            del small_compiler
            medium_compiler = DecodeGraphCompiler(
                self.model_path, self.tree_path, self.lexicon_compiler, **self.hclg_options
            )
            medium_compiler.compile_g_fst(self.medium_arpa_path)
            medium_compiler.export_g(medium_g_path)
            del medium_compiler
            carpa_compiler = DecodeGraphCompiler(
                self.model_path, self.tree_path, self.lexicon_compiler, **self.hclg_options
            )
            carpa_compiler.compile_g_carpa(self.big_arpa_path, carpa_path)
            del carpa_compiler
            if hclg_path.exists():
                self.callback((True, hclg_path))
            else:
                self.callback((False, hclg_path))


class DecodeFunction(KaldiFunction):
    """
    Multiprocessing function for performing decoding

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.decode_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`
        Arguments for the function
    """

    def __init__(self, args: DecodeArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.hclg_paths = args.hclg_paths
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.decode", self.log_path, job_name=self.job_name) as decode_logger,
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                )
            ]

            for d in job.dictionaries:
                decode_logger.debug(f"Decoding for dictionary {d.id}")
                decode_logger.debug(f"Decoding with model: {self.model_path}")
                dict_id = d.id

                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)

                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                alignment_file_name = job.construct_path(
                    self.working_directory, "ali", "ark", dict_id
                )
                words_path = job.construct_path(self.working_directory, "words", "ark", dict_id)
                hclg_fst = ConstFst.Read(str(self.hclg_paths[dict_id]))
                boost_silence = self.decode_options.pop("boost_silence", 1.0)
                decoder = GmmDecoder(self.model_path, hclg_fst, **self.decode_options)
                if boost_silence != 1.0:
                    decoder.boost_silence(boost_silence, silence_phones)
                decoder.export_lattices(
                    lat_path,
                    feature_archive,
                    word_file_name=words_path,
                    alignment_file_name=alignment_file_name,
                    callback=self.callback,
                )


class LmRescoreFunction(KaldiFunction):
    """
    Multiprocessing function rescore lattices by replacing the small G.fst with the medium G.fst

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.lm_rescore_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-lmrescore-pruned`
        Relevant Kaldi binary
    :openfst_src:`fstproject`
        Relevant OpenFst binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreArguments`
        Arguments for the function
    """

    def __init__(self, args: LmRescoreArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.old_g_paths = args.old_g_paths
        self.new_g_paths = args.new_g_paths
        self.lm_rescore_options = args.lm_rescore_options

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.lm", self.log_path, job_name=self.job_name),
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            for d in job.dictionaries:
                dict_id = d.id
                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                tmp_lat_path = job.construct_path(
                    self.working_directory, "lat.tmp", "ark", dict_id
                )
                os.rename(lat_path, tmp_lat_path)
                old_g_path = self.old_g_paths[dict_id]
                new_g_path = self.new_g_paths[dict_id]
                olg_g = VectorFst.Read(str(old_g_path))
                new_lm = VectorFst.Read(str(new_g_path))
                rescorer = LmRescorer(olg_g, **self.lm_rescore_options)
                lattice_archive = LatticeArchive(tmp_lat_path, determinized=True)
                rescorer.export_lattices(lat_path, lattice_archive, new_lm, callback=self.callback)
                lattice_archive.close()
                os.remove(tmp_lat_path)


class CarpaLmRescoreFunction(KaldiFunction):
    """
    Multiprocessing function to rescore lattices by replacing medium G.fst with large G.carpa

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.carpa_lm_rescore_arguments`
        Job method for generating arguments for this function
    :openfst_src:`fstproject`
        Relevant OpenFst binary
    :kaldi_src:`lattice-lmrescore`
        Relevant Kaldi binary
    :kaldi_src:`lattice-lmrescore-const-arpa`
        Relevant Kaldi binary

    Parameters
    ----------
    args: CarpaLmRescoreArguments
        Arguments
    """

    def __init__(self, args: CarpaLmRescoreArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.lm_rescore_options = args.lm_rescore_options
        self.old_g_paths = args.old_g_paths
        self.new_g_paths = args.new_g_paths

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.lm", self.log_path, job_name=self.job_name),
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            for d in job.dictionaries:
                dict_id = d.id
                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                tmp_lat_path = job.construct_path(
                    self.working_directory, "lat.tmp", "ark", dict_id
                )
                os.rename(lat_path, tmp_lat_path)
                old_g_path = self.old_g_paths[dict_id]
                new_g_path = self.new_g_paths[dict_id]
                olg_g = VectorFst.Read(str(old_g_path))
                new_lm = ConstArpaLm()
                ReadKaldiObject(str(new_g_path), new_lm)
                rescorer = LmRescorer(olg_g, **self.lm_rescore_options)
                lattice_archive = LatticeArchive(tmp_lat_path, determinized=True)
                rescorer.export_lattices(lat_path, lattice_archive, new_lm, callback=self.callback)
                lattice_archive.close()
                os.remove(tmp_lat_path)


class InitialFmllrFunction(KaldiFunction):
    """
    Multiprocessing function for running initial fMLLR calculation

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.initial_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-post-to-gpost`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr-gpost`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrArguments`
        Arguments for the function
    """

    def __init__(self, args: InitialFmllrArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.fmllr", self.log_path, job_name=self.job_name) as fmllr_logger,
        ):
            fmllr_logger.debug(f"Using acoustic model: {self.model_path}\n")
            job: typing.Optional[Job] = session.get(
                Job, self.job_name, options=[joinedload(Job.dictionaries), joinedload(Job.corpus)]
            )
            lda_mat_path = self.working_directory.joinpath("lda.mat")
            if not lda_mat_path.exists():
                lda_mat_path = None
            for dict_id in job.dictionary_ids:
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                utt2spk_path = job.construct_path(
                    job.corpus.current_subset_directory, "utt2spk", "scp", dictionary_id=dict_id
                )
                spk2utt_path = job.construct_path(
                    job.corpus.current_subset_directory, "spk2utt", "scp", dictionary_id=dict_id
                )
                utt2spk = KaldiMapping()
                utt2spk.load(utt2spk_path)
                spk2utt = KaldiMapping(list_mapping=True)
                spk2utt.load(spk2utt_path)
                feature_archive = FeatureArchive(
                    feat_path,
                    utt2spk=utt2spk,
                    lda_mat_file_name=lda_mat_path,
                    deltas=True,
                )
                silence_phones = [
                    x
                    for x, in session.query(Phone.mapping_id).filter(
                        Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                    )
                ]
                computer = FmllrComputer(
                    self.model_path,
                    silence_phones,
                    spk2utt=spk2utt,
                    two_models=False,
                    **self.fmllr_options,
                )
                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                fmllr_logger.debug(f"Processing {lat_path} with features from {feat_path}")
                lattice_archive = LatticeArchive(lat_path, determinized=False)
                temp_trans_path = job.construct_path(
                    self.working_directory, "trans", "ark", dict_id
                )
                computer.export_transforms(
                    temp_trans_path,
                    feature_archive,
                    lattice_archive,
                    callback=self.callback,
                )
                feature_archive.close()
                lattice_archive.close()
                del feature_archive
                del lattice_archive
                del computer
                trans_archive = MatrixArchive(temp_trans_path)
                write_specifier = generate_write_specifier(
                    job.construct_path(
                        job.corpus.current_subset_directory, "trans", "ark", dictionary_id=dict_id
                    ),
                    write_scp=True,
                )
                writer = BaseFloatMatrixWriter(write_specifier)
                for speaker, trans in trans_archive:
                    writer.Write(str(speaker), trans)
                writer.Close()
                trans_archive.close()
                del trans_archive
                os.remove(temp_trans_path)


class FinalFmllrFunction(KaldiFunction):

    """
    Multiprocessing function for running final fMLLR estimation

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.final_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-determinize-pruned`
        Relevant Kaldi binary
    :kaldi_src:`lattice-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr`
        Relevant Kaldi binary
    :kaldi_src:`compose-transforms`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrArguments`
        Arguments for the function
    """

    def __init__(self, args: FinalFmllrArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.fmllr", self.log_path, job_name=self.job_name) as fmllr_logger,
        ):
            fmllr_logger.debug(f"Using acoustic model: {self.model_path}\n")
            job: typing.Optional[Job] = session.get(
                Job, self.job_name, options=[joinedload(Job.dictionaries), joinedload(Job.corpus)]
            )
            lda_mat_path = self.working_directory.joinpath("lda.mat")
            if not lda_mat_path.exists():
                lda_mat_path = None
            for dict_id in job.dictionary_ids:
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                fmllr_trans_path = job.construct_path(
                    job.corpus.split_directory, "trans", "scp", dictionary_id=dict_id
                )
                previous_transform_archive = None
                if not fmllr_trans_path.exists():
                    fmllr_logger.debug("Computing transforms from scratch")
                    fmllr_trans_path = None
                else:
                    fmllr_logger.debug(f"Updating previous transforms {fmllr_trans_path}")
                    previous_transform_archive = MatrixArchive(fmllr_trans_path)
                utt2spk_path = job.construct_path(
                    job.corpus.current_subset_directory, "utt2spk", "scp", dictionary_id=dict_id
                )
                spk2utt_path = job.construct_path(
                    job.corpus.current_subset_directory, "spk2utt", "scp", dictionary_id=dict_id
                )
                utt2spk = KaldiMapping()
                utt2spk.load(utt2spk_path)
                spk2utt = KaldiMapping(list_mapping=True)
                spk2utt.load(spk2utt_path)
                feature_archive = FeatureArchive(
                    feat_path,
                    utt2spk=utt2spk,
                    lda_mat_file_name=lda_mat_path,
                    transform_file_name=fmllr_trans_path,
                    deltas=True,
                )
                silence_phones = [
                    x
                    for x, in session.query(Phone.mapping_id).filter(
                        Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                    )
                ]
                computer = FmllrComputer(
                    self.model_path,
                    silence_phones,
                    spk2utt=spk2utt,
                    two_models=True,
                    **self.fmllr_options,
                )
                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                fmllr_logger.debug(f"Processing {lat_path} with features from {feat_path}")
                lattice_archive = LatticeArchive(lat_path, determinized=False)
                temp_trans_path = job.construct_path(
                    self.working_directory, "trans", "ark", dict_id
                )
                computer.export_transforms(
                    temp_trans_path,
                    feature_archive,
                    lattice_archive,
                    previous_transform_archive=previous_transform_archive,
                    callback=self.callback,
                )
                feature_archive.close()
                del previous_transform_archive
                del feature_archive
                del lattice_archive
                del computer
                if fmllr_trans_path is not None:
                    os.remove(fmllr_trans_path)
                    os.remove(fmllr_trans_path.with_suffix(".ark"))
                trans_archive = MatrixArchive(temp_trans_path)
                write_specifier = generate_write_specifier(
                    job.construct_path(
                        job.corpus.current_subset_directory, "trans", "ark", dictionary_id=dict_id
                    ),
                    write_scp=True,
                )
                writer = BaseFloatMatrixWriter(write_specifier)
                for speaker, trans in trans_archive:
                    writer.Write(str(speaker), trans)
                writer.Close()
                del trans_archive
                os.remove(temp_trans_path)


class FmllrRescoreFunction(KaldiFunction):
    """
    Multiprocessing function to rescore lattices following fMLLR estimation

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.fmllr_rescore_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-rescore-lattice`
        Relevant Kaldi binary
    :kaldi_src:`lattice-determinize-pruned`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreArguments`
        Arguments for the function
    """

    def __init__(self, args: FmllrRescoreArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.rescore_options = args.rescore_options

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.decode", self.log_path, job_name=self.job_name) as decode_logger,
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            rescorer = GmmRescorer(self.model_path, **self.rescore_options)
            for d in job.dictionaries:
                decode_logger.debug(f"Aligning for dictionary {d.id}")
                decode_logger.debug(f"Aligning with model: {self.model_path}")
                dict_id = d.id
                fst_path = job.construct_path(self.working_directory, "fsts", "ark", dict_id)
                decode_logger.debug(f"Training graph archive: {fst_path}")

                fmllr_path = job.construct_path(
                    job.corpus.current_subset_directory, "trans", "scp", dict_id
                )
                if not fmllr_path.exists():
                    fmllr_path = None
                lda_mat_path = self.working_directory.joinpath("lda.mat")
                if not lda_mat_path.exists():
                    lda_mat_path = None
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                utt2spk_path = job.construct_path(
                    job.corpus.current_subset_directory, "utt2spk", "scp", dict_id
                )
                utt2spk = KaldiMapping()
                utt2spk.load(utt2spk_path)
                decode_logger.debug(f"Feature path: {feat_path}")
                decode_logger.debug(f"LDA transform path: {lda_mat_path}")
                decode_logger.debug(f"Speaker transform path: {fmllr_path}")
                decode_logger.debug(f"utt2spk path: {utt2spk_path}")
                feature_archive = FeatureArchive(
                    feat_path,
                    utt2spk=utt2spk,
                    lda_mat_file_name=lda_mat_path,
                    transform_file_name=fmllr_path,
                    deltas=True,
                )
                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                tmp_lat_path = job.construct_path(
                    self.working_directory, "lat.tmp", "ark", dict_id
                )
                os.rename(lat_path, tmp_lat_path)
                lattice_archive = LatticeArchive(tmp_lat_path, determinized=True)
                rescorer.export_lattices(
                    lat_path, lattice_archive, feature_archive, callback=self.callback
                )
                lattice_archive.close()
                os.remove(tmp_lat_path)


@dataclass
class PerSpeakerDecodeArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.validation.corpus_validator.PerSpeakerDecodeFunction`"""

    working_directory: Path
    model_path: Path
    tree_path: Path
    decode_options: MetaDict
    order: int
    method: str


class PerSpeakerDecodeFunction(KaldiFunction):
    """
    Multiprocessing function to test utterance transcriptions with utterance and speaker ngram models

    See Also
    --------
    :kaldi_src:`compile-train-graphs-fsts`
        Relevant Kaldi binary
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary
    :kaldi_src:`lattice-oracle`
        Relevant Kaldi binary
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
    args: :class:`~montreal_forced_aligner.validation.corpus_validator.PerSpeakerDecodeArguments`
        Arguments for the function
    """

    def __init__(self, args: PerSpeakerDecodeArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.model_path = args.model_path
        self.decode_options = args.decode_options
        self.tree_path = args.tree_path
        self.order = args.order
        self.method = args.method
        self.word_symbols_paths = {}

    def _run(self) -> typing.Generator[typing.Tuple[int, str]]:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.decode", self.log_path, job_name=self.job_name) as decode_logger,
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                )
            ]

            for d in job.dictionaries:
                decode_logger.debug(f"Decoding for dictionary {d.id}")
                decode_logger.debug(f"Decoding with model: {self.model_path}")
                dict_id = d.id

                fmllr_path = job.construct_path(
                    job.corpus.current_subset_directory, "trans", "scp", dict_id
                )
                if not fmllr_path.exists():
                    fmllr_path = None
                lda_mat_path = self.working_directory.joinpath("lda.mat")
                if not lda_mat_path.exists():
                    lda_mat_path = None
                feat_path = job.construct_path(
                    job.corpus.current_subset_directory, "feats", "scp", dictionary_id=dict_id
                )
                utt2spk_path = job.construct_path(
                    job.corpus.current_subset_directory, "utt2spk", "scp", dict_id
                )
                utt2spk = KaldiMapping()
                utt2spk.load(utt2spk_path)
                decode_logger.debug(f"Feature path: {feat_path}")
                decode_logger.debug(f"LDA transform path: {lda_mat_path}")
                decode_logger.debug(f"Speaker transform path: {fmllr_path}")
                decode_logger.debug(f"utt2spk path: {utt2spk_path}")
                feature_archive = FeatureArchive(
                    feat_path,
                    utt2spk=utt2spk,
                    lda_mat_file_name=lda_mat_path,
                    transform_file_name=fmllr_path,
                    deltas=True,
                )

                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                alignment_file_name = job.construct_path(
                    self.working_directory, "ali", "ark", dict_id
                )
                words_path = job.construct_path(self.working_directory, "words", "ark", dict_id)
                boost_silence = self.decode_options.pop("boost_silence", 1.0)

                current_speaker = None
                write_specifier = generate_write_specifier(lat_path, write_scp=False)
                alignment_writer = None
                if alignment_file_name:
                    alignment_write_specifier = generate_write_specifier(
                        alignment_file_name, write_scp=False
                    )
                    alignment_writer = Int32VectorWriter(alignment_write_specifier)
                word_writer = None
                if words_path:
                    word_write_specifier = generate_write_specifier(words_path, write_scp=False)
                    word_writer = Int32VectorWriter(word_write_specifier)
                writer = CompactLatticeWriter(write_specifier)
                for utt_id, speaker_id in (
                    session.query(Utterance.kaldi_id, Utterance.speaker_id)
                    .filter(Utterance.job_id == job.id)
                    .order_by(Utterance.kaldi_id)
                ):
                    if speaker_id != current_speaker:
                        lm_path = os.path.join(d.temp_directory, f"{speaker_id}.fst")
                        hclg_fst = ConstFst.Read(str(lm_path))
                        decoder = GmmDecoder(self.model_path, hclg_fst, **self.decode_options)
                        if boost_silence != 1.0:
                            decoder.boost_silence(boost_silence, silence_phones)
                    for transcription in decoder.decode_utterances(feature_archive):
                        if transcription is None:
                            continue
                        utt_id = int(transcription.utterance_id.split("-")[-1])
                        self.callback((utt_id, transcription.likelihood))
                        writer.Write(str(transcription.utterance_id), transcription.lattice)
                        if alignment_writer is not None:
                            alignment_writer.Write(
                                str(transcription.utterance_id), transcription.alignment
                            )
                        if word_writer is not None:
                            word_writer.Write(str(transcription.utterance_id), transcription.words)
                writer.Close()
                if alignment_writer is not None:
                    alignment_writer.Close()
                if word_writer is not None:
                    word_writer.Close()


class DecodePhoneFunction(KaldiFunction):
    """
    Multiprocessing function for performing decoding

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.decode_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`
        Arguments for the function
    """

    def __init__(self, args: DecodePhoneArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.hclg_path = args.hclg_path
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def _run(self) -> None:
        """Run the function"""
        with (
            self.session() as session,
            thread_logger("kalpy.decode", self.log_path, job_name=self.job_name) as decode_logger,
        ):
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            silence_phones = [
                x
                for x, in session.query(Phone.mapping_id).filter(
                    Phone.phone_type.in_([PhoneType.silence, PhoneType.oov])
                )
            ]
            phones = session.query(Phone.mapping_id, Phone.phone)
            reversed_phone_mapping = {}
            for p_id, phone in phones:
                reversed_phone_mapping[p_id] = phone
            hclg_fst = ConstFst.Read(str(self.hclg_path))
            for d in job.dictionaries:
                decode_logger.debug(f"Decoding for dictionary {d.id}")
                decode_logger.debug(f"Decoding with model: {self.model_path}")
                dict_id = d.id
                feature_archive = job.construct_feature_archive(self.working_directory, dict_id)
                lat_path = job.construct_path(self.working_directory, "lat", "ark", dict_id)
                alignment_file_name = job.construct_path(
                    self.working_directory, "ali", "ark", dict_id
                )
                words_path = job.construct_path(self.working_directory, "words", "ark", dict_id)

                boost_silence = self.decode_options.pop("boost_silence", 1.0)
                decoder = GmmDecoder(self.model_path, hclg_fst, **self.decode_options)
                if boost_silence != 1.0:
                    decoder.boost_silence(boost_silence, silence_phones)
                decoder.export_lattices(
                    lat_path,
                    feature_archive,
                    word_file_name=words_path,
                    alignment_file_name=alignment_file_name,
                    callback=self.callback,
                )
