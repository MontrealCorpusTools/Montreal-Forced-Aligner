"""
Transcription
=============

"""
from __future__ import annotations

import csv
import itertools
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import tqdm
from praatio import textgrid
from sqlalchemy.orm import joinedload, selectinload

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.alignment.multiprocessing import construct_output_path
from montreal_forced_aligner.data import TextFileType, TextgridFormats
from montreal_forced_aligner.db import (
    Corpus,
    Dictionary,
    File,
    SoundFile,
    Speaker,
    SpeakerOrdering,
    Utterance,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError, PlatformError
from montreal_forced_aligner.helper import load_configuration, parse_old_features, score_wer
from montreal_forced_aligner.models import AcousticModel, LanguageModel
from montreal_forced_aligner.transcription.multiprocessing import (
    CarpaLmRescoreArguments,
    CarpaLmRescoreFunction,
    CreateHclgArguments,
    CreateHclgFunction,
    DecodeArguments,
    DecodeFunction,
    FinalFmllrArguments,
    FinalFmllrFunction,
    FmllrRescoreArguments,
    FmllrRescoreFunction,
    InitialFmllrArguments,
    InitialFmllrFunction,
    LatGenFmllrArguments,
    LatGenFmllrFunction,
    LmRescoreArguments,
    LmRescoreFunction,
    ScoreArguments,
    ScoreFunction,
)
from montreal_forced_aligner.utils import (
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    thirdparty_binary,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from montreal_forced_aligner.abc import MetaDict

__all__ = ["Transcriber", "TranscriberMixin"]


class TranscriberMixin:
    """Abstract class for MFA transcribers

    Parameters
    ----------
    transition_scale: float
        Transition scale, defaults to 1.0
    acoustic_scale: float
        Acoustic scale, defaults to 0.1
    self_loop_scale: float
        Self-loop scale, defaults to 0.1
    beam: int
        Size of the beam to use in decoding, defaults to 10
    silence_weight: float
        Weight on silence in fMLLR estimation
    max_active: int
        Max active for decoding
    lattice_beam: int
        Beam width for decoding lattices
    first_beam: int
        Beam for decoding in initial speaker-independent pass, only used if ``uses_speaker_adaptation`` is true
    first_max_active: int
        Max active for decoding in initial speaker-independent pass, only used if ``uses_speaker_adaptation`` is true
    language_model_weight: float
        Weight of language model
    word_insertion_penalty: float
        Penalty for inserting words
    """

    def __init__(
        self,
        transition_scale: float = 1.0,
        acoustic_scale: float = 0.083333,
        self_loop_scale: float = 0.1,
        beam: int = 10,
        silence_weight: float = 0.01,
        max_active: int = 7000,
        lattice_beam: int = 6,
        first_beam: int = 10,
        first_max_active: int = 2000,
        language_model_weight: int = 10,
        word_insertion_penalty: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.beam = beam
        self.acoustic_scale = acoustic_scale
        self.self_loop_scale = self_loop_scale
        self.transition_scale = transition_scale
        self.silence_weight = silence_weight
        self.max_active = max_active
        self.lattice_beam = lattice_beam
        self.first_beam = first_beam
        self.first_max_active = first_max_active
        self.language_model_weight = language_model_weight
        self.word_insertion_penalty = word_insertion_penalty

    def save_transcription_evaluation(self, output_directory: str) -> None:
        """
        Save transcription evaluation to an output directory

        Parameters
        ----------
        output_directory: str
            Directory to save evaluation
        """
        output_path = os.path.join(output_directory, "transcription_evaluation.csv")
        with open(output_path, "w", newline="", encoding="utf8") as f, self.session() as session:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file",
                    "speaker",
                    "begin",
                    "end",
                    "duration",
                    "word_count",
                    "oov_count",
                    "gold_transcript",
                    "hypothesis",
                    "WER",
                    "CER",
                ]
            )
            utterances = (
                session.query(
                    Speaker.name,
                    File.name,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.duration,
                    Utterance.normalized_text,
                    Utterance.transcription_text,
                    Utterance.oovs,
                    Utterance.word_error_rate,
                    Utterance.character_error_rate,
                )
                .join(Utterance.speaker)
                .join(Utterance.file)
                .filter(Utterance.normalized_text != None)  # noqa
                .filter(Utterance.normalized_text != "")
            )

            for (
                speaker,
                file,
                begin,
                end,
                duration,
                text,
                transcription_text,
                oovs,
                word_error_rate,
                character_error_rate,
            ) in utterances:
                word_count = text.count(" ") + 1
                oov_count = oovs.count(" ") + 1
                writer.writerow(
                    [
                        file,
                        speaker,
                        begin,
                        end,
                        duration,
                        word_count,
                        oov_count,
                        text,
                        transcription_text,
                        word_error_rate,
                        character_error_rate,
                    ]
                )

    def compute_wer(self):
        """
        Evaluates the transcripts if there are reference transcripts

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        if not hasattr(self, "db_engine"):
            raise Exception("Must be used as part of a class with a database engine")
        self.log_info("Evaluating transcripts...")
        # Sentence-level measures
        incorrect = 0
        total_count = 0
        # Word-level measures
        total_word_edits = 0
        total_word_length = 0

        # Character-level measures
        total_character_edits = 0
        total_character_length = 0

        indices = []
        to_comp = []

        update_mappings = []
        with self.session() as session:
            utterances = session.query(Utterance)
            utterances = utterances.filter(Utterance.normalized_text != None)  # noqa
            utterances = utterances.filter(Utterance.normalized_text != "")
            for utt in utterances:
                g = utt.normalized_text.split()
                total_count += 1
                total_word_length += len(g)
                character_length = len("".join(g))
                total_character_length += character_length

                if not utt.transcription_text:
                    incorrect += 1
                    total_word_edits += len(g)
                    total_character_edits += character_length
                    update_mappings.append(
                        {"id": utt.id, "word_error_rate": 1.0, "character_error_rate": 1.0}
                    )
                    continue

                h = utt.transcription_text.split()
                if g != h:
                    indices.append(utt.id)
                    to_comp.append((g, h))
                    incorrect += 1
                else:
                    update_mappings.append(
                        {"id": utt.id, "word_error_rate": 0.0, "character_error_rate": 0.0}
                    )

            with mp.Pool(self.num_jobs) as pool:
                gen = pool.starmap(score_wer, to_comp)
                for i, (word_edits, word_length, character_edits, character_length) in enumerate(
                    gen
                ):
                    utt_id = indices[i]
                    update_mappings.append(
                        {
                            "id": utt_id,
                            "word_error_rate": word_edits / word_length,
                            "character_error_rate": character_edits / character_length,
                        }
                    )
                    total_word_edits += word_edits
                    total_character_edits += character_edits

            session.bulk_update_mappings(Utterance, update_mappings)
            session.commit()
        ser = incorrect / total_count
        wer = total_word_edits / total_word_length
        cer = total_character_edits / total_character_length
        return ser, wer, cer

    @property
    def decode_options(self) -> MetaDict:
        """Options needed for decoding"""
        return {
            "first_beam": self.first_beam,
            "beam": self.beam,
            "first_max_active": self.first_max_active,
            "max_active": self.max_active,
            "lattice_beam": self.lattice_beam,
            "acoustic_scale": self.acoustic_scale,
            "uses_speaker_adaptation": self.uses_speaker_adaptation,
        }

    @property
    def score_options(self) -> MetaDict:
        """Options needed for scoring lattices"""
        return {
            "language_model_weight": self.language_model_weight,
            "word_insertion_penalty": self.word_insertion_penalty,
        }

    @property
    def transcribe_fmllr_options(self) -> MetaDict:
        """Options needed for calculating fMLLR transformations"""
        return {
            "acoustic_scale": self.acoustic_scale,
            "silence_weight": self.silence_weight,
            "lattice_beam": self.lattice_beam,
        }

    @property
    def lm_rescore_options(self) -> MetaDict:
        """Options needed for rescoring the language model"""
        return {
            "acoustic_scale": self.acoustic_scale,
        }


class Transcriber(TranscriberMixin, CorpusAligner, TopLevelMfaWorker):
    """
    Class for performing transcription.

    Parameters
    ----------
    acoustic_model_path : str
        Path to acoustic model
    language_model_path : str
        Path to language model model
    evaluation_mode: bool
        Flag for evaluating generated transcripts against the actual transcripts, defaults to False
    min_language_model_weight: int
        Minimum language model weight to use in evaluation mode, defaults to 7
    max_language_model_weight: int
        Maximum language model weight to use in evaluation mode, defaults to 17
    word_insertion_penalties: list[float]
        List of word insertion penalties to use in evaluation mode, defaults to [0, 0.5, 1.0]

    See Also
    --------
    :class:`~montreal_forced_aligner.transcription.transcriber.TranscriberMixin`
        For transcription parameters
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For corpus and dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.FileExporterMixin`
        For file exporting parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters

    Attributes
    ----------
    acoustic_model: AcousticModel
        Acoustic model
    language_model: LanguageModel
        Language model
    """

    def __init__(
        self,
        acoustic_model_path: str,
        language_model_path: str,
        evaluation_mode: bool = False,
        min_language_model_weight: int = 7,
        max_language_model_weight: int = 17,
        word_insertion_penalties: List[float] = None,
        output_type: str = "transcription",
        **kwargs,
    ):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        kwargs.update(self.acoustic_model.parameters)
        super(Transcriber, self).__init__(**kwargs)
        self.language_model = LanguageModel(language_model_path, self.model_directory)
        if word_insertion_penalties is None:
            word_insertion_penalties = [0, 0.5, 1.0]
        self.min_language_model_weight = min_language_model_weight
        self.max_language_model_weight = max_language_model_weight
        self.evaluation_mode = evaluation_mode
        self.word_insertion_penalties = word_insertion_penalties
        self.output_type = output_type

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:
        """
        Parse configuration parameters from a config file and command line arguments

        Parameters
        ----------
        config_path: str, optional
            Path to yaml configuration file
        args: :class:`~argparse.Namespace`, optional
            Arguments parsed by argparse
        unknown_args: list[str], optional
            List of unknown arguments from argparse

        Returns
        -------
        dict[str, Any]
            Dictionary of specified configuration parameters
        """
        global_params = {}
        if config_path and os.path.exists(config_path):
            data = load_configuration(config_path)
            data = parse_old_features(data)
            for k, v in data.items():
                if k == "features":
                    global_params.update(v)
                else:
                    if v is None and k in cls.nullable_fields:
                        v = []
                    global_params[k] = v

        global_params.update(cls.parse_args(args, unknown_args))
        if hasattr(args, "language_model_weight") and args.language_model_weight is not None:
            global_params["min_language_model_weight"] = args.language_model_weight
            global_params["max_language_model_weight"] = args.language_model_weight + 1
        if hasattr(args, "word_insertion_penalty") and args.word_insertion_penalty is not None:
            global_params["word_insertion_penalties"] = [args.word_insertion_penalty]
        return global_params

    def setup(self) -> None:
        """Set up transcription"""
        if self.initialized:
            return
        begin = time.time()
        os.makedirs(self.working_log_directory, exist_ok=True)
        check = self.check_previous_run()
        if check:
            self.log_debug(
                "There were some differences in the current run compared to the last one. "
                "This may cause issues, run with --clean, if you hit an error."
            )
        self.load_corpus()
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(dirty_path):
            shutil.rmtree(self.working_directory, ignore_errors=True)
        os.makedirs(self.working_log_directory, exist_ok=True)
        dirty_path = os.path.join(self.model_directory, "dirty")

        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.model_directory)
        log_dir = os.path.join(self.model_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.acoustic_model.validate(self)
        self.acoustic_model.export_model(self.model_directory)
        self.acoustic_model.export_model(self.working_directory)
        logger = logging.getLogger(self.identifier)
        self.acoustic_model.log_details(logger)
        self.create_decoding_graph()
        self.initialized = True
        self.log_debug(f"Setup for transcription in {time.time() - begin} seconds")

    def create_hclgs_arguments(self) -> Dict[str, CreateHclgArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgFunction`

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgArguments`]
            Per dictionary arguments for HCLG
        """
        args = {}
        with self.session() as session:
            for d in session.query(Dictionary):
                args[d.id] = CreateHclgArguments(
                    d.id,
                    getattr(self, "db_path", ""),
                    os.path.join(self.model_directory, "log", f"hclg.{d.id}.log"),
                    self.model_directory,
                    os.path.join(self.model_directory, "{file_name}" + f".{d.id}.fst"),
                    os.path.join(self.model_directory, f"words.{d.id}.txt"),
                    os.path.join(self.model_directory, f"G.{d.id}.carpa"),
                    self.language_model.small_arpa_path,
                    self.language_model.medium_arpa_path,
                    self.language_model.carpa_path,
                    self.model_path,
                    d.lexicon_disambig_fst_path,
                    d.disambiguation_symbols_int_path,
                    self.hclg_options,
                    self.word_mapping(d.id),
                )
        return args

    def decode_arguments(self) -> List[DecodeArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`]
            Arguments for processing
        """
        feat_string = self.construct_feature_proc_strings()
        return [
            DecodeArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"decode.{j.name}.log"),
                j.dictionary_ids,
                feat_string[j.name],
                self.decode_options,
                self.alignment_model_path,
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "words", "txt"),
                j.construct_dictionary_dependent_paths(self.model_directory, "HCLG", "fst"),
            )
            for j in self.jobs
        ]

    def score_arguments(self) -> List[ScoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreArguments`]
            Arguments for processing
        """
        return [
            ScoreArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.evaluation_directory, f"score.{j.name}.log"),
                j.dictionary_ids,
                self.score_options,
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.rescored", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.carpa.rescored", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "words", "txt"),
                j.construct_path_dictionary(self.evaluation_directory, "tra", "scp"),
                j.construct_path_dictionary(self.evaluation_directory, "ali", "ark"),
            )
            for j in self.jobs
        ]

    def lm_rescore_arguments(self) -> List[LmRescoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreArguments`]
            Arguments for processing
        """
        return [
            LmRescoreArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"lm_rescore.{j.name}.log"),
                j.dictionary_ids,
                self.lm_rescore_options,
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.rescored", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G.small", "fst"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G.med", "fst"),
            )
            for j in self.jobs
        ]

    def carpa_lm_rescore_arguments(self) -> List[CarpaLmRescoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreArguments`]
            Arguments for processing
        """
        return [
            CarpaLmRescoreArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"carpa_lm_rescore.{j.name}.log"),
                j.dictionary_ids,
                j.construct_path_dictionary(self.working_directory, "lat.rescored", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.carpa.rescored", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G.med", "fst"),
                j.construct_dictionary_dependent_paths(self.model_directory, "G", "carpa"),
            )
            for j in self.jobs
        ]

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for calculating fMLLR"""
        options = super().fmllr_options
        options["acoustic_scale"] = self.acoustic_scale
        options["sil_phones"] = self.silence_csl
        options["lattice_beam"] = self.lattice_beam
        return options

    def initial_fmllr_arguments(self) -> List[InitialFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            InitialFmllrArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"initial_fmllr.{j.name}.log"),
                j.dictionary_ids,
                feat_strings[j.name],
                self.model_path,
                self.fmllr_options,
                j.construct_path_dictionary(self.working_directory, "trans", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                j.construct_path_dictionary(self.data_directory, "spk2utt", "scp"),
            )
            for j in self.jobs
        ]

    def lat_gen_fmllr_arguments(self) -> List[LatGenFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            LatGenFmllrArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"lat_gen_fmllr.{j.name}.log"),
                j.dictionary_ids,
                feat_strings[j.name],
                self.model_path,
                self.decode_options,
                j.construct_dictionary_dependent_paths(self.model_directory, "words", "txt"),
                j.construct_dictionary_dependent_paths(self.model_directory, "HCLG", "fst"),
                j.construct_path_dictionary(self.working_directory, "lat.tmp", "ark"),
            )
            for j in self.jobs
        ]

    def final_fmllr_arguments(self) -> List[FinalFmllrArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            FinalFmllrArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"final_fmllr.{j.name}.log"),
                j.dictionary_ids,
                feat_strings[j.name],
                self.model_path,
                self.fmllr_options,
                j.construct_path_dictionary(self.working_directory, "trans", "ark"),
                j.construct_path_dictionary(self.data_directory, "spk2utt", "scp"),
                j.construct_path_dictionary(self.working_directory, "lat.tmp", "ark"),
            )
            for j in self.jobs
        ]

    def fmllr_rescore_arguments(self) -> List[FmllrRescoreArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            FmllrRescoreArguments(
                j.name,
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"fmllr_rescore.{j.name}.log"),
                j.dictionary_ids,
                feat_strings[j.name],
                self.model_path,
                self.fmllr_options,
                j.construct_path_dictionary(self.working_directory, "lat.tmp", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
            )
            for j in self.jobs
        ]

    @property
    def workflow_identifier(self) -> str:
        """Transcriber identifier"""
        return "transcriber"

    @property
    def evaluation_directory(self):
        """Evaluation directory path for the current language model weight and word insertion penalty"""
        eval_string = f"eval_{self.language_model_weight}_{self.word_insertion_penalty}"
        path = os.path.join(self.working_directory, eval_string)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def evaluation_log_directory(self) -> str:
        """Log directory for the current evaluation"""
        return os.path.join(self.evaluation_directory, "log")

    @property
    def model_directory(self) -> str:
        """Model directory for the transcriber"""
        return os.path.join(self.output_directory, "models")

    @property
    def model_path(self) -> str:
        """Acoustic model file path"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def alignment_model_path(self) -> str:
        """Alignment (speaker-independent) acoustic model file path"""
        path = os.path.join(self.working_directory, "final.alimdl")
        if os.path.exists(path):
            return path
        return self.model_path

    @property
    def hclg_options(self):
        """Options for constructing HCLG FSTs"""
        context_width, central_pos = self.get_tree_info()
        return {
            "context_width": context_width,
            "central_pos": central_pos,
            "self_loop_scale": self.self_loop_scale,
            "transition_scale": self.transition_scale,
        }

    def get_tree_info(self) -> Tuple[int, int]:
        """
        Get the context width and central position for the acoustic model

        Returns
        -------
        int
            Context width
        int
            Central position
        """
        tree_proc = subprocess.Popen(
            [thirdparty_binary("tree-info"), os.path.join(self.model_directory, "tree")],
            encoding="utf8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = tree_proc.communicate()
        context_width = 1
        central_pos = 0
        for line in stdout.split("\n"):
            text = line.strip().split(" ")
            if text[0] == "context-width":
                context_width = int(text[1])
            elif text[0] == "central-position":
                central_pos = int(text[1])
        return context_width, central_pos

    def create_hclgs(self):
        """
        Create HCLG.fst files for every dictionary being used by a :class:`~montreal_forced_aligner.transcription.transcriber.Transcriber`
        """

        dict_arguments = self.create_hclgs_arguments()
        dict_arguments = list(dict_arguments.values())
        self.log_info("Generating HCLG.fst...")
        if self.use_mp:
            error_dict = {}
            return_queue = mp.Queue()
            stopped = Stopped()
            procs = []
            for i, args in enumerate(dict_arguments):
                function = CreateHclgFunction(args)
                p = KaldiProcessWorker(i, return_queue, function, stopped)
                procs.append(p)
                p.start()
            with tqdm.tqdm(
                total=len(dict_arguments), disable=getattr(self, "quiet", False)
            ) as pbar:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    result, hclg_path = result
                    if result:
                        self.log_debug(f"Done generating {hclg_path}!")
                    else:
                        self.log_warning(f"There was an error in generating {hclg_path}")
                    pbar.update(1)
            for p in procs:
                p.join()
            if error_dict:
                for v in error_dict.values():
                    raise v
        else:
            for args in dict_arguments:
                function = CreateHclgFunction(args)
                with tqdm.tqdm(
                    total=len(dict_arguments), disable=getattr(self, "quiet", False)
                ) as pbar:
                    for result, hclg_path in function.run():
                        if result:
                            self.log_debug(f"Done generating {hclg_path}!")
                        else:
                            self.log_warning(f"There was an error in generating {hclg_path}")
                        pbar.update(1)
        error_logs = []
        for arg in dict_arguments:
            if not os.path.exists(arg.hclg_path):
                error_logs.append(arg.log_path)
        if error_logs:
            raise KaldiProcessingError(error_logs)

    def create_decoding_graph(self) -> None:
        """
        Create decoding graph for use in transcription

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        done_path = os.path.join(self.model_directory, "done")
        if os.path.exists(done_path):
            self.log_info("Graph construction already done, skipping!")
        log_dir = os.path.join(self.model_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.write_lexicon_information(write_disambiguation=True)
        with self.session() as session:
            for d in session.query(Dictionary):
                words_path = os.path.join(self.model_directory, f"words.{d.id}.txt")
                shutil.copyfile(d.words_symbol_path, words_path)

        big_arpa_path = self.language_model.carpa_path
        small_arpa_path = self.language_model.small_arpa_path
        medium_arpa_path = self.language_model.medium_arpa_path
        if not os.path.exists(small_arpa_path) or not os.path.exists(medium_arpa_path):
            self.log_warning(
                "Creating small and medium language models from scratch, this may take some time. "
                "Running `mfa train_lm` on the ARPA file will remove this warning."
            )
            if sys.platform == "win32":
                raise PlatformError("ngram")
            self.log_info("Parsing large ngram model...")
            mod_path = os.path.join(self.model_directory, "base_lm.mod")
            new_carpa_path = os.path.join(self.model_directory, "base_lm.arpa")
            with open(big_arpa_path, "r", encoding="utf8") as inf, open(
                new_carpa_path, "w", encoding="utf8"
            ) as outf:
                for line in inf:
                    outf.write(line.lower())
            big_arpa_path = new_carpa_path
            subprocess.call(["ngramread", "--ARPA", big_arpa_path, mod_path])

            if not os.path.exists(small_arpa_path):
                self.log_info(
                    "Generating small model from the large ARPA with a pruning threshold of 3e-7"
                )
                prune_thresh_small = 0.0000003
                small_mod_path = mod_path.replace(".mod", "_small.mod")
                subprocess.call(
                    [
                        "ngramshrink",
                        "--method=relative_entropy",
                        f"--theta={prune_thresh_small}",
                        mod_path,
                        small_mod_path,
                    ]
                )
                subprocess.call(["ngramprint", "--ARPA", small_mod_path, small_arpa_path])

            if not os.path.exists(medium_arpa_path):
                self.log_info(
                    "Generating medium model from the large ARPA with a pruning threshold of 1e-7"
                )
                prune_thresh_medium = 0.0000001
                med_mod_path = mod_path.replace(".mod", "_med.mod")
                subprocess.call(
                    [
                        "ngramshrink",
                        "--method=relative_entropy",
                        f"--theta={prune_thresh_medium}",
                        mod_path,
                        med_mod_path,
                    ]
                )
                subprocess.call(["ngramprint", "--ARPA", med_mod_path, medium_arpa_path])
        try:
            self.create_hclgs()
        except Exception as e:
            dirty_path = os.path.join(self.model_directory, "dirty")
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                import logging

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
            raise

    def score(self) -> None:
        """
        Score the decoded transcriptions

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreFunction`
            Multiprocessing function
        :class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreArguments`
            Arguments for function
        """
        with tqdm.tqdm(
            total=self.num_utterances, disable=getattr(self, "quiet", False)
        ) as pbar, open(
            os.path.join(self.evaluation_directory, "score_costs.csv"), "w", encoding="utf8"
        ) as log_file:
            log_file.write("utterance,graph_cost,acoustic_cost,total_cost,num_frames\n")
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.score_arguments()):
                    function = ScoreFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)

                    (
                        utterance,
                        graph_cost,
                        acoustic_cost,
                        total_cost,
                        num_frames,
                    ) = result
                    log_file.write(
                        f"{utterance},{graph_cost},{acoustic_cost},{total_cost},{num_frames}\n"
                    )
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.score_arguments():
                    function = ScoreFunction(args)
                    for (
                        utterance,
                        graph_cost,
                        acoustic_cost,
                        total_cost,
                        num_frames,
                    ) in function.run():
                        log_file.write(
                            f"{utterance},{graph_cost},{acoustic_cost},{total_cost},{num_frames}\n"
                        )
                        pbar.update(1)

    def score_transcriptions(self):
        """
        Score transcriptions if reference text is available in the corpus

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.ScoreFunction`
            Multiprocessing helper function for each job
        :meth:`.Transcriber.score_arguments`
            Job method for generating arguments for this function

        """
        if self.evaluation_mode:
            best_wer = 10000
            best = None
            evaluations = list(
                itertools.product(
                    range(self.min_language_model_weight, self.max_language_model_weight),
                    self.word_insertion_penalties,
                )
            )
            with tqdm.tqdm(total=len(evaluations), disable=getattr(self, "quiet", False)) as pbar:
                for lmwt, wip in evaluations:
                    pbar.update(1)
                    self.language_model_weight = lmwt
                    self.word_insertion_penalty = wip
                    os.makedirs(self.evaluation_log_directory, exist_ok=True)

                    self.log_debug(
                        f"Evaluating with language model weight={lmwt} and word insertion penalty={wip}..."
                    )
                    self.score()

                    ser, wer = self.evaluate_transcriptions()
                    if wer < best_wer:
                        best = (lmwt, wip)
                        best_wer = wer
            self.language_model_weight = best[0]
            self.word_insertion_penalty = best[1]
            self.log_info(
                f"Best language model weight={best[0]}, best word insertion penalty={best[1]}, WER={best_wer:.2f}%"
            )
            for score_args in self.score_arguments():
                for p in score_args.tra_paths.values():
                    shutil.copyfile(
                        p,
                        p.replace(self.evaluation_directory, self.working_directory),
                    )
        else:
            self.score()

    def calc_initial_fmllr(self):
        """
        Calculate initial fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`
            Multiprocessing function
        :meth:`.Transcriber.initial_fmllr_arguments`
            Arguments for function
        """
        self.log_info("Calculating initial fMLLR transforms...")
        sum_errors = 0
        with tqdm.tqdm(total=self.num_speakers, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.initial_fmllr_arguments()):
                    function = InitialFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.initial_fmllr_arguments():
                    function = InitialFmllrFunction(args)
                    for _ in function.run():
                        pbar.update(1)
            if sum_errors:
                self.log_warning(f"{sum_errors} utterances had errors on calculating fMLLR.")

    def lat_gen_fmllr(self):
        """
        Generate lattice with fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`
            Multiprocessing function
        :meth:`.Transcriber.lat_gen_fmllr_arguments`
            Arguments for function
        """
        self.log_info("Regenerating lattices with fMLLR transforms...")
        with tqdm.tqdm(
            total=self.num_utterances, disable=getattr(self, "quiet", False)
        ) as pbar, open(
            os.path.join(self.working_log_directory, "lat_gen_fmllr_log_like.csv"),
            "w",
            encoding="utf8",
        ) as log_file:
            log_file.write("utterance,log_likelihood,num_frames\n")
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.lat_gen_fmllr_arguments()):
                    function = LatGenFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    utterance, log_likelihood, num_frames = result
                    log_file.write(f"{utterance},{log_likelihood},{num_frames}\n")
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.lat_gen_fmllr_arguments():
                    function = LatGenFmllrFunction(args)
                    for utterance, log_likelihood, num_frames in function.run():
                        log_file.write(f"{utterance},{log_likelihood},{num_frames}\n")
                        pbar.update(1)

    def calc_final_fmllr(self):
        """
        Calculate final fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`
            Multiprocessing function
        :meth:`.Transcriber.final_fmllr_arguments`
            Arguments for function
        """
        self.log_info("Calculating final fMLLR transforms...")
        sum_errors = 0
        with tqdm.tqdm(total=self.num_speakers, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.final_fmllr_arguments()):
                    function = FinalFmllrFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.final_fmllr_arguments():
                    function = FinalFmllrFunction(args)
                    for _ in function.run():
                        pbar.update(1)
            if sum_errors:
                self.log_warning(f"{sum_errors} utterances had errors on calculating fMLLR.")

    def fmllr_rescore(self):
        """
        Rescore lattices with final fMLLR transforms

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`
            Multiprocessing function
        :meth:`.Transcriber.fmllr_rescore_arguments`
            Arguments for function
        """
        self.log_info("Rescoring fMLLR lattices with final transform...")
        sum_errors = 0
        with tqdm.tqdm(total=self.num_utterances, disable=getattr(self, "quiet", False)) as pbar:
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.fmllr_rescore_arguments()):
                    function = FmllrRescoreFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    done, errors = result
                    sum_errors += errors
                    pbar.update(done + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.fmllr_rescore_arguments():
                    function = FmllrRescoreFunction(args)
                    for done, errors in function.run():
                        sum_errors += errors
                        pbar.update(done + errors)
            if sum_errors:
                self.log_warning(f"{errors} utterances had errors on calculating fMLLR.")

    def transcribe_fmllr(self) -> None:
        """
        Run fMLLR estimation over initial decoding lattices and rescore

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`
            Multiprocessing helper function for each job

        """
        self.calc_initial_fmllr()

        self.speaker_independent = False

        self.lat_gen_fmllr()

        self.calc_final_fmllr()

        self.fmllr_rescore()

        self.lm_rescore()

        self.carpa_lm_rescore()

    def decode(self):
        """
        Generate lattices

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`
            Multiprocessing function
        :meth:`.Transcriber.decode_arguments`
            Arguments for function
        """
        self.log_info("Generating lattices...")
        with tqdm.tqdm(
            total=self.num_utterances, disable=getattr(self, "quiet", False)
        ) as pbar, open(
            os.path.join(self.working_log_directory, "decode_log_like.csv"), "w", encoding="utf8"
        ) as log_file:
            log_file.write("utterance,log_likelihood,num_frames\n")
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(self.decode_arguments()):
                    function = DecodeFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    utterance, log_likelihood, num_frames = result
                    log_file.write(f"{utterance},{log_likelihood},{num_frames}\n")
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in self.decode_arguments():
                    function = DecodeFunction(args)
                    for utterance, log_likelihood, num_frames in function.run():
                        log_file.write(f"{utterance},{log_likelihood},{num_frames}\n")
                        pbar.update(1)

    def lm_rescore(self):
        """
        Rescore lattices with bigger language model

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`
            Multiprocessing function
        :meth:`.Transcriber.lm_rescore_arguments`
            Arguments for function
        """
        self.log_info("Rescoring lattices with medium G.fst...")
        if self.use_mp:
            error_dict = {}
            return_queue = mp.Queue()
            stopped = Stopped()
            procs = []
            for i, args in enumerate(self.lm_rescore_arguments()):
                function = LmRescoreFunction(args)
                p = KaldiProcessWorker(i, return_queue, function, stopped)
                procs.append(p)
                p.start()
            with tqdm.tqdm(
                total=self.num_utterances, disable=getattr(self, "quiet", False)
            ) as pbar:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    succeeded, failed = result
                    if failed:
                        self.log_warning("Some lattices failed to be rescored")
                    pbar.update(succeeded + failed)
            for p in procs:
                p.join()
            if error_dict:
                for v in error_dict.values():
                    raise v
        else:
            for args in self.lm_rescore_arguments():
                function = LmRescoreFunction(args)
                with tqdm.tqdm(total=self.num_jobs, disable=getattr(self, "quiet", False)) as pbar:
                    for succeeded, failed in function.run():
                        if failed:
                            self.log_warning("Some lattices failed to be rescored")
                        pbar.update(succeeded + failed)

    def carpa_lm_rescore(self):
        """
        Rescore lattices with CARPA language model

        See Also
        -------
        :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`
            Multiprocessing function
        :meth:`.Transcriber.carpa_lm_rescore_arguments`
            Arguments for function
        """
        self.log_info("Rescoring lattices with large G.carpa...")
        if self.use_mp:
            error_dict = {}
            return_queue = mp.Queue()
            stopped = Stopped()
            procs = []
            for i, args in enumerate(self.carpa_lm_rescore_arguments()):
                function = CarpaLmRescoreFunction(args)
                p = KaldiProcessWorker(i, return_queue, function, stopped)
                procs.append(p)
                p.start()
            with tqdm.tqdm(
                total=self.num_utterances, disable=getattr(self, "quiet", False)
            ) as pbar:
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    succeeded, failed = result
                    if failed:
                        self.log_warning("Some lattices failed to be rescored")
                    pbar.update(succeeded + failed)
            for p in procs:
                p.join()
            if error_dict:
                for v in error_dict.values():
                    raise v
        else:
            for args in self.carpa_lm_rescore_arguments():
                function = CarpaLmRescoreFunction(args)
                with tqdm.tqdm(
                    total=self.num_utterances, disable=getattr(self, "quiet", False)
                ) as pbar:
                    for succeeded, failed in function.run():
                        if failed:
                            self.log_warning("Some lattices failed to be rescored")
                        pbar.update(succeeded + failed)

    def transcribe(self) -> None:
        """
        Transcribe the corpus

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`
            Multiprocessing helper function for each job

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.log_info("Beginning transcription...")
        done_path = os.path.join(self.working_directory, "done")
        dirty_path = os.path.join(self.working_directory, "dirty")
        try:
            if not os.path.exists(done_path):
                self.speaker_independent = True

                self.decode()
                if self.uses_speaker_adaptation:
                    self.log_info("Performing speaker adjusted transcription...")
                    self.transcribe_fmllr()
                else:
                    self.lm_rescore()
                    self.carpa_lm_rescore()
            else:
                self.log_info("Transcription already done, skipping!")
            self.score_transcriptions()
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                import logging

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
            raise

    def evaluate_transcriptions(self):
        """
        Evaluates the transcripts if there are reference transcripts

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.log_info("Evaluating transcripts...")
        self._load_transcripts()
        ser, wer, cer = self.compute_wer()
        self.log_info(f"SER: {100 * ser:.2f}%, WER: {100 * wer:.2f}%, CER: {100 * cer:.2f}%")
        return ser, wer

    def _load_transcripts(self):
        """Load transcripts from Kaldi temporary files"""
        with self.session() as session:
            records = []
            for score_args in self.score_arguments():
                for dict_id, tra_path in score_args.tra_paths.items():
                    lookup = self.reversed_word_mapping(dict_id)
                    with open(tra_path, "r", encoding="utf8") as f:
                        for line in f:
                            t = line.strip().split(" ")
                            utt = int(t[0].split("-")[-1])
                            ints = t[1:]
                            if not ints:
                                continue
                            records.append(
                                {
                                    "id": utt,
                                    "transcription_text": " ".join(lookup[int(x)] for x in ints),
                                }
                            )
            session.bulk_update_mappings(Utterance, records)
            self.transcription_done = True
            session.query(Corpus).update({"transcription_done": True})
            session.commit()

    def collect_alignments(self) -> None:
        """
        Collect word and phone alignments from alignment archives
        """
        if self.alignment_done:
            if self.export_output_directory is not None:
                self.export_textgrids()
            return
        self._collect_alignments()

    def export_transcriptions(self):
        self._load_transcripts()
        with self.session() as session:
            files = session.query(File).options(
                selectinload(File.utterances),
                selectinload(File.speakers).selectinload(SpeakerOrdering.speaker),
                joinedload(File.sound_file, innerjoin=True).load_only(SoundFile.duration),
            )
            for file in files:
                utterance_count = len(file.utterances)
                duration = file.sound_file.duration

                if utterance_count == 0:
                    self.log_debug(f"Could not find any utterances for {file.name}")
                elif (
                    utterance_count == 1
                    and file.utterances[0].begin == 0
                    and file.utterances[0].end == duration
                ):
                    output_format = "lab"
                else:
                    output_format = TextgridFormats.SHORT_TEXTGRID
                output_path = construct_output_path(
                    file.name,
                    file.relative_path,
                    self.export_output_directory,
                    output_format=output_format,
                )
                data = file.construct_transcription_tiers()
                if output_format == "lab":
                    for intervals in data.values():
                        with open(output_path, "w", encoding="utf8") as f:
                            f.write(intervals[0].label)
                else:

                    tg = textgrid.Textgrid()
                    tg.maxTimestamp = duration
                    for speaker in file.speakers:
                        speaker = speaker.speaker.name
                        intervals = data[speaker]
                        tier = textgrid.IntervalTier(
                            speaker, [x.to_tg_interval() for x in intervals], minT=0, maxT=duration
                        )

                        tg.addTier(tier)
                    tg.save(output_path, includeBlankSpaces=True, format=output_format)
        if self.evaluation_mode:
            self.save_transcription_evaluation(self.export_output_directory)

    def export_files(
        self,
        output_directory: str,
        output_format: Optional[str] = None,
        include_original_text: bool = False,
    ) -> None:
        """
        Export transcriptions

        Parameters
        ----------
        output_directory: str
            Directory to save transcriptions
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        self.export_output_directory = output_directory
        os.makedirs(self.export_output_directory, exist_ok=True)
        if self.output_type == "transcription":
            self.export_transcriptions()
        else:
            self.export_textgrids(output_format, include_original_text)
