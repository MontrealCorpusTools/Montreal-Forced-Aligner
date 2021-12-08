"""
Transcription
=============

"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import yaml

from ..abc import FileExporterMixin, TopLevelMfaWorker
from ..corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from ..exceptions import KaldiProcessingError, PlatformError
from ..helper import parse_old_features, score
from ..models import AcousticModel, LanguageModel
from ..utils import log_kaldi_errors, run_mp, run_non_mp, thirdparty_binary
from .multiprocessing import (
    CarpaLmRescoreArguments,
    CreateHclgArguments,
    DecodeArguments,
    FinalFmllrArguments,
    FmllrRescoreArguments,
    InitialFmllrArguments,
    LatGenFmllrArguments,
    LmRescoreArguments,
    ScoreArguments,
    carpa_lm_rescore_func,
    create_hclg_func,
    decode_func,
    final_fmllr_est_func,
    fmllr_rescore_func,
    initial_fmllr_func,
    lat_gen_fmllr_func,
    lm_rescore_func,
    score_func,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from ..abc import MetaDict

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
        first_beam: int = 0,
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

    @abstractmethod
    def create_decoding_graph(self) -> None:
        """Create decoding graph for use in transcription"""
        ...

    @abstractmethod
    def transcribe(self) -> None:
        """Perform transcription"""
        ...

    @property
    @abstractmethod
    def model_path(self) -> str:
        """Acoustic model file path"""
        ...

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


class Transcriber(
    AcousticCorpusPronunciationMixin, TranscriberMixin, FileExporterMixin, TopLevelMfaWorker
):
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
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                data = parse_old_features(data)
                for k, v in data.items():
                    if k == "features":
                        global_params.update(v)
                    else:
                        global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    def setup(self) -> None:
        """Set up transcription"""
        if self.initialized:
            return
        begin = time.time()
        os.makedirs(self.working_log_directory, exist_ok=True)
        check = self.check_previous_run()
        if check:
            self.logger.debug(
                "There were some differences in the current run compared to the last one. "
                "This may cause issues, run with --clean, if you hit an error."
            )
        self.load_corpus()
        self.acoustic_model.validate(self)
        self.acoustic_model.export_model(self.model_directory)
        self.acoustic_model.export_model(self.working_directory)
        self.acoustic_model.log_details(self.logger)
        self.create_decoding_graph()
        self.initialized = True
        self.logger.debug(f"Setup for transcription in {time.time() - begin} seconds")

    def create_hclgs_arguments(self) -> Dict[str, CreateHclgArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.create_hclg_func`

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgArguments`]
            Per dictionary arguments for HCLG
        """
        args = {}
        for dict_name, dictionary in self.dictionary_mapping.items():
            args[dict_name] = CreateHclgArguments(
                os.path.join(self.model_directory, "log", f"hclg.{dict_name}.log"),
                self.model_directory,
                os.path.join(self.model_directory, "{file_name}" + f".{dict_name}.fst"),
                os.path.join(self.model_directory, f"words.{dict_name}.txt"),
                os.path.join(self.model_directory, f"G.{dict_name}.carpa"),
                self.language_model.small_arpa_path,
                self.language_model.medium_arpa_path,
                self.language_model.carpa_path,
                self.model_path,
                dictionary.lexicon_disambig_fst_path,
                os.path.join(dictionary.phones_dir, "disambiguation_symbols.int"),
                self.hclg_options,
                dictionary.words_mapping,
            )
        return args

    def decode_arguments(self) -> List[DecodeArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.decode_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`]
            Arguments for processing
        """
        feat_string = self.construct_feature_proc_strings()
        return [
            DecodeArguments(
                os.path.join(self.working_log_directory, f"decode.{j.name}.log"),
                j.current_dictionary_names,
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
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.score_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreArguments`]
            Arguments for processing
        """
        return [
            ScoreArguments(
                os.path.join(self.working_log_directory, f"score.{j.name}.log"),
                j.current_dictionary_names,
                self.score_options,
                j.construct_path_dictionary(self.working_directory, "lat", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.rescored", "ark"),
                j.construct_path_dictionary(self.working_directory, "lat.carpa.rescored", "ark"),
                j.construct_dictionary_dependent_paths(self.model_directory, "words", "txt"),
                j.construct_path_dictionary(self.evaluation_directory, "tra", "scp"),
            )
            for j in self.jobs
        ]

    def lm_rescore_arguments(self) -> List[LmRescoreArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.lm_rescore_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreArguments`]
            Arguments for processing
        """
        return [
            LmRescoreArguments(
                os.path.join(self.working_log_directory, f"lm_rescore.{j.name}.log"),
                j.current_dictionary_names,
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
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.carpa_lm_rescore_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreArguments`]
            Arguments for processing
        """
        return [
            CarpaLmRescoreArguments(
                os.path.join(self.working_log_directory, f"carpa_lm_rescore.{j.name}.log"),
                j.current_dictionary_names,
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
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.initial_fmllr_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            InitialFmllrArguments(
                os.path.join(self.working_log_directory, f"initial_fmllr.{j.name}.log"),
                j.current_dictionary_names,
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
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.lat_gen_fmllr_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            LatGenFmllrArguments(
                os.path.join(self.working_log_directory, f"lat_gen_fmllr.{j.name}.log"),
                j.current_dictionary_names,
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
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.final_fmllr_est_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            FinalFmllrArguments(
                os.path.join(self.working_log_directory, f"final_fmllr.{j.name}.log"),
                j.current_dictionary_names,
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
        Generate Job arguments for :func:`~montreal_forced_aligner.transcription.multiprocessing.fmllr_rescore_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            FmllrRescoreArguments(
                os.path.join(self.working_log_directory, f"fmllr_rescore.{j.name}.log"),
                j.current_dictionary_names,
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
            text=True,
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
        if self.use_mp:
            run_mp(create_hclg_func, dict_arguments, self.working_log_directory)
        else:
            run_non_mp(create_hclg_func, dict_arguments, self.working_log_directory)
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
        dirty_path = os.path.join(self.model_directory, "dirty")

        if os.path.exists(dirty_path):  # if there was an error, let's redo from scratch
            shutil.rmtree(self.model_directory)
        log_dir = os.path.join(self.model_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        self.write_lexicon_information(write_disambiguation=True)
        for dict_name, dictionary in self.dictionary_mapping.items():
            words_path = os.path.join(self.model_directory, f"words.{dict_name}.txt")
            shutil.copyfile(dictionary.words_symbol_path, words_path)

        big_arpa_path = self.language_model.carpa_path
        small_arpa_path = self.language_model.small_arpa_path
        medium_arpa_path = self.language_model.medium_arpa_path
        if not os.path.exists(small_arpa_path) or not os.path.exists(medium_arpa_path):
            self.logger.warning(
                "Creating small and medium language models from scratch, this may take some time. "
                "Running `mfa train_lm` on the ARPA file will remove this warning."
            )
            if sys.platform == "win32":
                raise PlatformError("ngram")
            self.logger.info("Parsing large ngram model...")
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
                self.logger.info(
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
                self.logger.info(
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
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def score_transcriptions(self):
        """
        Score transcriptions if reference text is available in the corpus

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.score_func`
            Multiprocessing helper function for each job
        :meth:`.Transcriber.score_arguments`
            Job method for generating arguments for this function

        """
        if self.evaluation_mode:
            best_wer = 10000
            best = None
            for lmwt in range(self.min_language_model_weight, self.max_language_model_weight):
                for wip in self.word_insertion_penalties:
                    self.language_model_weight = lmwt
                    self.word_insertion_penalty = wip
                    os.makedirs(self.evaluation_log_directory, exist_ok=True)

                    jobs = self.score_arguments()
                    if self.use_mp:
                        run_mp(score_func, jobs, self.evaluation_log_directory)
                    else:
                        run_non_mp(score_func, jobs, self.evaluation_log_directory)
                    ser, wer = self.evaluate()
                    if wer < best_wer:
                        best = (lmwt, wip)
            self.language_model_weight = best[0]
            self.word_insertion_penalty = best[1]
            for score_args in self.score_arguments():
                for p in score_args.tra_paths.values():
                    shutil.copyfile(
                        p,
                        p.replace(self.evaluation_directory, self.working_directory),
                    )
        else:
            jobs = self.score_arguments()
            if self.use_mp:
                run_mp(score_func, jobs, self.working_log_directory)
            else:
                run_non_mp(score_func, jobs, self.working_log_directory)

    def transcribe_fmllr(self) -> None:
        """
        Run fMLLR estimation over initial decoding lattices and rescore

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.initial_fmllr_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.lat_gen_fmllr_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.final_fmllr_est_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.fmllr_rescore_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.lm_rescore_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.carpa_lm_rescore_func`
            Multiprocessing helper function for each job

        """
        jobs = self.initial_fmllr_arguments()

        if self.use_mp:
            run_mp(initial_fmllr_func, jobs, self.working_log_directory)
        else:
            run_non_mp(initial_fmllr_func, jobs, self.working_log_directory)

        self.speaker_independent = False

        jobs = self.lat_gen_fmllr_arguments()

        if self.use_mp:
            run_mp(lat_gen_fmllr_func, jobs, self.working_log_directory)
        else:
            run_non_mp(lat_gen_fmllr_func, jobs, self.working_log_directory)

        jobs = self.final_fmllr_arguments()

        if self.use_mp:
            run_mp(final_fmllr_est_func, jobs, self.working_log_directory)
        else:
            run_non_mp(final_fmllr_est_func, jobs, self.working_log_directory)

        jobs = self.fmllr_rescore_arguments()

        if self.use_mp:
            run_mp(fmllr_rescore_func, jobs, self.working_log_directory)
        else:
            run_non_mp(fmllr_rescore_func, jobs, self.working_log_directory)

        jobs = self.lm_rescore_arguments()

        if self.use_mp:
            run_mp(lm_rescore_func, jobs, self.working_log_directory)
        else:
            run_non_mp(lm_rescore_func, jobs, self.working_log_directory)

        jobs = self.carpa_lm_rescore_arguments()

        if self.use_mp:
            run_mp(carpa_lm_rescore_func, jobs, self.working_log_directory)
        else:
            run_non_mp(carpa_lm_rescore_func, jobs, self.working_log_directory)

    def transcribe(self) -> None:
        """
        Transcribe the corpus

        See Also
        --------
        :func:`~montreal_forced_aligner.transcription.multiprocessing.decode_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.lm_rescore_func`
            Multiprocessing helper function for each job
        :func:`~montreal_forced_aligner.transcription.multiprocessing.carpa_lm_rescore_func`
            Multiprocessing helper function for each job

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.logger.info("Beginning transcription...")
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(dirty_path):
            shutil.rmtree(self.working_directory, ignore_errors=True)
        os.makedirs(self.working_log_directory, exist_ok=True)
        try:
            self.speaker_independent = True
            jobs = self.decode_arguments()

            if self.use_mp:
                run_mp(decode_func, jobs, self.working_log_directory)
            else:
                run_non_mp(decode_func, jobs, self.working_log_directory)

            jobs = self.lm_rescore_arguments()

            if self.use_mp:
                run_mp(lm_rescore_func, jobs, self.working_log_directory)
            else:
                run_non_mp(lm_rescore_func, jobs, self.working_log_directory)

            jobs = self.carpa_lm_rescore_arguments()

            if self.use_mp:
                run_mp(carpa_lm_rescore_func, jobs, self.working_log_directory)
            else:
                run_non_mp(carpa_lm_rescore_func, jobs, self.working_log_directory)
            if self.uses_speaker_adaptation:
                self.logger.info("Performing speaker adjusted transcription...")
                self.transcribe_fmllr()
            self.score_transcriptions()
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def evaluate(self):
        """
        Evaluates the transcripts if there are reference transcripts

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.logger.info("Evaluating transcripts...")
        self._load_transcripts()
        # Sentence-level measures

        correct = 0
        incorrect = 0
        # Word-level measures
        total_edits = 0
        total_length = 0
        issues = []
        with mp.Pool(self.num_jobs) as pool:
            to_comp = []
            for utt_name, utterance in self.utterances.items():
                g = utterance.text.split()
                if not utterance.transcription_text:
                    incorrect += 1
                    total_edits += len(g)
                    total_length += len(g)

                h = utterance.transcription_text.split()
                if g != h:
                    issues.append((utt_name, g, h))
                to_comp.append((g, h))
            gen = pool.starmap(score, to_comp)
            for (edits, length) in gen:
                if edits == 0:
                    correct += 1
                else:
                    incorrect += 1
                total_edits += edits
                total_length += length
        ser = 100 * incorrect / (correct + incorrect)
        wer = 100 * total_edits / total_length
        output_path = os.path.join(self.evaluation_directory, "transcription_issues.csv")
        with open(output_path, "w", encoding="utf8") as f:
            for utt, g, h in issues:
                g = " ".join(g)
                h = " ".join(h)
                f.write(f"{utt},{g},{h}\n")
        self.logger.info(f"SER: {ser:.2f}%, WER: {wer:.2f}%")
        return ser, wer

    def _load_transcripts(self):
        """Load transcripts from Kaldi temporary files"""
        for score_args in self.score_arguments():
            for tra_path in score_args.tra_paths.values():

                with open(tra_path, "r", encoding="utf8") as f:
                    for line in f:
                        t = line.strip().split(" ")
                        utt = t[0]
                        utterance = self.utterances[utt]
                        speaker = utterance.speaker
                        lookup = speaker.dictionary.reversed_word_mapping
                        ints = t[1:]
                        if not ints:
                            continue
                        transcription = []
                        for i in ints:
                            transcription.append(lookup[int(i)])
                        utterance.transcription_text = " ".join(transcription)

    def export_files(self, output_directory: str) -> None:
        """
        Export transcriptions

        Parameters
        ----------
        output_directory: str
            Directory to save transcriptions
        """
        backup_output_directory = None
        if not self.overwrite:
            backup_output_directory = os.path.join(self.working_directory, "transcriptions")
            os.makedirs(backup_output_directory, exist_ok=True)
        self._load_transcripts()
        for file in self.files:
            file.save(output_directory, backup_output_directory)
