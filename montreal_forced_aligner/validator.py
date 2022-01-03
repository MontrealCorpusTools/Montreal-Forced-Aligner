"""
Validating corpora
==================

"""
from __future__ import annotations

import os
import subprocess
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import yaml

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import CorpusAligner, PretrainedAligner
from montreal_forced_aligner.alignment.multiprocessing import compile_information_func
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import (
    TerminalPrinter,
    comma_join,
    edit_distance,
    load_scp,
    save_scp,
)
from montreal_forced_aligner.utils import log_kaldi_errors, run_mp, run_non_mp, thirdparty_binary

if TYPE_CHECKING:
    from argparse import Namespace

    from .abc import MetaDict


__all__ = ["TrainingValidator", "PretrainedValidator"]


class CompileUtteranceTrainGraphsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.validator.compile_utterance_train_graphs_func`"""

    log_path: str
    dictionaries: List[str]
    disambig_int_paths: Dict[str, str]
    disambig_L_fst_paths: Dict[str, str]
    fst_paths: Dict[str, str]
    graphs_paths: Dict[str, str]
    model_path: str
    tree_path: str


class TestUtterancesArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.validator.test_utterances_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    words_paths: Dict[str, str]
    graphs_paths: Dict[str, str]
    text_int_paths: Dict[str, str]
    edits_paths: Dict[str, str]
    out_int_paths: Dict[str, str]
    model_path: str


def test_utterances_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    words_paths: Dict[str, str],
    graphs_paths: Dict[str, str],
    text_int_paths: Dict[str, str],
    edits_paths: Dict[str, str],
    out_int_paths: Dict[str, str],
    model_path: str,
):
    """
    Multiprocessing function to test utterance transcriptions

    See Also
    --------
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary
    :kaldi_src:`lattice-oracle`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Log path
    dictionaries: list[str]
        List of dictionaries
    feature_strings: dict[str, str]
        Dictionary of feature strings per dictionary name
    words_paths: dict[str, str]
        Dictionary of word mapping files per dictionary name
    graphs_paths: dict[str, str]
        Dictionary of utterance FST graph archives per dictionary name
    text_int_paths: dict[str, str]
        Dictionary of text.int files per dictionary name
    edits_paths: dict[str, str]
        Dictionary of paths to save transcription differences per dictionary name
    out_int_paths: dict[str, str]
        Dictionary of output .int files per dictionary name
    model_path: str
        Acoustic model path
    """
    acoustic_scale = 0.1
    beam = 15.0
    lattice_beam = 8.0
    max_active = 750
    with open(log_path, "w") as log_file:
        for dict_name in dictionaries:
            words_path = words_paths[dict_name]
            graphs_path = graphs_paths[dict_name]
            feature_string = feature_strings[dict_name]
            edits_path = edits_paths[dict_name]
            text_int_path = text_int_paths[dict_name]
            out_int_path = out_int_paths[dict_name]
            latgen_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-latgen-faster"),
                    f"--acoustic-scale={acoustic_scale}",
                    f"--beam={beam}",
                    f"--max-active={max_active}",
                    f"--lattice-beam={lattice_beam}",
                    f"--word-symbol-table={words_path}",
                    model_path,
                    "ark:" + graphs_path,
                    feature_string,
                    "ark:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
            )

            oracle_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-oracle"),
                    "ark:-",
                    f"ark,t:{text_int_path}",
                    f"ark,t:{out_int_path}",
                    f"ark,t:{edits_path}",
                ],
                stderr=log_file,
                stdin=latgen_proc.stdout,
            )
            oracle_proc.communicate()


def compile_utterance_train_graphs_func(
    log_path: str,
    dictionaries: List[str],
    disambig_int_paths: Dict[str, str],
    disambig_L_fst_paths: Dict[str, str],
    fst_paths: Dict[str, str],
    graphs_paths: Dict[str, str],
    model_path: str,
    tree_path: str,
):
    """
    Multiprocessing function to compile utterance FSTs

    See Also
    --------
    :kaldi_src:`compile-train-graphs-fsts`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Log path
    dictionaries: list[str]
        List of dictionaries
    disambig_int_paths: dict[str, str]
        Dictionary of disambiguation symbol int files per dictionary name
    disambig_L_fst_paths: dict[str, str]
        Dictionary of disambiguation lexicon FSTs per dictionary name
    fst_paths: dict[str, str]
        Dictionary of pregenerated utterance FST scp files per dictionary name
    graphs_paths: dict[str, str]
        Dictionary of utterance FST graph archives per dictionary name
    model_path: str
        Acoustic model path
    tree_path: str
        Acoustic model's tree path
    """
    with open(log_path, "w") as log_file:
        for dict_name in dictionaries:
            disambig_int_path = disambig_int_paths[dict_name]
            disambig_L_fst_path = disambig_L_fst_paths[dict_name]
            fst_path = fst_paths[dict_name]
            graphs_path = graphs_paths[dict_name]
            proc = subprocess.Popen(
                [
                    thirdparty_binary("compile-train-graphs-fsts"),
                    "--transition-scale=1.0",
                    "--self-loop-scale=0.1",
                    f"--read-disambig-syms={disambig_int_path}",
                    tree_path,
                    model_path,
                    disambig_L_fst_path,
                    f"ark:{fst_path}",
                    f"ark:{graphs_path}",
                ],
                stderr=log_file,
            )

            proc.communicate()


class ValidationMixin(CorpusAligner):
    """
    Mixin class for performing validation on a corpus

    Parameters
    ----------
    ignore_acoustics: bool
        Flag for whether feature generation and training/alignment should be skipped
    test_transcriptions: bool
        Flag for whether utterance transcriptions should be tested with a unigram language model

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.base.CorpusAligner`
        For corpus, dictionary, and alignment parameters

    Attributes
    ----------
    printer: TerminalPrinter
        Printer for output messages
    """

    def __init__(
        self, ignore_acoustics: bool = False, test_transcriptions: bool = False, **kwargs
    ):
        kwargs["clean"] = True
        super().__init__(**kwargs)
        self.ignore_acoustics = ignore_acoustics
        self.test_transcriptions = test_transcriptions
        self.printer = TerminalPrinter()

    def utt2fst_scp_data(
        self, num_frequent_words: int = 10
    ) -> List[Dict[str, List[Tuple[str, str]]]]:
        """
        Generate Kaldi style utt2fst scp data

        Parameters
        ----------
        num_frequent_words: int
            Number of frequent words to include in the unigram language model

        Returns
        -------
        dict[str, list[tuple[str, str]]]
            Utterance FSTs per dictionary name
        """
        job_data = []
        most_frequent = {}
        for j in self.jobs:
            data = {x: [] for x in j.current_dictionary_names}
            for utterance in j.current_utterances:
                new_text = []
                dictionary = utterance.speaker.dictionary
                dict_name = utterance.speaker.dictionary_name
                if dict_name not in most_frequent:
                    word_frequencies = self.get_word_frequency()
                    most_frequent[dict_name] = sorted(
                        word_frequencies.items(), key=lambda x: -x[1]
                    )[:num_frequent_words]

                for t in utterance.text:
                    lookup = utterance.speaker.dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                    new_text.extend(x for x in lookup if x != "")
                data[dict_name].append(
                    (
                        utterance.name,
                        dictionary.create_utterance_fst(new_text, most_frequent[dictionary.name]),
                    )
                )
            job_data.append(data)
        return job_data

    def output_utt_fsts(self, num_frequent_words: int = 10) -> None:
        """
        Write utterance FSTs

        Parameters
        ----------
        num_frequent_words: int
            Number of frequent words
        """
        utt2fst = self.utt2fst_scp_data(num_frequent_words)
        for i, job_data in enumerate(utt2fst):
            for dict_name, scp in job_data.items():
                utt2fst_scp_path = os.path.join(
                    self.split_directory, f"utt2fst.{dict_name}.{i}.scp"
                )
                save_scp(scp, utt2fst_scp_path, multiline=True)

    def compile_utterance_train_graphs_arguments(
        self,
    ) -> List[CompileUtteranceTrainGraphsArguments]:
        """
        Generate Job arguments for :func:`compile_utterance_train_graphs_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.validator.CompileUtteranceTrainGraphsArguments`]
            Arguments for processing
        """
        disambig_paths = {
            k: self.disambiguation_symbols_int_path for k, v in self.dictionary_mapping.items()
        }
        lexicon_fst_paths = {
            k: v.lexicon_disambig_fst_path for k, v in self.dictionary_mapping.items()
        }
        return [
            CompileUtteranceTrainGraphsArguments(
                os.path.join(self.working_log_directory, f"utterance_fst.{j.name}.log"),
                j.current_dictionary_names,
                disambig_paths,
                lexicon_fst_paths,
                j.construct_path_dictionary(self.data_directory, "utt2fst", "scp"),
                j.construct_path_dictionary(self.working_directory, "utterance_graphs", "fst"),
                self.model_path,
                self.tree_path,
            )
            for j in self.jobs
        ]

    def test_utterances_arguments(self) -> List[TestUtterancesArguments]:
        """
        Generate Job arguments for :func:`test_utterances_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.validator.TestUtterancesArguments`]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        words_paths = {k: v.words_symbol_path for k, v in self.dictionary_mapping.items()}
        return [
            TestUtterancesArguments(
                os.path.join(self.working_directory, f"utterance_fst.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                words_paths,
                j.construct_path_dictionary(self.working_directory, "utterance_graphs", "fst"),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.construct_path_dictionary(self.working_directory, "edits", "scp"),
                j.construct_path_dictionary(self.working_directory, "aligned", "int"),
                self.model_path,
            )
            for j in self.jobs
        ]

    @property
    def working_log_directory(self) -> str:
        """Working log directory"""
        return os.path.join(self.working_directory, "log")

    def setup(self):
        """
        Set up the corpus and validator

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        try:
            self.load_corpus()
            self.write_lexicon_information()
            if self.test_transcriptions:
                self.write_lexicon_information(write_disambiguation=True)
            if self.ignore_acoustics:
                self.logger.info("Skipping acoustic feature generation")
            else:
                self.generate_features()
            self.calculate_oovs_found()

            if not self.ignore_acoustics and self.test_transcriptions:
                self.initialize_utt_fsts()
            else:
                self.logger.info("Skipping transcription testing")
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def analyze_setup(self) -> None:
        """
        Analyzes the set up process and outputs info to the console
        """
        begin = time.time()
        total_duration = sum(x.duration for x in self.files)
        total_duration = Decimal(str(total_duration)).quantize(Decimal("0.001"))
        self.log_debug(f"Duration calculation took {time.time() - begin}")

        begin = time.time()
        ignored_count = len(self.no_transcription_files)
        ignored_count += len(self.textgrid_read_errors)
        ignored_count += len(self.decode_error_files)
        self.log_debug(f"Ignored count calculation took {time.time() - begin}")

        self.printer.print_header("Corpus")

        self.printer.print_green_stat(self.files.sound_file_count, "sound files")
        self.printer.print_green_stat(self.files.lab_count, "lab files")
        self.printer.print_green_stat(self.files.textgrid_count, "textgrid files")
        if len(self.no_transcription_files):
            self.printer.print_yellow_stat(
                len(self.no_transcription_files),
                "sound files without corresponding transcriptions",
            )
        if len(self.decode_error_files):
            self.printer.print_red_stat(len(self.decode_error_files), "read errors for lab files")
        if len(self.textgrid_read_errors):
            self.printer.print_red_stat(
                len(self.textgrid_read_errors), "read errors for TextGrid files"
            )

        self.printer.print_green_stat(len(self.speakers), "speakers")
        self.printer.print_green_stat(self.num_utterances, "utterances")
        self.printer.print_green_stat(total_duration, "seconds total duration")
        print()
        self.analyze_wav_errors()
        self.analyze_missing_features()
        self.analyze_files_with_no_transcription()
        self.analyze_transcriptions_with_no_wavs()

        if len(self.decode_error_files) or self.files.lab_count:
            self.analyze_unreadable_text_files()
        if len(self.textgrid_read_errors) or self.files.textgrid_count:
            self.analyze_textgrid_read_errors()

        self.printer.print_header("Dictionary")
        self.analyze_oovs()

    def analyze_oovs(self) -> None:
        """
        Analyzes OOVs in the corpus and constructs message
        """
        self.printer.print_sub_header("Out of vocabulary words")
        output_dir = self.output_directory
        oov_types = self.oovs_found
        calculate_frequency = not oov_types
        oov_path = os.path.join(output_dir, "oovs_found.txt")
        utterance_oov_path = os.path.join(output_dir, "utterance_oovs.txt")

        total_instances = 0
        with open(utterance_oov_path, "w", encoding="utf8") as f:
            for utterance in self.utterances:
                if not utterance.oovs:
                    continue
                total_instances += len(utterance.oovs)
                f.write(f"{utterance.name} {', '.join(utterance.oovs)}\n")
                if calculate_frequency:
                    self.oovs_found.update(utterance.oovs)
        if self.oovs_found:
            self.save_oovs_found(output_dir)
            self.printer.print_yellow_stat(len(oov_types), "OOV word types")
            self.printer.print_yellow_stat(total_instances, "total OOV tokens")
            lines = [
                "",
                "For a full list of the word types, please see:",
                "",
                self.printer.indent_string + self.printer.colorize(oov_path, "bright"),
                "",
                "For a by-utterance breakdown of missing words, see:",
                "",
                self.printer.indent_string + self.printer.colorize(utterance_oov_path, "bright"),
                "",
            ]
            self.printer.print_info_lines(lines)
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'yellow')} missing words from the dictionary. If you plan on using the a model trained "
                "on this dataset to align other datasets in the future, it is recommended that there be at "
                "least some missing words."
            )
        self.printer.print_end_section()

    def analyze_wav_errors(self) -> None:
        """
        Analyzes any sound file issues in the corpus and constructs message
        """
        self.printer.print_sub_header("Sound file read errors")

        output_dir = self.output_directory
        wav_read_errors = self.sound_file_errors
        if wav_read_errors:
            path = os.path.join(output_dir, "sound_file_errors.csv")
            with open(path, "w") as f:
                for p in wav_read_errors:
                    f.write(f"{p}\n")

            self.printer.print_info_lines(
                f"There were {self.printer.colorize(len(wav_read_errors), 'red')} issues reading sound files. "
                f"Please see {self.printer.colorize(path, 'bright')} for a list."
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} issues reading sound files."
            )

        self.printer.print_end_section()

    def analyze_missing_features(self) -> None:
        """
        Analyzes issues in feature generation in the corpus and constructs message
        """
        self.printer.print_sub_header("Feature generation")
        if self.ignore_acoustics:
            self.printer.print_info_lines("Acoustic feature generation was skipped.")
            self.printer.print_end_section()
            return
        output_dir = self.output_directory
        missing_features = [x for x in self.utterances if x.ignored]
        if missing_features:
            path = os.path.join(output_dir, "missing_features.csv")
            with open(path, "w") as f:
                for utt in missing_features:
                    if utt.begin is not None:

                        f.write(f"{utt.file.wav_path},{utt.begin},{utt.end}\n")
                    else:
                        f.write(f"{utt.file.wav_path}\n")

            self.printer.print_info_lines(
                f"There were {self.printer.colorize(len(missing_features), 'red')} utterances missing features. "
                f"Please see {self.printer.colorize(path, 'bright')} for a list."
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} utterances missing features."
            )
        self.printer.print_end_section()

    def analyze_files_with_no_transcription(self) -> None:
        """
        Analyzes issues with sound files that have no transcription files
        in the corpus and constructs message
        """
        self.printer.print_sub_header("Files without transcriptions")
        output_dir = self.output_directory
        if self.no_transcription_files:
            path = os.path.join(output_dir, "missing_transcriptions.csv")
            with open(path, "w") as f:
                for file_path in self.no_transcription_files:
                    f.write(f"{file_path}\n")
            self.printer.print_info_lines(
                f"There were {self.printer.colorize(len(self.no_transcription_files), 'red')} sound files missing transcriptions. "
                f"Please see {self.printer.colorize(path, 'bright')} for a list."
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} sound files missing transcriptions."
            )
        self.printer.print_end_section()

    def analyze_transcriptions_with_no_wavs(self) -> None:
        """
        Analyzes issues with transcription that have no sound files
        in the corpus and constructs message
        """
        self.printer.print_sub_header("Transcriptions without sound files")
        output_dir = self.output_directory
        if self.transcriptions_without_wavs:
            path = os.path.join(output_dir, "transcriptions_missing_sound_files.csv")
            with open(path, "w") as f:
                for file_path in self.transcriptions_without_wavs:
                    f.write(f"{file_path}\n")
            self.printer.print_info_lines(
                f"There were {self.printer.colorize(len(self.transcriptions_without_wavs), 'red')} transcription files missing sound files. "
                f"Please see {self.printer.colorize(path, 'bright')} for a list."
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} transcription files missing sound files."
            )
        self.printer.print_end_section()

    def analyze_textgrid_read_errors(self) -> None:
        """
        Analyzes issues with reading TextGrid files
        in the corpus and constructs message
        """
        self.printer.print_sub_header("TextGrid read errors")
        output_dir = self.output_directory
        if self.textgrid_read_errors:
            path = os.path.join(output_dir, "textgrid_read_errors.txt")
            with open(path, "w") as f:
                for k, v in self.textgrid_read_errors.items():
                    f.write(
                        f"The TextGrid file {k} gave the following error on load:\n\n{v}\n\n\n"
                    )
            self.printer.print_info_lines(
                [
                    f"There were {self.printer.colorize(len(self.textgrid_read_errors), 'red')} TextGrid files that could not be loaded. "
                    "For details, please see:",
                    "",
                    self.printer.indent_string + self.printer.colorize(path, "bright"),
                ]
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} issues reading TextGrids."
            )

        self.printer.print_end_section()

    def analyze_unreadable_text_files(self) -> None:
        """
        Analyzes issues with reading text files
        in the corpus and constructs message
        """
        self.printer.print_sub_header("Text file read errors")
        output_dir = self.output_directory
        if self.decode_error_files:
            path = os.path.join(output_dir, "utf8_read_errors.csv")
            with open(path, "w") as f:
                for file_path in self.decode_error_files:
                    f.write(f"{file_path}\n")
            self.printer.print_info_lines(
                f"There were {self.printer.colorize(len(self.decode_error_files), 'red')} text files that could not be read. "
                f"Please see {self.printer.colorize(path, 'bright')} for a list."
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} issues reading text files."
            )

        self.printer.print_end_section()

    def compile_information(self) -> None:
        """
        Compiles information about alignment, namely what the overall log-likelihood was
        and how many files were unaligned.

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.multiprocessing.compile_information_func`
            Multiprocessing helper function for each job
        :meth:`.AlignMixin.compile_information_arguments`
            Job method for generating arguments for the helper function
        """
        self.logger.debug("Analyzing alignment information")
        compile_info_begin = time.time()

        jobs = self.compile_information_arguments()

        if self.use_mp:
            alignment_info = run_mp(
                compile_information_func, jobs, self.working_log_directory, True
            )
        else:
            alignment_info = run_non_mp(
                compile_information_func, jobs, self.working_log_directory, True
            )

        avg_like_sum = 0
        avg_like_frames = 0
        average_logdet_sum = 0
        average_logdet_frames = 0
        beam_too_narrow_count = 0
        too_short_count = 0
        unaligned_utts = []
        for data in alignment_info.values():
            unaligned_utts.extend(data["unaligned"])
            beam_too_narrow_count += len(data["unaligned"])
            too_short_count += len(data["too_short"])
            avg_like_frames += data["total_frames"]
            avg_like_sum += data["log_like"] * data["total_frames"]
            if "logdet_frames" in data:
                average_logdet_frames += data["logdet_frames"]
                average_logdet_sum += data["logdet"] * data["logdet_frames"]

        if not avg_like_frames:
            self.logger.debug(
                "No utterances were aligned, this likely indicates serious problems with the aligner."
            )
            self.printer.print_red_stat(0, f"of {len(self.utterances)} utterances were aligned")
        else:
            if too_short_count:
                self.printer.print_red_stat(
                    too_short_count, "utterances were too short to be aligned"
                )
            else:
                self.printer.print_green_stat(0, "utterances were too short to be aligned")
            if beam_too_narrow_count:
                self.logger.debug(
                    f"There were {beam_too_narrow_count} utterances that could not be aligned with "
                    f"the current beam settings."
                )
                self.printer.print_yellow_stat(
                    beam_too_narrow_count, "utterances that need a larger beam to align"
                )
            else:
                self.printer.print_green_stat(0, "utterances that need a larger beam to align")

            num_utterances = self.num_utterances
            if unaligned_utts:
                path = os.path.join(self.output_directory, "unalignable_files.csv")
                with open(path, "w") as f:
                    f.write("File path,begin,end,duration,text length\n")
                    for utt in unaligned_utts:
                        utterance = self.utterances[utt]
                        utt_duration = utterance.duration
                        utt_length_words = utterance.text.count(" ") + 1
                        if utterance.begin is not None:
                            f.write(
                                f"{utterance.file.wav_path},{utterance.begin},{utterance.end},{utt_duration},{utt_length_words}\n"
                            )
                        else:
                            f.write(
                                f"{utterance.file.wav_path},,,{utt_duration},{utt_length_words}\n"
                            )
                self.printer.print_info_lines(
                    [
                        f"There were {self.printer.colorize(len(unaligned_utts), 'red')} unaligned utterances out of {self.printer.colorize(self.num_utterances, 'bright')} after initial training. "
                        f"For details, please see:",
                        "",
                        self.printer.indent_string + self.printer.colorize(path, "bright"),
                    ]
                )

            self.printer.print_green_stat(
                num_utterances - beam_too_narrow_count - too_short_count,
                "utterances were successfully aligned",
            )
            average_log_like = avg_like_sum / avg_like_frames
            if average_logdet_sum:
                average_log_like += average_logdet_sum / average_logdet_frames
            self.logger.debug(f"Average per frame likelihood for alignment: {average_log_like}")
        self.logger.debug(f"Compiling information took {time.time() - compile_info_begin}")

    def initialize_utt_fsts(self) -> None:
        """
        Construct utterance FSTs
        """
        self.logger.info("Initializing for testing transcriptions...")
        self.output_utt_fsts()

    def test_utterance_transcriptions(self) -> None:
        """
        Tests utterance transcriptions with simple unigram models based on the utterance text and frequent
        words in the corpus

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.logger.info("Checking utterance transcriptions...")

        try:

            jobs = self.compile_utterance_train_graphs_arguments()
            if self.use_mp:
                run_mp(compile_utterance_train_graphs_func, jobs, self.working_log_directory)
            else:
                run_non_mp(compile_utterance_train_graphs_func, jobs, self.working_log_directory)
            self.logger.info("Utterance FSTs compiled!")
            self.logger.info("Decoding utterances (this will take some time)...")
            jobs = self.test_utterances_arguments()
            if self.use_mp:
                run_mp(test_utterances_func, jobs, self.working_log_directory)
            else:
                run_non_mp(test_utterances_func, jobs, self.working_log_directory)
            self.logger.info("Finished decoding utterances!")

            errors = {}

            for job in jobs:
                for dict_name in job.dictionaries:
                    word_mapping = self.dictionary_mapping[dict_name].reversed_word_mapping
                    aligned_int = load_scp(job.out_int_paths[dict_name])
                    for utt, line in sorted(aligned_int.items()):
                        text = []
                        for t in line:
                            text.append(word_mapping[int(t)])
                        ref_text = self.utterances[utt].text.split()
                        edits = edit_distance(text, ref_text)

                        if edits:
                            errors[utt] = (edits, ref_text, text)
            if not errors:

                print(
                    f"There were {self.printer.colorize('no', 'green')} utterances with transcription issues."
                )
            else:
                out_path = os.path.join(self.output_directory, "transcription_problems.csv")
                with open(out_path, "w") as problemf:
                    problemf.write("Utterance,Edits,Reference,Decoded\n")
                    for utt, (edits, ref_text, text) in sorted(
                        errors.items(), key=lambda x: -1 * (len(x[1][1]) + len(x[1][2]))
                    ):
                        problemf.write(f"{utt},{edits},{' '.join(ref_text)},{' '.join(text)}\n")
                print(
                    f"There were {self.printer.colorize(len(errors), 'red')} of {self.printer.colorize(self.num_utterances, 'bright')} utterances with at least one transcription issue. "
                    f"Please see {self.printer.colorize(out_path, 'bright')} for a list."
                )

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise


class TrainingValidator(TrainableAligner, ValidationMixin):
    """
    Validator class for checking whether a corpus and a dictionary will work together
    for training

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.trainer.TrainableAligner`
        For training configuration
    :class:`~montreal_forced_aligner.validator.ValidationMixin`
        For validation parameters

    Attributes
    ----------
    training_configs: dict[str, :class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonophoneTrainer`]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_configs = {}
        self.add_config("monophone", {})

    @property
    def workflow_identifier(self) -> str:
        """Identifier for validation"""
        return "validate_training"

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:

        """
        Parse parameters for validation from a config path or command-line arguments

        Parameters
        ----------
        config_path: str
            Config path
        args: :class:`~argparse.Namespace`
            Command-line arguments from argparse
        unknown_args: list[str], optional
            Extra command-line arguments

        Returns
        -------
        dict[str, Any]
            Configuration parameters
        """
        global_params = {}
        training_params = []
        use_default = True
        if config_path:
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for k, v in data.items():
                    if k == "training":
                        for t in v:
                            for k2, v2 in t.items():
                                if "features" in v2:
                                    global_params.update(v2["features"])
                                    del v2["features"]
                                training_params.append((k2, v2))
                    elif k == "features":
                        if "type" in v:
                            v["feature_type"] = v["type"]
                            del v["type"]
                        global_params.update(v)
                    else:
                        if v is None and k in {
                            "punctuation",
                            "compound_markers",
                            "clitic_markers",
                        }:
                            v = []
                        global_params[k] = v
                if training_params:
                    use_default = False
        if use_default:  # default training configuration
            training_params.append(("monophone", {}))
        if training_params:
            if training_params[0][0] != "monophone":
                raise ConfigError("The first round of training must be monophone.")
        global_params["training_configuration"] = training_params
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    def setup(self):
        """
        Set up the corpus and validator

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        if self.initialized:
            return
        try:
            all_begin = time.time()
            self.dictionary_setup()
            self.log_debug(f"Loaded dictionary in {time.time() - all_begin}")

            begin = time.time()
            self._load_corpus()
            self.log_debug(f"Loaded corpus in {time.time() - begin}")

            begin = time.time()
            self.set_lexicon_word_set(self.corpus_word_set)
            self.log_debug(f"Set up lexicon word set in {time.time() - begin}")

            begin = time.time()
            for speaker in self.speakers:
                speaker.set_dictionary(self.get_dictionary(speaker.name))
            self.log_debug(f"Set dictionaries for speakers in {time.time() - begin}")

            self.calculate_oovs_found()

            begin = time.time()
            self.write_lexicon_information()
            self.log_debug(f"Wrote lexicon information in {time.time() - begin}")

            if self.ignore_acoustics:
                self.logger.info("Skipping acoustic feature generation")
            else:

                begin = time.time()
                self.initialize_jobs()
                self.log_debug(f"Initialized jobs in {time.time() - begin}")

                begin = time.time()
                self.write_corpus_information()
                self.log_debug(f"Wrote corpus information in {time.time() - begin}")

                begin = time.time()
                self.create_corpus_split()
                self.log_debug(f"Created corpus split directory in {time.time() - begin}")
                if self.test_transcriptions:
                    begin = time.time()
                    self.write_lexicon_information(write_disambiguation=True)
                    self.log_debug(f"Wrote lexicon information in {time.time() - begin}")
                begin = time.time()
                self.generate_features()
                self.log_debug(f"Generated features in {time.time() - begin}")
                if self.test_transcriptions:
                    begin = time.time()
                    self.initialize_utt_fsts()
                    self.log_debug(f"Initialized utterance FSTs in {time.time() - begin}")
                begin = time.time()
                self.calculate_oovs_found()
                self.log_debug(f"Calculated OOVs in {time.time() - begin}")

            self.initialized = True
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def calculate_oovs_found(self) -> None:
        """Sum the counts of oovs found in pronunciation dictionaries"""
        begin = time.time()
        self.logger.info("Calculating OOVs...")
        for u in self.utterances:
            self.oovs_found.update(u.oovs)
        self.save_oovs_found(self.output_directory)
        self.log_debug(f"Calculated OOVs in {time.time() - begin}")

    def validate(self):
        """
        Performs validation of the corpus
        """
        begin = time.time()
        self.log_debug(f"Setup took {time.time() - begin}")
        self.setup()
        self.analyze_setup()
        self.log_debug(f"Setup took {time.time() - begin}")
        if self.ignore_acoustics:
            self.printer.print_info_lines("Skipping test alignments.")
            return
        self.printer.print_header("Training")
        self.train(True)
        if self.test_transcriptions:
            self.printer.print_header("Test transcriptions")
            self.test_utterance_transcriptions()


class PretrainedValidator(PretrainedAligner, ValidationMixin):
    """
    Validator class for checking whether a corpus, a dictionary, and
    an acoustic model will work together for alignment

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.pretrained.PretrainedAligner`
        For alignment configuration
    :class:`~montreal_forced_aligner.validator.ValidationMixin`
        For validation parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def workflow_identifier(self) -> str:
        """Identifier for validation"""
        return "validate_pretrained"

    def setup(self):
        """
        Set up the corpus and validator

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        if self.initialized:
            return
        try:
            self.dictionary_setup()
            self._load_corpus()
            self.set_lexicon_word_set(self.corpus_word_set)

            for speaker in self.speakers:
                speaker.set_dictionary(self.get_dictionary(speaker.name))

            self.calculate_oovs_found()

            if self.ignore_acoustics:
                self.logger.info("Skipping acoustic feature generation")
            else:
                self.write_lexicon_information()
                self.initialize_jobs()
                self.write_corpus_information()
                self.create_corpus_split()
                if self.test_transcriptions:
                    self.write_lexicon_information(write_disambiguation=True)
                self.generate_features()
                if self.test_transcriptions:
                    self.initialize_utt_fsts()
                else:
                    self.logger.info("Skipping transcription testing")
            self.acoustic_model.validate(self)
            self.acoustic_model.export_model(self.working_directory)
            self.acoustic_model.log_details(self.logger)

            self.initialized = True
            self.logger.info("Finished initializing!")
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def align(self) -> None:
        """
        Validate alignment

        """
        done_path = os.path.join(self.working_directory, "done")
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.debug("Alignment already done, skipping.")
            return
        try:
            log_dir = os.path.join(self.working_directory, "log")
            os.makedirs(log_dir, exist_ok=True)
            self.compile_train_graphs()

            self.logger.debug("Performing first-pass alignment...")
            self.speaker_independent = True
            self.align_utterances()
            if self.uses_speaker_adaptation:
                self.logger.debug("Calculating fMLLR for speaker adaptation...")
                self.calc_fmllr()

                self.speaker_independent = False
                self.logger.debug("Performing second-pass alignment...")
                self.align_utterances()

        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        with open(done_path, "w"):
            pass

    def validate(self) -> None:
        """
        Performs validation of the corpus
        """
        self.setup()
        self.analyze_setup()
        self.analyze_missing_phones()
        if self.ignore_acoustics:
            self.log_info("Skipping test alignments.")
            return
        self.printer.print_header("Alignment")
        self.align()
        self.compile_information()
        if self.test_transcriptions:
            self.printer.print_header("Test transcriptions")
            self.test_utterance_transcriptions()

    def analyze_missing_phones(self) -> None:
        """Analyzes dictionary and acoustic model for phones in the dictionary that don't have acoustic models"""
        self.printer.print_sub_header("Acoustic model compatibility")
        if self.excluded_pronunciation_count:
            self.printer.print_yellow_stat(
                len(self.excluded_phones), "phones not in acoustic model"
            )
            self.printer.print_yellow_stat(
                self.excluded_pronunciation_count, "ignored pronunciations"
            )

            phone_string = [self.printer.colorize(x, "red") for x in sorted(self.excluded_phones)]
            self.printer.print_info_lines(
                [
                    "",
                    "Phones missing acoustic models:",
                    "",
                    self.printer.indent_string + comma_join(phone_string),
                ]
            )
        else:
            self.printer.print_info_lines(
                f"There were {self.printer.colorize('no', 'green')} phones in the dictionary without acoustic models."
            )
        self.printer.print_end_section()
