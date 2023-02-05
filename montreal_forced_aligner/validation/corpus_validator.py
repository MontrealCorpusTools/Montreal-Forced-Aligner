"""
Validating corpora
==================
"""
from __future__ import annotations

import logging
import os
import time
import typing
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Optional

import sqlalchemy
from sqlalchemy.orm import joinedload

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.alignment.multiprocessing import compile_information_func
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import Corpus, File, SoundFile, Speaker, TextFile, Utterance
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import (
    TerminalPrinter,
    comma_join,
    load_configuration,
    mfa_open,
)
from montreal_forced_aligner.utils import log_kaldi_errors, run_mp, run_non_mp

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict


__all__ = ["TrainingValidator", "PretrainedValidator"]

logger = logging.getLogger("mfa")


class ValidationMixin:
    """
    Mixin class for performing validation on a corpus

    Parameters
    ----------
    ignore_acoustics: bool
        Flag for whether feature generation and training/alignment should be skipped
    test_transcriptions: bool
        Flag for whether utterance transcriptions should be tested with a unigram language model
    phone_alignment: bool
        Flag for whether alignments should be compared to a phone-based system
    target_num_ngrams: int
        Target number of ngrams from speaker models to use

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
        self,
        ignore_acoustics: bool = False,
        test_transcriptions: bool = False,
        target_num_ngrams: int = 100,
        order: int = 3,
        method: str = "kneser_ney",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_acoustics = ignore_acoustics
        self.test_transcriptions = test_transcriptions
        self.target_num_ngrams = target_num_ngrams
        self.order = order
        self.method = method
        self.printer = TerminalPrinter(print_function=logger.info)

    @property
    def working_log_directory(self) -> str:
        """Working log directory"""
        return os.path.join(self.working_directory, "log")

    def analyze_setup(self) -> None:
        """
        Analyzes the setup process and outputs info to the console
        """
        begin = time.time()

        with self.session() as session:
            sound_file_count = session.query(SoundFile).count()
            text_file_count = session.query(TextFile).count()
            total_duration = session.query(sqlalchemy.func.sum(Utterance.duration)).scalar()

        total_duration = Decimal(str(total_duration)).quantize(Decimal("0.001"))
        logger.debug(f"Duration calculation took {time.time() - begin:.3f} seconds")

        begin = time.time()
        ignored_count = len(self.no_transcription_files)
        ignored_count += len(self.textgrid_read_errors)
        ignored_count += len(self.decode_error_files)
        logger.debug(f"Ignored count calculation took {time.time() - begin:.3f} seconds")

        self.printer.print_header("Corpus")
        self.printer.print_green_stat(sound_file_count, "sound files")
        self.printer.print_green_stat(text_file_count, "text files")
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

        self.printer.print_green_stat(self.num_speakers, "speakers")
        self.printer.print_green_stat(self.num_utterances, "utterances")
        self.printer.print_green_stat(total_duration, "seconds total duration")
        print()
        self.analyze_wav_errors()
        self.analyze_missing_features()
        self.analyze_files_with_no_transcription()
        self.analyze_transcriptions_with_no_wavs()

        if len(self.decode_error_files):
            self.analyze_unreadable_text_files()
        if len(self.textgrid_read_errors):
            self.analyze_textgrid_read_errors()

        self.printer.print_header("Dictionary")
        self.analyze_oovs()

    def analyze_oovs(self) -> None:
        """
        Analyzes OOVs in the corpus and constructs message
        """
        self.printer.print_sub_header("Out of vocabulary words")
        output_dir = self.output_directory
        oov_path = os.path.join(output_dir, "oovs_found.txt")
        utterance_oov_path = os.path.join(output_dir, "utterance_oovs.txt")

        total_instances = 0
        with mfa_open(utterance_oov_path, "w") as f, self.session() as session:
            utterances = (
                session.query(
                    File.name,
                    File.relative_path,
                    Speaker.name,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.oovs,
                )
                .join(Utterance.file)
                .join(Utterance.speaker)
                .filter(Utterance.oovs != None)  # noqa
                .filter(Utterance.oovs != "")
            )

            for file_name, relative_path, speaker_name, begin, end, oovs in utterances:
                total_instances += len(oovs)
                f.write(
                    f"{relative_path +'/' + file_name}, {speaker_name}: {begin}-{end}: {', '.join(oovs)}\n"
                )
                self.oovs_found.update(oovs)
        if self.oovs_found:
            self.save_oovs_found(self.output_directory)
            self.printer.print_yellow_stat(len(self.oovs_found), "OOV word types")
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
            with mfa_open(path, "w") as f:
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
        with self.session() as session:
            utterances = (
                session.query(File.name, File.relative_path, Utterance.begin, Utterance.end)
                .join(Utterance.file)
                .filter(Utterance.ignored == True)  # noqa
            )
            if utterances.count():
                path = os.path.join(output_dir, "missing_features.csv")
                with mfa_open(path, "w") as f:
                    for file_name, relative_path, begin, end in utterances:

                        f.write(f"{relative_path + '/' + file_name},{begin},{end}\n")

                self.printer.print_info_lines(
                    f"There were {self.printer.colorize(utterances.count(), 'red')} utterances missing features. "
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
            with mfa_open(path, "w") as f:
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
            with mfa_open(path, "w") as f:
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
            with mfa_open(path, "w") as f:
                for e in self.textgrid_read_errors:
                    f.write(
                        f"The TextGrid file {e.file_name} gave the following error on load:\n\n{e}\n\n\n"
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
            with mfa_open(path, "w") as f:
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
        logger.debug("Analyzing alignment information")
        compile_info_begin = time.time()
        self.collect_alignments()
        jobs = self.compile_information_arguments()

        if GLOBAL_CONFIG.use_mp:
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

        self.printer.print_header("Alignment")
        if not avg_like_frames:
            logger.debug(
                "No utterances were aligned, this likely indicates serious problems with the aligner."
            )
            self.printer.print_red_stat(0, f"of {self.num_utterances} utterances were aligned")
        else:
            if too_short_count:
                self.printer.print_red_stat(
                    too_short_count, "utterances were too short to be aligned"
                )
            else:
                self.printer.print_green_stat(0, "utterances were too short to be aligned")
            if beam_too_narrow_count:
                logger.debug(
                    f"There were {beam_too_narrow_count} utterances that could not be aligned with "
                    f"the current beam settings."
                )
                self.printer.print_yellow_stat(
                    beam_too_narrow_count, "utterances that need a larger beam to align"
                )
            else:
                self.printer.print_green_stat(0, "utterances that need a larger beam to align")

            num_utterances = self.num_utterances
            with self.session() as session:
                unaligned_utts = (
                    session.query(Utterance)
                    .options(joinedload(Utterance.file).load_only(File.name))
                    .filter_by(alignment_log_likelihood=None)
                )
                unaligned_count = unaligned_utts.count()
                if unaligned_count:
                    path = os.path.join(self.output_directory, "unalignable_files.csv")
                    with mfa_open(path, "w") as f:
                        f.write("file,begin,end,duration,text length\n")
                        for u in unaligned_utts:
                            utt_length_words = u.text.count(" ") + 1
                            f.write(
                                f"{u.file.name},{u.begin},{u.end},{u.duration},{utt_length_words}\n"
                            )
                    self.printer.print_info_lines(
                        [
                            f"There were {self.printer.colorize(unaligned_count, 'red')} unaligned utterances out of {self.printer.colorize(self.num_utterances, 'bright')} after initial training. "
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
            logger.debug(f"Average per frame likelihood for alignment: {average_log_like}")
        logger.debug(f"Compiling information took {time.time() - compile_info_begin:.3f} seconds")

    def test_utterance_transcriptions(self) -> None:
        """
        Tests utterance transcriptions with simple unigram models based on the utterance text and frequent
        words in the corpus

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        logger.info("Checking utterance transcriptions...")

        try:
            self.train_speaker_lms()

            self.transcribe(WorkflowType.per_speaker_transcription)

            self.printer.print_header("Test transcriptions")
            ser, wer, cer = self.compute_wer()
            if ser < 0.3:
                self.printer.print_green_stat(f"{ser*100:.2f}%", "sentence error rate")
            elif ser < 0.8:
                self.printer.print_yellow_stat(f"{ser*100:.2f}%", "sentence error rate")
            else:
                self.printer.print_red_stat(f"{ser*100:.2f}%", "sentence error rate")

            if wer < 0.25:
                self.printer.print_green_stat(f"{wer*100:.2f}%", "word error rate")
            elif wer < 0.75:
                self.printer.print_yellow_stat(f"{wer*100:.2f}%", "word error rate")
            else:
                self.printer.print_red_stat(f"{wer*100:.2f}%", "word error rate")

            if cer < 0.25:
                self.printer.print_green_stat(f"{cer*100:.2f}%", "character error rate")
            elif cer < 0.75:
                self.printer.print_yellow_stat(f"{cer*100:.2f}%", "character error rate")
            else:
                self.printer.print_red_stat(f"{cer*100:.2f}%", "character error rate")

            self.save_transcription_evaluation(self.output_directory)
            out_path = os.path.join(self.output_directory, "transcription_evaluation.csv")
            print(f"See {self.printer.colorize(out_path, 'bright')} for more details.")

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise


class TrainingValidator(TrainableAligner, ValidationMixin):
    """
    Validator class for checking whether a corpus and a dictionary will work together
    for training

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.trainer.TrainableAligner`
        For training configuration
    :class:`~montreal_forced_aligner.validation.corpus_validator.ValidationMixin`
        For validation parameters

    Attributes
    ----------
    training_configs: dict[str, :class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonophoneTrainer`]
    """

    def __init__(self, **kwargs):
        training_configuration = kwargs.pop("training_configuration", None)
        super().__init__(**kwargs)
        self.training_configs = {}
        if training_configuration is None:
            training_configuration = [("monophone", {})]
        for k, v in training_configuration:
            self.add_config(k, v)

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:

        """
        Parse parameters for validation from a config path or command-line arguments

        Parameters
        ----------
        config_path: str
            Config path
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

        Returns
        -------
        dict[str, Any]
            Configuration parameters
        """
        global_params = {}
        training_params = []
        use_default = True
        if config_path:
            data = load_configuration(config_path)
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
                    if v is None and k in cls.nullable_fields:
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

    def setup(self) -> None:
        """
        Set up the corpus and validator

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.check_previous_run()
        if hasattr(self, "initialize_database"):
            self.initialize_database()
        if self.initialized:
            return
        try:
            all_begin = time.time()
            self.dictionary_setup()
            logger.debug(f"Loaded dictionary in {time.time() - all_begin:.3f} seconds")

            begin = time.time()
            self._load_corpus()
            logger.debug(f"Loaded corpus in {time.time() - begin:.3f} seconds")

            begin = time.time()
            self.initialize_jobs()
            logger.debug(f"Initialized jobs in {time.time() - begin:.3f} seconds")

            self.normalize_text()

            self.save_oovs_found(self.output_directory)

            begin = time.time()
            self.write_lexicon_information()
            self.write_training_information()
            if self.test_transcriptions:
                self.write_lexicon_information(write_disambiguation=True)
            logger.debug(f"Wrote lexicon information in {time.time() - begin:.3f} seconds")

            if self.ignore_acoustics:
                logger.info("Skipping acoustic feature generation")
            else:
                begin = time.time()
                self.create_corpus_split()
                logger.debug(
                    f"Created corpus split directory in {time.time() - begin:.3f} seconds"
                )
                begin = time.time()
                self.generate_features()
                logger.debug(f"Generated features in {time.time() - begin:.3f} seconds")
                begin = time.time()
                self.save_oovs_found(self.output_directory)
                logger.debug(f"Calculated OOVs in {time.time() - begin:.3f} seconds")
                self.setup_trainers()

            self.initialized = True
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def validate(self) -> None:
        """
        Performs validation of the corpus
        """
        begin = time.time()
        logger.debug(f"Setup took {time.time() - begin:.3f} seconds")
        self.setup()
        self.analyze_setup()
        logger.debug(f"Setup took {time.time() - begin:.3f} seconds")
        if self.ignore_acoustics:
            self.printer.print_info_lines("Skipping test alignments.")
            return
        self.printer.print_header("Training")
        self.train()
        if self.test_transcriptions:
            self.test_utterance_transcriptions()
            self.get_phone_confidences()


class PretrainedValidator(PretrainedAligner, ValidationMixin):
    """
    Validator class for checking whether a corpus, a dictionary, and
    an acoustic model will work together for alignment

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.pretrained.PretrainedAligner`
        For alignment configuration
    :class:`~montreal_forced_aligner.validation.corpus_validator.ValidationMixin`
        For validation parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self) -> None:
        """
        Set up the corpus and validator

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.initialize_database()
        if self.initialized:
            return
        try:
            self.setup_acoustic_model()
            self.dictionary_setup()
            self._load_corpus()
            self.initialize_jobs()
            self.normalize_text()

            self.save_oovs_found(self.output_directory)

            if self.ignore_acoustics:
                logger.info("Skipping acoustic feature generation")
            else:
                self.write_lexicon_information()

                self.create_corpus_split()
                if self.test_transcriptions:
                    self.write_lexicon_information(write_disambiguation=True)
                self.generate_features()
            self.acoustic_model.validate(self)
            self.acoustic_model.log_details()

            self.initialized = True
            logger.info("Finished initializing!")
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def validate(self) -> None:
        """
        Performs validation of the corpus
        """
        self.initialize_database()
        self.create_new_current_workflow(WorkflowType.alignment)
        self.setup()
        self.analyze_setup()
        self.analyze_missing_phones()
        if self.ignore_acoustics:
            logger.info("Skipping test alignments.")
            return
        self.align()
        self.collect_alignments()
        self.compile_information()
        if self.phone_confidence:
            self.get_phone_confidences()

        if self.use_phone_model:
            self.create_new_current_workflow(WorkflowType.phone_transcription)
            self.transcribe()
            self.collect_alignments()
        if self.test_transcriptions:
            self.test_utterance_transcriptions()
            self.collect_alignments()
            self.transcription_done = True
            with self.session() as session:
                session.query(Corpus).update({"transcription_done": True})
                session.commit()

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
