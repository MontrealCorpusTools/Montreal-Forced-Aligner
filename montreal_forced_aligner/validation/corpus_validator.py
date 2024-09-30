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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import sqlalchemy

from montreal_forced_aligner.acoustic_modeling.trainer import TrainableAligner
from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import Corpus, File, SoundFile, Speaker, TextFile, Utterance
from montreal_forced_aligner.exceptions import ConfigError, KaldiProcessingError
from montreal_forced_aligner.helper import comma_join, load_configuration, mfa_open
from montreal_forced_aligner.utils import log_kaldi_errors

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

    @property
    def working_log_directory(self) -> str:
        """Working log directory"""
        return self.working_directory.joinpath("log")

    def analyze_setup(self, output_directory: Path = None) -> None:
        """
        Analyzes the setup process and outputs info to the console

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
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

        logger.info("Corpus")
        logger.info(f"{sound_file_count} sound files")
        logger.info(f"{text_file_count} text files")
        if len(self.no_transcription_files):
            logger.warning(
                f"{len(self.no_transcription_files)} sound files without corresponding transcriptions",
            )
        if len(self.decode_error_files):
            logger.error(f"{len(self.decode_error_files)} read errors for lab files")
        if len(self.textgrid_read_errors):
            logger.error(f"{len(self.textgrid_read_errors)} read errors for TextGrid files")

        logger.info(f"{self.num_speakers} speakers")
        logger.info(f"{self.num_utterances} utterances")
        logger.info(f"{total_duration} seconds total duration")
        self.analyze_wav_errors()
        self.analyze_missing_features()
        self.analyze_files_with_no_transcription()
        self.analyze_transcriptions_with_no_wavs()

        if len(self.decode_error_files):
            self.analyze_unreadable_text_files()
        if len(self.textgrid_read_errors):
            self.analyze_textgrid_read_errors()

        logger.info("Dictionary")
        self.analyze_oovs(output_directory=output_directory)

    def analyze_oovs(self, output_directory: Path = None) -> None:
        """
        Analyzes OOVs in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("Out of vocabulary words")
        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        oov_path = os.path.join(output_directory, "oovs_found.txt")
        utterance_oov_path = os.path.join(output_directory, "utterance_oovs.txt")

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
                    f"{relative_path.joinpath(file_name)}, {speaker_name}: {begin}-{end}: {', '.join(oovs)}\n"
                )
                self.oovs_found.update(oovs)
        if self.oovs_found:
            self.save_oovs_found(output_directory)
            logger.warning(f"{len(self.oovs_found)} OOV word types")
            logger.warning(f"{total_instances}total OOV tokens")
            logger.warning(
                f"For a full list of the word types, please see: {oov_path}. "
                f"For a by-utterance breakdown of missing words, see: {utterance_oov_path}"
            )
        else:
            logger.info(
                "There were no missing words from the dictionary. If you plan on using the a model trained "
                "on this dataset to align other datasets in the future, it is recommended that there be at "
                "least some missing words."
            )

    def analyze_wav_errors(self, output_directory: Path = None) -> None:
        """
        Analyzes any sound file issues in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("Sound file read errors")

        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        wav_read_errors = self.sound_file_errors
        if wav_read_errors:
            path = os.path.join(output_directory, "sound_file_errors.csv")
            with mfa_open(path, "w") as f:
                for p in wav_read_errors:
                    f.write(f"{p}\n")

            logger.error(
                f"There were {len(wav_read_errors)} issues reading sound files. "
                f"Please see {path} for a list."
            )
        else:
            logger.info("There were no issues reading sound files.")

    def analyze_missing_features(self, output_directory: Path = None) -> None:
        """
        Analyzes issues in feature generation in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("Feature generation")
        if self.ignore_acoustics:
            logger.info("Acoustic feature generation was skipped.")
            return

        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        with self.session() as session:
            utterances = (
                session.query(File.name, File.relative_path, Utterance.begin, Utterance.end)
                .join(Utterance.file)
                .filter(Utterance.ignored == True)  # noqa
            )
            if utterances.count():
                path = os.path.join(output_directory, "missing_features.csv")
                with mfa_open(path, "w") as f:
                    for file_name, relative_path, begin, end in utterances:
                        f.write(f"{relative_path.joinpath(file_name)},{begin},{end}\n")

                logger.error(
                    f"There were {utterances.count()} utterances missing features. "
                    f"Please see {path} for a list."
                )
            else:
                logger.info("There were no utterances missing features.")

    def analyze_files_with_no_transcription(self, output_directory: Path = None) -> None:
        """
        Analyzes issues with sound files that have no transcription files
        in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("Files without transcriptions")
        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        if self.no_transcription_files:
            path = os.path.join(output_directory, "missing_transcriptions.csv")
            with mfa_open(path, "w") as f:
                for file_path in self.no_transcription_files:
                    f.write(f"{file_path}\n")
            logger.error(
                f"There were {len(self.no_transcription_files)} sound files missing transcriptions."
            )
            logger.error(f"Please see {path} for a list.")
        else:
            logger.info("There were no sound files missing transcriptions.")

    def analyze_transcriptions_with_no_wavs(self, output_directory: Path = None) -> None:
        """
        Analyzes issues with transcription that have no sound files
        in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("Transcriptions without sound files")
        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        if self.transcriptions_without_wavs:
            path = os.path.join(output_directory, "transcriptions_missing_sound_files.csv")
            with mfa_open(path, "w") as f:
                for file_path in self.transcriptions_without_wavs:
                    f.write(f"{file_path}\n")
            logger.error(
                f"There were {len(self.transcriptions_without_wavs)} transcription files missing sound files. "
                f"Please see {path} for a list."
            )
        else:
            logger.info("There were no transcription files missing sound files.")

    def analyze_textgrid_read_errors(self, output_directory: Path = None) -> None:
        """
        Analyzes issues with reading TextGrid files
        in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("TextGrid read errors")
        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        if self.textgrid_read_errors:
            path = os.path.join(output_directory, "textgrid_read_errors.txt")
            with mfa_open(path, "w") as f:
                for e in self.textgrid_read_errors:
                    f.write(
                        f"The TextGrid file {e.file_name} gave the following error on load:\n\n{e}\n\n\n"
                    )
            logger.error(
                f"There were {len(self.textgrid_read_errors)} TextGrid files that could not be loaded. "
                f"For details, please see: {path}",
            )
        else:
            logger.info("There were no issues reading TextGrids.")

    def analyze_unreadable_text_files(self, output_directory: Path = None) -> None:
        """
        Analyzes issues with reading text files
        in the corpus and constructs message

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        logger.info("Text file read errors")
        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        if self.decode_error_files:
            path = os.path.join(output_directory, "utf8_read_errors.csv")
            with mfa_open(path, "w") as f:
                for file_path in self.decode_error_files:
                    f.write(f"{file_path}\n")
            logger.error(
                f"There were {len(self.decode_error_files)} text files that could not be read. "
                f"Please see {path} for a list."
            )
        else:
            logger.info("There were no issues reading text files.")

    def test_utterance_transcriptions(self, output_directory: Path = None) -> None:
        """
        Tests utterance transcriptions with simple unigram models based on the utterance text and frequent
        words in the corpus

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """

        if output_directory is None:
            output_directory = self.output_directory
        os.makedirs(output_directory, exist_ok=True)
        try:
            self.subset_lexicon(write_disambiguation=True)
            self.train_speaker_lms()

            self.transcribe(WorkflowType.per_speaker_transcription)

            logger.info("Test transcriptions")
            ser, wer, cer = self.compute_wer()
            if ser < 0.3:
                logger.info(f"{ser * 100:.2f}% sentence error rate")
            elif ser < 0.8:
                logger.warning(f"{ser * 100:.2f}% sentence error rate")
            else:
                logger.error(f"{ser * 100:.2f}% sentence error rate")

            if wer < 0.25:
                logger.info(f"{wer * 100:.2f}% word error rate")
            elif wer < 0.75:
                logger.warning(f"{wer * 100:.2f}% word error rate")
            else:
                logger.error(f"{wer * 100:.2f}% word error rate")

            if cer < 0.25:
                logger.info(f"{cer * 100:.2f}% character error rate")
            elif cer < 0.75:
                logger.warning(f"{cer * 100:.2f}% character error rate")
            else:
                logger.error(f"{cer * 100:.2f}% character error rate")

            self.save_transcription_evaluation(output_directory)
            out_path = os.path.join(output_directory, "transcription_evaluation.csv")
            logger.info(f"See {out_path} for more details.")

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

    @property
    def working_directory(self) -> Path:
        if self.current_workflow.workflow_type in [
            WorkflowType.transcription,
            WorkflowType.per_speaker_transcription,
        ]:
            return self.output_directory.joinpath(self._current_workflow)
        return super().working_directory

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters for validation from a config path or command-line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`
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
                self.generate_features()
                logger.debug(f"Generated features in {time.time() - begin:.3f} seconds")
                begin = time.time()
                logger.debug(f"Calculated OOVs in {time.time() - begin:.3f} seconds")
                self.setup_trainers()

            self.initialized = True
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def validate(self, output_directory: Path = None) -> None:
        """
        Performs validation of the corpus

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        begin = time.time()
        logger.debug(f"Setup took {time.time() - begin:.3f} seconds")
        self.setup()
        self.analyze_setup(output_directory=output_directory)
        logger.debug(f"Setup took {time.time() - begin:.3f} seconds")
        if self.ignore_acoustics:
            logger.info("Skipping test alignments.")
            return
        logger.info("Training")
        self.train()
        if self.test_transcriptions:
            self.test_utterance_transcriptions(output_directory=output_directory)
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
        self.dirty = True  # Always reset validate
        self.initialize_database()
        if self.initialized:
            return
        try:
            self.setup_acoustic_model()
            self.dictionary_setup()
            self._load_corpus()
            self.initialize_jobs()
            self.normalize_text()

            if self.ignore_acoustics:
                logger.info("Skipping acoustic feature generation")
            else:
                self.write_lexicon_information()

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

    def validate(self, output_directory: Path = None) -> None:
        """
        Performs validation of the corpus

        Parameters
        ----------
        output_directory: Path, optional
            Optional directory to save output files in
        """
        self.initialize_database()
        self.create_new_current_workflow(WorkflowType.alignment)
        self.setup()
        self.analyze_setup(output_directory=output_directory)
        self.analyze_missing_phones()
        if self.ignore_acoustics:
            logger.info("Skipping test alignments.")
            return
        self.align()
        self.collect_alignments()
        if self.phone_confidence:
            self.get_phone_confidences()

        if self.use_phone_model:
            self.create_new_current_workflow(WorkflowType.phone_transcription)
            self.transcribe()
            self.collect_alignments()
        if self.test_transcriptions:
            self.test_utterance_transcriptions(output_directory=output_directory)
            self.collect_alignments()
            self.transcription_done = True
            with self.session() as session:
                session.query(Corpus).update({"transcription_done": True})
                session.commit()

    def analyze_missing_phones(self) -> None:
        """Analyzes dictionary and acoustic model for phones in the dictionary that don't have acoustic models"""
        logger.info("Acoustic model compatibility")
        if self.excluded_pronunciation_count:
            logger.warning(len(self.excluded_phones), "phones not in acoustic model")
            logger.warning(self.excluded_pronunciation_count, "ignored pronunciations")

            logger.error(
                f"Phones missing acoustic models: {comma_join(sorted(self.excluded_phones))}"
            )
        else:
            logger.info("There were no phones in the dictionary without acoustic models.")
