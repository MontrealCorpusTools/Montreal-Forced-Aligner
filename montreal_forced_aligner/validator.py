"""
Validating corpora
==================

"""
from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from .abc import AcousticModelWorker
from .aligner.pretrained import PretrainedAligner
from .config import FeatureConfig
from .exceptions import CorpusError, KaldiProcessingError
from .helper import edit_distance, load_scp
from .multiprocessing import run_mp, run_non_mp
from .multiprocessing.alignment import compile_utterance_train_graphs_func, test_utterances_func
from .trainers import MonophoneTrainer
from .utils import log_kaldi_errors

if TYPE_CHECKING:
    from .corpus.base import Corpus
    from .dictionary import MultispeakerDictionary


__all__ = ["CorpusValidator"]


class CorpusValidator(AcousticModelWorker):
    """
    Validator class for checking whether a corpus, a dictionary, and (optionally) an acoustic model work together

    Parameters
    ----------
    corpus : :class:`~montreal_forced_aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~montreal_forced_aligner.dictionary.MultispeakerDictionary`
        MultispeakerDictionary object for the pronunciation dictionary
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    ignore_acoustics: bool, optional
        Flag for whether all acoustics should be ignored, which speeds up the validation, defaults to False
    test_transcriptions: bool, optional
        Flag for whether the validator should test transcriptions, defaults to False
    use_mp: bool, optional
        Flag for whether to use multiprocessing
    logger: :class:`~logging.Logger`, optional
        Logger to use

    Attributes
    ----------
    corpus_analysis_template: str
        Template for output message
    alignment_analysis_template: str
        Template for output message
    transcription_analysis_template: str
        Template for output message
    """

    corpus_analysis_template = """
    =========================================Corpus=========================================
    {} sound files
    {} sound files with .lab transcription files
    {} sound files with TextGrids transcription files
    {} additional sound files ignored (see below)
    {} speakers
    {} utterances
    {} seconds total duration

    DICTIONARY
    ----------
    {}

    SOUND FILE READ ERRORS
    ----------------------
    {}

    FEATURE CALCULATION
    -------------------
    {}

    FILES WITHOUT TRANSCRIPTIONS
    ----------------------------
    {}

    TRANSCRIPTIONS WITHOUT FILES
    --------------------
    {}

    TEXTGRID READ ERRORS
    --------------------
    {}

    UNREADABLE TEXT FILES
    --------------------
    {}
    """

    alignment_analysis_template = """
    =======================================Alignment========================================
    {}
    """

    transcription_analysis_template = """
    ====================================Transcriptions======================================
    {}
    """

    def __init__(
        self,
        corpus: Corpus,
        dictionary: MultispeakerDictionary,
        temp_directory: Optional[str] = None,
        ignore_acoustics: bool = False,
        test_transcriptions: bool = False,
        use_mp: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(corpus, dictionary)
        self.temp_directory = temp_directory
        self.test_transcriptions = test_transcriptions
        self.ignore_acoustics = ignore_acoustics
        self.trainer: MonophoneTrainer = MonophoneTrainer(FeatureConfig())
        self.logger = logger
        self.trainer.logger = logger
        self.trainer.update({"use_mp": use_mp})
        self.setup()

    @property
    def working_directory(self) -> str:
        return os.path.join(self.temp_directory, "validation")

    @property
    def working_log_directory(self) -> str:
        return os.path.join(self.working_directory, "log")

    def setup(self):
        """
        Set up the corpus and validator

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.dictionary.set_word_set(self.corpus.word_set)
        self.dictionary.write()
        if self.test_transcriptions:
            self.dictionary.write(write_disambiguation=True)
        if self.ignore_acoustics:
            fc = None
            if self.logger is not None:
                self.logger.info("Skipping acoustic feature generation")
        else:
            fc = self.trainer.feature_config
        try:
            self.corpus.initialize_corpus(self.dictionary, fc)
            if self.test_transcriptions:
                self.corpus.initialize_utt_fsts()
        except CorpusError:
            if self.logger is not None:
                self.logger.warning(
                    "There was an error when initializing the corpus, likely due to missing sound files. Ignoring acoustic generation..."
                )
            self.ignore_acoustics = True

    def analyze_setup(self):
        """
        Analyzes the set up process and outputs info to the console
        """
        total_duration = sum(x.duration for x in self.corpus.files.values())
        total_duration = Decimal(str(total_duration)).quantize(Decimal("0.001"))

        ignored_count = len(self.corpus.no_transcription_files)
        ignored_count += len(self.corpus.textgrid_read_errors)
        ignored_count += len(self.corpus.decode_error_files)
        self.logger.info(
            self.corpus_analysis_template.format(
                sum(1 for x in self.corpus.files.values() if x.wav_path is not None),
                sum(1 for x in self.corpus.files.values() if x.text_type == "lab"),
                sum(1 for x in self.corpus.files.values() if x.text_type == "textgrid"),
                ignored_count,
                len(self.corpus.speakers),
                self.corpus.num_utterances,
                total_duration,
                self.analyze_oovs(),
                self.analyze_wav_errors(),
                self.analyze_missing_features(),
                self.analyze_files_with_no_transcription(),
                self.analyze_transcriptions_with_no_wavs(),
                self.analyze_textgrid_read_errors(),
                self.analyze_unreadable_text_files(),
            )
        )

    def analyze_oovs(self) -> str:
        """
        Analyzes OOVs in the corpus and constructs message

        Returns
        -------
        str
            OOV validation result message
        """
        output_dir = self.corpus.output_directory
        oov_types = self.dictionary.oovs_found
        oov_path = os.path.join(output_dir, "oovs_found.txt")
        utterance_oov_path = os.path.join(output_dir, "utterance_oovs.txt")
        if oov_types:
            total_instances = 0
            with open(utterance_oov_path, "w", encoding="utf8") as f:
                for utt, utterance in sorted(self.corpus.utterances.items()):
                    if not utterance.oovs:
                        continue
                    total_instances += len(utterance.oovs)
                    f.write(f"{utt} {', '.join(utterance.oovs)}\n")
            self.dictionary.save_oovs_found(output_dir)
            message = (
                f"There were {len(oov_types)} word types not found in the dictionary with a total of {total_instances} instances.\n\n"
                f"    Please see \n\n        {oov_path}\n\n    for a full list of the word types and \n\n        {utterance_oov_path}\n\n    for a by-utterance breakdown of "
                f"missing words."
            )
        else:
            message = (
                "There were no missing words from the dictionary. If you plan on using the a model trained "
                "on this dataset to align other datasets in the future, it is recommended that there be at "
                "least some missing words."
            )
        return message

    def analyze_wav_errors(self) -> str:
        """
        Analyzes any sound file issues in the corpus and constructs message

        Returns
        -------
        str
            Sound file validation result message
        """
        output_dir = self.corpus.output_directory
        wav_read_errors = self.corpus.sound_file_errors
        if wav_read_errors:
            path = os.path.join(output_dir, "sound_file_errors.csv")
            with open(path, "w") as f:
                for p in wav_read_errors:
                    f.write(f"{p}\n")

            message = (
                f"There were {len(wav_read_errors)} sound files that could not be read. "
                f"Please see {path} for a list."
            )
        else:
            message = "There were no sound files that could not be read."

        return message

    def analyze_missing_features(self) -> str:
        """
        Analyzes issues in feature generation in the corpus and constructs message

        Returns
        -------
        str
            Feature validation result message
        """
        if self.ignore_acoustics:
            return "Acoustic feature generation was skipped."
        output_dir = self.corpus.output_directory
        missing_features = [x for x in self.corpus.utterances.values() if x.ignored]
        if missing_features:
            path = os.path.join(output_dir, "missing_features.csv")
            with open(path, "w") as f:
                for utt in missing_features:
                    if utt.begin is not None:

                        f.write(f"{utt.file.wav_path},{utt.begin},{utt.end}\n")
                    else:
                        f.write(f"{utt.file.wav_path}\n")

            message = (
                f"There were {len(missing_features)} utterances missing features. "
                f"Please see {path} for a list."
            )
        else:
            message = "There were no utterances missing features."
        return message

    def analyze_files_with_no_transcription(self) -> str:
        """
        Analyzes issues with sound files that have no transcription files
        in the corpus and constructs message

        Returns
        -------
        str
            File matching validation result message
        """
        output_dir = self.corpus.output_directory
        if self.corpus.no_transcription_files:
            path = os.path.join(output_dir, "missing_transcriptions.csv")
            with open(path, "w") as f:
                for file_path in self.corpus.no_transcription_files:
                    f.write(f"{file_path}\n")
            message = (
                f"There were {len(self.corpus.no_transcription_files)} sound files missing transcriptions. "
                f"Please see {path} for a list."
            )
        else:
            message = "There were no sound files missing transcriptions."
        return message

    def analyze_transcriptions_with_no_wavs(self) -> str:
        """
        Analyzes issues with transcription that have no sound files
        in the corpus and constructs message

        Returns
        -------
        str
            File matching validation result message
        """
        output_dir = self.corpus.output_directory
        if self.corpus.transcriptions_without_wavs:
            path = os.path.join(output_dir, "transcriptions_missing_sound_files.csv")
            with open(path, "w") as f:
                for file_path in self.corpus.transcriptions_without_wavs:
                    f.write(f"{file_path}\n")
            message = (
                f"There were {len(self.corpus.transcriptions_without_wavs)} transcription files missing sound files. "
                f"Please see {path} for a list."
            )
        else:
            message = "There were no transcription files missing sound files."
        return message

    def analyze_textgrid_read_errors(self) -> str:
        """
        Analyzes issues with reading TextGrid files
        in the corpus and constructs message

        Returns
        -------
        str
            TextGrid validation result message
        """
        output_dir = self.corpus.output_directory
        if self.corpus.textgrid_read_errors:
            path = os.path.join(output_dir, "textgrid_read_errors.txt")
            with open(path, "w") as f:
                for k, v in self.corpus.textgrid_read_errors.items():
                    f.write(
                        f"The TextGrid file {k} gave the following error on load:\n\n{v}\n\n\n"
                    )
            message = (
                f"There were {len(self.corpus.textgrid_read_errors)} TextGrid files that could not be parsed. "
                f"Please see {path} for a list."
            )
        else:
            message = "There were no issues reading TextGrids."
        return message

    def analyze_unreadable_text_files(self) -> str:
        """
        Analyzes issues with reading text files
        in the corpus and constructs message

        Returns
        -------
        str
            Text file validation result message
        """
        output_dir = self.corpus.output_directory
        if self.corpus.decode_error_files:
            path = os.path.join(output_dir, "utf8_read_errors.csv")
            with open(path, "w") as f:
                for file_path in self.corpus.decode_error_files:
                    f.write(f"{file_path}\n")
            message = (
                f"There were {len(self.corpus.decode_error_files)} text files that could not be parsed. "
                f"Please see {path} for a list."
            )
        else:
            message = "There were no issues reading text files."
        return message

    def analyze_unaligned_utterances(self) -> None:
        """
        Analyzes issues with any unaligned files following training
        """
        unaligned_utts = self.trainer.get_unaligned_utterances()
        num_utterances = self.corpus.num_utterances
        if unaligned_utts:
            path = os.path.join(self.corpus.output_directory, "unalignable_files.csv")
            with open(path, "w") as f:
                f.write("File path,begin,end,duration,text length\n")
                for utt in unaligned_utts:
                    utterance = self.corpus.utterances[utt]
                    utt_duration = utterance.duration
                    utt_length_words = utterance.text.count(" ") + 1
                    if utterance.begin is not None:
                        f.write(
                            f"{utterance.file.wav_path},{utterance.begin},{utterance.end},{utt_duration},{utt_length_words}\n"
                        )
                    else:
                        f.write(f"{utterance.file.wav_path},,,{utt_duration},{utt_length_words}\n")
            message = (
                f"There were {len(unaligned_utts)} unalignable utterances out of {num_utterances} after the initial training. "
                f"Please see {path} for a list."
            )
        else:
            message = f"All {num_utterances} utterances were successfully aligned!"
        print(self.alignment_analysis_template.format(message))

    def validate(self):
        """
        Performs validation of the corpus
        """
        self.analyze_setup()
        if self.ignore_acoustics:
            print("Skipping test alignments.")
            return
        if not isinstance(self.trainer, PretrainedAligner):
            self.trainer.init_training(
                self.trainer.train_type, self.temp_directory, self.corpus, self.dictionary, None
            )
            self.trainer.train()
        self.trainer.align(None)
        self.analyze_unaligned_utterances()
        if self.test_transcriptions:
            self.test_utterance_transcriptions()

    def test_utterance_transcriptions(self):
        """
        Tests utterance transcriptions with simple unigram models based on the utterance text and frequent
        words in the corpus

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.logger.info("Checking utterance transcriptions...")

        model_directory = self.trainer.align_directory
        log_directory = os.path.join(model_directory, "log")

        try:

            jobs = [x.compile_utterance_train_graphs_arguments(self) for x in self.corpus.jobs]
            if self.trainer.feature_config.use_mp:
                run_mp(compile_utterance_train_graphs_func, jobs, log_directory)
            else:
                run_non_mp(compile_utterance_train_graphs_func, jobs, log_directory)
            self.logger.info("Utterance FSTs compiled!")
            self.logger.info("Decoding utterances (this will take some time)...")
            jobs = [x.test_utterances_arguments(self) for x in self.corpus.jobs]
            if self.trainer.feature_config.use_mp:
                run_mp(test_utterances_func, jobs, log_directory)
            else:
                run_non_mp(test_utterances_func, jobs, log_directory)
            self.logger.info("Finished decoding utterances!")

            errors = {}

            for job in jobs:
                for dict_name in job.dictionaries:
                    word_mapping = self.dictionary.dictionary_mapping[
                        dict_name
                    ].reversed_word_mapping
                    aligned_int = load_scp(job.out_int_paths[dict_name])
                    for utt, line in sorted(aligned_int.items()):
                        text = []
                        for t in line:
                            text.append(word_mapping[int(t)])
                        ref_text = self.corpus.utterances[utt].text.split()
                        edits = edit_distance(text, ref_text)

                        if edits:
                            errors[utt] = (edits, ref_text, text)
            if not errors:
                message = "There were no utterances with transcription issues."
            else:
                out_path = os.path.join(self.corpus.output_directory, "transcription_problems.csv")
                with open(out_path, "w") as problemf:
                    problemf.write("Utterance,Edits,Reference,Decoded\n")
                    for utt, (edits, ref_text, text) in sorted(
                        errors.items(), key=lambda x: -1 * (len(x[1][1]) + len(x[1][2]))
                    ):
                        problemf.write(f"{utt},{edits},{' '.join(ref_text)},{' '.join(text)}\n")
                message = (
                    f"There were {len(errors)} of {self.corpus.num_utterances} utterances with at least one transcription issue. "
                    f"Please see the outputted csv file {out_path}."
                )

            self.logger.info(self.transcription_analysis_template.format(message))

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger.handlers[0].baseFilename)
            raise
