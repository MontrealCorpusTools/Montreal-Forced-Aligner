"""Class definitions for aligning with pretrained acoustic models"""
from __future__ import annotations

import os
import subprocess
import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

import yaml

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import align_phones, parse_old_features
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import log_kaldi_errors, run_mp, run_non_mp, thirdparty_binary

if TYPE_CHECKING:
    from argparse import Namespace

    from montreal_forced_aligner.abc import MetaDict

__all__ = ["PretrainedAligner"]


def generate_pronunciations_func(
    log_path: str,
    dictionaries: List[str],
    text_int_paths: Dict[str, str],
    word_boundary_paths: Dict[str, str],
    ali_paths: Dict[str, str],
    model_path: str,
    pron_paths: Dict[str, str],
):
    """
    Multiprocessing function for generating pronunciations

    See Also
    --------
    :meth:`.DictionaryTrainer.export_lexicons`
        Main function that calls this function in parallel
    :meth:`.DictionaryTrainer.generate_pronunciations_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`linear-to-nbest`
        Kaldi binary this uses

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: list[str]
        List of dictionary names
    text_int_paths: dict[str, str]
        Dictionary of text int files per dictionary name
    word_boundary_paths: dict[str, str]
        Dictionary of word boundary files per dictionary name
    ali_paths: dict[str, str]
        Dictionary of alignment archives per dictionary name
    model_path: str
        Path to acoustic model file
    pron_paths: dict[str, str]
        Dictionary of pronunciation archives per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            text_int_path = text_int_paths[dict_name]
            word_boundary_path = word_boundary_paths[dict_name]
            ali_path = ali_paths[dict_name]
            pron_path = pron_paths[dict_name]

            lin_proc = subprocess.Popen(
                [
                    thirdparty_binary("linear-to-nbest"),
                    f"ark:{ali_path}",
                    f"ark:{text_int_path}",
                    "",
                    "",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            align_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-align-words"),
                    word_boundary_path,
                    model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=lin_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            prons_proc = subprocess.Popen(
                [thirdparty_binary("nbest-to-prons"), model_path, "ark:-", pron_path],
                stdin=align_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            prons_proc.communicate()


class GeneratePronunciationsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.alignment.pretrained.generate_pronunciations_func`"""

    log_path: str
    dictionaries: List[str]
    text_int_paths: Dict[str, str]
    word_boundary_paths: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    pron_paths: Dict[str, str]


class PretrainedAligner(CorpusAligner, TopLevelMfaWorker):
    """
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    acoustic_model_path : str
        Path to acoustic model

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.base.CorpusAligner`
        For dictionary and corpus parsing parameters and alignment parameters
    :class:`~montreal_forced_aligner.abc.TopLevelMfaWorker`
        For top-level parameters
    """

    def __init__(
        self,
        acoustic_model_path: str,
        **kwargs,
    ):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        kw = self.acoustic_model.parameters
        kw.update(kwargs)
        super().__init__(**kw)

    @property
    def working_directory(self) -> str:
        """Working directory"""
        return self.workflow_directory

    def setup(self) -> None:
        """Setup for alignment"""
        if self.initialized:
            return
        begin = time.time()
        try:
            os.makedirs(self.working_log_directory, exist_ok=True)
            check = self.check_previous_run()
            if check:
                self.logger.debug(
                    "There were some differences in the current run compared to the last one. "
                    "This may cause issues, run with --clean, if you hit an error."
                )
            self.load_corpus()
            if self.excluded_pronunciation_count:
                self.logger.warning(
                    f"There were {self.excluded_pronunciation_count} pronunciations in the dictionary that"
                    f"were ignored for containing one of {len(self.excluded_phones)} phones not present in the"
                    f"trained acoustic model.  Please run `mfa validate` to get more details."
                )
            self.acoustic_model.validate(self)
            self.acoustic_model.export_model(self.working_directory)
            self.acoustic_model.log_details(self.logger)

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        self.initialized = True
        self.logger.debug(f"Setup for alignment in {time.time() - begin} seconds")

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters from a config path or command-line arguments

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
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                data = parse_old_features(data)
                for k, v in data.items():
                    if k == "features":
                        global_params.update(v)
                    else:
                        if v is None and k in {
                            "punctuation",
                            "compound_markers",
                            "clitic_markers",
                        }:
                            v = []
                        global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    @property
    def configuration(self) -> MetaDict:
        """Configuration for aligner"""
        config = super().configuration
        config.update(
            {
                "acoustic_model": self.acoustic_model.name,
            }
        )
        return config

    @property
    def backup_output_directory(self) -> Optional[str]:
        """Backup output directory if overwriting is not allowed"""
        if self.overwrite:
            return None
        return os.path.join(self.working_directory, "textgrids")

    @property
    def workflow_identifier(self) -> str:
        """Aligner identifier"""
        return "pretrained_aligner"

    def evaluate(
        self,
        mapping: Optional[Dict[str, str]] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        """
        Evaluate alignments against a reference directory

        Parameters
        ----------
        reference_directory: str
            Directory containing reference TextGrid alignments
        mapping: dict[str, Union[str, list[str]]], optional
            Mapping between phones that should be considered equal across different phone set types
        output_directory: str, optional
            Directory to save results, if not specified, it will be saved in the log directory
        """
        # Set up
        self.log_info("Evaluating alignments...")
        self.log_debug(f"Mapping: {mapping}")

        score_count = 0
        score_sum = 0
        phone_edit_sum = 0
        phone_length_sum = 0
        if output_directory:
            csv_path = os.path.join(output_directory, "alignment_evaluation.csv")
        else:
            csv_path = os.path.join(self.working_log_directory, "alignment_evaluation.csv")
        with open(csv_path, "w", encoding="utf8") as f:
            f.write(
                "utterance,file,speaker,duration,word_count,oov_count,reference_phone_count,score,phone_error_rate\n"
            )
            for utterance in self.utterances:
                if not utterance.reference_phone_labels:
                    continue
                speaker = utterance.speaker_name
                file = utterance.file_name
                duration = utterance.duration
                reference_phone_count = len(utterance.reference_phone_labels)
                word_count = len(utterance.text.split())
                oov_count = len(utterance.oovs)
                if not utterance.phone_labels:  # couldn't be aligned
                    utterance.alignment_score = None
                    utterance.phone_error_rate = len(utterance.reference_phone_labels)
                    f.write(
                        f"{utterance.name},{file},{speaker},{duration},{word_count},{oov_count},{reference_phone_count},na,{len(utterance.reference_phone_labels)}\n"
                    )

                    continue
                score, phone_error_rate = align_phones(
                    utterance.reference_phone_labels,
                    utterance.phone_labels,
                    self.optional_silence_phone,
                    mapping,
                )
                if score is None:
                    continue
                utterance.alignment_score = score
                utterance.phone_error_rate = phone_error_rate
                f.write(
                    f"{utterance.name},{file},{speaker},{duration},{word_count},{oov_count},{reference_phone_count},{score},{phone_error_rate}\n"
                )
                score_count += 1
                score_sum += score
                phone_edit_sum += int(phone_error_rate * reference_phone_count)
                phone_length_sum += reference_phone_count
        self.logger.info(f"Average overlap score: {score_sum/score_count}")
        self.logger.info(f"Average phone error rate: {phone_edit_sum/phone_length_sum}")

    def align(self) -> None:
        """Run the aligner"""
        self.setup()
        done_path = os.path.join(self.working_directory, "done")
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(done_path):
            self.logger.info("Alignment already done, skipping.")
            return
        try:
            log_dir = os.path.join(self.working_directory, "log")
            os.makedirs(log_dir, exist_ok=True)
            self.compile_train_graphs()

            self.logger.info("Performing first-pass alignment...")
            self.speaker_independent = True
            self.align_utterances()
            self.compile_information()
            if self.uses_speaker_adaptation:
                self.calc_fmllr()

                self.speaker_independent = False
                self.logger.info("Performing second-pass alignment...")
                self.align_utterances()

                self.compile_information()
            self.collect_alignments()
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        with open(done_path, "w"):
            pass


class DictionaryTrainer(PretrainedAligner):
    """
    Aligner for calculating pronunciation probabilities of dictionary entries

    Parameters
    ----------
    calculate_silence_probs: bool
        Flag for whether to calculate silence probabilities, default is False
    min_count: int
        Specifies the minimum count of words to include in derived probabilities,
        affects probabilities of infrequent words more, default is 1

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.pretrained.PretrainedAligner`
        For dictionary and corpus parsing parameters and alignment parameters
    """

    def __init__(
        self,
        calculate_silence_probs: bool = False,
        min_count: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.calculate_silence_probs = calculate_silence_probs
        self.min_count = min_count

    def generate_pronunciations_arguments(
        self,
    ) -> List[GeneratePronunciationsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.pretrained.generate_pronunciations_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.pretrained.GeneratePronunciationsArguments`]
            Arguments for processing
        """
        return [
            GeneratePronunciationsArguments(
                os.path.join(self.working_log_directory, f"generate_pronunciations.{j.name}.log"),
                j.current_dictionary_names,
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.word_boundary_int_files(),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.model_path,
                j.construct_path_dictionary(self.working_directory, "prons", "scp"),
            )
            for j in self.jobs
        ]

    def export_lexicons(self, output_directory: str) -> None:
        """
        Generate pronunciation probabilities for the dictionary

        Parameters
        ----------
        output_directory: str
            Directory in which to save new dictionaries

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.pretrained.generate_pronunciations_func`
            Multiprocessing helper function for each job
        :meth:`.DictionaryTrainer.generate_pronunciations_arguments`
            Job method for generating arguments for helper function

        """
        os.makedirs(output_directory, exist_ok=True)
        jobs = self.generate_pronunciations_arguments()
        if self.use_mp:
            run_mp(generate_pronunciations_func, jobs, self.working_log_directory)
        else:
            run_non_mp(generate_pronunciations_func, jobs, self.working_log_directory)
        pron_counts = {}
        utt_mapping = {}
        for j in self.jobs:
            args = jobs[j.name]
            for dict_name, pron_path in args.pron_paths.items():
                if dict_name not in pron_counts:
                    pron_counts[dict_name] = defaultdict(Counter)
                    utt_mapping[dict_name] = {}
                with open(pron_path, "r", encoding="utf8") as f:
                    last_utt = None
                    for line in f:
                        line = line.split()
                        utt = line[0]
                        if utt not in utt_mapping[dict_name]:
                            if last_utt is not None:
                                utt_mapping[dict_name][last_utt].append("</s>")
                            utt_mapping[dict_name][utt] = ["<s>"]
                            last_utt = utt
                        dictionary = self.get_dictionary(self.utterances[utt].speaker_name)
                        word = dictionary.reversed_word_mapping[int(line[3])]
                        if word == "<eps>":
                            utt_mapping[dict_name][utt].append(word)
                        else:
                            pron = tuple(
                                dictionary.reversed_phone_mapping[int(x)].split("_")[0]
                                for x in line[4:]
                            )
                            pron_string = " ".join(pron)
                            utt_mapping[dict_name][utt].append(word + " " + pron_string)
                            pron_counts[dict_name][word][pron] += 1
        for dict_name, dictionary in self.dictionary_mapping.items():
            counts = pron_counts[dict_name]
            mapping = utt_mapping[dict_name]
            if self.calculate_silence_probs:
                sil_before_counts = Counter()
                nonsil_before_counts = Counter()
                sil_after_counts = Counter()
                nonsil_after_counts = Counter()
                sils = ["<s>", "</s>", "<eps>"]
                for v in mapping.values():
                    for i, w in enumerate(v):
                        if w in sils:
                            continue
                        prev_w = v[i - 1]
                        next_w = v[i + 1]
                        if prev_w in sils:
                            sil_before_counts[w] += 1
                        else:
                            nonsil_before_counts[w] += 1
                        if next_w in sils:
                            sil_after_counts[w] += 1
                        else:
                            nonsil_after_counts[w] += 1

            dictionary.pronunciation_probabilities = True
            for word, prons in dictionary.actual_words.items():
                if word not in counts:
                    for p in prons:
                        p.probability = 1
                else:
                    total = 0
                    best_pron = 0
                    best_count = 0
                    for p in prons:
                        p.probability = self.min_count
                        if p.pronunciation in counts[word]:
                            p.probability += counts[word][p.pronunciation]
                        total += p.probability
                        if p.probability > best_count:
                            best_pron = p.pronunciation
                            best_count = p.probability
                    for p in prons:
                        if p.pronunciation == best_pron:
                            p.probability = 1
                        else:
                            p.probability /= total
                    dictionary.words[word] = prons
            output_path = os.path.join(output_directory, dict_name + ".txt")
            dictionary.export_lexicon(output_path, probability=True)
