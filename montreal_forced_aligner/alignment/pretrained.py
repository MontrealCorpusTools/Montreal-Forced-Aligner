"""Class definitions for aligning with pretrained acoustic models"""
from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.data import PhoneType
from montreal_forced_aligner.db import (
    Dictionary,
    Grapheme,
    Phone,
    PhoneInterval,
    Speaker,
    Utterance,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignerError, KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, parse_old_features
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.online.alignment import (
    OnlineAlignmentArguments,
    OnlineAlignmentFunction,
)
from montreal_forced_aligner.utils import log_kaldi_errors

if TYPE_CHECKING:
    from argparse import Namespace

    from montreal_forced_aligner.abc import MetaDict

__all__ = ["PretrainedAligner"]


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

    def setup_acoustic_model(self) -> None:
        """Set up the acoustic model"""
        self.acoustic_model.export_model(self.working_directory)
        os.makedirs(self.phones_dir, exist_ok=True)
        exist_check = os.path.exists(self.db_path)
        if not exist_check:
            self.initialize_database()
        for f in ["phones.txt", "graphemes.txt"]:
            path = os.path.join(self.working_directory, f)
            if os.path.exists(path):
                os.rename(path, os.path.join(self.phones_dir, f))
        dict_info = self.acoustic_model.meta.get("dictionaries", None)
        if not dict_info:
            return
        os.makedirs(self.dictionary_output_directory, exist_ok=True)
        self.oov_word = dict_info["oov_word"]
        self.silence_word = dict_info["silence_word"]
        self.bracketed_word = dict_info["bracketed_word"]
        self.use_g2p = dict_info["use_g2p"]
        self.laughter_word = dict_info["laughter_word"]
        self.clitic_marker = dict_info["clitic_marker"]
        self.position_dependent_phones = dict_info["position_dependent_phones"]
        self.compile_regexes()
        if not self.use_g2p:
            return
        dictionary_id_cache = {}
        with self.session() as session:
            for speaker_id, speaker_name, dictionary_id, dict_name, path in (
                session.query(
                    Speaker.id, Speaker.name, Dictionary.id, Dictionary.name, Dictionary.path
                )
                .outerjoin(Speaker.dictionary)
                .filter(Dictionary.default == False)  # noqa
            ):
                if speaker_id is not None:
                    self._speaker_ids[speaker_name] = speaker_id
                dictionary_id_cache[path] = dictionary_id
                self.dictionary_lookup[dict_name] = dictionary_id
            dictionary = (
                session.query(Dictionary).filter(Dictionary.default == True).first()  # noqa
            )
            if dictionary:
                self._default_dictionary_id = dictionary.id
                dictionary_id_cache[dictionary.path] = self._default_dictionary_id
                self.dictionary_lookup[dictionary.name] = dictionary.id
            for dict_name in dict_info["names"]:
                dictionary = Dictionary(
                    name=dict_name,
                    path=dict_name,
                    phone_set_type=self.phone_set_type,
                    root_temp_directory=self.dictionary_output_directory,
                    position_dependent_phones=self.position_dependent_phones,
                    clitic_marker=self.clitic_marker,
                    bracket_regex=self.bracket_regex.pattern,
                    clitic_cleanup_regex=self.clitic_cleanup_regex.pattern,
                    laughter_regex=self.laughter_regex.pattern,
                    default=dict_name == dict_info["default"],
                    use_g2p=self.use_g2p,
                    max_disambiguation_symbol=0,
                    silence_word=self.silence_word,
                    oov_word=self.oov_word,
                    bracketed_word=self.bracketed_word,
                    laughter_word=self.laughter_word,
                    optional_silence_phone=self.optional_silence_phone,
                )
                session.add(dictionary)
                session.flush()
                dictionary_id_cache[dict_name] = dictionary.id
                if dictionary.default:
                    self._default_dictionary_id = dictionary.id
                fst_path = os.path.join(self.acoustic_model.dirname, dict_name + ".fst")
                if os.path.exists(fst_path):
                    os.makedirs(dictionary.temp_directory, exist_ok=True)
                    shutil.copyfile(fst_path, os.path.join(dictionary.temp_directory, "L.fst"))
            phone_objs = []
            with open(self.phone_symbol_table_path, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    phone, mapping_id = line.split()
                    mapping_id = int(mapping_id)
                    phone_type = PhoneType.non_silence
                    if phone.startswith("#"):
                        phone_type = PhoneType.disambiguation
                    elif phone in self.kaldi_silence_phones:
                        phone_type = PhoneType.silence
                    phone_objs.append(
                        {
                            "id": mapping_id + 1,
                            "mapping_id": mapping_id,
                            "phone": phone,
                            "phone_type": phone_type,
                        }
                    )
            grapheme_objs = []
            with open(self.grapheme_symbol_table_path, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    grapheme, mapping_id = line.split()
                    mapping_id = int(mapping_id)
                    grapheme_objs.append(
                        {"id": mapping_id + 1, "mapping_id": mapping_id, "grapheme": grapheme}
                    )
            session.bulk_insert_mappings(Grapheme, grapheme_objs)
            session.bulk_insert_mappings(Phone, phone_objs)
            session.commit()

    def setup(self) -> None:
        """Setup for alignment"""
        if self.initialized:
            return
        begin = time.time()
        try:
            os.makedirs(self.working_log_directory, exist_ok=True)
            check = self.check_previous_run()
            if check:
                self.log_debug(
                    "There were some differences in the current run compared to the last one. "
                    "This may cause issues, run with --clean, if you hit an error."
                )
            self.setup_acoustic_model()
            self.load_corpus()
            if self.excluded_pronunciation_count:
                self.log_warning(
                    f"There were {self.excluded_pronunciation_count} pronunciations in the dictionary that "
                    f"were ignored for containing one of {len(self.excluded_phones)} phones not present in the"
                    f"trained acoustic model.  Please run `mfa validate` to get more details."
                )
            self.acoustic_model.validate(self)
            import logging

            logger = logging.getLogger(self.identifier)
            self.acoustic_model.log_details(logger)

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                import logging

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
            raise
        self.initialized = True
        self.log_debug(f"Setup for alignment in {time.time() - begin} seconds")

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
    def workflow_identifier(self) -> str:
        """Aligner identifier"""
        return "pretrained_aligner"

    def align_one_utterance(self, utterance: Utterance, session: Session) -> None:
        """
        Align a single utterance

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.db.Utterance`
            Utterance object to align
        session: :class:`~sqlalchemy.orm.session.Session`
            Session to use
        """
        dictionary = utterance.speaker.dictionary
        self.acoustic_model.export_model(self.working_directory)
        sox_string = utterance.file.sound_file.sox_string
        if not sox_string:
            sox_string = utterance.file.sound_file.sound_file_path
        text_int_path = os.path.join(self.working_directory, "text.int")
        with open(text_int_path, "w", encoding="utf8") as f:
            f.write(f"{utterance.kaldi_id} {utterance.normalized_text_int}\n")
        if utterance.features:
            feats_path = os.path.join(self.working_directory, "feats.scp")
            with open(feats_path, "w", encoding="utf8") as f:
                f.write(f"{utterance.kaldi_id} {utterance.features}\n")
        else:
            wav_path = os.path.join(self.working_directory, "wav.scp")
            segment_path = os.path.join(self.working_directory, "segments.scp")
            with open(wav_path, "w", encoding="utf8") as f:
                f.write(f"{utterance.file_id} {sox_string}\n")
            with open(segment_path, "w", encoding="utf8") as f:
                f.write(
                    f"{utterance.kaldi_id} {utterance.file_id} {utterance.begin} {utterance.end} {utterance.channel}\n"
                )
        if utterance.speaker.cmvn:
            cmvn_path = os.path.join(self.working_directory, "cmvn.scp")
            with open(cmvn_path, "w", encoding="utf8") as f:
                f.write(f"{utterance.speaker.id} {utterance.speaker.cmvn}\n")
        spk2utt_path = os.path.join(self.working_directory, "spk2utt.scp")
        utt2spk_path = os.path.join(self.working_directory, "utt2spk.scp")
        with open(spk2utt_path, "w") as f:
            f.write(f"{utterance.speaker.id} {utterance.kaldi_id}\n")
        with open(utt2spk_path, "w") as f:
            f.write(f"{utterance.kaldi_id} {utterance.speaker.id}\n")

        args = OnlineAlignmentArguments(
            0,
            self.db_path,
            os.path.join(self.working_directory, "align.log"),
            self.working_directory,
            sox_string,
            utterance.to_data(),
            self.mfcc_options,
            self.pitch_options,
            self.feature_options,
            self.align_options,
            self.alignment_model_path,
            self.tree_path,
            self.disambiguation_symbols_int_path,
            dictionary.lexicon_fst_path,
            dictionary.word_boundary_int_path,
            self.reversed_phone_mapping,
            self.optional_silence_phone,
            {self.silence_word},
        )
        func = OnlineAlignmentFunction(args)
        word_intervals, phone_intervals, log_likelihood = func.run()
        session.query(PhoneInterval).filter(PhoneInterval.utterance_id == utterance.id).delete()
        session.query(WordInterval).filter(WordInterval.utterance_id == utterance.id).delete()
        session.flush()
        for wi in word_intervals:
            session.add(WordInterval.from_ctm(wi, utterance))
        for pi in phone_intervals:
            session.add(PhoneInterval.from_ctm(pi, utterance))
        utterance.alignment_log_likelihood = log_likelihood
        session.commit()

    def align(self) -> None:
        """Run the aligner"""
        self.setup()
        done_path = os.path.join(self.working_directory, "done")
        dirty_path = os.path.join(self.working_directory, "dirty")
        if os.path.exists(done_path):
            self.log_info("Alignment already done, skipping.")
            return
        try:
            log_dir = os.path.join(self.working_directory, "log")
            os.makedirs(log_dir, exist_ok=True)
            self.compile_train_graphs()

            self.log_info("Performing first-pass alignment...")
            self.speaker_independent = True
            self.align_utterances()
            self.compile_information()
            if self.uses_speaker_adaptation:
                if self.alignment_model_path.endswith(".mdl"):
                    if os.path.exists(self.alignment_model_path.replace(".mdl", ".alimdl")):
                        raise AlignerError(
                            "Not using speaker independent model when it is available"
                        )
                self.calc_fmllr()

                self.speaker_independent = False
                assert self.alignment_model_path.endswith(".mdl")
                self.log_info("Performing second-pass alignment...")
                self.align_utterances()

                self.compile_information()
        except Exception as e:
            with open(dirty_path, "w"):
                pass
            if isinstance(e, KaldiProcessingError):
                import logging

                logger = logging.getLogger(self.identifier)
                log_kaldi_errors(e.error_logs, logger)
                e.update_log_file(logger)
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

    def export_lexicons(
        self, output_directory: str, silence_probabilities: Optional[bool] = False
    ) -> None:
        """
        Generate pronunciation probabilities for the dictionary

        Parameters
        ----------
        output_directory: str
            Directory in which to save new dictionaries
        silence_probabilities: bool
            Flag for whether to save silence probabilities as well

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`
            Multiprocessing helper function for each job
        :meth:`.CorpusAligner.generate_pronunciations_arguments`
            Job method for generating arguments for helper function

        """
        self.compute_pronunciation_probabilities()
        os.makedirs(output_directory, exist_ok=True)
        with self.session() as session:
            for dictionary in session.query(Dictionary):
                self.export_lexicon(
                    dictionary.id,
                    os.path.join(output_directory, dictionary.name + ".dict"),
                    probability=True,
                    silence_probabilities=silence_probabilities,
                )
