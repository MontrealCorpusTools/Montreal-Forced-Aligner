"""Class definitions for aligning with pretrained acoustic models"""
from __future__ import annotations

import datetime
import logging
import os
import shutil
import time
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import sqlalchemy
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from kalpy.data import Segment
from kalpy.utils import read_kaldi_object
from kalpy.utterance import Utterance as KalpyUtterance
from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.data import PhoneType, WordType, WorkflowType
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    Dictionary,
    Grapheme,
    Phone,
    PhoneInterval,
    Pronunciation,
    Speaker,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import (
    load_configuration,
    mfa_open,
    parse_old_features,
    split_phone_position,
)
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.online.alignment import align_utterance_online
from montreal_forced_aligner.transcription.transcriber import TranscriberMixin
from montreal_forced_aligner.utils import log_kaldi_errors

if TYPE_CHECKING:

    from montreal_forced_aligner.abc import MetaDict

__all__ = ["PretrainedAligner", "DictionaryTrainer"]

logger = logging.getLogger("mfa")


class PretrainedAligner(TranscriberMixin, TopLevelMfaWorker):
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
        acoustic_model_path: Path = None,
        **kwargs,
    ):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        kw = self.acoustic_model.parameters
        kw.update(kwargs)
        super().__init__(**kw)
        self.final_alignment = True

    def setup_acoustic_model(self) -> None:
        """Set up the acoustic model"""
        self.acoustic_model.export_model(self.working_directory)
        os.makedirs(self.phones_dir, exist_ok=True)
        for f in ["phones.txt", "graphemes.txt"]:
            path = self.working_directory.joinpath(f)
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
                    shutil.copyfile(fst_path, dictionary.lexicon_fst_path)
                fst_path = os.path.join(self.acoustic_model.dirname, dict_name + "_align.fst")
                if os.path.exists(fst_path):
                    os.makedirs(dictionary.temp_directory, exist_ok=True)
                    shutil.copyfile(fst_path, dictionary.align_lexicon_path)
            phone_objs = []
            with mfa_open(self.phone_symbol_table_path, "r") as f:
                for line in f:
                    line = line.strip()
                    phone_label, mapping_id = line.split()
                    mapping_id = int(mapping_id)
                    phone_type = PhoneType.non_silence
                    if phone_label.startswith("#"):
                        phone_type = PhoneType.disambiguation
                    elif phone_label in self.kaldi_silence_phones:
                        phone_type = PhoneType.silence
                    phone, pos = split_phone_position(phone_label)
                    phone_objs.append(
                        {
                            "id": mapping_id + 1,
                            "mapping_id": mapping_id,
                            "phone": phone,
                            "position": pos,
                            "kaldi_label": phone_label,
                            "phone_type": phone_type,
                        }
                    )
            grapheme_objs = []
            with mfa_open(self.grapheme_symbol_table_path, "r") as f:
                for line in f:
                    line = line.strip()
                    grapheme, mapping_id = line.split()
                    mapping_id = int(mapping_id)
                    grapheme_objs.append(
                        {"id": mapping_id + 1, "mapping_id": mapping_id, "grapheme": grapheme}
                    )
            session.bulk_insert_mappings(
                Grapheme, grapheme_objs, return_defaults=False, render_nulls=True
            )
            session.bulk_insert_mappings(
                Phone, phone_objs, return_defaults=False, render_nulls=True
            )
            session.commit()

    def setup(self) -> None:
        """Setup for alignment"""
        self.ignore_empty_utterances = True
        super(PretrainedAligner, self).setup()
        if self.initialized:
            return
        begin = time.time()
        try:
            os.makedirs(self.working_log_directory, exist_ok=True)
            check = self.check_previous_run()
            if check:
                logger.debug(
                    "There were some differences in the current run compared to the last one. "
                    "This may cause issues, run with --clean, if you hit an error."
                )
            self.setup_acoustic_model()
            self.load_corpus()
            if self.excluded_pronunciation_count:
                logger.warning(
                    f"There were {self.excluded_pronunciation_count} pronunciations in the dictionary that "
                    f"were ignored for containing one of {len(self.excluded_phones)} phones not present in the "
                    f"trained acoustic model.  Please run `mfa validate` to get more details."
                )
            self.acoustic_model.validate(self)
            self.acoustic_model.log_details()

        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise
        self.initialized = True
        logger.debug(f"Setup for alignment in {time.time() - begin:.3f} seconds")

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters from a config path or command-line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`
            Config path
        args: dict[str, Any]
            Parsed arguments
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
        dictionary_id = utterance.speaker.dictionary_id
        self.acoustic_model.export_model(self.working_directory)
        workflow = self.get_latest_workflow_run(WorkflowType.online_alignment, session)
        if workflow is None:
            workflow = CorpusWorkflow(
                name=f"{utterance.id}_ali",
                workflow_type=WorkflowType.online_alignment,
                time_stamp=datetime.datetime.now(),
                working_directory=self.working_directory,
            )
            session.add(workflow)
            session.flush()
        segment = Segment(
            str(utterance.file.sound_file.sound_file_path),
            utterance.begin,
            utterance.end,
            utterance.channel,
        )
        cmvn_string = utterance.speaker.cmvn
        cmvn = None
        if cmvn_string:
            cmvn = read_kaldi_object(DoubleMatrix, cmvn_string)
        fmllr_string = utterance.speaker.fmllr
        fmllr_trans = None
        if fmllr_string:
            fmllr_trans = read_kaldi_object(FloatMatrix, fmllr_string)

        text = utterance.normalized_text
        if self.use_g2p:
            text = utterance.normalized_character_text
        utterance_data = KalpyUtterance(segment, text, cmvn_string, fmllr_string)
        ctm = align_utterance_online(
            self.acoustic_model,
            utterance_data,
            self.lexicon_compilers[dictionary_id],
            cmvn=cmvn,
            fmllr_trans=fmllr_trans,
            **self.align_options,
        )

        max_phone_interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
        if max_phone_interval_id is None:
            max_phone_interval_id = 0
        max_word_interval_id = session.query(sqlalchemy.func.max(WordInterval.id)).scalar()
        if max_word_interval_id is None:
            max_word_interval_id = 0
        mapping_id = session.query(sqlalchemy.func.max(Word.mapping_id)).scalar()
        if mapping_id is None:
            mapping_id = -1
        mapping_id += 1
        word_index = self.get_next_primary_key(Word)
        new_phone_interval_mappings = []
        new_word_interval_mappings = []
        words = (
            session.query(Word.word, Word.id)
            .filter(Word.dictionary_id == dictionary_id)
            .filter(Word.word.in_(utterance.normalized_text.split()))
        )
        phone_to_phone_id = {}
        word_mapping = {}
        pronunciation_mapping = {}
        ds = session.query(Phone.id, Phone.mapping_id).all()
        for p_id, mapping_id in ds:
            phone_to_phone_id[mapping_id] = p_id
        new_words = []
        for w, w_id in words:
            word_mapping[w] = w_id
        pronunciations = (
            session.query(Word.word, Pronunciation.pronunciation, Pronunciation.id)
            .join(Pronunciation.word)
            .filter(Word.dictionary_id == dictionary_id)
            .filter(Word.word.in_(utterance.normalized_text.split()))
        )
        for w, pron, p_id in pronunciations:
            pronunciation_mapping[(w, pron)] = p_id

        for word_interval in ctm.word_intervals:
            if word_interval.label not in word_mapping:
                new_words.append(
                    {
                        "id": word_index,
                        "mapping_id": mapping_id,
                        "word": word_interval.label,
                        "dictionary_id": 1,
                        "word_type": WordType.oov,
                    }
                )
                word_mapping[word_interval.label] = word_index
                word_id = word_index
            else:
                word_id = word_mapping[word_interval.label]
            max_word_interval_id += 1
            pronunciation_id = pronunciation_mapping.get(
                (word_interval.label, word_interval.pronunciation), None
            )

            new_word_interval_mappings.append(
                {
                    "id": max_word_interval_id,
                    "begin": word_interval.begin,
                    "end": word_interval.end,
                    "word_id": word_id,
                    "pronunciation_id": pronunciation_id,
                    "utterance_id": utterance.id,
                    "workflow_id": workflow.id,
                }
            )
            for interval in word_interval.phones:
                max_phone_interval_id += 1
                new_phone_interval_mappings.append(
                    {
                        "id": max_phone_interval_id,
                        "begin": interval.begin,
                        "end": interval.end,
                        "phone_id": phone_to_phone_id[interval.symbol],
                        "utterance_id": utterance.id,
                        "workflow_id": workflow.id,
                        "word_interval_id": max_word_interval_id,
                        "phone_goodness": interval.confidence if interval.confidence else 0.0,
                    }
                )
        session.query(Utterance).filter(Utterance.id == utterance.id).update(
            {Utterance.alignment_log_likelihood: ctm.likelihood}
        )

        session.query(PhoneInterval).filter(PhoneInterval.utterance_id == utterance.id).filter(
            PhoneInterval.workflow_id == workflow.id
        ).delete()
        session.query(WordInterval).filter(WordInterval.utterance_id == utterance.id).filter(
            WordInterval.workflow_id == workflow.id
        ).delete()
        session.flush()
        if new_words:

            session.bulk_insert_mappings(Word, new_words, return_defaults=False, render_nulls=True)
            session.flush()
        if new_word_interval_mappings:
            session.bulk_insert_mappings(
                WordInterval, new_word_interval_mappings, return_defaults=False, render_nulls=True
            )
            session.bulk_insert_mappings(
                PhoneInterval,
                new_phone_interval_mappings,
                return_defaults=False,
                render_nulls=True,
            )
        session.commit()

    def align(self, workflow_name=None) -> None:
        """Run the aligner"""
        self.initialize_database()
        self.create_new_current_workflow(WorkflowType.alignment, workflow_name)
        wf = self.current_workflow
        if wf.done:
            logger.info("Alignment already done, skipping.")
            return
        self.setup()
        super().align()


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

    def export_lexicons(self, output_directory: str) -> None:
        """
        Generate pronunciation probabilities for the dictionary

        Parameters
        ----------
        output_directory: str
            Directory in which to save new dictionaries

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
                )
