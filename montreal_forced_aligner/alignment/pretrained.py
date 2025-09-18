"""Class definitions for aligning with pretrained acoustic models"""
from __future__ import annotations

import datetime
import logging
import os
import shutil
import time
import typing
from pathlib import Path

import sqlalchemy
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from kalpy.aligner import KalpyAligner
from kalpy.data import Segment
from kalpy.utils import read_kaldi_object
from kalpy.utterance import Utterance as KalpyUtterance
from sqlalchemy.orm import Session, subqueryload

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.multiprocessing import (
    AnalyzeTranscriptsFunction,
    FineTuneArguments,
    FineTuneFunction,
)
from montreal_forced_aligner.data import PhoneType, WorkflowType
from montreal_forced_aligner.db import (
    Corpus,
    CorpusWorkflow,
    Dictionary,
    Grapheme,
    Job,
    Phone,
    PhoneInterval,
    Speaker,
    Utterance,
    WordInterval,
    bulk_update,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import (
    load_configuration,
    mfa_open,
    parse_old_features,
    split_phone_position,
)
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.online.alignment import update_utterance_intervals
from montreal_forced_aligner.transcription.transcriber import TranscriberMixin
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function

if typing.TYPE_CHECKING:
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
        use_reference_alignments: bool = False,
        fine_tune: bool = False,
        fine_tune_boundary_tolerance: typing.Optional[float] = None,
        **kwargs,
    ):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        self.use_reference_alignments = use_reference_alignments
        kw = self.acoustic_model.parameters
        kw.update(kwargs)
        super().__init__(**kw)
        self.fine_tune = fine_tune
        self.fine_tune = fine_tune_boundary_tolerance
        self.final_alignment = True
        self.kalpy_aligner = None

    def setup_acoustic_model(self) -> None:
        """Set up the acoustic model"""
        self.acoustic_model.export_model(self.working_directory)
        os.makedirs(self.phones_dir, exist_ok=True)
        for f in ["phones.txt", "graphemes.txt"]:
            path = self.working_directory.joinpath(f)
            if os.path.exists(path) and not os.path.exists(os.path.join(self.phones_dir, f)):
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
            duration_information = self.acoustic_model.meta.get("duration_information", {})
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
                    if phone_label in duration_information:
                        mean_duration, sd_duration = duration_information[phone_label]
                        phone_objs[-1]["mean_duration"] = mean_duration
                        phone_objs[-1]["sd_duration"] = sd_duration
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

    def analyze_alignments(self, calculate_duration_statistics=True):
        super().analyze_alignments(calculate_duration_statistics=calculate_duration_statistics)

    def setup(self) -> None:
        """Setup for alignment"""
        self.ignore_empty_utterances = True
        super(PretrainedAligner, self).setup()
        if self.initialized:
            return
        begin = time.time()
        try:
            self.initialize_database()
            self.create_new_current_workflow(WorkflowType.alignment)
            wf = self.current_workflow
            if wf.done:
                logger.info("Alignment already done, skipping.")
                return
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
            self.kalpy_aligner = KalpyAligner(
                self.acoustic_model,
                self.lexicon_compilers,
                **self.align_options,
            )

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
        config_path: typing.Optional[Path] = None,
        args: typing.Optional[typing.Dict[str, typing.Any]] = None,
        unknown_args: typing.Optional[typing.Iterable[str]] = None,
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
        workflow = self.get_latest_workflow_run(WorkflowType.online_alignment, session)
        if workflow is None:
            workflow = CorpusWorkflow(
                name="online_alignment",
                workflow_type=WorkflowType.online_alignment,
                time_stamp=datetime.datetime.now(),
                working_directory=self.output_directory.joinpath("online_alignment"),
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
        ctm = self.kalpy_aligner.align_utterance(
            utterance_data, cmvn, fmllr_trans, dictionary_id=dictionary_id
        )
        update_utterance_intervals(session, utterance, ctm)

    def verify_transcripts(self, workflow_name=None) -> None:
        self.initialize_database()
        self.create_new_current_workflow(WorkflowType.transcript_verification, workflow_name)
        wf = self.current_workflow
        if wf.done:
            logger.info("Transcript verification already done, skipping.")
            return
        self.setup()
        self.write_lexicon_information(write_disambiguation=True)
        super().align()

        arguments = self.analyze_alignments_arguments()
        update_mappings = []
        for utt_id, word_error_rate, duration_deviation, transcript in run_kaldi_function(
            AnalyzeTranscriptsFunction, arguments, total_count=self.num_current_utterances
        ):
            update_mappings.append(
                {
                    "id": utt_id,
                    "word_error_rate": word_error_rate,
                    "duration_deviation": duration_deviation,
                    "transcription_text": transcript,
                }
            )
        with self.session() as session:
            bulk_update(session, Utterance, update_mappings)
            session.commit()

    def reset_manual_alignments(self):
        with self.session() as session:
            corpus = session.query(Corpus).first()
            if corpus is not None and corpus.has_reference_alignments:
                session.query(Utterance).update({Utterance.manual_alignments: False})
                session.commit()
                jobs = session.query(Job).options(subqueryload(Job.dictionaries))
                for job in jobs:
                    for d in job.training_dictionaries:
                        reference_phones_path = job.construct_path(
                            job.corpus.current_subset_directory, "ref_phones", "ark", d.name
                        )
                        reference_phones_path.unlink(missing_ok=True)

    def check_manual_alignments(self):
        with self.session() as session:
            corpus = session.query(Corpus).first()
            if corpus is not None and corpus.has_reference_alignments:
                ref_phone_counts = (
                    session.query(Utterance.id, sqlalchemy.func.count())
                    .join(Utterance.reference_phone_intervals)
                    .group_by(Utterance.id)
                )
                mapping = []
                for u_id, c in ref_phone_counts:
                    if c > 0:
                        mapping.append({"id": u_id, "manual_alignments": True})
                bulk_update(session, Utterance, mapping)
                session.commit()

    def _align(self):
        if not self.use_reference_alignments:
            self.reset_manual_alignments()
        super()._align()
        if self.fine_tune:
            self.fine_tune_alignments()

    def align(self) -> None:
        """Run the aligner"""
        self.setup()
        super().align()

    def fine_tune_alignments(self) -> None:
        """
        Fine tune aligned boundaries to millisecond precision
        """
        logger.info("Fine tuning alignments...")
        all_begin = time.time()
        with self.session() as session:
            arguments = self.fine_tune_arguments()
            update_mappings = []
            for result in run_kaldi_function(
                FineTuneFunction, arguments, total_count=self.num_utterances
            ):
                update_mappings.extend(result)
            bulk_update(session, PhoneInterval, update_mappings)
            session.flush()
            word_update_mappings = []
            word_intervals = (
                session.query(
                    WordInterval.id,
                    sqlalchemy.func.min(PhoneInterval.begin),
                    sqlalchemy.func.max(PhoneInterval.end),
                )
                .join(PhoneInterval.word_interval)
                .group_by(WordInterval.id)
            )
            for wi_id, begin, end in word_intervals:
                word_update_mappings.append({"id": wi_id, "begin": begin, "end": end})
            bulk_update(session, WordInterval, word_update_mappings)
            session.commit()
        self.export_frame_shift = round(self.export_frame_shift / 10, 4)
        logger.debug(f"Fine tuning alignments took {time.time() - all_begin:.3f} seconds")

    def fine_tune_arguments(self) -> typing.List[FineTuneArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneArguments`]
            Arguments for processing
        """
        args = []
        for j in self.jobs:
            log_path = self.working_log_directory.joinpath(f"fine_tune.{j.id}.log")
            args.append(
                FineTuneArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    log_path,
                    self.acoustic_model,
                    self.lexicon_compilers,
                    self.fine_tune_boundary_tolerance,
                )
            )
        return args


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
