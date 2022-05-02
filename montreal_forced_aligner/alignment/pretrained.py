"""Class definitions for aligning with pretrained acoustic models"""
from __future__ import annotations

import csv
import functools
import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import sqlalchemy
from sqlalchemy.orm import Session, joinedload, subqueryload

from montreal_forced_aligner.abc import TopLevelMfaWorker
from montreal_forced_aligner.alignment.base import CorpusAligner
from montreal_forced_aligner.db import (
    Corpus,
    DictBundle,
    Dictionary,
    File,
    PhoneInterval,
    Speaker,
    Utterance,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignerError, KaldiProcessingError
from montreal_forced_aligner.helper import align_phones, load_configuration, parse_old_features
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
            self.load_corpus()
            if self.excluded_pronunciation_count:
                self.log_warning(
                    f"There were {self.excluded_pronunciation_count} pronunciations in the dictionary that "
                    f"were ignored for containing one of {len(self.excluded_phones)} phones not present in the"
                    f"trained acoustic model.  Please run `mfa validate` to get more details."
                )
            self.acoustic_model.validate(self)
            self.acoustic_model.export_model(self.working_directory)
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

    def evaluate(
        self,
        mapping: Optional[Dict[str, str]] = None,
        output_directory: Optional[str] = None,
    ) -> None:
        """
        Evaluate alignments against a reference directory

        Parameters
        ----------
        mapping: dict[str, Union[str, list[str]]], optional
            Mapping between phones that should be considered equal across different phone set types
        output_directory: str, optional
            Directory to save results, if not specified, it will be saved in the log directory
        """
        begin = time.time()
        if output_directory:
            csv_path = os.path.join(output_directory, "alignment_evaluation.csv")
        else:
            csv_path = os.path.join(self.working_log_directory, "alignment_evaluation.csv")
        csv_header = [
            "file",
            "begin",
            "end",
            "speaker",
            "duration",
            "normalized_text",
            "oovs",
            "reference_phone_count",
            "alignment_score",
            "phone_error_rate",
            "alignment_log_likelihood",
            "word_count",
            "oov_count",
        ]

        score_count = 0
        score_sum = 0
        phone_edit_sum = 0
        phone_length_sum = 0
        if self.alignment_evaluation_done:
            self.log_info("Exporting saved evaluation...")
            with self.session() as session, open(csv_path, "w", encoding="utf8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
                bn = DictBundle(
                    "evaluation_data",
                    File.c.name.label("file"),
                    Utterance.begin,
                    Utterance.end,
                    Speaker.c.name.label("speaker"),
                    Utterance.duration,
                    Utterance.normalized_text,
                    Utterance.oovs,
                    sqlalchemy.func.count(Utterance.reference_phone_intervals).label(
                        "reference_phone_count"
                    ),
                    Utterance.alignment_score,
                    Utterance.phone_error_rate,
                    Utterance.alignment_log_likelihood,
                )
                utterances = (
                    session.query(bn)
                    .join(Utterance.speaker)
                    .join(Utterance.file)
                    .group_by(Utterance.id)
                    .join(Utterance.reference_phone_intervals)
                )
                for line in utterances:
                    data = line["evaluation_data"]
                    data["word_count"] = len(data["normalized_text"].split())
                    data["oov_count"] = len(data["oovs"].split())
                    phone_error_rate = data["phone_error_rate"]
                    reference_phone_count = data["reference_phone_count"]
                    if data["alignment_score"] is not None:
                        score_count += 1
                        score_sum += data["alignment_score"]
                    phone_edit_sum += int(phone_error_rate * reference_phone_count)
                    phone_length_sum += reference_phone_count
                    writer.writerow(data)
        else:
            # Set up
            self.log_info("Evaluating alignments...")
            self.log_debug(f"Mapping: {mapping}")
            update_mappings = []
            indices = []
            to_comp = []
            score_func = functools.partial(
                align_phones,
                silence_phone=self.optional_silence_phone,
                ignored_phones={self.oov_phone},
                custom_mapping=mapping,
            )
            with self.session() as session:
                unaligned_utts = []
                utterances = session.query(Utterance).options(
                    subqueryload(Utterance.reference_phone_intervals),
                    subqueryload(Utterance.phone_intervals),
                    joinedload(Utterance.file, innerjoin=True),
                    joinedload(Utterance.speaker, innerjoin=True),
                )
                for u in utterances:
                    reference_phone_count = len(u.reference_phone_intervals)
                    if not reference_phone_count:
                        continue
                    if u.alignment_log_likelihood is None:  # couldn't be aligned
                        phone_error_rate = reference_phone_count
                        unaligned_utts.append(u)
                        update_mappings.append(
                            {
                                "id": u.id,
                                "alignment_score": None,
                                "phone_error_rate": phone_error_rate,
                            }
                        )
                        continue
                    reference_phone_labels = [x.as_ctm() for x in u.reference_phone_intervals]
                    phone_labels = [x.as_ctm() for x in u.phone_intervals]
                    indices.append(u)
                    to_comp.append((reference_phone_labels, phone_labels))

                with mp.Pool(self.num_jobs) as pool, open(
                    csv_path, "w", encoding="utf8", newline=""
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                    writer.writeheader()
                    gen = pool.starmap(score_func, to_comp)
                    for u in unaligned_utts:
                        word_count = len(u.normalized_text.split())
                        oov_count = len(u.oovs.split())
                        reference_phone_count = len(u.reference_phone_intervals)
                        writer.writerow(
                            {
                                "file": u.file_name,
                                "begin": u.begin,
                                "end": u.end,
                                "speaker": u.speaker_name,
                                "duration": u.duration,
                                "normalized_text": u.normalized_text,
                                "oovs": u.oovs,
                                "reference_phone_count": reference_phone_count,
                                "alignment_score": None,
                                "phone_error_rate": reference_phone_count,
                                "alignment_log_likelihood": None,
                                "word_count": word_count,
                                "oov_count": oov_count,
                            }
                        )
                    for i, (score, phone_error_rate) in enumerate(gen):
                        if score is None:
                            continue
                        u = indices[i]

                        word_count = len(u.normalized_text.split())
                        oov_count = len(u.oovs.split())
                        reference_phone_count = len(u.reference_phone_intervals)
                        update_mappings.append(
                            {
                                "id": u.id,
                                "alignment_score": score,
                                "phone_error_rate": phone_error_rate,
                            }
                        )
                        writer.writerow(
                            {
                                "file": u.file_name,
                                "begin": u.begin,
                                "end": u.end,
                                "speaker": u.speaker_name,
                                "duration": u.duration,
                                "normalized_text": u.normalized_text,
                                "oovs": u.oovs,
                                "reference_phone_count": reference_phone_count,
                                "alignment_score": score,
                                "phone_error_rate": phone_error_rate,
                                "alignment_log_likelihood": u.alignment_log_likelihood,
                                "word_count": word_count,
                                "oov_count": oov_count,
                            }
                        )
                        score_count += 1
                        score_sum += score
                        phone_edit_sum += int(phone_error_rate * reference_phone_count)
                        phone_length_sum += reference_phone_count
                session.bulk_update_mappings(Utterance, update_mappings)
                session.query(Corpus).update({"alignment_evaluation_done": True})
                session.commit()
        self.log_info(f"Average overlap score: {score_sum/score_count}")
        self.log_info(f"Average phone error rate: {phone_edit_sum/phone_length_sum}")
        self.log_debug(f"Alignment evaluation took {time.time()-begin} seconds")

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
