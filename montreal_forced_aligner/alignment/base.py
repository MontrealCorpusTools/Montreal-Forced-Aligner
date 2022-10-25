"""Class definitions for base aligner"""
from __future__ import annotations

import collections
import csv
import functools
import multiprocessing as mp
import os
import subprocess
import time
import typing
from queue import Empty
from typing import Dict, List, Optional

import sqlalchemy
import tqdm
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner.abc import FileExporterMixin
from montreal_forced_aligner.alignment.mixins import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    AlignmentExtractionArguments,
    AlignmentExtractionFunction,
    ExportTextGridArguments,
    ExportTextGridProcessWorker,
    FineTuneArguments,
    FineTuneFunction,
    GeneratePronunciationsArguments,
    GeneratePronunciationsFunction,
    TranscriptionAlignmentExtractionFunction,
    construct_output_path,
    construct_output_tiers,
)
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.data import (
    PronunciationProbabilityCounter,
    TextFileType,
    WorkflowType,
)
from montreal_forced_aligner.db import (
    Corpus,
    Dictionary,
    File,
    PhoneInterval,
    PhonologicalRule,
    Pronunciation,
    RuleApplication,
    SoundFile,
    Speaker,
    TextFile,
    Utterance,
    Word,
    WordInterval,
    WordType,
    bulk_update,
)
from montreal_forced_aligner.exceptions import AlignmentExportError
from montreal_forced_aligner.helper import (
    align_phones,
    format_correction,
    format_probability,
    mfa_open,
)
from montreal_forced_aligner.textgrid import export_textgrid, output_textgrid_writing_errors
from montreal_forced_aligner.utils import Counter, KaldiProcessWorker, Stopped, thirdparty_binary

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

__all__ = ["CorpusAligner"]


class CorpusAligner(AcousticCorpusPronunciationMixin, AlignMixin, FileExporterMixin):
    """
    Mixin class that aligns corpora with pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.alignment.mixins.AlignMixin`
        For alignment parameters
    :class:`~montreal_forced_aligner.abc.FileExporterMixin`
        For file exporting parameters
    """

    def __init__(self, max_active: int = 2500, lattice_beam: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.export_output_directory = None
        self.max_active = max_active
        self.lattice_beam = lattice_beam
        self.phone_lm_order = 2
        self.phone_lm_method = "unsmoothed"

    @property
    def hclg_options(self) -> MetaDict:
        """Options for constructing HCLG FSTs"""
        context_width, central_pos = self.get_tree_info()
        return {
            "context_width": context_width,
            "central_pos": central_pos,
            "self_loop_scale": self.self_loop_scale,
            "transition_scale": self.transition_scale,
        }

    @property
    def decode_options(self) -> MetaDict:
        """Options needed for decoding"""
        return {
            "first_beam": getattr(self, "first_beam", 10),
            "beam": self.beam,
            "first_max_active": getattr(self, "first_max_active", 2000),
            "max_active": self.max_active,
            "lattice_beam": self.lattice_beam,
            "acoustic_scale": self.acoustic_scale,
            "transition_scale": self.transition_scale,
            "self_loop_scale": self.self_loop_scale,
            "uses_speaker_adaptation": self.uses_speaker_adaptation,
        }

    @property
    def score_options(self) -> MetaDict:
        """Options needed for scoring lattices"""
        return {
            "frame_shift": round(self.frame_shift / 1000, 3),
            "language_model_weight": getattr(self, "language_model_weight", 10),
            "word_insertion_penalty": getattr(self, "word_insertion_penalty", 0.5),
        }

    def alignment_extraction_arguments(
        self, from_transcription=False
    ) -> List[AlignmentExtractionArguments]:
        """
        Generate Job arguments for
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionArguments`]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            if not j.has_data:
                continue
            ali_paths = j.construct_path_dictionary(self.working_directory, "ali", "ark")
            if from_transcription:
                ali_paths = j.construct_path_dictionary(self.working_directory, "lat", "ark")
            arguments.append(
                AlignmentExtractionArguments(
                    j.name,
                    getattr(self, "read_only_db_string", ""),
                    os.path.join(self.working_log_directory, f"get_phone_ctm.{j.name}.log"),
                    self.alignment_model_path,
                    round(self.frame_shift / 1000, 4),
                    ali_paths,
                    j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                    self.phone_symbol_table_path,
                    self.score_options,
                )
            )

        return arguments

    def export_textgrid_arguments(
        self, output_format: str, workflow_id: int, include_original_text: bool = False
    ) -> List[ExportTextGridArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`

        Parameters
        ----------
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridArguments`]
            Arguments for processing
        """
        return [
            ExportTextGridArguments(
                j.name,
                getattr(self, "read_only_db_string", ""),
                os.path.join(self.working_log_directory, f"export_textgrids.{j.name}.log"),
                self.export_frame_shift,
                GLOBAL_CONFIG.cleanup_textgrids,
                self.clitic_marker,
                self.export_output_directory,
                output_format,
                include_original_text,
                workflow_id,
            )
            for j in self.jobs
            if j.has_data
        ]

    def generate_pronunciations_arguments(
        self,
    ) -> List[GeneratePronunciationsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsArguments`]
            Arguments for processing
        """

        return [
            GeneratePronunciationsArguments(
                j.name,
                getattr(self, "read_only_db_string", ""),
                os.path.join(self.working_log_directory, f"generate_pronunciations.{j.name}.log"),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.model_path,
                False,
            )
            for j in self.jobs
            if j.has_data
        ]

    def compute_pronunciation_probabilities(self):
        """
        Multiprocessing function that computes pronunciation probabilities from alignments

        Parameters
        ----------
        compute_silence_probabilities: bool
            Flag for whether to compute silence probabilities for pronunciations, defaults to True

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`
            Multiprocessing helper function for each job
        :meth:`.CorpusAligner.generate_pronunciations_arguments`
            Job method for generating arguments for the helper function
        :kaldi_steps:`align_si`
            Reference Kaldi script
        :kaldi_steps:`align_fmllr`
            Reference Kaldi script
        """

        begin = time.time()
        dictionary_counters = {
            dict_id: PronunciationProbabilityCounter()
            for dict_id in self.dictionary_lookup.values()
        }
        self.log_info("Generating pronunciations...")
        arguments = self.generate_pronunciations_arguments()
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = GeneratePronunciationsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    dict_id, utterance_counter = result
                    dictionary_counters[dict_id].add_counts(utterance_counter)
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                self.log_debug("Not using multiprocessing...")
                for args in arguments:
                    function = GeneratePronunciationsFunction(args)
                    for dict_id, utterance_counter in function.run():
                        dictionary_counters[dict_id].add_counts(utterance_counter)
                        pbar.update(1)

        initial_key = ("<s>", "")
        final_key = ("</s>", "")
        lambda_2 = 2
        silence_prob_sum = 0
        initial_silence_prob_sum = 0
        final_silence_correction_sum = 0
        final_non_silence_correction_sum = 0
        with mfa_open(
            os.path.join(self.working_log_directory, "pronunciation_probability_calculation.log"),
            "w",
            encoding="utf8",
        ) as log_file, self.session() as session:
            dictionaries = session.query(Dictionary.id)
            dictionary_mappings = []
            for (d_id,) in dictionaries:
                counter = dictionary_counters[d_id]
                log_file.write(f"For {d_id}:\n")
                words = (
                    session.query(Word.word)
                    .filter(Word.dictionary_id == d_id)
                    .filter(Word.word_type != WordType.silence)
                )
                pronunciations = (
                    session.query(Word.word, Pronunciation.pronunciation, Pronunciation.id)
                    .join(Pronunciation.word)
                    .filter(Word.dictionary_id == d_id)
                    .filter(Word.word_type != WordType.silence)
                )
                pron_mapping = {}
                for w, p, p_id in pronunciations:
                    pron_mapping[(w, p)] = {"id": p_id}
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    counter.word_pronunciation_counts[w][p] += 1  # Add one smoothing
                for (w,) in words:
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    if w not in counter.word_pronunciation_counts:
                        continue
                    pron_counts = counter.word_pronunciation_counts[w]
                    max_value = max(pron_counts.values())
                    for p, c in pron_counts.items():
                        pron_mapping[(w, p)]["count"] = c
                        pron_mapping[(w, p)]["probability"] = format_probability(c / max_value)

                silence_count = sum(counter.silence_before_counts.values())
                non_silence_count = sum(counter.non_silence_before_counts.values())
                log_file.write(f"Total silence count was {silence_count}\n")
                log_file.write(f"Total non silence count was {non_silence_count}\n")
                silence_probability = silence_count / (silence_count + non_silence_count)
                silence_prob_sum += silence_probability
                silence_probabilities = {}
                for w, p, _ in pronunciations:
                    count = counter.silence_following_counts[(w, p)]
                    total_count = (
                        counter.silence_following_counts[(w, p)]
                        + counter.non_silence_following_counts[(w, p)]
                    )
                    pron_mapping[(w, p)]["silence_following_count"] = count
                    pron_mapping[(w, p)][
                        "non_silence_following_count"
                    ] = counter.non_silence_following_counts[(w, p)]
                    w_p_silence_count = count + (silence_probability * lambda_2)
                    prob = format_probability(w_p_silence_count / (total_count + lambda_2))
                    silence_probabilities[(w, p)] = prob
                    if w not in {initial_key[0], final_key[0], self.silence_word}:
                        pron_mapping[(w, p)]["silence_after_probability"] = prob
                lambda_3 = 2
                bar_count_silence_wp = collections.defaultdict(float)
                bar_count_non_silence_wp = collections.defaultdict(float)
                for (w_p1, w_p2), counts in counter.ngram_counts.items():
                    if w_p1 not in silence_probabilities:
                        silence_prob = 0.01
                    else:
                        silence_prob = silence_probabilities[w_p1]
                    total_count = counts["silence"] + counts["non_silence"]
                    bar_count_silence_wp[w_p2] += total_count * silence_prob
                    bar_count_non_silence_wp[w_p2] += total_count * (1 - silence_prob)
                for w, p, _ in pronunciations:
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    silence_count = counter.silence_before_counts[(w, p)]
                    non_silence_count = counter.non_silence_before_counts[(w, p)]
                    pron_mapping[(w, p)]["silence_before_correction"] = format_correction(
                        (silence_count + lambda_3) / (bar_count_silence_wp[(w, p)] + lambda_3)
                    )

                    pron_mapping[(w, p)]["non_silence_before_correction"] = format_correction(
                        (non_silence_count + lambda_3)
                        / (bar_count_non_silence_wp[(w, p)] + lambda_3)
                    )
                session.bulk_update_mappings(Pronunciation, pron_mapping.values())
                session.flush()
                initial_silence_count = counter.silence_before_counts[initial_key] + (
                    silence_probability * lambda_2
                )
                initial_non_silence_count = counter.non_silence_before_counts[initial_key] + (
                    (1 - silence_probability) * lambda_2
                )
                initial_silence_probability = format_probability(
                    initial_silence_count / (initial_silence_count + initial_non_silence_count)
                )

                final_silence_correction = format_correction(
                    (counter.silence_before_counts[final_key] + lambda_3)
                    / (bar_count_silence_wp[final_key] + lambda_3)
                )

                final_non_silence_correction = format_correction(
                    (counter.non_silence_before_counts[final_key] + lambda_3)
                    / (bar_count_non_silence_wp[final_key] + lambda_3)
                )
                initial_silence_prob_sum += initial_silence_probability
                final_silence_correction_sum += final_silence_correction
                final_non_silence_correction_sum += final_non_silence_correction
                dictionary_mappings.append(
                    {
                        "id": d_id,
                        "silence_probability": silence_probability,
                        "initial_silence_probability": initial_silence_probability,
                        "final_silence_correction": final_silence_correction,
                        "final_non_silence_correction": final_non_silence_correction,
                    }
                )
            self.silence_probability = silence_prob_sum / self.num_dictionaries
            self.initial_silence_probability = initial_silence_prob_sum / self.num_dictionaries
            self.final_silence_correction = final_silence_correction_sum / self.num_dictionaries
            self.final_non_silence_correction = (
                final_non_silence_correction_sum / self.num_dictionaries
            )
            bulk_update(session, Dictionary, dictionary_mappings)
            session.commit()
            rules: List[PhonologicalRule] = (
                session.query(PhonologicalRule)
                .options(
                    subqueryload(PhonologicalRule.pronunciations).joinedload(
                        RuleApplication.pronunciation, innerjoin=True
                    )
                )
                .all()
            )
            if rules:
                for r in rules:
                    base_count = 0
                    base_sil_after_count = 0
                    base_nonsil_after_count = 0

                    rule_count = 0
                    rule_sil_before_correction = 0
                    base_sil_before_correction = 0
                    rule_nonsil_before_correction = 0
                    base_nonsil_before_correction = 0
                    rule_sil_after_count = 0
                    rule_nonsil_after_count = 0
                    rule_correction_count = 0
                    base_correction_count = 0
                    non_application_query = session.query(Pronunciation).filter(
                        Pronunciation.pronunciation.regexp_match(r.match_regex.pattern),
                        Pronunciation.count > 1,
                    )
                    for p in non_application_query:
                        base_count += p.count

                        if p.silence_before_correction:
                            base_sil_before_correction += p.silence_before_correction
                            base_nonsil_before_correction += p.non_silence_before_correction
                            base_correction_count += 1

                        base_sil_after_count += (
                            p.silence_following_count if p.silence_following_count else 0
                        )
                        base_nonsil_after_count += (
                            p.non_silence_following_count if p.non_silence_following_count else 0
                        )

                    for p in r.pronunciations:
                        p = p.pronunciation
                        if p.count == 1:
                            continue
                        rule_count += p.count

                        if p.silence_before_correction:
                            rule_sil_before_correction += p.silence_before_correction
                            rule_nonsil_before_correction += p.non_silence_before_correction
                            rule_correction_count += 1

                        rule_sil_after_count += (
                            p.silence_following_count if p.silence_following_count else 0
                        )
                        rule_nonsil_after_count += (
                            p.non_silence_following_count if p.non_silence_following_count else 0
                        )
                    if not rule_count:
                        continue
                    r.probability = format_probability(rule_count / (rule_count + base_count))
                    if rule_correction_count:
                        rule_sil_before_correction = (
                            rule_sil_before_correction / rule_correction_count
                        )
                        rule_nonsil_before_correction = (
                            rule_nonsil_before_correction / rule_correction_count
                        )
                        base_sil_before_correction = (
                            base_sil_before_correction / base_correction_count
                        )
                        base_nonsil_before_correction = (
                            base_nonsil_before_correction / base_correction_count
                        )
                        r.silence_before_correction = format_correction(
                            base_sil_before_correction - rule_sil_before_correction
                        )
                        r.non_silence_before_correction = format_correction(
                            base_nonsil_before_correction - rule_nonsil_before_correction
                        )

                    r.silence_after_probability = format_probability(
                        (rule_sil_after_count + lambda_2)
                        / (rule_sil_after_count + rule_nonsil_after_count + lambda_2)
                    )

            session.commit()
        self.calculate_phonological_variant_probability()
        self.log_debug(f"Calculating pronunciation probabilities took {time.time() - begin}")

    def _collect_alignments(self, workflow: WorkflowType = WorkflowType.alignment) -> None:
        """
        Process alignment archives to extract word or phone alignments

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionFunction`
            Multiprocessing function for extracting alignments
        :meth:`.CorpusAligner.word_alignment_arguments`
            Arguments for word CTMs
        :meth:`.CorpusAligner.phone_alignment_arguments`
            Arguments for phone alignment
        """
        if self.has_alignments(workflow):
            return
        self.log_info(f"Collecting phone and word alignments from {workflow.name} lattices...")

        arguments = self.alignment_extraction_arguments(
            from_transcription=workflow is not WorkflowType.alignment
        )  # Phone CTM jobs
        with self.session() as session, tqdm.tqdm(
            total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            workflow_id = self.get_latest_workflow_run(workflow, session).id
            extraction_function = TranscriptionAlignmentExtractionFunction
            if workflow is WorkflowType.alignment:
                extraction_function = AlignmentExtractionFunction
            max_phone_interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
            if max_phone_interval_id is None:
                max_phone_interval_id = 0
            max_word_interval_id = session.query(sqlalchemy.func.max(WordInterval.id)).scalar()
            if max_word_interval_id is None:
                max_word_interval_id = 0
            phone_interval_mappings = []
            word_interval_mappings = []
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = extraction_function(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    utterance, word_intervals, phone_intervals, phone_word_mapping = result
                    new_phone_interval_mappings = []
                    new_word_interval_mappings = []
                    for interval in phone_intervals:
                        max_phone_interval_id += 1
                        new_phone_interval_mappings.append(
                            {
                                "id": max_phone_interval_id,
                                "begin": interval.begin,
                                "end": interval.end,
                                "phone_id": interval.label,
                                "utterance_id": utterance,
                                "workflow_id": workflow_id,
                                "phone_goodness": interval.confidence,
                            }
                        )
                    for interval in word_intervals:
                        max_word_interval_id += 1
                        new_word_interval_mappings.append(
                            {
                                "id": max_word_interval_id,
                                "begin": interval.begin,
                                "end": interval.end,
                                "word_id": interval.word_id,
                                "pronunciation_id": interval.pronunciation_id,
                                "utterance_id": utterance,
                                "workflow_id": workflow_id,
                            }
                        )
                    for i, index in enumerate(phone_word_mapping):
                        new_phone_interval_mappings[i][
                            "word_interval_id"
                        ] = new_word_interval_mappings[index]["id"]
                    phone_interval_mappings.extend(new_phone_interval_mappings)
                    word_interval_mappings.extend(new_word_interval_mappings)
                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = extraction_function(args)
                    for (
                        utterance,
                        word_intervals,
                        phone_intervals,
                        phone_word_mapping,
                    ) in function.run():
                        new_phone_interval_mappings = []
                        new_word_interval_mappings = []
                        for interval in phone_intervals:
                            max_phone_interval_id += 1
                            new_phone_interval_mappings.append(
                                {
                                    "id": max_phone_interval_id,
                                    "begin": interval.begin,
                                    "end": interval.end,
                                    "phone_id": interval.label,
                                    "utterance_id": utterance,
                                    "workflow_id": workflow_id,
                                    "phone_goodness": interval.confidence,
                                }
                            )
                        for interval in word_intervals:
                            max_word_interval_id += 1
                            new_word_interval_mappings.append(
                                {
                                    "id": max_word_interval_id,
                                    "begin": interval.begin,
                                    "end": interval.end,
                                    "word_id": interval.word_id,
                                    "pronunciation_id": interval.pronunciation_id,
                                    "utterance_id": utterance,
                                    "workflow_id": workflow_id,
                                }
                            )
                        for i, index in enumerate(phone_word_mapping):
                            new_phone_interval_mappings[i][
                                "word_interval_id"
                            ] = new_word_interval_mappings[index]["id"]
                        phone_interval_mappings.extend(new_phone_interval_mappings)
                        word_interval_mappings.extend(new_word_interval_mappings)

                        pbar.update(1)
            self.alignment_checks[workflow] = True
            session.bulk_insert_mappings(
                PhoneInterval, phone_interval_mappings, return_defaults=False, render_nulls=True
            )
            if word_interval_mappings:
                session.bulk_insert_mappings(
                    WordInterval, word_interval_mappings, return_defaults=False, render_nulls=True
                )
            session.commit()

    def fine_tune_alignments(self, workflow: WorkflowType) -> None:
        """
        Fine tune aligned boundaries to millisecond precision

        Parameters
        ----------
        workflow: :class:`~montreal_forced_aligner.data.WorkflowType`
            Workflow type to fine tune

        """
        self.log_info("Fine tuning alignments...")
        begin = time.time()
        with self.session() as session:

            with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                update_mappings = []
                workflow_id = self.get_latest_workflow_run(workflow, session).id
                arguments = self.fine_tune_arguments(workflow_id)
                if GLOBAL_CONFIG.use_mp:
                    error_dict = {}
                    return_queue = mp.Queue()
                    stopped = Stopped()
                    procs = []
                    for i, args in enumerate(arguments):
                        function = FineTuneFunction(args)
                        p = KaldiProcessWorker(i, return_queue, function, stopped)
                        procs.append(p)
                        p.start()
                    while True:
                        try:
                            result = return_queue.get(timeout=1)
                            if isinstance(result, Exception):
                                error_dict[getattr(result, "job_name", 0)] = result
                                continue
                            if stopped.stop_check():
                                continue
                        except Empty:
                            for proc in procs:
                                if not proc.finished.stop_check():
                                    break
                            else:
                                break
                            continue

                        update_mappings.extend(result[0])
                        update_mappings.extend(
                            [{"id": x, "begin": 0, "end": 0, "label": ""} for x in result[1]]
                        )
                        pbar.update(1)
                    for p in procs:
                        p.join()

                    if error_dict:
                        for v in error_dict.values():
                            raise v

                else:
                    self.log_debug("Not using multiprocessing...")
                    for args in arguments:
                        function = FineTuneFunction(args)
                        for result in function.run():
                            update_mappings.extend(result[0])
                            update_mappings.extend(
                                [{"id": x, "begin": 0, "end": 0, "label": ""} for x in result[1]]
                            )
                            pbar.update(1)
                bulk_update(session, PhoneInterval, update_mappings)
                session.flush()
                session.execute(PhoneInterval.__table__.delete().where(PhoneInterval.end == 0))
                session.flush()
                word_update_mappings = []
                word_intervals = (
                    session.query(
                        WordInterval.id,
                        sqlalchemy.func.min(PhoneInterval.begin),
                        sqlalchemy.func.min(PhoneInterval.end),
                    )
                    .join(PhoneInterval.word_interval)
                    .group_by(WordInterval.id)
                )
                for wi_id, begin, end in word_intervals:
                    word_update_mappings.append({"id": wi_id, "begin": begin, "end": end})
                bulk_update(session, WordInterval, word_update_mappings)
                session.commit()
            self.export_frame_shift = round(self.export_frame_shift / 10, 4)
            self.log_debug(f"Fine tuning alignments took {time.time() - begin}")

    def fine_tune_arguments(self, workflow_id: int) -> List[FineTuneArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneArguments`]
            Arguments for processing
        """
        args = []
        for j in self.jobs:
            if not j.has_data:
                continue
            log_path = os.path.join(self.working_log_directory, f"fine_tune.{j.name}.log")
            fmllr_paths = None
            if not getattr(self, "speaker_independent", True):
                fmllr_paths = j.construct_path_dictionary(self.working_directory, "trans", "ark")
            cmvn_paths = j.construct_path_dictionary(self.data_directory, "cmvn", "scp")
            lda_mat_path = os.path.join(self.working_directory, "lda.mat")
            if not os.path.exists(lda_mat_path):
                lda_mat_path = None
            args.append(
                FineTuneArguments(
                    j.name,
                    getattr(self, "read_only_db_string", ""),
                    log_path,
                    self.working_directory,
                    self.phone_symbol_table_path,
                    self.disambiguation_symbols_int_path,
                    self.tree_path,
                    self.model_path,
                    self.frame_shift,
                    cmvn_paths,
                    fmllr_paths,
                    lda_mat_path,
                    self.mfcc_options,
                    self.pitch_options,
                    self.lda_options,
                    self.align_options,
                    workflow_id,
                    self.position_dependent_phones,
                    self.kaldi_grouped_phones,
                )
            )
        return args

    def collect_alignments(self, workflow: WorkflowType = WorkflowType.alignment) -> None:
        """
        Collect word and phone alignments from alignment archives
        """
        if self.has_alignments(workflow):
            if self.export_output_directory is not None:
                self.export_textgrids(workflow=workflow)
            return
        self._collect_alignments(workflow)

    def get_tree_info(self) -> typing.Tuple[int, int]:
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
            [thirdparty_binary("tree-info"), self.tree_path],
            encoding="utf8",
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

    def export_textgrids(
        self,
        output_format: str = TextFileType.TEXTGRID.value,
        include_original_text: bool = False,
        workflow: WorkflowType = WorkflowType.alignment,
    ) -> None:
        """
        Exports alignments to TextGrid files

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`
            Multiprocessing helper function for TextGrid export
        :meth:`.CorpusAligner.export_textgrid_arguments`
            Job method for TextGrid export

        Parameters
        ----------
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        """
        if not self.has_alignments(workflow):
            self._collect_alignments(workflow)
        begin = time.time()
        error_dict = {}

        with tqdm.tqdm(total=self.num_files, disable=GLOBAL_CONFIG.quiet) as pbar:
            with self.session() as session:
                workflow_id = self.get_latest_workflow_run(workflow, session).id
                files = (
                    session.query(
                        File.id,
                        File.name,
                        File.relative_path,
                        SoundFile.duration,
                        TextFile.text_file_path,
                    )
                    .join(File.sound_file)
                    .join(File.text_file)
                )
                if GLOBAL_CONFIG.use_mp and GLOBAL_CONFIG.num_jobs > 1:
                    stopped = Stopped()

                    finished_adding = Stopped()
                    for_write_queue = mp.Queue()
                    return_queue = mp.Queue()
                    export_args = self.export_textgrid_arguments(
                        output_format, workflow_id, include_original_text
                    )
                    exported_file_count = Counter()
                    export_procs = []
                    self.db_engine.dispose()
                    for j in self.jobs:
                        if not j.has_data:
                            continue
                        export_proc = ExportTextGridProcessWorker(
                            self.read_only_db_string,
                            for_write_queue,
                            return_queue,
                            stopped,
                            finished_adding,
                            export_args[j.name],
                            exported_file_count,
                        )
                        export_proc.start()
                        export_procs.append(export_proc)
                    try:
                        for args in files:
                            for_write_queue.put(args)
                        time.sleep(1)
                        finished_adding.stop()
                        while True:
                            try:
                                result = return_queue.get(timeout=1)
                                if isinstance(result, AlignmentExportError):
                                    error_dict[getattr(result, "path", 0)] = result
                                    continue
                                if self.stopped.stop_check():
                                    continue
                            except Empty:
                                for proc in export_procs:
                                    if not proc.finished_processing.stop_check():
                                        break
                                else:
                                    break
                                continue
                            if isinstance(result, int):
                                pbar.update(1)
                    except Exception:
                        stopped.stop()
                        raise
                    finally:
                        for i in range(GLOBAL_CONFIG.num_jobs):
                            if not self.jobs[i].has_data:
                                continue
                            export_procs[i].join()
                else:
                    self.log_debug("Not using multiprocessing for TextGrid export")

                    for file_id, name, relative_path, duration, text_file_path in files:
                        output_path = construct_output_path(
                            name,
                            relative_path,
                            self.export_output_directory,
                            text_file_path,
                            output_format,
                        )

                        data = construct_output_tiers(
                            session,
                            file_id,
                            workflow_id,
                            GLOBAL_CONFIG.cleanup_textgrids,
                            self.clitic_marker,
                            include_original_text,
                        )
                        export_textgrid(
                            data,
                            output_path,
                            duration,
                            self.export_frame_shift,
                            output_format=output_format,
                        )
                        pbar.update(1)

        if error_dict:
            self.log_warning(
                f"There were {len(error_dict)} errors encountered in generating TextGrids. "
                f"Check {os.path.join(self.export_output_directory, 'output_errors.txt')} "
                f"for more details"
            )
        output_textgrid_writing_errors(self.export_output_directory, error_dict)
        self.log_info(f"Finished exporting TextGrids to {self.export_output_directory}!")
        self.log_debug(f"Exported TextGrids in a total of {time.time() - begin} seconds")

    def export_files(
        self,
        output_directory: str,
        output_format: Optional[str] = None,
        include_original_text: bool = False,
        workflow: WorkflowType = WorkflowType.alignment,
    ) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        include_original_text: bool
            Flag for including the original text of the corpus files as a tier
        workflow: :class:`~montreal_forced_aligner.data.WorkflowType`
            Workflow to use when exporting files
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        self.export_output_directory = output_directory

        self.log_info(f"Exporting {workflow.name} TextGrids to {self.export_output_directory}...")
        os.makedirs(self.export_output_directory, exist_ok=True)
        self.export_textgrids(output_format, include_original_text, workflow)

    def evaluate_alignments(
        self,
        mapping: Optional[Dict[str, str]] = None,
        output_directory: Optional[str] = None,
        comparison_source=WorkflowType.alignment,
        reference_source=WorkflowType.reference,
    ) -> None:
        """
        Evaluate alignments against a reference directory

        Parameters
        ----------
        mapping: dict[str, Union[str, list[str]]], optional
            Mapping between phones that should be considered equal across different phone set types
        output_directory: str, optional
            Directory to save results, if not specified, it will be saved in the log directory
        comparison_source: :class:`~montreal_forced_aligner.data.WorkflowType`
            Workflow to compare to the reference intervals, defaults to :attr:`~montreal_forced_aligner.data.WorkflowType.alignment`
        comparison_source: :class:`~montreal_forced_aligner.data.WorkflowType`
            Workflow to use as the reference intervals, defaults to :attr:`~montreal_forced_aligner.data.WorkflowType.reference`
        """
        from montreal_forced_aligner.config import GLOBAL_CONFIG

        begin = time.time()
        if output_directory:
            csv_path = os.path.join(
                output_directory,
                f"{comparison_source.name}_{reference_source.name}_evaluation.csv",
            )
        else:
            csv_path = os.path.join(
                self.working_log_directory,
                f"{comparison_source.name}_{reference_source.name}_evaluation.csv",
            )
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
        with self.session() as session:
            # Set up
            self.log_info("Evaluating alignments...")
            self.log_debug(f"Mapping: {mapping}")
            reference_workflow_id = self.get_latest_workflow_run(reference_source, session).id
            comparison_workflow_id = self.get_latest_workflow_run(comparison_source, session).id
            update_mappings = []
            indices = []
            to_comp = []
            score_func = functools.partial(
                align_phones,
                silence_phone=self.optional_silence_phone,
                custom_mapping=mapping,
            )
            unaligned_utts = []
            utterances = session.query(Utterance).options(
                joinedload(Utterance.file, innerjoin=True),
                joinedload(Utterance.speaker, innerjoin=True),
                subqueryload(Utterance.phone_intervals).options(
                    joinedload(PhoneInterval.phone, innerjoin=True),
                    joinedload(PhoneInterval.workflow, innerjoin=True),
                ),
                subqueryload(Utterance.word_intervals).options(
                    joinedload(WordInterval.word, innerjoin=True),
                    joinedload(WordInterval.workflow, innerjoin=True),
                ),
            )
            reference_phone_counts = {}
            for u in utterances:
                reference_phones = u.phone_intervals_for_workflow(reference_workflow_id)
                comparison_phones = u.phone_intervals_for_workflow(comparison_workflow_id)
                reference_phone_counts[u.id] = len(reference_phones)
                if not reference_phone_counts[u.id]:
                    continue
                if not comparison_phones:  # couldn't be aligned
                    phone_error_rate = reference_phone_counts[u.id]
                    unaligned_utts.append(u)
                    update_mappings.append(
                        {
                            "id": u.id,
                            "alignment_score": None,
                            "phone_error_rate": phone_error_rate,
                        }
                    )
                    continue
                indices.append(u)
                to_comp.append((reference_phones, comparison_phones))
            with mp.Pool(GLOBAL_CONFIG.num_jobs) as pool:
                gen = pool.starmap(score_func, to_comp)
                for i, (score, phone_error_rate) in enumerate(gen):
                    if score is None:
                        continue
                    u = indices[i]
                    reference_phone_count = reference_phone_counts[u.id]
                    update_mappings.append(
                        {
                            "id": u.id,
                            "alignment_score": score,
                            "phone_error_rate": phone_error_rate,
                        }
                    )
                    score_count += 1
                    score_sum += score
                    phone_edit_sum += int(phone_error_rate * reference_phone_count)
                    phone_length_sum += reference_phone_count
            bulk_update(session, Utterance, update_mappings)
            self.alignment_evaluation_done = True
            session.query(Corpus).update({Corpus.alignment_evaluation_done: True})
            session.commit()
            self.log_info("Exporting evaluation...")
            with mfa_open(csv_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
                utterances = (
                    session.query(
                        File.name,
                        Utterance.begin,
                        Utterance.end,
                        Speaker.name,
                        Utterance.duration,
                        Utterance.normalized_text,
                        Utterance.oovs,
                        Utterance.alignment_score,
                        Utterance.phone_error_rate,
                        Utterance.alignment_log_likelihood,
                    )
                    .join(Utterance.speaker)
                    .join(Utterance.file)
                    .join(Utterance.phone_intervals)
                    .where(PhoneInterval.workflow_id == reference_workflow_id)
                )
                for (
                    file_name,
                    begin,
                    end,
                    speaker_name,
                    duration,
                    normalized_text,
                    oovs,
                    alignment_score,
                    phone_error_rate,
                    alignment_log_likelihood,
                ) in utterances:
                    data = {
                        "file": file_name,
                        "begin": begin,
                        "end": end,
                        "speaker": speaker_name,
                        "duration": duration,
                        "normalized_text": normalized_text,
                        "oovs": oovs,
                        "reference_phone_count": reference_phone_counts[u.id],
                        "alignment_score": alignment_score,
                        "phone_error_rate": phone_error_rate,
                        "alignment_log_likelihood": alignment_log_likelihood,
                    }
                    data["word_count"] = len(data["normalized_text"].split())
                    data["oov_count"] = len(data["oovs"].split())
                    if alignment_score is not None:
                        score_count += 1
                        score_sum += alignment_score
                    writer.writerow(data)
        self.log_info(f"Average overlap score: {score_sum/score_count}")
        self.log_info(f"Average phone error rate: {phone_edit_sum/phone_length_sum}")
        self.log_debug(f"Alignment evaluation took {time.time()-begin} seconds")
