"""Class definitions for base aligner"""
from __future__ import annotations

import collections
import csv
import functools
import multiprocessing as mp
import os
import time
from queue import Empty
from typing import Dict, List, Optional

import sqlalchemy
import tqdm
from sqlalchemy.orm import joinedload, selectinload, subqueryload

from montreal_forced_aligner.abc import FileExporterMixin
from montreal_forced_aligner.alignment.mixins import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    AlignmentExtractionArguments,
    AlignmentExtractionFunction,
    ExportTextGridArguments,
    ExportTextGridProcessWorker,
    GeneratePronunciationsArguments,
    GeneratePronunciationsFunction,
    construct_output_path,
)
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.data import CtmInterval, PronunciationProbabilityCounter, TextFileType
from montreal_forced_aligner.db import (
    Corpus,
    DictBundle,
    Dictionary,
    File,
    PhoneInterval,
    Pronunciation,
    SoundFile,
    Speaker,
    TextFile,
    Utterance,
    Word,
    WordInterval,
    WordType,
)
from montreal_forced_aligner.exceptions import AlignmentExportError
from montreal_forced_aligner.helper import align_phones
from montreal_forced_aligner.textgrid import export_textgrid, output_textgrid_writing_errors
from montreal_forced_aligner.utils import Counter, KaldiProcessWorker, Stopped

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.export_output_directory = None

    def alignment_extraction_arguments(self) -> List[AlignmentExtractionArguments]:
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
            arguments.append(
                AlignmentExtractionArguments(
                    j.name,
                    getattr(self, "db_path", ""),
                    os.path.join(self.working_log_directory, f"get_phone_ctm.{j.name}.log"),
                    self.alignment_model_path,
                    round(self.frame_shift / 1000, 4),
                    self.cleanup_textgrids,
                    j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                    j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                )
            )

        return arguments

    def export_textgrid_arguments(
        self, output_format: str, include_original_text: bool = False
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
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"export_textgrids.{j.name}.log"),
                self.frame_shift,
                self.export_output_directory,
                output_format,
                include_original_text,
            )
            for j in self.jobs
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
                getattr(self, "db_path", ""),
                os.path.join(self.working_log_directory, f"generate_pronunciations.{j.name}.log"),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.model_path,
                False,
            )
            for j in self.jobs
        ]

    def compute_pronunciation_probabilities(self, compute_silence_probabilities=True):
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

        def format_probability(probability_value: float) -> float:
            """Format a probability to have two decimal places and be between 0.01 and 0.99"""
            return min(max(round(probability_value, 2), 0.01), 0.99)

        def format_correction(correction_value: float) -> float:
            """Format a probability correction value to have two decimal places and be  greater than 0.01"""
            correction_value = round(correction_value, 2)
            if correction_value == 0:
                correction_value = 0.01
            return correction_value

        begin = time.time()
        dictionary_counters = {
            dict_id: PronunciationProbabilityCounter()
            for dict_id in self.dictionary_lookup.values()
        }
        self.log_info("Generating pronunciations...")
        arguments = self.generate_pronunciations_arguments()
        with tqdm.tqdm(
            total=self.num_current_utterances, disable=getattr(self, "quiet", False)
        ) as pbar:
            if self.use_mp:
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
        with open(
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
                    pron_counts = counter.word_pronunciation_counts[w]
                    max_value = max(pron_counts.values())
                    for p, c in pron_counts.items():
                        pron_mapping[(w, p)]["probability"] = format_probability(c / max_value)

                if not compute_silence_probabilities:
                    log_file.write("Skipping silence calculations")
                    continue
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
            if compute_silence_probabilities:
                self.silence_probability = silence_prob_sum / self.num_dictionaries
                self.initial_silence_probability = initial_silence_prob_sum / self.num_dictionaries
                self.final_silence_correction = (
                    final_silence_correction_sum / self.num_dictionaries
                )
                self.final_non_silence_correction = (
                    final_non_silence_correction_sum / self.num_dictionaries
                )
            session.bulk_update_mappings(Dictionary, dictionary_mappings)
            session.commit()
        self.log_debug(f"Calculating pronunciation probabilities took {time.time() - begin}")

    def _collect_alignments(self):
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
        self.log_info("Collecting phone and word alignments from alignment lattices...")
        jobs = self.alignment_extraction_arguments()  # Phone CTM jobs
        with self.session() as session, tqdm.tqdm(
            total=self.num_current_utterances, disable=getattr(self, "quiet", False)
        ) as pbar:
            phone_interval_mappings = []
            word_interval_mappings = []
            if self.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(jobs):
                    function = AlignmentExtractionFunction(args)
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
                    utterance, word_intervals, phone_intervals = result
                    for interval in phone_intervals:
                        phone_interval_mappings.append(
                            {
                                "begin": interval.begin,
                                "end": interval.end,
                                "label": interval.label,
                                "utterance_id": utterance,
                            }
                        )
                    for interval in word_intervals:
                        word_interval_mappings.append(
                            {
                                "begin": interval.begin,
                                "end": interval.end,
                                "label": interval.label,
                                "utterance_id": utterance,
                            }
                        )

                    pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in jobs:
                    function = AlignmentExtractionFunction(args)
                    for utterance, word_intervals, phone_intervals in function.run():

                        for interval in phone_intervals:
                            phone_interval_mappings.append(
                                {
                                    "begin": interval.begin,
                                    "end": interval.end,
                                    "label": interval.label,
                                    "utterance_id": utterance,
                                }
                            )
                        for interval in word_intervals:
                            word_interval_mappings.append(
                                {
                                    "begin": interval.begin,
                                    "end": interval.end,
                                    "label": interval.label,
                                    "utterance_id": utterance,
                                }
                            )

                        pbar.update(1)
            self.alignment_done = True
            session.query(Corpus).update({"alignment_done": True})
            session.bulk_insert_mappings(
                PhoneInterval, phone_interval_mappings, return_defaults=False, render_nulls=True
            )
            session.bulk_insert_mappings(
                WordInterval, word_interval_mappings, return_defaults=False, render_nulls=True
            )
            session.commit()

    def collect_alignments(self) -> None:
        """
        Collect word and phone alignments from alignment archives
        """
        if self.alignment_done:
            if self.export_output_directory is not None:
                self.export_textgrids()
            return
        self._collect_alignments()

    def export_textgrids(
        self, output_format: str = TextFileType.TEXTGRID.value, include_original_text: bool = False
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
        if not self.alignment_done:
            self._collect_alignments()
        begin = time.time()
        error_dict = {}

        with tqdm.tqdm(total=self.num_files, disable=getattr(self, "quiet", False)) as pbar:

            with self.session() as session:
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
                if self.use_mp:
                    stopped = Stopped()

                    finished_adding = Stopped()
                    for_write_queue = mp.Queue()
                    return_queue = mp.Queue()
                    export_args = self.export_textgrid_arguments(
                        output_format, include_original_text
                    )
                    exported_file_count = Counter()
                    export_procs = []
                    self.db_engine.dispose()
                    for j in self.jobs:
                        export_proc = ExportTextGridProcessWorker(
                            self.db_path,
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

                        for i in range(self.num_jobs):
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
                        utterances = (
                            session.query(Utterance)
                            .options(
                                joinedload(Utterance.speaker, innerjoin=True).load_only(
                                    Speaker.name
                                ),
                                selectinload(Utterance.phone_intervals),
                                selectinload(Utterance.word_intervals),
                            )
                            .filter(Utterance.file_id == file_id)
                        )
                        data = {}
                        for utt in utterances:
                            if utt.speaker.name not in data:
                                data[utt.speaker.name] = {"words": [], "phones": []}
                            for wi in utt.word_intervals:
                                data[utt.speaker.name]["words"].append(
                                    CtmInterval(wi.begin, wi.end, wi.label, utt.id)
                                )

                            for pi in utt.phone_intervals:
                                data[utt.speaker.name]["phones"].append(
                                    CtmInterval(pi.begin, pi.end, pi.label, utt.id)
                                )
                        export_textgrid(data, output_path, duration, self.frame_shift)
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
    ) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        self.export_output_directory = output_directory

        self.log_info(f"Exporting TextGrids to {self.export_output_directory}...")
        os.makedirs(self.export_output_directory, exist_ok=True)
        self.export_textgrids(output_format, include_original_text)

    def evaluate_alignments(
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
