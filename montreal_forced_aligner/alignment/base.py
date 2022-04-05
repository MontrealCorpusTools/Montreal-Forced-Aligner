"""Class definitions for base aligner"""
from __future__ import annotations

import collections
import multiprocessing as mp
import os
import shutil
import time
from queue import Empty
from typing import List, Optional

import tqdm
from sqlalchemy.orm import Session, joinedload

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
    construct_output_tiers,
)
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.corpus.db import Corpus, File, PhoneInterval, WordInterval
from montreal_forced_aligner.data import PronunciationProbabilityCounter, TextFileType
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

    def export_textgrid_arguments(self) -> List[ExportTextGridArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`

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
                self.backup_output_directory,
            )
            for j in self.jobs
        ]

    @property
    def backup_output_directory(self) -> Optional[str]:
        """Backup directory if overwriting is not allowed"""
        return None

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

        def format_probability(probability_value):
            return min(max(round(probability_value, 2), 0.01), 0.99)

        def format_correction(correction_value):
            correction_value = round(correction_value, 2)
            if correction_value == 0:
                correction_value = 0.01
            return correction_value

        begin = time.time()
        dictionary_counters = {
            dict_name: PronunciationProbabilityCounter()
            for dict_name in self.dictionary_mapping.keys()
        }
        self.log_info("Generating pronunciations...")
        arguments = self.generate_pronunciations_arguments()
        with tqdm.tqdm(
            total=self.num_current_utterances, disable=getattr(self, "quiet", False)
        ) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = GeneratePronunciationsFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        dict_name, utterance_counter = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    dictionary_counters[dict_name].add_counts(utterance_counter)
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
                    for dict_name, utterance_counter in function.run():
                        dictionary_counters[dict_name].add_counts(utterance_counter)
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
        ) as log_file:
            for dict_name, counter in dictionary_counters.items():
                dictionary = self.dictionary_mapping[dict_name]
                log_file.write(f"For {dict_name}:\n")
                floored_pronunciations = []
                for w, pron_counts in counter.word_pronunciation_counts.items():
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    for p in dictionary.words[w]:  # Add one smoothing
                        pron_counts[p.pronunciation] += 1
                    max_value = max(pron_counts.values())
                    log_file.write(f"{w} max value: {max_value}\n")
                    for p in dictionary.words[w]:
                        c = pron_counts[p.pronunciation]
                        p.probability = format_probability(c / max_value)
                        if p.probability == 0.01:
                            floored_pronunciations.append((w, p.pronunciation))
                self.log_debug(
                    f"The following word/pronunciation combos had near zero probability "
                    f"and could likely be removed from {dict_name}:"
                )
                self.log_debug(
                    "  " + ", ".join(f'{w} /{" ".join(p)}/' for w, p in floored_pronunciations)
                )
                if not compute_silence_probabilities:
                    log_file.write("Skipping silence calculations")
                    continue
                silence_count = sum(counter.silence_before_counts.values())
                non_silence_count = sum(counter.non_silence_before_counts.values())
                log_file.write(f"Total silence count was {silence_count}\n")
                log_file.write(f"Total non silence count was {non_silence_count}\n")
                dictionary.silence_probability = silence_count / (
                    silence_count + non_silence_count
                )
                silence_prob_sum += dictionary.silence_probability
                silence_probabilities = {}
                for w_p, count in counter.silence_following_counts.items():
                    w_p_silence_count = count + (dictionary.silence_probability * lambda_2)
                    w_p_non_silence_count = counter.non_silence_following_counts[w_p] + (
                        (1 - dictionary.silence_probability) * lambda_2
                    )
                    prob = format_probability(
                        w_p_silence_count / (w_p_silence_count + w_p_non_silence_count)
                    )
                    silence_probabilities[w_p] = prob
                    if w_p[0] not in {initial_key[0], final_key[0], self.silence_word}:
                        pron = dictionary.words[w_p[0]][w_p[1]]
                        pron.silence_after_probability = prob
                        log_file.write(
                            f"{w_p[0], w_p[1]} silence after: {w_p_silence_count}\tnon silence after: {w_p_non_silence_count}\n"
                        )
                        log_file.write(
                            f"{w_p[0], w_p[1]} silence after prob: {pron.silence_after_probability}\n"
                        )
                lambda_3 = 2
                bar_count_silence_wp = collections.defaultdict(float)
                bar_count_non_silence_wp = collections.defaultdict(float)
                for (w_p1, w_p2), counts in counter.ngram_counts.items():
                    if w_p1 not in silence_probabilities:
                        silence_prob = 0.01
                    else:
                        silence_prob = silence_probabilities[w_p1]
                    bar_count_silence_wp[w_p2] += counts["silence"] * silence_prob
                    bar_count_non_silence_wp[w_p2] += counts["non_silence"] * (1 - silence_prob)
                for w_p, silence_count in counter.silence_before_counts.items():
                    if w_p[0] in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    non_silence_count = counter.non_silence_before_counts[w_p]
                    pron = dictionary.words[w_p[0]][w_p[1]]
                    pron.silence_before_correction = format_correction(
                        (silence_count + lambda_3) / (bar_count_silence_wp[w_p] + lambda_3)
                    )

                    pron.non_silence_before_correction = format_correction(
                        (non_silence_count + lambda_3) / (bar_count_non_silence_wp[w_p] + lambda_3)
                    )
                    log_file.write(
                        f"{w_p[0], w_p[1]} silence count: {silence_count}\tnon silence count: {non_silence_count}\n"
                    )
                    log_file.write(
                        f"{w_p[0], w_p[1]} silence count: {pron.silence_before_correction}\tnon silence before correction: {pron.non_silence_before_correction}\n"
                    )
                initial_silence_count = counter.silence_before_counts[initial_key] + (
                    dictionary.silence_probability * lambda_2
                )
                initial_non_silence_count = counter.non_silence_before_counts[initial_key] + (
                    (1 - dictionary.silence_probability) * lambda_2
                )
                dictionary.initial_silence_probability = format_probability(
                    initial_silence_count / (initial_silence_count + initial_non_silence_count)
                )

                dictionary.final_silence_correction = format_correction(
                    (counter.silence_before_counts[final_key] + lambda_3)
                    / (bar_count_silence_wp[final_key] + lambda_3)
                )

                dictionary.final_non_silence_correction = format_correction(
                    (counter.non_silence_before_counts[final_key] + lambda_3)
                    / (bar_count_non_silence_wp[final_key] + lambda_3)
                )
                initial_silence_prob_sum += dictionary.initial_silence_probability
                final_silence_correction_sum += dictionary.final_silence_correction
                final_non_silence_correction_sum += dictionary.final_non_silence_correction
        if compute_silence_probabilities:
            self.silence_probability = silence_prob_sum / len(self.dictionary_mapping)
            self.initial_silence_probability = initial_silence_prob_sum / len(
                self.dictionary_mapping
            )
            self.final_silence_correction = final_silence_correction_sum / len(
                self.dictionary_mapping
            )
            self.final_non_silence_correction = final_non_silence_correction_sum / len(
                self.dictionary_mapping
            )
        self.log_debug(f"Alignment round took {time.time() - begin}")

    def _collect_alignments(self):
        """
        Process alignment archives to extract word or phone alignments

        Parameters
        ----------
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids', 'short_textgrids' (the default), or 'json', passed to praatio

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
        with Session(self.db_engine) as session, tqdm.tqdm(
            total=self.num_current_utterances, disable=getattr(self, "quiet", False)
        ) as pbar:
            phone_interval_mappings = []
            word_interval_mappings = []
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(jobs):
                    function = AlignmentExtractionFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        utterance, word_intervals, phone_intervals = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue

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
            session.bulk_insert_mappings(PhoneInterval, phone_interval_mappings)
            session.bulk_insert_mappings(WordInterval, word_interval_mappings)
            session.commit()
        self.alignment_done = True
        with self.session() as session:
            session.query(Corpus).update({"alignment_done": True})
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

    def export_textgrids(self, output_format: str = TextFileType.TEXTGRID.value) -> None:
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
            Format to save alignments, one of 'long_textgrids', 'short_textgrids' (the default), or 'json', passed to praatio
        """
        if not self.alignment_done:
            self._collect_alignments()
        begin = time.time()
        export_errors = {}

        with tqdm.tqdm(total=self.num_files, disable=getattr(self, "quiet", False)) as pbar:

            with Session(self.db_engine) as session:
                files = session.query(File).options(joinedload(File.sound_file))
                if self.use_mp:
                    manager = mp.Manager()
                    textgrid_errors = manager.dict()
                    stopped = Stopped()

                    finished_processing = Stopped()
                    for_write_queue = mp.JoinableQueue()
                    export_args = self.export_textgrid_arguments()
                    exported_file_count = Counter()
                    export_procs = []
                    self.db_engine.dispose()
                    for j in self.jobs:
                        export_proc = ExportTextGridProcessWorker(
                            self.db_path,
                            for_write_queue,
                            stopped,
                            finished_processing,
                            textgrid_errors,
                            output_format,
                            self.export_output_directory,
                            export_args[j.name],
                            exported_file_count,
                        )
                        export_proc.start()
                        export_procs.append(export_proc)
                    try:
                        for f in files:
                            for_write_queue.put(
                                (f.id, f.name, f.relative_path, f.sound_file.duration)
                            )
                        last_value = 0
                        while exported_file_count.value() < self.num_files - 2:
                            new_value = exported_file_count.value()
                            if new_value != last_value:
                                pbar.update(new_value - last_value)
                                last_value = new_value
                            time.sleep(5)
                    except Exception:
                        stopped.stop()
                        raise
                    finally:
                        finished_processing.stop()

                        for_write_queue.join()
                        for i in range(self.num_jobs):
                            export_procs[i].join()
                        export_errors.update(textgrid_errors)
                else:
                    self.log_debug("Not using multiprocessing for TextGrid export")

                    for f in files:
                        output_path = construct_output_path(
                            f.name, f.relative_path, self.export_output_directory
                        )
                        data = construct_output_tiers(session, f.id)
                        export_textgrid(data, output_path, f.sound_file.duration, self.frame_shift)
                        pbar.update(1)

        if export_errors:
            self.log_warning(
                f"There were {len(export_errors)} errors encountered in generating TextGrids. "
                f"Check {os.path.join(self.export_output_directory, 'output_errors.txt')} "
                f"for more details"
            )
        output_textgrid_writing_errors(self.export_output_directory, export_errors)
        self.log_info(f"Finished exporting TextGrids to {self.export_output_directory}!")
        self.log_debug(f"Exported TextGrids in a total of {time.time() - begin} seconds")

    def export_files(self, output_directory: str, output_format: Optional[str] = None) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        self.export_output_directory = output_directory
        if self.backup_output_directory and os.path.exists(self.backup_output_directory):
            shutil.rmtree(self.backup_output_directory, ignore_errors=True)
        if os.path.exists(self.export_output_directory) and not self.overwrite:
            self.export_output_directory = self.backup_output_directory
            self.log_debug(
                f"Not overwriting existing directory, exporting to {self.export_output_directory}"
            )

        self.log_info(f"Exporting TextGrids to {self.export_output_directory}...")
        os.makedirs(self.export_output_directory, exist_ok=True)
        self.export_textgrids(output_format)
