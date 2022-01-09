"""Class definitions for base aligner"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import time
from queue import Empty
from typing import List, Optional

import tqdm

from montreal_forced_aligner.abc import FileExporterMixin
from montreal_forced_aligner.alignment.mixins import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    ExportTextGridArguments,
    ExportTextGridProcessWorker,
    PhoneAlignmentArguments,
    PhoneAlignmentFunction,
    WordAlignmentArguments,
    WordAlignmentFunction,
)
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.textgrid import export_textgrid, output_textgrid_writing_errors
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped

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

    def word_alignment_arguments(self) -> List[WordAlignmentArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmArguments`]
            Arguments for processing
        """
        return [
            WordAlignmentArguments(
                os.path.join(self.working_log_directory, f"get_word_ctm.{j.name}.log"),
                self.alignment_model_path,
                round(self.frame_shift / 1000, 4),
                self.cleanup_textgrids,
                self.oov_word,
                self.sanitize_function,
                j.current_dictionary_names,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.word_boundary_int_files(),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                {d.name: d.reversed_word_mapping for d in self.dictionary_mapping.values()},
                j.text_scp_data(),
                j.utt2spk_scp_data(),
            )
            for j in self.jobs
        ]

    def phone_alignment_arguments(self) -> List[PhoneAlignmentArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmArguments`]
            Arguments for processing
        """
        return [
            PhoneAlignmentArguments(
                os.path.join(self.working_log_directory, f"get_phone_ctm.{j.name}.log"),
                self.alignment_model_path,
                round(self.frame_shift / 1000, 4),
                self.position_dependent_phones,
                self.cleanup_textgrids,
                self.optional_silence_phone,
                j.current_dictionary_names,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.word_boundary_int_files(),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                self.reversed_phone_mapping,
            )
            for j in self.jobs
        ]

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

    def _collect_alignments(self, word_mode=True):
        """
        Process alignment archives to extract word or phone alignments

        Parameters
        ----------
        word_mode: bool
            Flag for collecting word or phone alignments

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.WordAlignmentFunction`
            Multiprocessing function for words alignments
        :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneAlignmentFunction`
            Multiprocessing function for phone alignments
        :meth:`.CorpusAligner.word_alignment_arguments`
            Arguments for word CTMS
        :meth:`.CorpusAligner.phone_alignment_arguments`
            Arguments for phone alignment
        """
        if word_mode:
            self.logger.info("Collecting word alignments from alignment lattices...")
            jobs = self.word_alignment_arguments()  # Word CTM jobs
        else:
            self.logger.info("Collecting phone alignments from alignment lattices...")
            jobs = self.phone_alignment_arguments()  # Phone CTM jobs
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(jobs):
                    if word_mode:
                        function = WordAlignmentFunction(args)
                    else:
                        function = PhoneAlignmentFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        utterance, intervals = return_queue.get(timeout=1)
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    if word_mode:
                        self.utterances[utterance].add_word_intervals(intervals)
                    else:
                        self.utterances[utterance].add_phone_intervals(intervals)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in jobs:
                    if word_mode:
                        function = WordAlignmentFunction(args)
                    else:
                        function = PhoneAlignmentFunction(args)
                    for utterance, intervals in function.run():
                        if word_mode:
                            self.utterances[utterance].add_word_intervals(intervals)
                        else:
                            self.utterances[utterance].add_phone_intervals(intervals)
                        pbar.update(1)

    def collect_word_alignments(self):
        self._collect_alignments(True)

    def collect_phone_alignments(self):
        self._collect_alignments(False)

    def collect_alignments(self):
        if self.alignment_done:
            return
        self.collect_word_alignments()
        self.collect_phone_alignments()
        self.alignment_done = True

    def export_textgrids(self) -> None:
        """
        Exports alignments to TextGrid files

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`
            Multiprocessing helper function for TextGrid export
        :meth:`.CorpusAligner.export_textgrid_arguments`
            Job method for TextGrid export
        """
        begin = time.time()
        self.logger.info("Exporting TextGrids...")
        os.makedirs(self.export_output_directory, exist_ok=True)
        if self.backup_output_directory:
            os.makedirs(self.backup_output_directory, exist_ok=True)

        export_errors = {}
        total_files = len(self.files)
        with tqdm.tqdm(total=total_files) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                textgrid_errors = manager.dict()
                stopped = Stopped()

                finished_processing = Stopped()
                for_write_queue = mp.JoinableQueue()
                export_args = self.export_textgrid_arguments()

                export_procs = []
                for j in self.jobs:
                    export_proc = ExportTextGridProcessWorker(
                        for_write_queue,
                        stopped,
                        finished_processing,
                        textgrid_errors,
                        export_args[j.name],
                    )
                    export_proc.start()
                    export_procs.append(export_proc)
                try:
                    for file in self.files:
                        tiers = file.aligned_data
                        output_path = file.construct_output_path(
                            self.export_output_directory, self.backup_output_directory
                        )
                        duration = file.duration
                        for_write_queue.put((tiers, output_path, duration))
                        pbar.update(1)
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
                for file in self.files:
                    data = file.aligned_data

                    backup_output_directory = None
                    if not self.overwrite:
                        backup_output_directory = self.backup_output_directory
                        os.makedirs(backup_output_directory, exist_ok=True)
                    output_path = file.construct_output_path(
                        self.export_output_directory, backup_output_directory
                    )
                    export_textgrid(data, output_path, file.duration, self.frame_shift)
                    pbar.update(1)

        if export_errors:
            self.logger.warning(
                f"There were {len(export_errors)} errors encountered in generating TextGrids. "
                f"Check the output_errors.txt file in {os.path.join(self.export_output_directory)} "
                f"for more details"
            )
        output_textgrid_writing_errors(self.export_output_directory, export_errors)
        self.logger.info("Finished exporting TextGrids!")
        self.logger.debug(f"Exported TextGrids in a total of {time.time() - begin} seconds")

    def export_files(self, output_directory: str) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        """
        self.export_output_directory = output_directory
        if self.backup_output_directory is not None and os.path.exists(
            self.backup_output_directory
        ):
            shutil.rmtree(self.backup_output_directory, ignore_errors=True)
        self.export_textgrids()
