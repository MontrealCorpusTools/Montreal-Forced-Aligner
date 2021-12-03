"""Class definitions for base aligner"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from typing import Optional

from montreal_forced_aligner.abc import FileExporterMixin
from montreal_forced_aligner.alignment.mixins import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    AliToCtmArguments,
    CleanupWordCtmArguments,
    CleanupWordCtmProcessWorker,
    CombineCtmArguments,
    CombineProcessWorker,
    ExportPreparationProcessWorker,
    ExportTextGridArguments,
    ExportTextGridProcessWorker,
    NoCleanupWordCtmArguments,
    NoCleanupWordCtmProcessWorker,
    PhoneCtmArguments,
    PhoneCtmProcessWorker,
    ali_to_ctm_func,
)
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.exceptions import AlignmentExportError
from montreal_forced_aligner.textgrid import (
    ctm_to_textgrid,
    output_textgrid_writing_errors,
    parse_from_phone,
    parse_from_word,
    parse_from_word_no_cleanup,
    process_ctm_line,
)
from montreal_forced_aligner.utils import Stopped, run_mp, run_non_mp

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

    def cleanup_word_ctm_arguments(self) -> list[CleanupWordCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CleanupWordCtmProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.CleanupWordCtmArguments`]
            Arguments for processing
        """
        return [
            CleanupWordCtmArguments(
                j.construct_path_dictionary(self.working_directory, "word", "ctm"),
                j.current_dictionary_names,
                j.job_utts(),
                j.dictionary_data(),
            )
            for j in self.jobs
        ]

    def no_cleanup_word_ctm_arguments(self) -> list[NoCleanupWordCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.NoCleanupWordCtmProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.NoCleanupWordCtmArguments`]
            Arguments for processing
        """
        return [
            NoCleanupWordCtmArguments(
                j.construct_path_dictionary(self.working_directory, "word", "ctm"),
                j.current_dictionary_names,
                j.job_utts(),
                j.dictionary_data(),
            )
            for j in self.jobs
        ]

    def phone_ctm_arguments(self) -> list[PhoneCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmArguments`]
            Arguments for processing
        """
        return [
            PhoneCtmArguments(
                j.construct_path_dictionary(self.working_directory, "phone", "ctm"),
                j.current_dictionary_names,
                j.job_utts(),
                j.reversed_phone_mappings(),
                j.positions(),
            )
            for j in self.jobs
        ]

    def combine_ctm_arguments(self) -> list[CombineCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.CombineProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.CombineCtmArguments`]
            Arguments for processing
        """
        return [
            CombineCtmArguments(
                j.current_dictionary_names,
                j.job_files(),
                j.job_speakers(),
                j.dictionary_data(),
                self.cleanup_textgrids,
            )
            for j in self.jobs
        ]

    def export_textgrid_arguments(self) -> list[ExportTextGridArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridArguments`]
            Arguments for processing
        """
        return [
            ExportTextGridArguments(
                self.files,
                self.frame_shift,
                self.textgrid_output,
                self.backup_output_directory,
            )
            for _ in self.jobs
        ]

    @property
    def backup_output_directory(self) -> Optional[str]:
        """Backup directory if overwriting is not allowed"""
        return None

    def ctms_to_textgrids_mp(self):
        """
        Multiprocessing function for exporting alignment CTM information as TextGrids

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.multiprocessing.ali_to_ctm_func`
            Multiprocessing helper function for converting ali archives to CTM format
        :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmProcessWorker`
            Multiprocessing helper class for processing CTM files
        :meth:`.CorpusAligner.phone_ctm_arguments`
            Job method for generating arguments for PhoneCtmProcessWorker
        :class:`~montreal_forced_aligner.alignment.multiprocessing.CleanupWordCtmProcessWorker`
            Multiprocessing helper class for processing CTM files
        :meth:`.CorpusAligner.cleanup_word_ctm_arguments`
            Job method for generating arguments for CleanupWordCtmProcessWorker
        :class:`~montreal_forced_aligner.alignment.multiprocessing.NoCleanupWordCtmProcessWorker`
            Multiprocessing helper class for processing CTM files
        :meth:`.CorpusAligner.no_cleanup_word_ctm_arguments`
            Job method for generating arguments for NoCleanupWordCtmProcessWorker
        :class:`~montreal_forced_aligner.alignment.multiprocessing.CombineProcessWorker`
            Multiprocessing helper class for combining word and phone alignments
        :meth:`.CorpusAligner.combine_ctm_arguments`
            Job method for generating arguments for NoCleanupWordCtmProcessWorker
        :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportPreparationProcessWorker`
            Multiprocessing helper class for generating TextGrid tiers
        :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`
            Multiprocessing helper class for exporting TextGrid files
        :meth:`.CorpusAligner.export_textgrid_arguments`
            Job method for generating arguments for ExportTextGridProcessWorker
        :kaldi_steps:`get_train_ctm`
            Reference Kaldi script

        """
        export_begin = time.time()
        manager = mp.Manager()
        textgrid_errors = manager.dict()
        error_catching = manager.dict()
        stopped = Stopped()
        if not self.overwrite:
            os.makedirs(self.backup_output_directory, exist_ok=True)

        self.logger.debug("Beginning to process ctm files...")
        ctm_begin_time = time.time()
        word_procs = []
        phone_procs = []
        combine_procs = []
        finished_signals = [Stopped() for _ in range(self.num_jobs)]
        finished_processing = Stopped()
        to_process_queue = [mp.JoinableQueue() for _ in range(self.num_jobs)]
        to_export_queue = mp.JoinableQueue()
        for_write_queue = mp.JoinableQueue()
        finished_combining = Stopped()

        if self.cleanup_textgrids:
            word_ctm_args = self.cleanup_word_ctm_arguments()
        else:
            word_ctm_args = self.no_cleanup_word_ctm_arguments()
        phone_ctm_args = self.phone_ctm_arguments()
        combine_ctm_args = self.combine_ctm_arguments()
        export_args = self.export_textgrid_arguments()
        for j in self.jobs:
            if self.cleanup_textgrids:
                word_p = CleanupWordCtmProcessWorker(
                    j.name,
                    to_process_queue[j.name],
                    stopped,
                    error_catching,
                    word_ctm_args[j.name],
                )
            else:
                word_p = NoCleanupWordCtmProcessWorker(
                    j.name,
                    to_process_queue[j.name],
                    stopped,
                    error_catching,
                    word_ctm_args[j.name],
                )

            word_procs.append(word_p)
            word_p.start()

            phone_p = PhoneCtmProcessWorker(
                j.name,
                to_process_queue[j.name],
                stopped,
                error_catching,
                phone_ctm_args[j.name],
            )
            phone_p.start()
            phone_procs.append(phone_p)

            combine_p = CombineProcessWorker(
                j.name,
                to_process_queue[j.name],
                to_export_queue,
                stopped,
                finished_signals[j.name],
                error_catching,
                combine_ctm_args[j.name],
            )
            combine_p.start()
            combine_procs.append(combine_p)
        preparation_proc = ExportPreparationProcessWorker(
            to_export_queue, for_write_queue, stopped, finished_combining, self.files
        )
        preparation_proc.start()

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

        self.logger.debug("Waiting for processes to finish...")
        for i in range(self.num_jobs):
            word_procs[i].join()
            phone_procs[i].join()
            finished_signals[i].stop()

        self.logger.debug(f"Ctm parsers took {time.time() - ctm_begin_time} seconds")

        self.logger.debug("Waiting for processes to finish...")
        for i in range(self.num_jobs):
            to_process_queue[i].join()
            combine_procs[i].join()
        finished_combining.stop()

        self.logger.debug(f"Combiners took {time.time() - ctm_begin_time} seconds")
        self.logger.debug("Beginning export...")

        to_export_queue.join()
        preparation_proc.join()

        self.logger.debug(f"Adding jobs for export took {time.time() - export_begin}")
        self.logger.debug("Waiting for export processes to join...")

        finished_processing.stop()
        for i in range(self.num_jobs):
            export_procs[i].join()
        for_write_queue.join()
        self.logger.debug(f"Export took {time.time() - export_begin} seconds")

        if error_catching:
            self.logger.error("Error was encountered in processing CTMs")
            for key, error in error_catching.items():
                self.logger.error(f"{key}:\n\n{error}")
            raise AlignmentExportError(error_catching)

        if textgrid_errors:
            self.logger.warning(
                f"There were {len(textgrid_errors)} errors encountered in generating TextGrids. "
                f"Check the output_errors.txt file in {os.path.join(self.textgrid_output)} "
                f"for more details"
            )
        output_textgrid_writing_errors(self.textgrid_output, textgrid_errors)

    def convert_ali_to_textgrids(self) -> None:
        """
        Multiprocessing function that aligns based on the current model.

        See Also
        --------
        :func:`~montreal_forced_aligner.alignment.multiprocessing.ali_to_ctm_func`
            Multiprocessing helper function for each job
        :meth:`.CorpusAligner.ali_to_word_ctm_arguments`
            Job method for generating arguments for this function
        :meth:`.CorpusAligner.ali_to_phone_ctm_arguments`
            Job method for generating arguments for this function
        :kaldi_steps:`get_train_ctm`
            Reference Kaldi script
        """
        log_directory = self.working_log_directory
        os.makedirs(self.textgrid_output, exist_ok=True)
        jobs = self.ali_to_word_ctm_arguments()  # Word CTM jobs
        jobs += self.ali_to_phone_ctm_arguments()  # Phone CTM jobs
        self.logger.info("Generating CTMs from alignment...")
        if self.use_mp:
            run_mp(ali_to_ctm_func, jobs, log_directory)
        else:
            run_non_mp(ali_to_ctm_func, jobs, log_directory)
        self.logger.info("Finished generating CTMs!")

        self.logger.info("Exporting TextGrids from CTMs...")
        if self.use_mp:
            self.ctms_to_textgrids_mp()
        else:
            self.ctms_to_textgrids_non_mp()
        self.logger.info("Finished exporting TextGrids!")

    def ctms_to_textgrids_non_mp(self) -> None:
        """
        Parse CTM files to TextGrids without using multiprocessing
        """

        def process_current_word_labels():
            """Process the current stack of word labels"""
            speaker = cur_utt.speaker

            text = cur_utt.text.split()
            if self.cleanup_textgrids:
                actual_labels = parse_from_word(current_labels, text, speaker.dictionary_data)
            else:
                actual_labels = parse_from_word_no_cleanup(
                    current_labels, speaker.dictionary_data.reversed_words_mapping
                )
            cur_utt.word_labels = actual_labels

        def process_current_phone_labels():
            """Process the current stack of phone labels"""
            speaker = cur_utt.speaker

            cur_utt.phone_labels = parse_from_phone(
                current_labels,
                speaker.dictionary.reversed_phone_mapping,
                speaker.dictionary.positions,
            )

        export_errors = {}
        if self.cleanup_textgrids:
            w_args = self.cleanup_word_ctm_arguments()
        else:
            w_args = self.no_cleanup_word_ctm_arguments()
        p_args = self.phone_ctm_arguments()
        for j in self.jobs:

            word_arguments = w_args[j.name]
            phone_arguments = p_args[j.name]
            self.logger.debug(f"Parsing ctms for job {j.name}...")
            cur_utt = None
            current_labels = []
            for dict_name in word_arguments.dictionaries:
                with open(word_arguments.ctm_paths[dict_name], "r") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        ctm_interval = process_ctm_line(line)
                        utt = self.utterances[ctm_interval.utterance]
                        if cur_utt is None:
                            cur_utt = utt
                        if utt.is_segment:
                            utt_begin = utt.begin
                        else:
                            utt_begin = 0
                        if utt != cur_utt:
                            process_current_word_labels()
                            cur_utt = utt
                            current_labels = []

                        ctm_interval.shift_times(utt_begin)
                        current_labels.append(ctm_interval)
                if current_labels:
                    process_current_word_labels()
            cur_utt = None
            current_labels = []
            for dict_name in phone_arguments.dictionaries:
                with open(phone_arguments.ctm_paths[dict_name], "r") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        ctm_interval = process_ctm_line(line)
                        utt = self.utterances[ctm_interval.utterance]
                        if cur_utt is None:
                            cur_utt = utt
                        if utt.is_segment:
                            utt_begin = utt.begin
                        else:
                            utt_begin = 0
                        if utt != cur_utt and cur_utt is not None:
                            process_current_phone_labels()
                            cur_utt = utt
                            current_labels = []

                        ctm_interval.shift_times(utt_begin)
                        current_labels.append(ctm_interval)
                if current_labels:
                    process_current_phone_labels()

            self.logger.debug(f"Generating TextGrids for job {j.name}...")
            processed_files = set()
            for file in j.job_files().values():
                first_file_write = True
                if file.name in processed_files:
                    first_file_write = False
                try:
                    ctm_to_textgrid(file, self, first_file_write)
                    processed_files.add(file.name)
                except Exception:
                    if self.debug:
                        raise
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    export_errors[file.name] = "\n".join(
                        traceback.format_exception(exc_type, exc_value, exc_traceback)
                    )
        if export_errors:
            self.logger.warning(
                f"There were {len(export_errors)} errors encountered in generating TextGrids. "
                f"Check the output_errors.txt file in {os.path.join(self.textgrid_output)} "
                f"for more details"
            )
        output_textgrid_writing_errors(self.textgrid_output, export_errors)

    def export_files(self, output_directory: str) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: str
            Directory to save to
        """
        begin = time.time()
        self.textgrid_output = output_directory
        if self.backup_output_directory is not None and os.path.exists(
            self.backup_output_directory
        ):
            shutil.rmtree(self.backup_output_directory, ignore_errors=True)
        self.convert_ali_to_textgrids()
        self.logger.debug(f"Exported TextGrids in a total of {time.time() - begin} seconds")

    def ali_to_word_ctm_arguments(self) -> list[AliToCtmArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.ali_to_ctm_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmArguments`]
            Arguments for processing
        """
        return [
            AliToCtmArguments(
                os.path.join(self.working_log_directory, f"get_word_ctm.{j.name}.log"),
                j.current_dictionary_names,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.word_boundary_int_files(),
                round(self.frame_shift / 1000, 4),
                self.alignment_model_path,
                j.construct_path_dictionary(self.working_directory, "word", "ctm"),
                True,
            )
            for j in self.jobs
        ]

    def ali_to_phone_ctm_arguments(self) -> list[AliToCtmArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.ali_to_ctm_func`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmArguments`]
            Arguments for processing
        """
        return [
            AliToCtmArguments(
                os.path.join(self.working_log_directory, f"get_phone_ctm.{j.name}.log"),
                j.current_dictionary_names,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                j.construct_path_dictionary(self.data_directory, "text", "int.scp"),
                j.word_boundary_int_files(),
                round(self.frame_shift / 1000, 4),
                self.alignment_model_path,
                j.construct_path_dictionary(self.working_directory, "phone", "ctm"),
                False,
            )
            for j in self.jobs
        ]
