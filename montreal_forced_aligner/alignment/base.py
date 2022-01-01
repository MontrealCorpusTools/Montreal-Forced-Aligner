"""Class definitions for base aligner"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from queue import Empty
from typing import List, Optional

import tqdm

from montreal_forced_aligner.abc import FileExporterMixin
from montreal_forced_aligner.alignment.mixins import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    AliToCtmArguments,
    AliToCtmFunction,
    ExportTextGridArguments,
    ExportTextGridProcessWorker,
    PhoneCtmArguments,
    PhoneCtmProcessWorker,
    WordCtmArguments,
    WordCtmProcessWorker,
)
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.exceptions import AlignmentExportError
from montreal_forced_aligner.textgrid import (
    export_textgrid,
    output_textgrid_writing_errors,
    process_ctm_line,
)
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

    def word_ctm_arguments(self) -> List[WordCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmProcessWorker`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmArguments`]
            Arguments for processing
        """
        return [
            WordCtmArguments(
                j.construct_path_dictionary(self.working_directory, "word", "ctm"),
                j.current_dictionary_names,
                {d.name: d.reversed_word_mapping for d in self.dictionary_mapping.values()},
                j.text_scp_data(),
                j.utt2spk_scp_data(),
                self.sanitize_function,
                self.cleanup_textgrids,
                self.oov_word,
            )
            for j in self.jobs
        ]

    def phone_ctm_arguments(self) -> List[PhoneCtmArguments]:
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
                self.reversed_phone_mapping,
                self.position_dependent_phones,
                self.cleanup_textgrids,
                self.optional_silence_phone,
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
                self.textgrid_output,
                self.backup_output_directory,
            )
            for j in self.jobs
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
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmFunction`
            Multiprocessing helper function for converting ali archives to CTM format
        :class:`~montreal_forced_aligner.alignment.multiprocessing.PhoneCtmProcessWorker`
            Multiprocessing helper class for processing CTM files
        :meth:`.CorpusAligner.phone_ctm_arguments`
            Job method for generating arguments for PhoneCtmProcessWorker
        :class:`~montreal_forced_aligner.alignment.multiprocessing.WordCtmProcessWorker`
            Multiprocessing helper class for processing word CTM files
        :meth:`.CorpusAligner.word_ctm_arguments`
            Job method for generating arguments for WordCtmProcessWorker
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
        word_procs = []
        phone_procs = []
        finished_processing = Stopped()
        to_process_queue = mp.JoinableQueue()
        for_write_queue = mp.JoinableQueue()
        total_files = len(self.files)
        word_ctm_args = self.word_ctm_arguments()
        phone_ctm_args = self.phone_ctm_arguments()
        export_args = self.export_textgrid_arguments()
        for j in self.jobs:
            word_p = WordCtmProcessWorker(
                j.name,
                to_process_queue,
                stopped,
                error_catching,
                word_ctm_args[j.name],
            )

            word_procs.append(word_p)
            word_p.start()

            phone_p = PhoneCtmProcessWorker(
                j.name,
                to_process_queue,
                stopped,
                error_catching,
                phone_ctm_args[j.name],
            )
            phone_p.start()
            phone_procs.append(phone_p)

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
            with tqdm.tqdm(total=total_files) as pbar:
                while True:
                    try:
                        w_p, intervals = to_process_queue.get(timeout=1)
                    except Empty:
                        for proc in word_procs:
                            if not proc.finished_signal.stop_check():
                                break
                        for proc in phone_procs:
                            if not proc.finished_signal.stop_check():
                                break
                        else:
                            break
                        continue
                    to_process_queue.task_done()
                    if self.stopped.stop_check():
                        self.logger.debug("Got stop check, exiting")
                        continue
                    utt = self.utterances[intervals[0].utterance]
                    if w_p == "word":
                        utt.add_word_intervals(intervals)
                    else:
                        utt.add_phone_intervals(intervals)
                    file = self.files[utt.file_name]
                    if file.is_fully_aligned:
                        tiers = file.aligned_data
                        output_path = file.construct_output_path(
                            self.textgrid_output, self.backup_output_directory
                        )
                        duration = file.duration
                        for_write_queue.put((tiers, output_path, duration))
                        pbar.update(1)
        except Exception:
            stopped.stop()
            while True:
                try:
                    _ = to_process_queue.get(timeout=1)
                except Empty:
                    for proc in word_procs:
                        if not proc.finished_signal.stop_check():
                            break
                    for proc in phone_procs:
                        if not proc.finished_signal.stop_check():
                            break
                    else:
                        break
                    continue
                to_process_queue.task_done()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_catching["main"] = "\n".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
        finally:
            finished_processing.stop()
            self.logger.debug("Waiting for processes to finish...")
            for i in range(self.num_jobs):
                word_procs[i].join()
                phone_procs[i].join()
            to_process_queue.join()

            for_write_queue.join()
            for i in range(self.num_jobs):
                export_procs[i].join()
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

    def ali_to_ctm(self, word_mode=True):
        """
        Convert alignment archives to CTM format

        Parameters
        ----------
        word_mode: bool
            Flag for generating word or phone CTMs

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmFunction`
            Multiprocessing function
        :meth:`.CorpusAligner.ali_to_word_ctm_arguments`
            Arguments for word CTMS
        :meth:`.CorpusAligner.ali_to_phone_ctm_arguments`
            Arguments for phone CTMS
        """
        if word_mode:
            self.logger.info("Generating word CTM files from alignment lattices...")
            jobs = self.ali_to_word_ctm_arguments()  # Word CTM jobs
        else:
            self.logger.info("Generating phone CTM files from alignment lattices...")
            jobs = self.ali_to_phone_ctm_arguments()  # Phone CTM jobs
        sum_errors = 0
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(jobs):
                    function = AliToCtmFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        done, errors = return_queue.get(timeout=1)
                        sum_errors += errors
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(done + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in jobs:
                    function = AliToCtmFunction(args)
                    for done, errors in function.run():
                        sum_errors += errors
                        pbar.update(done + errors)
            if sum_errors:
                self.logger.warning(f"{errors} utterances had errors during creating CTM files.")

    def convert_ali_to_textgrids(self) -> None:
        """
        Multiprocessing function that aligns based on the current model.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmFunction`
            Multiprocessing helper function for each job
        :meth:`.CorpusAligner.ali_to_word_ctm_arguments`
            Job method for generating arguments for this function
        :meth:`.CorpusAligner.ali_to_phone_ctm_arguments`
            Job method for generating arguments for this function
        :kaldi_steps:`get_train_ctm`
            Reference Kaldi script
        """
        os.makedirs(self.textgrid_output, exist_ok=True)
        self.logger.info("Generating CTMs from alignment...")
        self.ali_to_ctm(True)
        self.ali_to_ctm(False)
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
        self.log_debug("Not using multiprocessing for TextGrid export")
        export_errors = {}
        w_args = self.word_ctm_arguments()
        p_args = self.phone_ctm_arguments()
        for j in self.jobs:

            word_arguments = w_args[j.name]
            phone_arguments = p_args[j.name]
            self.logger.debug(f"Parsing ctms for job {j.name}...")

            for dict_name in word_arguments.dictionaries:
                with open(word_arguments.ctm_paths[dict_name], "r") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        interval = process_ctm_line(line)
                        utt = self.utterances[interval.utterance]
                        dictionary = self.get_dictionary(utt.speaker_name)
                        label = dictionary.reversed_word_mapping[int(interval.label)]

                        interval.label = label
                        utt.add_word_intervals(interval)

            for dict_name in phone_arguments.dictionaries:
                with open(phone_arguments.ctm_paths[dict_name], "r") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        interval = process_ctm_line(line)
                        utt = self.utterances[interval.utterance]
                        dictionary = self.get_dictionary(utt.speaker_name)

                        label = dictionary.reversed_phone_mapping[int(interval.label)]
                        if self.position_dependent_phones:
                            for p in dictionary.positions:
                                if label.endswith(p):
                                    label = label[: -1 * len(p)]
                        interval.label = label
                        utt.add_phone_intervals(interval)
        for file in self.files:
            data = file.aligned_data

            backup_output_directory = None
            if not self.overwrite:
                backup_output_directory = self.backup_output_directory
                os.makedirs(backup_output_directory, exist_ok=True)
            output_path = file.construct_output_path(self.textgrid_output, backup_output_directory)
            export_textgrid(data, output_path, file.duration, self.frame_shift)

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

    def ali_to_word_ctm_arguments(self) -> List[AliToCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmFunction`

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

    def ali_to_phone_ctm_arguments(self) -> List[AliToCtmArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.AliToCtmFunction`

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
