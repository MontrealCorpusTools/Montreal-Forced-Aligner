"""Class definitions for corpora"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from queue import Empty

from montreal_forced_aligner.abc import MfaWorker, TemporaryDirectoryMixin
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import File
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.corpus.multiprocessing import CorpusProcessWorker
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import TextGridParseError, TextParseError
from montreal_forced_aligner.utils import Stopped


class TextCorpusMixin(CorpusMixin):
    """
    Abstract mixin class for processing text corpora

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.base.CorpusMixin`
        For corpus parsing parameters
    """

    def _load_corpus_from_source_mp(self) -> None:
        """
        Load a corpus using multiprocessing
        """
        if self.stopped is None:
            self.stopped = Stopped()
        try:
            sanitize_function = self.sanitize_function
        except AttributeError:
            sanitize_function = None
        begin_time = time.time()
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        return_dict["decode_error_files"] = manager.list()
        return_dict["textgrid_read_errors"] = manager.dict()
        finished_adding = Stopped()
        procs = []
        for i in range(self.num_jobs):
            p = CorpusProcessWorker(
                i,
                job_queue,
                return_dict,
                return_queue,
                self.stopped,
                finished_adding,
                self.speaker_characters,
                sanitize_function,
            )
            procs.append(p)
            p.start()
        try:
            for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                exts = find_exts(files)
                relative_path = root.replace(self.corpus_directory, "").lstrip("/").lstrip("\\")

                if self.stopped.stop_check():
                    break
                for file_name in exts.identifiers:
                    if self.stopped.stop_check():
                        break
                    wav_path = None
                    if file_name in exts.lab_files:
                        lab_name = exts.lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)

                    elif file_name in exts.textgrid_files:
                        tg_name = exts.textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    else:
                        continue
                    job_queue.put((file_name, wav_path, transcription_path, relative_path))

            finished_adding.stop()
            self.log_debug("Finished adding jobs!")
            job_queue.join()

            self.log_debug("Waiting for workers to finish...")
            for p in procs:
                p.join()

            while True:
                try:
                    file = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.stop_check():
                            break
                    else:
                        break
                    continue

                self.add_file(File.load_from_mp_data(file))

            if "error" in return_dict:
                raise return_dict["error"][1]

            for k in ["decode_error_files", "textgrid_read_errors"]:
                if hasattr(self, k):
                    if return_dict[k]:
                        self.log_info(
                            "There were some issues with files in the corpus. "
                            "Please look at the log file or run the validator for more information."
                        )
                        self.log_debug(f"{k} showed {len(return_dict[k])} errors:")
                        if k == "textgrid_read_errors":
                            getattr(self, k).update(return_dict[k])
                            for f, e in return_dict[k].items():
                                self.log_debug(f"{f}: {e.error}")
                        else:
                            self.log_debug(", ".join(return_dict[k]))
                            setattr(self, k, return_dict[k])

        except KeyboardInterrupt:
            self.log_info("Detected ctrl-c, please wait a moment while we clean everything up...")
            self.stopped.stop()
            finished_adding.stop()
            job_queue.join()
            self.stopped.set_sigint_source()
            while True:
                try:
                    _ = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.stop_check():
                            break
                    else:
                        break
        finally:

            finished_adding.stop()
            job_queue.join()
            for p in procs:
                p.join()
            if self.stopped.stop_check():
                self.log_info(f"Stopped parsing early ({time.time() - begin_time} seconds)")
                if self.stopped.source():
                    sys.exit(0)
            else:
                self.log_debug(
                    f"Parsed corpus directory with {self.num_jobs} jobs in {time.time() - begin_time} seconds"
                )

    def _load_corpus_from_source(self) -> None:
        """
        Load a corpus without using multiprocessing
        """
        begin_time = time.time()
        self.stopped = False

        try:
            sanitize_function = self.sanitize_function
        except AttributeError:
            sanitize_function = None
        for root, _, files in os.walk(self.corpus_directory, followlinks=True):
            exts = find_exts(files)
            relative_path = root.replace(self.corpus_directory, "").lstrip("/").lstrip("\\")
            if self.stopped:
                return
            for file_name in exts.identifiers:

                wav_path = None
                if file_name in exts.lab_files:
                    lab_name = exts.lab_files[file_name]
                    transcription_path = os.path.join(root, lab_name)
                elif file_name in exts.textgrid_files:
                    tg_name = exts.textgrid_files[file_name]
                    transcription_path = os.path.join(root, tg_name)
                else:
                    continue
                try:
                    file = File.parse_file(
                        file_name,
                        wav_path,
                        transcription_path,
                        relative_path,
                        self.speaker_characters,
                        sanitize_function,
                    )
                    self.add_file(file)
                except TextParseError as e:
                    self.decode_error_files.append(e)
                except TextGridParseError as e:
                    self.textgrid_read_errors[e.file_name] = e
        if self.decode_error_files or self.textgrid_read_errors:
            self.log_info(
                "There were some issues with files in the corpus. "
                "Please look at the log file or run the validator for more information."
            )
            if self.decode_error_files:
                self.log_debug(
                    f"There were {len(self.decode_error_files)} errors decoding text files:"
                )
                self.log_debug(", ".join(self.decode_error_files))
            if self.textgrid_read_errors:
                self.log_debug(
                    f"There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:"
                )
                for f, e in self.textgrid_read_errors.items():
                    self.log_debug(f"{f}: {e.error}")

        self.log_debug(f"Parsed corpus directory in {time.time()-begin_time} seconds")


class DictionaryTextCorpusMixin(TextCorpusMixin, MultispeakerDictionaryMixin):
    """
    Abstract mixin class for processing text corpora with pronunciation dictionaries.

    This is primarily useful for training language models, as you can treat words in the language model as OOV if they
    aren't in your pronunciation dictionary

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.text_corpus.TextCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.dictionary_setup()
        self._load_corpus()
        self.set_lexicon_word_set(self.corpus_word_set)
        self.write_lexicon_information()

        for speaker in self.speakers:
            speaker.set_dictionary(self.get_dictionary(speaker.name))
        self.initialize_jobs()
        self.write_corpus_information()
        self.create_corpus_split()


class TextCorpus(DictionaryTextCorpusMixin, MfaWorker, TemporaryDirectoryMixin):
    """
    Standalone class for working with text corpora and pronunciation dictionaries

    Most MFA functionality will use the :class:`~montreal_forced_aligner.corpus.text_corpus.DictionaryTextCorpusMixin` class rather than this class.

    Parameters
    ----------
    num_jobs: int
        Number of jobs to use when loading the corpus

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.text_corpus.DictionaryTextCorpusMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters
    """

    def __init__(self, num_jobs=3, **kwargs):
        super().__init__(**kwargs)
        self.num_jobs = num_jobs

    def load_corpus(self) -> None:
        """Load the corpus"""
        self._load_corpus()

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all corpus and dictionary files"""
        return os.path.join(self.temporary_directory, self.identifier)

    @property
    def working_directory(self) -> str:
        """Working directory"""
        return self.output_directory
