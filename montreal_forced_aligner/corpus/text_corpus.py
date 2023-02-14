"""Class definitions for corpora"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from queue import Empty

import tqdm

from montreal_forced_aligner.abc import MfaWorker, TemporaryDirectoryMixin
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.base import CorpusMixin
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.corpus.multiprocessing import CorpusProcessWorker
from montreal_forced_aligner.data import DatabaseImportData
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import TextGridParseError, TextParseError
from montreal_forced_aligner.utils import Stopped

logger = logging.getLogger("mfa")


class TextCorpusMixin(CorpusMixin):
    """
    Abstract mixin class for processing text corpora

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.base.CorpusMixin`
        For corpus parsing parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_corpus_from_source_mp(self) -> None:
        """
        Load a corpus using multiprocessing
        """
        if self.stopped is None:
            self.stopped = Stopped()
        begin_time = time.time()
        job_queue = mp.Queue()
        return_queue = mp.Queue()
        error_dict = {}
        finished_adding = Stopped()
        procs = []
        for i in range(GLOBAL_CONFIG.num_jobs):
            p = CorpusProcessWorker(
                i,
                job_queue,
                return_queue,
                self.stopped,
                finished_adding,
                self.speaker_characters,
                sample_rate=0,
            )
            procs.append(p)
            p.start()
        import_data = DatabaseImportData()
        try:
            file_count = 0
            with tqdm.tqdm(
                total=1, disable=GLOBAL_CONFIG.quiet
            ) as pbar, self.session() as session:
                for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                    exts = find_exts(files)
                    relative_path = (
                        root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")
                    )

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
                        file_count += 1
                        pbar.total = file_count

                finished_adding.stop()

                while True:
                    try:
                        file = return_queue.get(timeout=1)
                        if isinstance(file, tuple):
                            error_type = file[0]
                            error = file[1]
                            if error_type == "error":
                                error_dict[error_type] = error
                            else:
                                if error_type not in error_dict:
                                    error_dict[error_type] = []
                                error_dict[error_type].append(error)
                            continue
                        if self.stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished_processing.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    import_data.add_objects(self.generate_import_objects(file))

                logger.debug("Waiting for workers to finish...")
                for p in procs:
                    p.join()

                if "error" in error_dict:
                    session.rollback()
                    raise error_dict["error"][1]

                self._finalize_load(session, import_data)

                for k in ["decode_error_files", "textgrid_read_errors"]:
                    if hasattr(self, k):
                        if k in error_dict:
                            logger.info(
                                "There were some issues with files in the corpus. "
                                "Please look at the log file or run the validator for more information."
                            )
                            logger.debug(f"{k} showed {len(error_dict[k])} errors:")
                            if k == "textgrid_read_errors":
                                getattr(self, k).extend(error_dict[k])
                                for e in error_dict[k]:
                                    logger.debug(f"{e.file_name}: {e.error}")
                            else:
                                logger.debug(", ".join(error_dict[k]))
                                setattr(self, k, error_dict[k])

        except KeyboardInterrupt:
            logger.info("Detected ctrl-c, please wait a moment while we clean everything up...")
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
            for p in procs:
                p.join()
            if self.stopped.stop_check():
                logger.info(f"Stopped parsing early ({time.time() - begin_time:.3f} seconds)")
                if self.stopped.source():
                    sys.exit(0)
            else:
                logger.debug(
                    f"Parsed corpus directory with {GLOBAL_CONFIG.num_jobs} jobs in {time.time() - begin_time:.3f} seconds"
                )

    def _load_corpus_from_source(self) -> None:
        """
        Load a corpus without using multiprocessing
        """
        begin_time = time.time()
        self.stopped = False

        import_data = DatabaseImportData()
        sanitize_function = getattr(self, "sanitize_function", None)
        with self.session() as session:
            for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                exts = find_exts(files)
                relative_path = (
                    root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")
                )
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
                        file = FileData.parse_file(
                            file_name,
                            wav_path,
                            transcription_path,
                            relative_path,
                            self.speaker_characters,
                            sanitize_function,
                        )
                        import_data.add_objects(self.generate_import_objects(file))
                    except TextParseError as e:
                        self.decode_error_files.append(e)
                    except TextGridParseError as e:
                        self.textgrid_read_errors.append(e)
            self._finalize_load(session, import_data)
        if self.decode_error_files or self.textgrid_read_errors:
            logger.info(
                "There were some issues with files in the corpus. "
                "Please look at the log file or run the validator for more information."
            )
            if self.decode_error_files:
                logger.debug(
                    f"There were {len(self.decode_error_files)} errors decoding text files:"
                )
                logger.debug(", ".join(self.decode_error_files))
            if self.textgrid_read_errors:
                logger.debug(
                    f"There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:"
                )
                for e in self.textgrid_read_errors:
                    logger.debug(f"{e.file_name}: {e.error}")

        logger.debug(f"Parsed corpus directory in {time.time()-begin_time} seconds")


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
        self.initialize_database()
        self.dictionary_setup()

        self._load_corpus()
        self.initialize_jobs()
        self.normalize_text()
        self.write_lexicon_information()
        self.create_corpus_split()


class TextCorpus(TextCorpusMixin, MfaWorker, TemporaryDirectoryMixin):
    """
    Standalone class for working with text corpora without a pronunciation dictionary

    Most MFA functionality will use the :class:`~montreal_forced_aligner.corpus.text_corpus.TextCorpusMixin` class rather than this class.

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.text_corpus.DictionaryTextCorpusMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.initialize_database()

        self._load_corpus()
        self.initialize_jobs()
        self.create_corpus_split()

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all corpus and dictionary files"""
        return os.path.join(GLOBAL_CONFIG.temporary_directory, self.identifier)

    @property
    def working_directory(self) -> str:
        """Working directory"""
        return self.corpus_output_directory


class DictionaryTextCorpus(DictionaryTextCorpusMixin, MfaWorker, TemporaryDirectoryMixin):
    """
    Standalone class for working with text corpora and pronunciation dictionaries

    Most MFA functionality will use the :class:`~montreal_forced_aligner.corpus.text_corpus.DictionaryTextCorpusMixin` class rather than this class.

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.text_corpus.DictionaryTextCorpusMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def identifier(self) -> str:
        """Identifier for the corpus"""
        return self.data_source_identifier

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all corpus and dictionary files"""
        return os.path.join(GLOBAL_CONFIG.temporary_directory, self.identifier)

    @property
    def working_directory(self) -> str:
        """Working directory"""
        return self.corpus_output_directory
