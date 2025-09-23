"""Classes for remapping alignments from one phone set to another"""
from __future__ import annotations

import logging
import os
import threading
import time
import typing
from pathlib import Path
from queue import Empty, Queue

import yaml
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import PhoneRemapperMixin, TopLevelMfaWorker
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.corpus.multiprocessing import AlignmentRemapperWorker
from montreal_forced_aligner.helper import mfa_open

logger = logging.getLogger("mfa")

__all__ = ["AlignmentRemapper"]


class AlignmentRemapper(PhoneRemapperMixin, TopLevelMfaWorker):
    def __init__(
        self,
        corpus_directory: typing.Union[str, Path],
        split_percentage: float = 0.5,
        **kwargs,
    ):
        self.corpus_directory = Path(corpus_directory)
        self._data_source = self.corpus_directory.stem
        self.split_percentage = split_percentage
        super().__init__(**kwargs)
        self.stopped = None

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return self._data_source

    @property
    def data_directory(self) -> Path:
        """Data directory for trainer"""
        return self.working_directory

    def setup(self) -> None:
        """Setup for dictionary remapping"""
        super().setup()
        self.load_mapping()
        self.validate_mapping()
        if self.initialized:
            return
        self.initialized = True

    def load_mapping(self) -> None:
        with mfa_open(self.phone_mapping_path, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        for key, value in data.items():
            if isinstance(value, list):
                value = value[0]
                logger.warning(
                    f"Found ambiguous mapping for {key}, using first value ({value}) as the target."
                )
            if " " in value:
                value = tuple(value.split())
            self.phone_remapping[key] = value

    def validate_mapping(self):
        covered_phones = set()
        found_splitting = False

        for key, value in self.phone_remapping.items():
            if isinstance(value, tuple):
                found_splitting = True
            if " " in key:
                for p in key.split():
                    covered_phones.add(p)
            else:
                covered_phones.add(key)
        if found_splitting and self.split_percentage != 0.5:
            logger.warning(
                "Found instances of splitting one phone to multiple phones, "
                "be aware that new segments will receive equal distribution of duration. "
                "If a different point is better, use --split_percentage 0.75 to specify 75%, for instance, "
                "but this will only affect behavior when splitting to two phones."
            )

    def remap_alignments(
        self,
        output_directory: typing.Union[Path, str],
        output_format: typing.Literal[
            "short_textgrid", "long_textgrid", "json", "textgrid_json"
        ] = "short_textgrid",
    ):
        if self.stopped is None:
            self.stopped = threading.Event()
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        begin_time = time.time()
        job_queue = Queue()
        return_queue = Queue()
        error_dict = {}
        finished_adding = threading.Event()
        procs = []
        for i in range(config.NUM_JOBS):
            p = AlignmentRemapperWorker(
                i,
                job_queue,
                return_queue,
                self.stopped,
                finished_adding,
                self.phone_remapping,
                self.split_percentage,
                output_format,
            )
            procs.append(p)
            p.start()

        try:
            file_count = 0
            with tqdm(total=1, disable=config.QUIET) as pbar:
                for root, _, files in os.walk(self.corpus_directory, followlinks=True):
                    if self.stopped.is_set():
                        break
                    if root.startswith("."):  # Ignore hidden directories
                        continue
                    exts = find_exts(files)
                    relative_path = (
                        root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")
                    )
                    for tg_name in exts.textgrid_files.values():
                        if self.stopped.is_set():
                            break
                        input_path = os.path.join(root, tg_name)
                        output_dir = output_directory.joinpath(relative_path)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir.joinpath(tg_name)
                        job_queue.put((input_path, output_path))
                        file_count += 1
                        pbar.total = file_count

                finished_adding.set()

                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, tuple):
                            error_type = result[0]
                            error = result[1]
                            if error_type == "error":
                                error_dict[error_type] = error
                            else:
                                if error_type not in error_dict:
                                    error_dict[error_type] = []
                                error_dict[error_type].append(error)
                            continue
                        if self.stopped.is_set():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished_processing.is_set():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    return_queue.task_done()

                logger.debug("Waiting for workers to finish...")
                for p in procs:
                    p.join()

                if "error" in error_dict:
                    raise error_dict["error"]

        except KeyboardInterrupt:
            logger.info("Detected ctrl-c, please wait a moment while we clean everything up...")
            self.stopped.set()
            finished_adding.set()
            while True:
                try:
                    _ = return_queue.get(timeout=1)
                    return_queue.task_done()
                except Empty:
                    for proc in procs:
                        if not proc.finished_processing.is_set():
                            break
                    else:
                        break
        finally:
            finished_adding.set()
            for p in procs:
                p.join()
            if self.stopped.is_set():
                logger.info(f"Stopped parsing early ({time.time() - begin_time:.3f} seconds)")
            else:
                logger.debug(
                    f"Remapped alignments with {config.NUM_JOBS} jobs in {time.time() - begin_time:.3f} seconds"
                )
