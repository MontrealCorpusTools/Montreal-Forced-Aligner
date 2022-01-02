"""
Segmenting files
================

.. autosummary::

   :toctree: generated/

"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union

import tqdm
import yaml

from montreal_forced_aligner.abc import FileExporterMixin, MetaDict, TopLevelMfaWorker
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.classes import File, Speaker, Utterance
from montreal_forced_aligner.corpus.features import VadConfigMixin
from montreal_forced_aligner.data import TextFileType
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import load_scp
from montreal_forced_aligner.utils import (
    KaldiFunction,
    KaldiProcessWorker,
    Stopped,
    log_kaldi_errors,
    parse_logs,
)

if TYPE_CHECKING:
    from argparse import Namespace

SegmentationType = List[Dict[str, float]]

__all__ = ["Segmenter", "SegmentVadFunction", "SegmentVadArguments"]


class SegmentVadArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`"""

    vad_path: str
    segmentation_options: MetaDict


def get_initial_segmentation(frames: List[Union[int, str]], frame_shift: int) -> SegmentationType:
    """
    Compute initial segmentation over voice activity

    Parameters
    ----------
    frames: list[Union[int, str]]
        List of frames with VAD output
    frame_shift: int
        Frame shift of features in ms

    Returns
    -------
    SegmentationType
        Initial segmentation
    """
    segs = []
    cur_seg = None
    silent_frames = 0
    non_silent_frames = 0
    for i, f in enumerate(frames):
        if int(f) > 0:
            non_silent_frames += 1
            if cur_seg is None:
                cur_seg = {"begin": i * frame_shift}
        else:
            silent_frames += 1
            if cur_seg is not None:
                cur_seg["end"] = (i - 1) * frame_shift
                segs.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        cur_seg["end"] = len(frames) * frame_shift
        segs.append(cur_seg)
    return segs


def merge_segments(
    segments: SegmentationType,
    min_pause_duration: float,
    max_segment_length: float,
    snap_boundary_threshold: float,
) -> SegmentationType:
    """
    Merge segments together

    Parameters
    ----------
    segments: SegmentationType
        Initial segments
    min_pause_duration: float
        Minimum amount of silence time to mark an utterance boundary
    max_segment_length: float
        Maximum length of segments before they're broken up
    snap_boundary_threshold:
        Boundary threshold to snap boundaries together

    Returns
    -------
    SegmentationType
        Merged segments
    """
    merged_segs = []
    for s in segments:
        if (
            not merged_segs
            or s["begin"] > merged_segs[-1]["end"] + min_pause_duration
            or s["end"] - merged_segs[-1]["begin"] > max_segment_length
        ):
            if s["end"] - s["begin"] > min_pause_duration:
                if merged_segs and snap_boundary_threshold:
                    boundary_gap = s["begin"] - merged_segs[-1]["end"]
                    if boundary_gap < snap_boundary_threshold:
                        half_boundary = boundary_gap / 2
                    else:
                        half_boundary = snap_boundary_threshold / 2
                    merged_segs[-1]["end"] += half_boundary
                    s["begin"] -= half_boundary

                merged_segs.append(s)
        else:
            merged_segs[-1]["end"] = s["end"]
    return merged_segs


class SegmentVadFunction(KaldiFunction):
    """
    Multiprocessing function to generate segments from VAD output.

    See Also
    --------
    :meth:`montreal_forced_aligner.segmenter.Segmenter.segment_vad`
        Main function that calls this function in parallel
    :meth:`montreal_forced_aligner.segmenter.Segmenter.segment_vad_arguments`
        Job method for generating arguments for this function
    :kaldi_utils:`segmentation.pl`
        Kaldi utility

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.segmenter.SegmentVadArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*processed (?P<done>\d+) utterances.*(?P<no_feats>\d+) had.*(?P<unvoiced>\d+) were.*"
    )

    def __init__(self, args: SegmentVadArguments):
        self.vad_path = args.vad_path
        self.segmentation_options = args.segmentation_options

    def run(self):
        """Run the function"""

        speaker = Speaker("speech")

        vad = load_scp(self.vad_path, data_type=int)
        for recording, frames in vad.items():
            file = File(recording)
            initial_segments = get_initial_segmentation(
                frames, self.segmentation_options["frame_shift"]
            )

            merged = merge_segments(
                initial_segments,
                self.segmentation_options["min_pause_duration"],
                self.segmentation_options["max_segment_length"],
                self.segmentation_options["snap_boundary_threshold"],
            )
            for seg in merged:
                yield recording, Utterance(
                    speaker, file, begin=seg["begin"], end=seg["end"], text="speech"
                )


class Segmenter(VadConfigMixin, AcousticCorpusMixin, FileExporterMixin, TopLevelMfaWorker):
    """
    Class for performing speaker classification

    Parameters
    ----------
    max_segment_length : float
        Maximum duration of segments
    min_pause_duration : float
        Minimum duration of pauses
    snap_boundary_threshold : float
        Threshold for snapping segment boundaries to each other
    """

    def __init__(
        self,
        max_segment_length: float = 30,
        min_pause_duration: float = 0.05,
        snap_boundary_threshold: float = 0.15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_segment_length = max_segment_length
        self.min_pause_duration = min_pause_duration
        self.snap_boundary_threshold = snap_boundary_threshold

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Namespace] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters for segmentation from a config path or command-line arguments

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
            with open(config_path, "r", encoding="utf8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for k, v in data.items():
                    if k == "features":
                        if "type" in v:
                            v["feature_type"] = v["type"]
                            del v["type"]
                        global_params.update(v)
                    else:
                        if v is None and k in {
                            "punctuation",
                            "compound_markers",
                            "clitic_markers",
                        }:
                            v = []
                        global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    def segment_vad_arguments(self) -> List[SegmentVadArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`

        Returns
        -------
        list[SegmentVadArguments]
            Arguments for processing
        """
        return [
            SegmentVadArguments(
                j.construct_path(self.split_directory, "vad", "scp"),
                self.segmentation_options,
            )
            for j in self.jobs
        ]

    @property
    def segmentation_options(self):
        """Options for segmentation"""
        return {
            "max_segment_length": self.max_segment_length,
            "min_pause_duration": self.min_pause_duration,
            "snap_boundary_threshold": self.snap_boundary_threshold,
            "frame_shift": round(self.frame_shift / 1000, 2),
        }

    @property
    def workflow_identifier(self) -> str:
        """Segmentation workflow"""
        return "segmentation"

    def segment_vad(self) -> None:
        """
        Run segmentation based off of VAD.

        See Also
        --------
        :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`
            Multiprocessing helper function for each job
        segment_vad_arguments
            Job method for generating arguments for helper function
        """

        arguments = self.segment_vad_arguments()
        old_utts = set()
        with tqdm.tqdm(total=self.num_utterances) as pbar:
            if self.use_mp:
                manager = mp.Manager()
                error_dict = manager.dict()
                return_queue = manager.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = SegmentVadFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, error_dict, stopped)
                    procs.append(p)
                    p.start()
                    while True:
                        try:
                            old_utt, utterance = return_queue.get(timeout=1)
                            # print(utterance)
                            if stopped.stop_check():
                                continue
                        except Empty:
                            for proc in procs:
                                if not proc.finished.stop_check():
                                    break
                            else:
                                break
                            continue
                        if old_utt not in old_utts:
                            old_utt = self.utterances[old_utt]
                            file = old_utt.file
                            if self.ignore_speakers:
                                if utterance.speaker_name not in self.speakers:
                                    self.speakers[utterance.speaker_name] = Speaker(
                                        utterance.speaker_name
                                    )
                                speaker = self.speakers[utterance.speaker_name]
                            else:
                                speaker = old_utt.speaker
                            utterance.file = file
                            utterance.set_speaker(speaker)
                            self.add_utterance(utterance)
                            old_utts.add(old_utt)
                            pbar.update(1)
                for p in procs:
                    p.join()
                if error_dict:
                    for v in error_dict.values():
                        raise v
            else:
                for args in arguments:
                    function = SegmentVadFunction(args)
                    for old_utt, utterance in function.run():
                        if old_utt not in old_utts:
                            old_utt = self.utterances[old_utt]
                            file = old_utt.file
                            if self.ignore_speakers:
                                if utterance.speaker_name not in self.speakers:
                                    self.speakers[utterance.speaker_name] = Speaker(
                                        utterance.speaker_name
                                    )
                                speaker = self.speakers[utterance.speaker_name]
                            else:
                                speaker = old_utt.speaker
                            utterance.file = file
                            utterance.set_speaker(speaker)
                            self.add_utterance(utterance)
                            old_utts.add(old_utt)
                            pbar.update(1)
        for u in old_utts:
            self.delete_utterance(u)

    def setup(self) -> None:
        """Setup segmentation"""
        self.check_previous_run()
        log_dir = os.path.join(self.working_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.load_corpus()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise

    def segment(self) -> None:
        """
        Performs VAD and segmentation into utterances

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.setup()
        log_directory = os.path.join(self.working_directory, "log")
        done_path = os.path.join(self.working_directory, "done")
        if os.path.exists(done_path):
            self.logger.info("Classification already done, skipping.")
            return
        try:
            self.compute_vad()
            self.uses_vad = True
            self.segment_vad()
            parse_logs(log_directory)
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs, self.logger)
                e.update_log_file(self.logger)
            raise
        with open(done_path, "w"):
            pass

    @property
    def backup_output_directory(self) -> str:
        """Backup output directory"""
        return os.path.join(self.workflow_directory, "backup")

    def export_files(self, output_directory: str) -> None:
        """
        Export the results of segmentation as TextGrids

        Parameters
        ----------
        output_directory: str
            Directory to save segmentation TextGrids
        """
        backup_output_directory = None
        if not self.overwrite:
            backup_output_directory = os.path.join(self.working_directory, "transcriptions")
            os.makedirs(backup_output_directory, exist_ok=True)
        for f in self.files:
            f.save(output_directory, backup_output_directory, text_type=TextFileType.TEXTGRID)
