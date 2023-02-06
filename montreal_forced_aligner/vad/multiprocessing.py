"""Multiprocessing functionality for VAD"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import typing
from typing import TYPE_CHECKING, List, Union

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import CtmInterval, MfaArguments
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import read_feats, thirdparty_binary

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        from speechbrain.pretrained import VAD

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError):
    FOUND_SPEECHBRAIN = False
    VAD = None

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from montreal_forced_aligner.abc import MetaDict


class SegmentVadArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`"""

    vad_path: str
    segmentation_options: MetaDict


def get_initial_segmentation(
    frames: List[Union[int, str]], frame_shift: float
) -> List[CtmInterval]:
    """
    Compute initial segmentation over voice activity

    Parameters
    ----------
    frames: list[Union[int, str]]
        List of frames with VAD output
    frame_shift: float
        Frame shift of features in seconds

    Returns
    -------
    List[CtmInterval]
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
                cur_seg = CtmInterval(begin=i * frame_shift, end=0, label="speech")
        else:
            silent_frames += 1
            if cur_seg is not None:
                cur_seg.end = (i - 1) * frame_shift
                segs.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        cur_seg.end = len(frames) * frame_shift
        segs.append(cur_seg)
    return segs


def merge_segments(
    segments: List[CtmInterval],
    min_pause_duration: float,
    max_segment_length: float,
    min_segment_length: float,
) -> List[CtmInterval]:
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
    List[CtmInterval]
        Merged segments
    """
    merged_segs = []
    snap_boundary_threshold = min_pause_duration / 2
    for s in segments:
        if (
            not merged_segs
            or s.begin > merged_segs[-1].end + min_pause_duration
            or s.end - merged_segs[-1].begin > max_segment_length
        ):
            if s.end - s.begin > min_pause_duration:
                if merged_segs and snap_boundary_threshold:
                    boundary_gap = s.begin - merged_segs[-1].end
                    if boundary_gap < snap_boundary_threshold:
                        half_boundary = boundary_gap / 2
                    else:
                        half_boundary = snap_boundary_threshold / 2
                    merged_segs[-1].end += half_boundary
                    s.begin -= half_boundary

                merged_segs.append(s)
        else:
            merged_segs[-1].end = s.end
    return [x for x in merged_segs if x.end - x.begin > min_segment_length]


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
        super().__init__(args)
        self.vad_path = args.vad_path
        self.segmentation_options = args.segmentation_options

    def _run(self) -> typing.Generator[typing.Tuple[int, float, float]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-vector"),
                    "--binary=false",
                    f"scp:{self.vad_path}",
                    "ark,t:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            for utt_id, frames in read_feats(copy_proc):
                initial_segments = get_initial_segmentation(
                    frames, self.segmentation_options["frame_shift"]
                )

                merged = merge_segments(
                    initial_segments,
                    self.segmentation_options["close_th"],
                    self.segmentation_options["large_chunk_size"],
                    self.segmentation_options["len_th"],
                )
                yield utt_id, merged
