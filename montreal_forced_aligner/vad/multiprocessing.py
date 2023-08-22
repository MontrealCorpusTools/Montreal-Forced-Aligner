"""Multiprocessing functionality for VAD"""
from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import librosa
import numpy
import numpy as np
import pynini
import pywrapfst
from _kalpy.util import SequentialBaseFloatVectorReader
from Bio import pairwise2
from kalpy.utils import generate_read_specifier

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import CtmInterval, MfaArguments
from montreal_forced_aligner.db import SoundFile, Utterance

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        import torch
        from speechbrain.pretrained import VAD

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError):
    FOUND_SPEECHBRAIN = False
    VAD = None

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from dataclasses import dataclass

    from montreal_forced_aligner.abc import MetaDict
else:
    from dataclassy import dataclass


@dataclass
class SegmentVadArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`"""

    vad_path: Path
    segmentation_options: MetaDict


def get_initial_segmentation(frames: numpy.ndarray, frame_shift: float) -> List[CtmInterval]:
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
    segments = []
    cur_segment = None
    silent_frames = 0
    non_silent_frames = 0
    for i in range(frames.shape[0]):
        f = frames[i]
        if int(f) > 0:
            non_silent_frames += 1
            if cur_segment is None:
                cur_segment = CtmInterval(begin=i * frame_shift, end=0, label="speech")
        else:
            silent_frames += 1
            if cur_segment is not None:
                cur_segment.end = (i - 1) * frame_shift
                segments.append(cur_segment)
                cur_segment = None
    if cur_segment is not None:
        cur_segment.end = len(frames) * frame_shift
        segments.append(cur_segment)
    return segments


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
    min_segment_length: float
        Minimum length of segments returned

    Returns
    -------
    List[CtmInterval]
        Merged segments
    """
    merged_segments = []
    snap_boundary_threshold = min_pause_duration / 2
    for s in segments:
        if (
            not merged_segments
            or s.begin > merged_segments[-1].end + min_pause_duration
            or s.end - merged_segments[-1].begin > max_segment_length
        ):
            if s.end - s.begin > min_pause_duration:
                if merged_segments and snap_boundary_threshold:
                    boundary_gap = s.begin - merged_segments[-1].end
                    if boundary_gap < snap_boundary_threshold:
                        half_boundary = boundary_gap / 2
                    else:
                        half_boundary = snap_boundary_threshold / 2
                    merged_segments[-1].end += half_boundary
                    s.begin -= half_boundary

                merged_segments.append(s)
        else:
            merged_segments[-1].end = s.end
    return [x for x in merged_segments if x.end - x.begin > min_segment_length]


def construct_utterance_segmentation_fst(
    text: str,
    word_symbol_table: pywrapfst.SymbolTable,
    interjection_words: typing.List[str] = None,
):
    if interjection_words is None:
        interjection_words = []
    words = text.split()
    fst = pynini.Fst()
    start_state = fst.add_state()
    fst.set_start(start_state)
    fst.add_states(len(words))

    for i, w in enumerate(words):
        next_state = i + 1
        label = word_symbol_table.find(w)
        if i != 0:
            fst.add_arc(
                start_state,
                pywrapfst.Arc(label, label, pywrapfst.Weight.one(fst.weight_type()), next_state),
            )
        fst.add_arc(
            i, pywrapfst.Arc(label, label, pywrapfst.Weight.one(fst.weight_type()), next_state)
        )

        fst.set_final(next_state, pywrapfst.Weight(fst.weight_type(), 1))
        for interjection in interjection_words:
            start_interjection_state = fst.add_state()
            fst.add_arc(
                next_state,
                pywrapfst.Arc(
                    word_symbol_table.find("<eps>"),
                    word_symbol_table.find("<eps>"),
                    pywrapfst.Weight(fst.weight_type(), 10),
                    start_interjection_state,
                ),
            )
            if " " in interjection:
                i_words = interjection.split()
                for j, iw in enumerate(i_words):
                    next_interjection_state = fst.add_state()
                    if j == 0:
                        prev_state = start_interjection_state
                    else:
                        prev_state = next_interjection_state - 1
                    label = word_symbol_table.find(iw)
                    weight = pywrapfst.Weight.one(fst.weight_type())
                    fst.add_arc(
                        prev_state, pywrapfst.Arc(label, label, weight, next_interjection_state)
                    )
                final_interjection_state = next_interjection_state
            else:
                final_interjection_state = fst.add_state()
                label = word_symbol_table.find(interjection)
                weight = pywrapfst.Weight.one(fst.weight_type())
                fst.add_arc(
                    start_interjection_state,
                    pywrapfst.Arc(label, label, weight, final_interjection_state),
                )
            # Path to next word in text
            weight = pywrapfst.Weight.one(fst.weight_type())
            fst.add_arc(
                final_interjection_state,
                pywrapfst.Arc(
                    word_symbol_table.find("<eps>"),
                    word_symbol_table.find("<eps>"),
                    weight,
                    next_state,
                ),
            )
    for interjection in interjection_words:
        start_interjection_state = fst.add_state()
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                word_symbol_table.find("<eps>"),
                word_symbol_table.find("<eps>"),
                pywrapfst.Weight(fst.weight_type(), 10),
                start_interjection_state,
            ),
        )
        if " " in interjection:
            i_words = interjection.split()
            for j, iw in enumerate(i_words):
                next_interjection_state = fst.add_state()
                if j == 0:
                    prev_state = start_interjection_state
                else:
                    prev_state = next_interjection_state - 1
                label = word_symbol_table.find(iw)
                weight = pywrapfst.Weight.one(fst.weight_type())
                fst.add_arc(
                    prev_state, pywrapfst.Arc(label, label, weight, next_interjection_state)
                )
            final_interjection_state = next_interjection_state
        else:
            final_interjection_state = fst.add_state()
            label = word_symbol_table.find(interjection)
            weight = pywrapfst.Weight.one(fst.weight_type())
            fst.add_arc(
                start_interjection_state,
                pywrapfst.Arc(label, label, weight, final_interjection_state),
            )
        # Path to next word in text
        weight = pywrapfst.Weight.one(fst.weight_type())
        fst.add_arc(
            final_interjection_state,
            pywrapfst.Arc(
                word_symbol_table.find("<eps>"),
                word_symbol_table.find("<eps>"),
                weight,
                start_state,
            ),
        )
    fst.set_final(next_state, pywrapfst.Weight.one(fst.weight_type()))
    fst = pynini.determinize(fst)
    fst = pynini.rmepsilon(fst)
    fst = pynini.disambiguate(fst)
    fst = pynini.determinize(fst)
    return fst


def align_text(split_utterance_texts, text, oovs, oov_word, interjection_words):
    text = text.split()
    split_utterance_text = []
    lengths = []
    indices = list(split_utterance_texts.keys())
    for t in split_utterance_texts.values():
        t = t.split()
        lengths.append(len(t))
        split_utterance_text.extend(t)

    def score_func(first_element, second_element):
        if first_element == second_element:
            return 0
        if first_element == oov_word and second_element in oovs:
            return 0
        if first_element == oov_word and second_element not in oovs:
            return -10
        if first_element in interjection_words:
            return -10
        return -2

    alignments = pairwise2.align.globalcs(
        split_utterance_text, text, score_func, -0.5, -0.1, gap_char=["-"], one_alignment_only=True
    )
    results = [[]]
    split_ind = 0
    current_size = 0
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "<unk>":
                sa = sb
            if sa != "-":
                if (
                    split_ind < len(lengths) - 1
                    and sa not in split_utterance_texts[indices[split_ind]].split()
                    and split_utterance_texts[indices[split_ind + 1]].split()[0] == sa
                ):
                    results.append([])
                    split_ind += 1
                    current_size = 0
                results[-1].append(sa)
                current_size += 1
                if split_ind < len(lengths) - 1 and current_size >= lengths[split_ind]:
                    results.append([])
                    split_ind += 1
                    current_size = 0
            elif sb != "-":
                results[-1].append(sb)
    results = {k: " ".join(r) for k, r in zip(split_utterance_texts.keys(), results)}
    return results


def segment_utterance_vad_speech_brain(
    utterance: Utterance, sound_file: SoundFile, vad_model: VAD, segmentation_options: MetaDict
) -> np.ndarray:
    y, _ = librosa.load(
        sound_file.sound_file_path,
        sr=16000,
        mono=False,
        offset=utterance.begin,
        duration=utterance.duration,
    )
    if len(y.shape) > 1:
        y = y[:, utterance.channel]
    prob_chunks = vad_model.get_speech_prob_chunk(
        torch.tensor(y[np.newaxis, :], device=vad_model.device)
    ).cpu()
    prob_th = vad_model.apply_threshold(
        prob_chunks,
        activation_th=segmentation_options["activation_th"],
        deactivation_th=segmentation_options["deactivation_th"],
    ).float()
    # Compute the boundaries of the speech segments
    boundaries = vad_model.get_boundaries(prob_th, output_value="seconds")
    boundaries += utterance.begin
    # Apply energy-based VAD on the detected speech segments
    if True or segmentation_options["apply_energy_VAD"]:
        boundaries = vad_model.energy_VAD(
            sound_file.sound_file_path,
            boundaries,
            activation_th=segmentation_options["en_activation_th"],
            deactivation_th=segmentation_options["en_deactivation_th"],
        )

    # Merge short segments
    boundaries = vad_model.merge_close_segments(
        boundaries, close_th=segmentation_options["close_th"]
    )

    # Remove short segments
    boundaries = vad_model.remove_short_segments(boundaries, len_th=segmentation_options["len_th"])

    # Double check speech segments
    if segmentation_options["double_check"]:
        boundaries = vad_model.double_check_speech_segments(
            boundaries, sound_file.sound_file_path, speech_th=segmentation_options["speech_th"]
        )
    boundaries[:, 0] -= round(segmentation_options["close_th"] / 3, 3)
    boundaries[:, 1] += round(segmentation_options["close_th"] / 3, 3)
    return boundaries.numpy()


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

    def __init__(self, args: SegmentVadArguments):
        super().__init__(args)
        self.vad_path = args.vad_path
        self.segmentation_options = args.segmentation_options

    def _run(self):
        """Run the function"""
        reader = SequentialBaseFloatVectorReader(generate_read_specifier(self.vad_path))

        while not reader.Done():
            utt_id = reader.Key()
            frames = reader.Value()
            initial_segments = get_initial_segmentation(
                frames.numpy(), self.segmentation_options["frame_shift"]
            )

            merged = merge_segments(
                initial_segments,
                self.segmentation_options["close_th"],
                self.segmentation_options["large_chunk_size"],
                self.segmentation_options["len_th"],
            )
            self.callback((int(utt_id.split("-")[-1]), merged))
            reader.Next()
        reader.Close()
