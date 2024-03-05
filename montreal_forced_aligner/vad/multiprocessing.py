"""Multiprocessing functionality for VAD"""
from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy
import numpy as np
import pynini
import pywrapfst
from _kalpy.decoder import LatticeFasterDecoder, LatticeFasterDecoderConfig
from _kalpy.fstext import GetLinearSymbolSequence
from _kalpy.gmm import DecodableAmDiagGmmScaled
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from _kalpy.util import SequentialBaseFloatVectorReader
from kalpy.data import Segment
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.cmvn import CmvnComputer
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.vad import VadComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.utils import generate_read_specifier, read_kaldi_object
from kalpy.utterance import Utterance as KalpyUtterance
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.data import CtmInterval, MfaArguments
from montreal_forced_aligner.db import File, Job, Speaker, Utterance
from montreal_forced_aligner.exceptions import SegmenterError
from montreal_forced_aligner.models import AcousticModel, G2PModel

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        import torch

        try:
            from speechbrain.pretrained import VAD
        except ImportError:  # speechbrain 1.0
            from speechbrain.inference.VAD import VAD

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

__all__ = [
    "SegmentTranscriptArguments",
    "SegmentVadArguments",
    "SegmentTranscriptFunction",
    "SegmentVadFunction",
    "get_initial_segmentation",
    "merge_segments",
    "segment_utterance_transcript",
    "segment_utterance_vad",
    "segment_utterance_vad_speech_brain",
]


@dataclass
class SegmentVadArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`"""

    vad_path: Path
    segmentation_options: MetaDict


@dataclass
class SegmentTranscriptArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.segmenter.SegmentTranscriptFunction`"""

    acoustic_model: AcousticModel
    vad_model: typing.Optional[VAD]
    lexicon_compilers: typing.Dict[int, LexiconCompiler]
    mfcc_options: MetaDict
    vad_options: MetaDict
    segmentation_options: MetaDict
    decode_options: MetaDict


def segment_utterance_transcript(
    acoustic_model: AcousticModel,
    utterance: KalpyUtterance,
    lexicon_compiler: LexiconCompiler,
    vad_model: VAD,
    segmentation_options: MetaDict,
    cmvn: DoubleMatrix = None,
    fmllr_trans: FloatMatrix = None,
    mfcc_options: MetaDict = None,
    vad_options: MetaDict = None,
    g2p_model: G2PModel = None,
    interjection_words: typing.List[str] = None,
    acoustic_scale: float = 0.1,
    beam: float = 16.0,
    lattice_beam: float = 10.0,
    max_active: int = 7000,
    min_active: int = 200,
    prune_interval: int = 25,
    beam_delta: float = 0.5,
    hash_ratio: float = 2.0,
    prune_scale: float = 0.1,
    boost_silence: float = 1.0,
):
    """
    Split an utterance and its transcript into multiple transcribed utterances

    Parameters
    ----------
    acoustic_model: :class:`~montreal_forced_aligner.models.AcousticModel`
        Acoustic model to use in splitting transcriptions
    utterance: :class:`~kalpy.utterance.Utterance`
        Utterance to split
    lexicon_compiler: :class:`~kalpy.fstext.lexicon.LexiconCompiler`
        Lexicon compiler
    vad_model: :class:`~speechbrain.pretrained.VAD` or None
        VAD model from SpeechBrain, if None, then Kaldi's energy-based VAD is used
    segmentation_options: dict[str, Any]
        Segmentation options
    cmvn: :class:`~_kalpy.matrix.DoubleMatrix`
        CMVN stats to apply
    fmllr_trans: :class:`~_kalpy.matrix.FloatMatrix`
        fMLLR transformation matrix for speaker adaptation
    mfcc_options: dict[str, Any], optional
        MFCC options for energy based VAD
    vad_options: dict[str, Any], optional
        Options for energy based VAD
    acoustic_scale: float, optional
        Defaults to 0.1
    beam: float, optional
        Defaults to 16
    lattice_beam: float, optional
        Defaults to 10
    max_active: int, optional
        Defaults to 7000
    min_active: int, optional
        Defaults to 250
    prune_interval: int, optional
        Defaults to 25
    beam_delta: float, optional
        Defaults to 0.5
    hash_ratio: float, optional
        Defaults to 2.0
    prune_scale: float, optional
        Defaults to 0.1
    boost_silence: float, optional
        Defaults to 1.0

    Returns
    -------
    list[:class:`~kalpy.utterance.Utterance`]
        Split utterances
    """
    graph_compiler = TrainingGraphCompiler(
        acoustic_model.alignment_model_path,
        acoustic_model.tree_path,
        lexicon_compiler,
        lexicon_compiler.word_table,
    )
    if utterance.cmvn_string:
        cmvn = read_kaldi_object(DoubleMatrix, utterance.cmvn_string)
    if utterance.fmllr_string:
        fmllr_trans = read_kaldi_object(FloatMatrix, utterance.fmllr_string)
    if cmvn is None and acoustic_model.uses_cmvn:
        utterance.generate_mfccs(acoustic_model.mfcc_computer)
        cmvn_computer = CmvnComputer()
        cmvn = cmvn_computer.compute_cmvn_from_features([utterance.mfccs])
    current_transcript = utterance.transcript
    if vad_model is None:
        segments = segment_utterance_vad(
            utterance, mfcc_options, vad_options, segmentation_options
        )
    else:
        segments = segment_utterance_vad_speech_brain(utterance, vad_model, segmentation_options)

    config = LatticeFasterDecoderConfig()
    config.beam = beam
    config.lattice_beam = lattice_beam
    config.max_active = max_active
    config.min_active = min_active
    config.prune_interval = prune_interval
    config.beam_delta = beam_delta
    config.hash_ratio = hash_ratio
    config.prune_scale = prune_scale
    new_utts = []
    am, transition_model = acoustic_model.acoustic_model, acoustic_model.transition_model
    if boost_silence != 1.0:
        am.boost_silence(transition_model, lexicon_compiler.silence_symbols, boost_silence)
    for seg in segments:
        new_utt = KalpyUtterance(seg, current_transcript)
        new_utt.generate_mfccs(acoustic_model.mfcc_computer)
        if acoustic_model.uses_cmvn:
            new_utt.apply_cmvn(cmvn)
        feats = new_utt.generate_features(
            acoustic_model.mfcc_computer,
            acoustic_model.pitch_computer,
            lda_mat=acoustic_model.lda_mat,
            fmllr_trans=fmllr_trans,
        )
        unknown_words = []
        unknown_word_index = 0
        for w in new_utt.transcript.split():
            if not lexicon_compiler.word_table.member(w):
                unknown_words.append(w)
        fst = graph_compiler.compile_fst(new_utt.transcript, interjection_words)
        decodable = DecodableAmDiagGmmScaled(am, transition_model, feats, acoustic_scale)

        d = LatticeFasterDecoder(fst, config)
        ans = d.Decode(decodable)
        if not ans:
            raise SegmenterError(f"Did not successfully decode: {current_transcript}")
        ans, decoded = d.GetBestPath()
        if decoded.NumStates() == 0:
            raise SegmenterError("Error getting best path from decoder for utterance")
        alignment, words, weight = GetLinearSymbolSequence(decoded)

        words = words[:-1]
        new_transcript = []
        for w in words:
            w = lexicon_compiler.word_table.find(w)
            if w == lexicon_compiler.oov_word:
                w = unknown_words[unknown_word_index]
                unknown_word_index += 1
            new_transcript.append(w)
        transcript = " ".join(new_transcript)
        if interjection_words:
            current_transcript = align_interjection_words(
                transcript, current_transcript, interjection_words, lexicon_compiler
            )
        else:
            current_transcript = " ".join(current_transcript.split()[len(words) :])
        new_utt.transcript = transcript
        new_utt.mfccs = None
        new_utt.cmvn_string = utterance.cmvn_string
        new_utt.fmllr_string = utterance.fmllr_string
        new_utts.append(new_utt)
    if current_transcript:
        new_utts[-1].transcript += " " + current_transcript
    return new_utts


def align_interjection_words(
    transcript,
    original_transcript,
    interjection_words: typing.List[str],
    lexicon_compiler: LexiconCompiler,
):
    g = pynini.Fst()
    start_state = g.add_state()
    g.set_start(start_state)
    for w in original_transcript.split():
        word_symbol = lexicon_compiler.to_int(w)
        word_initial_state = g.add_state()
        for iw in interjection_words:
            if not lexicon_compiler.word_table.member(iw):
                continue
            iw_symbol = lexicon_compiler.to_int(iw)
            g.add_arc(
                word_initial_state - 1,
                pywrapfst.Arc(
                    iw_symbol,
                    lexicon_compiler.word_table.find("<eps>"),
                    pywrapfst.Weight(g.weight_type(), 4.0),
                    word_initial_state,
                ),
            )
        word_final_state = g.add_state()
        g.add_arc(
            word_initial_state,
            pywrapfst.Arc(
                word_symbol, word_symbol, pywrapfst.Weight.one(g.weight_type()), word_final_state
            ),
        )
        g.add_arc(
            word_initial_state - 1,
            pywrapfst.Arc(
                word_symbol, word_symbol, pywrapfst.Weight.one(g.weight_type()), word_final_state
            ),
        )
        g.set_final(word_initial_state, pywrapfst.Weight.one(g.weight_type()))
        g.set_final(word_final_state, pywrapfst.Weight.one(g.weight_type()))

    a = pynini.accep(
        " ".join(
            [
                x if lexicon_compiler.word_table.member(x) else lexicon_compiler.oov_word
                for x in transcript.split()
            ]
        ),
        token_type=lexicon_compiler.word_table,
    )
    interjections_removed = (
        pynini.compose(a, g).project("output").string(lexicon_compiler.word_table)
    )
    return " ".join(original_transcript.split()[len(interjections_removed.split()) :])


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


def segment_utterance_vad(
    utterance: KalpyUtterance,
    mfcc_options: MetaDict,
    vad_options: MetaDict,
    segmentation_options: MetaDict,
) -> typing.List[Segment]:
    mfcc_computer = MfccComputer(**mfcc_options)
    vad_computer = VadComputer(**vad_options)
    feats = mfcc_computer.compute_mfccs_for_export(utterance.segment, compress=False)
    vad = vad_computer.compute_vad(feats).numpy()
    segments = get_initial_segmentation(vad, mfcc_computer.frame_shift)
    segments = merge_segments(
        segments,
        segmentation_options["close_th"],
        segmentation_options["large_chunk_size"],
        segmentation_options["len_th"],
    )
    new_segments = []
    for s in segments:
        seg = Segment(
            utterance.segment.file_path,
            s.begin + utterance.segment.begin,
            s.end + utterance.segment.begin,
            utterance.segment.channel,
        )
        new_segments.append(seg)
    return new_segments


def segment_utterance_vad_speech_brain(
    utterance: KalpyUtterance, vad_model: VAD, segmentation_options: MetaDict
) -> typing.List[Segment]:
    y = utterance.segment.wave
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
    boundaries += utterance.segment.begin

    # Apply energy-based VAD on the detected speech segments
    if segmentation_options["apply_energy_VAD"]:
        boundaries = vad_model.energy_VAD(
            utterance.segment.file_path,
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
            boundaries, utterance.segment.file_path, speech_th=segmentation_options["speech_th"]
        )
    boundaries[:, 0] -= round(segmentation_options["close_th"] / 2, 3)
    boundaries[:, 1] += round(segmentation_options["close_th"] / 2, 3)
    boundaries = boundaries.numpy()
    segments = []
    for i in range(boundaries.shape[0]):
        begin, end = boundaries[i]
        begin = max(begin, 0)
        end = min(end, utterance.segment.end)
        seg = Segment(
            utterance.segment.file_path, float(begin), float(end), utterance.segment.channel
        )
        segments.append(seg)
    return segments


class SegmentVadFunction(KaldiFunction):
    """
    Multiprocessing function to generate segments from VAD output.

    See Also
    --------
    :meth:`montreal_forced_aligner.segmenter.Segmenter.segment_vad`
        Main function that calls this function in parallel
    :meth:`montreal_forced_aligner.segmenter.VadSegmenter.segment_vad_arguments`
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


class SegmentTranscriptFunction(KaldiFunction):
    """
    Multiprocessing function to segment utterances with transcripts from VAD output.

    See Also
    --------
    :meth:`montreal_forced_aligner.segmenter.Segmenter.segment_vad`
        Main function that calls this function in parallel
    :meth:`montreal_forced_aligner.segmenter.TranscriptionSegmenter.segment_transcript_arguments`
        Job method for generating arguments for this function
    :kaldi_utils:`segmentation.pl`
        Kaldi utility

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.segmenter.SegmentTranscriptArguments`
        Arguments for the function
    """

    def __init__(self, args: SegmentTranscriptArguments):
        super().__init__(args)
        self.acoustic_model = args.acoustic_model
        self.vad_model = args.vad_model
        self.lexicon_compilers = args.lexicon_compilers
        self.segmentation_options = args.segmentation_options
        self.mfcc_options = args.mfcc_options
        self.vad_options = args.vad_options
        self.decode_options = args.decode_options
        self.speechbrain = self.vad_model is not None

    def _run(self):
        """Run the function"""
        with self.session() as session:
            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )

            for d in job.dictionaries:
                utterances = (
                    session.query(Utterance)
                    .join(Utterance.speaker)
                    .options(
                        joinedload(Utterance.file).joinedload(File.sound_file),
                        joinedload(Utterance.speaker),
                    )
                    .filter(
                        Utterance.job_id == self.job_name,
                        Utterance.duration >= 0.1,
                        Speaker.dictionary_id == d.id,
                    )
                    .order_by(Utterance.kaldi_id)
                )
                for u in utterances:
                    new_utterances = segment_utterance_transcript(
                        self.acoustic_model,
                        u.to_kalpy(),
                        self.lexicon_compilers[d.id],
                        self.vad_model if self.speechbrain else None,
                        self.segmentation_options,
                        mfcc_options=self.mfcc_options if not self.speechbrain else None,
                        vad_options=self.vad_options if not self.speechbrain else None,
                        **self.decode_options,
                    )
                    self.callback((u.id, new_utterances))
