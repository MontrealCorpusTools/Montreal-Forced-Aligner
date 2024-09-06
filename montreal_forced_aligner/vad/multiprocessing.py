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
        import torchaudio

        try:
            from speechbrain.pretrained import VAD
        except ImportError:  # speechbrain 1.0
            from speechbrain.inference.VAD import VAD

        class MfaVAD(VAD):
            def energy_VAD(
                self,
                audio_file: typing.Union[str, Path, np.ndarray],
                boundaries,
                activation_th=0.5,
                deactivation_th=0.0,
                eps=1e-6,
            ):
                """Applies energy-based VAD within the detected speech segments.The neural
                network VAD often creates longer segments and tends to merge segments that
                are close with each other.

                The energy VAD post-processes can be useful for having a fine-grained voice
                activity detection.

                The energy VAD computes the energy within the small chunks. The energy is
                normalized within the segment to have mean 0.5 and +-0.5 of std.
                This helps to set the energy threshold.

                Arguments
                ---------
                audio_file: path
                    Path of the audio file containing the recording. The file is read
                    with torchaudio.
                boundaries: torch.Tensor
                    torch.Tensor containing the speech boundaries. It can be derived using the
                    get_boundaries method.
                activation_th: float
                    A new speech segment is started it the energy is above activation_th.
                deactivation_th: float
                    The segment is considered ended when the energy is <= deactivation_th.
                eps: float
                    Small constant for numerical stability.

                Returns
                -------
                new_boundaries
                    The new boundaries that are post-processed by the energy VAD.
                """
                if not isinstance(audio_file, np.ndarray):
                    # Getting the total size of the input file
                    sample_rate, audio_len = self._get_audio_info(audio_file)

                    if sample_rate != self.sample_rate:
                        raise ValueError(
                            "The detected sample rate is different from that set in the hparam file"
                        )
                else:
                    sample_rate = self.sample_rate

                # Computing the chunk length of the energy window
                chunk_len = int(self.time_resolution * sample_rate)
                new_boundaries = []

                # Processing speech segments
                for i in range(boundaries.shape[0]):
                    begin_sample = int(boundaries[i, 0] * sample_rate)
                    end_sample = int(boundaries[i, 1] * sample_rate)
                    seg_len = end_sample - begin_sample

                    if not isinstance(audio_file, np.ndarray):
                        # Reading the speech segment
                        segment, _ = torchaudio.load(
                            audio_file, frame_offset=begin_sample, num_frames=seg_len
                        )
                    else:
                        segment = audio_file[begin_sample : begin_sample + seg_len]

                    # Create chunks
                    segment_chunks = self.create_chunks(
                        segment, chunk_size=chunk_len, chunk_stride=chunk_len
                    )

                    # Energy computation within each chunk
                    energy_chunks = segment_chunks.abs().sum(-1) + eps
                    energy_chunks = energy_chunks.log()

                    # Energy normalization
                    energy_chunks = (
                        (energy_chunks - energy_chunks.mean()) / (2 * energy_chunks.std())
                    ) + 0.5
                    energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

                    # Apply threshold based on the energy value
                    energy_vad = self.apply_threshold(
                        energy_chunks,
                        activation_th=activation_th,
                        deactivation_th=deactivation_th,
                    )

                    # Get the boundaries
                    energy_boundaries = self.get_boundaries(energy_vad, output_value="seconds")

                    # Get the final boundaries in the original signal
                    for j in range(energy_boundaries.shape[0]):
                        start_en = boundaries[i, 0] + energy_boundaries[j, 0]
                        end_end = boundaries[i, 0] + energy_boundaries[j, 1]
                        new_boundaries.append([start_en, end_end])

                # Convert boundaries to tensor
                new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
                return new_boundaries

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError):
    FOUND_SPEECHBRAIN = False
    MfaVAD = None

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


def segment_utterance(
    segment: Segment,
    vad_model: MfaVAD,
    segmentation_options: MetaDict,
    mfcc_options: MetaDict = None,
    vad_options: MetaDict = None,
    allow_empty: bool = True,
) -> typing.List[Segment]:
    """
    Split an utterance and its transcript into multiple transcribed utterances

    Parameters
    ----------
    segment: :class:`~kalpy.data.Segment`
        Segment to split
    vad_model: :class:`~montreal_forced_aligner.vad.multiprocessing.VAD` or None
        VAD model from SpeechBrain, if None, then Kaldi's energy-based VAD is used
    segmentation_options: dict[str, Any]
        Segmentation options
    mfcc_options: dict[str, Any], optional
        MFCC options for energy based VAD
    vad_options: dict[str, Any], optional
        Options for energy based VAD

    Returns
    -------
    list[:class:`~kalpy.data.Segment`]
        Split segments
    """
    if vad_model is None:
        segments = segment_utterance_vad(segment, mfcc_options, vad_options, segmentation_options)
    else:
        segments = segment_utterance_vad_speech_brain(
            segment, vad_model, segmentation_options, allow_empty=allow_empty
        )
    if not segments:
        return [segment]
    return segments


def segment_utterance_transcript(
    acoustic_model: AcousticModel,
    utterance: KalpyUtterance,
    lexicon_compiler: LexiconCompiler,
    vad_model: MfaVAD,
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
    segments = segment_utterance(
        utterance.segment, vad_model, segmentation_options, mfcc_options, vad_options
    )
    if not segments:
        return [utterance]
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
    segment: Segment,
    mfcc_options: MetaDict,
    vad_options: MetaDict,
    segmentation_options: MetaDict,
    adaptive: bool = True,
    allow_empty: bool = True,
) -> typing.List[Segment]:
    mfcc_options["use_energy"] = True
    mfcc_options["raw_energy"] = False
    mfcc_options["dither"] = 0.0
    mfcc_options["energy_floor"] = 0.0
    mfcc_computer = MfccComputer(**mfcc_options)
    feats = mfcc_computer.compute_mfccs_for_export(segment, compress=False)
    if adaptive:
        vad_options["energy_mean_scale"] = 0.0
        mfccs = feats.numpy()
        vad_options["energy_threshold"] = mfccs[:, 0].mean()
    vad_computer = VadComputer(**vad_options)
    vad = vad_computer.compute_vad(feats).numpy()
    segments = get_initial_segmentation(vad, mfcc_computer.frame_shift)
    segments = merge_segments(
        segments,
        segmentation_options["close_th"],
        segmentation_options["large_chunk_size"],
        segmentation_options["len_th"] if allow_empty else 0.02,
    )
    new_segments = []
    for s in segments:
        seg = Segment(
            segment.file_path,
            s.begin + segment.begin,
            s.end + segment.begin,
            segment.channel,
        )
        new_segments.append(seg)
    return new_segments


def segment_utterance_vad_speech_brain(
    segment: Segment,
    vad_model: MfaVAD,
    segmentation_options: MetaDict,
    allow_empty: bool = True,
) -> typing.List[Segment]:
    y = segment.wave
    prob_chunks = vad_model.get_speech_prob_chunk(torch.tensor(y[np.newaxis, :])).float()
    prob_th = vad_model.apply_threshold(
        prob_chunks,
        activation_th=segmentation_options["activation_th"],
        deactivation_th=segmentation_options["deactivation_th"],
    ).float()

    # Compute the boundaries of the speech segments
    boundaries = vad_model.get_boundaries(prob_th, output_value="seconds").cpu()
    if segment.begin is not None:
        boundaries += segment.begin
    # Apply energy-based VAD on the detected speech segments
    if segmentation_options["apply_energy_VAD"]:
        vad_boundaries = vad_model.energy_VAD(
            segment.file_path,
            boundaries,
            activation_th=segmentation_options["en_activation_th"],
            deactivation_th=segmentation_options["en_deactivation_th"],
        )
        if vad_boundaries.size(0) != 0 or allow_empty:
            boundaries = vad_boundaries

    # Merge short segments
    boundaries = vad_model.merge_close_segments(
        boundaries, close_th=segmentation_options["close_th"]
    )

    # Remove short segments
    filtered_boundaries = vad_model.remove_short_segments(
        boundaries, len_th=segmentation_options["len_th"]
    )
    if filtered_boundaries.size(0) != 0 or allow_empty:
        boundaries = filtered_boundaries

    # Double check speech segments
    if segmentation_options["double_check"]:
        checked_boundaries = vad_model.double_check_speech_segments(
            boundaries, segment.file_path, speech_th=segmentation_options["speech_th"]
        )
        if checked_boundaries.size(0) != 0 or allow_empty:
            boundaries = checked_boundaries
    boundaries[:, 0] -= round(segmentation_options["close_th"] / 2, 3)
    boundaries[:, 1] += round(segmentation_options["close_th"] / 2, 3)
    segments = []
    for i in range(boundaries.numpy().shape[0]):
        begin, end = boundaries[i]
        if i == 0:
            begin = max(begin, 0)
        if i == boundaries.numpy().shape[0] - 1:
            end = min(end, segment.end)
        seg = Segment(segment.file_path, float(begin), float(end), segment.channel)
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
