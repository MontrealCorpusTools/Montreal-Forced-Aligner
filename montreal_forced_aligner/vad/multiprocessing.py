"""Multiprocessing functionality for VAD"""
from __future__ import annotations

import typing
from pathlib import Path
from typing import TYPE_CHECKING, Union

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
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import File, Job, Speaker, Utterance
from montreal_forced_aligner.exceptions import SegmenterError
from montreal_forced_aligner.models import AcousticModel, G2PModel
from montreal_forced_aligner.vad.models import MfaVAD, get_initial_segmentation, merge_segments

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
    "segment_utterance_transcript",
    "segment_utterance_vad",
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
    vad_model: typing.Optional[MfaVAD]
    lexicon_compilers: typing.Dict[int, LexiconCompiler]
    mfcc_options: MetaDict
    vad_options: MetaDict
    segmentation_options: MetaDict
    decode_options: MetaDict


def segment_utterance(
    segment: Segment,
    vad_model: typing.Optional[MfaVAD],
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
        segments = vad_model.segment_utterance(
            segment, **segmentation_options, allow_empty=allow_empty
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
        segmentation_options["min_pause_duration"],
        segmentation_options["max_segment_length"],
        segmentation_options["min_segment_length"] if allow_empty else 0.02,
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
                self.segmentation_options["min_pause_duration"],
                self.segmentation_options["max_segment_length"],
                self.segmentation_options["min_segment_length"],
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
