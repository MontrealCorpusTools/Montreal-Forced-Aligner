"""Classes for calculating alignments online"""
from __future__ import annotations

from _kalpy.fstext import ConstFst
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.data import HierarchicalCtm
from kalpy.gmm.decode import GmmDecoder
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner.exceptions import AlignerError
from montreal_forced_aligner.models import AcousticModel


def transcribe_utterance_online(
    acoustic_model: AcousticModel,
    utterance: KalpyUtterance,
    lexicon_compiler: LexiconCompiler,
    hclg_fst: ConstFst,
    cmvn: DoubleMatrix = None,
    fmllr_trans: FloatMatrix = None,
    acoustic_scale: float = 0.1,
    boost_silence: float = 1.0,
    beam: int = 10,
    lattice_beam: int = 10,
    max_active: int = 7000,
    min_active: int = 200,
    prune_interval: int = 25,
    beam_delta: float = 0.5,
    hash_ratio: float = 2.0,
    prune_scale: float = 0.1,
    allow_partial: bool = True,
) -> HierarchicalCtm:
    if utterance.mfccs is None:
        utterance.generate_mfccs(acoustic_model.mfcc_computer)
        if acoustic_model.uses_cmvn:
            if cmvn is None:
                cmvn_computer = CmvnComputer()
                cmvn = cmvn_computer.compute_cmvn_from_features([utterance.mfccs])
            utterance.apply_cmvn(cmvn)
    feats = utterance.generate_features(
        acoustic_model.mfcc_computer,
        acoustic_model.pitch_computer,
        lda_mat=acoustic_model.lda_mat,
        fmllr_trans=fmllr_trans,
    )
    decoder = GmmDecoder(
        acoustic_model.alignment_model_path,
        hclg_fst,
        acoustic_scale=acoustic_scale,
        beam=beam,
        lattice_beam=lattice_beam,
        max_active=max_active,
        min_active=min_active,
        prune_interval=prune_interval,
        beam_delta=beam_delta,
        hash_ratio=hash_ratio,
        prune_scale=prune_scale,
        allow_partial=allow_partial,
        fast=True,
    )
    if boost_silence != 1.0:
        decoder.boost_silence(boost_silence, lexicon_compiler.silence_symbols)
    alignment = decoder.decode_utterance(feats)
    if alignment is None:
        raise AlignerError(
            f"Could not transcribe the file with the current beam size ({decoder.beam}, "
            "please try increasing the beam size via `--beam X`"
        )
    phone_intervals = alignment.generate_ctm(
        decoder.transition_model,
        lexicon_compiler.phone_table,
        acoustic_model.mfcc_computer.frame_shift,
    )
    ctm = lexicon_compiler.phones_to_pronunciations(
        alignment.words,
        phone_intervals,
        transcription=False,
        text=utterance.transcript,
    )
    ctm.likelihood = alignment.likelihood
    ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
    return ctm
