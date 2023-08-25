"""Classes for calculating alignments online"""
from __future__ import annotations

from _kalpy.matrix import DoubleMatrix, FloatMatrix
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import HierarchicalCtm
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner.exceptions import AlignerError
from montreal_forced_aligner.models import AcousticModel


def align_utterance_online(
    acoustic_model: AcousticModel,
    utterance: KalpyUtterance,
    lexicon_compiler: LexiconCompiler,
    cmvn: DoubleMatrix = None,
    fmllr_trans: FloatMatrix = None,
    beam: int = 10,
    retry_beam: int = 40,
    transition_scale: float = 1.0,
    acoustic_scale: float = 0.1,
    self_loop_scale: float = 0.1,
    boost_silence: float = 1.0,
) -> HierarchicalCtm:
    graph_compiler = TrainingGraphCompiler(
        acoustic_model.alignment_model_path,
        acoustic_model.tree_path,
        lexicon_compiler,
        lexicon_compiler.word_table,
    )
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
    fst = graph_compiler.compile_fst(utterance.transcript)
    aligner = GmmAligner(
        acoustic_model.alignment_model_path if fmllr_trans is None else acoustic_model.model_path,
        beam=beam,
        retry_beam=retry_beam,
        transition_scale=transition_scale,
        acoustic_scale=acoustic_scale,
        self_loop_scale=self_loop_scale,
    )
    if boost_silence != 1.0:
        aligner.boost_silence(boost_silence, lexicon_compiler.silence_symbols)
    alignment = aligner.align_utterance(fst, feats)
    if alignment is None:
        raise AlignerError(
            f"Could not align the file with the current beam size ({aligner.beam}, "
            "please try increasing the beam size via `--beam X`"
        )
    phone_intervals = alignment.generate_ctm(
        aligner.transition_model,
        lexicon_compiler.phone_table,
        acoustic_model.mfcc_computer.frame_shift,
    )
    ctm = lexicon_compiler.phones_to_pronunciations(
        utterance.transcript, alignment.words, phone_intervals, transcription=False
    )
    ctm.likelihood = alignment.likelihood
    ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
    return ctm
