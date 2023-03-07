"""Multiprocessing functionality for VAD"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import typing
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import librosa
import numpy as np
import pynini
import pywrapfst
import sqlalchemy
from Bio import pairwise2

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.corpus.features import online_feature_proc
from montreal_forced_aligner.data import CtmInterval, MfaArguments, WordType
from montreal_forced_aligner.db import Dictionary, File, SoundFile, Speaker, Utterance, Word
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import parse_ctm_output, read_feats, thirdparty_binary

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
    segments = []
    cur_segment = None
    silent_frames = 0
    non_silent_frames = 0
    for i, f in enumerate(frames):
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


def segment_utterance(
    session: sqlalchemy.orm.Session,
    working_directory: Path,
    utterance_id: int,
    vad_model: VAD,
    segmentation_options: MetaDict,
    mfcc_options: MetaDict,
    pitch_options: MetaDict,
    lda_options: MetaDict,
    decode_options: MetaDict,
):
    log_path = working_directory.joinpath("log", "utterance_segmentation.log")
    utterance, speaker, dictionary, sound_file = (
        session.query(Utterance, Speaker, Dictionary, SoundFile)
        .join(Utterance.speaker)
        .join(Speaker.dictionary)
        .join(Utterance.file)
        .join(File.sound_file)
        .filter(Utterance.id == utterance_id)
        .first()
    )

    text = utterance.normalized_text
    if not text:
        text = utterance.text
    oovs = utterance.oovs.split()
    normalized_text = " ".join([x if x not in oovs else dictionary.oov_word for x in text.split()])
    words = set(normalized_text.split() + [dictionary.bracketed_word])
    interjection_words = (
        session.query(Word.word)
        .filter(Word.dictionary_id == dictionary.id)
        .filter(Word.word_type == WordType.interjection)
        .all()
    )
    words.update(interjection_words)
    query = session.query(Word.word, Word.mapping_id, Word.initial_cost, Word.final_cost).filter(
        Word.dictionary_id == dictionary.id
    )
    initial_costs = {}
    final_costs = {}
    reversed_word_mapping = {}
    for w, m_id, ic, fc in query:
        reversed_word_mapping[m_id] = w
        if w not in words:
            continue
        if ic is not None:
            initial_costs[w] = ic
        if fc is not None:
            final_costs[w] = fc
    segments = segment_utterance_vad_speech_brain(
        utterance, sound_file, vad_model, segmentation_options
    )
    word_symbol_table = pywrapfst.SymbolTable.read_text(dictionary.words_symbol_path)
    utterance_fst_path = working_directory.joinpath("utterance.fst")
    utterance_fst = construct_utterance_segmentation_fst(
        normalized_text,
        word_symbol_table,
        interjection_words=interjection_words,
    )

    utterance_fst.write(utterance_fst_path)
    wav_path = working_directory.joinpath("wav.scp")
    segment_path = working_directory.joinpath("segments.scp")
    utt2spk_path = working_directory.joinpath("utt2spk.scp")
    cmvn_path = working_directory.joinpath("cmvn.scp")
    trans_path = working_directory.joinpath("trans.scp")
    with mfa_open(wav_path, "w") as f:
        f.write(f"{utterance.file_id} {sound_file.sox_string}\n")
    if speaker.cmvn:
        with mfa_open(cmvn_path, "w") as f:
            f.write(f"{utterance.speaker_id} {speaker.cmvn}\n")
    if speaker.fmllr:
        with mfa_open(trans_path, "w") as f:
            f.write(f"{utterance.speaker_id} {speaker.fmllr}\n")
    sub_utterance_information = {}
    with mfa_open(segment_path, "w") as f, mfa_open(utt2spk_path, "w") as utt2spk_f:
        for i in range(segments.shape[0]):
            begin, end = segments[i]
            begin = max(begin - 0.05, 0)
            f.write(
                f"{utterance.speaker_id}-{i} {utterance.file_id} {begin} {end} {utterance.channel}\n"
            )
            utt2spk_f.write(f"{utterance.speaker_id}-{i} {utterance.speaker_id}\n")
            sub_utterance_information[i] = {
                "file_id": utterance.file_id,
                "begin": float(begin),
                "end": float(end),
                "channel": utterance.channel,
                "speaker_id": utterance.speaker_id,
            }
    model_path = working_directory.joinpath("final.alimdl")
    tree_path = working_directory.joinpath("tree")
    hclg_path = working_directory.joinpath("hclg.fst")
    with open(working_directory.joinpath("utterance.text_fst"), "w", encoding="utf8") as f:
        utterance_fst.set_input_symbols(word_symbol_table)
        utterance_fst.set_output_symbols(word_symbol_table)
        f.write(str(utterance_fst))
    with mfa_open(log_path, "w") as log_file:

        proc = subprocess.Popen(
            [
                thirdparty_binary("compile-graph"),
                f"--read-disambig-syms={dictionary.disambiguation_symbols_int_path}",
                f"--transition-scale={decode_options['transition_scale']}",
                f"--self-loop-scale={decode_options['self_loop_scale']}",
                tree_path,
                model_path,
                dictionary.lexicon_disambig_fst_path,
                utterance_fst_path,
                f"{hclg_path}",
            ],
            stderr=log_file,
            env=os.environ,
        )
        proc.wait()
        if proc.returncode != 0:
            raise KaldiProcessingError([log_path])
        feature_proc = online_feature_proc(
            working_directory,
            wav_path,
            segment_path,
            mfcc_options,
            pitch_options,
            lda_options,
            log_file,
        )
        if decode_options.get("boost_silence", 1.0) != 1.0:
            mdl_string = f"gmm-boost-silence --boost={decode_options['boost_silence']} {decode_options['optional_silence_csl']} {model_path} - |"

        else:
            mdl_string = model_path
        latgen_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-latgen-faster"),
                f"--acoustic-scale={decode_options['acoustic_scale']}",
                f"--beam={decode_options['beam']}",
                f"--max-active={decode_options['max_active']}",
                f"--lattice-beam={decode_options['lattice_beam']}",
                f"--word-symbol-table={dictionary.words_symbol_path}",
                "--allow-partial=true",
                mdl_string,
                hclg_path,
                "ark,s,cs:-",
                "ark:-",
            ],
            stderr=log_file,
            stdin=feature_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        lat_align_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-align-words-lexicon"),
                dictionary.align_lexicon_int_path,
                mdl_string,
                "ark,s,cs:-",
                "ark:-",
            ],
            stderr=log_file,
            stdin=latgen_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        ctm_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-to-ctm-conf"),
                f"--acoustic-scale={decode_options['acoustic_scale']}",
                "ark,s,cs:-",
                "-",
            ],
            stderr=log_file,
            stdin=lat_align_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
            encoding="utf8",
        )
        split_texts = {}
        for sub_id, intervals in parse_ctm_output(ctm_proc, reversed_word_mapping):
            split_text = " ".join([x.label for x in intervals if x.confidence == 1.0])
            if not split_text:
                del sub_utterance_information[sub_id]
                continue
            split_texts[sub_id] = split_text
        ctm_proc.wait()
        split_texts = align_text(split_texts, text, oovs, dictionary.oov_word, interjection_words)
        for i, split_text in split_texts.items():
            split_oovs = set(x for x in oovs if x in split_text.split())
            sub_utterance_information[i]["text"] = split_text
            sub_utterance_information[i]["oovs"] = " ".join(split_oovs)
            sub_utterance_information[i]["normalized_text"] = split_text
        sub_utterance_information = {
            k: v for k, v in sub_utterance_information.items() if "text" in v
        }
    return utterance_id, sub_utterance_information


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
