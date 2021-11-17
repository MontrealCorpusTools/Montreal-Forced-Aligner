"""
Multiprocessing classes
-----------------------

"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Collection, Dict, List, NamedTuple, Optional, Set, Tuple

if TYPE_CHECKING:
    from ..corpus.classes import File, Speaker, Utterance

from ..abc import IvectorExtractor, MetaDict, MfaWorker
from ..helper import output_mapping, save_scp

if TYPE_CHECKING:
    from ..abc import Aligner, MappingType, ReversedMappingType, WordsType
    from ..aligner.adapting import AdaptingAligner
    from ..aligner.base import BaseAligner
    from ..config import FeatureConfig
    from ..corpus import Corpus
    from ..dictionary import DictionaryData
    from ..segmenter import Segmenter
    from ..trainers import (
        BaseTrainer,
        IvectorExtractorTrainer,
        LdaTrainer,
        MonophoneTrainer,
        SatTrainer,
    )
    from ..transcriber import Transcriber
    from ..validator import CorpusValidator


__all__ = [
    "Job",
    "AlignArguments",
    "VadArguments",
    "SegmentVadArguments",
    "CreateHclgArguments",
    "AccGlobalStatsArguments",
    "AccStatsArguments",
    "AccIvectorStatsArguments",
    "AccStatsTwoFeatsArguments",
    "AliToCtmArguments",
    "MfccArguments",
    "ScoreArguments",
    "DecodeArguments",
    "PhoneCtmArguments",
    "CombineCtmArguments",
    "CleanupWordCtmArguments",
    "NoCleanupWordCtmArguments",
    "LmRescoreArguments",
    "AlignmentImprovementArguments",
    "ConvertAlignmentsArguments",
    "CalcFmllrArguments",
    "CalcLdaMlltArguments",
    "GmmGselectArguments",
    "FinalFmllrArguments",
    "LatGenFmllrArguments",
    "FmllrRescoreArguments",
    "TreeStatsArguments",
    "LdaAccStatsArguments",
    "MapAccStatsArguments",
    "GaussToPostArguments",
    "InitialFmllrArguments",
    "ExtractIvectorsArguments",
    "ExportTextGridArguments",
    "CompileTrainGraphsArguments",
    "CompileInformationArguments",
    "CompileUtteranceTrainGraphsArguments",
    "MonoAlignEqualArguments",
    "TestUtterancesArguments",
    "CarpaLmRescoreArguments",
    "GeneratePronunciationsArguments",
]


class VadArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.features.compute_vad_func`"""

    log_path: str
    dictionaries: List[str]
    feats_scp_paths: Dict[str, str]
    vad_scp_paths: Dict[str, str]
    vad_options: MetaDict


class MfccArguments(NamedTuple):
    """
    Arguments for :func:`~montreal_forced_aligner.multiprocessing.features.mfcc_func`
    """

    log_path: str
    dictionaries: List[str]
    feats_scp_paths: Dict[str, str]
    lengths_paths: Dict[str, str]
    segment_paths: Dict[str, str]
    wav_paths: Dict[str, str]
    mfcc_options: MetaDict


class CompileTrainGraphsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compile_train_graphs_func`"""

    log_path: str
    dictionaries: List[str]
    tree_path: str
    model_path: str
    text_int_paths: Dict[str, str]
    disambig_paths: Dict[str, str]
    lexicon_fst_paths: Dict[str, str]
    fst_scp_paths: Dict[str, str]


class MonoAlignEqualArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.mono_align_equal_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    fst_scp_paths: Dict[str, str]
    ali_ark_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str


class AccStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.acc_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str


class AlignArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.align_func`"""

    log_path: str
    dictionaries: List[str]
    fst_scp_paths: Dict[str, str]
    feature_strings: Dict[str, str]
    model_path: str
    ali_paths: Dict[str, str]
    score_paths: Dict[str, str]
    loglike_paths: Dict[str, str]
    align_options: MetaDict


class CompileInformationArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compile_information_func`"""

    align_log_paths: str


class AliToCtmArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.ali_to_ctm_func`"""

    log_path: str
    dictionaries: List[str]
    ali_paths: Dict[str, str]
    text_int_paths: Dict[str, str]
    word_boundary_int_paths: Dict[str, str]
    frame_shift: float
    model_path: str
    ctm_paths: Dict[str, str]
    word_mode: bool


class CleanupWordCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.multiprocessing.alignment.CleanupWordCtmProcessWorker`"""

    ctm_paths: Dict[str, str]
    dictionaries: List[str]
    utterances: Dict[str, Dict[str, Utterance]]
    dictionary_data: Dict[str, DictionaryData]


class NoCleanupWordCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.multiprocessing.alignment.NoCleanupWordCtmProcessWorker`"""

    ctm_paths: Dict[str, str]
    dictionaries: List[str]
    utterances: Dict[str, Dict[str, Utterance]]
    dictionary_data: Dict[str, DictionaryData]


class PhoneCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.multiprocessing.alignment.PhoneCtmProcessWorker`"""

    ctm_paths: Dict[str, str]
    dictionaries: List[str]
    utterances: Dict[str, Dict[str, Utterance]]
    reversed_phone_mappings: Dict[str, ReversedMappingType]
    positions: Dict[str, List[str]]


class CombineCtmArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.multiprocessing.alignment.CombineProcessWorker`"""

    dictionaries: List[str]
    files: Dict[str, File]
    dictionary_data: Dict[str, DictionaryData]
    cleanup_textgrids: bool


class ExportTextGridArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.multiprocessing.alignment.ExportTextGridProcessWorker`"""

    files: Dict[str, File]
    frame_shift: int
    output_directory: str
    backup_output_directory: str


class TreeStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.tree_stats_func`"""

    log_path: str
    dictionaries: List[str]
    ci_phones: str
    model_path: str
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    treeacc_paths: Dict[str, str]


class ConvertAlignmentsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.convert_alignments_func`"""

    log_path: str
    dictionaries: List[str]
    model_path: str
    tree_path: str
    align_model_path: str
    ali_paths: Dict[str, str]
    new_ali_paths: Dict[str, str]


class AlignmentImprovementArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compute_alignment_improvement_func`"""

    log_path: str
    dictionaries: List[str]
    model_path: str
    text_int_paths: Dict[str, str]
    word_boundary_paths: Dict[str, str]
    ali_paths: Dict[str, str]
    frame_shift: int
    reversed_phone_mappings: Dict[str, Dict[int, str]]
    positions: Dict[str, List[str]]
    phone_ctm_paths: Dict[str, str]


class CalcFmllrArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.calc_fmllr_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    ali_model_path: str
    model_path: str
    spk2utt_paths: Dict[str, str]
    trans_paths: Dict[str, str]
    fmllr_options: MetaDict


class AccStatsTwoFeatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.acc_stats_two_feats_func`"""

    log_path: str
    dictionaries: List[str]
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str
    feature_strings: Dict[str, str]
    si_feature_strings: Dict[str, str]


class LdaAccStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.lda_acc_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    lda_options: MetaDict
    acc_paths: Dict[str, str]


class CalcLdaMlltArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.calc_lda_mllt_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    lda_options: MetaDict
    macc_paths: Dict[str, str]


class MapAccStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.map_acc_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]


class GmmGselectArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.gmm_gselect_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    dubm_model: str
    gselect_paths: Dict[str, str]


class AccGlobalStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.acc_global_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    gselect_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    dubm_path: str


class GaussToPostArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.gauss_to_post_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    post_paths: Dict[str, str]
    dubm_path: str


class AccIvectorStatsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.acc_ivector_stats_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    ie_path: str
    post_paths: Dict[str, str]
    acc_init_paths: Dict[str, str]


class ExtractIvectorsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.extract_ivectors_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: MetaDict
    ali_paths: Dict[str, str]
    ie_path: str
    ivector_paths: Dict[str, str]
    weight_paths: Dict[str, str]
    model_path: str
    dubm_path: str


class CompileUtteranceTrainGraphsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compile_utterance_train_graphs_func`"""

    log_path: str
    dictionaries: List[str]
    disambig_int_paths: Dict[str, str]
    disambig_L_fst_paths: Dict[str, str]
    fst_paths: Dict[str, str]
    graphs_paths: Dict[str, str]
    model_path: str
    tree_path: str


class TestUtterancesArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.test_utterances_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    words_paths: Dict[str, str]
    graphs_paths: Dict[str, str]
    text_int_paths: Dict[str, str]
    edits_paths: Dict[str, str]
    out_int_paths: Dict[str, str]
    model_path: str


class SegmentVadArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.segment_vad_func`"""

    dictionaries: List[str]
    vad_paths: Dict[str, str]
    segmentation_options: MetaDict


class GeneratePronunciationsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.pronunciations.generate_pronunciations_func`"""

    log_path: str
    dictionaries: List[str]
    text_int_paths: Dict[str, str]
    word_boundary_paths: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    pron_paths: Dict[str, str]


class CreateHclgArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.create_hclg_func`"""

    log_path: str
    working_directory: str
    path_template: str
    words_path: str
    carpa_path: str
    small_arpa_path: str
    medium_arpa_path: str
    big_arpa_path: str
    model_path: str
    disambig_L_path: str
    disambig_int_path: str
    hclg_options: MetaDict
    words_mapping: MappingType

    @property
    def hclg_path(self) -> str:
        return self.path_template.format(file_name="HCLG")


class DecodeArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.decode_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    decode_options: MetaDict
    model_path: str
    lat_paths: Dict[str, str]
    words_paths: Dict[str, str]
    hclg_paths: Dict[str, str]


class ScoreArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.score_func`"""

    log_path: str
    dictionaries: List[str]
    score_options: MetaDict
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    carpa_rescored_lat_paths: Dict[str, str]
    words_paths: Dict[str, str]
    tra_paths: Dict[str, str]


class LmRescoreArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.lm_rescore_func`"""

    log_path: str
    dictionaries: List[str]
    lm_rescore_options: MetaDict
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    old_g_paths: Dict[str, str]
    new_g_paths: Dict[str, str]


class CarpaLmRescoreArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.carpa_lm_rescore_func`"""

    log_path: str
    dictionaries: List[str]
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    old_g_paths: Dict[str, str]
    new_g_paths: Dict[str, str]


class InitialFmllrArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.initial_fmllr_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: MetaDict
    pre_trans_paths: Dict[str, str]
    lat_paths: Dict[str, str]
    spk2utt_paths: Dict[str, str]


class LatGenFmllrArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.lat_gen_fmllr_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    decode_options: MetaDict
    words_paths: Dict[str, str]
    hclg_paths: Dict[str, str]
    tmp_lat_paths: Dict[str, str]


class FinalFmllrArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.final_fmllr_est_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: MetaDict
    trans_paths: Dict[str, str]
    spk2utt_paths: Dict[str, str]
    tmp_lat_paths: Dict[str, str]


class FmllrRescoreArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.fmllr_rescore_func`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: MetaDict
    tmp_lat_paths: Dict[str, str]
    final_lat_paths: Dict[str, str]


class Job:
    """
    Class representing information about corpus jobs that will be run in parallel.
    Jobs have a set of speakers that they will process, along with all files and utterances associated with that speaker.
    As such, Jobs also have a set of dictionaries that the speakers use, and argument outputs are largely dependent on
    the pronunciation dictionaries in use.

    Parameters
    ----------
    name: int
        Job number is the job's identifier

    Attributes
    ----------
    speakers: List[:class:`~montreal_forced_aligner.corpus.Speaker`]
        List of speakers associated with this job
    dictionaries: Set[:class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`]
        Set of dictionaries that the job's speakers use
    subset_utts: Set[:class:`~montreal_forced_aligner.corpus.Utterance`]
        When trainers are just using a subset of the corpus, the subset of utterances on each job will be set and used to
        filter the job's utterances
    subset_speakers: Set[:class:`~montreal_forced_aligner.corpus.Speaker`]
        When subset_utts is set, this property will be calculated as the subset of speakers that the utterances correspond to
    subset_dictionaries: Set[:class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`]
        Subset of dictionaries that the subset of speakers use

    """

    def __init__(self, name: int):
        self.name = name
        self.speakers: List[Speaker] = []
        self.dictionaries = set()

        self.subset_utts = set()
        self.subset_speakers = set()
        self.subset_dictionaries = set()

    def add_speaker(self, speaker: Speaker) -> None:
        """
        Add a speaker to a job

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.Speaker`
            Speaker to add
        """
        self.speakers.append(speaker)
        self.dictionaries.add(speaker.dictionary)

    def set_subset(self, subset_utts: Optional[Collection[Utterance]]) -> None:
        """
        Set the current subset for the trainer

        Parameters
        ----------
        subset_utts: Collection[:class:`~montreal_forced_aligner.corpus.Utterance`], optional
            Subset of utterances for this job to use
        """
        if subset_utts is None:
            self.subset_utts = set()
            self.subset_speakers = set()
            self.subset_dictionaries = set()
        else:
            self.subset_utts = set(subset_utts)
            self.subset_speakers = {u.speaker for u in subset_utts if u.speaker in self.speakers}
            self.subset_dictionaries = {s.dictionary for s in self.subset_speakers}

    def text_scp_data(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate the job's data for Kaldi's text scp files

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            Text for each utterance, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if not u.text:
                    continue
                data[key][u.name] = u.text_for_scp()
        return data

    def text_int_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's text int scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Text converted to integer IDs for each utterance, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                continue
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if u.ignored:
                    continue
                if not u.text:
                    continue
                data[key][u.name] = " ".join(map(str, u.text_int_for_scp()))
        return data

    def wav_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's wav scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Wav scp strings for each file, per dictionary name
        """
        data = {}
        done = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
                done[key] = set()
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if not u.is_segment:
                    data[key][u.name] = u.file.for_wav_scp()
                elif u.file.name not in done:
                    data[key][u.file.name] = u.file.for_wav_scp()
                    done[key].add(u.file.name)
        return data

    def utt2spk_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's utt2spk scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Utterance to speaker mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                data[key][u.name] = s.name
        return data

    def feat_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's feature scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Utterance to feature archive ID mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if u.features:
                    data[key][u.name] = u.features
        return data

    def spk2utt_scp_data(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate the job's data for Kaldi's spk2utt scp files

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            Speaker to utterance mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            data[key][s.name] = sorted(
                [
                    u.name
                    for u in s.utterances.values()
                    if not u.ignored and not (self.subset_utts and u not in self.subset_utts)
                ]
            )
        return data

    def cmvn_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's CMVN scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Speaker to CMVN mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            if s.cmvn:
                data[key][s.name] = s.cmvn
        return data

    def segments_scp_data(self) -> Dict[str, Dict[str, str]]:
        """
        Generate the job's data for Kaldi's segments scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Utterance to segment mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if not u.is_segment:
                    continue
                data[key][u.name] = u.segment_for_scp()
        return data

    def construct_path_dictionary(
        self, directory: str, identifier: str, extension: str
    ) -> Dict[str, str]:
        """
        Helper function for constructing dictionary-dependent paths for the Job

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        Dict[str, str]
            Path for each dictionary
        """
        output = {}
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(
                directory, f"{identifier}.{dict_name}.{self.name}.{extension}"
            )
        return output

    def construct_dictionary_dependent_paths(
        self, directory: str, identifier: str, extension: str
    ) -> Dict[str, str]:
        """
        Helper function for constructing paths that depend only on the dictionaries of the job, and not the job name itself.
        These paths should be merged with all other jobs to get a full set of dictionary paths.

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        Dict[str, str]
            Path for each dictionary
        """
        output = {}
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(directory, f"{identifier}.{dict_name}.{extension}")
        return output

    @property
    def dictionary_count(self):
        """Number of dictionaries currently used"""
        if self.subset_dictionaries:
            return len(self.subset_dictionaries)
        return len(self.dictionaries)

    @property
    def current_dictionaries(self):
        """Current dictionaries depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return self.subset_dictionaries
        return self.dictionaries

    @property
    def current_dictionary_names(self):
        """Current dictionary names depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return sorted(x.name for x in self.subset_dictionaries)
        if self.dictionaries == {None}:
            return [None]
        return sorted(x.name for x in self.dictionaries)

    def set_feature_config(self, feature_config: FeatureConfig) -> None:
        """
        Set the feature configuration to use for the Job

        Parameters
        ----------
        feature_config: :class:`~montreal_forced_aligner.config.FeatureConfig`
            Feature configuration
        """
        self.feature_config = feature_config

    def construct_base_feature_string(self, corpus: Corpus, all_feats: bool = False) -> str:
        """
        Construct the base feature string independent of job name

        Used in initialization of MonophoneTrainer (to get dimension size) and IvectorTrainer (uses all feats)

        Parameters
        ----------
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus to use as the source
        all_feats: bool
            Flag for whether all features across all jobs should be taken into account

        Returns
        -------
        str
            Feature string
        """
        if all_feats:
            feat_path = os.path.join(corpus.output_directory, "feats.scp")
            utt2spk_path = os.path.join(corpus.output_directory, "utt2spk.scp")
            cmvn_path = os.path.join(corpus.output_directory, "cmvn.scp")
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            feats += " add-deltas ark:- ark:- |"
            return feats
        utt2spks = self.construct_path_dictionary(corpus.split_directory, "utt2spk", "scp")
        cmvns = self.construct_path_dictionary(corpus.split_directory, "cmvn", "scp")
        features = self.construct_path_dictionary(corpus.split_directory, "feats", "scp")
        for dict_name in self.current_dictionary_names:
            feat_path = features[dict_name]
            cmvn_path = cmvns[dict_name]
            utt2spk_path = utt2spks[dict_name]
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            if self.feature_config.deltas:
                feats += " add-deltas ark:- ark:- |"

            return feats

    def construct_feature_proc_strings(
        self,
        aligner: MfaWorker,
        speaker_independent: bool = False,
    ) -> Dict[str, str]:
        """
        Constructs a feature processing string to supply to Kaldi binaries, taking into account corpus features and the
        current working directory of the aligner (whether fMLLR or LDA transforms should be used, etc).

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.abc.MfaWorker`
            Aligner, Transcriber or other main utility class that uses the features
        speaker_independent: bool
            Flag for whether features should be speaker-independent regardless of the presence of fMLLR transforms

        Returns
        -------
        Dict[str, str]
            Feature strings per dictionary name
        """
        lda_mat_path = None
        fmllrs = {}
        if aligner.working_directory is not None:
            lda_mat_path = os.path.join(aligner.working_directory, "lda.mat")
            if not os.path.exists(lda_mat_path):
                lda_mat_path = None

            fmllrs = self.construct_path_dictionary(aligner.working_directory, "trans", "ark")
        utt2spks = self.construct_path_dictionary(aligner.data_directory, "utt2spk", "scp")
        cmvns = self.construct_path_dictionary(aligner.data_directory, "cmvn", "scp")
        features = self.construct_path_dictionary(aligner.data_directory, "feats", "scp")
        vads = self.construct_path_dictionary(aligner.data_directory, "vad", "scp")
        feat_strings = {}
        for dict_name in self.current_dictionary_names:
            feat_path = features[dict_name]
            cmvn_path = cmvns[dict_name]
            utt2spk_path = utt2spks[dict_name]
            fmllr_trans_path = None
            try:
                fmllr_trans_path = fmllrs[dict_name]
                if not os.path.exists(fmllr_trans_path):
                    fmllr_trans_path = None
            except KeyError:
                pass
            vad_path = vads[dict_name]
            if aligner.uses_voiced:
                feats = f"ark,s,cs:add-deltas scp:{feat_path} ark:- |"
                if aligner.uses_cmvn:
                    feats += " apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
                feats += f" select-voiced-frames ark:- scp,s,cs:{vad_path} ark:- |"
            elif not os.path.exists(cmvn_path) and aligner.uses_cmvn:
                feats = f"ark,s,cs:add-deltas scp:{feat_path} ark:- |"
                if aligner.uses_cmvn:
                    feats += " apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
            else:
                feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
                if lda_mat_path is not None:
                    if not os.path.exists(lda_mat_path):
                        raise Exception(f"Could not find {lda_mat_path}")
                    feats += f" splice-feats --left-context={self.feature_config.splice_left_context} --right-context={self.feature_config.splice_right_context} ark:- ark:- |"
                    feats += f" transform-feats {lda_mat_path} ark:- ark:- |"
                elif aligner.uses_splices:
                    feats += f" splice-feats --left-context={self.feature_config.splice_left_context} --right-context={self.feature_config.splice_right_context} ark:- ark:- |"
                elif self.feature_config.deltas:
                    feats += " add-deltas ark:- ark:- |"

                if fmllr_trans_path is not None and not (
                    aligner.speaker_independent or speaker_independent
                ):
                    if not os.path.exists(fmllr_trans_path):
                        raise Exception(f"Could not find {fmllr_trans_path}")
                    feats += f" transform-feats --utt2spk=ark:{utt2spk_path} ark:{fmllr_trans_path} ark:- ark:- |"
            feat_strings[dict_name] = feats
        return feat_strings

    def compile_utterance_train_graphs_arguments(
        self, validator: CorpusValidator
    ) -> CompileUtteranceTrainGraphsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compile_utterance_train_graphs_func`

        Parameters
        ----------
        validator: :class:`~montreal_forced_aligner.validator.CorpusValidator`
            Validator

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CompileUtteranceTrainGraphsArguments`
            Arguments for processing
        """
        dictionary_paths = validator.dictionary.output_paths
        disambig_paths = {
            k: os.path.join(v, "phones", "disambiguation_symbols.int")
            for k, v in dictionary_paths.items()
        }
        lexicon_fst_paths = {
            k: os.path.join(v, "L_disambig.fst") for k, v in dictionary_paths.items()
        }
        return CompileUtteranceTrainGraphsArguments(
            os.path.join(
                validator.trainer.working_log_directory, f"utterance_fst.{self.name}.log"
            ),
            self.current_dictionary_names,
            disambig_paths,
            lexicon_fst_paths,
            self.construct_path_dictionary(validator.trainer.data_directory, "utt2fst", "scp"),
            self.construct_path_dictionary(
                validator.trainer.working_directory, "utterance_graphs", "fst"
            ),
            validator.trainer.current_model_path,
            validator.trainer.tree_path,
        )

    def test_utterances_arguments(self, validator: CorpusValidator) -> TestUtterancesArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.test_utterances_func`

        Parameters
        ----------
        validator: :class:`~montreal_forced_aligner.validator.CorpusValidator`
            Validator

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.TestUtterancesArguments`
            Arguments for processing
        """
        dictionary_paths = validator.dictionary.output_paths
        words_paths = {k: os.path.join(v, "words.txt") for k, v in dictionary_paths.items()}
        return TestUtterancesArguments(
            os.path.join(validator.trainer.working_directory, f"utterance_fst.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(validator.trainer),
            words_paths,
            self.construct_path_dictionary(
                validator.trainer.working_directory, "utterance_graphs", "fst"
            ),
            self.construct_path_dictionary(validator.trainer.data_directory, "text", "int.scp"),
            self.construct_path_dictionary(validator.trainer.working_directory, "edits", "scp"),
            self.construct_path_dictionary(validator.trainer.working_directory, "aligned", "int"),
            validator.trainer.current_model_path,
        )

    def extract_ivector_arguments(
        self, ivector_extractor: IvectorExtractor
    ) -> ExtractIvectorsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.extract_ivectors_func`

        Parameters
        ----------
        ivector_extractor: :class:`~montreal_forced_aligner.abc.IvectorExtractor`
            Ivector extractor

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.ExtractIvectorsArguments`
            Arguments for processing
        """
        return ExtractIvectorsArguments(
            os.path.join(
                ivector_extractor.working_log_directory, f"extract_ivectors.{self.name}.log"
            ),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(ivector_extractor),
            ivector_extractor.ivector_options,
            self.construct_path_dictionary(ivector_extractor.working_directory, "ali", "ark"),
            ivector_extractor.ie_path,
            self.construct_path_dictionary(ivector_extractor.working_directory, "ivectors", "scp"),
            self.construct_path_dictionary(ivector_extractor.working_directory, "weights", "ark"),
            ivector_extractor.model_path,
            ivector_extractor.dubm_path,
        )

    def create_hclgs_arguments(self, transcriber: Transcriber) -> Dict[str, CreateHclgArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.create_hclg_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        Dict[str, :class:`~montreal_forced_aligner.multiprocessing.classes.CreateHclgArguments`]
            Per dictionary arguments for HCLG
        """
        args = {}

        for dictionary in self.current_dictionaries:
            dict_name = dictionary.name
            args[dict_name] = CreateHclgArguments(
                os.path.join(transcriber.model_directory, "log", f"hclg.{dict_name}.log"),
                transcriber.model_directory,
                os.path.join(transcriber.model_directory, "{file_name}" + f".{dict_name}.fst"),
                os.path.join(transcriber.model_directory, f"words.{dict_name}.txt"),
                os.path.join(transcriber.model_directory, f"G.{dict_name}.carpa"),
                transcriber.language_model.small_arpa_path,
                transcriber.language_model.medium_arpa_path,
                transcriber.language_model.carpa_path,
                transcriber.model_path,
                dictionary.disambig_path,
                os.path.join(dictionary.phones_dir, "disambiguation_symbols.int"),
                transcriber.hclg_options,
                dictionary.words_mapping,
            )
        return args

    def decode_arguments(self, transcriber: Transcriber) -> DecodeArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.decode_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.DecodeArguments`
            Arguments for processing
        """
        return DecodeArguments(
            os.path.join(transcriber.working_log_directory, f"decode.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.transcribe_config.decode_options,
            transcriber.alignment_model_path,
            self.construct_path_dictionary(transcriber.working_directory, "lat", "ark"),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "words", "txt"),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "HCLG", "fst"),
        )

    def score_arguments(self, transcriber: Transcriber) -> ScoreArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.score_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.ScoreArguments`
            Arguments for processing
        """
        return ScoreArguments(
            os.path.join(transcriber.working_log_directory, f"score.{self.name}.log"),
            self.current_dictionary_names,
            transcriber.transcribe_config.score_options,
            self.construct_path_dictionary(transcriber.working_directory, "lat", "ark"),
            self.construct_path_dictionary(transcriber.working_directory, "lat.rescored", "ark"),
            self.construct_path_dictionary(
                transcriber.working_directory, "lat.carpa.rescored", "ark"
            ),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "words", "txt"),
            self.construct_path_dictionary(transcriber.evaluation_directory, "tra", "scp"),
        )

    def lm_rescore_arguments(self, transcriber: Transcriber) -> LmRescoreArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.lm_rescore_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.LmRescoreArguments`
            Arguments for processing
        """
        return LmRescoreArguments(
            os.path.join(transcriber.working_log_directory, f"lm_rescore.{self.name}.log"),
            self.current_dictionary_names,
            transcriber.transcribe_config.lm_rescore_options,
            self.construct_path_dictionary(transcriber.working_directory, "lat", "ark"),
            self.construct_path_dictionary(transcriber.working_directory, "lat.rescored", "ark"),
            self.construct_dictionary_dependent_paths(
                transcriber.model_directory, "G.small", "fst"
            ),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "G.med", "fst"),
        )

    def carpa_lm_rescore_arguments(self, transcriber: Transcriber) -> CarpaLmRescoreArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.carpa_lm_rescore_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CarpaLmRescoreArguments`
            Arguments for processing
        """
        return CarpaLmRescoreArguments(
            os.path.join(transcriber.working_log_directory, f"carpa_lm_rescore.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_path_dictionary(transcriber.working_directory, "lat.rescored", "ark"),
            self.construct_path_dictionary(
                transcriber.working_directory, "lat.carpa.rescored", "ark"
            ),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "G.med", "fst"),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "G", "carpa"),
        )

    def initial_fmllr_arguments(self, transcriber: Transcriber) -> InitialFmllrArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.initial_fmllr_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.InitialFmllrArguments`
            Arguments for processing
        """
        return InitialFmllrArguments(
            os.path.join(transcriber.working_log_directory, f"initial_fmllr.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.fmllr_options,
            self.construct_path_dictionary(transcriber.working_directory, "trans", "ark"),
            self.construct_path_dictionary(transcriber.working_directory, "lat", "ark"),
            self.construct_path_dictionary(transcriber.data_directory, "spk2utt", "scp"),
        )

    def lat_gen_fmllr_arguments(self, transcriber: Transcriber) -> LatGenFmllrArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.lat_gen_fmllr_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.LatGenFmllrArguments`
            Arguments for processing
        """
        return LatGenFmllrArguments(
            os.path.join(transcriber.working_log_directory, f"lat_gen_fmllr.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.transcribe_config.decode_options,
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "words", "txt"),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, "HCLG", "fst"),
            self.construct_path_dictionary(transcriber.working_directory, "lat.tmp", "ark"),
        )

    def final_fmllr_arguments(self, transcriber: Transcriber) -> FinalFmllrArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.final_fmllr_est_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.FinalFmllrArguments`
            Arguments for processing
        """
        return FinalFmllrArguments(
            os.path.join(transcriber.working_log_directory, f"final_fmllr.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.fmllr_options,
            self.construct_path_dictionary(transcriber.working_directory, "trans", "ark"),
            self.construct_path_dictionary(transcriber.data_directory, "spk2utt", "scp"),
            self.construct_path_dictionary(transcriber.working_directory, "lat.tmp", "ark"),
        )

    def fmllr_rescore_arguments(self, transcriber: Transcriber) -> FmllrRescoreArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.transcription.fmllr_rescore_func`

        Parameters
        ----------
        transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
            Transcriber

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.FmllrRescoreArguments`
            Arguments for processing
        """
        return FmllrRescoreArguments(
            os.path.join(transcriber.working_log_directory, f"fmllr_rescore.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.fmllr_options,
            self.construct_path_dictionary(transcriber.working_directory, "lat.tmp", "ark"),
            self.construct_path_dictionary(transcriber.working_directory, "lat", "ark"),
        )

    def vad_arguments(self, corpus: Corpus) -> VadArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.features.compute_vad_func`

        Parameters
        ----------
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.VadArguments`
            Arguments for processing
        """
        return VadArguments(
            os.path.join(corpus.split_directory, "log", f"compute_vad.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_path_dictionary(corpus.split_directory, "feats", "scp"),
            self.construct_path_dictionary(corpus.split_directory, "vad", "scp"),
            corpus.vad_config,
        )

    def segments_vad_arguments(self, segmenter: Segmenter) -> SegmentVadArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.segment_vad_func`

        Parameters
        ----------
        segmenter: :class:`~montreal_forced_aligner.segmenter.Segmenter`
            Segmenter

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.SegmentVadArguments`
            Arguments for processing
        """
        return SegmentVadArguments(
            self.current_dictionary_names,
            self.construct_path_dictionary(segmenter.corpus.split_directory, "vad", "scp"),
            segmenter.segmentation_config.segmentation_options,
        )

    def mfcc_arguments(self, corpus: Corpus) -> MfccArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.features.mfcc_func`

        Parameters
        ----------
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.MfccArguments`
            Arguments for processing
        """
        return MfccArguments(
            os.path.join(corpus.split_directory, "log", f"make_mfcc.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_path_dictionary(corpus.split_directory, "feats", "scp"),
            self.construct_path_dictionary(corpus.split_directory, "utterance_lengths", "scp"),
            self.construct_path_dictionary(corpus.split_directory, "segments", "scp"),
            self.construct_path_dictionary(corpus.split_directory, "wav", "scp"),
            self.feature_config.mfcc_options,
        )

    def acc_stats_arguments(self, aligner: BaseTrainer) -> AccStatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.acc_stats_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.BaseTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AccStatsArguments`
            Arguments for processing
        """
        return AccStatsArguments(
            os.path.join(
                aligner.working_directory, "log", f"acc.{aligner.iteration}.{self.name}.log"
            ),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.construct_path_dictionary(
                aligner.working_directory, str(aligner.iteration), "acc"
            ),
            aligner.current_model_path,
        )

    def mono_align_equal_arguments(self, aligner: MonophoneTrainer) -> MonoAlignEqualArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.mono_align_equal_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.MonophoneTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.MonoAlignEqualArguments`
            Arguments for processing
        """
        return MonoAlignEqualArguments(
            os.path.join(aligner.working_log_directory, f"mono_align_equal.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, "fsts", "scp"),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.working_directory, "0", "acc"),
            aligner.current_model_path,
        )

    def align_arguments(self, aligner: Aligner) -> AlignArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.align_func`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AlignArguments`
            Arguments for processing
        """
        if aligner.iteration is not None:
            log_path = os.path.join(
                aligner.working_log_directory, f"align.{aligner.iteration}.{self.name}.log"
            )
        else:
            log_path = os.path.join(aligner.working_log_directory, f"align.{self.name}.log")
        return AlignArguments(
            log_path,
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, "fsts", "scp"),
            self.construct_feature_proc_strings(aligner),
            aligner.alignment_model_path,
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.working_directory, "ali", "scores"),
            self.construct_path_dictionary(aligner.working_directory, "ali", "loglikes"),
            aligner.align_options,
        )

    def compile_information_arguments(self, aligner: BaseTrainer) -> CompileInformationArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compile_information_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.BaseTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CompileInformationArguments`
            Arguments for processing
        """
        if aligner.iteration is not None:
            log_path = os.path.join(
                aligner.working_log_directory, f"align.{aligner.iteration}.{self.name}.log"
            )
        else:
            log_path = os.path.join(aligner.working_log_directory, f"align.{self.name}.log")
        return CompileInformationArguments(log_path)

    def word_boundary_int_files(self) -> Dict[str, str]:
        """
        Generate mapping for dictionaries to word boundary int files

        Returns
        -------
        Dict[str, ReversedMappingType]
            Per dictionary word boundary int files
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = os.path.join(dictionary.phones_dir, "word_boundary.int")
        return data

    def reversed_phone_mappings(self) -> Dict[str, ReversedMappingType]:
        """
        Generate mapping for dictionaries to reversed phone mapping

        Returns
        -------
        Dict[str, ReversedMappingType]
            Per dictionary reversed phone mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_phone_mapping
        return data

    def reversed_word_mappings(self) -> Dict[str, ReversedMappingType]:
        """
        Generate mapping for dictionaries to reversed word mapping

        Returns
        -------
        Dict[str, ReversedMappingType]
            Per dictionary reversed word mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_word_mapping
        return data

    def words_mappings(self) -> Dict[str, MappingType]:
        """
        Generate mapping for dictionaries to word mapping

        Returns
        -------
        Dict[str, MappingType]
            Per dictionary word mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words_mapping
        return data

    def words(self) -> Dict[str, WordsType]:
        """
        Generate mapping for dictionaries to words

        Returns
        -------
        Dict[str, WordsType]
            Per dictionary words
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words
        return data

    def punctuation(self):
        """
        Generate mapping for dictionaries to punctuation

        Returns
        -------
        Dict[str, str]
            Per dictionary punctuation
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.punctuation
        return data

    def clitic_set(self) -> Dict[str, Set[str]]:
        """
        Generate mapping for dictionaries to clitic sets

        Returns
        -------
        Dict[str, str]
            Per dictionary clitic sets
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_set
        return data

    def clitic_markers(self) -> Dict[str, str]:
        """
        Generate mapping for dictionaries to clitic markers

        Returns
        -------
        Dict[str, str]
            Per dictionary clitic markers
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_markers
        return data

    def compound_markers(self) -> Dict[str, str]:
        """
        Generate mapping for dictionaries to compound markers

        Returns
        -------
        Dict[str, str]
            Per dictionary compound markers
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.compound_markers
        return data

    def strip_diacritics(self) -> Dict[str, List[str]]:
        """
        Generate mapping for dictionaries to diacritics to strip

        Returns
        -------
        Dict[str, List[str]]
            Per dictionary strip diacritics
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.strip_diacritics
        return data

    def oov_codes(self) -> Dict[str, str]:
        """
        Generate mapping for dictionaries to oov symbols

        Returns
        -------
        Dict[str, str]
            Per dictionary oov symbols
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_code
        return data

    def oov_ints(self) -> Dict[str, int]:
        """
        Generate mapping for dictionaries to oov ints

        Returns
        -------
        Dict[str, int]
            Per dictionary oov ints
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_int
        return data

    def positions(self) -> Dict[str, List[str]]:
        """
        Generate mapping for dictionaries to positions

        Returns
        -------
        Dict[str, List[str]]
            Per dictionary positions
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.positions
        return data

    def silences(self) -> Dict[str, Set[str]]:
        """
        Generate mapping for dictionaries to silence symbols

        Returns
        -------
        Dict[str, Set[str]]
            Per dictionary silence symbols
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.silences
        return data

    def multilingual_ipa(self) -> Dict[str, bool]:
        """
        Generate mapping for dictionaries to multilingual IPA flags

        Returns
        -------
        Dict[str, bool]
            Per dictionary multilingual IPA flags
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.multilingual_ipa
        return data

    def generate_pronunciations_arguments(
        self, aligner: Aligner
    ) -> GeneratePronunciationsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.pronunciations.generate_pronunciations_func`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.GeneratePronunciationsArguments`
            Arguments for processing
        """
        return GeneratePronunciationsArguments(
            os.path.join(
                aligner.working_log_directory, f"generate_pronunciations.{self.name}.log"
            ),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.data_directory, "text", "int.scp"),
            self.word_boundary_int_files(),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            aligner.model_path,
            self.construct_path_dictionary(aligner.working_directory, "prons", "scp"),
        )

    def alignment_improvement_arguments(
        self, aligner: BaseTrainer
    ) -> AlignmentImprovementArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compute_alignment_improvement_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.BaseTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AlignmentImprovementArguments`
            Arguments for processing
        """
        return AlignmentImprovementArguments(
            os.path.join(aligner.working_log_directory, f"alignment_analysis.{self.name}.log"),
            self.current_dictionary_names,
            aligner.current_model_path,
            self.construct_path_dictionary(aligner.data_directory, "text", "int.scp"),
            self.word_boundary_int_files(),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.feature_config.frame_shift,
            self.reversed_phone_mappings(),
            self.positions(),
            self.construct_path_dictionary(
                aligner.working_directory, f"phone.{aligner.iteration}", "ctm"
            ),
        )

    def ali_to_word_ctm_arguments(self, aligner: BaseAligner) -> AliToCtmArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.ali_to_ctm_func`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AliToCtmArguments`
            Arguments for processing
        """
        return AliToCtmArguments(
            os.path.join(aligner.working_log_directory, f"get_word_ctm.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.data_directory, "text", "int.scp"),
            self.word_boundary_int_files(),
            round(self.feature_config.frame_shift / 1000, 4),
            aligner.alignment_model_path,
            self.construct_path_dictionary(aligner.working_directory, "word", "ctm"),
            True,
        )

    def ali_to_phone_ctm_arguments(self, aligner: Aligner) -> AliToCtmArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.ali_to_ctm_func`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AliToCtmArguments`
            Arguments for processing
        """
        return AliToCtmArguments(
            os.path.join(aligner.working_log_directory, f"get_phone_ctm.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.data_directory, "text", "int.scp"),
            self.word_boundary_int_files(),
            round(self.feature_config.frame_shift / 1000, 4),
            aligner.alignment_model_path,
            self.construct_path_dictionary(aligner.working_directory, "phone", "ctm"),
            False,
        )

    def job_utts(self) -> Dict[str, Dict[str, Utterance]]:
        """
        Generate utterances by dictionary name for the Job

        Returns
        -------
        Dict[str, Dict[str, :class:`~montreal_forced_aligner.corpus.Utterance`]]
            Mapping of dictionary name to Utterance mappings
        """
        data = {}
        speakers = self.subset_speakers
        if not speakers:
            speakers = self.speakers
        for s in speakers:
            if s.dictionary.name not in data:
                data[s.dictionary.name] = {}
            data[s.dictionary.name].update(s.utterances)
        return data

    def job_files(self) -> Dict[str, File]:
        """
        Generate files for the Job

        Returns
        -------
        Dict[str, :class:`~montreal_forced_aligner.corpus.File`]
            Mapping of file name to File objects
        """
        data = {}
        speakers = self.subset_speakers
        if not speakers:
            speakers = self.speakers
        for s in speakers:
            for f in s.files:
                for sf in f.speaker_ordering:
                    if sf.name == s.name:
                        sf.dictionary_data = s.dictionary_data
                data[f.name] = f
        return data

    def cleanup_word_ctm_arguments(self, aligner: Aligner) -> CleanupWordCtmArguments:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.multiprocessing.CleanupWordCtmProcessWorker`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CleanupWordCtmArguments`
            Arguments for processing
        """
        return CleanupWordCtmArguments(
            self.construct_path_dictionary(aligner.align_directory, "word", "ctm"),
            self.current_dictionary_names,
            self.job_utts(),
            self.dictionary_data(),
        )

    def no_cleanup_word_ctm_arguments(self, aligner: Aligner) -> NoCleanupWordCtmArguments:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.multiprocessing.NoCleanupWordCtmProcessWorker`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.NoCleanupWordCtmArguments`
            Arguments for processing
        """
        return NoCleanupWordCtmArguments(
            self.construct_path_dictionary(aligner.align_directory, "word", "ctm"),
            self.current_dictionary_names,
            self.job_utts(),
            self.dictionary_data(),
        )

    def phone_ctm_arguments(self, aligner: Aligner) -> PhoneCtmArguments:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.multiprocessing.PhoneCtmProcessWorker`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.PhoneCtmArguments`
            Arguments for processing
        """
        return PhoneCtmArguments(
            self.construct_path_dictionary(aligner.align_directory, "phone", "ctm"),
            self.current_dictionary_names,
            self.job_utts(),
            self.reversed_phone_mappings(),
            self.positions(),
        )

    def dictionary_data(self) -> Dict[str, DictionaryData]:
        """
        Generate dictionary data for the job

        Returns
        -------
        Dict[str, DictionaryData]
            Mapping of dictionary name to dictionary data
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.data()
        return data

    def combine_ctm_arguments(self, aligner: Aligner) -> CombineCtmArguments:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.multiprocessing.CombineProcessWorker`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CombineCtmArguments`
            Arguments for processing
        """
        return CombineCtmArguments(
            self.current_dictionary_names,
            self.job_files(),
            self.dictionary_data(),
            aligner.align_config.cleanup_textgrids,
        )

    def export_textgrid_arguments(self, aligner: Aligner) -> ExportTextGridArguments:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.multiprocessing.ExportTextGridProcessWorker`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.ExportTextGridArguments`
            Arguments for processing
        """
        return ExportTextGridArguments(
            aligner.corpus.files,
            aligner.feature_config.frame_shift,
            aligner.textgrid_output,
            aligner.backup_output_directory,
        )

    def tree_stats_arguments(self, aligner: BaseTrainer) -> TreeStatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.tree_stats_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.BaseTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.TreeStatsArguments`
            Arguments for processing
        """
        return TreeStatsArguments(
            os.path.join(aligner.working_log_directory, f"acc_tree.{self.name}.log"),
            self.current_dictionary_names,
            aligner.dictionary.config.silence_csl,
            aligner.previous_trainer.alignment_model_path,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.previous_trainer.align_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.working_directory, "tree", "acc"),
        )

    def convert_alignment_arguments(self, aligner: BaseTrainer) -> ConvertAlignmentsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.convert_alignments_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.BaseTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.ConvertAlignmentsArguments`
            Arguments for processing
        """
        return ConvertAlignmentsArguments(
            os.path.join(aligner.working_log_directory, f"convert_alignments.{self.name}.log"),
            self.current_dictionary_names,
            aligner.current_model_path,
            aligner.tree_path,
            aligner.previous_trainer.alignment_model_path,
            self.construct_path_dictionary(
                aligner.previous_trainer.working_directory, "ali", "ark"
            ),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
        )

    def calc_fmllr_arguments(self, aligner: Aligner) -> CalcFmllrArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.calc_fmllr_func`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CalcFmllrArguments`
            Arguments for processing
        """
        return CalcFmllrArguments(
            os.path.join(aligner.working_log_directory, f"calc_fmllr.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            aligner.alignment_model_path,
            aligner.model_path,
            self.construct_path_dictionary(aligner.data_directory, "spk2utt", "scp"),
            self.construct_path_dictionary(aligner.working_directory, "trans", "ark"),
            aligner.fmllr_options,
        )

    def acc_stats_two_feats_arguments(self, aligner: SatTrainer) -> AccStatsTwoFeatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.acc_stats_two_feats_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.SatTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AccStatsTwoFeatsArguments`
            Arguments for processing
        """
        return AccStatsTwoFeatsArguments(
            os.path.join(aligner.working_log_directory, f"acc_stats_two_feats.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.working_directory, "two_feat_acc", "ark"),
            aligner.current_model_path,
            self.construct_feature_proc_strings(aligner),
            self.construct_feature_proc_strings(aligner, speaker_independent=True),
        )

    def lda_acc_stats_arguments(self, aligner: LdaTrainer) -> LdaAccStatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.lda_acc_stats_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.LdaTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.LdaAccStatsArguments`
            Arguments for processing
        """
        return LdaAccStatsArguments(
            os.path.join(aligner.working_log_directory, f"lda_acc_stats.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(
                aligner.previous_trainer.working_directory, "ali", "ark"
            ),
            aligner.previous_trainer.alignment_model_path,
            aligner.lda_options,
            self.construct_path_dictionary(aligner.working_directory, "lda", "acc"),
        )

    def calc_lda_mllt_arguments(self, aligner: LdaTrainer) -> CalcLdaMlltArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.calc_lda_mllt_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.LdaTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CalcLdaMlltArguments`
            Arguments for processing
        """
        return CalcLdaMlltArguments(
            os.path.join(aligner.working_log_directory, f"lda_mllt.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, "ali", "ark"),
            aligner.current_model_path,
            aligner.lda_options,
            self.construct_path_dictionary(aligner.working_directory, "lda", "macc"),
        )

    def ivector_acc_stats_arguments(
        self, trainer: IvectorExtractorTrainer
    ) -> AccIvectorStatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.acc_ivector_stats_func`

        Parameters
        ----------
        trainer: :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AccIvectorStatsArguments`
            Arguments for processing
        """
        return AccIvectorStatsArguments(
            os.path.join(trainer.working_log_directory, f"ivector_acc.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(trainer),
            trainer.ivector_options,
            trainer.current_ie_path,
            self.construct_path_dictionary(trainer.working_directory, "post", "ark"),
            self.construct_path_dictionary(trainer.working_directory, "ivector", "acc"),
        )

    def map_acc_stats_arguments(self, aligner: AdaptingAligner) -> MapAccStatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.map_acc_stats_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.aligner.AdaptingAligner`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.MapAccStatsArguments`
            Arguments for processing
        """
        return MapAccStatsArguments(
            os.path.join(aligner.working_log_directory, f"map_acc_stats.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.current_model_path,
            self.construct_path_dictionary(aligner.previous_aligner.align_directory, "ali", "ark"),
            self.construct_path_dictionary(aligner.working_directory, "map", "acc"),
        )

    def gmm_gselect_arguments(self, aligner: IvectorExtractorTrainer) -> GmmGselectArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.gmm_gselect_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.GmmGselectArguments`
            Arguments for processing
        """
        return GmmGselectArguments(
            os.path.join(aligner.working_log_directory, f"gmm_gselect.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.ivector_options,
            aligner.current_dubm_path,
            self.construct_path_dictionary(aligner.working_directory, "gselect", "ark"),
        )

    def acc_global_stats_arguments(
        self, aligner: IvectorExtractorTrainer
    ) -> AccGlobalStatsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.acc_global_stats_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligners.trainers.IvectorExtractorTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.AccGlobalStatsArguments`
            Arguments for processing
        """
        return AccGlobalStatsArguments(
            os.path.join(
                aligner.working_log_directory,
                f"acc_global_stats.{aligner.iteration}.{self.name}.log",
            ),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.ivector_options,
            self.construct_path_dictionary(aligner.working_directory, "gselect", "ark"),
            self.construct_path_dictionary(
                aligner.working_directory, f"global.{aligner.iteration}", "acc"
            ),
            aligner.current_dubm_path,
        )

    def gauss_to_post_arguments(self, aligner: IvectorExtractorTrainer) -> GaussToPostArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.gauss_to_post_func`

        Parameters
        ----------
        aligner: :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.GaussToPostArguments`
            Arguments for processing
        """
        return GaussToPostArguments(
            os.path.join(aligner.working_log_directory, f"gauss_to_post.{self.name}.log"),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.ivector_options,
            self.construct_path_dictionary(aligner.working_directory, "post", "ark"),
            aligner.current_dubm_path,
        )

    def compile_train_graph_arguments(self, aligner: Aligner) -> CompileTrainGraphsArguments:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.alignment.compile_train_graphs_func`

        Parameters
        ----------
        aligner: Aligner
            Aligner

        Returns
        -------
        :class:`~montreal_forced_aligner.multiprocessing.classes.CompileTrainGraphsArguments`
            Arguments for processing
        """
        dictionary_paths = aligner.dictionary.output_paths
        disambig_paths = {
            k: os.path.join(v, "phones", "disambiguation_symbols.int")
            for k, v in dictionary_paths.items()
        }
        lexicon_fst_paths = {k: os.path.join(v, "L.fst") for k, v in dictionary_paths.items()}
        model_path = aligner.model_path
        if not os.path.exists(model_path):
            model_path = aligner.alignment_model_path
        return CompileTrainGraphsArguments(
            os.path.join(aligner.working_log_directory, f"compile_train_graphs.{self.name}.log"),
            self.current_dictionary_names,
            os.path.join(aligner.working_directory, "tree"),
            model_path,
            self.construct_path_dictionary(aligner.data_directory, "text", "int.scp"),
            disambig_paths,
            lexicon_fst_paths,
            self.construct_path_dictionary(aligner.working_directory, "fsts", "scp"),
        )

    def utt2fst_scp_data(
        self, corpus: Corpus, num_frequent_words: int = 10
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate Kaldi style utt2fst scp data

        Parameters
        ----------
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus to generate data for
        num_frequent_words: int
            Number of frequent words to include in the unigram language model

        Returns
        -------
        Dict[str, List[Tuple[str, str]]]
            Utterance FSTs per dictionary name
        """
        data = {}
        most_frequent = {}
        for dict_name, utterances in self.job_utts().items():
            data[dict_name] = []
            for u_name, utterance in utterances.items():
                new_text = []
                dictionary = utterance.speaker.dictionary
                if dictionary.name not in most_frequent:
                    word_frequencies = corpus.get_word_frequency()
                    most_frequent[dictionary.name] = sorted(
                        word_frequencies.items(), key=lambda x: -x[1]
                    )[:num_frequent_words]

                for t in utterance.text:
                    lookup = utterance.speaker.dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                    new_text.extend(x for x in lookup if x != "")
                data[dict_name].append(
                    (
                        u_name,
                        dictionary.create_utterance_fst(new_text, most_frequent[dictionary.name]),
                    )
                )
        return data

    def output_utt_fsts(self, corpus: Corpus, num_frequent_words: int = 10) -> None:
        """
        Write utterance FSTs

        Parameters
        ----------
        corpus: :class:`~montreal_forced_aligner.corpus.Corpus`
            Corpus to generate FSTs for
        num_frequent_words: int
            Number of frequent words
        """
        utt2fst = self.utt2fst_scp_data(corpus, num_frequent_words)
        for dict_name, scp in utt2fst.items():
            utt2fst_scp_path = os.path.join(
                corpus.split_directory, f"utt2fst.{dict_name}.{self.name}.scp"
            )
            save_scp(scp, utt2fst_scp_path, multiline=True)

    def output_to_directory(self, split_directory: str) -> None:
        """
        Output job information to a directory

        Parameters
        ----------
        split_directory: str
            Directory to output to
        """
        wav = self.wav_scp_data()
        for dict_name, scp in wav.items():
            wav_scp_path = os.path.join(split_directory, f"wav.{dict_name}.{self.name}.scp")
            output_mapping(scp, wav_scp_path, skip_safe=True)

        spk2utt = self.spk2utt_scp_data()
        for dict_name, scp in spk2utt.items():
            spk2utt_scp_path = os.path.join(
                split_directory, f"spk2utt.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, spk2utt_scp_path)

        feats = self.feat_scp_data()
        for dict_name, scp in feats.items():
            feats_scp_path = os.path.join(split_directory, f"feats.{dict_name}.{self.name}.scp")
            output_mapping(scp, feats_scp_path)

        cmvn = self.cmvn_scp_data()
        for dict_name, scp in cmvn.items():
            cmvn_scp_path = os.path.join(split_directory, f"cmvn.{dict_name}.{self.name}.scp")
            output_mapping(scp, cmvn_scp_path)

        utt2spk = self.utt2spk_scp_data()
        for dict_name, scp in utt2spk.items():
            utt2spk_scp_path = os.path.join(
                split_directory, f"utt2spk.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, utt2spk_scp_path)

        segments = self.segments_scp_data()
        for dict_name, scp in segments.items():
            segments_scp_path = os.path.join(
                split_directory, f"segments.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, segments_scp_path)

        text_scp = self.text_scp_data()
        for dict_name, scp in text_scp.items():
            if not scp:
                continue
            text_scp_path = os.path.join(split_directory, f"text.{dict_name}.{self.name}.scp")
            output_mapping(scp, text_scp_path)

        text_int = self.text_int_scp_data()
        for dict_name, scp in text_int.items():
            if dict_name is None:
                continue
            if not scp:
                continue
            text_int_scp_path = os.path.join(
                split_directory, f"text.{dict_name}.{self.name}.int.scp"
            )
            output_mapping(scp, text_int_scp_path, skip_safe=True)
