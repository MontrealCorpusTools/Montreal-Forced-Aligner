from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Union, Callable, NamedTuple, Set

if TYPE_CHECKING:
    from ..dictionary import Dictionary, DictionaryData
    from .base import BaseCorpus
    from ..validator import CorpusValidator
    from ..trainers import BaseTrainer, MonophoneTrainer, LdaTrainer, SatTrainer, IvectorExtractorTrainer
    from ..speaker_classifier import SpeakerClassifier
    from ..segmenter import Segmenter
    from ..aligner.base import BaseAligner
    from ..config import FeatureConfig
    from ..aligner.adapting import AdaptingAligner
    from ..corpus import CorpusType, SegmentsType, OneToOneMappingType, OneToManyMappingType
    from ..dictionary import ReversedMappingType, MultiSpeakerMappingType, MappingType, WordsType, DictionaryType, \
        PunctuationType, IpaType
    from ..config.align_config import AlignConfig, ConfigDict
    from ..textgrid import CtmType
    from ..transcriber import Transcriber
    from ..config import SegmentationConfig

    ConfigType = Union[BaseTrainer, AlignConfig]
    FmllrConfigType = Union[SatTrainer, AlignConfig]
    LdaConfigType = Union[LdaTrainer, AlignConfig]

    IterationType = Union[str, int]

    AlignerType = Union[BaseTrainer, BaseAligner]

import os
import traceback
import sys

from praatio import textgrid
from praatio.utilities.constants import Interval
from collections import namedtuple

from .helper import get_wav_info, parse_transcription, load_text
from ..helper import output_mapping, save_scp
from ..exceptions import CorpusError, TextParseError, TextGridParseError


def parse_file(utt_name: str, wav_path: str, text_path: str,
               relative_path: str, speaker_characters: Union[int, str], sample_rate: int = 16000,
               punctuation: Optional[str] = None,
               clitic_markers: Optional[str] = None,
               stop_check: Optional[Callable] = None) -> File:
    file = File(wav_path, text_path, relative_path=relative_path)
    if file.has_sound_file:
        root = os.path.dirname(wav_path)
        file.wav_info = get_wav_info(wav_path, sample_rate=sample_rate)
    else:
        root = os.path.dirname(text_path)
    if not speaker_characters:
        speaker_name = os.path.basename(root)
    elif isinstance(speaker_characters, int):
        speaker_name = utt_name[:speaker_characters]
    elif speaker_characters == 'prosodylab':
        speaker_name = utt_name.split('_')[1]
    else:
        speaker_name = utt_name
    root_speaker = None
    if speaker_characters or file.text_type != 'textgrid':
        root_speaker = Speaker(speaker_name)
    file.load_text(root_speaker=root_speaker, punctuation=punctuation, clitic_markers=clitic_markers,
                   stop_check=stop_check)
    return file


class Speaker(object):
    def __init__(self, name):
        self.name = name
        self.utterances: Dict[str, Utterance] = {}
        self.cmvn = None
        self.dictionary: Optional[Dictionary] = None
        self.dictionary_data: Optional[DictionaryData] = None

    def __getstate__(self):
        data = {'name': self.name,
                'cmvn': self.cmvn
                }
        if self.dictionary_data is not None:
            data['dictionary_data'] = self.dictionary_data
        return data

    def __setstate__(self, state):
        self.name = state['name']
        self.cmvn = state['cmvn']
        if 'dictionary_data' in state:
            self.dictionary_data = state['dictionary_data']

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Speaker):
            return other.name == self.name
        elif isinstance(other, str):
            return self.name == other
        raise NotImplementedError

    def __lt__(self, other):
        if isinstance(other, Speaker):
            return other.name < self.name
        elif isinstance(other, str):
            return self.name < other
        raise NotImplementedError

    def __lte__(self, other):
        if isinstance(other, Speaker):
            return other.name <= self.name
        elif isinstance(other, str):
            return self.name <= other
        raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, Speaker):
            return other.name > self.name
        elif isinstance(other, str):
            return self.name > other
        raise NotImplementedError

    def __gte__(self, other):
        if isinstance(other, Speaker):
            return other.name >= self.name
        elif isinstance(other, str):
            return self.name >= other
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def add_utterance(self, utterance: Utterance):
        utterance.speaker = self
        self.utterances[utterance] = utterance

    def delete_utterance(self, utterance: Utterance):
        utterance.speaker = None
        del self.utterances[utterance]

    def merge(self, speaker: Speaker):
        for u in speaker.utterances.values():
            self.add_utterance(u)
        speaker.utterances = []

    def word_set(self):
        words = set()
        for u in self.utterances.values():
            if u.text:
                words.update(u.text.split())
        return words

    def set_dictionary(self, dictionary: Dictionary):
        self.dictionary = dictionary
        self.dictionary_data = dictionary.data(self.word_set())

    @property
    def files(self):
        files = set()
        for u in self.utterances.values():
            files.add(u.file)
        return files

    @property
    def meta(self):
        data = {
            'name': self.name,
            'cmvn': self.cmvn,
        }
        if self.dictionary is not None:
            data['dictionary'] = self.dictionary.name
        return data


class File(object):
    def __init__(self, wav_path=None, text_path=None, relative_path=None):
        self.wav_path = wav_path
        self.text_path = text_path
        if self.wav_path is not None:
            self.name = os.path.splitext(os.path.basename(self.wav_path))[0]
        elif self.text_path is not None:
            self.name = os.path.splitext(os.path.basename(self.text_path))[0]
        else:
            raise CorpusError('File objects must have either a wav_path or text_path')
        self.relative_path = relative_path
        self.wav_info = None
        self.speaker_ordering: List[Speaker] = []
        self.utterances: Dict[str, Utterance] = {}
        self.aligned = False

    def __repr__(self):
        return f'<File {self.name} Sound path="{self.wav_path}" Text path="{self.text_path}">'

    def __getstate__(self):
        return {
            'name': self.name,
            'wav_path': self.wav_path,
            'text_path': self.text_path,
            'relative_path': self.relative_path,
            'aligned': self.aligned,
            'wav_info': self.wav_info,
            'speaker_ordering': [x.__getstate__() for x in self.speaker_ordering],
            'utterances': self.utterances.values()
        }

    def __setstate__(self, state):
        self.name = state['name']
        self.wav_path = state['wav_path']
        self.text_path = state['text_path']
        self.relative_path = state['relative_path']
        self.wav_info = state['wav_info']
        self.aligned = state['aligned']
        self.speaker_ordering = state['speaker_ordering']
        self.utterances = {}
        for i, s in enumerate(self.speaker_ordering):
            self.speaker_ordering[i] = Speaker('')
            self.speaker_ordering[i].__setstate__(s)
        for u in state['utterances']:
            u.file = self
            for s in self.speaker_ordering:
                if s.name == u.speaker_name:
                    u.speaker = s
                    s.add_utterance(u)
            self.add_utterance(u)

    def save(self, output_directory: Optional[str]=None, backup_output_directory: Optional[str] = None):
        utterance_count = len(self.utterances)
        if utterance_count == 1:
            utterance = next(iter(self.utterances.values()))
            if utterance.begin is None and not utterance.phone_labels:
                output_path = self.construct_output_path(output_directory, backup_output_directory, enforce_lab=True)
                with open(output_path, 'w', encoding='utf8') as f:
                    if utterance.transcription_text is not None:
                        f.write(utterance.transcription_text)
                    else:
                        f.write(utterance.text)
                return
        output_path = self.construct_output_path(output_directory, backup_output_directory)
        max_time = self.duration
        tiers = {}
        for speaker in self.speaker_ordering:
            if speaker is None:
                tiers['speech'] = textgrid.IntervalTier('speech', [], minT=0, maxT=max_time)
            else:
                tiers[speaker] = textgrid.IntervalTier(speaker.name, [], minT=0, maxT=max_time)

        tg = textgrid.Textgrid()
        tg.maxTimestamp = max_time
        for utt_name, utterance in self.utterances.items():

            if utterance.speaker is None:
                speaker = 'speech'
            else:
                speaker = utterance.speaker
            if not self.aligned:

                if utterance.transcription_text is not None:
                    tiers[speaker].entryList.append(
                        Interval(start=utterance.begin, end=utterance.end, label=utterance.transcription_text))
                else:
                    tiers[speaker].entryList.append(
                        Interval(start=utterance.begin, end=utterance.end, label=utterance.text))
        for t in tiers.values():
            tg.addTier(t)
        tg.save(output_path,
                includeBlankSpaces=True, format='long_textgrid')

    @property
    def meta(self):
        return {
            'wav_path': self.wav_path,
            'text_path': self.text_path,
            'name': self.name,
            'relative_path': self.relative_path,
            'wav_info': self.wav_info,
            'speaker_ordering': [x.name for x in self.speaker_ordering],
        }

    @property
    def has_sound_file(self):
        if self.wav_path is not None and os.path.exists(self.wav_path):
            return True
        return False

    @property
    def has_text_file(self):
        if self.text_path is not None and os.path.exists(self.text_path):
            return True
        return False

    @property
    def text_type(self):
        if self.has_text_file:
            if os.path.splitext(self.text_path)[1].lower() == '.textgrid':
                return 'textgrid'
            else:
                return 'lab'
        return None

    def construct_output_path(self, output_directory: Optional[str]=None,
                              backup_output_directory: Optional[str]=None,
                              enforce_lab:bool = False):
        if enforce_lab:
            extension = '.lab'
        else:
            extension = '.TextGrid'
        if output_directory is None:
            if self.text_path is None:
                return os.path.splitext(self.wav_path)[0] + extension
            return self.text_path
        if self.relative_path:
            relative = os.path.join(output_directory, self.relative_path)
        else:
            relative = output_directory
        tg_path = os.path.join(relative, self.name + extension)
        if backup_output_directory is not None and os.path.exists(tg_path):
            tg_path = tg_path.replace(output_directory, backup_output_directory)
        os.makedirs(os.path.dirname(tg_path), exist_ok=True)
        return tg_path

    def load_text(self, root_speaker: Optional[Speaker] = None,
                  punctuation: Optional[str] = None,
                  clitic_markers: Optional[str] = None,
                  stop_check: Optional[Callable] = None):
        if self.text_type == 'lab':
            try:
                text = load_text(self.text_path)
            except UnicodeDecodeError:
                raise TextParseError(self.text_path)
            words = parse_transcription(text, punctuation, clitic_markers)
            utterance = Utterance(speaker=root_speaker, file=self, text=' '.join(words))
            self.add_utterance(utterance)
        elif self.text_type == 'textgrid':
            try:
                tg = textgrid.openTextgrid(self.text_path, includeEmptyIntervals=False)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise TextGridParseError(self.text_path,
                                         '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

            num_tiers = len(tg.tierNameList)
            if num_tiers == 0:
                raise TextGridParseError(self.text_path, 'Number of tiers parsed was zero')
            if self.num_channels > 2:
                raise (Exception('More than two channels'))
            for i, tier_name in enumerate(tg.tierNameList):
                ti = tg.tierDict[tier_name]
                if tier_name.lower() == 'notes':
                    continue
                if not isinstance(ti, textgrid.IntervalTier):
                    continue
                if not root_speaker:
                    speaker_name = tier_name.strip()
                    speaker = Speaker(speaker_name)
                    self.add_speaker(speaker)
                else:
                    speaker = root_speaker
                for begin, end, text in ti.entryList:
                    if stop_check is not None and stop_check():
                        return
                    text = text.lower().strip()
                    words = parse_transcription(text, punctuation, clitic_markers)
                    if not words:
                        continue
                    begin, end = round(begin, 4), round(end, 4)
                    end = min(end, self.duration)
                    utt = Utterance(speaker=speaker, file=self, begin=begin, end=end, text=' '.join(words))
                    self.add_utterance(utt)
        else:
            utterance = Utterance(speaker=root_speaker, file=self)
            self.add_utterance(utterance)

    def add_speaker(self, speaker: Speaker):
        if speaker not in self.speaker_ordering:
            self.speaker_ordering.append(speaker)

    def add_utterance(self, utterance: Utterance):
        utterance.file = self
        self.utterances[utterance.name] = utterance
        self.add_speaker(utterance.speaker)

    def delete_utterance(self, utterance: Utterance):
        utterance.file = None
        del self.utterances[utterance]

    def load_info(self):
        if self.wav_path is not None:
            self.wav_info = get_wav_info(self.wav_path)

    @property
    def duration(self):
        if self.wav_path is None:
            return 0
        if not self.wav_info:
            self.load_info()
        return self.wav_info['duration']

    @property
    def num_channels(self):
        if self.wav_path is None:
            return 0
        if not self.wav_info:
            self.load_info()
        return self.wav_info['num_channels']

    @property
    def format(self):
        if not self.wav_info:
            self.load_info()
        return self.wav_info['format']

    @property
    def sox_string(self):
        if not self.wav_info:
            self.load_info()
        return self.wav_info['sox_string']

    def for_wav_scp(self):
        if self.sox_string:
            return self.sox_string
        return self.wav_path


class Utterance(object):
    def __init__(self, speaker: Speaker, file: File, begin: Optional[float] = None,
                 end: Optional[float] = None, channel: Optional[int] = 0, text: Optional[str] = None):
        self.speaker = speaker
        self.file = file
        self.file_name = file.name
        self.speaker_name = speaker.name
        self.begin = begin
        self.end = end
        self.channel = channel
        self.text = text
        self.transcription_text = None
        self.ignored = False
        self.features = None
        self.feature_length = None
        self.phone_labels: Optional[CtmType] = None
        self.word_labels: Optional[CtmType] = None
        self.oovs = []
        self.speaker.add_utterance(self)
        self.file.add_utterance(self)

    def __getstate__(self):
        return {'file_name': self.file_name,
                'speaker_name': self.speaker_name,
                'begin': self.begin,
                'end': self.end,
                'channel': self.channel,
                'text': self.text,
                'transcription_text': self.transcription_text,
                'oovs': self.oovs,
                'ignored': self.ignored,
                'features': self.features,
                'feature_length': self.feature_length,
                'phone_labels': self.phone_labels,
                'word_labels': self.word_labels,
                }

    def __setstate__(self, state):
        self.file_name = state['file_name']
        self.speaker_name = state['speaker_name']
        self.begin = state['begin']
        self.end = state['end']
        self.channel = state['channel']
        self.text = state['text']
        self.transcription_text = state['transcription_text']
        self.oovs = state['oovs']
        self.ignored = state['ignored']
        self.features = state['features']
        self.feature_length = state['feature_length']
        self.phone_labels = state['phone_labels']
        self.word_labels = state['word_labels']

    def delete(self):
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<Utterance "{self.name}">'

    def __eq__(self, other):
        if isinstance(other, Utterance):
            return other.name == self.name
        elif isinstance(other, str):
            return self.name == other
        raise NotImplementedError

    def __lt__(self, other):
        if isinstance(other, Utterance):
            return other.name < self.name
        elif isinstance(other, str):
            return self.name < other
        raise NotImplementedError

    def __lte__(self, other):
        if isinstance(other, Utterance):
            return other.name <= self.name
        elif isinstance(other, str):
            return self.name <= other
        raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, Utterance):
            return other.name > self.name
        elif isinstance(other, str):
            return self.name > other
        raise NotImplementedError

    def __gte__(self, other):
        if isinstance(other, Utterance):
            return other.name >= self.name
        elif isinstance(other, str):
            return self.name >= other
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    @property
    def duration(self):
        if self.begin is not None and self.end is not None:
            return self.end - self.begin
        return self.file.duration

    @property
    def meta(self):
        return {
            'speaker': self.speaker.name,
            'file': self.file.name,
            'begin': self.begin,
            'end': self.end,
            'channel': self.channel,
            'text': self.text,
            'ignored': self.ignored,
            'features': self.features,
            'feature_length': self.feature_length,
        }

    def set_speaker(self, speaker: Speaker):
        self.speaker = speaker
        self.speaker.add_utterance(self)
        self.file.add_utterance(self)

    @property
    def is_segment(self):
        return self.begin is not None and self.end is not None

    def text_for_scp(self):
        return self.text.split()

    def text_int_for_scp(self):
        if self.speaker.dictionary is None:
            return
        text = self.text_for_scp()
        new_text = []
        for i in range(len(text)):
            t = text[i]
            lookup = self.speaker.dictionary.to_int(t)
            for w in lookup:
                if w == self.speaker.dictionary.oov_int:
                    self.oovs.append(text[i])
                new_text.append(w)
        return new_text

    def segment_for_scp(self):
        return [self.file.name, self.begin, self.end, self.channel]

    @property
    def name(self):
        base = f'{self.file_name}-{self.speaker_name}'
        if self.is_segment:
            base = f'{self.file_name}-{self.speaker_name}-{self.begin}-{self.end}'
        return base.replace(' ', '-space-').replace('.', '-').replace('_', '-')


class VadArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feats_scp_paths: Dict[str, str]
    vad_scp_paths: Dict[str, str]
    vad_options: ConfigDict


class MfccArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feats_scp_paths: Dict[str, str]
    lengths_paths: Dict[str, str]
    segment_paths: Dict[str, str]
    wav_paths: Dict[str, str]
    mfcc_options: ConfigDict


class CompileTrainGraphsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    tree_path: str
    model_path: str
    text_int_paths: Dict[str, str]
    disambig_paths: Dict[str, str]
    lexicon_fst_paths: Dict[str, str]
    fst_scp_paths: Dict[str, str]


class MonoAlignEqualArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    fst_scp_paths: Dict[str, str]
    ali_ark_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str


class AccStatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str


class AlignArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    fst_scp_paths: Dict[str, str]
    feature_strings: Dict[str, str]
    model_path: str
    ali_paths: Dict[str, str]
    score_paths: Dict[str, str]
    loglike_paths: Dict[str, str]
    align_options: ConfigDict


class CompileInformationArguments(NamedTuple):
    align_log_paths: str


class AliToCtmArguments(NamedTuple):
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
    ctm_paths: Dict[str, str]
    dictionaries: List[str]
    utterances: Dict[str, Dict[str, Utterance]]
    dictionary_data: Dict[str, DictionaryData]


class NoCleanupWordCtmArguments(NamedTuple):
    ctm_paths: Dict[str, str]
    dictionaries: List[str]
    utterances: Dict[str, Dict[str, Utterance]]
    dictionary_data: Dict[str, DictionaryData]


class PhoneCtmArguments(NamedTuple):
    ctm_paths: Dict[str, str]
    dictionaries: List[str]
    utterances: Dict[str, Dict[str, Utterance]]
    reversed_phone_mappings: Dict[str, ReversedMappingType]
    positions: Dict[str, List[str]]


class CombineCtmArguments(NamedTuple):
    dictionaries: List[str]
    files: Dict[str, File]
    dictionary_data: Dict[str, DictionaryData]
    cleanup_textgrids: bool


class ExportTextGridArguments(NamedTuple):
    files: Dict[str, File]
    frame_shift: int
    output_directory: str
    backup_output_directory: str


class TreeStatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    ci_phones: str
    model_path: str
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    treeacc_paths: Dict[str, str]


class ConvertAlignmentsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    model_path: str
    tree_path: str
    align_model_path: str
    ali_paths: Dict[str, str]
    new_ali_paths: Dict[str, str]


class AlignmentImprovementArguments(NamedTuple):
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
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    spk2utt_paths: Dict[str, str]
    trans_paths: Dict[str, str]
    fmllr_options: ConfigDict


class AccStatsTwoFeatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    model_path: str
    feature_strings: Dict[str, str]
    si_feature_strings: Dict[str, str]


class LdaAccStatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    lda_options: ConfigDict
    acc_paths: Dict[str, str]


class CalcLdaMlltArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    lda_options: ConfigDict
    macc_paths: Dict[str, str]


class MapAccStatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    ali_paths: Dict[str, str]
    acc_paths: Dict[str, str]


class GmmGselectArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: ConfigDict
    dubm_model: str
    gselect_paths: Dict[str, str]


class AccGlobalStatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: ConfigDict
    gselect_paths: Dict[str, str]
    acc_paths: Dict[str, str]
    dubm_path: str


class GaussToPostArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: ConfigDict
    post_paths: Dict[str, str]
    dubm_path: str


class AccIvectorStatsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: ConfigDict
    ie_path: str
    post_paths: Dict[str, str]
    acc_init_paths: Dict[str, str]


class ExtractIvectorsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ivector_options: ConfigDict
    ali_paths: Dict[str, str]
    ie_path: str
    ivector_paths: Dict[str, str]
    weight_paths: Dict[str, str]
    model_path: str
    dubm_path: str


class CompileUtteranceTrainGraphsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    disambig_int_paths: Dict[str, str]
    disambig_L_fst_paths: Dict[str, str]
    fst_paths: Dict[str, str]
    graphs_paths: Dict[str, str]
    model_path: str
    tree_path: str


class TestUtterancesArguments(NamedTuple):
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
    log_path: str
    dictionaries: List[str]
    vad_paths: Dict[str, str]
    segmentation_options: ConfigDict


class ClassifySpeakersArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    model_path: str
    labels_path: str
    ivector_paths: Dict[str, str]


class GeneratePronunciationsArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    text_int_paths: Dict[str, str]
    word_boundary_paths: Dict[str, str]
    ali_paths: Dict[str, str]
    model_path: str
    pron_paths: Dict[str, str]


class CreateHclgArguments(NamedTuple):
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
    hclg_options: ConfigDict
    words_mapping: MappingType


class DecodeArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    decode_options: ConfigDict
    model_path: str
    lat_paths: Dict[str, str]
    words_paths: Dict[str, str]
    hclg_paths: Dict[str, str]


class ScoreArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    score_options: ConfigDict
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    carpa_rescored_lat_paths: Dict[str, str]
    words_paths: Dict[str, str]
    tra_paths: Dict[str, str]


class LmRescoreArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    lm_rescore_options: ConfigDict
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    old_g_paths: Dict[str, str]
    new_g_paths: Dict[str, str]


class CarpaLmRescoreArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    old_g_paths: Dict[str, str]
    new_g_paths: Dict[str, str]


class InitialFmllrArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: ConfigDict
    pre_trans_paths: Dict[str, str]
    lat_paths: Dict[str, str]
    spk2utt_paths: Dict[str, str]


class LatGenFmllrArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    decode_options: ConfigDict
    words_paths: Dict[str, str]
    hclg_paths: Dict[str, str]
    tmp_lat_paths: Dict[str, str]


class FinalFmllrArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: ConfigDict
    trans_paths: Dict[str, str]
    spk2utt_paths: Dict[str, str]
    tmp_lat_paths: Dict[str, str]


class FmllrRescoreArguments(NamedTuple):
    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: ConfigDict
    tmp_lat_paths: Dict[str, str]
    final_lat_paths: Dict[str, str]


class Job(object):
    def __init__(self, name: int):
        self.name = name
        self.speakers: List[Speaker] = []
        self.dictionaries: Set[Dictionary] = set()

        self.subset_utts: Set[Utterance] = set()
        self.subset_speakers: Set[Speaker] = set()
        self.subset_dictionaries: Set[Dictionary] = set()

    def add_speaker(self, speaker: Speaker):
        self.speakers.append(speaker)
        self.dictionaries.add(speaker.dictionary)

    def set_subset(self, subset_utts):
        if subset_utts is None:
            self.subset_utts = set()
            self.subset_speakers = set()
            self.subset_dictionaries = set()
        else:
            self.subset_utts = set(subset_utts)
            self.subset_speakers = set(u.speaker for u in subset_utts if u.speaker in self.speakers)
            self.subset_dictionaries = set(s.dictionary for s in self.subset_speakers)

    def text_scp_data(self):
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

    def text_int_scp_data(self):
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
                data[key][u.name] = u.text_int_for_scp()
        return data

    def wav_scp_data(self):
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

    def utt2spk_scp_data(self):
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

    def feat_scp_data(self):
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

    def spk2utt_scp_data(self):
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
            data[key][s.name] = sorted([u.name for u in s.utterances.values() if not u.ignored and not (
                    self.subset_utts and u not in self.subset_utts)])
        return data

    def cmvn_scp_data(self):
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

    def segments_scp_data(self):
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

    def construct_path_dictionary(self, directory, identifier, extension):
        output = {}
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(directory, f'{identifier}.{dict_name}.{self.name}.{extension}')
        return output

    def construct_dictionary_dependent_paths(self, directory, identifier, extension):
        output = {}
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(directory, f'{identifier}.{dict_name}.{extension}')
        return output

    @property
    def dictionary_count(self):
        if self.subset_dictionaries:
            return len(self.subset_dictionaries)
        return len(self.dictionaries)

    @property
    def current_dictionaries(self):
        if self.subset_dictionaries:
            return self.subset_dictionaries
        return self.dictionaries

    @property
    def current_dictionary_names(self):
        if self.subset_dictionaries:
            return sorted(x.name for x in self.subset_dictionaries)
        if self.dictionaries == {None}:
            return [None]
        return sorted(x.name for x in self.dictionaries)

    def set_feature_config(self, feature_config: FeatureConfig):
        self.feature_config = feature_config

    def construct_base_feature_string(self, corpus: CorpusType, all_feats: bool = False ) -> str:
        if all_feats:
            feat_path = os.path.join(corpus.output_directory, 'feats.scp')
            utt2spk_path = os.path.join(corpus.output_directory, 'utt2spk.scp')
            cmvn_path = os.path.join(corpus.output_directory, 'cmvn.scp')
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            feats += " add-deltas ark:- ark:- |"
            return feats
        utt2spks = self.construct_path_dictionary(corpus.split_directory, 'utt2spk', 'scp')
        cmvns = self.construct_path_dictionary(corpus.split_directory, 'cmvn', 'scp')
        features = self.construct_path_dictionary(corpus.split_directory, 'feats', 'scp')
        for dict_name in self.current_dictionary_names:
            feat_path = features[dict_name]
            cmvn_path = cmvns[dict_name]
            utt2spk_path = utt2spks[dict_name]
            feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
            if self.feature_config.deltas:
                feats += " add-deltas ark:- ark:- |"

            return feats

    def construct_feature_proc_strings(self, aligner: Union[AlignerType, SpeakerClassifier, Transcriber],
                                       speaker_independent: bool = False) -> Dict[str, str]:
        lda_mat_path = None
        fmllrs = {}
        if aligner.working_directory is not None:
            lda_mat_path = os.path.join(aligner.working_directory, 'lda.mat')
            if not os.path.exists(lda_mat_path):
                lda_mat_path = None

            fmllrs = self.construct_path_dictionary(aligner.working_directory, 'trans', 'ark')
        utt2spks = self.construct_path_dictionary(aligner.data_directory, 'utt2spk', 'scp')
        cmvns = self.construct_path_dictionary(aligner.data_directory, 'cmvn', 'scp')
        features = self.construct_path_dictionary(aligner.data_directory, 'feats', 'scp')
        vads = self.construct_path_dictionary(aligner.data_directory, 'vad', 'scp')
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
                feats = f'ark,s,cs:add-deltas scp:{feat_path} ark:- |'
                if aligner.uses_cmvn:
                    feats += ' apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |'
                feats += f' select-voiced-frames ark:- scp,s,cs:{vad_path} ark:- |'
            elif not os.path.exists(cmvn_path) and aligner.uses_cmvn:
                feats = f'ark,s,cs:add-deltas scp:{feat_path} ark:- |'
                if aligner.uses_cmvn:
                    feats += ' apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |'
            else:
                feats = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
                if lda_mat_path is not None:
                    if not os.path.exists(lda_mat_path):
                        raise Exception(f'Could not find {lda_mat_path}')
                    feats += f' splice-feats --left-context={self.feature_config.splice_left_context} --right-context={self.feature_config.splice_right_context} ark:- ark:- |'
                    feats += f" transform-feats {lda_mat_path} ark:- ark:- |"
                elif aligner.uses_splices:
                    feats += f' splice-feats --left-context={self.feature_config.splice_left_context} --right-context={self.feature_config.splice_right_context} ark:- ark:- |'
                elif self.feature_config.deltas:
                    feats += " add-deltas ark:- ark:- |"

                if fmllr_trans_path is not None and not (aligner.speaker_independent or speaker_independent):
                    if not os.path.exists(fmllr_trans_path):
                        raise Exception(f'Could not find {fmllr_trans_path}')
                    feats += f" transform-feats --utt2spk=ark:{utt2spk_path} ark:{fmllr_trans_path} ark:- ark:- |"
            feat_strings[dict_name] = feats
        return feat_strings

    def compile_utterance_train_graphs_arguments(self, validator: CorpusValidator):
        dictionary_paths = validator.dictionary.output_paths
        disambig_paths = {k: os.path.join(v, 'phones', 'disambig.int') for k, v in dictionary_paths.items()}
        lexicon_fst_paths = {k: os.path.join(v, 'L_disambig.fst') for k, v in dictionary_paths.items()}
        return CompileUtteranceTrainGraphsArguments(
            os.path.join(validator.trainer.working_log_directory, f'utterance_fst.{self.name}.log'),
            self.current_dictionary_names,
            disambig_paths,
            lexicon_fst_paths,
            self.construct_path_dictionary(validator.trainer.data_directory, 'utt2fst', 'scp'),
            self.construct_path_dictionary(validator.trainer.working_directory, 'utterance_graphs', 'fst'),
            validator.trainer.current_model_path,
            validator.trainer.tree_path,
        )

    def test_utterances_arguments(self, validator: CorpusValidator):
        dictionary_paths = validator.dictionary.output_paths
        words_paths = {k: os.path.join(v, 'words.txt') for k, v in dictionary_paths.items()}
        return TestUtterancesArguments(
            os.path.join(validator.trainer.working_directory, f'utterance_fst.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(validator.trainer),
            words_paths,
            self.construct_path_dictionary(validator.trainer.working_directory, 'utterance_graphs', 'fst'),
            self.construct_path_dictionary(validator.trainer.data_directory, 'text', 'int.scp'),
            self.construct_path_dictionary(validator.trainer.working_directory, 'edits', 'scp'),
            self.construct_path_dictionary(validator.trainer.working_directory, 'aligned', 'int'),
            validator.trainer.current_model_path
        )

    def extract_ivector_arguments(self, ivector_extractor: SpeakerClassifier):
        return ExtractIvectorsArguments(
            os.path.join(ivector_extractor.working_log_directory, f'extract_ivectors.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(ivector_extractor),
            ivector_extractor.ivector_options,
            self.construct_path_dictionary(ivector_extractor.working_directory, 'ali', 'ark'),
            ivector_extractor.ie_path,
            self.construct_path_dictionary(ivector_extractor.working_directory, 'ivectors', 'scp'),
            self.construct_path_dictionary(ivector_extractor.working_directory, 'weights', 'ark'),
            ivector_extractor.model_path,
            ivector_extractor.dubm_path,
        )

    def classify_speaker_arguments(self, ivector_extractor: SpeakerClassifier):
        return ClassifySpeakersArguments(
            os.path.join(ivector_extractor.working_log_directory, f'classify_speakers.{self.name}.log'),
            self.current_dictionary_names,
            ivector_extractor.speaker_classification_model_path,
            ivector_extractor.speaker_labels_path,
            self.construct_path_dictionary(ivector_extractor.working_directory, 'ivectors', 'scp'),
        )

    def create_hclgs_arguments(self, transcriber: Transcriber):
        args = {}

        for dictionary in self.current_dictionaries:
            dict_name = dictionary.name
            args[dict_name] = CreateHclgArguments(
                os.path.join(transcriber.model_directory, 'log', f'hclg.{dict_name}.log'),
                transcriber.model_directory,
                os.path.join(transcriber.model_directory, '{file_name}' + f'.{dict_name}.fst'),
                os.path.join(transcriber.model_directory, f'words.{dict_name}.txt'),
                os.path.join(transcriber.model_directory, f'G.{dict_name}.carpa'),
                transcriber.language_model.small_arpa_path,
                transcriber.language_model.medium_arpa_path,
                transcriber.language_model.carpa_path,
                transcriber.model_path,
                dictionary.disambig_path,
                os.path.join(dictionary.phones_dir, 'disambig.int'),
                transcriber.hclg_options,
                dictionary.words_mapping
            )
        return args

    def decode_arguments(self, transcriber: Transcriber):
        return DecodeArguments(
            os.path.join(transcriber.working_log_directory, f'decode.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.transcribe_config.decode_options,
            transcriber.alignment_model_path,
            self.construct_path_dictionary(transcriber.working_directory, 'lat', 'ark'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'words', 'txt'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'hclg', 'fst'),
        )

    def score_arguments(self, transcriber: Transcriber):
        return ScoreArguments(
            os.path.join(transcriber.working_log_directory, f'score.{self.name}.log'),
            self.current_dictionary_names,
            transcriber.transcribe_config.score_options,
            self.construct_path_dictionary(transcriber.working_directory, 'lat', 'ark'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat.rescored', 'ark'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat.carpa.rescored', 'ark'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'words', 'txt'),
            self.construct_path_dictionary(transcriber.evaluation_directory, 'tra', 'scp'),
        )

    def lm_rescore_arguments(self, transcriber: Transcriber):
        return LmRescoreArguments(
            os.path.join(transcriber.working_log_directory, f'lm_rescore.{self.name}.log'),
            self.current_dictionary_names,
            transcriber.transcribe_config.lm_rescore_options,
            self.construct_path_dictionary(transcriber.working_directory, 'lat', 'ark'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat.rescored', 'ark'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'G.small', 'fst'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'G.med', 'fst'),
        )

    def carpa_lm_rescore_arguments(self, transcriber: Transcriber):
        return CarpaLmRescoreArguments(
            os.path.join(transcriber.working_log_directory, f'carpa_lm_rescore.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(transcriber.working_directory, 'lat.rescored', 'ark'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat.carpa.rescored', 'ark'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'G.med', 'fst'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'G', 'carpa'),
        )

    def initial_fmllr_arguments(self, transcriber: Transcriber):
        return InitialFmllrArguments(
            os.path.join(transcriber.working_log_directory, f'initial_fmllr.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.fmllr_options,
            self.construct_path_dictionary(transcriber.working_directory, 'trans', 'ark'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat', 'ark'),
            self.construct_path_dictionary(transcriber.data_directory, 'spk2utt', 'scp'),
        )

    def lat_gen_fmllr_arguments(self, transcriber: Transcriber):
        return LatGenFmllrArguments(
            os.path.join(transcriber.working_log_directory, f'lat_gen_fmllr.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.transcribe_config.decode_options,
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'words', 'txt'),
            self.construct_dictionary_dependent_paths(transcriber.model_directory, 'hclg', 'fst'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat.tmp', 'ark'),
        )

    def final_fmllr_arguments(self, transcriber: Transcriber):
        return FinalFmllrArguments(
            os.path.join(transcriber.working_log_directory, f'final_fmllr.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.fmllr_options,
            self.construct_path_dictionary(transcriber.working_directory, 'trans', 'ark'),
            self.construct_path_dictionary(transcriber.data_directory, 'spk2utt', 'scp'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat.tmp', 'ark'),
        )

    def fmllr_rescore_arguments(self, transcriber: Transcriber):
        return FmllrRescoreArguments(
            os.path.join(transcriber.working_log_directory, f'fmllr_rescore.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(transcriber),
            transcriber.model_path,
            transcriber.fmllr_options,
            self.construct_path_dictionary(transcriber.working_directory, 'lat.tmp', 'ark'),
            self.construct_path_dictionary(transcriber.working_directory, 'lat', 'ark'),
        )

    def vad_arguments(self, corpus: CorpusType):
        return VadArguments(
            os.path.join(corpus.split_directory, 'log', f'compute_vad.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(corpus.split_directory, 'feats', 'scp'),
            self.construct_path_dictionary(corpus.split_directory, 'vad', 'scp'),
            corpus.vad_config
        )

    def segments_vad_arguments(self, segmenter: Segmenter):
        return SegmentVadArguments(
            os.path.join(segmenter.corpus.split_directory, 'log', f'segment_vad.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(segmenter.corpus.split_directory, 'vad', 'scp'),
            segmenter.segmentation_config.segmentation_options
        )

    def mfcc_arguments(self, corpus: CorpusType):
        return MfccArguments(
            os.path.join(corpus.split_directory, 'log', f'make_mfcc.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(corpus.split_directory, 'feats', 'scp'),
            self.construct_path_dictionary(corpus.split_directory, 'utterance_lengths', 'scp'),
            self.construct_path_dictionary(corpus.split_directory, 'segments', 'scp'),
            self.construct_path_dictionary(corpus.split_directory, 'wav', 'scp'),
            self.feature_config.mfcc_options()
        )

    def acc_stats_arguments(self, aligner: AlignerType):
        return AccStatsArguments(
            os.path.join(aligner.working_directory, 'log', f'acc.{aligner.iteration}.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, str(aligner.iteration), 'acc'),
            aligner.current_model_path
        )

    def mono_align_equal_arguments(self, aligner: AlignerType) -> MonoAlignEqualArguments:
        return MonoAlignEqualArguments(
            os.path.join(aligner.working_log_directory, f'mono_align_equal.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, 'fsts', 'scp'),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, '0', 'acc'),
            aligner.current_model_path
        )

    def align_arguments(self, aligner: AlignerType):
        if aligner.iteration is not None:
            log_path = os.path.join(aligner.working_log_directory, f'align.{aligner.iteration}.{self.name}.log')
        else:
            log_path = os.path.join(aligner.working_log_directory, f'align.{self.name}.log')
        return AlignArguments(
            log_path,
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, 'fsts', 'scp'),
            self.construct_feature_proc_strings(aligner),
            aligner.alignment_model_path,
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'scores'),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'loglikes'),
            aligner.align_options
        )

    def compile_information_arguments(self, aligner: AlignerType):
        if aligner.iteration is not None:
            log_path = os.path.join(aligner.working_log_directory, f'align.{aligner.iteration}.{self.name}.log')
        else:
            log_path = os.path.join(aligner.working_log_directory, f'align.{self.name}.log')
        return CompileInformationArguments(
            log_path
        )

    def word_boundary_int_files(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = os.path.join(dictionary.phones_dir, 'word_boundary.int')
        return data

    def reversed_phone_mappings(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_phone_mapping
        return data

    def reversed_word_mappings(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_word_mapping
        return data

    def words_mappings(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words_mapping
        return data

    def words(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words
        return data

    def punctuation(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.punctuation
        return data

    def clitic_set(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_set
        return data

    def clitic_markers(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_markers
        return data

    def compound_markers(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.compound_markers
        return data

    def strip_diacritics(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.strip_diacritics
        return data

    def oov_codes(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_code
        return data

    def oov_ints(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_int
        return data

    def positions(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.positions
        return data

    def silences(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.silences
        return data

    def multilingual_ipa(self):
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.multilingual_ipa
        return data

    def generate_pronunciations_arguments(self, aligner: AlignerType):
        return GeneratePronunciationsArguments(
            os.path.join(aligner.working_log_directory, f'generate_pronunciations.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.data_directory, 'text', 'int.scp'),
            self.word_boundary_int_files(),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            aligner.current_model_path,
            self.construct_path_dictionary(aligner.working_directory, 'prons', 'scp'),
        )

    def alignment_improvement_arguments(self, aligner: AlignerType):
        return AlignmentImprovementArguments(
            os.path.join(aligner.working_log_directory, f'alignment_analysis.{self.name}.log'),
            self.current_dictionary_names,
            aligner.current_model_path,
            self.construct_path_dictionary(aligner.data_directory, 'text', 'int.scp'),
            self.word_boundary_int_files(),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.feature_config.frame_shift,
            self.reversed_phone_mappings(),
            self.positions(),
            self.construct_path_dictionary(aligner.working_directory, f'phone.{aligner.iteration}', 'ctm'),
        )

    def ali_to_word_ctm_arguments(self, aligner: AlignerType):
        return AliToCtmArguments(
            os.path.join(aligner.working_log_directory, f'get_word_ctm.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.data_directory, 'text', 'int.scp'),
            self.word_boundary_int_files(),
            round(self.feature_config.frame_shift / 1000, 4),
            aligner.alignment_model_path,
            self.construct_path_dictionary(aligner.working_directory, 'word', 'ctm'),
            True
        )

    def ali_to_phone_ctm_arguments(self, aligner: AlignerType):
        return AliToCtmArguments(
            os.path.join(aligner.working_log_directory, f'get_phone_ctm.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.data_directory, 'text', 'int.scp'),
            self.word_boundary_int_files(),
            round(self.feature_config.frame_shift / 1000, 4),
            aligner.alignment_model_path,
            self.construct_path_dictionary(aligner.working_directory, 'phone', 'ctm'),
            False
        )

    def job_utts(self):
        data = {}
        speakers = self.subset_speakers
        if not speakers:
            speakers = self.speakers
        for s in speakers:
            if s.dictionary.name not in data:
                data[s.dictionary.name] = {}
            data[s.dictionary.name].update(s.utterances)
        return data

    def job_files(self):
        data = {}
        speakers = self.subset_speakers
        if not speakers:
            speakers = self.speakers
        for s in speakers:
            for f in s.files:
                for i, sf in enumerate(f.speaker_ordering):
                    if sf.name == s.name:
                        sf.dictionary_data = s.dictionary_data
                data[f.name] = f
        return data

    def cleanup_word_ctm_arguments(self, aligner: AlignerType):
        return CleanupWordCtmArguments(
            self.construct_path_dictionary(aligner.align_directory, 'word', 'ctm'),
            self.current_dictionary_names,
            self.job_utts(),
            self.dictionary_data()
        )

    def no_cleanup_word_ctm_arguments(self, aligner: AlignerType):
        return NoCleanupWordCtmArguments(
            self.construct_path_dictionary(aligner.align_directory, 'word', 'ctm'),
            self.current_dictionary_names,
            self.job_utts(),
            self.dictionary_data()
        )

    def phone_ctm_arguments(self, aligner: AlignerType):
        return PhoneCtmArguments(
            self.construct_path_dictionary(aligner.align_directory, 'phone', 'ctm'),
            self.current_dictionary_names,
            self.job_utts(),
            self.reversed_phone_mappings(),
            self.positions()
        )

    def dictionary_data(self) -> Dict[str, DictionaryData]:
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.data()
        return data

    def combine_ctm_arguments(self, aligner: AlignerType):
        return CombineCtmArguments(
            self.current_dictionary_names,
            self.job_files(),
            self.dictionary_data(),
            aligner.align_config.cleanup_textgrids
        )

    def export_textgrid_arguments(self, aligner: AlignerType):
        return ExportTextGridArguments(
            aligner.corpus.files,
            aligner.feature_config.frame_shift,
            aligner.textgrid_output,
            aligner.backup_output_directory,
        )

    def tree_stats_arguments(self, aligner: AlignerType):
        return TreeStatsArguments(
            os.path.join(aligner.working_log_directory, f'acc_tree.{self.name}.log'),
            self.current_dictionary_names,
            aligner.dictionary.silence_csl,
            aligner.previous_trainer.alignment_model_path,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.previous_trainer.align_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, 'tree', 'acc'),
        )

    def convert_alignment_arguments(self, aligner: AlignerType):
        return ConvertAlignmentsArguments(
            os.path.join(aligner.working_log_directory, f'convert_alignments.{self.name}.log'),
            self.current_dictionary_names,
            aligner.current_model_path,
            aligner.tree_path,
            aligner.previous_trainer.alignment_model_path,
            self.construct_path_dictionary(aligner.previous_trainer.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
        )

    def calc_fmllr_arguments(self, aligner: AlignerType):

        return CalcFmllrArguments(
            os.path.join(aligner.working_log_directory, f'calc_fmllr.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            aligner.current_model_path,
            self.construct_path_dictionary(aligner.data_directory, 'spk2utt', 'scp'),
            self.construct_path_dictionary(aligner.working_directory, 'trans', 'ark'),
            aligner.fmllr_options
        )

    def acc_stats_two_feats_arguments(self, aligner: AlignerType):
        return AccStatsTwoFeatsArguments(
            os.path.join(aligner.working_log_directory, f'acc_stats_two_feats.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, 'acc', 'ark'),
            aligner.current_model_path,
            self.construct_feature_proc_strings(aligner),
            self.construct_feature_proc_strings(aligner, speaker_independent=True),
        )

    def lda_acc_stats_arguments(self, aligner: LdaTrainer):
        return LdaAccStatsArguments(
            os.path.join(aligner.working_log_directory, f'lda_acc_stats.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.previous_trainer.working_directory, 'ali', 'ark'),
            aligner.previous_trainer.alignment_model_path,
            aligner.lda_options,
            self.construct_path_dictionary(aligner.working_directory, 'lda', 'acc'),
        )

    def calc_lda_mllt_arguments(self, aligner: LdaTrainer):
        return CalcLdaMlltArguments(
            os.path.join(aligner.working_log_directory, f'lda_mllt.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            self.construct_path_dictionary(aligner.working_directory, 'ali', 'ark'),
            aligner.current_model_path,
            aligner.lda_options,
            self.construct_path_dictionary(aligner.working_directory, 'lda', 'macc'),
        )

    def ivector_acc_stats_arguments(self, trainer: IvectorExtractorTrainer):
        return AccIvectorStatsArguments(
            os.path.join(trainer.working_log_directory, f'ivector_acc.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(trainer),
            trainer.ivector_options,
            trainer.current_ie_path,
            self.construct_path_dictionary(trainer.working_directory, 'post', 'ark'),
            self.construct_path_dictionary(trainer.working_directory, 'ivector', 'acc'),
        )

    def map_acc_stats_arguments(self, aligner: AdaptingAligner):
        return MapAccStatsArguments(
            os.path.join(aligner.working_log_directory, f'map_acc_stats.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.current_model_path,
            self.construct_path_dictionary(aligner.previous_aligner.align_directory, 'ali', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, 'map', 'acc'),
        )

    def gmm_gselect_arguments(self, aligner: IvectorExtractorTrainer):
        return GmmGselectArguments(
            os.path.join(aligner.working_log_directory, f'gmm_gselect.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.ivector_options,
            aligner.current_dubm_path,
            self.construct_path_dictionary(aligner.working_directory, 'gselect', 'ark'),
        )

    def acc_global_stats_arguments(self, aligner: IvectorExtractorTrainer):
        return AccGlobalStatsArguments(
            os.path.join(aligner.working_log_directory, f'acc_global_stats.{aligner.iteration}.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.ivector_options,
            self.construct_path_dictionary(aligner.working_directory, 'gselect', 'ark'),
            self.construct_path_dictionary(aligner.working_directory, f'global.{aligner.iteration}', 'acc'),
            aligner.current_dubm_path,
        )

    def gauss_to_post_arguments(self, aligner: IvectorExtractorTrainer):
        return GaussToPostArguments(
            os.path.join(aligner.working_log_directory, f'gauss_to_post.{self.name}.log'),
            self.current_dictionary_names,
            self.construct_feature_proc_strings(aligner),
            aligner.ivector_options,
            self.construct_path_dictionary(aligner.working_directory, 'post', 'ark'),
            aligner.current_dubm_path,
        )

    def compile_train_graph_arguments(self, aligner: AlignerType):
        dictionary_paths = aligner.dictionary.output_paths
        disambig_paths = {k: os.path.join(v, 'phones', 'disambig.int') for k, v in dictionary_paths.items()}
        lexicon_fst_paths = {k: os.path.join(v, 'L.fst') for k, v in dictionary_paths.items()}
        model_path = aligner.current_model_path
        if not os.path.exists(model_path):
            model_path = aligner.alignment_model_path
        return CompileTrainGraphsArguments(
            os.path.join(aligner.working_log_directory, f'compile_train_graphs.{self.name}.log'),
            self.current_dictionary_names,
            os.path.join(aligner.working_directory, 'tree'),
            model_path,
            self.construct_path_dictionary(aligner.data_directory, 'text', 'int.scp'),
            disambig_paths,
            lexicon_fst_paths,
            self.construct_path_dictionary(aligner.working_directory, 'fsts', 'scp')
        )

    def utt2fst_scp_data(self, corpus: CorpusType, num_frequent_words: int=10) -> Dict[List[Tuple[str, str]]]:
        data = {}
        most_frequent = {}
        for dict_name, utterances in self.job_utts().items():
            data[dict_name] = []
            for u_name, utterance in utterances.items():
                new_text = []
                dictionary = utterance.speaker.dictionary
                if dictionary.name not in most_frequent:
                    word_frequencies = corpus.get_word_frequency(dictionary)
                    most_frequent[dictionary.name] = sorted(word_frequencies.items(), key=lambda x: -x[1])[:num_frequent_words]

                for t in utterance.text:
                    lookup = utterance.speaker.dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                    new_text.extend(x for x in lookup if x != '')
                data[dict_name].append((u_name, dictionary.create_utterance_fst(new_text, most_frequent[dictionary.name])))
        return data

    def output_utt_fsts(self, corpus: CorpusType, num_frequent_words: int=10):
        utt2fst = self.utt2fst_scp_data(corpus, num_frequent_words)
        for dict_name, scp in utt2fst.items():
            utt2fst_scp_path = os.path.join(corpus.split_directory, f'utt2fst.{dict_name}.{self.name}.scp')
            save_scp(scp, utt2fst_scp_path, multiline=True)

    def output_to_directory(self, split_directory):
        wav = self.wav_scp_data()
        for dict_name, scp in wav.items():
            wav_scp_path = os.path.join(split_directory, f'wav.{dict_name}.{self.name}.scp')
            output_mapping(scp, wav_scp_path, skip_safe=True)

        spk2utt = self.spk2utt_scp_data()
        for dict_name, scp in spk2utt.items():
            spk2utt_scp_path = os.path.join(split_directory, f'spk2utt.{dict_name}.{self.name}.scp')
            output_mapping(scp, spk2utt_scp_path)

        feats = self.feat_scp_data()
        for dict_name, scp in feats.items():
            feats_scp_path = os.path.join(split_directory, f'feats.{dict_name}.{self.name}.scp')
            output_mapping(scp, feats_scp_path)

        cmvn = self.cmvn_scp_data()
        for dict_name, scp in cmvn.items():
            cmvn_scp_path = os.path.join(split_directory, f'cmvn.{dict_name}.{self.name}.scp')
            output_mapping(scp, cmvn_scp_path)

        utt2spk = self.utt2spk_scp_data()
        for dict_name, scp in utt2spk.items():
            utt2spk_scp_path = os.path.join(split_directory, f'utt2spk.{dict_name}.{self.name}.scp')
            output_mapping(scp, utt2spk_scp_path)

        segments = self.segments_scp_data()
        for dict_name, scp in segments.items():
            segments_scp_path = os.path.join(split_directory, f'segments.{dict_name}.{self.name}.scp')
            output_mapping(scp, segments_scp_path)

        text_scp = self.text_scp_data()
        for dict_name, scp in text_scp.items():
            if not scp:
                continue
            text_scp_path = os.path.join(split_directory, f'text.{dict_name}.{self.name}.scp')
            output_mapping(scp, text_scp_path)

        text_int = self.text_int_scp_data()
        for dict_name, scp in text_int.items():
            if dict_name is None:
                continue
            if not scp:
                continue
            text_int_scp_path = os.path.join(split_directory, f'text.{dict_name}.{self.name}.int.scp')
            output_mapping(scp, text_int_scp_path)
