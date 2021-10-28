from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Union
if TYPE_CHECKING:
    from logging import Logger
    from ..features.config import FeatureConfig
    from ..dictionary import DictionaryType
    from . import CorpusGroupedOneToOne, CorpusGroupedOneToMany
import os
import sys
import time
import logging
import subprocess
import yaml
from collections import defaultdict

from .classes import Speaker, File, Utterance, Job, parse_file
from .helper import get_wav_info
from ..exceptions import CorpusError
from collections import Counter
from ..utils import thirdparty_binary
from ..helper import load_scp, output_mapping, save_groups, filter_scp

import multiprocessing as mp
from queue import Empty
from ..multiprocessing.helper import Stopped
from ..multiprocessing.corpus import CorpusProcessWorker
from ..exceptions import CorpusError, SoxError,  \
    TextParseError, TextGridParseError, KaldiProcessingError
from ..multiprocessing.features import mfcc, compute_vad, calc_cmvn
from ..features.config import FeatureConfig
from .helper import find_exts


class BaseCorpus(object):
    """
    Class that stores information about the dataset to align.

    Corpus objects have a number of mappings from either utterances or speakers
    to various properties, and mappings between utterances and speakers.

    See http://kaldi-asr.org/doc/data_prep.html for more information about
    the files that are created by this class.


    Parameters
    ----------
    directory : str
        Directory of the dataset to align
    output_directory : str
        Directory to store generated data for the Kaldi binaries
    speaker_characters : int, optional
        Number of characters in the filenames to count as the speaker ID,
        if not specified, speaker IDs are generated from directory names
    num_jobs : int, optional
        Number of processes to use, defaults to 3

    Raises
    ------
    CorpusError
        Raised if the specified corpus directory does not exist
    SampleRateError
        Raised if the wav files in the dataset do not share a consistent sample rate

    """

    def __init__(self, directory: str, output_directory: str,
                 speaker_characters: Union[int, str]=0,
                 num_jobs: int=3, sample_rate: int=16000, debug: bool=False, logger: Optional[Logger]=None, use_mp: bool=True,
                 punctuation: str=None, clitic_markers: str=None, audio_directory:
            Optional[str]=None, skip_load: bool=False, parse_text_only_files: bool=False):
        self.audio_directory = audio_directory
        self.punctuation = punctuation
        self.clitic_markers = clitic_markers
        self.debug = debug
        self.use_mp = use_mp
        log_dir = os.path.join(output_directory, 'logging')
        os.makedirs(log_dir, exist_ok=True)
        self.name = os.path.basename(directory)
        self.log_file = os.path.join(log_dir, 'corpus.log')
        if logger is None:
            self.logger = logging.getLogger('corpus_setup')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
            handler.setFormatter = logging.Formatter('%(name)s %(message)s')
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        if not os.path.exists(directory):
            raise CorpusError(f'The directory \'{directory}\' does not exist.')
        if not os.path.isdir(directory):
            raise CorpusError(f'The specified path for the corpus ({directory}) is not a directory.')

        num_jobs = max(num_jobs, 1)
        if num_jobs == 1:
            self.use_mp = False
        self.original_num_jobs = num_jobs
        self.logger.info('Setting up corpus information...')
        self.directory = directory
        self.output_directory = os.path.join(output_directory, 'corpus_data')
        self.temp_directory = os.path.join(self.output_directory, 'temp')
        os.makedirs(self.temp_directory, exist_ok=True)
        self.speaker_characters = speaker_characters
        if speaker_characters == 0:
            self.speaker_directories = True
        else:
            self.speaker_directories = False
        self.num_jobs = num_jobs
        self.speakers: Dict[str, Speaker] = {}
        self.files: Dict[str, File] = {}
        self.utterances: Dict[str, Utterance] = {}
        self.sound_file_errors = []
        self.decode_error_files = []
        self.transcriptions_without_wavs = []
        self.no_transcription_files = []
        self.textgrid_read_errors = {}
        self.groups = []
        self.speaker_groups = []
        self.word_counts = Counter()
        self.sample_rate = sample_rate
        if self.use_mp:
            self.stopped = Stopped()
        else:
            self.stopped = False

        self.skip_load = skip_load
        self.utterances_time_sorted = False
        self.parse_text_only_files = parse_text_only_files
        self.feature_config = FeatureConfig()
        self.vad_config = {'energy_threshold': 5.5,
                      'energy_mean_scale': 0.5}

    @property
    def file_speaker_mapping(self) -> Dict[str, List[str]]:
        return {file_name: file.speaker_ordering for file_name, file in self.files.items()}


    def _load_from_temp(self) -> bool:
        begin_time = time.time()
        for f in os.listdir(self.output_directory):
            if f.startswith('split'):
                old_num_jobs = int(f.replace('split', ''))
                if old_num_jobs != self.num_jobs:
                    self.logger.info(f'Found old run with {old_num_jobs} rather than the current {self.num_jobs}, '
                                     f'setting to {old_num_jobs}.  If you would like to use {self.num_jobs}, re-run the command '
                                     f'with --clean.')
                    self.num_jobs = old_num_jobs
        speakers_path = os.path.join(self.output_directory, 'speakers.yaml')
        files_path = os.path.join(self.output_directory, 'files.yaml')
        utterances_path = os.path.join(self.output_directory, 'utterances.yaml')

        if not os.path.exists(speakers_path):
            self.logger.debug(f'Could not find {speakers_path}, cannot load from temp')
            return False
        if not os.path.exists(files_path):
            self.logger.debug(f'Could not find {files_path}, cannot load from temp')
            return False
        if not os.path.exists(utterances_path):
            self.logger.debug(f'Could not find {utterances_path}, cannot load from temp')
            return False
        self.logger.debug('Loading from temporary files...')

        with open(speakers_path, 'r', encoding='utf8') as f:
            speaker_data = yaml.safe_load(f)

        for entry in speaker_data:
            self.speakers[entry['name']] = Speaker(entry['name'])
            self.speakers[entry['name']].cmvn = entry['cmvn']

        with open(files_path, 'r', encoding='utf8') as f:
            files_data = yaml.safe_load(f)
        for entry in files_data:
            self.files[entry['name']] = File(entry['wav_path'],  entry['text_path'], entry['relative_path'])
            self.files[entry['name']].speaker_ordering = [self.speakers[x] for x in entry['speaker_ordering']]
            self.files[entry['name']].wav_info = entry['wav_info']

        with open(utterances_path, 'r', encoding='utf8') as f:
            utterances_data = yaml.safe_load(f)
        for entry in utterances_data:
            s = self.speakers[entry['speaker']]
            f = self.files[entry['file']]
            u = Utterance(s, f, begin=entry['begin'], end=entry['end'],
                                                       channel=entry['channel'], text=entry['text'])
            self.utterances[u.name] = u
            if u.text:
                self.word_counts.update(u.text.split())
            self.utterances[u.name].features = entry['features']
            self.utterances[u.name].ignored = entry['ignored']

        self.logger.debug(f'Loaded from corpus_data temp directory in {time.time()-begin_time} seconds')
        return True


    def _load_from_source_mp(self) -> None:
        if self.stopped is None:
            self.stopped = Stopped()
        begin_time = time.time()
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        return_dict['sound_file_errors'] = manager.list()
        return_dict['decode_error_files'] = manager.list()
        return_dict['textgrid_read_errors'] = manager.dict()
        finished_adding = Stopped()
        procs = []
        for i in range(self.num_jobs):
            p = CorpusProcessWorker(job_queue, return_dict, return_queue, self.stopped,
                                    finished_adding)
            procs.append(p)
            p.start()
        try:

            use_audio_directory = False
            all_sound_files = {}
            if self.audio_directory and os.path.exists(self.audio_directory):
                use_audio_directory = True
                for root, dirs, files in os.walk(self.audio_directory, followlinks=True):
                    identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                    wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                    other_audio_files = {k: os.path.join(root, v) for k, v in other_audio_files.items()}
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)

            for root, dirs, files in os.walk(self.directory, followlinks=True):
                identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')

                if self.stopped.stop_check():
                    break
                if not use_audio_directory:
                    all_sound_files = {}
                    wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                    other_audio_files = {k: os.path.join(root, v) for k, v in other_audio_files.items()}
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)
                for file_name in identifiers:
                    if self.stopped.stop_check():
                        break
                    wav_path = None
                    transcription_path = None
                    if file_name in all_sound_files:
                        wav_path = all_sound_files[file_name]
                    if file_name in lab_files:
                        lab_name = lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)

                    elif file_name in textgrid_files:
                        tg_name = textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    if wav_path is None and not self.parse_text_only_files:
                        self.transcriptions_without_wavs.append(transcription_path)
                        continue
                    if transcription_path is None:
                        self.no_transcription_files.append(wav_path)
                    job_queue.put((file_name, wav_path, transcription_path, relative_path, self.speaker_characters,
                                   self.sample_rate, self.punctuation, self.clitic_markers))

            finished_adding.stop()
            self.logger.debug('Finished adding jobs!')
            job_queue.join()

            self.logger.debug('Waiting for workers to finish...')
            for p in procs:
                p.join()

            while True:
                try:
                    file = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    break

                self.add_file(file)

            if 'error' in return_dict:
                raise return_dict['error'][1]

            for k in ['sound_file_errors',
                      'decode_error_files', 'textgrid_read_errors']:
                if hasattr(self, k):
                    if return_dict[k]:
                        self.logger.info('There were some issues with files in the corpus. '
                                         'Please look at the log file or run the validator for more information.')
                        self.logger.debug(f'{k} showed {len(return_dict[k])} errors:')
                        if k == 'textgrid_read_errors':
                            getattr(self, k).update(return_dict[k])
                            for f, e in return_dict[k].items():
                                self.logger.debug(f'{f}: {e.error}')
                        else:
                            self.logger.debug(', '.join(return_dict[k]))
                            setattr(self, k, return_dict[k])

        except KeyboardInterrupt:
            self.logger.info('Detected ctrl-c, please wait a moment while we clean everything up...')
            self.stopped.stop()
            finished_adding.stop()
            job_queue.join()
            self.stopped.set_sigint_source()
            while True:
                try:
                    _ = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    break
        finally:

            if self.stopped.stop_check():
                self.logger.info(f'Stopped parsing early ({time.time() - begin_time} seconds)')
                if self.stopped.source():
                    sys.exit(0)
            else:
                self.logger.debug(
                    f'Parsed corpus directory with {self.num_jobs} jobs in {time.time() - begin_time} seconds')


    def _load_from_source(self) -> None:
        begin_time = time.time()
        self.stopped = False

        all_sound_files = {}
        use_audio_directory = False
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, dirs, files in os.walk(self.audio_directory, followlinks=True):
                if self.stopped:
                    return
                identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                other_audio_files = {k: os.path.join(root, v) for k, v in other_audio_files.items()}
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)

        for root, dirs, files in os.walk(self.directory, followlinks=True):
            identifiers, wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
            relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')
            if self.stopped:
                return
            if not use_audio_directory:
                all_sound_files = {}
                wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                other_audio_files = {k: os.path.join(root, v) for k, v in other_audio_files.items()}
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)
            for file_name in identifiers:

                wav_path = None
                transcription_path = None
                if file_name in all_sound_files:
                    wav_path = all_sound_files[file_name]
                if file_name in lab_files:
                    lab_name = lab_files[file_name]
                    transcription_path = os.path.join(root, lab_name)
                elif file_name in textgrid_files:
                    tg_name = textgrid_files[file_name]
                    transcription_path = os.path.join(root, tg_name)
                if wav_path is None and not self.parse_text_only_files:
                    self.transcriptions_without_wavs.append(transcription_path)
                    continue
                if transcription_path is None:
                    self.no_transcription_files.append(wav_path)

                try:
                    file = parse_file(file_name, wav_path, transcription_path, relative_path, self.speaker_characters,
                                        self.sample_rate, self.punctuation, self.clitic_markers)
                    self.add_file(file)
                except TextParseError  as e:
                    self.decode_error_files.append(e)
                except TextGridParseError as e:
                    self.textgrid_read_errors[e.file_name] = e
        if self.decode_error_files or self.textgrid_read_errors:
            self.logger.info('There were some issues with files in the corpus. '
                             'Please look at the log file or run the validator for more information.')
            if self.decode_error_files:
                self.logger.debug(f'There were {len(self.decode_error_files)} errors decoding text files:')
                self.logger.debug(', '.join(self.decode_error_files))
            if self.textgrid_read_errors:
                self.logger.debug(f'There were {len(self.textgrid_read_errors)} errors decoding reading TextGrid files:')
                for f, e in self.textgrid_read_errors.items():
                    self.logger.debug(f'{f}: {e.error}')


        self.logger.debug(f'Parsed corpus directory in {time.time()-begin_time} seconds')

    def add_file(self, file: File):
        self.files[file.name] = file
        for speaker in file.speaker_ordering:
            if speaker.name not in self.speakers:
                self.speakers[speaker.name] = speaker
            else:
                self.speakers[speaker.name].merge(speaker)
        for u in file.utterances.values():
            self.utterances[u.name] = u
            if u.text:
                self.word_counts.update(u.text.split())

    def get_word_frequency(self, dictionary: DictionaryType) -> Dict[str, float]:
        word_counts = Counter()
        for u in self.utterances.values():
            text = u.text
            speaker = u.speaker
            d = dictionary.get_dictionary(speaker)
            new_text = []
            text = text.split()
            for t in text:

                lookup = d.split_clitics(t)
                if lookup is None:
                    continue
                new_text.extend(x for x in lookup if x != '')
            word_counts.update(new_text)
        return {k: v / sum(word_counts.values()) for k, v in word_counts.items()}

    @property
    def word_set(self) -> List[str]:
        return sorted(self.word_counts)

    def add_utterance(self, utterance: Utterance) -> None:
        self.utterances[utterance.name] = utterance
        if utterance.speaker.name not in self.speakers:
            self.speakers[utterance.speaker.name] = utterance.speaker
        if utterance.file.name not in self.files:
            self.files[utterance.file.name] = utterance.file

    def delete_utterance(self, utterance: Union[str, Utterance]) -> None:
        if isinstance(utterance, str):
            utterance = self.utterances[utterance]
        utterance.speaker.delete_utterance(utterance)
        utterance.file.delete_utterance(utterance)
        del self.utterances[utterance.name]


    def initialize_jobs(self) -> None:
        if len(self.speakers) < self.num_jobs:
            self.num_jobs = len(self.speakers)
        self.jobs = [Job(i) for i in range(self.num_jobs)]
        job_ind = 0
        for s in self.speakers.values():
            self.jobs[job_ind].add_speaker(s)
            job_ind += 1
            if job_ind == self.num_jobs:
                job_ind = 0

    @property
    def num_utterances(self) -> int:
        return len(self.utterances)

    @property
    def features_directory(self) -> str:
        return os.path.join(self.output_directory, 'features')

    @property
    def features_log_directory(self) -> str:
        return os.path.join(self.split_directory, 'log')

    def speaker_utterance_info(self) -> str:
        num_speakers = len(self.speakers)
        if not num_speakers:
            raise CorpusError('There were no sound files found of the appropriate format. Please double check the corpus path '
                              'and/or run the validation utility (mfa validate).')
        average_utterances = sum(len(x.utterances) for x in self.speakers.values()) / num_speakers
        msg = f'Number of speakers in corpus: {num_speakers}, ' \
              f'average number of utterances per speaker: {average_utterances}'
        return msg

    def get_wav_duration(self, utt: str) -> float:
        return self.utterances[utt].file.duration

    @property
    def file_durations(self) -> Dict[str, float]:
        return {f: file.duration for f, file in self.files.items()}

    @property
    def split_directory(self) -> str:
        directory = os.path.join(self.output_directory, f'split{self.num_jobs}')
        return directory

    def generate_features(self, overwrite: bool=False, compute_cmvn: bool=True) -> None:
        if not overwrite and os.path.exists(os.path.join(self.output_directory, 'feats.scp')):
            return
        self.logger.info(f'Generating base features ({self.feature_config.type})...')
        if self.feature_config.type == 'mfcc':
            mfcc(self)
        self.combine_feats()
        if compute_cmvn:
            self.logger.info('Calculating CMVN...')
            calc_cmvn(self)
        self.write()
        self.split()

    def compute_vad(self, vad_config=None) -> None:
        if os.path.exists(os.path.join(self.split_directory, 'vad.0.scp')):
            self.logger.info('VAD already computed, skipping!')
            return
        self.logger.info('Computing VAD...')
        compute_vad(self)

    def combine_feats(self) -> None:
        split_directory = self.split_directory
        ignore_check = []
        for job in self.jobs:
            feats_paths  = job.construct_path_dictionary(split_directory, 'feats', 'scp')
            lengths_paths  = job.construct_path_dictionary(split_directory, 'utterance_lengths', 'scp')
            for dict_name in job.current_dictionary_names:
                path = feats_paths[dict_name]
                lengths_path = lengths_paths[dict_name]
                if os.path.exists(lengths_path):
                    with open(lengths_path, 'r') as inf:
                        for line in inf:
                            line = line.strip()
                            utt, length = line.split()
                            length = int(length)
                            if length < 13:  # Minimum length to align one phone plus silence
                                self.utterances[utt].ignored = True
                                ignore_check.append(utt)
                            self.utterances[utt].feature_length = length
                with open(path, 'r') as inf:
                    for line in inf:
                        line = line.strip()
                        if line == '':
                            continue
                        f = line.split(maxsplit=1)
                        if self.utterances[f[0]].ignored:
                            continue
                        self.utterances[f[0]].features = f[1]
        for u, utterance in self.utterances.items():
            if utterance.features is None:
                utterance.ignored = True
                ignore_check.append(u)
        if ignore_check:
            self.logger.warning('There were some utterances ignored due to short duration, see the log file for full '
                                'details or run `mfa validate` on the corpus.')
            self.logger.debug(f"The following utterances were too short to run alignment: "
                              f"{' ,'.join(ignore_check)}")
        self.write()

    def get_feat_dim(self) -> int:
        feature_string = self.jobs[0].construct_base_feature_string(self)
        with open(os.path.join(self.features_log_directory, 'feat-to-dim.log'), 'w') as log_file:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         feature_string, '-'],
                                        stdout=subprocess.PIPE,
                                        stderr=log_file
                                        )
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode('utf8').strip()
        return int(feats)

    def write(self) -> None:
        self._write_speakers()
        self._write_files()
        self._write_utterances()
        self._write_spk2utt()
        self._write_feats()

    def _write_spk2utt(self):
        data  = {speaker.name: sorted(speaker.utterances.keys()) for speaker in self.speakers.values()}
        output_mapping(data, os.path.join(self.output_directory, 'spk2utt.scp'))

    def write_utt2spk(self):
        data  = {u.name: u.speaker.name for u in self.utterances.values()}
        output_mapping(data, os.path.join(self.output_directory, 'utt2spk.scp'))

    def _write_feats(self):
        if any(x.features is not None for x in self.utterances.values()):
            with open(os.path.join(self.output_directory, 'feats.scp'), 'w', encoding='utf8') as f:
                for utterance in self.utterances.values():
                    if not utterance.features:
                        continue
                    f.write(f'{utterance.name} {utterance.features}\n')

    def _write_speakers(self):
        to_save = []
        for speaker in self.speakers.values():
            to_save.append(speaker.meta)
        with open(os.path.join(self.output_directory, 'speakers.yaml'), 'w', encoding='utf8') as f:
            yaml.safe_dump(to_save, f)

    def _write_files(self):
        to_save = []
        for file in self.files.values():
            to_save.append(file.meta)
        with open(os.path.join(self.output_directory, 'files.yaml'), 'w', encoding='utf8') as f:
            yaml.safe_dump(to_save, f)

    def _write_utterances(self):
        to_save = []
        for utterance in self.utterances.values():
            to_save.append(utterance.meta)
        with open(os.path.join(self.output_directory, 'utterances.yaml'), 'w', encoding='utf8') as f:
            yaml.safe_dump(to_save, f)

    def split(self) -> None:
        split_dir = self.split_directory
        os.makedirs(os.path.join(split_dir, 'log'), exist_ok=True)
        self.logger.info('Setting up training data...')
        for job in self.jobs:
            job.output_to_directory(split_dir)
