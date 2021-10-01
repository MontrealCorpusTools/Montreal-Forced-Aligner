import os
import sys
import time
import signal
import re

from .base import BaseCorpus, find_exts
from ..helper import load_scp
from ..dictionary import MultispeakerDictionary

from ..exceptions import CorpusError
from ..multiprocessing import segment_vad
import multiprocessing as mp
from queue import Empty
from ..multiprocessing.helper import Stopped
from ..multiprocessing.corpus import CorpusProcessWorker, parse_textgrid_file, parse_wav_file


class TranscribeCorpus(BaseCorpus):
    def __init__(self, directory, output_directory,
                 speaker_characters=0,
                 num_jobs=3, sample_rate=16000, debug=False, logger=None, use_mp=True, no_speakers=False,
                 ignore_transcriptions=False, audio_directory=None, skip_load=False):
        super(TranscribeCorpus, self).__init__(directory, output_directory,
                                               speaker_characters,
                                               num_jobs, sample_rate, debug, logger, use_mp,
                                               audio_directory=audio_directory, skip_load=skip_load)
        self.no_speakers = no_speakers
        self.ignore_transcriptions = ignore_transcriptions
        self.vad_segments = {}
        if self.use_mp:
            self.stopped = Stopped()
        else:
            self.stopped = False
        if not self.skip_load:
            self.load()

    def load(self):
        loaded = self._load_from_temp()
        if not loaded:
            if self.use_mp:
                self.logger.debug('Loading from source with multiprocessing')
                self._load_from_source_mp()
            else:
                self.logger.debug('Loading from source without multiprocessing')
                self._load_from_source()
        else:
            self.logger.debug('Successfully loaded from temporary files')
        self.find_best_groupings()

    def _load_from_temp(self):
        begin_time = time.time()
        for f in os.listdir(self.output_directory):
            if f.startswith('split'):
                old_num_jobs = int(f.replace('split', ''))
                if old_num_jobs != self.num_jobs:
                    self.logger.info(f'Found old run with {old_num_jobs} rather than the current {self.num_jobs}, '
                                     f'setting to {old_num_jobs}.  If you would like to use {self.num_jobs}, re-run the command '
                                     f'with --clean.')
                    self.num_jobs = old_num_jobs
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if not os.path.exists(feat_path):
            return False
        cmvn_path = os.path.join(self.output_directory, 'cmvn.scp')
        if not os.path.exists(cmvn_path):
            return False
        utt2spk_path = os.path.join(self.output_directory, 'utt2spk')
        if not os.path.exists(utt2spk_path):
            return False
        spk2utt_path = os.path.join(self.output_directory, 'spk2utt')
        if not os.path.exists(spk2utt_path):
            return False
        wav_path = os.path.join(self.output_directory, 'wav.scp')
        if not os.path.exists(wav_path):
            return False
        sox_strings_path = os.path.join(self.output_directory, 'sox_strings.scp')
        if not os.path.exists(sox_strings_path):
            return False
        file_directory_path = os.path.join(self.output_directory, 'file_directory.scp')
        if not os.path.exists(file_directory_path):
            return False
        wav_info_path = os.path.join(self.output_directory, 'wav_info.scp')
        if not os.path.exists(wav_info_path):
            return False
        self.feat_mapping = load_scp(feat_path)
        self.cmvn_mapping = load_scp(cmvn_path)
        self.utt_speak_mapping = load_scp(utt2spk_path)
        self.speak_utt_mapping = load_scp(spk2utt_path)
        self.utt_wav_mapping = load_scp(wav_path)
        self.sox_strings = load_scp(sox_strings_path)
        self.wav_info = load_scp(wav_info_path, float)
        self.file_directory_mapping = load_scp(file_directory_path)
        segments_path = os.path.join(self.output_directory, 'segments.scp')
        if os.path.exists(segments_path):
            self.segments = load_scp(segments_path)
            for k, v in self.segments.items():
                self.segments[k] = {'file_name': v[0], 'begin': round(float(v[1]), 4),
                                    'end': round(float(v[2]), 4), 'channel': int(v[3])}
        speaker_ordering_path = os.path.join(self.output_directory, 'speaker_ordering.scp')
        if os.path.exists(speaker_ordering_path):
            self.speaker_ordering = load_scp(speaker_ordering_path)
        text_path = os.path.join(self.output_directory, 'text')
        if os.path.exists(text_path):
            self.text_mapping = load_scp(text_path)
            for utt, text in self.text_mapping.items():
                self.text_mapping[utt] = ' '.join(text)
        self.logger.debug('Loaded from corpus_data temp directory in {} seconds'.format(time.time() - begin_time))
        return True

    def _load_from_source_mp(self):
        if self.stopped is None:
            self.stopped = Stopped()
        begin_time = time.time()
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        finished_adding = Stopped()

        all_sound_files = {}
        use_audio_directory = False
        processed = set()
        try:
            if self.audio_directory and os.path.exists(self.audio_directory):
                use_audio_directory = True
                for root, dirs, files in os.walk(self.audio_directory, followlinks=True):
                    wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                    wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                    other_audio_files = {k: os.path.join(root, v) for k, v in other_audio_files.items()}
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)

            procs = []
            for i in range(self.num_jobs):
                p = CorpusProcessWorker(job_queue, return_dict, return_queue, self.stopped,
                                        finished_adding)
                procs.append(p)
                p.start()

            for root, dirs, files in os.walk(self.directory, followlinks=True):
                wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')

                if not use_audio_directory:
                    all_sound_files = {}
                    all_sound_files.update(other_audio_files)
                    all_sound_files.update(wav_files)
                    for file_name, f in all_sound_files.items():
                        if self.stopped.stop_check():
                            break
                        wav_path = os.path.join(root, f)
                        transcription_path = None
                        if file_name in textgrid_files and not self.ignore_transcriptions:
                            tg_name = textgrid_files[file_name]
                            transcription_path = os.path.join(root, tg_name)
                        job_queue.put((file_name, wav_path, transcription_path, relative_path, self.speaker_characters, self.sample_rate))
                else:
                    for file_name, f in textgrid_files.items():
                        if file_name in all_sound_files:
                            wav_path = all_sound_files[file_name]
                        else:
                            continue
                        tg_name = textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                        job_queue.put((file_name, wav_path, transcription_path, relative_path, self.speaker_characters,
                                       self.sample_rate))
            if use_audio_directory:
                for file_name, f in all_sound_files.items():
                    if self.stopped.stop_check():
                        break
                    if f in processed:
                        continue
                    relative_path = os.path.dirname(f).replace(self.directory, '').lstrip('/').lstrip('\\')
                    wav_path = f
                    job_queue.put((file_name, wav_path, None, relative_path, self.speaker_characters,
                                   self.sample_rate))

            finished_adding.stop()
            self.logger.debug('Finished adding jobs!')
            job_queue.join()

            self.logger.debug('Waiting for workers to finish...')
            for p in procs:
                p.join()

            if 'error' in return_dict:
                raise return_dict['error'][1]

            self.logger.debug('Beginning processing of results queue...')

            while True:
                try:
                    info = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    break
                if 'segments' not in info:  # didn't have a textgrid file
                    self._parse_wav_file_info(info)
                else:
                    self._parse_textgrid_info(info)


            for k in ['wav_read_errors', 'unsupported_bit_depths',
                      'decode_error_files', 'textgrid_read_errors']:
                if hasattr(self, k):
                    if k in return_dict:
                        if k == 'textgrid_read_errors':
                            getattr(self, k).update(return_dict[k])
                        else:
                            setattr(self, k, return_dict[k])
        except KeyboardInterrupt:
            self.logger.info('Detected ctrl-c, please wait a moment while we clean everything up...')
            self.stopped.stop()
            finished_adding.stop()
            job_queue.join()
            self.stopped.set_sigint_source()
            while True:
                try:
                    info = return_queue.get(timeout=1)
                    if self.stopped.stop_check():
                        continue
                except Empty:
                    break
        finally:
            if self.stopped.stop_check():
                self.logger.debug('Stopped parsing early ({} seconds)'.format(time.time()-begin_time))
                if self.stopped.source():
                    sys.exit(0)
            else:
                self.logger.debug(
                'Parsed corpus directory with {} jobs in {} seconds'.format(self.num_jobs, time.time() - begin_time))

    def _parse_wav_file_info(self, info):
        utt_name = info['utt_name']
        speaker_name = info['speaker_name']
        wav_info = info['wav_info']
        if utt_name in self.utt_wav_mapping:
            ind = 0
            fixed_utt_name = utt_name
            while fixed_utt_name not in self.utt_wav_mapping:
                ind += 1
                fixed_utt_name = utt_name + '_{}'.format(ind)
            utt_name = fixed_utt_name
        file_name = info['file_name']

        self.wav_files.append(file_name)
        if self.no_speakers:
            self.utt_speak_mapping[utt_name] = utt_name
            self.speak_utt_mapping[utt_name] = [utt_name]

        else:
            self.speak_utt_mapping[speaker_name].append(utt_name)
            self.utt_speak_mapping[utt_name] = speaker_name
        if 'sox_string' in info:
            self.sox_strings[utt_name] = info['sox_string']
        self.utt_wav_mapping[utt_name] = info['wav_path']
        self.file_directory_mapping[utt_name] = info['relative_path']
        self.file_name_mapping[utt_name] = info['file_name']
        self.wav_info[file_name] = [wav_info['num_channels'], wav_info['sample_rate'], wav_info['duration']]

    def _parse_textgrid_info(self, info):
        wav_info = info['wav_info']
        file_name = info['recording_name']
        self.wav_files.append(file_name)
        self.speaker_ordering[file_name] = info['speaker_ordering']
        self.segments.update(info['segments'])
        self.utt_wav_mapping.update(info['utt_wav_mapping'])
        if 'sox_string' in info:
            self.sox_strings.update(info['sox_string'])
        for utt, words in info['text_mapping'].items():
            self.text_mapping[utt] = words
        if self.no_speakers:
            for utt in info['utt_speak_mapping']:
                self.utt_speak_mapping[utt] = utt
                self.speak_utt_mapping[utt] = [utt]
        else:
            self.utt_speak_mapping.update(info['utt_speak_mapping'])
            for speak, utts in info['speak_utt_mapping'].items():
                if speak not in self.speak_utt_mapping:
                    self.speak_utt_mapping[speak] = utts
                else:
                    self.speak_utt_mapping[speak].extend(utts)
        for fn in info['file_names']:
            self.file_directory_mapping[fn] = info['relative_path']
        self.wav_info[file_name] = [wav_info['num_channels'], wav_info['sample_rate'], wav_info['duration']]

    def _load_from_source(self):
        all_sound_files = {}
        use_audio_directory = False
        processed = set()
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, dirs, files in os.walk(self.directory, followlinks=True):
                wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                wav_files = {k: os.path.join(root, v) for k, v in wav_files.items()}
                other_audio_files = {k: os.path.join(root, v) for k, v in other_audio_files.items()}
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)

        for root, dirs, files in os.walk(self.directory, followlinks=True):
            wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
            relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')

            if not use_audio_directory:
                all_sound_files = {}
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)
                for file_name, f in all_sound_files.items():
                    wav_path = os.path.join(root, f)

                    if file_name in textgrid_files and not self.ignore_transcriptions:
                        tg_name = textgrid_files[file_name]
                        tg_path = os.path.join(root, tg_name)
                        info = parse_textgrid_file(file_name, wav_path, tg_path, relative_path, self.speaker_characters, self.sample_rate)
                        self._parse_textgrid_info(info)
                    else:
                        info = parse_wav_file(file_name, wav_path, None, relative_path, self.speaker_characters, self.sample_rate)
                        self._parse_wav_file_info(info)
            else:
                for file_name, f in textgrid_files.items():
                    if file_name in all_sound_files:
                        wav_path = all_sound_files[file_name]
                    else:
                        continue
                    tg_name = textgrid_files[file_name]
                    transcription_path = os.path.join(root, tg_name)
                    info = parse_textgrid_file(file_name, wav_path, transcription_path, relative_path, self.speaker_characters, self.sample_rate)
                    self._parse_textgrid_info(info)
                    processed.add(wav_path)
        if use_audio_directory:
            for file_name, f in all_sound_files.items():
                if f in processed:
                    continue
                relative_path = os.path.dirname(f).replace(self.directory, '').lstrip('/').lstrip('\\')
                wav_path = f
                info = parse_wav_file(file_name, wav_path, None, relative_path, self.speaker_characters, self.sample_rate)
                self._parse_wav_file_info(info)


    def initialize_corpus(self, dictionary=None, feature_config=None):
        if not self.utt_wav_mapping:
            raise CorpusError('There were no wav files found for transcribing this corpus. Please validate the corpus.')
        split_dir = self.split_directory()
        self.write()
        if not os.path.exists(split_dir):
            self.split()
        if feature_config is not None:
            feature_config.generate_features(self)
        if isinstance(dictionary, MultispeakerDictionary):
            self.split_by_dictionary(dictionary)
        self.figure_utterance_lengths()

    def create_vad_segments(self, segmentation_config):
        segment_vad(self, segmentation_config)
        directory = self.split_directory()
        self.vad_segments = {}
        for i in range(self.num_jobs):
            vad_segments_path = os.path.join(directory, 'vad_segments.{}.scp'.format(i))
            self.vad_segments.update(load_scp(vad_segments_path))
