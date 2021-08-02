import os
import random
import time
import re
import itertools

from praatio import tgio
from collections import Counter
from ..helper import output_mapping, save_groups, filter_scp, load_scp, log_kaldi_errors

from ..exceptions import CorpusError, WavReadError,  \
    TextParseError, TextGridParseError, KaldiProcessingError

from .base import BaseCorpus, find_exts
import multiprocessing as mp
from queue import Empty
from ..dictionary import MultispeakerDictionary
from ..multiprocessing.helper import Stopped, Counter as MPCounter
from ..multiprocessing.corpus import CorpusProcessWorker, parse_lab_file, parse_textgrid_file, parse_transcription


class AlignableCorpus(BaseCorpus):
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

    def __init__(self, directory, output_directory,
                 speaker_characters=0,
                 num_jobs=3, sample_rate=16000, debug=False, logger=None, use_mp=True,
                 punctuation=None, clitic_markers=None, parse_text_only_files=False, audio_directory=None):
        super(AlignableCorpus, self).__init__(directory, output_directory,
                                              speaker_characters,
                                              num_jobs, sample_rate, debug, logger, use_mp,
                                              punctuation, clitic_markers, audio_directory=audio_directory)
        self.utt_text_file_mapping = {}
        self.word_counts = Counter()
        self.utterance_oovs = {}
        self.no_transcription_files = []
        self.decode_error_files = []
        self.transcriptions_without_wavs = []
        self.tg_count = 0
        self.lab_count = 0
        self.parse_text_only_files = parse_text_only_files

        self.loaded_from_temp = self._load_from_temp()
        if not self.loaded_from_temp:
            if self.use_mp:
                self.logger.debug('Loading from source with multiprocessing')
                self._load_from_source_mp()
            else:
                self.logger.debug('Loading from source without multiprocessing')
                self._load_from_source()
        else:
            self.logger.debug('Successfully loaded from temporary files')
        self.check_warnings()
        self.find_best_groupings()

    def delete_utterance(self, utterance):
        super(AlignableCorpus, self).delete_utterance(utterance)

    def add_utterance(self, utterance, speaker, file, text, wav_file=None, seg=None):
        super(AlignableCorpus, self).add_utterance(utterance, speaker, file, text, wav_file, seg)

    def _load_from_temp(self):
        begin_time = time.time()
        utt2spk_path = os.path.join(self.output_directory, 'utt2spk')
        if not os.path.exists(utt2spk_path):
            return False
        spk2utt_path = os.path.join(self.output_directory, 'spk2utt')
        if not os.path.exists(spk2utt_path):
            return False
        utt_file_path = os.path.join(self.output_directory, 'utt2file')
        if not os.path.exists(utt_file_path):
            return False
        file_utt_path = os.path.join(self.output_directory, 'file2utt')
        if not os.path.exists(file_utt_path):
            return False
        text_path = os.path.join(self.output_directory, 'text')
        if not os.path.exists(text_path):
            return False
        sr_path = os.path.join(self.output_directory, 'sr.scp')
        if not os.path.exists(sr_path):
            return False
        wav_path = os.path.join(self.output_directory, 'wav.scp')
        if not os.path.exists(wav_path):
            return False
        sox_strings_path = os.path.join(self.output_directory, 'sox_strings.scp')
        if not os.path.exists(sox_strings_path):
            return False
        text_file_path = os.path.join(self.output_directory, 'text_file.scp')
        if not os.path.exists(text_file_path):
            return False
        file_directory_path = os.path.join(self.output_directory, 'file_directory.scp')
        if not os.path.exists(file_directory_path):
            return False
        file_name_mapping_path = os.path.join(self.output_directory, 'file_name.scp')
        if not os.path.exists(file_name_mapping_path):
            return False
        wav_info_path = os.path.join(self.output_directory, 'wav_info.scp')
        if not os.path.exists(wav_info_path):
            return False


        self.utt_speak_mapping = load_scp(utt2spk_path)
        self.speak_utt_mapping = load_scp(spk2utt_path)
        for speak, utts in self.speak_utt_mapping.items():
            if not isinstance(utts, list):
                self.speak_utt_mapping[speak] = [utts]
        self.utt_file_mapping = load_scp(utt_file_path)
        self.file_utt_mapping = load_scp(file_utt_path)
        for file, utts in self.file_utt_mapping.items():
            if not isinstance(utts, list):
                self.file_utt_mapping[file] = [utts]
        self.text_mapping = load_scp(text_path)
        for utt, text in self.text_mapping.items():
            if not isinstance(text, list):
                text = [text]
            for w in text:
                new_w = re.split(r"[-']", w)
                self.word_counts.update(new_w + [w])
            self.text_mapping[utt] = ' '.join(text)
        self.utt_wav_mapping = load_scp(wav_path)
        self.sox_strings = load_scp(sox_strings_path)
        self.wav_info = load_scp(wav_info_path, float)
        self.utt_text_file_mapping = load_scp(text_file_path)
        for p in self.utt_text_file_mapping.values():
            if p.lower().endswith('.textgrid'):
                self.tg_count += 1
            else:
                self.lab_count += 1
        self.file_directory_mapping = load_scp(file_directory_path)
        self.file_name_mapping = load_scp(file_name_mapping_path)
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if os.path.exists(feat_path):
            self.feat_mapping = load_scp(feat_path)
        cmvn_path = os.path.join(self.output_directory, 'cmvn.scp')
        if os.path.exists(cmvn_path):
            self.cmvn_mapping = load_scp(cmvn_path)
        segments_path = os.path.join(self.output_directory, 'segments.scp')
        if os.path.exists(segments_path):
            self.segments = load_scp(segments_path)
            for k, v in self.segments.items():
                self.segments[k] = {'file_name': v[0], 'begin': round(float(v[1]), 4),
                                    'end': round(float(v[2]), 4), 'channel': int(v[3])}
        speaker_ordering_path = os.path.join(self.output_directory, 'speaker_ordering.scp')
        if os.path.exists(speaker_ordering_path):
            self.speaker_ordering = load_scp(speaker_ordering_path)
            for file, speakers in self.speaker_ordering.items():
                if not isinstance(speakers, list):
                    self.speaker_ordering[file] = [speakers]
        self.logger.debug('Loaded from corpus_data temp directory in {} seconds'.format(time.time()-begin_time))
        return True

    def _load_from_source_mp(self):
        begin_time = time.time()
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        return_dict['wav_read_errors'] = manager.list()
        return_dict['unsupported_sample_rate'] = manager.list()
        return_dict['decode_error_files'] = manager.list()
        return_dict['textgrid_read_errors'] = manager.dict()
        stopped = Stopped()
        processed_counter = MPCounter()
        max_counter = MPCounter()

        text_only_files = []
        use_audio_directory = False
        all_sound_files = {}
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, dirs, files in os.walk(self.directory, followlinks=True):
                wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)

        procs = []
        for i in range(self.num_jobs):
            p = CorpusProcessWorker(job_queue, return_dict, return_queue, stopped,
                                    processed_counter, max_counter)
            procs.append(p)
            p.start()
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
            relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')
            if stopped.stop_check():
                break
            if not use_audio_directory:
                all_sound_files = {}
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)
                for file_name, f in all_sound_files.items():
                    if stopped.stop_check():
                        break
                    wav_path = os.path.join(root, f)
                    if file_name in lab_files:
                        lab_name = lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)

                    elif file_name in textgrid_files:
                        tg_name = textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    else:
                        self.no_transcription_files.append(wav_path)
                        continue
                    max_counter.increment()
                    job_queue.put((file_name, wav_path, transcription_path, relative_path, self.speaker_characters,
                                   self.sample_rate, self.punctuation, self.clitic_markers))
            else:
                for file_name, f in itertools.chain(lab_files.items(), textgrid_files.items()):
                    if stopped.stop_check():
                        break
                    if file_name in all_sound_files:
                        wav_path = all_sound_files[file_name]
                    else:
                        continue
                    if file_name in lab_files:
                        lab_name = lab_files[file_name]
                        transcription_path = os.path.join(root, lab_name)

                    elif file_name in textgrid_files:
                        tg_name = textgrid_files[file_name]
                        transcription_path = os.path.join(root, tg_name)
                    else:
                        self.no_transcription_files.append(wav_path)
                        continue
                    job_queue.put((file_name, wav_path, transcription_path, relative_path, self.speaker_characters,
                                   self.sample_rate, self.punctuation, self.clitic_markers))

            if self.parse_text_only_files:
                for file_name, f in itertools.chain(lab_files.items(), textgrid_files.items()):
                    if file_name in wav_files:
                        continue
                    text_path = os.path.join(root, f)
                    text_only_files.append((file_name, text_path))
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
            except Empty:
                break
            if 'segments' not in info:  # was a lab file
                utt_name = info['utt_name']
                speaker_name = info['speaker_name']
                wav_info = info['wav_info']
                if utt_name in self.utt_wav_mapping:
                    ind = 0
                    fixed_utt_name = utt_name
                    while fixed_utt_name not in self.utt_wav_mapping:
                        ind += 1
                        fixed_utt_name = utt_name + '-{}'.format(ind)
                    utt_name = fixed_utt_name
                file_name = utt_name
                words = info['words']
                words = words.split()
                for w in words:
                    new_w = re.split(r"[-']", w)
                    self.word_counts.update(new_w + [w])
                self.wav_files.append(file_name)
                self.text_mapping[utt_name] = ' '.join(words)
                self.utt_text_file_mapping[utt_name] = info['text_file']
                self.speak_utt_mapping[speaker_name].append(utt_name)
                self.utt_wav_mapping[utt_name] = info['wav_path']
                if 'sox_string' in info:
                    self.sox_strings[utt_name] = info['sox_string']
                self.utt_speak_mapping[utt_name] = speaker_name
                self.file_directory_mapping[utt_name] = info['relative_path']
                self.file_name_mapping[utt_name] = info['file_name']
                self.lab_count += 1
            else:
                wav_info = info['wav_info']
                file_name = info['recording_name']
                self.wav_files.append(file_name)
                self.speaker_ordering[file_name] = info['speaker_ordering']
                self.segments.update(info['segments'])
                self.utt_wav_mapping.update(info['utt_wav_mapping'])
                if 'sox_string' in info:
                    self.sox_strings.update(info['sox_string'])
                self.file_utt_mapping.update(info['file_utt_mapping'])
                self.utt_file_mapping.update(info['utt_file_mapping'])
                self.utt_text_file_mapping.update(info['utt_text_file_mapping'])
                for utt, words in info['text_mapping'].items():
                    words = words.split()
                    for w in words:
                        new_w = re.split(r"[-']", w)
                        self.word_counts.update(new_w + [w])
                    self.text_mapping[utt] = ' '.join(words)
                self.utt_speak_mapping.update(info['utt_speak_mapping'])
                for speak, utts in info['speak_utt_mapping'].items():
                    if speak not in self.speak_utt_mapping:
                        self.speak_utt_mapping[speak] = utts
                    else:
                        self.speak_utt_mapping[speak].extend(utts)
                for fn in info['file_names']:
                    self.file_directory_mapping[fn] = info['relative_path']
                self.tg_count += 1
            self.wav_info[file_name] = [wav_info['num_channels'], wav_info['sample_rate'], wav_info['duration']]
        for k in ['wav_read_errors',
                  'decode_error_files', 'textgrid_read_errors']:
            if hasattr(self, k):
                if return_dict[k]:
                    self.logger.info('There were some issues with files in the corpus. '
                                     'Please look at the log file or run the validator for more information.')
                    self.logger.debug('{} showed {} errors:'.format(k, len(return_dict[k])))
                    if k == 'textgrid_read_errors':
                        getattr(self, k).update(return_dict[k])
                        for f, e in return_dict[k].items():
                            self.logger.debug(f + ': ' + e.error)
                    else:
                        self.logger.debug(', '.join(return_dict[k]))
                        setattr(self, k, return_dict[k])
        if self.parse_text_only_files:
            for file_name, path in text_only_files:
                if path.lower().endswith('.textgrid'):
                    tg = tgio.openTextgrid(path)
                    for i, tier_name in enumerate(tg.tierNameList):
                        ti = tg.tierDict[tier_name]
                        if not isinstance(ti, tgio.IntervalTier):
                            continue
                        speaker_name = tier_name.strip().replace(' ', '_')
                        for begin, end, text in ti.entryList:
                            text = text.lower().strip()
                            words = parse_transcription(text, self.punctuation, self.clitic_markers)

                            utt_name = '{}_{}_{}_{}'.format(speaker_name, file_name, begin, end)
                            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
                            utt_name = utt_name.replace('_', '-')
                            if not words:
                                continue
                            for w in words:
                                new_w = re.split(r"[-']", w)
                                self.word_counts.update(new_w + [w])
                            self.text_mapping[utt_name] = ' '.join(words)
                else:
                    text_ind = 0
                    with open(path, 'r', encoding='utf8') as f:
                        for line in f:
                            line = line.strip().lower()
                            if not line:
                                continue
                            utt_name = '{}_{}'.format(file_name, text_ind)
                            words = parse_transcription(line, self.punctuation, self.clitic_markers)
                            for w in words:
                                new_w = re.split(r"[-']", w)
                                self.word_counts.update(new_w + [w])
                            self.text_mapping[utt_name] = ' '.join(words)

                            text_ind += 1
        self.logger.debug('Parsed corpus directory with {} jobs in {} seconds'.format(self.num_jobs, time.time()-begin_time))

    def _load_from_source(self):
        begin_time = time.time()
        text_only_files = []

        all_sound_files = {}
        if self.audio_directory and os.path.exists(self.audio_directory):
            for root, dirs, files in os.walk(self.directory, followlinks=True):
                wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)

        for root, dirs, files in os.walk(self.directory, followlinks=True):
            wav_files, lab_files, textgrid_files, other_audio_files = find_exts(files)
            all_sound_files.update(other_audio_files)
            all_sound_files.update(wav_files)
            relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')
            if self.parse_text_only_files:
                for filename, f in itertools.chain(lab_files.items(), textgrid_files.items()):
                    if filename in wav_files:
                        continue
                    text_path = os.path.join(root, f)
                    text_only_files.append((filename, text_path))

            for file_name, f in all_sound_files.items():
                wav_path = os.path.join(root, f)
                if file_name in lab_files:
                    lab_name = lab_files[file_name]
                    lab_path = os.path.join(root, lab_name)
                    try:
                        info = parse_lab_file(file_name, wav_path, lab_path, relative_path,
                                              self.speaker_characters, self.sample_rate, self.punctuation, self.clitic_markers)
                        utt_name = info['utt_name']
                        speaker_name = info['speaker_name']
                        wav_info = info['wav_info']
                        if utt_name in self.utt_wav_mapping:
                            ind = 0
                            fixed_utt_name = utt_name
                            while fixed_utt_name not in self.utt_wav_mapping:
                                ind += 1
                                fixed_utt_name = utt_name + '-{}'.format(ind)
                            utt_name = fixed_utt_name

                        words = info['words']
                        words = words.split()
                        for w in words:
                            new_w = re.split(r"[-']", w)
                            self.word_counts.update(new_w + [w])
                        self.text_mapping[utt_name] = ' '.join(words)
                        self.utt_text_file_mapping[utt_name] = lab_path
                        self.speak_utt_mapping[speaker_name].append(utt_name)
                        self.utt_wav_mapping[utt_name] = wav_path
                        if 'sox_string' in info:
                            self.sox_strings[utt_name] = info['sox_string']
                        self.utt_speak_mapping[utt_name] = speaker_name
                        self.file_directory_mapping[utt_name] = relative_path
                        self.file_name_mapping[utt_name] = info['file_name']
                        self.wav_info[file_name] = [wav_info['num_channels'],
                                                    wav_info['sample_rate'],
                                                    wav_info['duration']]
                        self.lab_count += 1
                    except WavReadError:
                        self.wav_read_errors.append(wav_path)
                    except TextParseError:
                        self.decode_error_files.append(lab_path)

                elif file_name in textgrid_files:
                    tg_name = textgrid_files[file_name]
                    tg_path = os.path.join(root, tg_name)
                    try:
                        info = parse_textgrid_file(file_name, wav_path, tg_path, relative_path,
                                                   self.speaker_characters, self.sample_rate,
                                                   self.punctuation, self.clitic_markers)
                        wav_info = info['wav_info']
                        self.wav_files.append(file_name)
                        self.speaker_ordering[file_name] = info['speaker_ordering']
                        self.segments.update(info['segments'])
                        self.utt_wav_mapping.update(info['utt_wav_mapping'])
                        if 'sox_strings' in info:
                            self.sox_strings.update(info['sox_strings'])
                        self.utt_text_file_mapping.update(info['utt_text_file_mapping'])
                        for utt, words in info['text_mapping'].items():
                            words = words.split()
                            for w in words:
                                new_w = re.split(r"[-']", w)
                                self.word_counts.update(new_w + [w])
                            self.text_mapping[utt] = ' '.join(words)
                        self.utt_speak_mapping.update(info['utt_speak_mapping'])
                        for speak, utts in info['speak_utt_mapping'].items():
                            if speak not in self.speak_utt_mapping:
                                self.speak_utt_mapping[speak] = utts
                            else:
                                self.speak_utt_mapping[speak].extend(utts)
                        for fn in info['file_names']:
                            self.file_directory_mapping[fn] = relative_path
                        self.wav_info[file_name] = [wav_info['num_channels'],
                                                    wav_info['sample_rate'],
                                                    wav_info['duration']]
                        self.tg_count += 1
                    except WavReadError:
                        self.wav_read_errors.append(wav_path)
                    except TextGridParseError as e:
                        self.textgrid_read_errors[tg_path] = e.error

                else:
                    self.no_transcription_files.append(wav_path)
                    continue
        if self.parse_text_only_files:
            for file_name, path in text_only_files:
                if path.lower().endswith('.textgrid'):
                    tg = tgio.openTextgrid(path)
                    for i, tier_name in enumerate(tg.tierNameList):
                        ti = tg.tierDict[tier_name]
                        if not isinstance(ti, tgio.IntervalTier):
                            continue
                        speaker_name = tier_name.strip().replace(' ', '_')
                        for begin, end, text in ti.entryList:
                            text = text.lower().strip()
                            words = parse_transcription(text, self.punctuation, self.clitic_markers)

                            utt_name = '{}_{}_{}_{}'.format(speaker_name, file_name, begin, end)
                            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
                            utt_name = utt_name.replace('_', '-')
                            if not words:
                                continue
                            for w in words:
                                new_w = re.split(r"[-']", w)
                                self.word_counts.update(new_w + [w])
                            self.text_mapping[utt_name] = ' '.join(words)
                else:
                    text_ind = 0
                    with open(path, 'r', encoding='utf8') as f:
                        for line in f:
                            line = line.strip().lower()
                            if not line:
                                continue
                            utt_name = '{}_{}'.format(file_name, text_ind)
                            words = parse_transcription(line, self.punctuation, self.clitic_markers)
                            for w in words:
                                new_w = re.split(r"[-']", w)
                                self.word_counts.update(new_w + [w])
                            self.text_mapping[utt_name] = ' '.join(words)
                            text_ind += 1

        self.logger.debug('Parsed corpus directory in {} seconds'.format(time.time()-begin_time))

    def check_warnings(self):
        self.issues_check = self.ignored_utterances or self.no_transcription_files or \
                            self.textgrid_read_errors or self.decode_error_files

    def save_text_file(self, file_name):
        if self.segments:
            text_file_path = self.utt_text_file_mapping[file_name]
            tg = tgio.Textgrid()
            tiers = {}

            duration = self.get_wav_duration(file_name)

            for utt in self.file_utt_mapping[file_name]:
                seg = self.segments[utt]
                fn, begin, end = seg['file_name'], seg['begin'], seg['end']
                begin = round(float(begin), 4)
                end = round(float(end), 4)
                text =  self.text_mapping[utt]
                speaker = self.utt_speak_mapping[utt]
                if speaker not in tiers:
                    tiers[speaker] = tgio.IntervalTier(speaker, [], minT=0, maxT=duration)
                if end > duration:
                    end = duration
                tiers[speaker].entryList.append((begin, end, text))

            for v in tiers.values():
                tg.addTier(v)
            tg.save(text_file_path, useShortForm=False)
        else:
            lab_path = self.utt_text_file_mapping[file_name]
            with open(lab_path, 'w', encoding='utf8') as f:
                f.write(self.text_mapping[file_name])

    @property
    def word_set(self):
        return list(self.word_counts)

    def normalized_text_iter(self, dictionary=None, min_count=1):
        unk_words = {k for k, v in self.word_counts.items() if v <= min_count}
        for u, text in self.text_mapping.items():
            text = text.split()
            new_text = []
            for t in text:
                if dictionary is not None:
                    dictionary.to_int(t)
                    lookup = dictionary.split_clitics(t)
                    if lookup is None:
                        continue
                else:
                    lookup = [t]
                for item in lookup:
                    if item in unk_words:
                        new_text.append('<unk>')
                    elif dictionary is not None and item not in dictionary.words:
                        new_text.append('<unk>')
                    else:
                        new_text.append(item)
            yield ' '.join(new_text)

    def grouped_text(self, dictionary=None):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                if u in self.ignored_utterances:
                    continue
                if dictionary is None:
                    try:
                        new_text = self.text_mapping[u]
                    except KeyError:
                        continue
                else:
                    if isinstance(dictionary, MultispeakerDictionary):
                        speaker = self.utt_speak_mapping[u]
                        d = dictionary.get_dictionary(speaker)
                    else:
                        d = dictionary
                    try:
                        text = self.text_mapping[u].split()
                    except KeyError:
                        continue
                    new_text = []
                    for t in text:
                        lookup = d.split_clitics(t)
                        if lookup is None:
                            continue
                        new_text.extend(x for x in lookup if x != '')
                output_g.append([u, new_text])
            output.append(output_g)
        return output

    def grouped_text_int(self, dictionary):
        oov_code = dictionary.oov_int
        self.utterance_oovs = {}
        output = []
        grouped_texts = self.grouped_text(dictionary)
        for g in grouped_texts:
            output_g = []
            for u, text in g:
                if u in self.ignored_utterances:
                    continue
                if isinstance(dictionary, MultispeakerDictionary):
                    speaker = self.utt_speak_mapping[u]
                    d = dictionary.get_dictionary(speaker)
                else:
                    d = dictionary
                oovs = []
                new_text = []
                for i in range(len(text)):
                    t = text[i]
                    lookup = d.to_int(t)
                    for w in lookup:
                        if w == oov_code:
                            oovs.append(text[i])
                        new_text.append(w)
                if oovs:
                    self.utterance_oovs[u] = oovs
                new_text = map(str, (x for x in new_text if isinstance(x, int)))
                output_g.append([u, ' '.join(new_text)])
            output.append(output_g)
        return output

    def get_word_frequency(self, dictionary):
        word_counts = Counter()
        for u, text in self.text_mapping.items():
            if isinstance(dictionary, MultispeakerDictionary):
                speaker = self.utt_speak_mapping[u]
                d = dictionary.get_dictionary(speaker)
            else:
                d = dictionary
            new_text = []
            text = text.split()
            for t in text:

                lookup = d.split_clitics(t)
                if lookup is None:
                    continue
                new_text.extend(x for x in lookup if x != '')
            word_counts.update(new_text)
        return {k: v / sum(word_counts.values()) for k, v in word_counts.items()}

    def grouped_utt2fst(self, dictionary, num_frequent_words=10):
        word_frequencies = self.get_word_frequency(dictionary)
        most_frequent = sorted(word_frequencies.items(), key=lambda x: -x[1])[:num_frequent_words]

        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                try:
                    text = self.text_mapping[u].split()
                except KeyError:
                    continue
                new_text = []
                if isinstance(dictionary, MultispeakerDictionary):
                    speaker = self.utt_speak_mapping[u]
                    d = dictionary.get_dictionary(speaker)
                else:
                    d = dictionary
                for t in text:
                    lookup = d.split_clitics(t)
                    if lookup is None:
                        continue
                    new_text.extend(x for x in lookup if x != '')
                try:
                    fst_text = d.create_utterance_fst(new_text, most_frequent)
                except ZeroDivisionError:
                    print(u, text, new_text)
                    raise
                output_g.append([u, fst_text])
            output.append(output_g)
        return output

    def subset_directory(self, subset, feature_config):
        if subset is None or subset > self.num_utterances or subset <= 0:
            return self.split_directory()
        directory = os.path.join(self.output_directory, 'subset_{}'.format(subset))
        self.create_subset(subset, feature_config)
        return directory

    def write(self):
        super(AlignableCorpus, self).write()
        self._write_text()
        self._write_utt_text_file()

    def _write_text(self):
        path = os.path.join(self.output_directory, 'text')
        output_mapping(self.text_mapping, path)

    def _write_utt_text_file(self):
        path = os.path.join(self.output_directory, 'text_file.scp')
        output_mapping(self.utt_text_file_mapping, path)

    def _split_utt2fst(self, directory, dictionary):
        if dictionary is None:
            return
        pattern = 'utt2fst.{}'
        save_groups(self.grouped_utt2fst(dictionary), directory, pattern, multiline=True)

    def _split_texts(self, directory, dictionary=None):
        pattern = 'text.{}'
        save_groups(self.grouped_text(dictionary), directory, pattern)
        if dictionary is not None:
            pattern = 'text.{}.int'
            ints = self.grouped_text_int(dictionary)
            save_groups(ints, directory, pattern)

    def split(self, dictionary):
        split_dir = self.split_directory()
        super(AlignableCorpus, self).split()
        self._split_texts(split_dir, dictionary)
        self._split_utt2fst(split_dir, dictionary)

    def initialize_corpus(self, dictionary=None, feature_config=None):
        if not self.utt_wav_mapping:
            raise CorpusError('There were no wav files found for transcribing this corpus. Please validate the corpus. '
                              'This error can also be caused if you\'re trying to find non-wav files without sox available '
                          'on the system path.')
        self.write()
        self.split(dictionary)
        if feature_config is not None:
            try:
                feature_config.generate_features(self)
            except Exception as e:
                if isinstance(e, KaldiProcessingError):
                    log_kaldi_errors(e.error_logs, self.logger)
                    e.update_log_file(self.logger.handlers[0].baseFilename)
                raise
        if isinstance(dictionary, MultispeakerDictionary):
            self.split_by_dictionary(dictionary)
        self.figure_utterance_lengths()

    def create_subset(self, subset, feature_config):

        split_directory = self.split_directory()
        subset_directory = os.path.join(self.output_directory, 'subset_{}'.format(subset))
        subset_utt_path = os.path.join(subset_directory, 'included_utts.txt')
        if os.path.exists(subset_utt_path):
            subset_utts = []
            with open(subset_utt_path, 'r', encoding='utf8') as f:
                for line in f:
                    subset_utts.append(line.strip())
        else:
            larger_subset_num = subset * 10
            if larger_subset_num < self.num_utterances:
                # Get all shorter utterances that are not one word long
                utts = sorted((x for x in self.utterance_lengths.keys() if ' ' in self.text_mapping[x]),
                              key=lambda x: self.utterance_lengths[x])
                larger_subset = utts[:larger_subset_num]
            else:
                larger_subset = self.utterance_lengths.keys()
            random.seed(1234)  # make it deterministic sampling
            subset_utts = set(random.sample(larger_subset, subset))
            log_dir = os.path.join(subset_directory, 'log')
            os.makedirs(log_dir, exist_ok=True)
            with open(subset_utt_path, 'w', encoding='utf8') as f:
                for u in subset_utts:
                    f.write('{}\n'.format(u))
        for j in range(self.num_jobs):
            for fn in ['text.{}', 'text.{}.int', 'utt2spk.{}']:
                sub_path = os.path.join(subset_directory, fn.format(j))
                with open(os.path.join(split_directory, fn.format(j)), 'r', encoding='utf8') as inf, \
                        open(sub_path, 'w', encoding='utf8') as outf:
                    for line in inf:
                        s = line.split()
                        if s[0] not in subset_utts:
                            continue
                        outf.write(line)
            subset_speakers = []
            sub_path = os.path.join(subset_directory, 'spk2utt.{}'.format(j))
            with open(os.path.join(split_directory, 'spk2utt.{}'.format(j)), 'r', encoding='utf8') as inf, \
                    open(sub_path, 'w', encoding='utf8') as outf:
                for line in inf:
                    line = line.split()
                    speaker, utts = line[0], line[1:]
                    filtered_utts = [x for x in utts if x in subset_utts]
                    if not filtered_utts:
                        continue
                    outf.write('{} {}\n'.format(speaker, ' '.join(filtered_utts)))
                    subset_speakers.append(speaker)
            sub_path = os.path.join(subset_directory, 'cmvn.{}.scp'.format(j))
            with open(os.path.join(split_directory, 'cmvn.{}.scp'.format(j)), 'r', encoding='utf8') as inf, \
                    open(sub_path, 'w', encoding='utf8') as outf:
                for line in inf:
                    line = line.split()
                    speaker, cmvn = line[0], line[1]
                    if speaker not in subset_speakers:
                        continue
                    outf.write('{} {}\n'.format(speaker, cmvn))
            if feature_config is not None:
                base_path = os.path.join(split_directory, feature_config.feature_id + '.{}.scp'.format(j))
                subset_scp = os.path.join(subset_directory, feature_config.feature_id + '.{}.scp'.format(j))
                if os.path.exists(subset_scp):
                    continue
                filtered = filter_scp(subset_utts, base_path)
                with open(subset_scp, 'w') as f:
                    for line in filtered:
                        f.write(line.strip() + '\n')

    def text_int_by_dictionay(self, multispeaker_dictionary, dictionary_name):
        dictionary = multispeaker_dictionary.dictionary_mapping[dictionary_name]
        oov_code = dictionary.oov_int
        self.utterance_oovs = {}
        output = []
        grouped_texts = self.grouped_text(dictionary)
        for g in grouped_texts:
            output_g = []
            for u, text in g:
                if u in self.ignored_utterances:
                    continue
                s = self.utt_speak_mapping[u]
                if multispeaker_dictionary.get_dictionary_name(s) != dictionary_name:
                    continue
                oovs = []
                new_text = []
                for i in range(len(text)):
                    t = text[i]
                    lookup = dictionary.to_int(t)
                    for w in lookup:
                        if w == oov_code:
                            oovs.append(text[i])
                        new_text.append(w)
                if oovs:
                    self.utterance_oovs[u] = oovs
                new_text = map(str, (x for x in new_text if isinstance(x, int)))
                output_g.append([u, ' '.join(new_text)])
            output.append(output_g)
        return output

    def split_by_dictionary(self, multispeaker_dictionary):
        super(AlignableCorpus, self).split_by_dictionary(multispeaker_dictionary)
        split_dir = self.split_directory()
        for name, d in multispeaker_dictionary.dictionary_mapping.items():
            spk2utt = self.text_int_by_dictionay(multispeaker_dictionary, name)
            pattern = 'text.{}.'+ name + '.int'
            save_groups(spk2utt, split_dir, pattern)
