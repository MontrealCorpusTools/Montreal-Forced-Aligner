import os
import subprocess
import sys
import traceback
import shutil
import struct
import wave
import re
import logging
import random
from collections import defaultdict, Counter
from textgrid import TextGrid, IntervalTier

from .helper import thirdparty_binary, load_text, load_scp, output_mapping, save_groups, filter_scp

from .exceptions import SampleRateError, CorpusError

from .dictionary import sanitize


def find_lab(filename, files):
    '''
    Finds a .lab file or .txt file that corresponds to a wav file.  The .lab extension is given priority.

    Parameters
    ----------
    filename : str
        Name of wav file
    files : list
        List of files to search in

    Returns
    -------
    str or None
        If a corresponding .lab or .txt file is found, returns it, otherwise returns None
    '''
    name, ext = os.path.splitext(filename)
    for f in files:
        fn, fext = os.path.splitext(f)
        if fn == name and fext.lower() == '.lab':
            return f
    for f in files: # Use .txt if no .lab file available
        fn, fext = os.path.splitext(f)
        if fn == name and fext.lower() == '.txt':
            return f
    return None


def find_wav(filename, files):
    '''
    Finds a .wav file that corresponds to a transcription file

    Parameters
    ----------
    filename : str
        Name of transcription file
    files : list
        List of files to search in

    Returns
    -------
    str or None
        If a corresponding .wav file is found, returns it, otherwise returns None
    '''
    name, ext = os.path.splitext(filename)
    for f in files:
        fn, fext = os.path.splitext(f)
        if fn == name and fext.lower() == '.wav':
            return f
    return None


def find_textgrid(filename, files):
    '''
    Finds a TextGrid file that corresponds to a wav file

    Parameters
    ----------
    filename : str
        Name of wav file
    files : list
        List of files to search in

    Returns
    -------
    str or None
        If a corresponding TextGrid is found, returns it, otherwise returns None
    '''
    name, ext = os.path.splitext(filename)
    for f in files:
        fn, fext = os.path.splitext(f)
        if fn == name and fext.lower() == '.textgrid':
            return f
    return None


def get_n_channels(file_path):
    '''
    Return the number of channels for a sound file

    Parameters
    ----------
    file_path : str
        Path to a wav file

    Returns
    -------
    int
        Number of channels (1 if mono, 2 if stereo)
    '''

    with wave.open(file_path, 'rb') as soundf:
        n_channels = soundf.getnchannels()
    return n_channels


def get_sample_rate(file_path):
    with wave.open(file_path, 'rb') as soundf:
        sr = soundf.getframerate()
    return sr


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as soundf:
        sr = soundf.getframerate()
        nframes = soundf.getnframes()
    return nframes / sr


def extract_temp_channels(wav_path, temp_directory):
    '''
    Extract a single channel from a stereo file to a new mono wav file

    Parameters
    ----------
    wav_path : str
        Path to stereo wav file
    temp_directory : str
        Directory to save extracted
    '''
    name, ext = os.path.splitext(wav_path)
    base = os.path.basename(name)
    A_path = os.path.join(temp_directory, base + '_A.wav')
    B_path = os.path.join(temp_directory, base + '_B.wav')
    samp_step = 1000000
    if not os.path.exists(A_path):
        with wave.open(wav_path, 'rb') as inf, \
                wave.open(A_path, 'wb') as af, \
                wave.open(B_path, 'wb') as bf:
            chans = inf.getnchannels()
            samps = inf.getnframes()
            samplerate = inf.getframerate()
            sampwidth = inf.getsampwidth()
            assert sampwidth == 2
            af.setnchannels(1)
            af.setframerate(samplerate)
            af.setsampwidth(sampwidth)
            bf.setnchannels(1)
            bf.setframerate(samplerate)
            bf.setsampwidth(sampwidth)
            cur_samp = 0
            while cur_samp < samps:
                s = inf.readframes(samp_step)
                cur_samp += samp_step
                act = samp_step
                if cur_samp > samps:
                    act -= (cur_samp - samps)

                unpstr = '<{0}h'.format(act * chans)  # little-endian 16-bit samples
                x = list(struct.unpack(unpstr, s))  # convert the byte string into a list of ints
                values = [struct.pack('h', d) for d in x[0::chans]]
                value_str = b''.join(values)
                af.writeframes(value_str)
                values = [struct.pack('h', d) for d in x[1::chans]]
                value_str = b''.join(values)
                bf.writeframes(value_str)
    return A_path, B_path


def parse_transcription(text):
    words = [sanitize(x) for x in text.split()]
    words = [x for x in words if x not in ['', '-', "'"]]
    return words


class Corpus(object):
    '''
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

    '''

    def __init__(self, directory, output_directory,
                 speaker_characters=0,
                 num_jobs=3, debug=False,
                 ignore_exceptions=False):
        self.debug = debug
        log_dir = os.path.join(output_directory, 'logging')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'corpus.log')
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
        handler.setFormatter = logging.Formatter('%(name)s %(message)s')
        root_logger.addHandler(handler)
        if not os.path.exists(directory):
            raise CorpusError('The directory \'{}\' does not exist.'.format(directory))
        if not os.path.isdir(directory):
            raise CorpusError('The specified path for the corpus ({}) is not a directory.'.format(directory))
        if num_jobs < 1:
            num_jobs = 1

        print('Setting up corpus information...')
        root_logger.info('Setting up corpus information...')
        self.directory = directory
        self.output_directory = os.path.join(output_directory, 'corpus_data')
        self.temp_directory = os.path.join(self.output_directory, 'temp')
        os.makedirs(self.temp_directory, exist_ok=True)
        self.num_jobs = num_jobs

        # Set up mapping dictionaries
        self.speak_utt_mapping = defaultdict(list)
        self.utt_speak_mapping = {}
        self.utt_wav_mapping = {}
        self.text_mapping = {}
        self.word_counts = Counter()
        self.segments = {}
        self.feat_mapping = {}
        self.cmvn_mapping = {}
        self.ignored_utterances = []
        self.wav_files = []
        self.wav_durations = {}
        self.utterance_lengths = {}
        self.utterance_oovs = {}
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if os.path.exists(feat_path):
            self.feat_mapping = load_scp(feat_path)

        if speaker_characters == 0:
            self.speaker_directories = True
        else:
            self.speaker_directories = False
        self.sample_rates = defaultdict(set)
        self.no_transcription_files = []
        self.decode_error_files = []
        self.unsupported_sample_rate = []
        self.wav_read_errors = []
        self.textgrid_read_errors = {}
        self.transcriptions_without_wavs = []
        self.file_directory_mapping = {}
        self.speaker_ordering = {}
        self.tg_count = 0
        self.lab_count = 0
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            for f in sorted(files):
                file_name, ext = os.path.splitext(f)
                if ext.lower() != '.wav':
                    if ext.lower() in ['.lab', '.textgrid']:
                        wav_path = find_wav(f, files)
                        if wav_path is None:
                            self.transcriptions_without_wavs.append(os.path.join(root, f))
                    continue
                lab_name = find_lab(f, files)
                wav_path = os.path.join(root, f)
                try:
                    sr = get_sample_rate(wav_path)
                except wave.Error:
                    self.wav_read_errors.append(wav_path)
                    continue
                if sr < 16000:
                    self.unsupported_sample_rate.append(wav_path)
                if lab_name is not None:
                    utt_name = file_name
                    if utt_name in self.utt_wav_mapping:
                        ind = 0
                        fixed_utt_name = utt_name
                        while fixed_utt_name not in self.utt_wav_mapping:
                            ind += 1
                            fixed_utt_name = utt_name + '_{}'.format(ind)
                        utt_name = fixed_utt_name
                    if self.feat_mapping and utt_name not in self.feat_mapping:
                        self.ignored_utterances.append(utt_name)
                        continue
                    lab_path = os.path.join(root, lab_name)
                    try:
                        text = load_text(lab_path)
                    except UnicodeDecodeError:
                        self.decode_error_files.append(lab_path)
                        continue
                    words = parse_transcription(text)
                    if not words:
                        continue
                    self.word_counts.update(words)
                    self.text_mapping[utt_name] = ' '.join(words)
                    if self.speaker_directories:
                        speaker_name = os.path.basename(root)
                    else:
                        if isinstance(speaker_characters, int):
                            speaker_name = f[:speaker_characters]
                        elif speaker_characters == 'prosodylab':
                            speaker_name = f.split('_')[1]
                    speaker_name = speaker_name.strip().replace(' ', '_')
                    utt_name = utt_name.strip().replace(' ', '_')
                    self.speak_utt_mapping[speaker_name].append(utt_name)
                    self.utt_wav_mapping[utt_name] = wav_path
                    self.sample_rates[get_sample_rate(wav_path)].add(speaker_name)
                    self.utt_speak_mapping[utt_name] = speaker_name
                    self.file_directory_mapping[utt_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')

                    self.lab_count += 1
                else:
                    tg_name = find_textgrid(f, files)
                    if tg_name is None:
                        self.no_transcription_files.append(wav_path)
                        continue
                    self.wav_files.append(file_name)
                    self.wav_durations[file_name] = get_wav_duration(wav_path)
                    tg_path = os.path.join(root, tg_name)
                    tg = TextGrid()
                    try:
                        tg.read(tg_path)
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        self.textgrid_read_errors[tg_path] = '\n'.join(
                            traceback.format_exception(exc_type, exc_value, exc_traceback))
                    n_channels = get_n_channels(wav_path)
                    num_tiers = len(tg.tiers)
                    if n_channels == 2:
                        A_name = file_name + "_A"
                        B_name = file_name + "_B"

                        A_path, B_path = extract_temp_channels(wav_path, self.temp_directory)
                    elif n_channels > 2:
                        raise (Exception('More than two channels'))
                    self.speaker_ordering[file_name] = []
                    if not self.speaker_directories:
                        if isinstance(speaker_characters, int):
                            speaker_name = f[:speaker_characters]
                        elif speaker_characters == 'prosodylab':
                            speaker_name = f.split('_')[1]
                        speaker_name = speaker_name.strip().replace(' ', '_')
                        self.speaker_ordering[file_name].append(speaker_name)
                    for i, ti in enumerate(tg.tiers):
                        if ti.name.lower() == 'notes':
                            continue
                        if not isinstance(ti, IntervalTier):
                            continue
                        if self.speaker_directories:
                            speaker_name = ti.name.strip().replace(' ', '_')
                            self.speaker_ordering[file_name].append(speaker_name)
                        self.sample_rates[get_sample_rate(wav_path)].add(speaker_name)
                        for interval in ti:
                            text = interval.mark.lower().strip()
                            words = parse_transcription(text)
                            if not words:
                                continue
                            begin, end = round(interval.minTime, 4), round(interval.maxTime, 4)
                            utt_name = '{}_{}_{}_{}'.format(speaker_name, file_name, begin, end)
                            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
                            if n_channels == 1:
                                if self.feat_mapping and utt_name not in self.feat_mapping:
                                    self.ignored_utterances.append(utt_name)
                                self.segments[utt_name] = '{} {} {}'.format(file_name, begin, end)
                                self.utt_wav_mapping[file_name] = wav_path
                            else:
                                if i < num_tiers / 2:
                                    utt_name += '_A'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                    self.segments[utt_name] = '{} {} {}'.format(A_name, begin, end)
                                    self.utt_wav_mapping[A_name] = A_path
                                else:
                                    utt_name += '_B'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                    self.segments[utt_name] = '{} {} {}'.format(B_name, begin, end)
                                    self.utt_wav_mapping[B_name] = B_path
                            self.text_mapping[utt_name] = ' '.join(words)
                            self.word_counts.update(words)
                            self.utt_speak_mapping[utt_name] = speaker_name
                            self.speak_utt_mapping[speaker_name].append(utt_name)
                    if n_channels == 2:
                        self.file_directory_mapping[A_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')
                        self.file_directory_mapping[B_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')
                    else:
                        self.file_directory_mapping[file_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')
                    self.tg_count += 1

        self.issues_check = self.ignored_utterances or self.no_transcription_files or \
                       self.textgrid_read_errors or self.unsupported_sample_rate or self.decode_error_files

        bad_speakers = []
        for speaker in self.speak_utt_mapping.keys():
            count = 0
            for k, v in self.sample_rates.items():
                if speaker in v:
                    count += 1
            if count > 1:
                bad_speakers.append(speaker)
        if bad_speakers:
            msg = 'The following speakers had multiple speaking rates: {}. ' \
                  'Please make sure that each speaker has a consistent sampling rate.'.format(', '.join(bad_speakers))
            root_logger.error(msg)
            raise (SampleRateError(msg))

        if len(self.speak_utt_mapping) < self.num_jobs:
            self.num_jobs = len(self.speak_utt_mapping)
        if self.num_jobs < len(self.sample_rates.keys()):
            self.num_jobs = len(self.sample_rates.keys())
            msg = 'The number of jobs was set to {}, due to the different sample rates in the dataset. ' \
                  'If you would like to use fewer parallel jobs, ' \
                  'please resample all wav files to the same sample rate.'.format(self.num_jobs)
            print('WARNING: ' + msg)
            root_logger.warning(msg)
        self.find_best_groupings()

    @property
    def ivector_directory(self):
        return os.path.join(self.output_directory, 'ivectors')

    @property
    def word_set(self):
        return set(self.word_counts)

    @property
    def utterances(self):
        return list(self.utt_speak_mapping.keys())

    def find_best_groupings(self):
        if self.segments:
            ratio = len(self.segments.keys()) / len(self.utt_speak_mapping.keys())
            segment_job_num = int(ratio * self.num_jobs)
            if segment_job_num == 0:
                segment_job_num = 1
        else:
            segment_job_num = 0
        full_wav_job_num = self.num_jobs - segment_job_num
        num_sample_rates = len(self.sample_rates.keys())
        jobs_per_sample_rate = {x: 1 for x in self.sample_rates.keys()}
        remaining_jobs = self.num_jobs - num_sample_rates
        while remaining_jobs > 0:
            min_num = min(jobs_per_sample_rate.values())
            addable = sorted([k for k, v in jobs_per_sample_rate.items() if v == min_num],
                             key=lambda x: -1 * len(self.sample_rates[x]))
            jobs_per_sample_rate[addable[0]] += 1
            remaining_jobs -= 1
        self.speaker_groups = []
        self.frequency_configs = []
        job_num = 0
        for k, v in jobs_per_sample_rate.items():
            speakers = sorted(self.sample_rates[k])
            groups = [[] for x in range(v)]

            configs = [(job_num + x, {'sample-frequency': k, 'low-freq': 20, 'high-freq': 7800}) for x in range(v)]
            ind = 0
            while speakers:
                s = speakers.pop(0)
                groups[ind].append(s)
                ind += 1
                if ind >= v:
                    ind = 0

            job_num += v
            self.speaker_groups.extend(groups)
            self.frequency_configs.extend(configs)
        self.groups = []
        for x in self.speaker_groups:
            g = []
            for s in x:
                g.extend(self.speak_utt_mapping[s])
            self.groups.append(g)

    def speaker_utterance_info(self):
        num_speakers = len(self.speak_utt_mapping.keys())
        average_utterances = sum(len(x) for x in self.speak_utt_mapping.values()) / num_speakers
        msg = 'Number of speakers in corpus: {}, average number of utterances per speaker: {}'.format(num_speakers,
                                                                                                      average_utterances)
        root_logger = logging.getLogger()
        root_logger.info(msg)
        return msg

    def parse_features_logs(self):
        line_reg_ex = r'Did not find features for utterance (\w+)'
        missing_features = []
        with open(os.path.join(self.features_log_directory, 'cmvn.log'), 'r') as f:
            for line in f:
                m = re.search(line_reg_ex, line)
                if m is not None:
                    missing_features.append(m.groups()[0])

    @property
    def num_utterances(self):
        return len(self.utt_speak_mapping)

    @property
    def features_directory(self):
        return os.path.join(self.output_directory, 'features')

    @property
    def features_log_directory(self):
        return os.path.join(self.features_directory, 'log')

    @property
    def grouped_wav(self):
        output = []
        for g in self.groups:
            done = set()
            output_g = []
            for u in g:
                if u in self.ignored_utterances:
                    continue
                if not self.segments:
                    try:
                        output_g.append([u, self.utt_wav_mapping[u]])
                    except KeyError:
                        pass
                else:
                    try:
                        r = self.segments[u].split(' ')[0]
                    except KeyError:
                        continue
                    if r not in done:
                        output_g.append([r, self.utt_wav_mapping[r]])
                        done.add(r)
            output.append(output_g)
        return output

    @property
    def grouped_feat(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                if u in self.ignored_utterances:
                    continue
                try:
                    output_g.append([u, self.feat_mapping[u]])
                except KeyError:
                    pass
            output.append(output_g)
        return output

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
                    try:
                        text = self.text_mapping[u].split()
                    except KeyError:
                        continue
                    new_text = []
                    for t in text:
                        lookup = dictionary.separate_clitics(t)
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
                oovs = []
                for i in range(len(text)):
                    t = text[i]
                    lookup = dictionary.to_int(t)
                    if lookup is None:
                        continue
                    if lookup == oov_code:
                        oovs.append(t)
                    text[i] = lookup
                if oovs:
                    self.utterance_oovs[u] = oovs
                new_text = map(str, (x for x in text if isinstance(x, int)))
                output_g.append([u, ' '.join(new_text)])
            output.append(output_g)
        return output

    @property
    def grouped_cmvn(self):
        output = []
        try:
            for g in self.speaker_groups:
                output_g = []
                for s in sorted(g):
                    try:
                        output_g.append([s, self.cmvn_mapping[s]])
                    except KeyError:
                        pass
                output.append(output_g)
        except KeyError:
            raise (CorpusError(
                'Something went wrong while setting up the corpus. Please delete the {} folder and try again.'.format(
                    self.output_directory)))
        return output

    @property
    def grouped_utt2spk(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in sorted(g):
                if u in self.ignored_utterances:
                    continue
                try:
                    output_g.append([u, self.utt_speak_mapping[u]])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    def get_word_frquency(self, dictionary):
        word_counts = Counter()
        for u, text in self.text_mapping.items():
            new_text = []
            text = text.split()
            for t in text:
                lookup = dictionary.separate_clitics(t)
                if lookup is None:
                    continue
                new_text.extend(x for x in lookup if x != '')
            word_counts.update(new_text)
        return {k: v / sum(word_counts.values()) for k, v in word_counts.items()}

    def grouped_utt2fst(self, dictionary, num_frequent_words=10):
        word_frequencies = self.get_word_frquency(dictionary)
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
                for t in text:
                    lookup = dictionary.separate_clitics(t)
                    if lookup is None:
                        continue
                    new_text.extend(x for x in lookup if x != '')
                try:
                    fst_text = dictionary.create_utterance_fst(new_text, most_frequent)
                except ZeroDivisionError:
                    print(u, text, new_text)
                    raise
                output_g.append([u, fst_text])
            output.append(output_g)
        return output

    @property
    def grouped_segments(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                try:
                    output_g.append([u, self.segments[u]])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    @property
    def grouped_spk2utt(self):
        output = []
        for g in self.speaker_groups:
            output_g = []
            for s in sorted(g):
                try:
                    output_g.append([s, sorted(self.speak_utt_mapping[s])])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    def get_wav_duration(self, utt):
        if utt in self.wav_durations:
            return self.wav_durations[utt]
        if not self.segments:
            wav_path = self.utt_wav_mapping[utt]
        else:
            rec = self.segments[utt].split(' ')[0]
            wav_path = self.utt_wav_mapping[rec]
        with wave.open(wav_path, 'rb') as soundf:
            sr = soundf.getframerate()
            nframes = soundf.getnframes()
        return nframes / sr

    def split_directory(self):
        directory = os.path.join(self.output_directory, 'split{}'.format(self.num_jobs))
        return directory

    def subset_directory(self, subset, feature_config):
        if subset is None or subset > self.num_utterances:
            return self.split_directory()
        directory = os.path.join(self.output_directory, 'subset_{}'.format(subset))
        if not os.path.exists(directory):
            self.create_subset(subset, feature_config)
        return directory

    def write(self):
        self._write_speak_utt()
        self._write_utt_speak()
        self._write_text()
        self._write_wavscp()

    def _write_utt_speak(self):
        utt2spk = os.path.join(self.output_directory, 'utt2spk')
        output_mapping(self.utt_speak_mapping, utt2spk)

    def _write_speak_utt(self):
        spk2utt = os.path.join(self.output_directory, 'spk2utt')
        output_mapping(self.speak_utt_mapping, spk2utt)

    def _write_text(self):
        text = os.path.join(self.output_directory, 'text')
        output_mapping(self.text_mapping, text)

    def _write_wavscp(self):
        wavscp = os.path.join(self.output_directory, 'wav.scp')
        output_mapping(self.utt_wav_mapping, wavscp)

    def _write_segments(self):
        if not self.segments:
            return
        segments = os.path.join(self.output_directory, 'segments')
        output_mapping(self.segments, segments)

    def _split_utt2spk(self, directory):
        pattern = 'utt2spk.{}'
        save_groups(self.grouped_utt2spk, directory, pattern)

    def _split_utt2fst(self, directory, dictionary):
        pattern = 'utt2fst.{}'
        save_groups(self.grouped_utt2fst(dictionary), directory, pattern, multiline=True)

    def _split_segments(self, directory):
        if not self.segments:
            return
        pattern = 'segments.{}'
        save_groups(self.grouped_segments, directory, pattern)

    def _split_spk2utt(self, directory):
        pattern = 'spk2utt.{}'
        save_groups(self.grouped_spk2utt, directory, pattern)

    def _split_wavs(self, directory):
        pattern = 'wav.{}.scp'
        save_groups(self.grouped_wav, directory, pattern)

    def _split_texts(self, directory, dictionary=None):
        pattern = 'text.{}'
        save_groups(self.grouped_text(dictionary), directory, pattern)
        if dictionary is not None:
            pattern = 'text.{}.int'
            ints = self.grouped_text_int(dictionary)
            save_groups(ints, directory, pattern)

    def _split_cmvns(self, directory):
        if not self.cmvn_mapping:
            cmvn_path = os.path.join(self.output_directory, 'cmvn.scp')
            self.cmvn_mapping = load_scp(cmvn_path)
        pattern = 'cmvn.{}.scp'
        save_groups(self.grouped_cmvn, directory, pattern)

    def combine_feats(self):
        self.feat_mapping = {}
        split_directory = self.split_directory()
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        with open(feat_path, 'w') as outf:
            for i in range(self.num_jobs):
                path = os.path.join(split_directory, 'feats.{}.scp'.format(i))
                with open(path, 'r') as inf:
                    for line in inf:
                        line = line.strip()
                        if line == '':
                            continue
                        f = line.split(maxsplit=1)
                        self.feat_mapping[f[0]] = f[1]
                        outf.write(line + '\n')
        if len(self.feat_mapping.keys()) != len(self.utt_speak_mapping.keys()):
            for k in self.utt_speak_mapping.keys():
                if k not in self.feat_mapping:
                    self.ignored_utterances.append(k)
            for k, v in self.speak_utt_mapping.items():
                self.speak_utt_mapping[k] = list(filter(lambda x: x in self.feat_mapping, v))
        self.figure_utterance_lengths()

    def get_feat_dim(self, feature_config):

        path = os.path.join(self.split_directory(), feature_config.feature_id + '.0.scp')
        with open(os.devnull, 'w') as devnull:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'scp:' + path, '-'],
                                        stdout=subprocess.PIPE,
                                        stderr=devnull)
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode('utf8').strip()
        return int(feats)

    def initialize_corpus(self, dictionary):
        root_logger = logging.getLogger()
        split_dir = self.split_directory()
        self.write()
        if not os.path.exists(split_dir):
            os.makedirs(os.path.join(split_dir, 'log'), exist_ok=True)
            root_logger.info('Setting up training data...')
            print('Setting up corpus_data directory...')
            self._split_wavs(split_dir)
            self._split_utt2spk(split_dir)
            self._split_spk2utt(split_dir)
            self._split_texts(split_dir, dictionary)
            self._split_utt2fst(split_dir, dictionary)
            self._split_segments(split_dir)
        self.figure_utterance_lengths()

    def figure_utterance_lengths(self):
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if os.path.exists(feat_path):
            with open(os.devnull, 'w') as devnull:
                dim_proc = subprocess.Popen([thirdparty_binary('feat-to-len'),
                                             'scp:' + feat_path, 'ark,t:-'],
                                            stdout=subprocess.PIPE,
                                            stderr=devnull)
                stdout, stderr = dim_proc.communicate()
                feats = stdout.decode('utf8').strip()
                for line in feats.splitlines():
                    line = line.strip()
                    line = line.split()
                    self.utterance_lengths[line[0]] = int(line[1])

    def create_subset(self, subset, feature_config):
        larger_subset_num = subset * 10
        if larger_subset_num < self.num_utterances:
            utts = sorted((x for x in self.utterance_lengths.keys() if ' ' in self.text_mapping[x]),
                          key=lambda x: self.utterance_lengths[x]) # Get all shorter utterances that are not one word long
            larger_subset = utts[:larger_subset_num]
        else:
            larger_subset = self.utterance_lengths.keys()

        subset_utts = set(random.sample(larger_subset, subset))
        split_directory = self.split_directory()
        subset_directory = os.path.join(self.output_directory, 'subset_{}'.format(subset))
        log_dir = os.path.join(subset_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        subset_utt_path = os.path.join(subset_directory, 'included_utts.txt')
        with open(subset_utt_path, 'w', encoding='utf8') as f:
            for u in subset_utts:
                f.write('{}\n'.format(u))
        for j in range(self.num_jobs):
            for fn in ['text.{}', 'text.{}.int', 'utt2spk.{}']:
                with open(os.path.join(split_directory, fn.format(j)), 'r', encoding='utf8') as inf, \
                        open(os.path.join(subset_directory, fn.format(j)), 'w', encoding='utf8') as outf:
                    for line in inf:
                        s = line.split()
                        if s[0] not in subset_utts:
                            continue
                        outf.write(line)
            with open(os.path.join(split_directory, 'spk2utt.{}'.format(j)), 'r', encoding='utf8') as inf, \
                    open(os.path.join(subset_directory, 'spk2utt.{}'.format(j)), 'w', encoding='utf8') as outf:
                for line in inf:
                    line = line.split()
                    speaker, utts = line[0], line[1:]
                    filtered_utts = [x for  x in utts if x in subset_utts]
                    outf.write('{} {}\n'.format(speaker, ' '.join(filtered_utts)))
            if feature_config is not None:
                base_path = os.path.join(split_directory, feature_config.feature_id + '.{}.scp'.format(j))
                subset_scp = os.path.join(subset_directory, feature_config.feature_id + '.{}.scp'.format(j))
                filtered = filter_scp(subset_utts, base_path)
                with open(subset_scp, 'w') as f:
                    for line in filtered:
                        f.write(line.strip() + '\n')

