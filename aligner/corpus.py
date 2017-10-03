import os
import subprocess
import sys
import traceback
import shutil
import struct
import wave
import logging
from collections import defaultdict, Counter
from textgrid import TextGrid, IntervalTier

from .helper import thirdparty_binary, load_text, make_safe
from .multiprocessing import mfcc

from .exceptions import SampleRateError, CorpusError

from .dictionary import sanitize

from .config import MfccConfig


def output_mapping(mapping, path):
    with open(path, 'w', encoding='utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            f.write('{} {}\n'.format(k, v))


def save_scp(scp, path, sort=True, multiline=False):
    with open(path, 'w', encoding='utf8') as f:
        if sort:
            scp = sorted(scp)
        for line in scp:
            if multiline:
                f.write('{}\n{}\n'.format(make_safe(line[0]), make_safe(line[1])))
            else:
                f.write('{}\n'.format(' '.join(map(make_safe, line))))


def save_groups(groups, seg_dir, pattern, multiline=False):
    for i, g in enumerate(groups):
        path = os.path.join(seg_dir, pattern.format(i))
        save_scp(g, path, multiline=multiline)


def load_scp(path):
    '''
    Load a Kaldi script file (.scp)

    See http://kaldi-asr.org/doc/io.html#io_sec_scp_details for more information

    Parameters
    ----------
    path : str
        Path to Kaldi script file

    Returns
    -------
    dict
        Dictionary where the keys are the first couple and the values are all
        other columns in the script file

    '''
    scp = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line_list = line.split()
            key = line_list.pop(0)
            if len(line_list) == 1:
                value = line_list[0]
            else:
                value = line_list
            scp[key] = value
    return scp


def find_lab(filename, files):
    '''
    Finds a .lab file that corresponds to a wav file

    Parameters
    ----------
    filename : str
        Name of wav file
    files : list
        List of files to search in

    Returns
    -------
    str or None
        If a corresponding .lab file is found, returns it, otherwise returns None
    '''
    name, ext = os.path.splitext(filename)
    for f in files:
        fn, fext = os.path.splitext(f)
        if fn == name and fext.lower() == '.lab':
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
    mfcc_config : MfccConfig
        Configuration object for how to calculate MFCCs
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
                 use_speaker_information=True,
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
        self.output_directory = os.path.join(output_directory, 'train')
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
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if os.path.exists(feat_path):
            self.feat_mapping = load_scp(feat_path)

        if speaker_characters == 0:
            self.speaker_directories = True
        else:
            self.speaker_directories = False
        self.sample_rates = defaultdict(set)
        no_transcription_files = []
        decode_error_files = []
        unsupported_sample_rate = []
        ignored_duplicates = False
        textgrid_read_errors = {}
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            for f in sorted(files):
                file_name, ext = os.path.splitext(f)
                if ext.lower() != '.wav':
                    continue
                lab_name = find_lab(f, files)
                wav_path = os.path.join(root, f)
                sr = get_sample_rate(wav_path)
                if sr < 16000:
                    unsupported_sample_rate.append(wav_path)
                    continue
                if lab_name is not None:
                    utt_name = file_name
                    if utt_name in self.utt_wav_mapping:
                        if not ignore_exceptions:
                            prev_wav = self.utt_wav_mapping[utt_name]
                            raise CorpusError(
                                'Files with the same file name are not permitted. Files with the same name are: {}, {}.'.format(
                                    prev_wav, wav_path))
                        else:
                            ignored_duplicates = True
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
                        decode_error_files.append(lab_path)
                        continue
                    words = [sanitize(x) for x in text.split()]
                    words = [x for x in words if x not in ['', '-', "'"]]
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
                else:
                    tg_name = find_textgrid(f, files)
                    if tg_name is None:
                        no_transcription_files.append(wav_path)
                        continue
                    self.wav_files.append(file_name)
                    self.wav_durations[file_name] = get_wav_duration(wav_path)
                    tg_path = os.path.join(root, tg_name)
                    tg = TextGrid()
                    try:
                        tg.read(tg_path)
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        textgrid_read_errors[tg_path] = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                    n_channels = get_n_channels(wav_path)
                    num_tiers = len(tg.tiers)
                    if n_channels == 2:
                        A_name = file_name + "_A"
                        B_name = file_name + "_B"

                        A_path, B_path = extract_temp_channels(wav_path, self.temp_directory)
                    elif n_channels > 2:
                        raise (Exception('More than two channels'))
                    if not self.speaker_directories:
                        if isinstance(speaker_characters, int):
                            speaker_name = f[:speaker_characters]
                        elif speaker_characters == 'prosodylab':
                            speaker_name = f.split('_')[1]
                        speaker_name = speaker_name.strip().replace(' ', '_')
                    for i, ti in enumerate(tg.tiers):
                        if ti.name.lower() == 'notes':
                            continue
                        if not isinstance(ti, IntervalTier):
                            continue
                        if self.speaker_directories:
                            speaker_name = ti.name.strip().replace(' ', '_')
                        self.sample_rates[get_sample_rate(wav_path)].add(speaker_name)
                        for interval in ti:
                            label = interval.mark.lower().strip()
                            #label = sanitize(label)
                            words = [sanitize(x) for x in label.split()]
                            words = [x for x in words if x not in ['', '-', "'"]]
                            if not words:
                                continue
                            begin, end = round(interval.minTime, 4), round(interval.maxTime, 4)
                            utt_name = '{}_{}_{}_{}'.format(speaker_name, file_name, begin, end)
                            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
                            if n_channels == 1:
                                if self.feat_mapping and utt_name not in self.feat_mapping:
                                    self.ignored_utterances.append(utt_name)
                                    continue
                                self.segments[utt_name] = '{} {} {}'.format(file_name, begin, end)
                                self.utt_wav_mapping[file_name] = wav_path
                            else:
                                if i < num_tiers / 2:
                                    utt_name += '_A'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                        continue
                                    self.segments[utt_name] = '{} {} {}'.format(A_name, begin, end)
                                    self.utt_wav_mapping[A_name] = A_path
                                else:
                                    utt_name += '_B'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                        continue
                                    self.segments[utt_name] = '{} {} {}'.format(B_name, begin, end)
                                    self.utt_wav_mapping[B_name] = B_path
                            self.text_mapping[utt_name] = ' '.join(words)
                            self.word_counts.update(words)
                            self.utt_speak_mapping[utt_name] = speaker_name
                            self.speak_utt_mapping[speaker_name].append(utt_name)
        if ignored_duplicates:
            print('At least one duplicate wav file name was found and treated as a different utterance.')
        if len(self.ignored_utterances) > 0:
            print('{} utterance(s) were ignored due to lack of features, please see {} for more information.'.format(
                len(self.ignored_utterances), self.log_file))
            root_logger.warning(
                'The following utterances were ignored due to lack of features: {}.  '
                'See relevant logs for more information'.format(', '.join(self.ignored_utterances)))
        if len(no_transcription_files) > 0:
            print(
                '{} wav file(s) were ignored because neither a .lab file or a .TextGrid file could be found, '
                'please see {} for more information'.format(len(no_transcription_files), self.log_file))
            root_logger.warning(
                'The following wav files were ignored due to lack of of a .lab or a .TextGrid file: {}.'.format(
                    ', '.join(no_transcription_files)))
        if textgrid_read_errors:
            print('{} TextGrid files were ignored due to errors loading them. '
                  'Please see {} for more information on the errors.'.format(len(textgrid_read_errors), self.log_file))
            for k, v in textgrid_read_errors.items():
                root_logger.warning('The TextGrid file {} gave the following error on load:\n\n{}'.format(k, v))
        if len(unsupported_sample_rate) > 0:
            print(
                '{} wav file(s) were ignored because they had a sample rate less than 16000, '
                'which is not currently supported, please see {} for more information'.format(
                    len(unsupported_sample_rate), self.log_file))
            root_logger.warning(
                'The following wav files were ignored due to a sample rate lower than 16000: {}.'.format(
                    ', '.join(unsupported_sample_rate)))
        if decode_error_files:
            print('There was an issue reading {} text file(s).  '
                  'Please see {} for more information.'.format(len(decode_error_files), self.log_file))
            root_logger.warning(
                'The following lab files were ignored because they could not be parsed with utf8: {}.'.format(
                    ', '.join(decode_error_files)))
        bad_speakers = []
        for speaker in self.speak_utt_mapping.keys():
            count = 0
            for k, v in self.sample_rates.items():
                if speaker in v:
                    count += 1
            if count > 1:
                bad_speakers.append(speaker)
        if bad_speakers:
            msg = 'The following speakers had multiple speaking rates: {}.  Please make sure that each speaker has a consistent sampling rate.'.format(
                ', '.join(bad_speakers))
            root_logger.error(msg)
            raise (SampleRateError(msg))

        if len(self.speak_utt_mapping) < self.num_jobs:
            self.num_jobs = len(self.speak_utt_mapping)
        if self.num_jobs < len(self.sample_rates.keys()):
            self.num_jobs = len(self.sample_rates.keys())
            msg = 'The number of jobs was set to {}, due to the different sample rates in the dataset.  If you would like to use fewer parallel jobs, please resample all wav files to the same sample rate.'.format(
                self.num_jobs)
            print(msg)
            root_logger.warning(msg)
        self.find_best_groupings()

    @property
    def word_set(self):
        return set(self.word_counts)

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
        self.mfcc_configs = []
        job_num = 0
        for k, v in jobs_per_sample_rate.items():
            speakers = sorted(self.sample_rates[k])
            groups = [[] for x in range(v)]

            configs = [MfccConfig(self.mfcc_directory, job=job_num + x, kwargs={'sample-frequency': k,
                                                                                'low-freq': 20,
                                                                                'high-freq': 7800}) for x in range(v)]
            ind = 0
            while speakers:
                s = speakers.pop(0)
                groups[ind].append(s)
                ind += 1
                if ind >= v:
                    ind = 0

            job_num += v
            self.speaker_groups.extend(groups)
            self.mfcc_configs.extend(configs)
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

    def parse_mfcc_logs(self):
        pass

    @property
    def num_utterances(self):
        return len(self.utt_speak_mapping)

    @property
    def mfcc_directory(self):
        return os.path.join(self.output_directory, 'mfcc')

    @property
    def mfcc_log_directory(self):
        return os.path.join(self.mfcc_directory, 'log')

    @property
    def grouped_wav(self):
        output = []
        for g in self.groups:
            done = set()
            output_g = []
            for u in g:
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
        all_oovs = []
        output = []
        grouped_texts = self.grouped_text(dictionary)
        for g in grouped_texts:
            output_g = []
            for u, text in g:
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
                    all_oovs.append(u + ' ' + ', '.join(oovs))
                new_text = map(str, (x for x in text if isinstance(x, int)))
                output_g.append([u, ' '.join(new_text)])
            output.append(output_g)
        return output, all_oovs

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

    @property
    def split_directory(self):
        return os.path.join(self.output_directory, 'split{}'.format(self.num_jobs))

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

    def _split_feats(self, directory):
        if not self.feat_mapping:
            feat_path = os.path.join(self.output_directory, 'feats.scp')
            self.feat_mapping = load_scp(feat_path)
        pattern = 'feats.{}.scp'
        save_groups(self.grouped_feat, directory, pattern)

    def _split_texts(self, directory, dictionary=None):
        pattern = 'text.{}'
        save_groups(self.grouped_text(dictionary), directory, pattern)
        if dictionary is not None:
            pattern = 'text.{}.int'
            ints, all_oovs = self.grouped_text_int(dictionary)
            save_groups(ints, directory, pattern)
            if all_oovs:
                with open(os.path.join(directory, 'utterance_oovs.txt'), 'w', encoding='utf8') as f:
                    for oov in sorted(all_oovs):
                        f.write(oov + '\n')
            dictionary.save_oovs_found(directory)

    def _split_cmvns(self, directory):
        if not self.cmvn_mapping:
            cmvn_path = os.path.join(self.output_directory, 'cmvn.scp')
            self.cmvn_mapping = load_scp(cmvn_path)
        pattern = 'cmvn.{}.scp'
        save_groups(self.grouped_cmvn, directory, pattern)

    def create_mfccs(self):
        log_directory = self.mfcc_log_directory
        os.makedirs(log_directory, exist_ok=True)
        if os.path.exists(os.path.join(self.mfcc_directory, 'cmvn')):
            print("Using previous MFCCs")
            return
        print('Calculating MFCCs...')
        self._split_wavs(self.mfcc_log_directory)
        self._split_segments(self.mfcc_log_directory)
        mfcc(self.mfcc_directory, log_directory, self.num_jobs, self.mfcc_configs)
        self.parse_mfcc_logs()
        self._combine_feats()
        print('Calculating CMVN...')
        self._calc_cmvn()

    def _combine_feats(self):
        root_logger = logging.getLogger()
        self.feat_mapping = {}
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        with open(feat_path, 'w') as outf:
            for i in range(self.num_jobs):
                path = os.path.join(self.mfcc_directory, 'raw_mfcc.{}.scp'.format(i))
                with open(path, 'r') as inf:
                    for line in inf:
                        line = line.strip()
                        if line == '':
                            continue
                        f = line.split(maxsplit=1)
                        self.feat_mapping[f[0]] = f[1]
                        outf.write(line + '\n')
                os.remove(path)
        if len(self.feat_mapping.keys()) != len(self.utt_speak_mapping.keys()):
            for k in self.utt_speak_mapping.keys():
                if k not in self.feat_mapping:
                    self.ignored_utterances.append(k)
            print('Some utterances were ignored due to lack of features, please see {} for more information.'.format(
                self.log_file))
            root_logger.warning(
                'The following utterances were ignored due to lack of features: {}.  See relevant logs for more information'.format(
                    ', '.join(self.ignored_utterances)))
            for k in self.ignored_utterances:
                del self.utt_speak_mapping[k]
                try:
                    del self.utt_wav_mapping[k]
                except KeyError:
                    pass
                try:
                    del self.segments[k]
                except KeyError:
                    pass
                try:
                    del self.text_mapping[k]
                except KeyError:
                    pass
            for k, v in self.speak_utt_mapping.items():
                self.speak_utt_mapping[k] = list(filter(lambda x: x in self.feat_mapping, v))

    def _calc_cmvn(self):
        spk2utt = os.path.join(self.output_directory, 'spk2utt')
        feats = os.path.join(self.output_directory, 'feats.scp')
        cmvn_directory = os.path.join(self.mfcc_directory, 'cmvn')
        os.makedirs(cmvn_directory, exist_ok=True)
        cmvn_ark = os.path.join(cmvn_directory, 'cmvn.ark')
        cmvn_scp = os.path.join(cmvn_directory, 'cmvn.scp')
        log_path = os.path.join(cmvn_directory, 'cmvn.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compute-cmvn-stats'),
                             '--spk2utt=ark:' + spk2utt,
                             'scp:' + feats, 'ark,scp:{},{}'.format(cmvn_ark, cmvn_scp)],
                            stderr=logf)
        shutil.copy(cmvn_scp, os.path.join(self.output_directory, 'cmvn.scp'))
        self.cmvn_mapping = load_scp(cmvn_scp)

    def _split_and_norm_feats(self):
        split_dir = self.split_directory
        log_dir = os.path.join(split_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'norm.log'), 'w') as logf:
            for i in range(self.num_jobs):
                path = os.path.join(split_dir, 'cmvndeltafeats.{}'.format(i))
                utt2spkpath = os.path.join(split_dir, 'utt2spk.{}'.format(i))
                cmvnpath = os.path.join(split_dir, 'cmvn.{}.scp'.format(i))
                featspath = os.path.join(split_dir, 'feats.{}.scp'.format(i))
                if not os.path.exists(path):
                    with open(path, 'wb') as outf:
                        cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                                      '--utt2spk=ark:' + utt2spkpath,
                                                      'scp:' + cmvnpath,
                                                      'scp:' + featspath,
                                                      'ark:-'], stdout=subprocess.PIPE,
                                                     stderr=logf
                                                     )
                        deltas_proc = subprocess.Popen([thirdparty_binary('add-deltas'),
                                                        'ark:-', 'ark:-'],
                                                       stdin=cmvn_proc.stdout,
                                                       stdout=outf,
                                                       stderr=logf
                                                       )
                        deltas_proc.communicate()
                    with open(path, 'rb') as inf, open(path + '_sub', 'wb') as outf:
                        subprocess.call([thirdparty_binary("subset-feats"),
                                         "--n=10", "ark:-", "ark:-"],
                                        stdin=inf, stderr=logf, stdout=outf)

    #
    def _norm_splice_feats(self):
        split_dir = self.split_directory
        log_dir = os.path.join(split_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'norm_splice.log'), 'w') as logf:
            for i in range(self.num_jobs):
                path = os.path.join(split_dir, 'cmvnsplicefeats.{}'.format(i))
                utt2spkpath = os.path.join(split_dir, 'utt2spk.{}'.format(i))
                cmvnpath = os.path.join(split_dir, 'cmvn.{}.scp'.format(i))
                featspath = os.path.join(split_dir, 'feats.{}.scp'.format(i))
                #if not os.path.exists(path):
                with open(path, 'wb') as outf:
                    cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                                  '--utt2spk=ark:' + utt2spkpath,
                                                  'scp:' + cmvnpath,
                                                  'scp:' + featspath,
                                                  'ark:-'], stdout=subprocess.PIPE,
                                                  stderr=logf
                                                  )
                    """with open(path, 'rb') as inf, open(path + '_sub', 'wb') as outf:
                        subprocess.call([thirdparty_binary("splice-feats"),
                                         '--left-context=3', '--right-context=3',
                                         'ark:-',
                                         'ark:-'], stdin=cmvn_proc.stdout,
                                         stderr=logf,
                                         stdout=outf
                                         )"""
                    splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                                         '--left-context=3', '--right-context=3',
                                                         'ark:-',

                                                         'ark:-'],
                                                         stdin=cmvn_proc.stdout,
                                                         stdout=outf,
                                                         stderr=logf)
                    splice_feats_proc.communicate()

    def _norm_splice_transform_feats(self, directory, num=0):
        split_dir = self.split_directory
        log_dir = os.path.join(split_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'norm_splice_transform.log'), 'w') as logf:
            for i in range(self.num_jobs):
                if num == 0:
                    path = os.path.join(split_dir, 'cmvnsplicetransformfeats.{}'.format(i))
                else:
                    path = os.path.join(split_dir, 'cmvnsplicetransformfeats_lda_mllt.{}'.format(i))
                utt2spkpath = os.path.join(split_dir, 'utt2spk.{}'.format(i))
                cmvnpath = os.path.join(split_dir, 'cmvn.{}.scp'.format(i))
                featspath = os.path.join(split_dir, 'feats.{}.scp'.format(i))
                #if not os.path.exists(path):
                with open(path, 'wb') as outf:
                    cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                                  '--utt2spk=ark:' + utt2spkpath,
                                                  'scp:' + cmvnpath,
                                                  'scp:' + featspath,
                                                  'ark:-'], stdout=subprocess.PIPE,
                                                  stderr=logf
                                                 )
                    splice_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                                    '--left-context=3', '--right-context=3',
                                                    'ark:-',
                                                    'ark:-'], stdin=cmvn_proc.stdout,
                                                    stderr=logf, stdout=subprocess.PIPE
                                                    )
                    """with open(path, 'rb') as inf, open(path + '_sub', 'wb') as outf:
                        transform_feats_proc = subprocess.Popen([thirdparty_binary("transform-feats"),
                                                                directory + '/{}.mat'.format(num),
                                                                'ark:-',
                                                                'ark:-'], stdin=splice_proc.stdout,
                                                                stderr=logf, stdout=outf
                                                                )
                        transform_feats_proc.communicate()"""

                    transform_feats_proc = subprocess.Popen([thirdparty_binary("transform-feats"),
                                                            directory + '/{}.mat'.format(num),
                                                            'ark:-',
                                                            'ark:-'],
                                                            stdin=splice_proc.stdout,
                                                            stderr=logf, stdout=outf
                                                            )
                    transform_feats_proc.communicate()

    #

    def get_feat_dim(self):
        directory = self.split_directory

        path = os.path.join(self.split_directory, 'cmvndeltafeats.0')
        with open(path, 'rb') as f, open(os.devnull, 'w') as devnull:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'ark,s,cs:-', '-'],
                                        stdin=f,
                                        stdout=subprocess.PIPE,
                                        stderr=devnull)
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode('utf8').strip()
        return feats

    def initialize_corpus(self, dictionary):
        root_logger = logging.getLogger()
        split_dir = self.split_directory
        self.write()
        split = False
        if not os.path.exists(split_dir):
            split = True
            root_logger.info('Setting up training data...')
            print('Setting up training data...')
            os.makedirs(split_dir)
            self._split_wavs(split_dir)
            self._split_utt2spk(split_dir)
            self._split_spk2utt(split_dir)
            self._split_texts(split_dir, dictionary)
            self._split_utt2fst(split_dir, dictionary)
        """# For testing
        split = True
        root_logger.info('Setting up training data...')
        print('Setting up training data...')
        os.makedirs(split_dir)
        self._split_wavs(split_dir)
        self._split_utt2spk(split_dir)
        self._split_spk2utt(split_dir)
        self._split_texts(split_dir, dictionary)
        self._split_utt2fst(split_dir, dictionary)"""

        self.create_mfccs()
        if split:
            self._split_feats(split_dir)
            self._split_cmvns(split_dir)
            self._split_and_norm_feats()
        #For nnet
        self._norm_splice_feats()
