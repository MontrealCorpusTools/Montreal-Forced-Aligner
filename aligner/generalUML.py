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
                with open(path, 'wb') as outf:
                    cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                                  '--utt2spk=ark:' + utt2spkpath,
                                                  'scp:' + cmvnpath,
                                                  'scp:' + featspath,
                                                  'ark:-'], stdout=subprocess.PIPE,
                                                  stderr=logf
                                                  )
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

    def initialize_corpus(self, dictionary, skip_input=True):
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
        if not skip_input and dictionary.oovs_found:
            user_input = input('There were words not found in the dictionary. Would you like to abort to fix them? (Y/N)')
            if user_input.lower() == 'y':
                    sys.exit(1)
        self.create_mfccs()
        if split:
            self._split_feats(split_dir)
            self._split_cmvns(split_dir)
            self._split_and_norm_feats()
        #For nnet
        self._norm_splice_feats()
import os

TEMP_DIR = os.path.expanduser('~/Documents/MFA')


def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class MonophoneConfig(object):
    '''
    Configuration class for monophone training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    If ``align_often`` is True in the keyword arguments, ``realign_iters`` will be::

        [1, 5, 10, 15, 20, 25, 30, 35, 38]

    Otherwise, ``realign_iters`` will be::

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform, defaults to 40
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_iter_inc : int
        Last iter to increase #Gauss on, defaults to 30
    totgauss : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to False
    do_lda_mllt : bool
        Spacifies whether to do LDA + MLLT transformation, default to True
    '''

    def __init__(self, **kwargs):
        self.num_iters = 40

        self.scale_opts = ['--transition-scale=1.0',
                           '--acoustic-scale=0.1',
                           '--self-loop-scale=0.1']
        self.beam = 10
        self.retry_beam = 40
        self.max_gauss_count = 1000
        self.boost_silence = 1.0
        if kwargs.get('align_often', False):
            self.realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14,
                                  16, 18, 20, 23, 26, 29, 32, 35, 38]
        else:
            self.realign_iters = [1, 5, 10, 15, 20, 25, 30, 35, 38]
        self.stage = -4
        self.power = 0.25

        self.do_fmllr = False
        self.do_lda_mllt = False

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def max_iter_inc(self):
        return self.num_iters - 10

    @property
    def inc_gauss_count(self):
        return int((self.max_gauss_count - self.initial_gauss_count) / self.max_iter_inc)


class TriphoneConfig(MonophoneConfig):
    '''
    Configuration class for triphone training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    If ``align_often`` is True in the keyword arguments, ``realign_iters`` will be::

        [1, 5, 10, 15, 20, 25, 30, 35, 38]

    Otherwise, ``realign_iters`` will be::

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform, defaults to 35
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_iter_inc : int
        Last iter to increase #Gauss on, defaults to 30
    totgauss : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to False
    do_lda_mllt : bool
        Spacifies whether to do LDA + MLLT transformation, default to False
    num_states : int
        Number of states in the decision tree, defaults to 3100
    num_gauss : int
        Number of gaussians in the decision tree, defaults to 50000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    '''

    def __init__(self, **kwargs):
        defaults = {'num_iters': 35,
                    'initial_gauss_count': 3100,
                    'max_gauss_count': 50000,
                    'cluster_threshold': 100,
                    'do_lda_mllt': False}
        defaults.update(kwargs)
        super(TriphoneConfig, self).__init__(**defaults)


class TriphoneFmllrConfig(TriphoneConfig):
    '''
    Configuration class for speaker-adapted triphone training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    If ``align_often`` is True in the keyword arguments, ``realign_iters`` will be::

        [1, 5, 10, 15, 20, 25, 30, 35, 38]

    Otherwise, ``realign_iters`` will be::

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]

    ``fmllr_iters`` defaults to::

        [2, 4, 6, 12]

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform, defaults to 35
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_iter_inc : int
        Last iter to increase #Gauss on, defaults to 30
    totgauss : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to True
    do_lda_mllt : bool
        Spacifies whether to do LDA + MLLT transformation, default to False
    num_states : int
        Number of states in the decision tree, defaults to 3100
    num_gauss : int
        Number of gaussians in the decision tree, defaults to 50000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to ``'full'``
    fmllr_iters : list
        List of iterations to perform fMLLR estimation
    fmllr_power : float
        Defaults to 0.2
    silence_weight : float
        Weight on silence in fMLLR estimation
    '''

    def __init__(self, align_often=True, **kwargs):
        defaults = {'do_fmllr': True,
                    'do_lda_mllt': False,
                    'fmllr_update_type': 'full',
                    'fmllr_iters': [2, 4, 6, 12],
                    'fmllr_power': 0.2,
                    'silence_weight': 0.0}
        defaults.update(kwargs)
        super(TriphoneFmllrConfig, self).__init__(**defaults)

# For nnets
class LdaMlltConfig(object):
    '''
    Configuration class for LDA + MLLT training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to False
    do_lda_mllt : bool
        Spacifies whether to do LDA + MLLT transformation, default to True
    scale_opts : list
        Options for specifying scaling in alignment
    num_gauss : int
        Number of gaussians in the decision tree, defaults to 50000
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    randprune : float
        Approximately the ratio by which we will speed up the LDA and MLLT calculations via randomized pruning
    '''
    def __init__(self, **kwargs):
        self.num_iters = 13
        self.do_fmllr = False
        self.do_lda_mllt = True

        self.scale_opts = ['--transition-scale=1.0',
                           '--acoustic-scale=0.1',
                           '--self-loop-scale=0.1']
        self.num_gauss = 5000
        self.beam = 10
        self.retry_beam = 40
        self.initial_gauss_count = 5000
        self.cluster_threshold = -1
        self.max_gauss_count = 10000
        self.boost_silence = 1.0
        self.realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.stage = -5
        self.power = 0.25

        self.dim = 40
        self.careful = False
        self.randprune = 4.0
        self.splice_opts = ['--left-context=3', '--right-context=3']
        self.cluster_thresh = -1
        self.norm_vars = False

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def max_iter_inc(self):
        return self.num_iters

    @property
    def inc_gauss_count(self):
        return int((self.max_gauss_count - self.initial_gauss_count) / self.max_iter_inc)

class DiagUbmConfig(object):
    '''
    Configuration class for diagonal UBM training

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform
    num_gselect : int
        Number of Gaussian-selection indices to use while training the model
    num_gauss : int
        Number of Gaussians after clustering down.

    '''
    def __init__(self, **kwargs):
        self.num_iters = 4
        self.num_gselect = 30
        self.num_frames = 400000
        self.num_gauss = 256

        self.num_iters_init = 20
        self.initial_gauss_proportion = 0.5
        self.subsample = 2
        self.cleanup = True
        self.min_gaussian_weight = 0.0001

        self.remove_low_count_gaussians = True
        self.num_threads = 32
        self.splice_opts = ['--left-context=3', '--right-context=3']

class iVectorExtractorConfig(object):
    '''
    Configuration class for i-vector extractor training

    Attributes
    ----------
    ivector_dim : int
        Dimension of the extracted i-vector
    ivector_period : int
        Number of frames between i-vector extractions
    num_iters : int
        Number of training iterations to perform
    num_gselect : int
        Gaussian-selection using diagonal model: number of Gaussians to select
    posterior_scale : float
        Scale on the acoustic posteriors, intended to account for inter-frame correlations
    min_post : float
        Minimum posterior to use (posteriors below this are pruned out)
    subsample : int
        Speeds up training; training on every x'th feature
    max_count : int
        The use of this option (e.g. --max-count 100) can make iVectors more consistent for different lengths of utterance, by scaling up the prior term when the data-count exceeds this value. The data-count is after posterior-scaling, so assuming the posterior-scale is 0.1, --max-count 100 starts having effect after 1000 frames, or 10 seconds of data.
    '''
    def __init__(self, **kwargs):
        self.ivector_dim = 100
        self.ivector_period = 10
        self.num_iters = 10
        self.num_gselect = 5
        self.posterior_scale = 0.1

        self.min_post = 0.025
        self.subsample = 2
        self.max_count = 0

        self.num_threads = 4
        self.num_processes = 4

        self.splice_opts = ['--left-context=3', '--right-context=3']
        self.compress = False

class NnetBasicConfig(object):
    '''
    Configuration class for neural network training

    Attributes
    ----------
    num_epochs : int
        Number of epochs of training; number of iterations is worked out from this
    iters_per_epoch : int
        Number of iterations per epoch
    realign_times : int
        How many times to realign during training; this will equally space them over the iterations
    beam : int
        Default beam width for alignment
    retry_beam : int
        Beam width to fall back on if no alignment is produced
    initial_learning_rate : float
        The initial learning rate at the beginning of training
    final_learning_rate : float
        The final learning rate by the end of training
    pnorm_input_dim : int
        The input dimension of the pnorm component
    pnorm_output_dim : int
        The output dimension of the pnorm component
    p : int
        Pnorm parameter
    hidden_layer_dim : int
        Dimension of a hidden layer
    samples_per_iter : int
        Number of samples seen per job per each iteration; used when getting examples
    shuffle_buffer_size : int
        This "buffer_size" variable controls randomization of the samples on each iter.  You could set it to 0 or to a large value for complete randomization, but this would both consume memory and cause spikes in disk I/O.  Smaller is easier on disk and memory but less random.  It's not a huge deal though, as samples are anyway randomized right at the start. (the point of this is to get data in different minibatches on different iterations, since in the preconditioning method, 2 samples in the same minibatch can affect each others' gradients.
    add_layers_period : int
        Number of iterations between addition of a new layer
    num_hidden_layers : int
        Number of hidden layers
    randprune : float
        Speeds up LDA
    alpha : float
        Relates to preconditioning
    mix_up : int
        Number of components to mix up to
    prior_subset_size : int
        Number of samples per job for computing priors
    update_period : int
        How often the preconditioning subspace is updated
    num_samples_history : int
        Relates to online preconditioning
    preconditioning_rank_in : int
        Relates to online preconditioning
    preconditioning_rank_out : int
        Relates to online preconditioning

    '''
    def __init__(self, **kwargs):
        self.num_epochs = 4
        self.num_epochs_extra = 5
        self.num_iters_final = 20
        self.iters_per_epoch = 2
        self.realign_times = 0

        self.beam = 10
        self.retry_beam = 15000000

        self.initial_learning_rate=0.32
        self.final_learning_rate=0.032
        self.bias_stddev = 0.5

        self.pnorm_input_dim = 3000
        self.pnorm_output_dim = 300
        self.p = 2

        self.shrink_interval = 5
        self.shrink = True
        self.num_frames_shrink = 2000

        self.final_learning_rate_factor = 0.5
        self.hidden_layer_dim = 50

        self.samples_per_iter = 200000
        self.shuffle_buffer_size = 5000
        self.add_layers_period = 2
        self.num_hidden_layers = 3
        self.modify_learning_rates = False

        self.last_layer_factor = 0.1
        self.first_layer_factor = 1.0

        self.splice_width = 3
        self.randprune = 4.0
        self.alpha = 4.0
        self.max_change = 10.0
        self.mix_up = 12000 # From run_nnet2.sh
        self.prior_subset_size = 10000
        self.boost_silence = 0.5

        self.update_period = 4
        self.num_samples_history = 2000
        self.max_change_per_sample = 0.075
        self.precondition_rank_in = 20
        self.precondition_rank_out = 80

class MfccConfig(object):
    '''
    Configuration class for MFCC generation

    The ``config_dict`` currently stores one key ``'use-energy'`` which
    defaults to False

    Parameters
    ----------
    output_directory : str
        Path to directory to save configuration files for Kaldi
    kwargs : dict, optional
        If specified, updates ``config_dict`` with this dictionary

    Attributes
    ----------
    config_dict : dict
        Dictionary of configuration parameters
    '''

    def __init__(self, output_directory, job=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.job = job
        self.config_dict = {'use-energy': False, 'frame-shift': 10}
        self.config_dict.update(kwargs)
        self.output_directory = output_directory
        self.write()

    def update(self, kwargs):
        '''
        Update configuration dictionary with new dictionary

        Parameters
        ----------
        kwargs : dict
            Dictionary of new parameter values
        '''
        self.config_dict.update(kwargs)
        self.write()

    @property
    def config_directory(self):
        path = os.path.join(self.output_directory, 'config')
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def path(self):
        if self.job is None:
            f = 'mfcc.conf'
        else:
            f = 'mfcc.{}.conf'.format(self.job)
        return os.path.join(self.config_directory, f)

    def write(self):
        '''
        Write configuration dictionary to a file for use in Kaldi binaries
        '''
        with open(self.path, 'w', encoding='utf8') as f:
            for k, v in self.config_dict.items():
                f.write('--{}={}\n'.format(k, make_safe(v)))
import os
import math
import subprocess
import re
from collections import defaultdict, Counter

from .helper import thirdparty_binary
from .exceptions import DictionaryPathError, DictionaryFileError, DictionaryError


def compile_graphemes(graphemes):
    if '-' in graphemes:
        base = r'^\W*([-{}]+)\W*'
    else:
        base = r'^\W*([{}]+)\W*'
    string = ''.join(x for x in graphemes if x != '-')
    try:
        return re.compile(base.format(string))
    except Exception:
        print(graphemes)
        raise


brackets = [('[', ']'), ('{', '}'), ('<', '>')]


def sanitize(item):
    if not item:
        return item
    for b in brackets:
        if item[0] == b[0] and item[-1] == b[1]:
            return item
    # Clitic markers are "-" and "'"
    sanitized = re.sub(r"^[^-\w']+", '', item)
    sanitized = re.sub(r"[^-\w']+$", '', sanitized)
    return sanitized


def sanitize_clitics(item):
    if not item:
        return item
    for b in brackets:
        if item[0] == b[0] and item[-1] == b[1]:
            return item
    # Clitic markers are "-" and "'"
    sanitized = re.sub(r"^\W+", '', item)
    sanitized = re.sub(r"\W+$", '', sanitized)
    return sanitized


class Dictionary(object):
    """
    Class containing information about a pronunciation dictionary

    Parameters
    ----------
    input_path : str
        Path to an input pronunciation dictionary
    output_directory : str
        Path to a directory to store files for Kaldi
    oov_code : str, optional
        What to label words not in the dictionary, defaults to ``'<unk>'``
    position_dependent_phones : bool, optional
        Specifies whether phones should be represented as dependent on their
        position in the word (beginning, middle or end), defaults to True
    num_sil_states : int, optional
        Number of states to use for silence phones, defaults to 5
    num_nonsil_states : int, optional
        Number of states to use for non-silence phones, defaults to 3
    shared_silence_phones : bool, optional
        Specify whether to share states across all silence phones, defaults
        to True
    pronunciation probabilities : bool, optional
        Specifies whether to model different pronunciation probabilities
        or to treat each entry as a separate word, defaults to True
    sil_prob : float, optional
        Probability of optional silences following words, defaults to 0.5
    """

    topo_template = '<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>'
    topo_sil_template = '<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>'
    topo_transition_template = '<Transition> {} {}'
    positions = ["_B", "_E", "_I", "_S"]
    clitic_markers = ["'", '-']

    def __init__(self, input_path, output_directory, oov_code='<unk>',
                 position_dependent_phones=True, num_sil_states=5,
                 num_nonsil_states=3, shared_silence_phones=True,
                 sil_prob=0.5, word_set=None, debug=False):
        if not os.path.exists(input_path):
            raise (DictionaryPathError(input_path))
        if not os.path.isfile(input_path):
            raise (DictionaryFileError(input_path))
        self.input_path = input_path
        self.debug = debug
        self.output_directory = os.path.join(output_directory, 'dictionary')
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.position_dependent_phones = position_dependent_phones

        self.words = defaultdict(list)
        self.nonsil_phones = set()
        self.sil_phones = {'sp', 'spn', 'sil'}
        self.optional_silence = 'sp'
        self.nonoptional_silence = 'sil'
        self.graphemes = set()
        if word_set is not None:
            word_set = {sanitize(x) for x in word_set}
        self.words['!sil'].append((('sp',), 1))
        self.words[self.oov_code].append((('spn',), 1))
        self.pronunciation_probabilities = True
        with open(input_path, 'r', encoding='utf8') as inf:
            for i, line in enumerate(inf):
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                word = line.pop(0).lower()
                if not line:
                    raise DictionaryError('Line {} of {} does not have a pronunciation.'.format(i, input_path))
                if word in ['!sil', oov_code]:
                    continue
                if word_set is not None and sanitize(word) not in word_set:
                    continue
                self.graphemes.update(word)
                try:
                    prob = float(line[0])
                    line = line[1:]
                except ValueError:
                    prob = None
                    self.pronunciation_probabilities = False
                pron = tuple(line)
                if not any(x in self.sil_phones for x in pron):
                    self.nonsil_phones.update(pron)
                if word in self.words and pron in set(x[0] for x in self.words[word]):
                    continue
                self.words[word].append((pron, prob))
        self.word_pattern = compile_graphemes(self.graphemes)
        self.phone_mapping = {}
        self.words_mapping = {}

    def generate_mappings(self):
        self.phone_mapping = {}
        i = 0
        self.phone_mapping['<eps>'] = i
        if self.position_dependent_phones:
            for p in self.positional_sil_phones:
                i += 1
                self.phone_mapping[p] = i
            for p in self.positional_nonsil_phones:
                i += 1
                self.phone_mapping[p] = i
        else:
            for p in sorted(self.sil_phones):
                i += 1
                self.phone_mapping[p] = i
            for p in sorted(self.nonsil_phones):
                i += 1
                self.phone_mapping[p] = i

        self.words_mapping = {}
        i = 0
        self.words_mapping['<eps>'] = i
        for w in sorted(self.words.keys()):
            i += 1
            self.words_mapping[w] = i

        self.words_mapping['#0'] = i + 1
        self.words_mapping['<s>'] = i + 2
        self.words_mapping['</s>'] = i + 3

        self.oovs_found = set()
        self.add_disambiguation()

    def add_disambiguation(self):
        subsequences = set()
        pronunciation_counts = defaultdict(int)

        for w, prons in self.words.items():
            for p in prons:
                pronunciation_counts[p[0]] += 1
                pron = [x for x in p[0]][:-1]
                while pron:
                    subsequences.add(tuple(p))
                    pron = pron[:-1]
        last_used = defaultdict(int)
        for w, prons in sorted(self.words.items()):
            new_prons = []
            for p in prons:
                if pronunciation_counts[p[0]] == 1 and not p[0] in subsequences:
                    disambig = None
                else:
                    pron = p[0]
                    last_used[pron] += 1
                    disambig = last_used[pron]
                new_prons.append((p[0], p[1], disambig))
            self.words[w] = new_prons
        if last_used:
            self.max_disambig = max(last_used.values())
        else:
            self.max_disambig = 0
        self.disambig = set('#{}'.format(x) for x in range(self.max_disambig + 1))
        i = max(self.phone_mapping.values())
        for p in sorted(self.disambig):
            i += 1
            self.phone_mapping[p] = i

    def create_utterance_fst(self, text, frequent_words):
        num_words = len(text)
        word_probs = Counter(text)
        word_probs = {k: v / num_words for k, v in word_probs.items()}
        word_probs.update(frequent_words)
        text = ''
        for k, v in word_probs.items():
            cost = -1 * math.log(v)
            text += '0 0 {w} {w} {cost}\n'.format(w=self.to_int(k), cost=cost)
        text += '0 {}\n'.format(-1 * math.log(1 / num_words))
        return text

    def to_int(self, item):
        """
        Convert a given word into its integer id
        """
        if item == '':
            return None
        item = self._lookup(item)
        if item not in self.words_mapping:
            self.oovs_found.add(item)
            return self.oov_int
        return self.words_mapping[item]

    def save_oovs_found(self, directory):
        """
        Save all out of vocabulary items to a file in the specified directory

        Parameters
        ----------
        directory : str
            Path to directory to save ``oovs_found.txt``
        """
        with open(os.path.join(directory, 'oovs_found.txt'), 'w', encoding='utf8') as f:
            for oov in sorted(self.oovs_found):
                f.write(oov + '\n')
        self.oovs_found = set()

    def _lookup(self, item):
        if item in self.words_mapping:
            return item
        sanitized = sanitize(item)
        if sanitized in self.words_mapping:
            return sanitized
        sanitized = sanitize_clitics(item)
        if sanitized in self.words_mapping:
            return sanitized
        return item

    def separate_clitics(self, item):
        """Separates words with apostrophes or hyphens if the subparts are in the lexicon.

        Checks whether the text on either side of an apostrophe or hyphen is in the dictionary. If so,
        splits the word. If neither part is in the dictionary, returns the word without splitting it.

        Parameters
        ----------
        item : string
            Lexical item

        Returns
        -------
        vocab_items: list
            List containing all words after any splits due to apostrophes or hyphens

        """
        unit_re = re.compile(r'^(\[.*\]|\{.*\}|<.*>)$')
        if unit_re.match(item) is not None:
            return [item]
        lookup = self._lookup(item)

        if lookup not in self.words_mapping:
            item = sanitize(item)
            vocab = []
            chars = list(item)
            count = 0
            for i in chars:
                if i in self.clitic_markers:
                    count += 1
            for i in range(count):
                for punc in chars:
                    if punc in self.clitic_markers:
                        idx = chars.index(punc)
                        option1withpunc = ''.join(chars[:idx + 1])
                        option1nopunc = ''.join(chars[:idx])
                        option2withpunc = ''.join(chars[idx:])
                        option2nopunc = ''.join(chars[idx + 1:])
                        if option1withpunc in self.words:
                            vocab.append(option1withpunc)
                            if option2nopunc in self.words:
                                vocab.append(option2nopunc)
                            elif all(x not in list(option2nopunc) for x in self.clitic_markers):
                                vocab.append(option2nopunc)
                        else:
                            vocab.append(option1nopunc)
                            if option2withpunc in self.words:
                                vocab.append(option2withpunc)
                            elif option2nopunc in self.words:
                                vocab.append(option2nopunc)
                            elif all(x not in list(option2nopunc) for x in self.clitic_markers):
                                vocab.append(option2nopunc)
                        chars = list(option2nopunc)
        else:
            return [lookup]
        if not vocab:
            return [lookup]
        else:
            unk = []
            for i in vocab:
                if i not in self.words:
                    unk.append(i)
            if len(unk) == count + 1:
                return [lookup]
            return vocab

    @property
    def reversed_word_mapping(self):
        """
        A mapping of integer ids to words
        """
        mapping = {}
        for k, v in self.words_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def reversed_phone_mapping(self):
        """
        A mapping of integer ids to phones
        """
        mapping = {}
        for k, v in self.phone_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def oov_int(self):
        """
        The integer id for out of vocabulary items
        """
        return self.words_mapping[self.oov_code]

    @property
    def positional_sil_phones(self):
        """
        List of silence phones with positions
        """
        sil_phones = []
        for p in sorted(self.sil_phones):
            sil_phones.append(p)
            for pos in self.positions:
                sil_phones.append(p + pos)
        return sil_phones

    @property
    def positional_nonsil_phones(self):
        """
        List of non-silence phones with positions
        """
        nonsil_phones = []
        for p in sorted(self.nonsil_phones):
            for pos in self.positions:
                nonsil_phones.append(p + pos)
        return nonsil_phones

    @property
    def optional_silence_csl(self):
        """
        Phone id of the optional silence phone
        """
        return '{}'.format(self.phone_mapping[self.optional_silence])

    @property
    def silence_csl(self):
        """
        A colon-separated list (as a string) of silence phone ids
        """
        if self.position_dependent_phones:
            return ':'.join(map(str, (self.phone_mapping[x] for x in self.positional_sil_phones)))
        else:
            return ':'.join(map(str, (self.phone_mapping[x] for x in self.sil_phones)))

    @property
    def phones_dir(self):
        """
        Directory to store information Kaldi needs about phones
        """
        return os.path.join(self.output_directory, 'phones')

    @property
    def phones(self):
        """
        The set of all phones (silence and non-silence)
        """
        return self.sil_phones | self.nonsil_phones

    def write(self):
        """
        Write the files necessary for Kaldi
        """
        print('Creating dictionary information...')
        os.makedirs(self.phones_dir, exist_ok=True)
        self.generate_mappings()
        self._write_graphemes()
        self._write_phone_map_file()
        self._write_phone_sets()
        self._write_phone_symbol_table()
        self._write_disambig()
        self._write_topo()
        self._write_word_boundaries()
        self._write_extra_questions()
        self._write_word_file()
        self._write_fst_text()
        self._write_fst_text(disambig=True)
        self._write_fst_binary()
        self._write_fst_binary(disambig=True)
        # self.cleanup()

    def cleanup(self):
        """
        Clean up temporary files in the output directory
        """
        os.remove(os.path.join(self.output_directory, 'temp.fst'))
        os.remove(os.path.join(self.output_directory, 'lexicon.text.fst'))

    def _write_graphemes(self):
        outfile = os.path.join(self.output_directory, 'graphemes.txt')
        with open(outfile, 'w', encoding='utf8') as f:
            for char in sorted(self.graphemes):
                f.write(char + '\n')

    def export_lexicon(self, path, disambig=False, probability=False):
        with open(path, 'w', encoding='utf8') as f:
            for w in sorted(self.words.keys()):
                for p in sorted(self.words[w]):
                    phones = ' '.join(p[0])
                    if disambig and p[2] is not None:
                        phones += ' #{}'.format(p[2])
                    if probability:
                        f.write('{}\t{}\t{}\n'.format(w, p[1], phones))
                    else:
                        f.write('{}\t{}\n'.format(w, phones))

    def _write_phone_map_file(self):
        outfile = os.path.join(self.output_directory, 'phone_map.txt')
        with open(outfile, 'w', encoding='utf8') as f:
            for sp in self.sil_phones:
                if self.position_dependent_phones:
                    new_phones = [sp + x for x in ['', ''] + self.positions]
                else:
                    new_phones = [sp]
                f.write(' '.join(new_phones) + '\n')
            for nsp in self.nonsil_phones:
                if self.position_dependent_phones:
                    new_phones = [nsp + x for x in [''] + self.positions]
                else:
                    new_phones = [nsp]
                f.write(' '.join(new_phones) + '\n')

    def _write_phone_symbol_table(self):
        outfile = os.path.join(self.output_directory, 'phones.txt')
        with open(outfile, 'w', encoding='utf8') as f:
            for p, i in sorted(self.phone_mapping.items(), key=lambda x: x[1]):
                f.write('{} {}\n'.format(p, i))

    def _write_word_boundaries(self):
        boundary_path = os.path.join(self.output_directory, 'phones', 'word_boundary.txt')
        boundary_int_path = os.path.join(self.output_directory, 'phones', 'word_boundary.int')
        with open(boundary_path, 'w', encoding='utf8') as f, \
                open(boundary_int_path, 'w', encoding='utf8') as intf:
            if self.position_dependent_phones:
                for p in sorted(self.phone_mapping.keys(), key=lambda x: self.phone_mapping[x]):
                    if p == '<eps>':
                        continue
                    cat = 'nonword'
                    if p.endswith('_B'):
                        cat = 'begin'
                    elif p.endswith('_S'):
                        cat = 'singleton'
                    elif p.endswith('_I'):
                        cat = 'internal'
                    elif p.endswith('_E'):
                        cat = 'end'
                    f.write(' '.join([p, cat]) + '\n')
                    intf.write(' '.join([str(self.phone_mapping[p]), cat]) + '\n')

    def _write_word_file(self):
        words_path = os.path.join(self.output_directory, 'words.txt')

        with open(words_path, 'w', encoding='utf8') as f:
            for w, i in sorted(self.words_mapping.items(), key=lambda x: x[1]):
                f.write('{} {}\n'.format(w, i))

    def _write_topo(self):
        filepath = os.path.join(self.output_directory, 'topo')
        sil_transp = 1 / (self.num_sil_states - 1)
        initial_transition = [self.topo_transition_template.format(x, sil_transp)
                              for x in range(self.num_sil_states - 1)]
        middle_transition = [self.topo_transition_template.format(x, sil_transp)
                             for x in range(1, self.num_sil_states)]
        final_transition = [self.topo_transition_template.format(self.num_sil_states - 1, 0.75),
                            self.topo_transition_template.format(self.num_sil_states, 0.25)]
        with open(filepath, 'w') as f:
            f.write('<Topology>\n')
            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            if self.position_dependent_phones:
                phones = self.positional_nonsil_phones
            else:
                phones = sorted(self.nonsil_phones)
            f.write("{}\n".format(' '.join(str(self.phone_mapping[x]) for x in phones)))
            f.write("</ForPhones>\n")
            states = [self.topo_template.format(cur_state=x, next_state=x + 1)
                      for x in range(self.num_nonsil_states)]
            f.write('\n'.join(states))
            f.write("\n<State> {} </State>\n".format(self.num_nonsil_states))
            f.write("</TopologyEntry>\n")

            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            if self.position_dependent_phones:
                phones = self.positional_sil_phones
            else:
                phones = self.sil_phones
            f.write("{}\n".format(' '.join(str(self.phone_mapping[x]) for x in phones)))
            f.write("</ForPhones>\n")
            states = []
            for i in range(self.num_sil_states):
                if i == 0:
                    transition = ' '.join(initial_transition)
                elif i == self.num_sil_states - 1:
                    transition = ' '.join(final_transition)
                else:
                    transition = ' '.join(middle_transition)
                states.append(self.topo_sil_template.format(cur_state=i, transitions=transition))
            f.write('\n'.join(states))
            f.write("\n<State> {} </State>\n".format(self.num_sil_states))
            f.write("</TopologyEntry>\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self):
        sharesplit = ['shared', 'split']
        if not self.shared_silence_phones:
            sil_sharesplit = ['not-shared', 'not-split']
        else:
            sil_sharesplit = sharesplit

        sets_file = os.path.join(self.output_directory, 'phones', 'sets.txt')
        roots_file = os.path.join(self.output_directory, 'phones', 'roots.txt')

        sets_int_file = os.path.join(self.output_directory, 'phones', 'sets.int')
        roots_int_file = os.path.join(self.output_directory, 'phones', 'roots.int')

        with open(sets_file, 'w', encoding='utf8') as setf, \
                open(roots_file, 'w', encoding='utf8') as rootf, \
                open(sets_int_file, 'w', encoding='utf8') as setintf, \
                open(roots_int_file, 'w', encoding='utf8') as rootintf:

            # process silence phones
            for i, sp in enumerate(self.sil_phones):
                if self.position_dependent_phones:
                    mapped = [sp + x for x in [''] + self.positions]
                else:
                    mapped = [sp]
                setf.write(' '.join(mapped) + '\n')
                setintf.write(' '.join(map(str, (self.phone_mapping[x] for x in mapped))) + '\n')
                if i == 0:
                    line = sil_sharesplit + mapped
                    lineint = sil_sharesplit + [self.phone_mapping[x] for x in mapped]
                else:
                    line = sharesplit + mapped
                    lineint = sharesplit + [self.phone_mapping[x] for x in mapped]
                rootf.write(' '.join(line) + '\n')
                rootintf.write(' '.join(map(str, lineint)) + '\n')

            # process nonsilence phones
            for nsp in sorted(self.nonsil_phones):
                if self.position_dependent_phones:
                    mapped = [nsp + x for x in self.positions]
                else:
                    mapped = [nsp]
                setf.write(' '.join(mapped) + '\n')
                setintf.write(' '.join(map(str, (self.phone_mapping[x] for x in mapped))) + '\n')
                line = sharesplit + mapped
                lineint = sharesplit + [self.phone_mapping[x] for x in mapped]
                rootf.write(' '.join(line) + '\n')
                rootintf.write(' '.join(map(str, lineint)) + '\n')

    def _write_extra_questions(self):
        phone_extra = os.path.join(self.phones_dir, 'extra_questions.txt')
        phone_extra_int = os.path.join(self.phones_dir, 'extra_questions.int')
        with open(phone_extra, 'w', encoding='utf8') as outf, \
                open(phone_extra_int, 'w', encoding='utf8') as intf:
            if self.position_dependent_phones:
                sils = sorted(self.positional_sil_phones)
            else:
                sils = sorted(self.sil_phones)
            outf.write(' '.join(sils) + '\n')
            intf.write(' '.join(map(str, (self.phone_mapping[x] for x in sils))) + '\n')

            if self.position_dependent_phones:
                nonsils = sorted(self.positional_nonsil_phones)
            else:
                nonsils = sorted(self.nonsil_phones)
            outf.write(' '.join(nonsils) + '\n')
            intf.write(' '.join(map(str, (self.phone_mapping[x] for x in nonsils))) + '\n')
            if self.position_dependent_phones:
                for p in self.positions:
                    line = [x + p for x in sorted(self.nonsil_phones)]
                    outf.write(' '.join(line) + '\n')
                    intf.write(' '.join(map(str, (self.phone_mapping[x] for x in line))) + '\n')
                for p in [''] + self.positions:
                    line = [x + p for x in sorted(self.sil_phones)]
                    outf.write(' '.join(line) + '\n')
                    intf.write(' '.join(map(str, (self.phone_mapping[x] for x in line))) + '\n')

    def _write_disambig(self):
        disambig = os.path.join(self.phones_dir, 'disambig.txt')
        disambig_int = os.path.join(self.phones_dir, 'disambig.int')
        with open(disambig, 'w', encoding='utf8') as outf, \
                open(disambig_int, 'w', encoding='utf8') as intf:
            for d in sorted(self.disambig):
                outf.write('{}\n'.format(d))
                intf.write('{}\n'.format(self.phone_mapping[d]))

    def _write_fst_binary(self, disambig=False):
        if disambig:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon_disambig.text.fst')
            output_fst = os.path.join(self.output_directory, 'L_disambig.fst')
        else:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon.text.fst')
            output_fst = os.path.join(self.output_directory, 'L.fst')

        phones_file_path = os.path.join(self.output_directory, 'phones.txt')
        words_file_path = os.path.join(self.output_directory, 'words.txt')

        log_path = os.path.join(self.output_directory, 'fst.log')
        temp_fst_path = os.path.join(self.output_directory, 'temp.fst')
        subprocess.call([thirdparty_binary('fstcompile'), '--isymbols={}'.format(phones_file_path),
                         '--osymbols={}'.format(words_file_path),
                         '--keep_isymbols=false', '--keep_osymbols=false',
                         lexicon_fst_path, temp_fst_path])

        subprocess.call([thirdparty_binary('fstarcsort'), '--sort_type=olabel',
                         temp_fst_path, output_fst])
        if self.debug:
            dot_path = os.path.join(self.output_directory, 'L.dot')
            with open(log_path, 'w') as logf:
                draw_proc = subprocess.Popen([thirdparty_binary('fstdraw'), '--portrait=true',
                                              '--isymbols={}'.format(phones_file_path),
                                              '--osymbols={}'.format(words_file_path), output_fst, dot_path],
                                             stderr=logf)
                draw_proc.communicate()
                dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-O', dot_path], stderr=logf)
                dot_proc.communicate()

    def _write_fst_text(self, disambig=False):
        if disambig:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon_disambig.text.fst')
        else:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon.text.fst')
        if self.sil_prob != 0:
            silphone = self.optional_silence
            nonoptsil = self.nonoptional_silence

            def is_sil(element):
                return element in [silphone, silphone + '_S']

            silcost = -1 * math.log(self.sil_prob)
            nosilcost = -1 * math.log(1.0 - self.sil_prob)
            startstate = 0
            loopstate = 1
            silstate = 2
        else:
            loopstate = 0
            nextstate = 1

        with open(lexicon_fst_path, 'w', encoding='utf8') as outf:
            if self.sil_prob != 0:
                outf.write('\t'.join(map(str, [startstate, loopstate, '<eps>', '<eps>', nosilcost])) + '\n')

                outf.write('\t'.join(map(str, [startstate, loopstate, nonoptsil, '<eps>', silcost])) + "\n")
                outf.write('\t'.join(map(str, [silstate, loopstate, silphone, '<eps>'])) + "\n")
                nextstate = 3
            for w in sorted(self.words.keys()):
                for phones, prob, disambig_symbol in sorted(self.words[w]):
                    phones = [x for x in phones]
                    if self.position_dependent_phones:
                        if len(phones) == 1:
                            phones[0] += '_S'
                        else:
                            for i in range(len(phones)):
                                if i == 0:
                                    phones[i] += '_B'
                                elif i == len(phones) - 1:
                                    phones[i] += '_E'
                                else:
                                    phones[i] += '_I'
                    if not self.pronunciation_probabilities:
                        pron_cost = 0
                    else:
                        if prob is None:
                            prob = 1.0
                        pron_cost = -1 * math.log(prob)

                    pron_cost_string = ''
                    if pron_cost != 0:
                        pron_cost_string = '\t{}'.format(pron_cost)

                    s = loopstate
                    word_or_eps = w
                    local_nosilcost = nosilcost + pron_cost
                    local_silcost = silcost + pron_cost
                    while len(phones) > 0:
                        p = phones.pop(0)
                        if len(phones) > 0 or (disambig and disambig_symbol is not None):
                            ns = nextstate
                            nextstate += 1
                            outf.write('\t'.join(map(str, [s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                            word_or_eps = '<eps>'
                            pron_cost_string = ""
                            pron_cost = 0.0
                            s = ns
                        elif self.sil_prob == 0:
                            ns = loopstate
                            outf.write('\t'.join(map(str, [s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                            word_or_eps = '<eps>'
                            pron_cost_string = ""
                            s = ns
                        else:
                            outf.write('\t'.join(map(str, [s, loopstate, p, word_or_eps, local_nosilcost])) + "\n")
                            outf.write('\t'.join(map(str, [s, silstate, p, word_or_eps, local_silcost])) + "\n")
                    if disambig and disambig_symbol is not None:
                        outf.write('\t'.join(map(str, [s, loopstate, '#{}'.format(disambig_symbol), word_or_eps,
                                                       local_nosilcost])) + "\n")
                        outf.write('\t'.join(
                            map(str, [s, silstate, '#{}'.format(disambig_symbol), word_or_eps, local_silcost])) + "\n")

            outf.write("{}\t{}\n".format(loopstate, 0))


class OrthographicDictionary(Dictionary):
    def __init__(self, input_dict, output_directory, oov_code='<unk>',
                 position_dependent_phones=True, num_sil_states=5,
                 num_nonsil_states=3, shared_silence_phones=False,
                 pronunciation_probabilities=True,
                 sil_prob=0.5, debug=False):
        self.debug = debug
        self.output_directory = os.path.join(output_directory, 'dictionary')
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.position_dependent_phones = position_dependent_phones
        self.pronunciation_probabilities = pronunciation_probabilities

        self.words = defaultdict(list)
        self.nonsil_phones = set()
        self.sil_phones = {'sp', 'spn', 'sil'}
        self.optional_silence = 'sp'
        self.nonoptional_silence = 'sil'
        self.graphemes = set()
        for w in input_dict:
            self.graphemes.update(w)
            pron = tuple(input_dict[w])
            self.words[w].append((pron, None))
            self.nonsil_phones.update(pron)
        self.word_pattern = compile_graphemes(self.graphemes)
        self.words['!SIL'].append((('sil',), None))
        self.words[self.oov_code].append((('spn',), None))
        self.phone_mapping = {}
        i = 0
        self.phone_mapping['<eps>'] = i
        if self.position_dependent_phones:
            for p in self.positional_sil_phones:
                i += 1
                self.phone_mapping[p] = i
            for p in self.positional_nonsil_phones:
                i += 1
                self.phone_mapping[p] = i
        else:
            for p in sorted(self.sil_phones):
                i += 1
                self.phone_mapping[p] = i
            for p in sorted(self.nonsil_phones):
                i += 1
                self.phone_mapping[p] = i

        self.words_mapping = {}
        i = 0
        self.words_mapping['<eps>'] = i
        for w in sorted(self.words.keys()):
            i += 1
            self.words_mapping[w] = i

        self.words_mapping['#0'] = i + 1
        self.words_mapping['<s>'] = i + 2
        self.words_mapping['</s>'] = i + 3

        self.oovs_found = set()
        self.add_disambiguation()
import os
import pickle
import yaml
import glob

from tempfile import mkdtemp
from shutil import copy, copyfile, rmtree, make_archive, unpack_archive

# default format for output
FORMAT = "zip"

from . import __version__
from .exceptions import PronunciationAcousticMismatchError, PronunciationOrthographyMismatchError


class Archive(object):
    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Largely duplicated from the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.
    """

    def __init__(self, source, is_tmpdir=False):
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
            self.is_tmpdir = is_tmpdir  # trust caller
        else:
            base = mkdtemp(dir=os.environ.get("TMPDIR", None))
            unpack_archive(source, base)
            (head, tail, _) = next(os.walk(base))
            if not tail:
                raise ValueError("'{}' is empty.".format(source))
            name = tail[0]
            if len(tail) > 1:
                if tail[0] != '__MACOSX':   # Zipping from Mac adds a directory
                    raise ValueError("'{}' is a bomb.".format(source))
                else:
                    name = tail[1]
            self.dirname = os.path.join(head, name)
            self.is_tmpdir = True  # ignore caller

    @classmethod
    def empty(cls, head):
        """
        Initialize an archive using an empty directory
        """
        base = mkdtemp(dir=os.environ.get("TMPDIR", None))
        source = os.path.join(base, head)
        os.makedirs(source, exist_ok=True)
        return cls(source, True)

    def add(self, source):
        """
        Add file into archive
        """
        copy(source, self.dirname)

    def __repr__(self):
        return "{}(dirname={!r})".format(self.__class__.__name__,
                                         self.dirname)

    def dump(self, sink, archive_fmt=FORMAT):
        """
        Write archive to disk, and return the name of final archive
        """
        return make_archive(sink, archive_fmt,
                            *os.path.split(self.dirname))

    def __del__(self):
        if self.is_tmpdir:
            rmtree(self.dirname)


class AcousticModel(Archive):
    def add_meta_file(self, aligner):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            yaml.dump(aligner.meta, f)

    @property
    def meta(self):
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'gmm-hmm'}
            else:
                with open(meta_path, 'r', encoding='utf8') as f:
                    self._meta = yaml.load(f)
            self._meta['phones'] = set(self._meta.get('phones', []))
        return self._meta

    def add_triphone_model(self, source):
        """
        Add file into archive
        """
        copyfile(os.path.join(source, 'final.mdl'), os.path.join(self.dirname, 'ali-final.mdl'))
        copyfile(os.path.join(source, 'final.occs'), os.path.join(self.dirname, 'ali-final.occs'))
        copyfile(os.path.join(source, 'tree'), os.path.join(self.dirname, 'ali-tree'))

    def add_triphone_fmllr_model(self, source):
        """
        Add file into archive
        """
        copy(os.path.join(source, 'final.mdl'), self.dirname)
        copy(os.path.join(source, 'final.occs'), self.dirname)
        copy(os.path.join(source, 'tree'), self.dirname)

    def add_nnet_model(self, source):
        """
        Add file into archive
        """
        copy(os.path.join(source, 'final.mdl'), self.dirname)
        copy(os.path.join(source, 'tree'), self.dirname)
        for file in glob.glob(os.path.join(source, 'alignfeats.*')):
            copy(os.path.join(source, file), self.dirname)
        for file in glob.glob(os.path.join(source, 'fsts.*')):
            copy(os.path.join(source, file), self.dirname)

    def export_triphone_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        copyfile(os.path.join(self.dirname, 'final.mdl'), os.path.join(destination, 'final.mdl'))
        copyfile(os.path.join(self.dirname, 'final.occs'), os.path.join(destination, 'final.occs'))
        copyfile(os.path.join(self.dirname, 'tree'), os.path.join(destination, 'tree'))

    def export_triphone_fmllr_model(self, destination):
        """
        """
        os.makedirs(destination, exist_ok=True)
        copy(os.path.join(self.dirname, 'final.mdl'), destination)
        copy(os.path.join(self.dirname, 'final.occs'), destination)
        copy(os.path.join(self.dirname, 'tree'), destination)

    def export_nnet_model(self, destination):
        os.makedirs(destination, exist_ok=True)
        copy(os.path.join(self.dirname, 'final.mdl'), destination)
        copy(os.path.join(self.dirname, 'tree'), destination)
        for file in glob.glob(os.path.join(self.dirname, 'fsts.*')):
            copy(os.path.join(self.dirname, file), destination)

    def validate(self, dictionary):
        if isinstance(dictionary, G2PModel):
            missing_phones = dictionary.meta['phones'] - set(self.meta['phones'])
        else:
            missing_phones = dictionary.nonsil_phones - set(self.meta['phones'])
        if missing_phones:
            #print('dictionary phones: {}'.format(dictionary.meta['phones']))
            print('dictionary phones: {}'.format(dictionary.nonsil_phones))
            print('model phones: {}'.format(self.meta['phones']))
            raise (PronunciationAcousticMismatchError(missing_phones))


class G2PModel(Archive):
    def add_meta_file(self, dictionary):
        with open(os.path.join(self.dirname, 'meta.yaml'), 'w', encoding='utf8') as f:
            meta = {'phones': sorted(dictionary.nonsil_phones),
                    'graphemes': sorted(dictionary.graphemes),
                    'architecture': 'phonetisaurus',
                    'version': __version__}
            yaml.dump(meta, f)

    @property
    def meta(self):
        if not self._meta:
            meta_path = os.path.join(self.dirname, 'meta.yaml')
            if not os.path.exists(meta_path):
                self._meta = {'version': '0.9.0',
                              'architecture': 'phonetisaurus'}
            else:
                with open(meta_path, 'r', encoding='utf8') as f:
                    self._meta = yaml.load(f)
            self._meta['phones'] = set(self._meta.get('phones', []))
            self._meta['graphemes'] = set(self._meta.get('graphemes', []))
        return self._meta

    @property
    def fst_path(self):
        return os.path.join(self.dirname, 'model.fst')

    def add_fst_model(self, source):
        """
        Add file into archive
        """
        copyfile(os.path.join(source, 'model.fst'), self.fst_path)

    def export_fst_model(self, destination):
        os.makedirs(destination, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, corpus):
        return True  # FIXME add actual validation

class IvectorExtractor(Archive):
    '''
    Archive for i-vector extractors (used with DNNs)
    '''
    def export_ivector_extractor(self, destination):
        os.makedirs(destination, exist_ok=True)
        copy(os.path.join(self.dirname, 'final.ie'), destination)           # i-vector extractor itself
        copy(os.path.join(self.dirname, 'global_cmvn.stats'), destination)  # Stats from diag UBM
        copy(os.path.join(self.dirname, 'final.dubm'), destination)         # Diag UBM itself
        copy(os.path.join(self.dirname, 'final.mat'), destination)          # LDA matrix
import os
import shutil
import glob
import subprocess
import re
import io
import math
import numpy as np
from tqdm import tqdm
from shutil import copy, copyfile, rmtree, make_archive, unpack_archive
from contextlib import redirect_stdout
from aligner.models import IvectorExtractor
from random import shuffle

from ..helper import thirdparty_binary, make_path_safe, awk_like, filter_scp

from ..config import (MonophoneConfig, TriphoneConfig, TriphoneFmllrConfig,
                      LdaMlltConfig, DiagUbmConfig, iVectorExtractorConfig,
                      NnetBasicConfig)

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               lda_acc_stats,
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train,
                               nnet_align, nnet_get_align_feats, extract_ivectors,
                               compute_prob, get_average_posteriors, relabel_egs)
#from ..accuracy_graph import get_accuracy_graph


from ..exceptions import NoSuccessfulAlignments

from .. import __version__

from ..config import TEMP_DIR


class BaseAligner(object):
    '''
    Base aligner class for common aligner functions

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    output_directory : str
        Path to export aligned TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    mono_params : :class:`~aligner.config.MonophoneConfig`, optional
        Monophone training parameters to use, if different from defaults
    tri_params : :class:`~aligner.config.TriphoneConfig`, optional
        Triphone training parameters to use, if different from defaults
    tri_fmllr_params : :class:`~aligner.config.TriphoneFmllrConfig`, optional
        Speaker-adapted triphone training parameters to use, if different from defaults
    '''

    def __init__(self, corpus, dictionary, output_directory,
                 temp_directory=None, num_jobs=3, call_back=None,
                 mono_params=None, tri_params=None,
                 tri_fmllr_params=None, lda_mllt_params=None,
                 diag_ubm_params=None, ivector_extractor_params=None,
                 nnet_basic_params=None,
                 debug=False, skip_input=False, nnet=False):
        self.nnet = nnet

        if mono_params is None:
            mono_params = {}
        if tri_params is None:
            tri_params = {}
        if tri_fmllr_params is None:
            tri_fmllr_params = {}

        if lda_mllt_params is None:
            lda_mllt_params = {}
        if diag_ubm_params is None:
            diag_ubm_params = {}
        if ivector_extractor_params is None:
            ivector_extractor_params = {}
        if nnet_basic_params is None:
            nnet_basic_params = {}

        self.mono_config = MonophoneConfig(**mono_params)
        self.tri_config = TriphoneConfig(**tri_params)
        self.tri_fmllr_config = TriphoneFmllrConfig(**tri_fmllr_params)

        self.lda_mllt_config = LdaMlltConfig(**lda_mllt_params)
        self.diag_ubm_config = DiagUbmConfig(**diag_ubm_params)
        self.ivector_extractor_config = iVectorExtractorConfig(**ivector_extractor_params)
        self.nnet_basic_config = NnetBasicConfig(**nnet_basic_params)

        self.corpus = corpus
        self.dictionary = dictionary
        self.output_directory = output_directory
        self.num_jobs = num_jobs
        if self.corpus.num_jobs != num_jobs:
            self.num_jobs = self.corpus.num_jobs
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False
        self.debug = debug
        self.skip_input = skip_input
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary, skip_input=self.skip_input)
        print(self.corpus.speaker_utterance_info())

    @property
    def meta(self):
        data = {'phones':sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture':'gmm-hmm',
                'features':'mfcc+deltas',
                }
        return data

    @property
    def mono_directory(self):
        return os.path.join(self.temp_directory, 'mono')

    @property
    def mono_final_model_path(self):
        return os.path.join(self.mono_directory, 'final.mdl')

    @property
    def mono_ali_directory(self):
        return os.path.join(self.temp_directory, 'mono_ali')

    @property
    def tri_directory(self):
        return os.path.join(self.temp_directory, 'tri')

    @property
    def tri_ali_directory(self):
        return os.path.join(self.temp_directory, 'tri_ali')

    @property
    def tri_final_model_path(self):
        return os.path.join(self.tri_directory, 'final.mdl')

    @property
    def tri_fmllr_directory(self):
        return os.path.join(self.temp_directory, 'tri_fmllr')

    @property
    def tri_fmllr_ali_directory(self):
        return os.path.join(self.temp_directory, 'tri_fmllr_ali')

    @property
    def tri_fmllr_final_model_path(self):
        return os.path.join(self.tri_fmllr_directory, 'final.mdl')

    # Beginning of nnet properties
    @property
    def lda_mllt_directory(self):
        return os.path.join(self.temp_directory, 'lda_mllt')

    @property
    def lda_mllt_ali_directory(self):
        return os.path.join(self.temp_directory, 'lda_mllt_ali')

    @property
    def lda_mllt_final_model_path(self):
        return os.path.join(self.lda_mllt_directory, 'final.mdl')

    @property
    def diag_ubm_directory(self):
        return os.path.join(self.temp_directory, 'diag_ubm')

    @property
    def diag_ubm_final_model_path(self):
        return os.path.join(self.diag_ubm_directory, 'final.dubm')

    @property
    def ivector_extractor_directory(self):
        return os.path.join(self.temp_directory, 'ivector_extractor')

    @property
    def ivector_extractor_final_model_path(self):
        return os.path.join(self.ivector_extractor_directory, 'final.ie')

    @property
    def extracted_ivector_directory(self):
        return os.path.join(self.temp_directory, 'extracted_ivector')

    @property
    def nnet_basic_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic')

    @property
    def nnet_basic_ali_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic_ali')

    @property
    def nnet_basic_final_model_path(self):
        return os.path.join(self.nnet_basic_directory, 'final.mdl')

    # End of nnet properties

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if os.path.exists(self.nnet_basic_final_model_path):
            model_directory = self.nnet_basic_directory
        elif os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory

        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                                 self.corpus, self.num_jobs)

    def get_num_gauss_mono(self):
        '''
        Get the number of gaussians for a monophone model
        '''
        with open(os.devnull, 'w') as devnull:
            proc = subprocess.Popen([thirdparty_binary('gmm-info'),
                                     '--print-args=false',
                                     os.path.join(self.mono_directory, '0.mdl')],
                                    stderr=devnull,
                                    stdout=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            num = stdout.decode('utf8')
            matches = re.search(r'gaussians (\d+)', num)
            num = int(matches.groups()[0])
        return num

    def _align_si(self, fmllr=False, lda_mllt=False, feature_name=None):
        '''
        Generate an alignment of the dataset
        '''
        if fmllr and os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config

        elif fmllr:     # First pass with fmllr, final path doesn't exist yet
            model_directory = self.tri_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config

        elif lda_mllt and os.path.exists(self.lda_mllt_final_model_path):
            model_directory = self.lda_mllt_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config

        elif lda_mllt:  # First pass with LDA + MLLT, final path doesn't exist yet
            model_directory = self.tri_fmllr_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
            output_directory = self.tri_ali_directory
            config = self.tri_config
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory
            output_directory = self.mono_ali_directory
            config = self.mono_config

        optional_silence = self.dictionary.optional_silence_csl
        oov = self.dictionary.oov_int

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(self.tri_fmllr_ali_directory, exist_ok=True)
        os.makedirs(self.lda_mllt_ali_directory, exist_ok=True)

        os.makedirs(log_dir, exist_ok=True)

        shutil.copyfile(os.path.join(model_directory, 'tree'),
                        os.path.join(output_directory, 'tree'))
        shutil.copyfile(os.path.join(model_directory, 'final.mdl'),
                        os.path.join(output_directory, '0.mdl'))

        shutil.copyfile(os.path.join(model_directory, 'final.occs'),
                        os.path.join(output_directory, '0.occs'))

        feat_type = 'delta'

        compile_train_graphs(output_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs, debug=self.debug)

        align(0, output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, config, feature_name=feature_name)
        shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(output_directory, 'final.mdl'))
        shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(output_directory, 'final.occs'))

        if output_directory == self.tri_fmllr_ali_directory:
            os.makedirs(self.tri_fmllr_directory, exist_ok=True)
            shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(self.tri_fmllr_directory, 'final.mdl'))
            shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(self.tri_fmllr_directory, 'final.occs'))
        elif output_directory == self.lda_mllt_ali_directory:
            os.makedirs(self.lda_mllt_directory, exist_ok=True)
            shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(self.lda_mllt_directory, 'final.mdl'))
            shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(self.lda_mllt_directory, 'final.occs'))

    def parse_log_directory(self, directory, iteration):
        '''
        Parse error files and relate relevant information about unaligned files
        '''
        if not self.verbose:
            return
        error_regex = re.compile(r'Did not successfully decode file (\w+),')
        too_little_data_regex = re.compile(
            r'Gaussian has too little data but not removing it because it is the last Gaussian')
        skipped_transition_regex = re.compile(r'(\d+) out of (\d+) transition-states skipped due to insuffient data')

        log_like_regex = re.compile(r'Overall avg like per frame = ([-0-9.]+|nan) over (\d+) frames')
        error_files = []
        for i in range(self.num_jobs):
            path = os.path.join(directory, 'align.{}.{}.log'.format(iteration - 1, i))
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                error_files.extend(error_regex.findall(f.read()))
        update_path = os.path.join(directory, 'update.{}.log'.format(iteration))
        if os.path.exists(update_path):
            with open(update_path, 'r') as f:
                data = f.read()
                m = log_like_regex.search(data)
                if m is not None:
                    log_like, tot_frames = m.groups()
                    if log_like == 'nan':
                        raise (NoSuccessfulAlignments('Could not align any files.  Too little data?'))
                    self.call_back('log-likelihood', float(log_like))
                skipped_transitions = skipped_transition_regex.search(data)
                self.call_back('skipped transitions', *skipped_transitions.groups())
                num_too_little_data = len(too_little_data_regex.findall(data))
                self.call_back('missing data gaussians', num_too_little_data)
        if error_files:
            self.call_back('could not align', error_files)

    def _align_fmllr(self):
        '''
        Align the dataset using speaker-adapted transforms
        '''
        model_directory = self.tri_directory        # Get final.mdl from here
        first_output_directory = self.tri_ali_directory
        second_output_directory = self.tri_fmllr_ali_directory
        self._align_si(fmllr=False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(first_output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        calc_fmllr(first_output_directory, self.corpus.split_directory,
                   sil_phones, self.num_jobs, self.tri_fmllr_config, initial=True)
        optional_silence = self.dictionary.optional_silence_csl
        align(0, first_output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, self.tri_fmllr_config)

        # Copy into the "correct" tri_fmllr_ali output directory
        for file in glob.glob(os.path.join(first_output_directory, 'ali.*')):
            shutil.copy(file, second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'tree'), second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'final.mdl'), second_output_directory)


    def _init_tri(self, fmllr=False):
        if fmllr:
            config = self.tri_fmllr_config
            directory = self.tri_fmllr_directory
            align_directory = self.tri_ali_directory
        else:
            config = self.tri_config
            directory = self.tri_directory
            align_directory = self.mono_ali_directory

        if not self.debug:
            if os.path.exists(os.path.join(directory, '1.mdl')):
                return

        if fmllr:
            print('Initializing speaker-adapted triphone training...')
        else:
            print('Initializing triphone training...')
        context_opts = []
        ci_phones = self.dictionary.silence_csl

        tree_stats(directory, align_directory,
                   self.corpus.split_directory, ci_phones, self.num_jobs)
        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)

        log_path = os.path.join(directory, 'log', 'mixup.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-mixup'),
                             '--mix-up={}'.format(config.initial_gauss_count),
                             mdl_path, occs_path, mdl_path], stderr=logf)

        #os.remove(treeacc_path)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        shutil.copy(occs_path, os.path.join(directory, '1.occs'))
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))

        convert_alignments(directory, align_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):
            for i in range(self.num_jobs):
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))



    def train_tri_fmllr(self):
        '''
        Perform speaker-adapted triphone training
        '''
        if not self.debug:
            if os.path.exists(self.tri_fmllr_final_model_path):
                print('Triphone FMLLR training already done, using previous final.mdl')
                return

        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()

        #self._align_fmllr()

        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok=True)
        self._init_tri(fmllr=True)
        self._do_tri_fmllr_training()

    def _do_tri_fmllr_training(self):
        self.call_back('Beginning speaker-adapted triphone training...')
        self._do_training(self.tri_fmllr_directory, self.tri_fmllr_config)

    def _do_training(self, directory, config):
        if config.realign_iters is None:
            config.realign_iters = list(range(0, config.num_iters, 10))
        num_gauss = config.initial_gauss_count
        sil_phones = self.dictionary.silence_csl
        inc_gauss = config.inc_gauss_count
        if self.call_back == print:
            iters = tqdm(range(1, config.num_iters))
        else:
            iters = range(1, config.num_iters)
        log_directory = os.path.join(directory, 'log')
        for i in iters:
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            occs_path = os.path.join(directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            if not self.debug:
                if os.path.exists(next_model_path):
                    continue

            if i in config.realign_iters:
                align(i, directory, self.corpus.split_directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config,
                      feature_name='cmvnsplicetransformfeats')
            if config.do_fmllr and i in config.fmllr_iters:
                calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                           self.num_jobs, config, initial=False, iteration=i)

            if config.do_lda_mllt and i <= config.num_iters:
                calc_lda_mllt(directory, self.corpus.split_directory,   # Could change this to make ali directory later
                #calc_lda_mllt(self.lda_mllt_ali_directory, sil_phones,
                              self.lda_mllt_directory, sil_phones,
                              self.num_jobs, config, config.num_iters,
                              initial=False, iteration=i, corpus=self.corpus)


            acc_stats(i, directory, self.corpus.split_directory, self.num_jobs,
                      config.do_fmllr, do_lda_mllt=config.do_lda_mllt)
            log_path = os.path.join(log_directory, 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.num_jobs)]
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                             '--write-occs=' + occs_path,
                                             '--mix-up=' + str(num_gauss), '--power=' + str(config.power),
                                             model_path,
                                             "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                               ' '.join(map(make_path_safe, acc_files))),
                                             next_model_path],
                                            stderr=logf)
                est_proc.communicate()
            self.parse_log_directory(log_directory, i)
            if i < config.max_iter_inc:
                num_gauss += inc_gauss

        shutil.copy(os.path.join(directory, '{}.mdl'.format(config.num_iters)),
                    os.path.join(directory, 'final.mdl'))

        shutil.copy(os.path.join(directory, '{}.occs'.format(config.num_iters)),
                    os.path.join(directory, 'final.occs'))

        if config.do_lda_mllt:
            shutil.copy(os.path.join(directory, '{}.mat'.format(config.num_iters-1)),
                        os.path.join(directory, 'final.mat'))

    def train_lda_mllt(self):
        '''
        Perform LDA + MLLT training
        '''

        if not self.debug:
            if os.path.exists(self.lda_mllt_final_model_path):
                print('LDA + MLLT training already done, using previous final.mdl')
                return

        # N.B: The function _align_lda_mllt() is half-developed, but there doesn't seem to
        # be a reason for it to actually ever be called (since people will always have
        # fmllr done immediately before in the pipeline. Can clean/delete later if determined
        # that we need to actually use it somewhere or not).
        #if not os.path.exists(self.lda_mllt_ali_directory):
        #    self._align_lda_mllt()
        #self._align_lda_mllt()  # half implemented, can come back later or make people run from fmllr

        os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)

        self._init_lda_mllt()
        self._do_lda_mllt_training()

    def _init_lda_mllt(self):
        '''
        Initialize LDA + MLLT training.
        '''
        config = self.lda_mllt_config
        directory = self.lda_mllt_directory
        align_directory = self.tri_fmllr_ali_directory  # The previous
        mdl_dir = self.tri_fmllr_directory

        if not self.debug:
            if os.path.exists(os.path.join(directory, '1.mdl')):
                return

        print('Initializing LDA + MLLT training...')

        context_opts = []
        ci_phones = self.dictionary.silence_csl

        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')

        final_mdl_path = os.path.join(self.tri_fmllr_directory)

        # Accumulate LDA stats
        lda_acc_stats(directory, self.corpus.split_directory, align_directory, config, ci_phones, self.num_jobs)

        # Accumulating tree stats
        self.corpus._norm_splice_transform_feats(self.lda_mllt_directory)
        tree_stats(directory, align_directory, self.corpus.split_directory, ci_phones,
                   self.num_jobs, feature_name='cmvnsplicetransformfeats')

        # Getting questions for tree clustering
        log_path = os.path.join(directory, 'log', 'cluster_phones.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        # Building the tree
        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        # Initializing the model
        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))
        shutil.copy(occs_path, os.path.join(directory, '1.occs'))

        convert_alignments(directory, align_directory, self.num_jobs)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):            
            for i in range(self.num_jobs):                                      
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))

    def _do_lda_mllt_training(self):
        self.call_back('Beginning LDA + MLLT training...')
        self._do_training(self.lda_mllt_directory, self.lda_mllt_config)

    def train_nnet_basic(self):
        '''
        Perform neural network training
        '''

        os.makedirs(os.path.join(self.nnet_basic_directory, 'log'), exist_ok=True)

        split_directory = self.corpus.split_directory
        config = self.nnet_basic_config
        tri_fmllr_config = self.tri_fmllr_config
        directory = self.nnet_basic_directory
        nnet_align_directory = self.nnet_basic_ali_directory
        align_directory = self.tri_fmllr_ali_directory
        lda_directory = self.lda_mllt_directory
        egs_directory = os.path.join(directory, 'egs')
        training_directory = self.corpus.output_directory

        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        L_fst_path = os.path.join(self.dictionary.output_directory, 'L.fst')
        ali_tree_path = os.path.join(align_directory, 'tree')
        shutil.copy(ali_tree_path, os.path.join(directory, 'tree'))

        mdl_path = os.path.join(align_directory, 'final.mdl')
        raw_feats = os.path.join(training_directory, 'feats.scp')

        tree_info_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                                          os.path.join(align_directory, 'tree')],
                                          stdout=subprocess.PIPE)
        tree_info = tree_info_proc.stdout.read()
        tree_info = tree_info.split()
        num_leaves = tree_info[1]
        num_leaves = num_leaves.decode("utf-8")

        lda_dim = self.lda_mllt_config.dim 

        # Extract iVectors
        self._extract_ivectors()

        # Get LDA matrix
        fixed_ivector_dir = self.extracted_ivector_directory
        get_lda_nnet(directory, align_directory, fixed_ivector_dir, training_directory,
                     split_directory, raw_feats, self.dictionary.optional_silence_csl, config, self.num_jobs)

        log_path = os.path.join(directory, 'log', 'lda_matrix.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(directory, 'lda.{}.acc'.format(x))
                         for x in range(self.num_jobs)]
            sum_lda_accs_proc = subprocess.Popen([thirdparty_binary('sum-lda-accs'),
                                                 os.path.join(directory, 'lda.acc')]
                                                 + acc_files,
                                                 stderr=logf)
            sum_lda_accs_proc.communicate()

            lda_mat_proc = subprocess.Popen([thirdparty_binary('nnet-get-feature-transform'),
                                            '--dim=' + str(lda_dim),
                                            os.path.join(directory, 'lda.mat'),
                                            os.path.join(directory, 'lda.acc')],
                                            stderr=logf)
            lda_mat_proc.communicate()
        lda_mat_path = os.path.join(directory, 'lda.mat')


        # Get examples for training
        os.makedirs(egs_directory, exist_ok=True)

        # # Get valid uttlist and train subset uttlist
        valid_uttlist = os.path.join(directory, 'valid_uttlist')
        train_subset_uttlist = os.path.join(directory, 'train_subset_uttlist')
        training_feats = os.path.join(directory, 'nnet_training_feats')
        num_utts_subset = 300
        log_path = os.path.join(directory, 'log', 'training_egs_feats.log')

        with open(log_path, 'w') as logf:
            with open(valid_uttlist, 'w') as outf:
                # Get first column from utt2spk (awk-like)
                utt2spk_col = awk_like(os.path.join(training_directory, 'utt2spk'), 0)
                # Shuffle the list from the column
                shuffle(utt2spk_col)
                # Take only the first num_utts_subset lines
                utt2spk_col = utt2spk_col[:num_utts_subset]
                # Write the result to file
                for line in utt2spk_col:
                    outf.write(line)
                    outf.write('\n')

            with open(train_subset_uttlist, 'w') as outf:
                # Get first column from utt2spk (awk-like)
                utt2spk_col = awk_like(os.path.join(training_directory, 'utt2spk'), 0)
                # Filter by the scp list
                filtered = filter_scp(valid_uttlist, utt2spk_col, exclude=True)
                # Shuffle the list
                shuffle(filtered)
                # Take only the first num_utts_subset lines
                filtered = filtered[:num_utts_subset]
                # Write the result to a file
                for line in filtered:
                    outf.write(line)
                    outf.write('\n')

        get_egs(directory, egs_directory, training_directory, split_directory, align_directory,
                fixed_ivector_dir, training_feats, valid_uttlist,
                train_subset_uttlist, config, self.num_jobs)

        # Initialize neural net
        print('Beginning DNN training...')
        stddev = float(1.0/config.pnorm_input_dim**0.5)
        online_preconditioning_opts = 'alpha={} num-samples-history={} update-period={} rank-in={} rank-out={} max-change-per-sample={}'.format(config.alpha, config.num_samples_history, config.update_period, config.precondition_rank_in, config.precondition_rank_out, config.max_change_per_sample)
        nnet_config_path = os.path.join(directory, 'nnet.config')
        hidden_config_path = os.path.join(directory, 'hidden.config')
        ivector_dim_path = os.path.join(directory, 'ivector_dim')
        with open(ivector_dim_path, 'r') as inf:
            ivector_dim = inf.read().strip()
        feat_dim = 13 + int(ivector_dim)

        with open(nnet_config_path, 'w') as nc:
            nc.write('SpliceComponent input-dim={} left-context={} right-context={} const-component-dim={}\n'.format(feat_dim, config.splice_width, config.splice_width, ivector_dim))
            nc.write('FixedAffineComponent matrix={}\n'.format(lda_mat_path))
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(lda_dim, config.pnorm_input_dim, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(config.pnorm_input_dim, config.pnorm_output_dim, config.p))
            nc.write('NormalizeComponent dim={}\n'.format(config.pnorm_output_dim))
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.pnorm_output_dim, num_leaves, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('SoftmaxComponent dim={}\n'.format(num_leaves))

        with open(hidden_config_path, 'w') as nc:
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.pnorm_output_dim, config.pnorm_input_dim, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(config.pnorm_input_dim, config.pnorm_output_dim, config.p))
            nc.write('NormalizeComponent dim={}\n'.format(config.pnorm_output_dim))

        log_path = os.path.join(directory, 'log', 'nnet_init.log')
        nnet_info_path = os.path.join(directory, 'log', 'nnet_info.log')
        with open(log_path, 'w') as logf:
            with open(nnet_info_path, 'w') as outf:
                nnet_am_init_proc = subprocess.Popen([thirdparty_binary('nnet-am-init'),
                                                     os.path.join(align_directory, 'tree'),
                                                     topo_path,
                                                     "{} {} -|".format(thirdparty_binary('nnet-init'),
                                                                       nnet_config_path),
                                                    os.path.join(directory, '0.mdl')],
                                                    stderr=logf)
                nnet_am_init_proc.communicate()

                nnet_am_info = subprocess.Popen([thirdparty_binary('nnet-am-info'),
                                                os.path.join(directory, '0.mdl')],
                                                stdout=outf,
                                                stderr=logf)
                nnet_am_info.communicate()


        # Train transition probabilities and set priors
        #   First combine all previous alignments
        ali_files = glob.glob(os.path.join(align_directory, 'ali.*'))
        prev_ali_path = os.path.join(directory, 'prev_ali.')
        with open(prev_ali_path, 'wb') as outfile:
            for ali_file in ali_files:
                with open(os.path.join(align_directory, ali_file), 'rb') as infile:
                    for line in infile:
                        outfile.write(line)
        nnet_train_trans(directory, align_directory, prev_ali_path, self.num_jobs)

        # Get iteration at which we will mix up
        num_tot_iters = config.num_epochs * config.iters_per_epoch
        finish_add_layers_iter = config.num_hidden_layers * config.add_layers_period
        first_modify_iter = finish_add_layers_iter + config.add_layers_period
        mix_up_iter = (num_tot_iters + finish_add_layers_iter)/2

        # Get iterations at which we will realign
        realign_iters = []
        if config.realign_times != 0:
            div = config.realign_times + 1 # (e.g. realign 2 times = iterations split into 3 sets)
            split_iters = np.array_split(range(num_tot_iters), div)
            for group in split_iters:
                realign_iters.append(group[0])

        # Training loop
        for i in range(num_tot_iters):
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            # Combine all examples (could integrate validation diagnostics, etc., later-- see egs functions)
            egs_files = []
            for file in os.listdir(egs_directory):
                if file.startswith('egs'):
                    egs_files.append(file)
            with open(os.path.join(egs_directory, 'all_egs.egs'), 'wb') as outfile:
                for egs_file in egs_files:
                    with open(os.path.join(egs_directory, egs_file), 'rb') as infile:
                        for line in infile:
                            outfile.write(line)

            # Get accuracy rates for the current iteration (to pull out graph later)
            #compute_prob(i, directory, egs_directory, model_path, self.num_jobs)
            log_path = os.path.join(directory, 'log', 'compute_prob_train.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                compute_prob_proc = subprocess.Popen([thirdparty_binary('nnet-compute-prob'),
                                                     model_path,
                                                     'ark:{}/all_egs.egs'.format(egs_directory)],
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                log_prob = compute_prob_proc.stdout.read().decode('utf-8').strip()
                compute_prob_proc.communicate()

            print("Iteration {} of {} \t\t Log-probability: {}".format(i+1, num_tot_iters, log_prob))

            # Pull out and save graphs
            # This is not quite working when done automatically - to be worked out with unit testing.
            #get_accuracy_graph(os.path.join(directory, 'log'), os.path.join(directory, 'log'))

            # If it is NOT the first iteration,
            # AND we still have layers to add,
            # AND it's the right time to add a layer...
            if i > 0 and i <= ((config.num_hidden_layers-1)*config.add_layers_period) and ((i-1)%config.add_layers_period) == 0:
                # Add a new hidden layer
                mdl = os.path.join(directory, 'tmp{}.mdl'.format(i))
                log_path = os.path.join(directory, 'log', 'temp_mdl.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    with open(mdl, 'w') as outf:
                        tmp_mdl_init_proc = subprocess.Popen([thirdparty_binary('nnet-init'),
                                                            '--srand={}'.format(i),
                                                            os.path.join(directory, 'hidden.config'),
                                                            '-'],
                                                            stdout=subprocess.PIPE,
                                                            stderr=logf)
                        tmp_mdl_ins_proc = subprocess.Popen([thirdparty_binary('nnet-insert'),
                                                            os.path.join(directory, '{}.mdl'.format(i)),
                                                            '-', '-'],
                                                            stdin=tmp_mdl_init_proc.stdout,
                                                            stdout=outf,
                                                            stderr=logf)
                        tmp_mdl_ins_proc.communicate()

            # Otherwise just use the past model
            else:
                mdl = os.path.join(directory, '{}.mdl'.format(i))

            # Shuffle examples and train nets with SGD
            nnet_train(directory, egs_directory, mdl, i, self.num_jobs)

            # Get nnet list from the various jobs on this iteration
            nnets_list = [os.path.join(directory, '{}.{}.mdl'.format((i+1), x))
                         for x in range(self.num_jobs)]

            if (i+1) >= num_tot_iters:
                learning_rate = config.final_learning_rate
            else:
                learning_rate = config.initial_learning_rate * math.exp(i * math.log(config.final_learning_rate/config.initial_learning_rate)/num_tot_iters)

            log_path = os.path.join(directory, 'log', 'average.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                nnet_avg_proc = subprocess.Popen([thirdparty_binary('nnet-am-average')]
                                                 + nnets_list
                                                 + ['-'],
                                                 stdout=subprocess.PIPE,
                                                 stderr=logf)
                nnet_copy_proc = subprocess.Popen([thirdparty_binary('nnet-am-copy'),
                                                  '--learning-rate={}'.format(learning_rate),
                                                  '-',
                                                  os.path.join(directory, '{}.mdl'.format(i+1))],
                                                  stdin=nnet_avg_proc.stdout,
                                                  stderr=logf)
                nnet_copy_proc.communicate()

            # If it's the right time, do mixing up
            if config.mix_up > 0 and i == mix_up_iter:
                log_path = os.path.join(directory, 'log', 'mix_up.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_am_mixup_proc = subprocess.Popen([thirdparty_binary('nnet-am-mixup'),
                                                          '--min-count=10',
                                                          '--num-mixtures={}'.format(config.mix_up),
                                                          os.path.join(directory, '{}.mdl'.format(i+1)),
                                                          os.path.join(directory, '{}.mdl'.format(i+1))],
                                                          stderr=logf)
                    nnet_am_mixup_proc.communicate()

            # Realign if it's the right time
            if i in realign_iters:
                prev_egs_directory = egs_directory
                egs_directory = os.path.join(directory, 'egs_{}'.format(i))
                os.makedirs(egs_directory, exist_ok=True)

                #   Get average posterior for purposes of adjusting priors
                get_average_posteriors(i, directory, prev_egs_directory, config, self.num_jobs)
                log_path = os.path.join(directory, 'log', 'vector_sum_exterior.{}.log'.format(i))
                vectors_to_sum = glob.glob(os.path.join(directory, 'post.{}.*.vec'.format(i)))

                with open(log_path, 'w') as logf:
                    vector_sum_proc = subprocess.Popen([thirdparty_binary('vector-sum')]
                                                       + vectors_to_sum
                                                       + [os.path.join(directory, 'post.{}.vec'.format(i))
                                                       ],
                                                       stderr=logf)
                    vector_sum_proc.communicate()

                #   Readjust priors based on computed posteriors
                log_path = os.path.join(directory, 'log', 'adjust_priors.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_adjust_priors_proc = subprocess.Popen([thirdparty_binary('nnet-adjust-priors'),
                                                               os.path.join(directory, '{}.mdl'.format(i)),
                                                               os.path.join(directory, 'post.{}.vec'.format(i)),
                                                               os.path.join(directory, '{}.mdl'.format(i))],
                                                               stderr=logf)
                    nnet_adjust_priors_proc.communicate()

                #   Realign:
                #       Compile train graphs (gets fsts.{} for alignment)
                compile_train_graphs(directory, self.dictionary.output_directory,
                                     self.corpus.split_directory, self.num_jobs)

                #       Get alignment feats
                nnet_get_align_feats(directory, self.corpus.split_directory, fixed_ivector_dir, config, self.num_jobs)

                #       Do alignment
                nnet_align(i, directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config)

                #     Finally, relabel the egs
                ali_files = glob.glob(os.path.join(directory, 'ali.*'))
                alignments = os.path.join(directory, 'alignments.')
                with open(alignments, 'wb') as outfile:
                    for ali_file in ali_files:
                        with open(os.path.join(directory, ali_file), 'rb') as infile:
                            for line in infile:
                                outfile.write(line)
                relabel_egs(i, directory, prev_egs_directory, alignments, egs_directory, self.num_jobs)


        # Rename the final model
        shutil.copy(os.path.join(directory, '{}.mdl'.format(num_tot_iters-1)), os.path.join(directory, 'final.mdl'))

        # Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        # Get alignment feats
        nnet_get_align_feats(directory, self.corpus.split_directory, fixed_ivector_dir, config, self.num_jobs)

        # Do alignment
        nnet_align("final", directory,
              self.dictionary.optional_silence_csl,
              self.num_jobs, config, mdl=os.path.join(directory, 'final.mdl'))

    def _extract_ivectors(self):
        '''
        Extracts i-vectors from a corpus using the trained i-vector extractor.
        '''
        print('Extracting i-vectors...')

        log_dir = os.path.join(self.extracted_ivector_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        # To do still for release: maybe add arguments to command line to tell MFA which
        # i-vector extractor to use.

        directory = self.extracted_ivector_directory

        # Only one option for now - make this an argument eventually.
        # Librispeech 100 chosen because of large number of speakers, not necessarily longer length. 
        # Thesis results tentatively confirmed that more speakers in ivector extractor => better results.
        ivector_extractor = IvectorExtractor(os.path.join(os.path.dirname(__file__), '../../pretrained_models/ls_100_ivector_extractor.zip'))
        ivector_extractor_directory = os.path.join(self.temp_directory, 'ivector_extractor')
        ivector_extractor.export_ivector_extractor(ivector_extractor_directory)

        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        config = self.ivector_extractor_config
        training_directory = self.corpus.output_directory

        # To make a directory for corpus with just 2 utterances per speaker
        # (left commented out in case we ever decide to do this)
        """max2_dir = os.path.join(directory, 'max2')
        os.makedirs(max2_dir, exist_ok=True)
        mfa_working_dir = os.getcwd()
        os.chdir("/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2")
        copy_data_sh = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2/copy_data_dir.sh"
        log_path = os.path.join(directory, 'log', 'max2.log')
        with open(log_path, 'w') as logf:
            command = [copy_data_sh, '--utts-per-spk-max', '2', train_dir, max2_dir]
            max2_proc = subprocess.Popen(command,
                                         stderr=logf)
            max2_proc.communicate()
        os.chdir(mfa_working_dir)"""

        # Write a "cmvn config" file (this is blank in the actual kaldi code, but it needs the argument passed)
        cmvn_config = os.path.join(directory, 'online_cmvn.conf')
        with open(cmvn_config, 'w') as cconf:
            cconf.write("")

        # Write a "splice config" file
        splice_config = os.path.join(directory, 'splice.conf')
        with open(splice_config, 'w') as sconf:
            sconf.write(config.splice_opts[0])
            sconf.write('\n')
            sconf.write(config.splice_opts[1])

        # Write a "config" file to input to the extraction binary
        ext_config = os.path.join(directory, 'ivector_extractor.conf')
        with open(ext_config, 'w') as ieconf:
            ieconf.write('--cmvn-config={}\n'.format(cmvn_config))
            ieconf.write('--ivector-period={}\n'.format(config.ivector_period))
            ieconf.write('--splice-config={}\n'.format(splice_config))
            ieconf.write('--lda-matrix={}\n'.format(os.path.join(ivector_extractor_directory, 'final.mat')))
            ieconf.write('--global-cmvn-stats={}\n'.format(os.path.join(ivector_extractor_directory, 'global_cmvn.stats')))
            ieconf.write('--diag-ubm={}\n'.format(os.path.join(ivector_extractor_directory, 'final.dubm')))
            ieconf.write('--ivector-extractor={}\n'.format(os.path.join(ivector_extractor_directory, 'final.ie')))
            ieconf.write('--num-gselect={}\n'.format(config.num_gselect))
            ieconf.write('--min-post={}\n'.format(config.min_post))
            ieconf.write('--posterior-scale={}\n'.format(config.posterior_scale))
            ieconf.write('--max-remembered-frames=1000\n')
            ieconf.write('--max-count={}\n'.format(0))

        # Extract i-vectors
        extract_ivectors(directory, training_directory, ext_config, config, self.num_jobs)

        # Combine i-vectors across jobs
        file_list = []
        for j in range(self.num_jobs):
            file_list.append(os.path.join(directory, 'ivector_online.{}.scp'.format(j)))

        with open(os.path.join(directory, 'ivector_online.scp'), 'w') as outfile:
            for fname in file_list:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
import os
import shutil
from tqdm import tqdm
import re
import glob

from .base import BaseAligner, TEMP_DIR, TriphoneFmllrConfig, TriphoneConfig, LdaMlltConfig, iVectorExtractorConfig, NnetBasicConfig

from ..exceptions import PronunciationAcousticMismatchError

from ..multiprocessing import (align, calc_fmllr, test_utterances, thirdparty_binary, subprocess,
                               convert_ali_to_textgrids, compile_train_graphs, nnet_get_align_feats, nnet_align)


def parse_transitions(path, phones_path):
    state_extract_pattern = re.compile(r'Transition-state (\d+): phone = (\w+)')
    id_extract_pattern = re.compile(r'Transition-id = (\d+)')
    cur_phone = None
    current = 0
    with open(path, encoding='utf8') as f, open(phones_path, 'w', encoding='utf8') as outf:
        outf.write('{} {}\n'.format('<eps>', 0))
        for line in f:
            line = line.strip()
            if line.startswith('Transition-state'):
                m = state_extract_pattern.match(line)
                _, phone = m.groups()
                if phone != cur_phone:
                    current = 0
                    cur_phone = phone
            else:
                m = id_extract_pattern.match(line)
                id = m.groups()[0]
                outf.write('{}_{} {}\n'.format(phone, current, id))
                current += 1


class PretrainedAligner(BaseAligner):
    '''
    Class for aligning a dataset using a pretrained acoustic model

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    acoustic_model : :class:`~aligner.models.AcousticModel`
        Archive containing the acoustic model and pronunciation dictionary
    output_directory : str
        Path to directory to save TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    '''

    def __init__(self, corpus, dictionary, acoustic_model, output_directory,
                 temp_directory=None, num_jobs=3, speaker_independent=False,
                 call_back=None, debug=False, skip_input=False, nnet=False):
        self.debug = debug
        self.nnet = nnet
        if temp_directory is None:
            temp_directory = TEMP_DIR
        self.acoustic_model = acoustic_model
        self.temp_directory = temp_directory
        self.output_directory = output_directory
        self.corpus = corpus
        self.speaker_independent = speaker_independent
        self.dictionary = dictionary
        self.skip_input = skip_input

        self.setup()

        if not nnet:
            self.acoustic_model.export_triphone_model(self.tri_directory)
            log_dir = os.path.join(self.tri_directory, 'log')
        else:
            self.acoustic_model.export_nnet_model(self.nnet_basic_directory)
            log_dir = os.path.join(self.nnet_basic_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        if self.corpus.num_jobs != num_jobs:
            num_jobs = self.corpus.num_jobs
        self.num_jobs = num_jobs
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False

        self.tri_fmllr_config = TriphoneFmllrConfig(**{'realign_iters': [1, 2],
                                                       'fmllr_iters': [1],
                                                       'num_iters': 3,
                                                       # 'boost_silence': 0
                                                       })
        self.tri_config = TriphoneConfig()
        self.lda_mllt_config = LdaMlltConfig()
        self.ivector_extractor_config = iVectorExtractorConfig()
        self.nnet_basic_config = NnetBasicConfig()

        if self.debug:
            os.makedirs(os.path.join(self.tri_directory, 'log'), exist_ok=True)
            mdl_path = os.path.join(self.tri_directory, 'final.mdl')
            tree_path = os.path.join(self.tri_directory, 'tree')
            occs_path = os.path.join(self.tri_directory, 'final.occs')
            log_path = os.path.join(self.tri_directory, 'log', 'show_transition.log')
            transition_path = os.path.join(self.tri_directory, 'transitions.txt')
            tree_pdf_path = os.path.join(self.tri_directory, 'tree.pdf')
            tree_dot_path = os.path.join(self.tri_directory, 'tree.dot')
            phones_path = os.path.join(self.dictionary.output_directory, 'phones.txt')
            triphones_path = os.path.join(self.tri_directory, 'triphones.txt')
            with open(log_path, 'w') as logf:
                with open(transition_path, 'w', encoding='utf8') as f:
                    subprocess.call([thirdparty_binary('show-transitions'), phones_path, mdl_path, occs_path], stdout=f,
                                    stderr=logf)
                parse_transitions(transition_path, triphones_path)
                if False:
                    with open(tree_dot_path, 'wb') as treef:
                        draw_tree_proc = subprocess.Popen([thirdparty_binary('draw-tree'), phones_path, tree_path],
                                                          stdout=treef, stderr=logf)
                        draw_tree_proc.communicate()
                    with open(tree_dot_path, 'rb') as treeinf, open(tree_pdf_path, 'wb') as treef:
                        dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-Gsize=8,10.5'], stdin=treeinf,
                                                    stdout=treef, stderr=logf)
                        dot_proc.communicate()
        print('Done with setup.')

    def setup(self):
        self.dictionary.nonsil_phones = self.acoustic_model.meta['phones']
        super(PretrainedAligner, self).setup()

    def test_utterance_transcriptions(self):
        return test_utterances(self)

    def do_align_nnet(self):
        '''
        Perform alignment using a previous DNN model
        '''

       # N.B.: This if ought to be commented out when developing.
        #if not os.path.exists(self.nnet_basic_ali_directory):
        print("doing align nnet")
        print("nnet basic directory is: {}".format(self.nnet_basic_directory))
        optional_silence = self.dictionary.optional_silence_csl

        # Extract i-vectors
        self._extract_ivectors()

        # Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(self.nnet_basic_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs, mdl='final')

        # Get alignment feats
        nnet_get_align_feats(self.nnet_basic_directory, self.corpus.split_directory, self.extracted_ivector_directory, self.nnet_basic_config, self.num_jobs)

        # Do nnet alignment
        nnet_align(0, self.nnet_basic_directory, optional_silence, self.num_jobs, self.nnet_basic_config, mdl='final')

    def do_align(self):
        '''
        Perform alignment while calculating speaker transforms (fMLLR estimation)
        '''
        self._init_tri()
        if not self.speaker_independent:
            self.train_tri_fmllr()

    def _align_fmllr(self):
        '''
        Align the dataset using speaker-adapted transforms
        '''
        model_directory = self.tri_directory        # Get final.mdl from here
        first_output_directory = self.tri_ali_directory
        second_output_directory = self.tri_fmllr_ali_directory
        os.makedirs(first_output_directory, exist_ok=True)
        os.makedirs(second_output_directory, exist_ok=True)
        if self.debug:
            shutil.copyfile(os.path.join(self.tri_directory, 'triphones.txt'),
                            os.path.join(self.tri_ali_directory, 'triphones.txt'))
        self._align_si(fmllr=False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(first_output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        if not self.speaker_independent:
            calc_fmllr(first_output_directory, self.corpus.split_directory,
                       sil_phones, self.num_jobs, self.tri_fmllr_config, initial=True)
            optional_silence = self.dictionary.optional_silence_csl
            align(0, first_output_directory, self.corpus.split_directory,
                  optional_silence, self.num_jobs, self.tri_fmllr_config)

        # Copy into the "correct" tri_fmllr_ali output directory
        for file in glob.glob(os.path.join(first_output_directory, 'ali.*')):
            shutil.copy(file, second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'tree'), second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'final.mdl'), second_output_directory)

    def _init_tri(self):
        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()
        if self.speaker_independent:
            return
        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok=True)

        shutil.copy(os.path.join(self.tri_directory, 'final.mdl'),
                    os.path.join(self.tri_fmllr_directory, '1.mdl'))

        for i in range(self.num_jobs):
            shutil.copy(os.path.join(self.tri_ali_directory, 'fsts.{}'.format(i)),
                        os.path.join(self.tri_fmllr_directory, 'fsts.{}'.format(i)))
            shutil.copy(os.path.join(self.tri_ali_directory, 'trans.{}'.format(i)),
                        os.path.join(self.tri_fmllr_directory, 'trans.{}'.format(i)))

    def train_tri_fmllr(self):
        directory = self.tri_fmllr_directory
        sil_phones = self.dictionary.silence_csl
        if self.call_back == print:
            iters = tqdm(range(1, self.tri_fmllr_config.num_iters))
        else:
            iters = range(1, self.tri_fmllr_config.num_iters)
        log_directory = os.path.join(directory, 'log')
        for i in iters:
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            occs_path = os.path.join(directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))
            if os.path.exists(next_model_path):
                continue
            align(i, directory, self.corpus.split_directory,
                  self.dictionary.optional_silence_csl,
                  self.num_jobs, self.tri_fmllr_config)
            calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                       self.num_jobs, self.tri_fmllr_config, initial=False, iteration=i)
            os.rename(model_path, next_model_path)
            self.parse_log_directory(log_directory, i)
        os.rename(next_model_path, os.path.join(directory, 'final.mdl'))

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if self.speaker_independent:
            model_directory = self.tri_ali_directory
        else:
            model_directory = self.tri_fmllr_directory
        if self.nnet:
            model_directory = self.nnet_basic_directory
        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                                 self.corpus, self.num_jobs)
        print("Exported textgrids to {}".format(self.output_directory))
        print("Log of export at {}".format(model_directory))
import os
import shutil
import subprocess
import re
import math
from tqdm import tqdm

from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               lda_acc_stats,
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train,
                               nnet_align, nnet_get_align_feats, extract_ivectors)

from ..exceptions import NoSuccessfulAlignments

from .base import BaseAligner

from ..models import AcousticModel


class TrainableAligner(BaseAligner):
    '''
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    output_directory : str
        Path to export aligned TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    mono_params : :class:`~aligner.config.MonophoneConfig`, optional
        Monophone training parameters to use, if different from defaults
    tri_params : :class:`~aligner.config.TriphoneConfig`, optional
        Triphone training parameters to use, if different from defaults
    tri_fmllr_params : :class:`~aligner.config.TriphoneFmllrConfig`, optional
        Speaker-adapted triphone training parameters to use, if different from defaults
    '''

    def save(self, path):
        '''
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        '''
        directory, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        acoustic_model = AcousticModel.empty(basename)
        acoustic_model.add_meta_file(self)
        if not self.nnet:
            acoustic_model.add_triphone_fmllr_model(self.tri_fmllr_directory)
        else:
            acoustic_model.add_nnet_model(self.nnet_basic_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        acoustic_model.dump(basename)
        print('Saved model to {}'.format(path))

    def _do_tri_training(self):
        self.call_back('Beginning triphone training...')
        self._do_training(self.tri_directory, self.tri_config)

    def train_tri(self):
        '''
        Perform triphone training
        '''

        if not self.debug:
            if os.path.exists(self.tri_final_model_path):
                print('Triphone training already done, using previous final.mdl')
                return

        if not os.path.exists(self.mono_ali_directory):
            self._align_si()

        os.makedirs(os.path.join(self.tri_directory, 'log'), exist_ok=True)

        self._init_tri(fmllr=False)
        self._do_tri_training()

    def _init_mono(self):
        '''
        Initialize monophone training
        '''
        print("Initializing monophone training...")
        log_dir = os.path.join(self.mono_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        tree_path = os.path.join(self.mono_directory, 'tree')
        mdl_path = os.path.join(self.mono_directory, '0.mdl')

        directory = self.corpus.split_directory
        feat_dim = self.corpus.get_feat_dim()
        path = os.path.join(directory, 'cmvndeltafeats.0_sub')
        feat_path = os.path.join(directory, 'cmvndeltafeats.0')
        shared_phones_opt = "--shared-phones=" + os.path.join(self.dictionary.phones_dir, 'sets.int')
        log_path = os.path.join(log_dir, 'log')
        with open(path, 'rb') as f, open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-mono'), shared_phones_opt,
                             "--train-feats=ark:-",
                             os.path.join(self.dictionary.output_directory, 'topo'),
                             feat_dim,
                             mdl_path,
                             tree_path],
                            stdin=f,
                            stderr=logf)
        num_gauss = self.get_num_gauss_mono()
        compile_train_graphs(self.mono_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)
        mono_align_equal(self.mono_directory,
                         self.corpus.split_directory, self.num_jobs)
        log_path = os.path.join(self.mono_directory, 'log', 'update.0.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(self.mono_directory, '0.{}.acc'.format(x)) for x in range(self.num_jobs)]
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                         '--min-gaussian-occupancy=3',
                                         '--mix-up={}'.format(num_gauss), '--power={}'.format(self.mono_config.power),
                                         mdl_path, "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                                     ' '.join(map(make_path_safe, acc_files))),
                                         os.path.join(self.mono_directory, '1.mdl')],
                                        stderr=logf)
            est_proc.communicate()

    def _do_mono_training(self):
        self.mono_config.initial_gauss_count = self.get_num_gauss_mono()
        self.call_back('Beginning monophone training...')
        self._do_training(self.mono_directory, self.mono_config)

    def train_mono(self):
        '''
        Perform monophone training
        '''
        final_mdl = os.path.join(self.mono_directory, 'final.mdl')

        if not self.debug:
            if os.path.exists(final_mdl):
                print('Monophone training already done, using previous final.mdl')
                return

        os.makedirs(os.path.join(self.mono_directory, 'log'), exist_ok=True)

        self._init_mono()
        self._do_mono_training()

    # Beginning of nnet functions


    def _align_lda_mllt(self):
        '''
        Align the dataset using LDA + MLLT transforms
        '''
        log_dir = os.path.join(self.lda_mllt_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        feat_name = "cmvnsplicetransformfeats"
        model_directory = self.tri_fmllr_directory  # Get final.mdl from here
        output_directory = self.lda_mllt_ali_directory  # Alignments end up here
        self._align_si(fmllr=False, lda_mllt=True, feature_name=feat_name)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        calc_lda_mllt(output_directory, self.corpus.split_directory,
                      self.tri_fmllr_directory,
                      sil_phones, self.num_jobs, self.lda_mllt_config,
                      self.lda_mllt_config.num_iters, initial=True)
        optional_silence = self.dictionary.optional_silence_csl
        align(0, model_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, self.lda_mllt_config, feature_name=feat_name)





    def train_diag_ubm(self):
        '''
        Train a diagonal UBM on the LDA + MLLT model
        '''
        
        if not self.debug:
            if os.path.exists(self.diag_ubm_final_model_path):
                print('Diagonal UBM training already done; using previous model')
                return
        log_dir = os.path.join(self.diag_ubm_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        lda_mllt_path = self.lda_mllt_directory
        directory = self.diag_ubm_directory

        cmvn_path = os.path.join(train_dir, 'cmvn.scp')

        old_config = self.lda_mllt_config
        config = self.diag_ubm_config
        ci_phones = self.dictionary.silence_csl

        final_mat_path = os.path.join(lda_mllt_path, 'final.mat')

        # Create global_cmvn.stats
        log_path = os.path.join(directory, 'log', 'make_global_cmvn.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('matrix-sum'),
                            '--binary=false',
                            'scp:' + cmvn_path,
                             os.path.join(directory, 'global_cmvn.stats')],
                             stderr=logf)

        # Get all feats
        all_feats_path = os.path.join(split_dir, 'cmvnonlinesplicetransformfeats')
        log_path = os.path.join(split_dir, 'log', 'cmvnonlinesplicetransform.log')
        with open(log_path, 'w') as logf:
            with open(all_feats_path, 'wb') as outf:
                apply_cmvn_online_proc = subprocess.Popen([thirdparty_binary('apply-cmvn-online'),
                                                          #'--config=' +
                                                          # This^ makes reference to a config file
                                                          # in Kaldi, but it's empty there
                                                          os.path.join(directory, 'global_cmvn.stats'),
                                                          'scp:' + train_dir + '/feats.scp',
                                                          'ark:-'],
                                                          stdout=subprocess.PIPE,
                                                          stderr=logf)
                splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats')]
                                                     + config.splice_opts +
                                                     ['ark:-', 'ark:-'],
                                                     stdin=apply_cmvn_online_proc.stdout,
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                transform_feats_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                                        os.path.join(lda_mllt_path, 'final.mat'),
                                                        'ark:-', 'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=outf,
                                                        stderr=logf)
                transform_feats_proc.communicate()

        # Initialize model from E-M in memory
        num_gauss_init = int(config.initial_gauss_proportion * int(config.num_gauss))
        log_path = os.path.join(directory, 'log', 'gmm_init.log')
        with open(log_path, 'w') as logf:
            gmm_init_proc = subprocess.Popen([thirdparty_binary('gmm-global-init-from-feats'),
                                             '--num-threads=' + str(config.num_threads),
                                             '--num-frames=' + str(config.num_frames),
                                             '--num_gauss=' + str(config.num_gauss),
                                             '--num_gauss_init=' + str(num_gauss_init),
                                             '--num_iters=' + str(config.num_iters_init),
                                             'ark:' + all_feats_path,
                                             os.path.join(directory, '0.dubm')],
                                             stderr=logf)
            gmm_init_proc.communicate()

        # Get subset of all feats
        subsample_feats_path = os.path.join(split_dir, 'cmvnonlinesplicetransformsubsamplefeats')
        log_path = os.path.join(split_dir, 'log', 'cmvnonlinesplicetransformsubsample.log')
        with open(log_path, 'w') as logf:
            with open(all_feats_path, 'r') as inf, open(subsample_feats_path, 'wb') as outf:
                subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                        '--n=' + str(config.subsample),
                                                        'ark:-',
                                                        'ark:-'],
                                                        stdin=inf,
                                                        stdout=outf,
                                                        stderr=logf)
                subsample_feats_proc.communicate()


        # Store Gaussian selection indices on disk
        gmm_gselect(directory, config, subsample_feats_path, self.num_jobs)

        # Training
        for i in range(config.num_iters):
            # Accumulate stats
            acc_global_stats(directory, config, subsample_feats_path, self.num_jobs, i)

            # Don't remove low-count Gaussians till the last tier,
            # or gselect info won't be valid anymore
            if i < config.num_iters-1:
                opt = '--remove-low-count-gaussians=false'
            else:
                opt = '--remove-low-count-gaussians=' + str(config.remove_low_count_gaussians)

            log_path = os.path.join(directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.num_jobs)]
                gmm_global_est_proc = subprocess.Popen([thirdparty_binary('gmm-global-est'),
                                                        opt,
                                                        '--min-gaussian-weight=' + str(config.min_gaussian_weight),
                                                        os.path.join(directory, '{}.dubm'.format(i)),
                                                        "{} - {}|".format(thirdparty_binary('gmm-global-sum-accs'),
                                                                          ' '.join(map(make_path_safe, acc_files))),
                                                        os.path.join(directory, '{}.dubm'.format(i+1))],
                                                        stderr=logf)
                gmm_global_est_proc.communicate()

        # Move files
        shutil.copy(os.path.join(directory, '{}.dubm'.format(config.num_iters)),
                    os.path.join(directory, 'final.dubm'))

    def ivector_extractor(self):
        '''
        Train i-vector extractor
        '''

        if not self.debug:
            if os.path.exists(self.ivector_extractor_final_model_path):
                print('i-vector training already done, using previous final.mdl')
                return

        os.makedirs(os.path.join(self.ivector_extractor_directory, 'log'), exist_ok=True)
        self._train_ivector_extractor()

    def _train_ivector_extractor(self):
        if not self.debug:
            if os.path.exists(self.ivector_extractor_final_model_path):
                print('i-vector extractor training already done, using previous final.ie')
                return

        log_dir = os.path.join(self.ivector_extractor_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        directory = self.ivector_extractor_directory
        split_dir = self.corpus.split_directory
        diag_ubm_path = self.diag_ubm_directory
        lda_mllt_path = self.lda_mllt_directory
        train_dir = self.corpus.output_directory
        config = self.ivector_extractor_config

        # Convert final.ubm to fgmm
        log_path = os.path.join(directory, 'log', 'global_to_fgmm.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-global-to-fgmm'),
                            os.path.join(diag_ubm_path, 'final.dubm'),
                            os.path.join(directory, '0.fgmm')],
                            stdout=subprocess.PIPE,
                            stderr=logf)

        # Initialize i-vector extractor
        log_path = os.path.join(directory, 'log', 'init.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('ivector-extractor-init'),
                            '--ivector-dim=' + str(config.ivector_dim),
                            '--use-weights=false',
                            os.path.join(directory, '0.fgmm'),
                            os.path.join(directory, '0.ie')],
                            stderr=logf)

        # Get GMM feats with online CMVN
        gmm_feats_path = os.path.join(split_dir, 'ivectorgmmfeats')
        log_path = os.path.join(split_dir, 'log', 'ivectorgmmfeats.log')
        with open(log_path, 'w') as logf:
            with open(gmm_feats_path, 'wb') as outf:
                apply_cmvn_online_proc = subprocess.Popen([thirdparty_binary('apply-cmvn-online'),
                                                          #'--config=' +
                                                          # This^ makes reference to a config file
                                                          # in Kaldi, but it's empty there
                                                          os.path.join(diag_ubm_path, 'global_cmvn.stats'),
                                                          'scp:' + train_dir + '/feats.scp',
                                                          'ark:-'],
                                                          stdout=subprocess.PIPE,
                                                          stderr=logf)
                splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats')]
                                                     + config.splice_opts +
                                                     ['ark:-', 'ark:-'],
                                                     stdin=apply_cmvn_online_proc.stdout,
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                transform_feats_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                                        os.path.join(lda_mllt_path, 'final.mat'),
                                                        'ark:-', 'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
                subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                        '--n=' + str(config.subsample),
                                                        'ark:-', 'ark:-'],
                                                        stdin=transform_feats_proc.stdout,
                                                        stdout=outf,
                                                        stderr=logf)
                subsample_feats_proc.communicate()


        # Do Gaussian selection and posterior extraction
        gauss_to_post(directory, config, diag_ubm_path, gmm_feats_path, self.num_jobs)

        # Get GMM feats without online CMVN
        feats_path = os.path.join(split_dir, 'ivectorfeats')
        log_path = os.path.join(split_dir, 'log', 'ivectorfeats.log')
        with open(log_path, 'w') as logf:
            with open(feats_path, 'wb') as outf:
                splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats')]
                                                     + config.splice_opts +
                                                     ['scp:' + os.path.join(train_dir, 'feats.scp'),
                                                     'ark:-'],
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                transform_feats_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                                        os.path.join(lda_mllt_path, 'final.mat'),
                                                        'ark:-', 'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
                subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                        '--n=' + str(config.subsample),
                                                        'ark:-', 'ark:-'],
                                                        stdin=transform_feats_proc.stdout,
                                                        stdout=outf,
                                                        stderr=logf)
                subsample_feats_proc.communicate()

        # Training loop
        for i in range(config.num_iters):

            # Accumulate stats and sum
            acc_ivector_stats(directory, config, feats_path, self.num_jobs, i)

            # Est extractor
            log_path = os.path.join(directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                extractor_est_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-est'),
                                                      os.path.join(directory, '{}.ie'.format(i)),
                                                      os.path.join(directory, 'acc.{}'.format(i)),
                                                      os.path.join(directory, '{}.ie'.format(i+1))],
                                                      stderr=logf)
                extractor_est_proc.communicate()
        # Rename to final
        shutil.copy(os.path.join(directory, '{}.ie'.format(config.num_iters)), os.path.join(directory, 'final.ie'))
import os
import shutil
import glob
import subprocess
import re
import io
import math
import numpy as np
from tqdm import tqdm
from shutil import copy, copyfile, rmtree, make_archive, unpack_archive
from contextlib import redirect_stdout
from aligner.models import IvectorExtractor
from random import shuffle

from ..helper import thirdparty_binary, make_path_safe, awk_like, filter_scp

from ..config import (MonophoneConfig, TriphoneConfig, TriphoneFmllrConfig,
                      LdaMlltConfig, DiagUbmConfig, iVectorExtractorConfig,
                      NnetBasicConfig)

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               lda_acc_stats,
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train,
                               nnet_align, nnet_get_align_feats, extract_ivectors,
                               compute_prob, get_average_posteriors, relabel_egs)
#from ..accuracy_graph import get_accuracy_graph


from ..exceptions import NoSuccessfulAlignments

from .. import __version__

from ..config import TEMP_DIR


class BaseAligner(object):
    '''
    Base aligner class for common aligner functions

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    output_directory : str
        Path to export aligned TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    mono_params : :class:`~aligner.config.MonophoneConfig`, optional
        Monophone training parameters to use, if different from defaults
    tri_params : :class:`~aligner.config.TriphoneConfig`, optional
        Triphone training parameters to use, if different from defaults
    tri_fmllr_params : :class:`~aligner.config.TriphoneFmllrConfig`, optional
        Speaker-adapted triphone training parameters to use, if different from defaults
    '''

    def __init__(self, corpus, dictionary, output_directory,
                 temp_directory=None, num_jobs=3, call_back=None,
                 mono_params=None, tri_params=None,
                 tri_fmllr_params=None, lda_mllt_params=None,
                 diag_ubm_params=None, ivector_extractor_params=None,
                 nnet_basic_params=None,
                 debug=False, skip_input=False, nnet=False):
        self.nnet = nnet

        if mono_params is None:
            mono_params = {}
        if tri_params is None:
            tri_params = {}
        if tri_fmllr_params is None:
            tri_fmllr_params = {}

        if lda_mllt_params is None:
            lda_mllt_params = {}
        if diag_ubm_params is None:
            diag_ubm_params = {}
        if ivector_extractor_params is None:
            ivector_extractor_params = {}
        if nnet_basic_params is None:
            nnet_basic_params = {}

        self.mono_config = MonophoneConfig(**mono_params)
        self.tri_config = TriphoneConfig(**tri_params)
        self.tri_fmllr_config = TriphoneFmllrConfig(**tri_fmllr_params)

        self.lda_mllt_config = LdaMlltConfig(**lda_mllt_params)
        self.diag_ubm_config = DiagUbmConfig(**diag_ubm_params)
        self.ivector_extractor_config = iVectorExtractorConfig(**ivector_extractor_params)
        self.nnet_basic_config = NnetBasicConfig(**nnet_basic_params)

        self.corpus = corpus
        self.dictionary = dictionary
        self.output_directory = output_directory
        self.num_jobs = num_jobs
        if self.corpus.num_jobs != num_jobs:
            self.num_jobs = self.corpus.num_jobs
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = temp_directory
        self.call_back = call_back
        if self.call_back is None:
            self.call_back = print
        self.verbose = False
        self.debug = debug
        self.skip_input = skip_input
        self.setup()

    def setup(self):
        self.dictionary.write()
        self.corpus.initialize_corpus(self.dictionary, skip_input=self.skip_input)
        print(self.corpus.speaker_utterance_info())

    @property
    def meta(self):
        data = {'phones':sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture':'gmm-hmm',
                'features':'mfcc+deltas',
                }
        return data

    @property
    def mono_directory(self):
        return os.path.join(self.temp_directory, 'mono')

    @property
    def mono_final_model_path(self):
        return os.path.join(self.mono_directory, 'final.mdl')

    @property
    def mono_ali_directory(self):
        return os.path.join(self.temp_directory, 'mono_ali')

    @property
    def tri_directory(self):
        return os.path.join(self.temp_directory, 'tri')

    @property
    def tri_ali_directory(self):
        return os.path.join(self.temp_directory, 'tri_ali')

    @property
    def tri_final_model_path(self):
        return os.path.join(self.tri_directory, 'final.mdl')

    @property
    def tri_fmllr_directory(self):
        return os.path.join(self.temp_directory, 'tri_fmllr')

    @property
    def tri_fmllr_ali_directory(self):
        return os.path.join(self.temp_directory, 'tri_fmllr_ali')

    @property
    def tri_fmllr_final_model_path(self):
        return os.path.join(self.tri_fmllr_directory, 'final.mdl')

    # Beginning of nnet properties
    @property
    def lda_mllt_directory(self):
        return os.path.join(self.temp_directory, 'lda_mllt')

    @property
    def lda_mllt_ali_directory(self):
        return os.path.join(self.temp_directory, 'lda_mllt_ali')

    @property
    def lda_mllt_final_model_path(self):
        return os.path.join(self.lda_mllt_directory, 'final.mdl')

    @property
    def diag_ubm_directory(self):
        return os.path.join(self.temp_directory, 'diag_ubm')

    @property
    def diag_ubm_final_model_path(self):
        return os.path.join(self.diag_ubm_directory, 'final.dubm')

    @property
    def ivector_extractor_directory(self):
        return os.path.join(self.temp_directory, 'ivector_extractor')

    @property
    def ivector_extractor_final_model_path(self):
        return os.path.join(self.ivector_extractor_directory, 'final.ie')

    @property
    def extracted_ivector_directory(self):
        return os.path.join(self.temp_directory, 'extracted_ivector')

    @property
    def nnet_basic_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic')

    @property
    def nnet_basic_ali_directory(self):
        return os.path.join(self.temp_directory, 'nnet_basic_ali')

    @property
    def nnet_basic_final_model_path(self):
        return os.path.join(self.nnet_basic_directory, 'final.mdl')

    # End of nnet properties

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        if os.path.exists(self.nnet_basic_final_model_path):
            model_directory = self.nnet_basic_directory
        elif os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory

        convert_ali_to_textgrids(self.output_directory, model_directory, self.dictionary,
                                 self.corpus, self.num_jobs)

    def get_num_gauss_mono(self):
        '''
        Get the number of gaussians for a monophone model
        '''
        with open(os.devnull, 'w') as devnull:
            proc = subprocess.Popen([thirdparty_binary('gmm-info'),
                                     '--print-args=false',
                                     os.path.join(self.mono_directory, '0.mdl')],
                                    stderr=devnull,
                                    stdout=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            num = stdout.decode('utf8')
            matches = re.search(r'gaussians (\d+)', num)
            num = int(matches.groups()[0])
        return num

    def _align_si(self, fmllr=False, lda_mllt=False, feature_name=None):
        '''
        Generate an alignment of the dataset
        '''
        if fmllr and os.path.exists(self.tri_fmllr_final_model_path):
            model_directory = self.tri_fmllr_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config

        elif fmllr:     # First pass with fmllr, final path doesn't exist yet
            model_directory = self.tri_directory
            output_directory = self.tri_fmllr_ali_directory
            config = self.tri_fmllr_config

        elif lda_mllt and os.path.exists(self.lda_mllt_final_model_path):
            model_directory = self.lda_mllt_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config

        elif lda_mllt:  # First pass with LDA + MLLT, final path doesn't exist yet
            model_directory = self.tri_fmllr_directory
            output_directory = self.lda_mllt_ali_directory
            config = self.lda_mllt_config
        elif os.path.exists(self.tri_final_model_path):
            model_directory = self.tri_directory
            output_directory = self.tri_ali_directory
            config = self.tri_config
        elif os.path.exists(self.mono_final_model_path):
            model_directory = self.mono_directory
            output_directory = self.mono_ali_directory
            config = self.mono_config

        optional_silence = self.dictionary.optional_silence_csl
        oov = self.dictionary.oov_int

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(self.tri_fmllr_ali_directory, exist_ok=True)
        os.makedirs(self.lda_mllt_ali_directory, exist_ok=True)

        os.makedirs(log_dir, exist_ok=True)

        shutil.copyfile(os.path.join(model_directory, 'tree'),
                        os.path.join(output_directory, 'tree'))
        shutil.copyfile(os.path.join(model_directory, 'final.mdl'),
                        os.path.join(output_directory, '0.mdl'))

        shutil.copyfile(os.path.join(model_directory, 'final.occs'),
                        os.path.join(output_directory, '0.occs'))

        feat_type = 'delta'

        compile_train_graphs(output_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs, debug=self.debug)

        align(0, output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, config, feature_name=feature_name)
        shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(output_directory, 'final.mdl'))
        shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(output_directory, 'final.occs'))

        if output_directory == self.tri_fmllr_ali_directory:
            os.makedirs(self.tri_fmllr_directory, exist_ok=True)
            shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(self.tri_fmllr_directory, 'final.mdl'))
            shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(self.tri_fmllr_directory, 'final.occs'))
        elif output_directory == self.lda_mllt_ali_directory:
            os.makedirs(self.lda_mllt_directory, exist_ok=True)
            shutil.copyfile(os.path.join(output_directory, '0.mdl'), os.path.join(self.lda_mllt_directory, 'final.mdl'))
            shutil.copyfile(os.path.join(output_directory, '0.occs'), os.path.join(self.lda_mllt_directory, 'final.occs'))

    def parse_log_directory(self, directory, iteration):
        '''
        Parse error files and relate relevant information about unaligned files
        '''
        if not self.verbose:
            return
        error_regex = re.compile(r'Did not successfully decode file (\w+),')
        too_little_data_regex = re.compile(
            r'Gaussian has too little data but not removing it because it is the last Gaussian')
        skipped_transition_regex = re.compile(r'(\d+) out of (\d+) transition-states skipped due to insuffient data')

        log_like_regex = re.compile(r'Overall avg like per frame = ([-0-9.]+|nan) over (\d+) frames')
        error_files = []
        for i in range(self.num_jobs):
            path = os.path.join(directory, 'align.{}.{}.log'.format(iteration - 1, i))
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                error_files.extend(error_regex.findall(f.read()))
        update_path = os.path.join(directory, 'update.{}.log'.format(iteration))
        if os.path.exists(update_path):
            with open(update_path, 'r') as f:
                data = f.read()
                m = log_like_regex.search(data)
                if m is not None:
                    log_like, tot_frames = m.groups()
                    if log_like == 'nan':
                        raise (NoSuccessfulAlignments('Could not align any files.  Too little data?'))
                    self.call_back('log-likelihood', float(log_like))
                skipped_transitions = skipped_transition_regex.search(data)
                self.call_back('skipped transitions', *skipped_transitions.groups())
                num_too_little_data = len(too_little_data_regex.findall(data))
                self.call_back('missing data gaussians', num_too_little_data)
        if error_files:
            self.call_back('could not align', error_files)

    def _align_fmllr(self):
        '''
        Align the dataset using speaker-adapted transforms
        '''
        model_directory = self.tri_directory        # Get final.mdl from here
        first_output_directory = self.tri_ali_directory
        second_output_directory = self.tri_fmllr_ali_directory
        self._align_si(fmllr=False)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(first_output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        calc_fmllr(first_output_directory, self.corpus.split_directory,
                   sil_phones, self.num_jobs, self.tri_fmllr_config, initial=True)
        optional_silence = self.dictionary.optional_silence_csl
        align(0, first_output_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, self.tri_fmllr_config)

        # Copy into the "correct" tri_fmllr_ali output directory
        for file in glob.glob(os.path.join(first_output_directory, 'ali.*')):
            shutil.copy(file, second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'tree'), second_output_directory)
        shutil.copy(os.path.join(first_output_directory, 'final.mdl'), second_output_directory)


    def _init_tri(self, fmllr=False):
        if fmllr:
            config = self.tri_fmllr_config
            directory = self.tri_fmllr_directory
            align_directory = self.tri_ali_directory
        else:
            config = self.tri_config
            directory = self.tri_directory
            align_directory = self.mono_ali_directory

        if not self.debug:
            if os.path.exists(os.path.join(directory, '1.mdl')):
                return

        if fmllr:
            print('Initializing speaker-adapted triphone training...')
        else:
            print('Initializing triphone training...')
        context_opts = []
        ci_phones = self.dictionary.silence_csl

        tree_stats(directory, align_directory,
                   self.corpus.split_directory, ci_phones, self.num_jobs)
        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)

        log_path = os.path.join(directory, 'log', 'mixup.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-mixup'),
                             '--mix-up={}'.format(config.initial_gauss_count),
                             mdl_path, occs_path, mdl_path], stderr=logf)

        #os.remove(treeacc_path)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        shutil.copy(occs_path, os.path.join(directory, '1.occs'))
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))

        convert_alignments(directory, align_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):
            for i in range(self.num_jobs):
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))



    def train_tri_fmllr(self):
        '''
        Perform speaker-adapted triphone training
        '''
        if not self.debug:
            if os.path.exists(self.tri_fmllr_final_model_path):
                print('Triphone FMLLR training already done, using previous final.mdl')
                return

        if not os.path.exists(self.tri_ali_directory):
            self._align_fmllr()

        #self._align_fmllr()

        os.makedirs(os.path.join(self.tri_fmllr_directory, 'log'), exist_ok=True)
        self._init_tri(fmllr=True)
        self._do_tri_fmllr_training()

    def _do_tri_fmllr_training(self):
        self.call_back('Beginning speaker-adapted triphone training...')
        self._do_training(self.tri_fmllr_directory, self.tri_fmllr_config)

    def _do_training(self, directory, config):
        if config.realign_iters is None:
            config.realign_iters = list(range(0, config.num_iters, 10))
        num_gauss = config.initial_gauss_count
        sil_phones = self.dictionary.silence_csl
        inc_gauss = config.inc_gauss_count
        if self.call_back == print:
            iters = tqdm(range(1, config.num_iters))
        else:
            iters = range(1, config.num_iters)
        log_directory = os.path.join(directory, 'log')
        for i in iters:
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            occs_path = os.path.join(directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            if not self.debug:
                if os.path.exists(next_model_path):
                    continue

            if i in config.realign_iters:
                align(i, directory, self.corpus.split_directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config,
                      feature_name='cmvnsplicetransformfeats')
            if config.do_fmllr and i in config.fmllr_iters:
                calc_fmllr(directory, self.corpus.split_directory, sil_phones,
                           self.num_jobs, config, initial=False, iteration=i)

            if config.do_lda_mllt and i <= config.num_iters:
                calc_lda_mllt(directory, self.corpus.split_directory,   # Could change this to make ali directory later
                #calc_lda_mllt(self.lda_mllt_ali_directory, sil_phones,
                              self.lda_mllt_directory, sil_phones,
                              self.num_jobs, config, config.num_iters,
                              initial=False, iteration=i, corpus=self.corpus)


            acc_stats(i, directory, self.corpus.split_directory, self.num_jobs,
                      config.do_fmllr, do_lda_mllt=config.do_lda_mllt)
            log_path = os.path.join(log_directory, 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.num_jobs)]
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                             '--write-occs=' + occs_path,
                                             '--mix-up=' + str(num_gauss), '--power=' + str(config.power),
                                             model_path,
                                             "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                               ' '.join(map(make_path_safe, acc_files))),
                                             next_model_path],
                                            stderr=logf)
                est_proc.communicate()
            self.parse_log_directory(log_directory, i)
            if i < config.max_iter_inc:
                num_gauss += inc_gauss

        shutil.copy(os.path.join(directory, '{}.mdl'.format(config.num_iters)),
                    os.path.join(directory, 'final.mdl'))

        shutil.copy(os.path.join(directory, '{}.occs'.format(config.num_iters)),
                    os.path.join(directory, 'final.occs'))

        if config.do_lda_mllt:
            shutil.copy(os.path.join(directory, '{}.mat'.format(config.num_iters-1)),
                        os.path.join(directory, 'final.mat'))

    def train_lda_mllt(self):
        '''
        Perform LDA + MLLT training
        '''

        if not self.debug:
            if os.path.exists(self.lda_mllt_final_model_path):
                print('LDA + MLLT training already done, using previous final.mdl')
                return

        # N.B: The function _align_lda_mllt() is half-developed, but there doesn't seem to
        # be a reason for it to actually ever be called (since people will always have
        # fmllr done immediately before in the pipeline. Can clean/delete later if determined
        # that we need to actually use it somewhere or not).
        #if not os.path.exists(self.lda_mllt_ali_directory):
        #    self._align_lda_mllt()
        #self._align_lda_mllt()  # half implemented, can come back later or make people run from fmllr

        os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)

        self._init_lda_mllt()
        self._do_lda_mllt_training()

    def _init_lda_mllt(self):
        '''
        Initialize LDA + MLLT training.
        '''
        config = self.lda_mllt_config
        directory = self.lda_mllt_directory
        align_directory = self.tri_fmllr_ali_directory  # The previous
        mdl_dir = self.tri_fmllr_directory

        if not self.debug:
            if os.path.exists(os.path.join(directory, '1.mdl')):
                return

        print('Initializing LDA + MLLT training...')

        context_opts = []
        ci_phones = self.dictionary.silence_csl

        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')

        final_mdl_path = os.path.join(self.tri_fmllr_directory)

        # Accumulate LDA stats
        lda_acc_stats(directory, self.corpus.split_directory, align_directory, config, ci_phones, self.num_jobs)

        # Accumulating tree stats
        self.corpus._norm_splice_transform_feats(self.lda_mllt_directory)
        tree_stats(directory, align_directory, self.corpus.split_directory, ci_phones,
                   self.num_jobs, feature_name='cmvnsplicetransformfeats')

        # Getting questions for tree clustering
        log_path = os.path.join(directory, 'log', 'cluster_phones.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        # Building the tree
        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        # Initializing the model
        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))
        shutil.copy(occs_path, os.path.join(directory, '1.occs'))

        convert_alignments(directory, align_directory, self.num_jobs)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):            
            for i in range(self.num_jobs):                                      
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                            os.path.join(directory, 'trans.{}'.format(i)))

    def _do_lda_mllt_training(self):
        self.call_back('Beginning LDA + MLLT training...')
        self._do_training(self.lda_mllt_directory, self.lda_mllt_config)

    def train_nnet_basic(self):
        '''
        Perform neural network training
        '''

        os.makedirs(os.path.join(self.nnet_basic_directory, 'log'), exist_ok=True)

        split_directory = self.corpus.split_directory
        config = self.nnet_basic_config
        tri_fmllr_config = self.tri_fmllr_config
        directory = self.nnet_basic_directory
        nnet_align_directory = self.nnet_basic_ali_directory
        align_directory = self.tri_fmllr_ali_directory
        lda_directory = self.lda_mllt_directory
        egs_directory = os.path.join(directory, 'egs')
        training_directory = self.corpus.output_directory

        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        L_fst_path = os.path.join(self.dictionary.output_directory, 'L.fst')
        ali_tree_path = os.path.join(align_directory, 'tree')
        shutil.copy(ali_tree_path, os.path.join(directory, 'tree'))

        mdl_path = os.path.join(align_directory, 'final.mdl')
        raw_feats = os.path.join(training_directory, 'feats.scp')

        tree_info_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                                          os.path.join(align_directory, 'tree')],
                                          stdout=subprocess.PIPE)
        tree_info = tree_info_proc.stdout.read()
        tree_info = tree_info.split()
        num_leaves = tree_info[1]
        num_leaves = num_leaves.decode("utf-8")

        lda_dim = self.lda_mllt_config.dim 

        # Extract iVectors
        self._extract_ivectors()

        # Get LDA matrix
        fixed_ivector_dir = self.extracted_ivector_directory
        get_lda_nnet(directory, align_directory, fixed_ivector_dir, training_directory,
                     split_directory, raw_feats, self.dictionary.optional_silence_csl, config, self.num_jobs)

        log_path = os.path.join(directory, 'log', 'lda_matrix.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(directory, 'lda.{}.acc'.format(x))
                         for x in range(self.num_jobs)]
            sum_lda_accs_proc = subprocess.Popen([thirdparty_binary('sum-lda-accs'),
                                                 os.path.join(directory, 'lda.acc')]
                                                 + acc_files,
                                                 stderr=logf)
            sum_lda_accs_proc.communicate()

            lda_mat_proc = subprocess.Popen([thirdparty_binary('nnet-get-feature-transform'),
                                            '--dim=' + str(lda_dim),
                                            os.path.join(directory, 'lda.mat'),
                                            os.path.join(directory, 'lda.acc')],
                                            stderr=logf)
            lda_mat_proc.communicate()
        lda_mat_path = os.path.join(directory, 'lda.mat')


        # Get examples for training
        os.makedirs(egs_directory, exist_ok=True)

        # # Get valid uttlist and train subset uttlist
        valid_uttlist = os.path.join(directory, 'valid_uttlist')
        train_subset_uttlist = os.path.join(directory, 'train_subset_uttlist')
        training_feats = os.path.join(directory, 'nnet_training_feats')
        num_utts_subset = 300
        log_path = os.path.join(directory, 'log', 'training_egs_feats.log')

        with open(log_path, 'w') as logf:
            with open(valid_uttlist, 'w') as outf:
                # Get first column from utt2spk (awk-like)
                utt2spk_col = awk_like(os.path.join(training_directory, 'utt2spk'), 0)
                # Shuffle the list from the column
                shuffle(utt2spk_col)
                # Take only the first num_utts_subset lines
                utt2spk_col = utt2spk_col[:num_utts_subset]
                # Write the result to file
                for line in utt2spk_col:
                    outf.write(line)
                    outf.write('\n')

            with open(train_subset_uttlist, 'w') as outf:
                # Get first column from utt2spk (awk-like)
                utt2spk_col = awk_like(os.path.join(training_directory, 'utt2spk'), 0)
                # Filter by the scp list
                filtered = filter_scp(valid_uttlist, utt2spk_col, exclude=True)
                # Shuffle the list
                shuffle(filtered)
                # Take only the first num_utts_subset lines
                filtered = filtered[:num_utts_subset]
                # Write the result to a file
                for line in filtered:
                    outf.write(line)
                    outf.write('\n')

        get_egs(directory, egs_directory, training_directory, split_directory, align_directory,
                fixed_ivector_dir, training_feats, valid_uttlist,
                train_subset_uttlist, config, self.num_jobs)

        # Initialize neural net
        print('Beginning DNN training...')
        stddev = float(1.0/config.pnorm_input_dim**0.5)
        online_preconditioning_opts = 'alpha={} num-samples-history={} update-period={} rank-in={} rank-out={} max-change-per-sample={}'.format(config.alpha, config.num_samples_history, config.update_period, config.precondition_rank_in, config.precondition_rank_out, config.max_change_per_sample)
        nnet_config_path = os.path.join(directory, 'nnet.config')
        hidden_config_path = os.path.join(directory, 'hidden.config')
        ivector_dim_path = os.path.join(directory, 'ivector_dim')
        with open(ivector_dim_path, 'r') as inf:
            ivector_dim = inf.read().strip()
        feat_dim = 13 + int(ivector_dim)

        with open(nnet_config_path, 'w') as nc:
            nc.write('SpliceComponent input-dim={} left-context={} right-context={} const-component-dim={}\n'.format(feat_dim, config.splice_width, config.splice_width, ivector_dim))
            nc.write('FixedAffineComponent matrix={}\n'.format(lda_mat_path))
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(lda_dim, config.pnorm_input_dim, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(config.pnorm_input_dim, config.pnorm_output_dim, config.p))
            nc.write('NormalizeComponent dim={}\n'.format(config.pnorm_output_dim))
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.pnorm_output_dim, num_leaves, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('SoftmaxComponent dim={}\n'.format(num_leaves))

        with open(hidden_config_path, 'w') as nc:
            nc.write('AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.pnorm_output_dim, config.pnorm_input_dim, online_preconditioning_opts, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(config.pnorm_input_dim, config.pnorm_output_dim, config.p))
            nc.write('NormalizeComponent dim={}\n'.format(config.pnorm_output_dim))

        log_path = os.path.join(directory, 'log', 'nnet_init.log')
        nnet_info_path = os.path.join(directory, 'log', 'nnet_info.log')
        with open(log_path, 'w') as logf:
            with open(nnet_info_path, 'w') as outf:
                nnet_am_init_proc = subprocess.Popen([thirdparty_binary('nnet-am-init'),
                                                     os.path.join(align_directory, 'tree'),
                                                     topo_path,
                                                     "{} {} -|".format(thirdparty_binary('nnet-init'),
                                                                       nnet_config_path),
                                                    os.path.join(directory, '0.mdl')],
                                                    stderr=logf)
                nnet_am_init_proc.communicate()

                nnet_am_info = subprocess.Popen([thirdparty_binary('nnet-am-info'),
                                                os.path.join(directory, '0.mdl')],
                                                stdout=outf,
                                                stderr=logf)
                nnet_am_info.communicate()


        # Train transition probabilities and set priors
        #   First combine all previous alignments
        ali_files = glob.glob(os.path.join(align_directory, 'ali.*'))
        prev_ali_path = os.path.join(directory, 'prev_ali.')
        with open(prev_ali_path, 'wb') as outfile:
            for ali_file in ali_files:
                with open(os.path.join(align_directory, ali_file), 'rb') as infile:
                    for line in infile:
                        outfile.write(line)
        nnet_train_trans(directory, align_directory, prev_ali_path, self.num_jobs)

        # Get iteration at which we will mix up
        num_tot_iters = config.num_epochs * config.iters_per_epoch
        finish_add_layers_iter = config.num_hidden_layers * config.add_layers_period
        first_modify_iter = finish_add_layers_iter + config.add_layers_period
        mix_up_iter = (num_tot_iters + finish_add_layers_iter)/2

        # Get iterations at which we will realign
        realign_iters = []
        if config.realign_times != 0:
            div = config.realign_times + 1 # (e.g. realign 2 times = iterations split into 3 sets)
            split_iters = np.array_split(range(num_tot_iters), div)
            for group in split_iters:
                realign_iters.append(group[0])

        # Training loop
        for i in range(num_tot_iters):
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            # Combine all examples (could integrate validation diagnostics, etc., later-- see egs functions)
            egs_files = []
            for file in os.listdir(egs_directory):
                if file.startswith('egs'):
                    egs_files.append(file)
            with open(os.path.join(egs_directory, 'all_egs.egs'), 'wb') as outfile:
                for egs_file in egs_files:
                    with open(os.path.join(egs_directory, egs_file), 'rb') as infile:
                        for line in infile:
                            outfile.write(line)

            # Get accuracy rates for the current iteration (to pull out graph later)
            #compute_prob(i, directory, egs_directory, model_path, self.num_jobs)
            log_path = os.path.join(directory, 'log', 'compute_prob_train.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                compute_prob_proc = subprocess.Popen([thirdparty_binary('nnet-compute-prob'),
                                                     model_path,
                                                     'ark:{}/all_egs.egs'.format(egs_directory)],
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                log_prob = compute_prob_proc.stdout.read().decode('utf-8').strip()
                compute_prob_proc.communicate()

            print("Iteration {} of {} \t\t Log-probability: {}".format(i+1, num_tot_iters, log_prob))

            # Pull out and save graphs
            # This is not quite working when done automatically - to be worked out with unit testing.
            #get_accuracy_graph(os.path.join(directory, 'log'), os.path.join(directory, 'log'))

            # If it is NOT the first iteration,
            # AND we still have layers to add,
            # AND it's the right time to add a layer...
            if i > 0 and i <= ((config.num_hidden_layers-1)*config.add_layers_period) and ((i-1)%config.add_layers_period) == 0:
                # Add a new hidden layer
                mdl = os.path.join(directory, 'tmp{}.mdl'.format(i))
                log_path = os.path.join(directory, 'log', 'temp_mdl.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    with open(mdl, 'w') as outf:
                        tmp_mdl_init_proc = subprocess.Popen([thirdparty_binary('nnet-init'),
                                                            '--srand={}'.format(i),
                                                            os.path.join(directory, 'hidden.config'),
                                                            '-'],
                                                            stdout=subprocess.PIPE,
                                                            stderr=logf)
                        tmp_mdl_ins_proc = subprocess.Popen([thirdparty_binary('nnet-insert'),
                                                            os.path.join(directory, '{}.mdl'.format(i)),
                                                            '-', '-'],
                                                            stdin=tmp_mdl_init_proc.stdout,
                                                            stdout=outf,
                                                            stderr=logf)
                        tmp_mdl_ins_proc.communicate()

            # Otherwise just use the past model
            else:
                mdl = os.path.join(directory, '{}.mdl'.format(i))

            # Shuffle examples and train nets with SGD
            nnet_train(directory, egs_directory, mdl, i, self.num_jobs)

            # Get nnet list from the various jobs on this iteration
            nnets_list = [os.path.join(directory, '{}.{}.mdl'.format((i+1), x))
                         for x in range(self.num_jobs)]

            if (i+1) >= num_tot_iters:
                learning_rate = config.final_learning_rate
            else:
                learning_rate = config.initial_learning_rate * math.exp(i * math.log(config.final_learning_rate/config.initial_learning_rate)/num_tot_iters)

            log_path = os.path.join(directory, 'log', 'average.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                nnet_avg_proc = subprocess.Popen([thirdparty_binary('nnet-am-average')]
                                                 + nnets_list
                                                 + ['-'],
                                                 stdout=subprocess.PIPE,
                                                 stderr=logf)
                nnet_copy_proc = subprocess.Popen([thirdparty_binary('nnet-am-copy'),
                                                  '--learning-rate={}'.format(learning_rate),
                                                  '-',
                                                  os.path.join(directory, '{}.mdl'.format(i+1))],
                                                  stdin=nnet_avg_proc.stdout,
                                                  stderr=logf)
                nnet_copy_proc.communicate()

            # If it's the right time, do mixing up
            if config.mix_up > 0 and i == mix_up_iter:
                log_path = os.path.join(directory, 'log', 'mix_up.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_am_mixup_proc = subprocess.Popen([thirdparty_binary('nnet-am-mixup'),
                                                          '--min-count=10',
                                                          '--num-mixtures={}'.format(config.mix_up),
                                                          os.path.join(directory, '{}.mdl'.format(i+1)),
                                                          os.path.join(directory, '{}.mdl'.format(i+1))],
                                                          stderr=logf)
                    nnet_am_mixup_proc.communicate()

            # Realign if it's the right time
            if i in realign_iters:
                prev_egs_directory = egs_directory
                egs_directory = os.path.join(directory, 'egs_{}'.format(i))
                os.makedirs(egs_directory, exist_ok=True)

                #   Get average posterior for purposes of adjusting priors
                get_average_posteriors(i, directory, prev_egs_directory, config, self.num_jobs)
                log_path = os.path.join(directory, 'log', 'vector_sum_exterior.{}.log'.format(i))
                vectors_to_sum = glob.glob(os.path.join(directory, 'post.{}.*.vec'.format(i)))

                with open(log_path, 'w') as logf:
                    vector_sum_proc = subprocess.Popen([thirdparty_binary('vector-sum')]
                                                       + vectors_to_sum
                                                       + [os.path.join(directory, 'post.{}.vec'.format(i))
                                                       ],
                                                       stderr=logf)
                    vector_sum_proc.communicate()

                #   Readjust priors based on computed posteriors
                log_path = os.path.join(directory, 'log', 'adjust_priors.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_adjust_priors_proc = subprocess.Popen([thirdparty_binary('nnet-adjust-priors'),
                                                               os.path.join(directory, '{}.mdl'.format(i)),
                                                               os.path.join(directory, 'post.{}.vec'.format(i)),
                                                               os.path.join(directory, '{}.mdl'.format(i))],
                                                               stderr=logf)
                    nnet_adjust_priors_proc.communicate()

                #   Realign:
                #       Compile train graphs (gets fsts.{} for alignment)
                compile_train_graphs(directory, self.dictionary.output_directory,
                                     self.corpus.split_directory, self.num_jobs)

                #       Get alignment feats
                nnet_get_align_feats(directory, self.corpus.split_directory, fixed_ivector_dir, config, self.num_jobs)

                #       Do alignment
                nnet_align(i, directory,
                      self.dictionary.optional_silence_csl,
                      self.num_jobs, config)

                #     Finally, relabel the egs
                ali_files = glob.glob(os.path.join(directory, 'ali.*'))
                alignments = os.path.join(directory, 'alignments.')
                with open(alignments, 'wb') as outfile:
                    for ali_file in ali_files:
                        with open(os.path.join(directory, ali_file), 'rb') as infile:
                            for line in infile:
                                outfile.write(line)
                relabel_egs(i, directory, prev_egs_directory, alignments, egs_directory, self.num_jobs)


        # Rename the final model
        shutil.copy(os.path.join(directory, '{}.mdl'.format(num_tot_iters-1)), os.path.join(directory, 'final.mdl'))

        # Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        # Get alignment feats
        nnet_get_align_feats(directory, self.corpus.split_directory, fixed_ivector_dir, config, self.num_jobs)

        # Do alignment
        nnet_align("final", directory,
              self.dictionary.optional_silence_csl,
              self.num_jobs, config, mdl=os.path.join(directory, 'final.mdl'))

    def _extract_ivectors(self):
        '''
        Extracts i-vectors from a corpus using the trained i-vector extractor.
        '''
        print('Extracting i-vectors...')

        log_dir = os.path.join(self.extracted_ivector_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        # To do still for release: maybe add arguments to command line to tell MFA which
        # i-vector extractor to use.

        directory = self.extracted_ivector_directory

        # Only one option for now - make this an argument eventually.
        # Librispeech 100 chosen because of large number of speakers, not necessarily longer length. 
        # Thesis results tentatively confirmed that more speakers in ivector extractor => better results.
        ivector_extractor = IvectorExtractor(os.path.join(os.path.dirname(__file__), '../../pretrained_models/ls_100_ivector_extractor.zip'))
        ivector_extractor_directory = os.path.join(self.temp_directory, 'ivector_extractor')
        ivector_extractor.export_ivector_extractor(ivector_extractor_directory)

        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        config = self.ivector_extractor_config
        training_directory = self.corpus.output_directory

        # To make a directory for corpus with just 2 utterances per speaker
        # (left commented out in case we ever decide to do this)
        """max2_dir = os.path.join(directory, 'max2')
        os.makedirs(max2_dir, exist_ok=True)
        mfa_working_dir = os.getcwd()
        os.chdir("/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2")
        copy_data_sh = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2/copy_data_dir.sh"
        log_path = os.path.join(directory, 'log', 'max2.log')
        with open(log_path, 'w') as logf:
            command = [copy_data_sh, '--utts-per-spk-max', '2', train_dir, max2_dir]
            max2_proc = subprocess.Popen(command,
                                         stderr=logf)
            max2_proc.communicate()
        os.chdir(mfa_working_dir)"""

        # Write a "cmvn config" file (this is blank in the actual kaldi code, but it needs the argument passed)
        cmvn_config = os.path.join(directory, 'online_cmvn.conf')
        with open(cmvn_config, 'w') as cconf:
            cconf.write("")

        # Write a "splice config" file
        splice_config = os.path.join(directory, 'splice.conf')
        with open(splice_config, 'w') as sconf:
            sconf.write(config.splice_opts[0])
            sconf.write('\n')
            sconf.write(config.splice_opts[1])

        # Write a "config" file to input to the extraction binary
        ext_config = os.path.join(directory, 'ivector_extractor.conf')
        with open(ext_config, 'w') as ieconf:
            ieconf.write('--cmvn-config={}\n'.format(cmvn_config))
            ieconf.write('--ivector-period={}\n'.format(config.ivector_period))
            ieconf.write('--splice-config={}\n'.format(splice_config))
            ieconf.write('--lda-matrix={}\n'.format(os.path.join(ivector_extractor_directory, 'final.mat')))
            ieconf.write('--global-cmvn-stats={}\n'.format(os.path.join(ivector_extractor_directory, 'global_cmvn.stats')))
            ieconf.write('--diag-ubm={}\n'.format(os.path.join(ivector_extractor_directory, 'final.dubm')))
            ieconf.write('--ivector-extractor={}\n'.format(os.path.join(ivector_extractor_directory, 'final.ie')))
            ieconf.write('--num-gselect={}\n'.format(config.num_gselect))
            ieconf.write('--min-post={}\n'.format(config.min_post))
            ieconf.write('--posterior-scale={}\n'.format(config.posterior_scale))
            ieconf.write('--max-remembered-frames=1000\n')
            ieconf.write('--max-count={}\n'.format(0))

        # Extract i-vectors
        extract_ivectors(directory, training_directory, ext_config, config, self.num_jobs)

        # Combine i-vectors across jobs
        file_list = []
        for j in range(self.num_jobs):
            file_list.append(os.path.join(directory, 'ivector_online.{}.scp'.format(j)))

        with open(os.path.join(directory, 'ivector_online.scp'), 'w') as outfile:
            for fname in file_list:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
