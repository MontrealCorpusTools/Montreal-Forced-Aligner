import os
import logging
import librosa
import soundfile
import re
import subprocess
from collections import defaultdict, Counter

from ..exceptions import SampleRateError, CorpusError
from ..helper import thirdparty_binary, load_text, load_scp, output_mapping, save_groups, filter_scp


def get_n_channels(file_path):
    """
    Return the number of channels for a sound file

    Parameters
    ----------
    file_path : str
        Path to a wav file

    Returns
    -------
    int
        Number of channels (1 if mono, 2 if stereo)
    """

    with soundfile.SoundFile(file_path, 'r') as inf:
        n_channels = inf.channels
        subtype = inf.subtype
        if not subtype.startswith('PCM'):
            raise SampleRateError('The file {} is not a PCM file.'.format(file_path))
    return n_channels


def get_sample_rate(file_path):
    return librosa.get_samplerate(file_path)


def get_bit_depth(file_path):
    with soundfile.SoundFile(file_path, 'r') as inf:
        subtype = inf.subtype
        bit_depth = int(subtype.replace('PCM_', ''))
    return bit_depth


def get_wav_duration(file_path):
    return librosa.get_duration(filename=file_path)


def extract_temp_channels(wav_path, temp_directory):
    """
    Extract a single channel from a stereo file to a new mono wav file

    Parameters
    ----------
    wav_path : str
        Path to stereo wav file
    temp_directory : str
        Directory to save extracted
    """
    name, ext = os.path.splitext(wav_path)
    base = os.path.basename(name)
    a_path = os.path.join(temp_directory, base + '_A.wav')
    b_path = os.path.join(temp_directory, base + '_B.wav')
    if not os.path.exists(a_path):
        with soundfile.SoundFile(wav_path, 'r') as inf:
            sr = inf.samplerate
            sound_format = inf.format
            endian = inf.endian
            subtype = inf.subtype
        stream = librosa.stream(wav_path,
                                block_length=256,
                                frame_length=2048,
                                hop_length=2048, mono=False)
        with soundfile.SoundFile(a_path, 'w', samplerate=sr, channels=1, endian=endian, subtype=subtype, format=sound_format) as af, \
             soundfile.SoundFile(b_path, 'w', samplerate=sr, channels=1, endian=endian, subtype=subtype, format=sound_format) as bf:

            for s in stream:
                af.write(s[0, :])
                bf.write(s[1, :])
    return a_path, b_path


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

    def __init__(self, directory, output_directory,
                 speaker_characters=0,
                 num_jobs=3, debug=False):
        self.debug = debug
        log_dir = os.path.join(output_directory, 'logging')
        os.makedirs(log_dir, exist_ok=True)
        self.name = os.path.basename(directory)
        self.log_file = os.path.join(log_dir, 'corpus.log')
        self.logger = logging.getLogger('corpus_setup')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file, 'w', 'utf-8')
        handler.setFormatter = logging.Formatter('%(name)s %(message)s')
        self.logger.addHandler(handler)
        if not os.path.exists(directory):
            raise CorpusError('The directory \'{}\' does not exist.'.format(directory))
        if not os.path.isdir(directory):
            raise CorpusError('The specified path for the corpus ({}) is not a directory.'.format(directory))
        if num_jobs < 1:
            num_jobs = 1

        print('Setting up corpus information...')
        self.logger.info('Setting up corpus information...')
        self.directory = directory
        self.output_directory = os.path.join(output_directory, 'corpus_data')
        self.temp_directory = os.path.join(self.output_directory, 'temp')
        os.makedirs(self.temp_directory, exist_ok=True)
        if speaker_characters == 0:
            self.speaker_directories = True
        else:
            self.speaker_directories = False
        self.num_jobs = num_jobs
        self.sample_rates = defaultdict(set)
        self.unsupported_sample_rate = []
        self.wav_files = []
        self.wav_durations = {}
        self.unsupported_bit_depths = []
        self.wav_read_errors = []
        self.speak_utt_mapping = defaultdict(list)
        self.utt_speak_mapping = {}
        self.utt_wav_mapping = {}
        self.feat_mapping = {}
        self.cmvn_mapping = {}
        self.file_directory_mapping = {}
        self.groups = []
        self.speaker_groups = []
        self.frequency_configs = []
        self.segments = {}
        self.ignored_utterances = []
        self.utterance_lengths = {}
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if os.path.exists(feat_path):
            self.feat_mapping = load_scp(feat_path)

    def find_best_groupings(self):
        if self.segments:
            ratio = len(self.segments.keys()) / len(self.utt_speak_mapping.keys())
            segment_job_num = int(ratio * self.num_jobs)
            if segment_job_num == 0:
                segment_job_num = 1
        else:
            segment_job_num = 0
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

    @property
    def utterances(self):
        return list(self.utt_speak_mapping.keys())

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

    def parse_features_logs(self):
        line_reg_ex = r'Did not find features for utterance (\w+)'
        missing_features = []
        with open(os.path.join(self.features_log_directory, 'cmvn.log'), 'r') as f:
            for line in f:
                m = re.search(line_reg_ex, line)
                if m is not None:
                    missing_features.append(m.groups()[0])

    def speaker_utterance_info(self):
        num_speakers = len(self.speak_utt_mapping.keys())
        average_utterances = sum(len(x) for x in self.speak_utt_mapping.values()) / num_speakers
        msg = 'Number of speakers in corpus: {}, average number of utterances per speaker: {}'.format(num_speakers,
                                                                                                      average_utterances)
        self.logger.info(msg)
        return msg

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
        return get_wav_duration(wav_path)

    def split_directory(self):
        directory = os.path.join(self.output_directory, 'split{}'.format(self.num_jobs))
        return directory

    def _write_utt_speak(self):
        utt2spk = os.path.join(self.output_directory, 'utt2spk')
        output_mapping(self.utt_speak_mapping, utt2spk)

    def _write_speak_utt(self):
        spk2utt = os.path.join(self.output_directory, 'spk2utt')
        output_mapping(self.speak_utt_mapping, spk2utt)

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

    def write(self):
        self._write_speak_utt()
        self._write_utt_speak()
        self._write_wavscp()

    def split(self):
        split_dir = self.split_directory()
        os.makedirs(os.path.join(split_dir, 'log'), exist_ok=True)
        self.logger.info('Setting up training data...')
        print('Setting up corpus_data directory...')
        self._split_wavs(split_dir)
        self._split_utt2spk(split_dir)
        self._split_spk2utt(split_dir)
        self._split_segments(split_dir)
