import os
import logging
import soundfile
import re
import subprocess
import shutil
from collections import defaultdict

from ..exceptions import SoxError, CorpusError
from ..helper import thirdparty_binary, load_scp, output_mapping, save_groups, filter_scp


supported_audio_extensions = ['.flac', '.ogg', '.aiff', '.mp3']


def get_wav_info(file_path, sample_rate=16000):
    if file_path.endswith('.mp3'):
        if not shutil.which('soxi'):
            raise SoxError('No sox found')
        sox_proc = subprocess.Popen(['soxi', '-D', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        if stderr.startswith('soxi FAIL formats'):
            raise SoxError('No support for mp3 in sox')
        return_dict = {'duration': float(stdout.strip()), 'format': 'MP3'}
        sox_proc = subprocess.Popen(['soxi', '-r', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        return_dict['sample_rate'] = int(stdout.strip())
        sox_proc = subprocess.Popen(['soxi', '-c', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        return_dict['num_channels'] = int(stdout.strip())
        sox_proc = subprocess.Popen(['soxi', '-p', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        return_dict['bit_depth'] = int(stdout.strip())
        use_sox = True
    else:
        with soundfile.SoundFile(file_path, 'r') as inf:
            subtype = inf.subtype
            if subtype == 'FLOAT':
                bit_depth = 32
            else:
                bit_depth = int(subtype.split('_')[-1])
            frames = inf.frames
            sr = inf.samplerate
            duration = frames / sr
            return_dict = {'num_channels': inf.channels, 'type': inf.subtype, 'bit_depth': bit_depth,
                           'sample_rate': sr, 'duration': duration, 'format': inf.format}
        use_sox = False
        if bit_depth != 16:
            use_sox = True
        if return_dict['format'] != 'WAV':
            use_sox = True
        if not subtype.startswith('PCM'):
            use_sox = True
    if use_sox:
        return_dict['sox_string'] = 'sox {} -t wav -b 16 -r {} - |'.format(file_path, sample_rate)
    return return_dict


def find_exts(files):
    wav_files = {}
    other_audio_files = {}
    lab_files = {}
    textgrid_files = {}
    for full_filename in files:
        filename, fext = os.path.splitext(full_filename)
        fext = fext.lower()
        if fext == '.wav':
            wav_files[filename] = full_filename
        elif fext == '.lab':
            lab_files[filename] = full_filename
        elif fext == '.txt' and filename not in lab_files:  # .lab files have higher priority than .txt files
            lab_files[filename] = full_filename
        elif fext == '.textgrid':
            textgrid_files[filename] = full_filename
        elif fext in supported_audio_extensions and shutil.which('sox') is not None:
            other_audio_files[filename] = full_filename
    return wav_files, lab_files, textgrid_files, other_audio_files


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
                 num_jobs=3, sample_rate=16000, debug=False, logger=None, use_mp=True,
                 punctuation=None, clitic_markers=None, audio_directory=None, skip_load=False):
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
            raise CorpusError('The directory \'{}\' does not exist.'.format(directory))
        if not os.path.isdir(directory):
            raise CorpusError('The specified path for the corpus ({}) is not a directory.'.format(directory))

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
        self.sample_rates = defaultdict(set)
        self.unsupported_sample_rate = []
        self.text_mapping = {}
        self.wav_files = []
        self.wav_info = {}
        self.sox_strings = {}
        self.unsupported_bit_depths = []
        self.wav_read_errors = []
        self.speak_utt_mapping = defaultdict(list)
        self.utt_speak_mapping = {}
        self.utt_wav_mapping = {}
        self.feat_mapping = {}
        self.cmvn_mapping = {}
        self.file_directory_mapping = {}
        self.file_name_mapping = {}
        self.textgrid_read_errors = {}
        self.speaker_ordering = {}
        self.groups = []
        self.speaker_groups = []
        self.segments = {}
        self.file_utt_mapping = {}
        self.utt_file_mapping = {}
        self.ignored_utterances = []
        self.utterance_lengths = {}
        self.sample_rate = sample_rate
        self.stopped = None
        self.skip_load = skip_load

    @property
    def file_speaker_mapping(self):
        return {file_name: [self.utt_speak_mapping[u] for u in utts] for file_name, utts in self.file_utt_mapping.items()}

    @property
    def speakers(self):
        return sorted(self.speak_utt_mapping.keys())

    def add_utterance(self, utterance, speaker, file, text, wav_file=None, seg=None):
        if seg is not None:
            self.segments[utterance] = seg
        self.utt_file_mapping[utterance] = file
        if file not in self.file_utt_mapping:
            self.file_utt_mapping[file] = []
            self.speaker_ordering[file] = []
        self.file_utt_mapping[file] = sorted(self.file_utt_mapping[file] + [utterance])

        self.utt_speak_mapping[utterance] = speaker
        self.speak_utt_mapping[speaker] = sorted(self.speak_utt_mapping[speaker] + [utterance])

        self.text_mapping[utterance] = text
        if wav_file is not None:
            self.utt_wav_mapping[utterance] = wav_file
        if speaker not in self.speaker_ordering[file]:
            self.speaker_ordering[file].append(speaker)

    def delete_utterance(self, utterance):
        if utterance in self.segments:
            del self.segments[utterance]
        file = self.utt_file_mapping[utterance]
        del self.utt_file_mapping[utterance]
        self.file_utt_mapping[file] = [x for x in self.file_utt_mapping[file] if x != utterance]
        if not self.file_utt_mapping[file]:
            del self.file_utt_mapping[file]
            del self.speaker_ordering[file]

        speaker = self.utt_speak_mapping[utterance]
        del self.utt_speak_mapping[utterance]
        self.speak_utt_mapping[speaker] = [x for x in self.speak_utt_mapping[speaker] if x != utterance]
        if not self.speak_utt_mapping[speaker]:
            del self.speak_utt_mapping[speaker]
        if utterance in self.feat_mapping:
            del self.feat_mapping[utterance]
        if utterance in self.utterance_lengths:
            del self.utterance_lengths[utterance]

        del self.text_mapping[utterance]
        if utterance in self.utt_wav_mapping:
            del self.utt_wav_mapping[utterance]

    def find_best_groupings(self):
        if len(self.speakers) < self.num_jobs:
            self.num_jobs = len(self.speakers)
        self.speaker_groups = [[] for _ in range(self.num_jobs)]
        job_ind = 0
        for s in self.speakers:
            self.speaker_groups[job_ind].append(s)
            job_ind += 1
            if job_ind == self.num_jobs:
                job_ind = 0
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
                        if u in self.sox_strings:
                            p = self.sox_strings[u]
                        else:
                            p = self.utt_wav_mapping[u]
                        output_g.append([u, p])
                    except KeyError:
                        pass
                else:
                    try:
                        r = self.segments[u]['file_name']
                    except KeyError:
                        continue
                    if r not in done:
                        if r in self.sox_strings:
                            p = self.sox_strings[r]
                        else:
                            p = self.utt_wav_mapping[r]
                        output_g.append([r, p])
                        done.add(r)
            output.append(output_g)
        return output

    def speaker_utterance_info(self):
        num_speakers = len(self.speak_utt_mapping.keys())
        if not num_speakers:
            raise CorpusError('There were no sound files found of the appropriate format. Please double check the corpus path '
                              'and/or run the validation utility (mfa validate).')
        average_utterances = sum(len(x) for x in self.speak_utt_mapping.values()) / num_speakers
        msg = 'Number of speakers in corpus: {}, average number of utterances per speaker: {}'.format(num_speakers,
                                                                                                      average_utterances)
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
            raise CorpusError(
                'Something went wrong while setting up the corpus. Please delete the {} folder and try again.'.format(
                    self.output_directory))
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
                    output_g.append([u, '{file_name} {begin} {end} {channel}'.format(**self.segments[u])])
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
        if utt in self.wav_info:
            return self.wav_info[utt][-1]
        if not self.segments:
            wav_path = self.utt_wav_mapping[utt]
        else:
            if utt in self.utt_wav_mapping:
                wav_path = self.utt_wav_mapping[utt]
            else:
                rec = self.segments[utt].split(' ')[0]
                if rec in self.wav_info:
                    return self.wav_info[rec][-1]
                wav_path = self.utt_wav_mapping[rec]
        return get_wav_info(wav_path)['duration']

    @property
    def file_durations(self):
        return {f: info[2] for f, info in self.wav_info.items()}

    def split_directory(self):
        directory = os.path.join(self.output_directory, 'split{}'.format(self.num_jobs))
        return directory

    def _write_utt_speak(self):
        utt2spk = os.path.join(self.output_directory, 'utt2spk')
        if os.path.exists(utt2spk):
            return
        output_mapping(self.utt_speak_mapping, utt2spk)

    def _write_speak_utt(self):
        spk2utt = os.path.join(self.output_directory, 'spk2utt')
        if os.path.exists(spk2utt):
            return
        output_mapping(self.speak_utt_mapping, spk2utt)

    def _write_utt_file(self):
        utt2file = os.path.join(self.output_directory, 'utt2file')
        if os.path.exists(utt2file):
            return
        output_mapping(self.utt_file_mapping, utt2file)

    def _write_file_utt(self):
        file2utt = os.path.join(self.output_directory, 'file2utt')
        if os.path.exists(file2utt):
            return
        output_mapping(self.file_utt_mapping, file2utt)

    def _write_wavscp(self):
        path = os.path.join(self.output_directory, 'wav.scp')
        if os.path.exists(path):
            return
        output_mapping(self.utt_wav_mapping, path)

    def _write_sox_strings(self):
        path = os.path.join(self.output_directory, 'sox_strings.scp')
        if os.path.exists(path):
            return
        output_mapping(self.sox_strings, path)

    def _write_wav_info(self):
        path = os.path.join(self.output_directory, 'wav_info.scp')
        if os.path.exists(path):
            return
        output_mapping(self.wav_info, path)

    def _write_file_directory(self):
        path = os.path.join(self.output_directory, 'file_directory.scp')
        if os.path.exists(path):
            return
        output_mapping(self.file_directory_mapping, path)

    def _write_file_name(self):
        path = os.path.join(self.output_directory, 'file_name.scp')
        if os.path.exists(path):
            return
        output_mapping(self.file_name_mapping, path)

    def _write_segments(self):
        if not self.segments:
            return
        path = os.path.join(self.output_directory, 'segments.scp')
        if os.path.exists(path):
            return
        segs = {}
        for k, v in self.segments.items():
            segs[k] = '{file_name} {begin} {end} {channel}'.format(**v)
        output_mapping(segs, path)

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

    def combine_feats(self):
        self.feat_mapping = {}
        split_directory = self.split_directory()
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        lengths_path = os.path.join(self.output_directory, 'utterance_lengths.scp')
        with open(feat_path, 'w') as outf, open(lengths_path, 'w') as lengths_out_f:
            for i in range(self.num_jobs):
                path = os.path.join(split_directory, 'feats.{}.scp'.format(i))
                run_filter = False
                lengths_path = os.path.join(split_directory, 'utterance_lengths.{}.scp'.format(i))
                if os.path.exists(lengths_path):
                    with open(lengths_path, 'r') as inf:
                        for line in inf:
                            line = line.strip()
                            utt, length = line.split()
                            length = int(length)
                            if length < 13:  # Minimum length to align one phone plus silence
                                self.ignored_utterances.append(utt)
                                run_filter = True
                            else:
                                self.utterance_lengths[utt] = length
                                lengths_out_f.write(line + '\n')
                    if run_filter:
                        filtered = filter_scp(self.ignored_utterances, path, exclude=True)
                        with open(path, 'w') as f:
                            for line in filtered:
                                f.write(line.strip() + '\n')
                with open(path, 'r') as inf:
                    for line in inf:
                        line = line.strip()
                        if line == '':
                            continue
                        f = line.split(maxsplit=1)
                        if f[0] in self.ignored_utterances:
                            continue
                        self.feat_mapping[f[0]] = f[1]
                        outf.write(line + '\n')
        for utt in self.utt_speak_mapping.keys():
            if utt not in self.feat_mapping and utt not in self.ignored_utterances:
                self.ignored_utterances.append(utt)
        if self.ignored_utterances:
            for k, v in self.speak_utt_mapping.items():
                self.speak_utt_mapping[k] = list(filter(lambda x: x in self.feat_mapping, v))
            self.logger.warning('There were some utterances ignored due to short duration, see the log file for full '
                                'details or run `mfa validate` on the corpus.')
            self.logger.debug('The following utterances were too short to run alignment: {}'.format(
                ' ,'.join(self.ignored_utterances)))

    def figure_utterance_lengths(self):
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        lengths_path = os.path.join(self.output_directory, 'utterance_lengths.scp')
        if os.path.exists(feat_path) and not self.utterance_lengths:
            if os.path.exists(lengths_path):
                self.utterance_lengths = load_scp(lengths_path, int)
            else:
                with open(os.devnull, 'w') as devnull:
                    dim_proc = subprocess.Popen([thirdparty_binary('feat-to-len'),
                                                 'scp:' + feat_path, 'ark,t:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=devnull)
                    stdout, stderr = dim_proc.communicate()
                    feats = stdout.decode('utf8').strip()
                    for line in feats.splitlines():
                        line = line.strip()
                        utt, length = line.split()
                        length = int(length)
                        if length < 13:  # Minimum length to align one phone plus silence
                            self.ignored_utterances.append(utt)
                        else:
                            self.utterance_lengths[line[0]] = int(line[1])
                output_mapping(self.utterance_lengths, lengths_path)

    def get_feat_dim(self, feature_config):

        feature_string = feature_config.construct_feature_proc_string(self.split_directory(), None, 0)
        with open(os.devnull, 'w') as devnull:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         feature_string, '-'],
                                        stdout=subprocess.PIPE,
                                        stderr=devnull
                                        )
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode('utf8').strip()
        return int(feats)

    def _write_speaker_ordering(self):
        path = os.path.join(self.output_directory, 'speaker_ordering.scp')
        output_mapping(self.speaker_ordering, path)

    def write(self):
        self._write_speak_utt()
        self._write_utt_speak()
        self._write_file_utt()
        self._write_utt_file()
        self._write_segments()
        self._write_wavscp()
        self._write_sox_strings()
        self._write_wav_info()
        self._write_file_directory()
        self._write_file_name()
        self._write_speaker_ordering()

    def split(self):
        split_dir = self.split_directory()
        os.makedirs(os.path.join(split_dir, 'log'), exist_ok=True)
        self.logger.info('Setting up training data...')
        self._split_wavs(split_dir)
        self._split_utt2spk(split_dir)
        self._split_spk2utt(split_dir)
        self._split_segments(split_dir)


    def spk2utt_by_dictionary(self, multispeaker_dictionary, dictionary_name):
        output = []
        for g in self.speaker_groups:
            output_g = []
            for s in sorted(g):
                if multispeaker_dictionary.get_dictionary_name(s) != dictionary_name:
                    continue
                try:
                    output_g.append([s, sorted(self.speak_utt_mapping[s])])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    def utt2spk_by_dictionary(self, multispeaker_dictionary, dictionary_name):
        output = []
        for g in self.groups:
            output_g = []
            for u in sorted(g):
                if u in self.ignored_utterances:
                    continue
                s = self.utt_speak_mapping[u]
                if multispeaker_dictionary.get_dictionary_name(s) != dictionary_name:
                    continue
                try:
                    output_g.append([u, s])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    def feats_by_dictionary(self, multispeaker_dictionary, dictionary_name):
        output = []
        for g in self.groups:
            output_g = []
            for u in sorted(g):
                if u in self.ignored_utterances:
                    continue
                s = self.utt_speak_mapping[u]
                if multispeaker_dictionary.get_dictionary_name(s) != dictionary_name:
                    continue
                try:
                    output_g.append([u, self.feat_mapping[u]])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    def cmvn_by_dictionary(self, multispeaker_dictionary, dictionary_name):
        output = []
        for g in self.speaker_groups:
            output_g = []
            for s in sorted(g):
                if multispeaker_dictionary.get_dictionary_name(s) != dictionary_name:
                    continue
                try:
                    output_g.append([s, self.cmvn_mapping[s]])
                except KeyError:
                    pass
            output.append(output_g)
        return output

    def split_by_dictionary(self, multispeaker_dictionary):
        split_dir = self.split_directory()
        for name, d in multispeaker_dictionary.dictionary_mapping.items():
            spk2utt = self.spk2utt_by_dictionary(multispeaker_dictionary, name)
            pattern = 'spk2utt.{}.'+ name
            save_groups(spk2utt, split_dir, pattern)

            utt2spk = self.utt2spk_by_dictionary(multispeaker_dictionary, name)
            pattern = 'utt2spk.{}.'+ name
            save_groups(utt2spk, split_dir, pattern)

            feats = self.feats_by_dictionary(multispeaker_dictionary, name)
            pattern = 'feats.{}.'+ name + '.scp'
            save_groups(feats, split_dir, pattern)

            cmvn = self.cmvn_by_dictionary(multispeaker_dictionary, name)
            pattern = 'cmvn.{}.'+ name + '.scp'
            save_groups(cmvn, split_dir, pattern)
