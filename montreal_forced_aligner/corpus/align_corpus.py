import os
import sys
import traceback
import random
import time
from collections import Counter
from textgrid import TextGrid, IntervalTier

from ..dictionary import sanitize
from ..helper import load_text, output_mapping, save_groups, filter_scp

from ..exceptions import SampleRateError, CorpusError

from .base import BaseCorpus, get_sample_rate, get_n_channels, get_wav_duration, extract_temp_channels, get_bit_depth, find_ext


def parse_transcription(text):
    words = [sanitize(x) for x in text.split()]
    words = [x for x in words if x not in ['', '-', "'"]]
    return words


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
                 num_jobs=3, debug=False, logger=None):
        super(AlignableCorpus, self).__init__(directory, output_directory,
                                              speaker_characters,
                                              num_jobs, debug, logger)
        # Set up mapping dictionaries
        self.utt_text_file_mapping = {}
        self.word_counts = Counter()
        self.utterance_oovs = {}
        self.no_transcription_files = []
        self.decode_error_files = []
        self.transcriptions_without_wavs = []
        self.tg_count = 0
        self.lab_count = 0
        begin_time = time.time()
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            wav_files = find_ext(files, '.wav')
            lab_files = find_ext(files, '.lab')
            txt_files = find_ext(files, '.txt')
            textgrid_files = find_ext(files, '.textgrid')
            for file_name, f in wav_files.items():
                wav_path = os.path.join(root, f)
                try:
                    sr = get_sample_rate(wav_path)
                except Exception:
                    self.wav_read_errors.append(wav_path)
                    continue
                if sr < 16000:
                    self.unsupported_sample_rate.append(wav_path)
                bit_depth = get_bit_depth(wav_path)
                if bit_depth != 16:
                    self.unsupported_bit_depths.append(wav_path)
                    continue
                # .lab files have higher priority than .txt files
                lab_name = lab_files[file_name] if file_name in lab_files else txt_files[file_name]
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
                        else:
                            speaker_name = f
                    speaker_name = speaker_name.strip().replace(' ', '_')
                    utt_name = utt_name.strip().replace(' ', '_')
                    self.utt_text_file_mapping[utt_name] = lab_path
                    self.speak_utt_mapping[speaker_name].append(utt_name)
                    self.utt_wav_mapping[utt_name] = wav_path
                    self.sample_rates[get_sample_rate(wav_path)].add(speaker_name)
                    self.utt_speak_mapping[utt_name] = speaker_name
                    self.file_directory_mapping[utt_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')

                    self.lab_count += 1
                else:
                    tg_name = textgrid_files[file_name]
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
                        a_name = file_name + "_A"
                        b_name = file_name + "_B"

                        a_path, b_path = extract_temp_channels(wav_path, self.temp_directory)
                    elif n_channels > 2:
                        raise (Exception('More than two channels'))
                    self.speaker_ordering[file_name] = []
                    if not self.speaker_directories:
                        if isinstance(speaker_characters, int):
                            speaker_name = f[:speaker_characters]
                        elif speaker_characters == 'prosodylab':
                            speaker_name = f.split('_')[1]
                        else:
                            speaker_name = f
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
                                    self.segments[utt_name] = '{} {} {}'.format(a_name, begin, end)
                                    self.utt_wav_mapping[a_name] = a_path
                                else:
                                    utt_name += '_B'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                    self.segments[utt_name] = '{} {} {}'.format(b_name, begin, end)
                                    self.utt_wav_mapping[b_name] = b_path
                            self.text_mapping[utt_name] = ' '.join(words)
                            self.utt_text_file_mapping[utt_name] = tg_path
                            self.word_counts.update(words)
                            self.utt_speak_mapping[utt_name] = speaker_name
                            self.speak_utt_mapping[speaker_name].append(utt_name)
                    if n_channels == 2:
                        self.file_directory_mapping[a_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')
                        self.file_directory_mapping[b_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')
                    self.file_directory_mapping[file_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')
                    self.tg_count += 1
        self.logger.debug('Parsed corpus directory in {} seconds'.format(time.time()-begin_time))
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
            self.logger.error(msg)
            raise (SampleRateError(msg))

        if len(self.speak_utt_mapping) < self.num_jobs:
            self.num_jobs = len(self.speak_utt_mapping)
        if self.num_jobs < len(self.sample_rates.keys()):
            self.num_jobs = len(self.sample_rates.keys())
            msg = 'The number of jobs was set to {}, due to the different sample rates in the dataset. ' \
                  'If you would like to use fewer parallel jobs, ' \
                  'please resample all wav files to the same sample rate.'.format(self.num_jobs)
            self.logger.warning(msg)
        self.find_best_groupings()

    def update_utterance_text(self, utterance, new_text):
        new_text = new_text.lower().strip()
        self.text_mapping[utterance] = new_text
        text_file_path = self.utt_text_file_mapping[utterance]
        if text_file_path.lower().endswith('.textgrid'):
            tg = TextGrid()
            found = False
            tg.read(text_file_path)

            speaker_name = utterance.split('_', maxsplit=1)
            wave_name, begin, end = self.segments[utterance].split(' ')
            begin = float(begin)
            end = float(end)
            if len(tg.tiers) == 1:
                for interval in tg.tiers[0]:
                    if abs(interval.minTime - begin) < 0.01 and abs(interval.maxTime - end) < 0.01:
                        interval.mark = new_text
                        found = True
                        break
            else:
                for tier in tg.tiers:
                    if tier.name == speaker_name:
                        for interval in tg.tiers[0]:
                            if abs(interval.minTime - begin) < 0.01 and abs(interval.maxTime - end) < 0.01:
                                interval.mark = new_text
                                found = True
                                break
            if found:
                tg.write(text_file_path)
            else:
                self.logger.warning('Unable to find utterance {} match in {}'.format(utterance, text_file_path))
        else:
            with open(text_file_path, 'w', encoding='utf8') as f:
                f.write(new_text)

    @property
    def word_set(self):
        return set(self.word_counts)

    def normalized_text_iter(self, dictionary=None, min_count=1):
        unk_words = set(k for k, v in self.word_counts.items() if v <= min_count)
        for u, text in self.text_mapping.items():
            text = text.split()
            new_text = []
            for t in text:
                if dictionary is not None:
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
                    try:
                        text = self.text_mapping[u].split()
                    except KeyError:
                        continue
                    new_text = []
                    for t in text:
                        lookup = dictionary.split_clitics(t)
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

    def get_word_frequency(self, dictionary):
        word_counts = Counter()
        for u, text in self.text_mapping.items():
            new_text = []
            text = text.split()
            for t in text:
                lookup = dictionary.split_clitics(t)
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
                for t in text:
                    lookup = dictionary.split_clitics(t)
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

    def subset_directory(self, subset, feature_config):
        if subset is None or subset > self.num_utterances or subset <= 0:
            return self.split_directory()
        directory = os.path.join(self.output_directory, 'subset_{}'.format(subset))
        if not os.path.exists(directory):
            self.create_subset(subset, feature_config)
        return directory

    def write(self):
        super(AlignableCorpus, self).write()
        self._write_text()

    def _write_text(self):
        text = os.path.join(self.output_directory, 'text')
        output_mapping(self.text_mapping, text)

    def _split_utt2fst(self, directory, dictionary):
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

    def initialize_corpus(self, dictionary):
        if not self.utt_wav_mapping:
            raise CorpusError('There were no wav files found for transcribing this corpus. Please validate the corpus.')
        split_dir = self.split_directory()
        self.write()
        if not os.path.exists(split_dir):
            self.split(dictionary)
        self.figure_utterance_lengths()

    def create_subset(self, subset, feature_config):
        larger_subset_num = subset * 10
        if larger_subset_num < self.num_utterances:
            # Get all shorter utterances that are not one word long
            utts = sorted((x for x in self.utterance_lengths.keys() if ' ' in self.text_mapping[x]),
                          key=lambda x: self.utterance_lengths[x])
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
                    filtered_utts = [x for x in utts if x in subset_utts]
                    outf.write('{} {}\n'.format(speaker, ' '.join(filtered_utts)))
            if feature_config is not None:
                base_path = os.path.join(split_directory, feature_config.feature_id + '.{}.scp'.format(j))
                subset_scp = os.path.join(subset_directory, feature_config.feature_id + '.{}.scp'.format(j))
                filtered = filter_scp(subset_utts, base_path)
                with open(subset_scp, 'w') as f:
                    for line in filtered:
                        f.write(line.strip() + '\n')
