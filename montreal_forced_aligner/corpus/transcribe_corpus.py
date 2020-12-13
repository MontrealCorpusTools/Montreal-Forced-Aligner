import os
import wave

from .base import BaseCorpus, get_sample_rate

from ..exceptions import SampleRateError, CorpusError


class TranscribeCorpus(BaseCorpus):
    def __init__(self, directory, output_directory,
                 speaker_characters=0,
                 num_jobs=3, debug=False):
        super(TranscribeCorpus, self).__init__(directory, output_directory,
                                               speaker_characters,
                                               num_jobs, debug)
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            for f in sorted(files):
                file_name, ext = os.path.splitext(f)
                if ext.lower() != '.wav':
                    continue

                wav_path = os.path.join(root, f)
                try:
                    sr = get_sample_rate(wav_path)
                except wave.Error:
                    self.wav_read_errors.append(wav_path)
                    continue
                if sr < 16000:
                    self.unsupported_sample_rate.append(wav_path)
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
                utt_name = file_name
                if utt_name in self.utt_wav_mapping:
                    ind = 0
                    fixed_utt_name = utt_name
                    while fixed_utt_name not in self.utt_wav_mapping:
                        ind += 1
                        fixed_utt_name = utt_name + '_{}'.format(ind)
                    utt_name = fixed_utt_name
                utt_name = utt_name.strip().replace(' ', '_')
                self.utt_wav_mapping[utt_name] = wav_path
                self.speak_utt_mapping[speaker_name].append(utt_name)
                self.sample_rates[get_sample_rate(wav_path)].add(speaker_name)
                self.utt_speak_mapping[utt_name] = speaker_name
                self.file_directory_mapping[utt_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')

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
            print('WARNING: ' + msg)
            self.logger.warning(msg)
        self.find_best_groupings()

    def initialize_corpus(self):
        split_dir = self.split_directory()
        self.write()
        if not os.path.exists(split_dir):
            self.split()
        self.figure_utterance_lengths()
