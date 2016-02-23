
import os
import subprocess
import shutil
from collections import defaultdict

from .helper import split_scp, thirdparty_binary, load_text
from .multiprocessing import mfcc

def output_mapping(mapping, path):
    with open(path, 'w', encoding = 'utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            f.write('{} {}\n'.format(k, v))

class Corpus(object):
    def __init__(self, directory, output_directory, mfcc_config,
                speaker_directories = True,
                num_jobs = 3):
        self.directory = directory
        self.output_directory = os.path.join(output_directory, 'train')
        os.makedirs(self.output_directory, exist_ok = True)
        self.speaker_directories = speaker_directories
        self.num_jobs = num_jobs
        self.mfcc_config = mfcc_config

        self.speak_utt_mapping = defaultdict(list)
        self.utt_speak_mapping = {}
        self.utt_wav_mapping = {}
        self.text_mapping = {}

        if self.speaker_directories:
            speaker_dirs = os.listdir(self.directory)
            for speaker_id in speaker_dirs:
                speaker_dir = os.path.join(self.directory, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue

                for f in os.listdir(speaker_dir):
                    if not f.endswith('.lab'):
                        continue
                    utt_name = os.path.splitext(f)[0]
                    path = os.path.join(speaker_dir, f)
                    wav_path = path.replace('.lab', '.wav')
                    self.text_mapping[utt_name] = load_text(path)
                    self.speak_utt_mapping[speaker_id].append(utt_name)
                    self.utt_wav_mapping[utt_name] = wav_path
                    self.utt_speak_mapping[utt_name] = speaker_id
        if len(self.utt_wav_mapping) < self.num_jobs:
            self.num_jobs = len(self.utt_wav_mapping)
    @property
    def mfcc_directory(self):
        return os.path.join(self.output_directory, 'mfcc')

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

    def create_mfccs(self):

        log_directory = os.path.join(self.mfcc_directory, 'log')
        os.makedirs(log_directory, exist_ok = True)
        split_scp(os.path.join(self.output_directory, 'wav.scp'),
                    log_directory, self.num_jobs)
        mfcc(self.mfcc_directory, log_directory, self.num_jobs, self.mfcc_config)
        self._combine_feats()
        self._calc_cmvn()

    def _combine_feats(self):
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        with open(feat_path, 'w') as outf:
            for i in range(self.num_jobs):
                path = os.path.join(self.mfcc_directory, 'raw_mfcc.{}.scp'.format(i+1))
                with open(path,'r') as inf:
                    for line in inf:
                        outf.write(line)
                os.remove(path)

    def _calc_cmvn(self):
        spk2utt = os.path.join(self.output_directory, 'spk2utt')
        feats = os.path.join(self.output_directory, 'feats.scp')
        cmvn_directory = os.path.join(self.mfcc_directory, 'cmvn')
        os.makedirs(cmvn_directory, exist_ok = True)
        cmvn_ark = os.path.join(cmvn_directory, 'cmvn.ark')
        cmvn_scp = os.path.join(cmvn_directory, 'cmvn.scp')
        subprocess.call([thirdparty_binary('compute-cmvn-stats'),
                        '--spk2utt=ark:'+spk2utt,
                        'scp:'+feats, 'ark,scp:{},{}'.format(cmvn_ark, cmvn_scp)])
        shutil.copy(cmvn_scp, os.path.join(self.output_directory, 'cmvn.scp'))
