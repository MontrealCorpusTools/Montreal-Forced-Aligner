
import os
import subprocess
import shutil
from collections import defaultdict

from .helper import thirdparty_binary, load_text, make_safe
from .multiprocessing import mfcc

def output_mapping(mapping, path):
    with open(path, 'w', encoding = 'utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            f.write('{} {}\n'.format(k, v))

def save_scp(scp, path):
    with open(path, 'w', encoding = 'utf8') as f:
        for line in scp:
            f.write('{}\n'.format(' '.join(map(make_safe,line))))

def save_groups(groups, seg_dir, pattern):
    for i, g in enumerate(groups):
        path = os.path.join(seg_dir, pattern.format(i))
        save_scp(g, path)

def load_scp(path):
    scp = {}
    with open(path, 'r', encoding = 'utf8') as f:
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
        if len(self.speak_utt_mapping) < self.num_jobs:
            self.num_jobs = len(self.utt_wav_mapping)
        self.groups = self.find_best_groupings()
        self.speaker_groups = []
        for g in self.groups:
            speaker_mapping = {}
            for k in sorted(g):
                v = self.utt_speak_mapping[k]
                if v not in speaker_mapping:
                    speaker_mapping[v] = []
                speaker_mapping[v].append(k)
            self.speaker_groups.append(speaker_mapping)

    @property
    def mfcc_directory(self):
        return os.path.join(self.output_directory, 'mfcc')

    @property
    def mfcc_log_directory(self):
        return os.path.join(self.mfcc_directory, 'log')

    @property
    def wav_scp(self):
        for k in sorted(self.utt_wav_mapping.keys()):
            v = self.utt_wav_mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            yield [k, v]

    @property
    def utt_scp(self):
        output = []
        for k in sorted(self.utt_speak_mapping.keys()):
            v = self.utt_speak_mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            yield [k, v]

    @property
    def grouped_wav(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                output_g.append([u, self.utt_wav_mapping[u]])
            output.append(output_g)
        return output

    @property
    def grouped_feat(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                output_g.append([u, self.feat_mapping[u]])
            output.append(output_g)
        return output

    @property
    def grouped_text(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                output_g.append([u, self.text_mapping[u]])
            output.append(output_g)
        return output

    def grouped_text_int(self, dictionary):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                text = self.text_mapping[u].split()
                for i in range(len(text)):
                    text[i] = str(dictionary.to_int(text[i]))
                output_g.append([u, ' '.join(text)])
            output.append(output_g)
        return output

    @property
    def grouped_cmvn(self):
        output = []
        for g in self.speaker_groups:
            output_g = []
            for s in sorted(g.keys()):
                output_g.append([s, self.cmvn_mapping[s]])
            output.append(output_g)
        return output

    @property
    def grouped_utt2spk(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                output_g.append([u, self.utt_speak_mapping[u]])
            output.append(output_g)
        return output

    @property
    def grouped_spk2utt(self):
        output = []
        for g in self.speaker_groups:
            output_g = []
            for s in sorted(g.keys()):
                output_g.append([s, self.speak_utt_mapping[s]])
            output.append(output_g)
        return output

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

    def _split_utt2spk(self, directory):
        pattern = 'utt2spk.{}'
        save_groups(self.grouped_utt2spk, directory, pattern)

    def _split_spk2utt(self, directory):
        pattern = 'spk2utt.{}'
        save_groups(self.grouped_spk2utt, directory, pattern)

    def _split_wavs(self, directory):
        pattern = 'wav.{}.scp'
        save_groups(self.grouped_wav, directory, pattern)

    def _split_feats(self, directory):
        pattern = 'feats.{}.scp'
        save_groups(self.grouped_feat, directory, pattern)

    def _split_texts(self, directory, dictionary = None):
        pattern = 'text.{}'
        save_groups(self.grouped_text, directory, pattern)
        if dictionary is not None:
            pattern = 'text.{}.int'
            save_groups(self.grouped_text_int(dictionary), directory, pattern)

    def _split_cmvns(self, directory):
        pattern = 'cmvn.{}.scp'
        save_groups(self.grouped_cmvn, directory, pattern)

    def find_best_groupings(self):
        scp = list(self.utt_scp)
        num_utt = len(scp)
        interval = int(num_utt / self.num_jobs)
        groups = []
        current_ind = 0
        for i in range(self.num_jobs):
            if i == self.num_jobs - 1:
                end_ind = num_utt
            else:
                end_ind = current_ind + interval
                utt = scp[end_ind][0]
                spk = scp[end_ind][1]
                for j in range(end_ind, num_utt):
                    if scp[j][1] != spk:
                        j -= 1
                        break
                else:
                    j = num_utt - 1
                if j - end_ind < i - end_ind:
                    end_ind = j
                else:
                    k = end_ind
                    for k in range(end_ind, 0, -1):
                        if scp[k][1] != spk:
                            k += 1
                            break
                    end_ind = k
            groups.append([x[0] for x in scp[current_ind:end_ind]])
            current_ind = end_ind
        return groups

    def create_mfccs(self):
        log_directory = self.mfcc_log_directory
        os.makedirs(log_directory, exist_ok = True)
        self._split_wavs(self.mfcc_log_directory)
        mfcc(self.mfcc_directory, log_directory, self.num_jobs, self.mfcc_config)
        self._combine_feats()
        self._calc_cmvn()

    def _combine_feats(self):
        self.feat_mapping = {}
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        with open(feat_path, 'w') as outf:
            for i in range(self.num_jobs):
                path = os.path.join(self.mfcc_directory, 'raw_mfcc.{}.scp'.format(i))
                with open(path,'r') as inf:
                    for line in inf:
                        line = line.strip()
                        if line == '':
                            continue
                        f = line.split(maxsplit=1)
                        self.feat_mapping[f[0]] = f[1]
                        outf.write(line + '\n')
                os.remove(path)

    def _calc_cmvn(self):
        spk2utt = os.path.join(self.output_directory, 'spk2utt')
        feats = os.path.join(self.output_directory, 'feats.scp')
        cmvn_directory = os.path.join(self.mfcc_directory, 'cmvn')
        os.makedirs(cmvn_directory, exist_ok = True)
        cmvn_ark = os.path.join(cmvn_directory, 'cmvn.ark')
        cmvn_scp = os.path.join(cmvn_directory, 'cmvn.scp')
        log_path = os.path.join(cmvn_directory, 'cmvn.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compute-cmvn-stats'),
                        '--spk2utt=ark:'+spk2utt,
                        'scp:'+feats, 'ark,scp:{},{}'.format(cmvn_ark, cmvn_scp)],
                                            stderr = logf)
        shutil.copy(cmvn_scp, os.path.join(self.output_directory, 'cmvn.scp'))
        self.cmvn_mapping = load_scp(cmvn_scp)

    def setup_splits(self, dictionary):
        split_dir = self.split_directory
        if os.path.exists(split_dir):
            return
        os.makedirs(split_dir)
        self._split_wavs(split_dir)
        self._split_utt2spk(split_dir)
        self._split_spk2utt(split_dir)
        self._split_feats(split_dir)
        self._split_texts(split_dir, dictionary)
        self._split_cmvns(split_dir)
        self._split_and_norm_feats()

    def _split_and_norm_feats(self):
        split_dir = self.split_directory
        log_dir = os.path.join(split_dir, 'log')
        os.makedirs(log_dir, exist_ok = True)
        with open(os.path.join(log_dir, 'norm.log'), 'w') as logf:
            for i in range(self.num_jobs):
                path = os.path.join(split_dir, 'cmvndeltafeats.{}'.format(i))
                utt2spkpath = os.path.join(split_dir, 'utt2spk.{}'.format(i))
                cmvnpath = os.path.join(split_dir, 'cmvn.{}.scp'.format(i))
                featspath = os.path.join(split_dir, 'feats.{}.scp'.format(i))
                if not os.path.exists(path):
                    with open(path, 'wb') as outf:
                        cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                    '--utt2spk=ark:'+utt2spkpath,
                                    'scp:'+cmvnpath,
                                    'scp:'+featspath,
                                    'ark:-'], stdout = subprocess.PIPE,
                                    stderr = logf
                                    )
                        deltas_proc = subprocess.Popen([thirdparty_binary('add-deltas'),
                                                'ark:-', 'ark:-'],
                                                stdin = cmvn_proc.stdout,
                                                stdout = outf,
                                                stderr = logf
                                                )
                        deltas_proc.communicate()
                    with open(path, 'rb') as inf, open(path+'_sub','wb') as outf:
                        subprocess.call([thirdparty_binary("subset-feats"),
                                    "--n=10", "ark:-", "ark:-"],
                                    stdin = inf, stdout = outf)

    def get_feat_dim(self):
        directory = self.split_directory

        path = os.path.join(self.split_directory, 'cmvndeltafeats.0')
        with open(path, 'rb') as f, open(os.devnull, 'w') as devnull:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                        'ark,s,cs:-', '-'],
                                        stdin = f,
                                        stdout = subprocess.PIPE,
                                        stderr = devnull)
            stdout, stderr = dim_proc.communicate()
            feats = stdout.decode('utf8').strip()
        return feats
