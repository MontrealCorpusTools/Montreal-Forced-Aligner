import os
import subprocess
import re

from .helper import load_phone_to_int, load_word_to_int

def prepare_lang(data_directory, lm_path, oov_code = "<unk>",
                position_dependent_phones = True,
                num_sil_states = 5,
                num_nonsil_states = 3,
                share_silence_phones = False,
                sil_prob = 0.5,
                reverse = False):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_dir = os.path.join(lang_directory, 'phones')

    format_lm(data_directory, lm_path)

class LM(object):
    ngram_pattern = re.compile(r'^\\(\d)-grams:$')
    def __init__(self, path):
        self.path = path

    def unigram_words(self):
        current_ngram = None
        with open(self.path, 'r', encoding = 'utf8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue

                ngram_match = self.ngram_pattern.match(line)
                if ngram_match is not None:
                    current_ngram = ngram_match.groups()[0]
                    if current_ngram != '1':
                        break
                    continue
                if current_ngram is None:
                    continue
                line = line.split('\t')
                yield line[1]


def find_oovs(words, lm):
    oovs = []
    for word in lm.unigram_words():
        if word not in words:
            oovs.append(word)
    return oovs

def oov_path(data_directory):
    lang_directory = os.path.join(data_directory, 'lang')
    return os.path.join(lang_directory, 'oov.txt')

def save_oovs(oovs, data_directory):
    with open(oov_path(data_directory), 'w', encoding = 'utf8') as f:
        for line in oovs:
            f.write(line + '\n')

def format_fst(fst_path, formatted_fst_path, oovs):
    ss = set(["<s>", "</s>"])
    oovs = set(oovs)
    with open(fst_path, 'r', encoding = 'utf8') as inf, \
        open(formatted_fst_path, 'w', encoding = 'utf8') as outf:
        for line in inf:
            line = line.strip()
            if line == '':
                outf.write(line)
                continue
            line = line.split()
            if len(line) >= 4:
                if line[2] in oovs or line[3] in oovs:
                    continue
                if line[2] == '#0' or line[3] == '#0':
                    raise(Exception('#0 a reserved symbol but found in the lm.'))

                if line[2] == '<eps>':
                    line[2] = '#0'

                if line[2] in ss:
                    line[2] = '<eps>'
                if line[3] in ss:
                    line[3] = '<eps>'

            line = '\t'.join(line)

            outf.write(line + '\n')

def format_lm(data_directory, lm_path):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')
    tmp_fst = os.path.join(lang_directory, 'temp.fst')
    formatted_fst = os.path.join(lang_directory, 'formatted.fst')
    words = load_word_to_int(lang_directory)
    word_path = os.path.join(lang_directory, 'words.txt')
    lm = LM(lm_path)
    oovs = find_oovs(words, lm)

    proc = subprocess.Popen(['arpa2fst', lm_path], stdout = subprocess.PIPE)
    with open(tmp_fst, 'w', encoding = 'utf8') as f:
        proc2 = subprocess.Popen(['fstprint'], stdin = proc.stdout, stdout = f)
        proc2.wait()

    format_fst(tmp_fst, formatted_fst, oovs)

    comp_proc = subprocess.Popen(['fstcompile', '--isymbols='+word_path,
        '--osymbols='+ word_path,
        '--keep_isymbols=false', '--keep_osymbols=false', formatted_fst], stdout = subprocess.PIPE)
    rmeps_proc = subprocess.Popen(['fstrmepsilon'], stdin = comp_proc.stdout, stdout = subprocess.PIPE)
    g_fst_path = os.path.join(lang_directory, 'G.fst')
    with open(g_fst_path, 'wb') as f:
        final_proc = subprocess.Popen(['fstarcsort',
                '--sort_type=ilabel'], stdin = rmeps_proc.stdout, stdout = f)
        final_proc.wait()
    subprocess.call(['fstisstochastic', g_fst_path])
