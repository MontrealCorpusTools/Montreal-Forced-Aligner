import subprocess
import os
import shutil
import re
import sys
import traceback
import time
from decimal import Decimal
import statistics

from .helper import make_path_safe, run_mp, run_non_mp, thirdparty_binary, parse_logs



def generate_pronunciations_func(model_directory, dictionary, corpus, job_name):
    text_int_path = os.path.join(corpus.split_directory(), 'text.{}.int'.format(job_name))
    log_path = os.path.join(model_directory, 'log', 'pronunciation.{}.log'.format(job_name))
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, 'final.mdl')
    aligned_path = os.path.join(model_directory, 'aligned.{}'.format(job_name))
    nbest_path = os.path.join(model_directory, 'nbest.{}'.format(job_name))
    pron_path = os.path.join(model_directory, 'prons.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark,t:' + nbest_path],
                                    stdout=subprocess.PIPE, stderr=log_file)

        lin_proc.communicate()
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark:-'],
                                    stdout=subprocess.PIPE, stderr=log_file)
        align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                       os.path.join(dictionary.phones_dir, 'word_boundary.int'), model_path,
                                       'ark:-', 'ark,t:' + aligned_path],
                                      stdin=lin_proc.stdout, stderr=log_file)
        align_proc.communicate()

        subprocess.call([thirdparty_binary('nbest-to-prons'),
                         model_path,
                         'ark:' + aligned_path,
                         pron_path],
                        stderr=log_file)


def generate_pronunciations(align_config, model_directory, dictionary, corpus, num_jobs):
    from collections import Counter, defaultdict
    log_directory = os.path.join(model_directory, 'log')
    os.makedirs(log_directory, exist_ok=True)
    jobs = [(model_directory, dictionary, corpus, x)
            for x in range(num_jobs)]
    if align_config.use_mp:
        run_mp(generate_pronunciations_func, jobs, log_directory)
    else:
        run_non_mp(generate_pronunciations_func, jobs, log_directory)

    word_lookup = dictionary.reversed_word_mapping
    phone_lookup = dictionary.reversed_phone_mapping
    pron_counts = defaultdict(Counter)
    for j in range(num_jobs):
        pron_path = os.path.join(model_directory, 'prons.{}'.format(j))
        with open(pron_path, 'r', encoding='utf8') as f:
            utt_mapping = {}
            last_utt = None
            for line in f:
                line = line.split()
                utt = line[0]
                if utt not in utt_mapping:
                    if last_utt is not None:
                        utt_mapping[last_utt].append('</s>')
                    utt_mapping[utt] = ['<s>']
                    last_utt = utt

                begin = line[1]
                end = line[2]
                word = word_lookup[int(line[3])]
                if word == '<eps>':
                    utt_mapping[utt].append(word)
                else:
                    pron = tuple(phone_lookup[int(x)].split('_')[0] for x in line[4:])
                    pron_string = ' '.join(pron)
                    utt_mapping[utt].append(word + ' ' + pron_string)
                    pron_counts[word][pron] += 1
                    print(word, pron)
    return pron_counts, utt_mapping

