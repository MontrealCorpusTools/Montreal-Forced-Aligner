import subprocess
import os
import re
import sys
import time
import traceback
from decimal import Decimal
import statistics
from collections import defaultdict
from multiprocessing import Lock

from .helper import make_path_safe, run_mp, run_non_mp, thirdparty_binary

from ..textgrid import parse_from_word, parse_from_word_no_cleanup, parse_from_phone, \
    ctms_to_textgrids_non_mp, output_textgrid_writing_errors, generate_tiers, export_textgrid, construct_output_path

from ..exceptions import AlignmentError
import multiprocessing as mp
from ..multiprocessing.helper import Stopped
from queue import Empty

queue_polling_timeout = 1


def acc_stats_func(directory, iteration, job_name, feature_string):
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, job_name))
    model_path = os.path.join(directory, '{}.mdl'.format(iteration))
    acc_path = os.path.join(directory, '{}.{}.acc'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), model_path,
                                     '{}'.format(feature_string), "ark:" + ali_path, acc_path],
                                    stderr=log_file)
        acc_proc.communicate()


def acc_stats(iteration, directory, split_directory, num_jobs, config):
    """
    Multiprocessing function that computes stats for GMM training

    See http://kaldi-asr.org/doc/gmm-acc-stats-ali_8cc.html for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh
    for the bash script this function was extracted from

    Parameters
    ----------
    iteration : int
        Iteration to calculate stats for
    directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    split_directory : str
        Directory of training data split into the number of jobs
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(directory, iteration, x,
             config.feature_config.construct_feature_proc_string(split_directory, directory, x)
             ) for x in range(num_jobs)]

    if config.use_mp:
        run_mp(acc_stats_func, jobs, config.log_directory)
    else:
        run_non_mp(acc_stats_func, jobs, config.log_directory)


def compile_train_graphs_func(directory, lang_directory, split_directory, job_name, dictionary_names=None, debug=True):
    tree_path = os.path.join(directory, 'tree')
    mdl_path = os.path.join(directory, '0.mdl')
    if not os.path.exists(mdl_path):
        mdl_path = os.path.join(directory, 'final.mdl')

    if dictionary_names is None:
        log_path = os.path.join(directory, 'log', 'compile-graphs.{}.log'.format(job_name))

        fst_scp_path = os.path.join(directory, 'fsts.{}.scp'.format(job_name))
        fst_ark_path = os.path.join(directory, 'fsts.{}.ark'.format(job_name))
        text_path = os.path.join(split_directory, 'text.{}.int'.format(job_name))

        with open(log_path, 'w', encoding='utf8') as log_file:
            proc = subprocess.Popen([thirdparty_binary('compile-train-graphs'),
                                     '--read-disambig-syms={}'.format(
                                         os.path.join(lang_directory, 'phones', 'disambig.int')),
                                     tree_path, mdl_path,
                                     os.path.join(lang_directory, 'L.fst'),
                                     "ark:" + text_path, "ark,scp:{},{}".format(fst_ark_path, fst_scp_path)],
                                    stderr=log_file)
            proc.communicate()
    else:
        for name in dictionary_names:
            log_path = os.path.join(directory, 'log', 'compile-graphs.{}.{}.log'.format(job_name, name))

            fst_scp_path = os.path.join(directory, 'fsts.{}.{}.scp'.format(job_name, name))
            fst_ark_path = os.path.join(directory, 'fsts.{}.{}.ark'.format(job_name, name))
            text_path = os.path.join(split_directory, 'text.{}.{}.int'.format(job_name, name))
            with open(log_path, 'w', encoding='utf8') as log_file:
                proc = subprocess.Popen([thirdparty_binary('compile-train-graphs'),
                                         '--read-disambig-syms={}'.format(
                                             os.path.join(lang_directory, 'phones', 'disambig.int')),
                                         tree_path, mdl_path,
                                         os.path.join(lang_directory, name, 'dictionary', 'L.fst'),
                                         "ark:" + text_path, "ark,scp:{},{}".format(fst_ark_path, fst_scp_path)],
                                        stderr=log_file)
                proc.communicate()

        fst_scp_path = os.path.join(directory, 'fsts.{}.scp'.format(job_name))
        lines = []
        for name in dictionary_names:
            with open(os.path.join(directory, 'fsts.{}.{}.scp'.format(job_name, name)), 'r', encoding='utf8') as inf:
                for line in inf:
                    lines.append(line)
        with open(fst_scp_path, 'w', encoding='utf8') as outf:
            for line in sorted(lines):
                outf.write(line)


def compile_train_graphs(directory, lang_directory, split_directory, num_jobs, aligner, debug=False):
    """
    Multiprocessing function that compiles training graphs for utterances

    See http://kaldi-asr.org/doc/compile-train-graphs_8cc.html for more details
    on the Kaldi binary this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh
    for the bash script that this function was extracted from.

    Parameters
    ----------
    directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    lang_directory : str
        Directory of the language model used
    split_directory : str
        Directory of training data split into the number of jobs
    num_jobs : int
        The number of processes to use
    """
    aligner.logger.info('Compiling training graphs...')
    begin = time.time()
    log_directory = os.path.join(directory, 'log')
    os.makedirs(log_directory, exist_ok=True)
    jobs = [(directory, lang_directory, split_directory, x, aligner.dictionaries_for_job(x), debug)
            for x in range(num_jobs)]
    if aligner.use_mp:
        run_mp(compile_train_graphs_func, jobs, log_directory)
    else:
        run_non_mp(compile_train_graphs_func, jobs, log_directory)
    aligner.logger.debug(f'Compiling training graphs took {time.time() - begin}')


def mono_align_equal_func(mono_directory, job_name, feature_string):
    fst_path = os.path.join(mono_directory, 'fsts.{}.scp'.format(job_name))
    mdl_path = os.path.join(mono_directory, '0.mdl')
    log_path = os.path.join(mono_directory, 'log', 'align.0.{}.log'.format(job_name))
    ali_path = os.path.join(mono_directory, 'ali.{}'.format(job_name))
    acc_path = os.path.join(mono_directory, '0.{}.acc'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        align_proc = subprocess.Popen([thirdparty_binary('align-equal-compiled'), "scp:" + fst_path,
                                       '{}'.format(feature_string), 'ark:' + ali_path],
                                      stderr=log_file)
        align_proc.communicate()
        stats_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), '--binary=true',
                                       mdl_path, '{}'.format(feature_string), 'ark:' + ali_path, acc_path],
                                      stdin=align_proc.stdout, stderr=log_file)
        stats_proc.communicate()


def mono_align_equal(mono_directory, split_directory, num_jobs, config):
    """
    Multiprocessing function that creates equal alignments for base monophone training

    See http://kaldi-asr.org/doc/align-equal-compiled_8cc.html for more details
    on the Kaldi binary this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh
    for the bash script that this function was extracted from.

    Parameters
    ----------
    mono_directory : str
        Directory of monophone training
    split_directory : str
        Directory of training data split into the number of jobs
    num_jobs : int
        The number of processes to use
    """

    jobs = [(mono_directory, x,
             config.feature_config.construct_feature_proc_string(split_directory, mono_directory, x),
             )
            for x in range(num_jobs)]

    if config.use_mp:
        run_mp(mono_align_equal_func, jobs, config.log_directory)
    else:
        run_non_mp(mono_align_equal_func, jobs, config.log_directory)


def align_func(directory, iteration, job_name, mdl, config, feature_string, output_directory, debug=False):
    fst_path = os.path.join(directory, 'fsts.{}.scp'.format(job_name))
    log_path = os.path.join(output_directory, 'log', 'align.{}.{}.log'.format(iteration, job_name))
    ali_path = os.path.join(output_directory, 'ali.{}'.format(job_name))
    score_path = os.path.join(output_directory, 'ali.{}.scores'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        log_file.write('DEBUG: {}'.format(debug))
        if debug:
            loglike_path = os.path.join(output_directory, 'ali.{}.loglikes'.format(job_name))
            com = [thirdparty_binary('gmm-align-compiled'),
                   '--transition-scale={}'.format(config['transition_scale']),
                   '--acoustic-scale={}'.format(config['acoustic_scale']),
                   '--self-loop-scale={}'.format(config['self_loop_scale']),
                   '--beam={}'.format(config['beam']),
                   '--retry-beam={}'.format(config['retry_beam']),
                   '--careful=false',
                   '--write-per-frame-acoustic-loglikes=ark,t:{}'.format(loglike_path),
                   mdl,
                   "scp:" + fst_path, '{}'.format(feature_string), "ark:" + ali_path,
                   "ark,t:" + score_path]
        else:
            com = [thirdparty_binary('gmm-align-compiled'),
                   '--transition-scale={}'.format(config['transition_scale']),
                   '--acoustic-scale={}'.format(config['acoustic_scale']),
                   '--self-loop-scale={}'.format(config['self_loop_scale']),
                   '--beam={}'.format(config['beam']),
                   '--retry-beam={}'.format(config['retry_beam']),
                   '--careful=false',
                   mdl,
                   "scp:" + fst_path, '{}'.format(feature_string), "ark:" + ali_path]
        align_proc = subprocess.Popen(com,
                                      stderr=log_file)
        align_proc.communicate()


def align(iteration, directory, split_directory, optional_silence, num_jobs, config, output_directory=None):
    """
    Multiprocessing function that aligns based on the current model

    See http://kaldi-asr.org/doc/gmm-align-compiled_8cc.html and
    http://kaldi-asr.org/doc/gmm-boost-silence_8cc.html for more details
    on the Kaldi binary this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/align_si.sh
    for the bash script this function was based on.

    Parameters
    ----------
    iteration : int or str
        Iteration to align
    directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    split_directory : str
        Directory of training data split into the number of jobs
    optional_silence : str
        Colon-separated list of silence phones to boost
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.MonophoneConfig`, :class:`~aligner.config.TriphoneConfig` or :class:`~aligner.config.TriphoneFmllrConfig`
        Configuration object for training
    """
    config.logger.info('Performing alignment...')
    begin = time.time()
    if output_directory is None:
        output_directory = directory
    log_directory = os.path.join(output_directory, 'log')
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    if config.boost_silence != 1.0:
        mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('gmm-boost-silence'),
                                               config.boost_silence, optional_silence, make_path_safe(mdl_path))
    else:
        mdl = mdl_path

    jobs = [(directory, iteration, x, mdl, config.align_options,
             config.feature_config.construct_feature_proc_string(split_directory, directory, x),
             output_directory, config.debug) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(align_func, jobs, log_directory)
    else:
        run_non_mp(align_func, jobs, log_directory)

    error_logs = []
    for i in range(num_jobs):
        log_path = os.path.join(output_directory, 'log', 'align.{}.{}.log'.format(iteration, i))
        with open(log_path, 'r', encoding='utf8') as f:
            for line in f:
                if line.strip().startswith('ERROR'):
                    error_logs.append(log_path)
                    break
    if error_logs:
        message = 'There were {} job(s) with errors.  For more information, please see the following logs:\n\n{}'
        raise (AlignmentError(message.format(len(error_logs), '\n'.join(error_logs))))
    config.logger.debug(f'Alignment round took {time.time() - begin}')


def compile_information_func(log_directory, split_directory, job_num):
    align_path = os.path.join(log_directory, 'align.final.{}.log'.format(job_num))

    log_like_pattern = re.compile(
        r'^LOG .* Overall log-likelihood per frame is (?P<log_like>[-0-9.]+) over (?P<frames>\d+) frames.*$')

    decode_error_pattern = re.compile(r'^WARNING .* Did not successfully decode file (?P<utt>.*?), .*$')

    feature_pattern = re.compile(r'Segment (?P<utt>.*?) too short')
    data = {'unaligned': [], 'too_short': [], 'log_like': 0, 'total_frames': 0}
    with open(align_path, 'r', encoding='utf8') as f:
        for line in f:
            decode_error_match = re.match(decode_error_pattern, line)
            if decode_error_match:
                data['unaligned'].append(decode_error_match.group('utt'))
                continue
            log_like_match = re.match(log_like_pattern, line)
            if log_like_match:
                log_like = log_like_match.group('log_like')
                frames = log_like_match.group('frames')
                data['log_like'] = float(log_like)
                data['total_frames'] = int(frames)
    features_path = os.path.join(split_directory, 'log', 'make_mfcc.{}.log'.format(job_num))
    with open(features_path, 'r', encoding='utf8') as f:
        for line in f:
            m = re.search(feature_pattern, line)
            if m is not None:
                utt = m.groups('utt')
                data['too_short'].append(utt)
    return data


def compile_information(model_directory, corpus, num_jobs, config):
    compile_info_begin = time.time()
    log_dir = os.path.join(model_directory, 'log')
    manager = mp.Manager()
    alignment_info = manager.dict()

    jobs = [(log_dir, corpus.split_directory(), x)
            for x in range(num_jobs)]

    if config.use_mp:
        run_mp(compile_information_func, jobs, log_dir, alignment_info)
    else:
        run_non_mp(compile_information_func, jobs, log_dir)

    unaligned = {}
    total_frames = sum(data['total_frames'] for data in alignment_info.values())
    average_log_like = 0
    for x, data in alignment_info.items():
        weight = data['total_frames'] / total_frames
        average_log_like += data['log_like'] * weight
        for u in data['unaligned']:
            unaligned[u] = 'Beam too narrow'
        for u in data['too_short']:
            unaligned[u] = 'Segment too short'

    if not total_frames:
        corpus.logger.warning('No files were aligned, this likely indicates serious problems with the aligner.')
    corpus.logger.debug(f'Compiling information took {time.time() - compile_info_begin}')
    return unaligned, average_log_like


def compute_alignment_improvement_func(iteration, data_directory, model_directory, phones_dir, job_name,
                                       frame_shift, reversed_phone_mapping, positions):
    try:
        text_int_path = os.path.join(data_directory, 'text.{}.int'.format(job_name))
        log_path = os.path.join(model_directory, 'log', 'get_ctm.{}.{}.log'.format(iteration, job_name))
        ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
        model_path = os.path.join(model_directory, '{}.mdl'.format(iteration))
        phone_ctm_path = os.path.join(model_directory, 'phone.{}.{}.ctm'.format(iteration, job_name))
        if os.path.exists(phone_ctm_path):
            return

        frame_shift = frame_shift / 1000
        with open(log_path, 'w', encoding='utf8') as log_file:
            lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                         "ark:" + text_int_path,
                                         '', '', 'ark:-'],
                                        stdout=subprocess.PIPE, stderr=log_file)
            det_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned'),
                                         'ark:-', 'ark:-'],
                                        stdin=lin_proc.stdout, stderr=log_file,
                                        stdout=subprocess.PIPE)
            align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                           os.path.join(phones_dir, 'word_boundary.int'), model_path,
                                           'ark:-', 'ark:-'],
                                          stdin=det_proc.stdout, stderr=log_file,
                                          stdout=subprocess.PIPE)
            phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                           'ark:-', "ark:-"],
                                          stdin=align_proc.stdout,
                                          stdout=subprocess.PIPE,
                                          stderr=log_file)
            nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                           '--frame-shift={}'.format(frame_shift),
                                           "ark:-", phone_ctm_path],
                                          stdin=phone_proc.stdout,
                                          stderr=log_file)
            nbest_proc.communicate()
        mapping = reversed_phone_mapping
        actual_lines = []
        with open(phone_ctm_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(' ')
                utt = line[0]
                begin = Decimal(line[2])
                duration = Decimal(line[3])
                end = begin + duration
                label = line[4]
                try:
                    label = mapping[int(label)]
                except KeyError:
                    pass
                for p in positions:
                    if label.endswith(p):
                        label = label[:-1 * len(p)]
                actual_lines.append([utt, begin, end, label])
        with open(phone_ctm_path, 'w', encoding='utf8') as f:
            for line in actual_lines:
                f.write('{}\n'.format(' '.join(map(str, line))))
    except Exception as e:
        raise (Exception(str(e)))


def parse_iteration_alignments(directory, iteration, num_jobs):
    data = {}
    for j in range(num_jobs):
        phone_ctm_path = os.path.join(directory, 'phone.{}.{}.ctm'.format(iteration, j))
        with open(phone_ctm_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(' ')
                utt = line[0]
                begin = Decimal(line[1])
                end = Decimal(line[2])
                label = line[3]
                if utt not in data:
                    data[utt] = []
                data[utt].append([begin, end, label])
    return data


def compare_alignments(alignments_one, alignments_two, frame_shift):
    utterances_aligned_diff = len(alignments_two) - len(alignments_one)
    utts_one = set(alignments_one.keys())
    utts_two = set(alignments_two.keys())
    common_utts = utts_one.intersection(utts_two)
    differences = []
    for u in common_utts:
        end = alignments_one[u][-1][1]
        t = Decimal('0.0')
        one_alignment = alignments_one[u]
        two_alignment = alignments_two[u]
        difference = 0
        while t < end:
            one_label = None
            two_label = None
            for b, e, l in one_alignment:
                if t < b:
                    continue
                if t >= e:
                    break
                one_label = l
            for b, e, l in two_alignment:
                if t < b:
                    continue
                if t >= e:
                    break
                two_label = l
            if one_label != two_label:
                difference += frame_shift
            t += frame_shift
        difference /= end
        differences.append(difference)
    if differences:
        mean_difference = statistics.mean(differences)
    else:
        mean_difference = 'N/A'
    return utterances_aligned_diff, mean_difference


def compute_alignment_improvement(iteration, config, model_directory, num_jobs):
    jobs = [(iteration, config.data_directory, model_directory, config.dictionary.phones_dir, x,
             config.feature_config.frame_shift, config.dictionary.reversed_phone_mapping, config.dictionary.positions)
            for x in range(num_jobs)]
    if config.use_mp:
        run_mp(compute_alignment_improvement_func, jobs, config.log_directory)
    else:
        run_non_mp(compute_alignment_improvement_func, jobs, config.log_directory)

    alignment_diff_path = os.path.join(model_directory, 'train_change.csv')
    if iteration == 0 or iteration not in config.realignment_iterations:
        return
    ind = config.realignment_iterations.index(iteration)
    if ind != 0:
        previous_iteration = config.realignment_iterations[ind - 1]
    else:
        previous_iteration = 0
    try:
        previous_alignments = parse_iteration_alignments(model_directory, previous_iteration, num_jobs)
    except FileNotFoundError:
        return
    current_alignments = parse_iteration_alignments(model_directory, iteration, num_jobs)
    utterance_aligned_diff, mean_difference = compare_alignments(previous_alignments, current_alignments,
                                                                 config.feature_config.frame_shift)
    if not os.path.exists(alignment_diff_path):
        with open(alignment_diff_path, 'w', encoding='utf8') as f:
            f.write('iteration,number_aligned,number_previously_aligned,'
                    'difference_in_utts_aligned,mean_boundary_change\n')
    if iteration in config.realignment_iterations:
        with open(alignment_diff_path, 'a', encoding='utf8') as f:
            f.write('{},{},{},{},{}\n'.format(iteration, len(current_alignments),
                                              len(previous_alignments), utterance_aligned_diff, mean_difference))
    if not config.debug:
        for j in range(num_jobs):
            phone_ctm_path = os.path.join(model_directory, 'phone.{}.{}.ctm'.format(previous_iteration, j))
            os.remove(phone_ctm_path)


def ali_to_ctm_func(model_directory, word_path, split_directory, job_name, frame_shift, word_mode=True):
    text_int_path = os.path.join(split_directory, 'text.{}.int'.format(job_name))
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, 'final.mdl')
    if word_mode:
        ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(job_name))
        log_path = os.path.join(model_directory, 'log', 'get_word_ctm_.{}.log'.format(job_name))
    else:
        ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(job_name))
        log_path = os.path.join(model_directory, 'log', 'get_phone_ctm_.{}.log'.format(job_name))
    if os.path.exists(ctm_path):
        return

    with open(log_path, 'w', encoding='utf8') as log_file:
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark:-'],
                                    stdout=subprocess.PIPE, stderr=log_file)
        align_words_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                             word_path, model_path,
                                             'ark:-', 'ark:-'],
                                            stdin=lin_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        if word_mode:
            nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                           '--frame-shift={}'.format(frame_shift),
                                           'ark:-',
                                           ctm_path],
                                          stderr=log_file, stdin=align_words_proc.stdout)
        else:
            phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                           'ark:-', "ark:-"],
                                          stdout=subprocess.PIPE, stdin=align_words_proc.stdout,
                                          stderr=log_file)
            nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                           '--frame-shift={}'.format(frame_shift),
                                           "ark:-", ctm_path],
                                          stdin=phone_proc.stdout,
                                          stderr=log_file)
        nbest_proc.communicate()


def process_line(line, utt_begin):
    line = line.split(' ')
    utt = line[0]
    begin = round(float(line[2]), 4)
    duration = float(line[3])
    end = round(begin + duration, 4)
    label = line[4]
    begin += utt_begin
    end += utt_begin
    return utt, begin, end, label


class NoCleanupWordCtmProcessWorker(mp.Process):
    def __init__(self, job_name, ctm_path, to_process_queue, stopped, error_catching,
                 segments, utt_speak_mapping,
                 reversed_word_mapping, speaker_mapping):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.ctm_path = ctm_path
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        # Corpus information
        self.segments = segments
        self.utt_speak_mapping = utt_speak_mapping

        # Dictionary information
        self.reversed_word_mapping = reversed_word_mapping
        self.speaker_mapping = speaker_mapping

    def run(self):
        current_file_data = {}

        def process_current(cur_utt, cur_file, current_labels):
            speaker = self.utt_speak_mapping[cur_utt]
            reversed_word_mapping = self.reversed_word_mapping
            if self.speaker_mapping is not None:
                dict_lookup_speaker = speaker
                if speaker not in self.speaker_mapping:
                    dict_lookup_speaker = 'default'
                reversed_word_mapping = self.reversed_word_mapping[self.speaker_mapping[dict_lookup_speaker]]

            actual_labels = parse_from_word_no_cleanup(current_labels, reversed_word_mapping)
            if speaker not in current_file_data:
                current_file_data[speaker] = []
            current_file_data[speaker].extend(actual_labels)

        def process_current_file(cur_file):
            self.to_process_queue.put(('word', cur_file, current_file_data))

        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        sum_time = 0
        count_time = 0
        try:
            with open(self.ctm_path, 'r') as word_file:
                for line in word_file:
                    line = line.strip()
                    if not line:
                        continue
                    utt, begin, end, label = process_line(line, utt_begin)
                    if cur_utt is None:
                        cur_utt = utt
                        begin_time = time.time()
                        if cur_utt in self.segments:
                            seg = self.segments[cur_utt]
                            cur_file = seg['file_name']
                            utt_begin = seg['begin']
                        else:
                            utt_begin = 0
                            cur_file = utt
                        begin += utt_begin
                        end += utt_begin

                    if utt != cur_utt:
                        process_current(cur_utt, cur_file, current_labels)
                        cur_utt = utt
                        if cur_utt in self.segments:
                            seg = self.segments[cur_utt]
                            file_name = seg['file_name']
                            utt_begin = seg['begin']
                        else:
                            utt_begin = 0
                            file_name = utt
                        if file_name != cur_file:
                            process_current_file(cur_file)
                            current_file_data = {}
                            sum_time += time.time() - begin_time
                            count_time += 1
                            begin_time = time.time()
                            cur_file = file_name
                        current_labels = []
                    current_labels.append([begin, end, label])
            if current_labels:
                process_current(cur_utt, cur_file, current_labels)
                process_current_file(cur_file)
                sum_time += time.time() - begin_time
                count_time += 1
        except Exception as e:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[('word', self.job_name)] = '\n'.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback))


class CleanupWordCtmProcessWorker(mp.Process):
    def __init__(self, job_name, ctm_path, to_process_queue, stopped, error_catching,
                 segments, text_mapping, utt_speak_mapping,
                 words_mapping, speaker_mapping,
                 punctuation, clitic_set, clitic_markers, compound_markers, oov_int):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.ctm_path = ctm_path
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        # Corpus information
        self.segments = segments
        self.text_mapping = text_mapping
        self.utt_speak_mapping = utt_speak_mapping

        # Dictionary information
        self.words_mapping = words_mapping
        self.speaker_mapping = speaker_mapping
        self.punctuation = punctuation
        self.clitic_set = clitic_set
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.oov_int = oov_int

    def run(self):
        current_file_data = {}

        def process_current(cur_utt, cur_file, current_labels):
            text = self.text_mapping[cur_utt].split()
            speaker = self.utt_speak_mapping[cur_utt]
            words_mapping = self.words_mapping
            oov_int = self.oov_int
            if self.speaker_mapping is not None:
                dict_lookup_speaker = speaker
                if speaker not in self.speaker_mapping:
                    dict_lookup_speaker = 'default'
                words_mapping = self.words_mapping[self.speaker_mapping[dict_lookup_speaker]]
                oov_int = self.oov_int[self.speaker_mapping[dict_lookup_speaker]]
            actual_labels = parse_from_word(current_labels, text, words_mapping, self.punctuation, self.clitic_set,
                                            self.clitic_markers, self.compound_markers, oov_int)
            if speaker not in current_file_data:
                current_file_data[speaker] = []
            current_file_data[speaker].extend(actual_labels)

        def process_current_file(cur_file):
            self.to_process_queue.put(('word', cur_file, current_file_data))

        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        sum_time = 0
        count_time = 0
        try:
            with open(self.ctm_path, 'r') as word_file:
                for line in word_file:
                    line = line.strip()
                    if not line:
                        continue
                    utt, begin, end, label = process_line(line, utt_begin)
                    if cur_utt is None:
                        cur_utt = utt
                        begin_time = time.time()
                        if cur_utt in self.segments:
                            seg = self.segments[cur_utt]
                            cur_file = seg['file_name']
                            utt_begin = seg['begin']
                        else:
                            utt_begin = 0
                            cur_file = utt
                        begin += utt_begin
                        end += utt_begin

                    if utt != cur_utt:
                        process_current(cur_utt, cur_file, current_labels)
                        cur_utt = utt
                        if cur_utt in self.segments:
                            seg = self.segments[cur_utt]
                            file_name = seg['file_name']
                            utt_begin = seg['begin']
                        else:
                            utt_begin = 0
                            file_name = utt
                        if file_name != cur_file:
                            process_current_file(cur_file)
                            current_file_data = {}
                            sum_time += time.time() - begin_time
                            count_time += 1
                            begin_time = time.time()
                            cur_file = file_name
                        current_labels = []
                    current_labels.append([begin, end, label])
            if current_labels:
                process_current(cur_utt, cur_file, current_labels)
                process_current_file(cur_file)
                sum_time += time.time() - begin_time
                count_time += 1
        except Exception as e:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[('word', self.job_name)] = '\n'.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback))


class PhoneCtmProcessWorker(mp.Process):
    def __init__(self, job_name, ctm_path, to_process_queue, stopped, error_catching,
                 segments, utt_speak_mapping,
                 reversed_phone_mapping, speaker_mapping, positions):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.ctm_path = ctm_path
        self.to_process_queue = to_process_queue
        self.stopped = stopped
        self.error_catching = error_catching

        self.segments = segments
        self.utt_speak_mapping = utt_speak_mapping

        self.reversed_phone_mapping = reversed_phone_mapping
        self.speaker_mapping = speaker_mapping
        self.positions = positions

    def run(self):
        main_begin = time.time()
        cur_utt = None
        cur_file = None
        utt_begin = 0
        current_labels = []
        sum_time = 0
        count_time = 0

        current_file_data = {}

        def process_current_utt(cur_utt, cur_file, current_labels):
            speaker = self.utt_speak_mapping[cur_utt]

            reversed_phone_mapping = self.reversed_phone_mapping
            if self.speaker_mapping is not None:
                dict_lookup_speaker = speaker
                if speaker not in self.speaker_mapping:
                    dict_lookup_speaker = 'default'
                reversed_phone_mapping = self.reversed_phone_mapping[self.speaker_mapping[dict_lookup_speaker]]
            actual_labels = parse_from_phone(current_labels, reversed_phone_mapping, self.positions)
            if speaker not in current_file_data:
                current_file_data[speaker] = []
            current_file_data[speaker].extend(actual_labels)

        def process_current_file(cur_file):
            self.to_process_queue.put(('phone', cur_file, current_file_data))

        try:
            with open(self.ctm_path, 'r') as word_file:
                for line in word_file:
                    line = line.strip()
                    if not line:
                        continue
                    utt, begin, end, label = process_line(line, utt_begin)
                    if cur_utt is None:
                        cur_utt = utt
                        begin_time = time.time()
                        if cur_utt in self.segments:
                            seg = self.segments[cur_utt]
                            cur_file = seg['file_name']
                            utt_begin = seg['begin']
                        else:
                            utt_begin = 0
                            cur_file = utt
                        begin += utt_begin
                        end += utt_begin

                    if utt != cur_utt:

                        process_current_utt(cur_utt, cur_file, current_labels)

                        cur_utt = utt
                        if cur_utt in self.segments:
                            seg = self.segments[cur_utt]
                            file_name = seg['file_name']
                            utt_begin = seg['begin']
                        else:
                            utt_begin = 0
                            file_name = utt
                        if file_name != cur_file:
                            process_current_file(cur_file)
                            current_file_data = {}
                            sum_time += time.time() - begin_time
                            count_time += 1
                            begin_time = time.time()
                            cur_file = file_name
                        current_labels = []
                    current_labels.append([begin, end, label])
            if current_labels:
                process_current_utt(cur_utt, cur_file, current_labels)
                process_current_file(cur_file)
                sum_time += time.time() - begin_time
                count_time += 1
        except Exception as e:
            self.stopped.stop()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.error_catching[('phone', self.job_name)] = '\n'.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback))


class CombineProcessWorker(mp.Process):
    def __init__(self, job_name, to_process_queue, to_export_queue, stopped, finished_combining, error_catching,
                 silences, multilingual_ipa, words_mapping, speaker_mapping,
                 punctuation, clitic_set, clitic_markers, compound_markers, oov_code, words,
                 strip_diacritics, cleanup_textgrids):
        mp.Process.__init__(self)
        self.job_name = job_name
        self.to_process_queue = to_process_queue
        self.to_export_queue = to_export_queue
        self.stopped = stopped
        self.finished_combining = finished_combining
        self.error_catching = error_catching

        self.silences = silences
        self.multilingual_ipa = multilingual_ipa
        self.words_mapping = words_mapping
        self.speaker_mapping = speaker_mapping
        self.punctuation = punctuation
        self.clitic_set = clitic_set
        self.clitic_markers = clitic_markers
        self.compound_markers = compound_markers
        self.oov_code = oov_code
        self.words = words
        self.strip_diacritics = strip_diacritics
        self.cleanup_textgrids = cleanup_textgrids

    def run(self):
        sum_time = 0
        count_time = 0
        phone_data = {}
        word_data = {}
        while True:
            try:
                w_p, file_name, data = self.to_process_queue.get(timeout=queue_polling_timeout)
                begin_time = time.time()
            except Empty as error:
                if self.finished_combining.stop_check():
                    break
                continue
            self.to_process_queue.task_done()
            if self.stopped.stop_check():
                continue
            if w_p == 'phone':
                if file_name in word_data:
                    word_ctm = word_data.pop(file_name)
                    phone_ctm = data
                else:
                    phone_data[file_name] = data
                    continue
            else:
                if file_name in phone_data:
                    phone_ctm = phone_data.pop(file_name)
                    word_ctm = data
                else:
                    word_data[file_name] = data
                    continue

            try:
                data = generate_tiers(word_ctm, phone_ctm, self.silences, self.multilingual_ipa,
                                      self.words_mapping, self.speaker_mapping,
                                      self.punctuation, self.clitic_set, self.clitic_markers, self.compound_markers,
                                      self.oov_code, self.words,
                                      self.strip_diacritics, cleanup_textgrids=self.cleanup_textgrids)
                self.to_export_queue.put((file_name, data))
            except Exception as e:
                self.stopped.stop()
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.error_catching[('combining', self.job_name)] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))

            sum_time += time.time() - begin_time
            count_time += 1


class ExportTextGridProcessWorker(mp.Process):
    def __init__(self, for_write_queue, stopped, finished_processing, textgrid_errors,
                 out_directory, backup_output_directory, wav_durations,
                 frame_shift, file_directory_mapping, file_name_mapping, speaker_ordering):
        mp.Process.__init__(self)
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_processing = finished_processing
        self.textgrid_errors = textgrid_errors

        self.out_directory = out_directory
        self.backup_output_directory = backup_output_directory

        self.wav_durations = wav_durations
        self.frame_shift = frame_shift
        self.file_directory_mapping = file_directory_mapping
        self.file_name_mapping = file_name_mapping
        self.speaker_ordering = speaker_ordering

    def run(self):
        while True:
            try:
                file_name, data = self.for_write_queue.get(timeout=queue_polling_timeout)
            except Empty as error:
                if self.finished_processing.stop_check():
                    break
                continue
            self.for_write_queue.task_done()
            if self.stopped.stop_check():
                continue
            overwrite = True
            speaker = None
            if len(data) == 1:
                speaker = next(iter(data))
            output_name, output_path = construct_output_path(file_name, self.out_directory,
                                                             self.file_directory_mapping, self.file_name_mapping,
                                                             speaker, self.backup_output_directory)
            max_time = round(self.wav_durations[output_name], 4)
            try:
                export_textgrid(file_name, output_path, data, max_time,
                                self.frame_shift, self.speaker_ordering, overwrite)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.textgrid_errors[file_name] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))


class ExportPreparationProcessWorker(mp.Process):
    def __init__(self, to_export_queue, for_write_queue, stopped, finished_combining, file_speaker_mapping):
        mp.Process.__init__(self)
        self.to_export_queue = to_export_queue
        self.for_write_queue = for_write_queue
        self.stopped = stopped
        self.finished_combining = finished_combining

        self.file_speaker_mapping = file_speaker_mapping

    def run(self):
        export_data = {}
        while True:
            try:
                file_name, data = self.to_export_queue.get(timeout=queue_polling_timeout)
            except Empty as error:
                if self.finished_combining.stop_check():
                    break
                continue
            self.to_export_queue.task_done()
            if self.stopped.stop_check():
                continue
            if file_name in self.file_speaker_mapping and len(self.file_speaker_mapping[file_name]) > 1:
                if file_name not in export_data:
                    export_data[file_name] = data
                else:
                    export_data[file_name].update(data)
                if len(export_data[file_name]) == len(self.file_speaker_mapping[file_name]):
                    data = export_data.pop(file_name)
                    self.for_write_queue.put((file_name, data))
            else:
                self.for_write_queue.put((file_name, data))

        for k, v in export_data.items():
            self.for_write_queue.put((k, v))


def ctms_to_textgrids_mp(align_config, output_directory, model_directory, dictionary, corpus, num_jobs):
    frame_shift = align_config.feature_config.frame_shift / 1000
    export_begin = time.time()
    manager = mp.Manager()
    textgrid_errors = manager.dict()
    error_catching = manager.dict()
    stopped = Stopped()
    backup_output_directory = None
    if not align_config.overwrite:
        backup_output_directory = os.path.join(model_directory, 'textgrids')
        os.makedirs(backup_output_directory, exist_ok=True)

    if dictionary.has_multiple:
        words_mapping = {}
        words = {}
        reversed_phone_mapping = {}
        reversed_word_mapping = {}
        for name, d in dictionary.dictionary_mapping.items():
            words_mapping[name] = d.words_mapping
            words[name] = d.words
            reversed_phone_mapping[name] = d.reversed_phone_mapping
            reversed_word_mapping[name] = d.reversed_word_mapping
        speaker_mapping = dictionary.speaker_mapping
        oov_int = {name: d.oov_int for name, d in dictionary.dictionary_mapping.items()}
    else:
        words_mapping = dictionary.words_mapping
        words = dictionary.words
        reversed_phone_mapping = dictionary.reversed_phone_mapping
        reversed_word_mapping = dictionary.reversed_word_mapping
        speaker_mapping = None
        oov_int = dictionary.oov_int
    punctuation = dictionary.punctuation
    clitic_set = dictionary.clitic_set
    clitic_markers = dictionary.clitic_markers
    compound_markers = dictionary.compound_markers

    corpus.logger.debug('Starting combination process...')
    silences = dictionary.silences
    corpus.logger.debug('Starting export process...')

    corpus.logger.debug('Beginning to process ctm files...')
    ctm_begin_time = time.time()
    word_procs = []
    phone_procs = []
    combine_procs = []
    finished_signals = [Stopped() for _ in range(num_jobs)]
    finished_processing = Stopped()
    to_process_queue = [mp.JoinableQueue() for _ in range(num_jobs)]
    to_export_queue = mp.JoinableQueue()
    for_write_queue = mp.JoinableQueue()
    finished_combining = Stopped()
    for i in range(num_jobs):
        word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(i))
        phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(i))
        if align_config.cleanup_textgrids:
            word_p = CleanupWordCtmProcessWorker(i, word_ctm_path, to_process_queue[i], stopped, error_catching,
                                                 corpus.segments, corpus.text_mapping, corpus.utt_speak_mapping,
                                                 words_mapping, speaker_mapping,
                                                 punctuation, clitic_set, clitic_markers, compound_markers, oov_int)
        else:
            print('no clean up!')
            word_p = NoCleanupWordCtmProcessWorker(i, word_ctm_path, to_process_queue[i], stopped, error_catching,
                                                   corpus.segments, corpus.utt_speak_mapping, reversed_word_mapping,
                                                   speaker_mapping)

        word_procs.append(word_p)
        word_p.start()

        phone_p = PhoneCtmProcessWorker(i, phone_ctm_path, to_process_queue[i], stopped, error_catching,
                                        corpus.segments, corpus.utt_speak_mapping, reversed_phone_mapping,
                                        speaker_mapping,
                                        dictionary.positions)
        phone_p.start()
        phone_procs.append(phone_p)

        combine_p = CombineProcessWorker(i, to_process_queue[i], to_export_queue, stopped, finished_signals[i],
                                         error_catching,
                                         silences,
                                         dictionary.multilingual_ipa, words_mapping, speaker_mapping,
                                         punctuation, clitic_set, clitic_markers, compound_markers,
                                         dictionary.oov_code, words, dictionary.strip_diacritics,
                                         align_config.cleanup_textgrids)
        combine_p.start()
        combine_procs.append(combine_p)
    preparation_proc = ExportPreparationProcessWorker(to_export_queue, for_write_queue, stopped, finished_combining,
                                                      corpus.file_speaker_mapping)
    preparation_proc.start()

    export_procs = []
    for i in range(num_jobs):
        export_proc = ExportTextGridProcessWorker(for_write_queue, stopped, finished_processing, textgrid_errors,
                                                  output_directory, backup_output_directory, corpus.file_durations,
                                                  frame_shift, corpus.file_directory_mapping, corpus.file_name_mapping,
                                                  corpus.speaker_ordering)
        export_proc.start()
        export_procs.append(export_proc)

    corpus.logger.debug('Waiting for processes to finish...')
    for i in range(num_jobs):
        word_procs[i].join()
        phone_procs[i].join()
        finished_signals[i].stop()

    corpus.logger.debug(f'Ctm parsers took {time.time() - ctm_begin_time} seconds')

    corpus.logger.debug('Waiting for processes to finish...')
    for i in range(num_jobs):
        to_process_queue[i].join()
        combine_procs[i].join()
    finished_combining.stop()

    to_export_queue.join()
    preparation_proc.join()

    corpus.logger.debug(f'Combiners took {time.time() - ctm_begin_time} seconds')
    corpus.logger.debug('Beginning export...')

    corpus.logger.debug(f'Adding jobs for export took {time.time() - export_begin}')
    corpus.logger.debug('Waiting for export processes to join...')

    for_write_queue.join()
    finished_processing.stop()
    for i in range(num_jobs):
        export_procs[i].join()
    for_write_queue.join()
    corpus.logger.debug(f'Export took {time.time() - export_begin} seconds')

    if error_catching:
        corpus.logger.error('Error was encountered in processing CTMs')
        for key, error in error_catching.items():
            corpus.logger.error(f'{key}:\n\n{error}')
        raise AlignmentError()

    output_textgrid_writing_errors(output_directory, textgrid_errors)


def convert_ali_to_textgrids(align_config, output_directory, model_directory, dictionary, corpus, num_jobs):
    """
    Multiprocessing function that aligns based on the current model

    See:

    - http://kaldi-asr.org/doc/linear-to-nbest_8cc.html
    - http://kaldi-asr.org/doc/lattice-align-words_8cc.html
    - http://kaldi-asr.org/doc/lattice-to-phone-lattice_8cc.html
    - http://kaldi-asr.org/doc/nbest-to-ctm_8cc.html

    for more details
    on the Kaldi binaries this function calls.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/get_train_ctm.sh
    for the bash script that this function was based on.

    Parameters
    ----------
    output_directory : str
        Directory to write TextGrid files to
    model_directory : str
        Directory of training (monophone, triphone, speaker-adapted triphone
        training directories)
    dictionary : :class:`~montreal_forced_aligner.dictionary.Dictionary`
        Dictionary object that has information about pronunciations
    corpus : :class:`~montreal_forced_aligner.corpus.AlignableCorpus`
        Corpus object that has information about the dataset
    num_jobs : int
        The number of processes to use in calculation

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)

    """
    log_directory = os.path.join(model_directory, 'log')
    frame_shift = align_config.feature_config.frame_shift / 1000
    word_path = os.path.join(dictionary.phones_dir, 'word_boundary.int')
    jobs = [(model_directory, word_path, corpus.split_directory(), x, frame_shift, True)  # Word CTM jobs
            for x in range(num_jobs)]
    jobs += [(model_directory, word_path, corpus.split_directory(), x, frame_shift, False)  # Phone CTM jobs
             for x in range(num_jobs)]
    corpus.logger.info('Generating CTMs from alignment...')
    if align_config.use_mp:
        run_mp(ali_to_ctm_func, jobs, log_directory)
    else:
        run_non_mp(ali_to_ctm_func, jobs, log_directory)
    corpus.logger.info('Finished generating CTMs!')

    corpus.logger.info('Exporting TextGrids from CTMs...')
    if align_config.use_mp:
        ctms_to_textgrids_mp(align_config, output_directory, model_directory, dictionary, corpus, num_jobs)
    else:
        ctms_to_textgrids_non_mp(align_config, output_directory, model_directory, dictionary, corpus, num_jobs)
    corpus.logger.info('Finished exporting TextGrids!')


def tree_stats_func(directory, ci_phones, mdl, feature_string, ali_path, job_name):
    context_opts = []
    log_path = os.path.join(directory, 'log', 'acc_tree.{}.log'.format(job_name))

    treeacc_path = os.path.join(directory, '{}.treeacc'.format(job_name))

    with open(log_path, 'w', encoding='utf8') as log_file:
        subprocess.call([thirdparty_binary('acc-tree-stats')] + context_opts +
                        ['--ci-phones=' + ci_phones, mdl, '{}'.format(feature_string),
                         "ark:" + ali_path,
                         treeacc_path], stderr=log_file)


def tree_stats(directory, align_directory, split_directory, ci_phones, num_jobs, config):
    """
    Multiprocessing function that computes stats for decision tree training

    See http://kaldi-asr.org/doc/acc-tree-stats_8cc.html for more details
    on the Kaldi binary this runs.

    Parameters
    ----------
    directory : str
        Directory of training (triphone, speaker-adapted triphone
        training directories)
    align_directory : str
        Directory of previous alignment
    split_directory : str
        Directory of training data split into the number of jobs
    ci_phones : str
        Colon-separated list of context-independent phones
    num_jobs : int
        The number of processes to use in calculation
    """

    mdl_path = os.path.join(align_directory, 'final.mdl')

    jobs = [(directory, ci_phones, mdl_path,
             config.feature_config.construct_feature_proc_string(split_directory, directory, x),
             os.path.join(align_directory, 'ali.{}'.format(x)), x) for x in range(num_jobs)]

    if config.use_mp:
        run_mp(tree_stats_func, jobs, config.log_directory)
    else:
        run_non_mp(tree_stats_func, jobs, config.log_directory)

    tree_accs = [os.path.join(directory, '{}.treeacc'.format(x)) for x in range(num_jobs)]
    log_path = os.path.join(directory, 'log', 'sum_tree_acc.log')
    with open(log_path, 'w', encoding='utf8') as log_file:
        subprocess.call([thirdparty_binary('sum-tree-stats'), os.path.join(directory, 'treeacc')] +
                        tree_accs, stderr=log_file)
    # for f in tree_accs:
    #    os.remove(f)


def convert_alignments_func(directory, align_directory, job_name):
    mdl_path = os.path.join(directory, '1.mdl')
    tree_path = os.path.join(directory, 'tree')
    ali_mdl_path = os.path.join(align_directory, 'final.mdl')
    ali_path = os.path.join(align_directory, 'ali.{}'.format(job_name))
    new_ali_path = os.path.join(directory, 'ali.{}'.format(job_name))

    log_path = os.path.join(directory, 'log', 'convert.{}.log'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subprocess.call([thirdparty_binary('convert-ali'), ali_mdl_path,
                         mdl_path, tree_path, "ark:" + ali_path,
                         "ark:" + new_ali_path], stderr=log_file)


def convert_alignments(directory, align_directory, num_jobs, config):
    """
    Multiprocessing function that converts alignments from previous training

    See http://kaldi-asr.org/doc/convert-ali_8cc.html for more details
    on the Kaldi binary this runs.

    Parameters
    ----------
    directory : str
        Directory of training (triphone, speaker-adapted triphone
        training directories)
    align_directory : str
        Directory of previous alignment
    num_jobs : int
        The number of processes to use in calculation

    """

    jobs = [(directory, align_directory, x)
            for x in range(num_jobs)]
    if config.use_mp:
        run_mp(convert_alignments_func, jobs, config.log_directory)
    else:
        run_non_mp(convert_alignments_func, jobs, config.log_directory)


def calc_fmllr_func(directory, split_directory, sil_phones, job_name, feature_string, config, initial,
                    model_name='final'):
    log_path = os.path.join(directory, 'log', 'fmllr.{}.{}.log'.format(model_name, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_name))
    if not initial:
        tmp_trans_path = os.path.join(directory, 'trans.temp.{}'.format(job_name))
    else:
        tmp_trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                      "ark:" + ali_path, 'ark:-'], stderr=log_file, stdout=subprocess.PIPE)

        weight_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'), '0.0',
                                        sil_phones, mdl_path, 'ark:-',
                                        'ark:-'], stderr=log_file, stdin=post_proc.stdout, stdout=subprocess.PIPE)

        if not initial:
            trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
            cmp_trans_path = os.path.join(directory, 'trans.cmp.{}'.format(job_name))
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                         '--verbose=4',
                                         '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                         '--spk2utt=ark:' + spk2utt_path, mdl_path, '{}'.format(feature_string),
                                         'ark:-', 'ark:-'],
                                        stderr=log_file, stdin=weight_proc.stdout, stdout=subprocess.PIPE)
            comp_proc = subprocess.Popen([thirdparty_binary('compose-transforms'),
                                          '--b-is-affine=true',
                                          'ark:-', 'ark:' + trans_path,
                                          'ark:' + cmp_trans_path], stderr=log_file, stdin=est_proc.stdout)
            comp_proc.communicate()

            os.remove(trans_path)
            os.rename(cmp_trans_path, trans_path)
        else:
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                         '--verbose=4',
                                         '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                         '--spk2utt=ark:' + spk2utt_path, mdl_path, '{}'.format(feature_string),
                                         'ark,s,cs:-', 'ark:' + tmp_trans_path],
                                        stderr=log_file, stdin=weight_proc.stdout)
            est_proc.communicate()


def calc_fmllr(directory, split_directory, sil_phones, num_jobs, config,
               initial=False, iteration=None):
    """
    Multiprocessing function that computes speaker adaptation (fMLLR)

    See:

    - http://kaldi-asr.org/doc/gmm-est-fmllr_8cc.html
    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/compose-transforms_8cc.html
    - http://kaldi-asr.org/doc/transform-feats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/align_fmllr.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of training (triphone, speaker-adapted triphone
        training directories)
    split_directory : str
        Directory of training data split into the number of jobs
    sil_phones : str
        Colon-separated list of silence phones
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.TriphoneFmllrConfig`
        Configuration object for training
    initial : bool, optional
        Whether this is the first computation of speaker-adaptation,
        defaults to False
    iteration : int or str
        Specifies the current iteration, defaults to None

    """
    config.logger.info('Calculating fMLLR for speaker adaptation...')
    begin = time.time()
    if iteration is None:
        if initial:
            model_name = '1'
        else:
            model_name = 'final'
    else:
        model_name = iteration
    log_directory = os.path.join(directory, 'log')

    jobs = [(directory, split_directory, sil_phones, x,
             config.feature_config.construct_feature_proc_string(split_directory, directory, x),
             config, initial, model_name) for x in range(num_jobs)]
    if config.use_fmllr_mp:
        run_mp(calc_fmllr_func, jobs, log_directory)
    else:
        run_non_mp(calc_fmllr_func, jobs, log_directory)
    config.logger.debug(f'Fmllr calculation took {time.time() - begin}')


def lda_acc_stats_func(directory, feature_string, align_directory, config, ci_phones, i):
    log_path = os.path.join(directory, 'log', 'ali_to_post.{}.log'.format(i))
    with open(log_path, 'w', encoding='utf8') as log_file:
        ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                             'ark:' + os.path.join(align_directory, 'ali.{}'.format(i)),
                                             'ark:-'],
                                            stderr=log_file, stdout=subprocess.PIPE)
        weight_silence_post_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                     str(config['boost_silence']), ci_phones,
                                                     os.path.join(align_directory, 'final.mdl'),
                                                     'ark:-', 'ark:-'],
                                                    stdin=ali_to_post_proc.stdout,
                                                    stderr=log_file, stdout=subprocess.PIPE)
        acc_lda_post_proc = subprocess.Popen([thirdparty_binary('acc-lda'),
                                              '--rand-prune=' + str(config['random_prune']),
                                              os.path.join(align_directory, 'final.mdl'),
                                              '{}'.format(feature_string),
                                              'ark,s,cs:-',
                                              os.path.join(directory, 'lda.{}.acc'.format(i))],
                                             stdin=weight_silence_post_proc.stdout,
                                             stderr=log_file)
        acc_lda_post_proc.communicate()


def lda_acc_stats(directory, split_directory, align_directory, config, ci_phones, num_jobs):
    """
    Multiprocessing function that accumulates LDA statistics

    See:

    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/acc-lda_8cc.html
    - http://kaldi-asr.org/doc/est-lda_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_lda_mllt.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of LDA+MLLT training
    split_directory : str
        Directory of training data split into the number of jobs
    align_directory : str
        Directory of previous alignment
    config : :class:`~aligner.config.LdaMlltConfig`
        Configuration object for training
    ci_phones : str
        Colon-separated list of context-independent phones
    num_jobs : int
        The number of processes to use in calculation

    """
    jobs = [(directory,
             config.feature_config.construct_feature_proc_string(split_directory, directory, x, splice=True),
             align_directory, config.lda_options, ci_phones, x) for x in range(num_jobs)]

    if config.use_mp:
        run_mp(lda_acc_stats_func, jobs, config.log_directory)
    else:
        run_non_mp(lda_acc_stats_func, jobs, config.log_directory)

    log_path = os.path.join(directory, 'log', 'lda_est.log')
    acc_list = []
    for x in range(num_jobs):
        acc_list.append(os.path.join(directory, 'lda.{}.acc'.format(x)))
    with open(log_path, 'w', encoding='utf8') as log_file:
        est_lda_proc = subprocess.Popen([thirdparty_binary('est-lda'),
                                         '--write-full-matrix=' + os.path.join(directory, 'full.mat'),
                                         '--dim=' + str(config.lda_dimension),
                                         os.path.join(directory, 'lda.mat')] + acc_list,
                                        stderr=log_file)
        est_lda_proc.communicate()


def calc_lda_mllt_func(directory, feature_string, sil_phones, job_name, config,
                       initial,
                       model_name='final'):
    log_path = os.path.join(directory, 'log', 'lda_mllt.{}.{}.log'.format(model_name, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    if not initial:
        mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    else:
        mdl_path = os.path.join(directory, '1.mdl')
        model_name = 1

    # Estimating MLLT
    with open(log_path, 'a', encoding='utf8') as log_file:
        post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                      "ark:" + ali_path, 'ark:-'],
                                     stdout=subprocess.PIPE, stderr=log_file)

        weight_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'), '0.0',
                                        sil_phones, mdl_path, 'ark:-',
                                        'ark:-'],
                                       stdin=post_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-mllt'),
                                     '--rand-prune=' + str(config['random_prune']),
                                     mdl_path,
                                     '{}'.format(feature_string),
                                     'ark:-',
                                     os.path.join(directory, '{}.{}.macc'.format(model_name, job_name))],
                                    stdin=weight_proc.stdout, stderr=log_file)
        acc_proc.communicate()


def calc_lda_mllt(directory, data_directory, sil_phones, num_jobs, config,
                  initial=False, iteration=None):
    """
    Multiprocessing function that calculates LDA+MLLT transformations

    See:

    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/gmm-acc-mllt_8cc.html
    - http://kaldi-asr.org/doc/est-mllt_8cc.html
    - http://kaldi-asr.org/doc/gmm-transform-means_8cc.html
    - http://kaldi-asr.org/doc/compose-transforms_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_lda_mllt.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of LDA+MLLT training
    data_directory : str
        Directory of training data split into the number of jobs
    sil_phones : str
        Colon-separated list of silence phones
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.LdaMlltConfig`
        Configuration object for training
    initial : bool
        Flag for first iteration
    iteration : int
        Current iteration

    """
    if iteration is None:
        model_name = 'final'
    else:
        model_name = iteration
    jobs = [(directory,
             config.feature_config.construct_feature_proc_string(data_directory, directory, x),
             sil_phones, x, config.lda_options, initial, model_name) for x in range(num_jobs)]

    if config.use_mp:
        run_mp(calc_lda_mllt_func, jobs, config.log_directory)
    else:
        run_non_mp(calc_lda_mllt_func, jobs, config.log_directory)

    mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    log_path = os.path.join(directory, 'log', 'transform_means.{}.log'.format(model_name))
    previous_mat_path = os.path.join(directory, 'lda.mat')
    new_mat_path = os.path.join(directory, 'lda_new.mat')
    composed_path = os.path.join(directory, 'lda_composed.mat')
    with open(log_path, 'a', encoding='utf8') as log_file:
        macc_list = []
        for x in range(num_jobs):
            macc_list.append(os.path.join(directory, '{}.{}.macc'.format(model_name, x)))
        subprocess.call([thirdparty_binary('est-mllt'),
                         new_mat_path]
                        + macc_list,
                        stderr=log_file)
        subprocess.call([thirdparty_binary('gmm-transform-means'),
                         new_mat_path,
                         mdl_path, mdl_path],
                        stderr=log_file)

        if os.path.exists(previous_mat_path):
            subprocess.call([thirdparty_binary('compose-transforms'),
                             new_mat_path,
                             previous_mat_path,
                             composed_path],
                            stderr=log_file)
            os.remove(previous_mat_path)
            os.rename(composed_path, previous_mat_path)
        else:
            os.rename(new_mat_path, previous_mat_path)
