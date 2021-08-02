import subprocess
import os
import shutil
import re
from decimal import Decimal
import statistics

from .helper import make_path_safe, run_mp, run_non_mp, thirdparty_binary

from ..textgrid import ctm_to_textgrid, parse_ctm

from ..exceptions import AlignmentError


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


def acc_stats_func(directory, iteration, job_name, feature_string):
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, job_name))
    model_path = os.path.join(directory, '{}.mdl'.format(iteration))
    acc_path = os.path.join(directory, '{}.{}.acc'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), model_path,
                                     '{}'.format(feature_string), "ark,t:" + ali_path, acc_path],
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

    log_path = os.path.join(directory, 'log', 'show_transition.log')
    transition_path = os.path.join(directory, 'transitions.txt')
    triphones_file_path = os.path.join(directory, 'triphones.txt')

    if dictionary_names is None:
        phones_file_path = os.path.join(lang_directory, 'phones.txt')
        if debug:
            with open(log_path, 'w', encoding='utf8') as log_file:
                with open(transition_path, 'w', encoding='utf8') as f:
                    subprocess.call([thirdparty_binary('show-transitions'), phones_file_path, mdl_path],
                                    stdout=f, stderr=log_file)
                parse_transitions(transition_path, triphones_file_path)
        log_path = os.path.join(directory, 'log', 'compile-graphs.{}.log'.format(job_name))

        if os.path.exists(triphones_file_path):
            phones_file_path = triphones_file_path
        words_file_path = os.path.join(lang_directory, 'words.txt')
        fst_scp_path = os.path.join(directory, 'fsts.{}.scp'.format(job_name))
        fst_ark_path = os.path.join(directory, 'fsts.{}.ark'.format(job_name))
        text_path = os.path.join(split_directory, 'text.{}.int'.format(job_name))

        with open(log_path, 'w', encoding='utf8') as log_file:
            proc = subprocess.Popen([thirdparty_binary('compile-train-graphs'),
                                     '--read-disambig-syms={}'.format(
                                         os.path.join(lang_directory, 'phones', 'disambig.int')),
                                     tree_path, mdl_path,
                                     os.path.join(lang_directory, 'L.fst'),
                                     "ark:"+ text_path, "ark,scp:{},{}".format(fst_ark_path, fst_scp_path)],
                                    stderr=log_file)
            proc.communicate()
    else:
        for name in dictionary_names:
            phones_file_path = os.path.join(lang_directory, 'phones.txt')
            if debug:
                with open(log_path, 'w', encoding='utf8') as log_file:
                    with open(transition_path, 'w', encoding='utf8') as f:
                        subprocess.call([thirdparty_binary('show-transitions'), phones_file_path, mdl_path],
                                        stdout=f, stderr=log_file)
                    parse_transitions(transition_path, triphones_file_path)
            log_path = os.path.join(directory, 'log', 'compile-graphs.{}.{}.log'.format(job_name, name))

            if os.path.exists(triphones_file_path):
                phones_file_path = triphones_file_path
            words_file_path = os.path.join(lang_directory, 'words.txt')
            fst_scp_path = os.path.join(directory, 'fsts.{}.{}.scp'.format(job_name, name))
            fst_ark_path = os.path.join(directory, 'fsts.{}.{}.ark'.format(job_name, name))
            text_path = os.path.join(split_directory, 'text.{}.{}.int'.format(job_name, name))
            with open(log_path, 'w', encoding='utf8') as log_file:

                proc = subprocess.Popen([thirdparty_binary('compile-train-graphs'),
                                         '--read-disambig-syms={}'.format(
                                             os.path.join(lang_directory, 'phones', 'disambig.int')),
                                         tree_path, mdl_path,
                                         os.path.join(lang_directory, name, 'dictionary', 'L.fst'),
                                         "ark:"+text_path, "ark,scp:{},{}".format(fst_ark_path, fst_scp_path)],
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


    if debug:
        utterances = []
        with open(os.path.join(split_directory, 'utt2spk.{}'.format(job_name)), 'r', encoding='utf8') as f:
            for line in f:
                utt = line.split()[0].strip()
                if not utt:
                    continue
                utterances.append(utt)

        with open(log_path, 'a', encoding='utf8') as log_file:

            temp_fst_path = os.path.join(directory, 'temp.fst.{}'.format(job_name))

            with open(fst_scp_path, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    utt = line.split()[0]

                    dot_path = os.path.join(directory, '{}.dot'.format(utt))
                    fst_proc = subprocess.Popen([thirdparty_binary('fstcopy'),
                                                 'scp:-',
                                                 'scp:echo {} {}|'.format(utt, temp_fst_path)],
                                                stdin=subprocess.PIPE, stderr=log_file)
                    fst_proc.communicate(input=line.encode())

                    draw_proc = subprocess.Popen([thirdparty_binary('fstdraw'), '--portrait=true',
                                                  '--isymbols={}'.format(phones_file_path),
                                                  '--osymbols={}'.format(words_file_path), temp_fst_path,
                                                  dot_path],
                                                 stderr=log_file)
                    draw_proc.communicate()
                    try:
                        dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-O', dot_path],
                                                    stderr=log_file)
                        dot_proc.communicate()
                    except FileNotFoundError:
                        pass


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
    log_directory = os.path.join(directory, 'log')
    os.makedirs(log_directory, exist_ok=True)
    jobs = [(directory, lang_directory, split_directory, x, aligner.dictionaries_for_job(x), debug)
            for x in range(num_jobs)]
    if aligner.use_mp:
        run_mp(compile_train_graphs_func, jobs, log_directory)
    else:
        run_non_mp(compile_train_graphs_func, jobs, log_directory)


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
                                       "scp:" + fst_path, '{}'.format(feature_string), "ark,t:" + ali_path,
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
                                       "scp:" + fst_path, '{}'.format(feature_string), "ark,t:" + ali_path,
                                       "ark,t:" + score_path]
        align_proc = subprocess.Popen(com,
                                      stderr=log_file)
        align_proc.communicate()


def align(iteration, directory, split_directory, optional_silence, num_jobs, config, output_directory=None, debug=False):
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
    if output_directory is None:
        output_directory = directory
    log_directory = os.path.join(output_directory, 'log')
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('gmm-boost-silence'),
                                           config.boost_silence, optional_silence, make_path_safe(mdl_path))

    jobs = [(directory, iteration, x, mdl, config.align_options,
             config.feature_config.construct_feature_proc_string(split_directory, directory, x),
                     output_directory) for x in range(num_jobs)]

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


def compile_information_func(log_directory, corpus, job_num):
    align_path = os.path.join(log_directory, 'align.final.{}.log'.format(job_num))
    unaligned = {}
    output_path = os.path.join(log_directory, 'unaligned.{}.log'.format(job_num))
    with open(align_path, 'r', encoding='utf8') as f:
        for line in f:
            m = re.search(r'Did not successfully decode file (.*?),', line)
            if m is not None:
                utt = m.groups()[0]
                unaligned[utt] = 'Could not decode (beam too narrow)'
    features_path = os.path.join(corpus.split_directory(), 'log', 'make_mfcc.{}.log'.format(job_num))
    with open(features_path, 'r', encoding='utf8') as f:
        for line in f:
            m = re.search(r'Segment (.*?) too short', line)
            if m is not None:
                utt = m.groups()[0]
                unaligned[utt] = 'Too short to get features'
    with open(output_path, 'w', encoding='utf8') as f:
        for k, v in unaligned.items():
            f.write('{} {}\n'.format(k, v))


def compile_information(model_directory, corpus, num_jobs, config):
    log_dir = os.path.join(model_directory, 'log')

    jobs = [(log_dir, corpus, x)
            for x in range(num_jobs)]

    run_non_mp(compile_information_func, jobs, log_dir)

    unaligned = {}
    for j in jobs:
        path = os.path.join(log_dir, 'unaligned.{}.log'.format(j[-1]))
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                utt, reason = line.split(' ', maxsplit=1)
                unaligned[utt] = reason
    return unaligned


def compute_alignment_improvement_func(iteration, config, model_directory, job_name):
    try:
        text_int_path = os.path.join(config.data_directory, 'text.{}.int'.format(job_name))
        log_path = os.path.join(model_directory, 'log', 'get_ctm.{}.{}.log'.format(iteration, job_name))
        ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
        model_path = os.path.join(model_directory, '{}.mdl'.format(iteration))
        phone_ctm_path = os.path.join(model_directory, 'phone.{}.{}.ctm'.format(iteration, job_name))
        if os.path.exists(phone_ctm_path):
            return

        frame_shift = config.feature_config.frame_shift / 1000
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
                                           os.path.join(config.dictionary.phones_dir, 'word_boundary.int'), model_path,
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
        mapping = config.dictionary.reversed_phone_mapping
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
                for p in config.dictionary.positions:
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
    jobs = [(iteration, config, model_directory, x) for x in range(num_jobs)]
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


def ali_to_textgrid_func(model_directory, word_path, split_directory, job_name, frame_shift):
    text_int_path = os.path.join(split_directory, 'text.{}.int'.format(job_name))
    log_path = os.path.join(model_directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, 'final.mdl')
    aligned_path = os.path.join(model_directory, 'aligned.{}'.format(job_name))
    nbest_path = os.path.join(model_directory, 'nbest.{}'.format(job_name))
    word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(job_name))
    if os.path.exists(word_ctm_path) and os.path.exists(phone_ctm_path):
        return

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
        det_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned'),
                                       'ark:-', 'ark:-'],
                                      stdin=lin_proc.stdout, stderr=log_file,
                                      stdout=subprocess.PIPE)
        align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                       word_path, model_path,
                                       'ark:-', 'ark,t:' + aligned_path],
                                      stdin=det_proc.stdout, stderr=log_file)
        align_proc.communicate()

        subprocess.call([thirdparty_binary('nbest-to-ctm'),
                         '--frame-shift={}'.format(frame_shift),
                         'ark:' + aligned_path,
                         word_ctm_path],
                        stderr=log_file)
        phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                       'ark:' + aligned_path, "ark:-"],
                                      stdout=subprocess.PIPE,
                                      stderr=log_file)
        nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                       '--frame-shift={}'.format(frame_shift),
                                       "ark:-", phone_ctm_path],
                                      stdin=phone_proc.stdout,
                                      stderr=log_file)
        nbest_proc.communicate()


def convert_ali_to_textgrids(align_config, output_directory, model_directory, dictionary, corpus, num_jobs, config):
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
    jobs = [(model_directory, word_path, corpus.split_directory(), x, frame_shift)
            for x in range(num_jobs)]
    if align_config.use_mp:
        run_mp(ali_to_textgrid_func, jobs, log_directory)
    else:
        run_non_mp(ali_to_textgrid_func, jobs, log_directory)

    if not corpus.segments: # Hack for better memory management for .lab files
        for i in range(num_jobs):
            word_ctm = {}
            phone_ctm = {}
            word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(i))
            phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(i))
            if not os.path.exists(word_ctm_path):
                continue
            parsed = parse_ctm(word_ctm_path, corpus, dictionary, mode='word')
            for k, v in parsed.items():
                if k not in word_ctm:
                    word_ctm[k] = v
                else:
                    word_ctm[k].update(v)
            parsed = parse_ctm(phone_ctm_path, corpus, dictionary, mode='phone')
            for k, v in parsed.items():
                if k not in phone_ctm:
                    phone_ctm[k] = v
                else:
                    phone_ctm[k].update(v)
            ctm_to_textgrid(word_ctm, phone_ctm, output_directory, corpus, dictionary, frame_shift=frame_shift)
    else:
        word_ctm = {}
        phone_ctm = {}
        for i in range(num_jobs):
            word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(i))
            phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(i))
            if not os.path.exists(word_ctm_path):
                continue
            parsed = parse_ctm(word_ctm_path, corpus, dictionary, mode='word')
            for k, v in parsed.items():
                if k not in word_ctm:
                    word_ctm[k] = v
                else:
                    word_ctm[k].update(v)
            parsed = parse_ctm(phone_ctm_path, corpus, dictionary, mode='phone')
            for k, v in parsed.items():
                if k not in phone_ctm:
                    phone_ctm[k] = v
                else:
                    phone_ctm[k].update(v)
        ctm_to_textgrid(word_ctm, phone_ctm, output_directory, corpus, dictionary, frame_shift=frame_shift)


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
        trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
        cmp_trans_path = os.path.join(directory, 'trans.cmp.{}'.format(job_name))
    else:
        tmp_trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
    post_path = os.path.join(directory, 'post.{}'.format(job_name))
    weight_path = os.path.join(directory, 'weight.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subprocess.call([thirdparty_binary('ali-to-post'),
                         "ark:" + ali_path, 'ark:' + post_path], stderr=log_file)

        subprocess.call([thirdparty_binary('weight-silence-post'), '0.0',
                         sil_phones, mdl_path, 'ark:' + post_path,
                         'ark:' + weight_path], stderr=log_file)

        subprocess.call([thirdparty_binary('gmm-est-fmllr'),
                         '--verbose=4',
                         '--fmllr-update-type={}'.format(config.fmllr_update_type),
                         '--spk2utt=ark:' + spk2utt_path, mdl_path, '{}'.format(feature_string),
                         'ark,s,cs:' + weight_path, 'ark:' + tmp_trans_path],
                        stderr=log_file)

        if not initial:
            subprocess.call([thirdparty_binary('compose-transforms'),
                             '--b-is-affine=true',
                             'ark:' + tmp_trans_path, 'ark:' + trans_path,
                             'ark:' + cmp_trans_path], stderr=log_file)
            os.remove(tmp_trans_path)
            os.remove(trans_path)
            os.rename(cmp_trans_path, trans_path)
        else:
            trans_path = tmp_trans_path


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

