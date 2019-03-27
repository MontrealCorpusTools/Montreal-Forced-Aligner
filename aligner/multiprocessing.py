import multiprocessing as mp
import subprocess
import os
import shutil
import re
import glob
from decimal import Decimal
import statistics

from .helper import make_path_safe, thirdparty_binary, filter_scp

from .textgrid import ctm_to_textgrid, parse_ctm

from .exceptions import CorpusError, AlignmentError


def init(env):
    os.environ = env


def acc_stats_func(directory, iteration, job_name, feat_path):  # pragma: no cover
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, job_name))
    model_path = os.path.join(directory, '{}.mdl'.format(iteration))
    next_model_path = os.path.join(directory, '{}.mdl'.format(iteration + 1))
    acc_path = os.path.join(directory, '{}.{}.acc'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as logf:
        acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), model_path,
                                     "scp:" + feat_path, "ark,t:" + ali_path, acc_path],
                                    stderr=logf)
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
    fmllr : bool, optional
        Whether the current training session is using fMLLR (speaker-adaptation),
        defaults to False

    """
    feat_name = config.feature_file_base_name

    feat_name += '.{}.scp'

    jobs = [(directory, iteration, x, os.path.join(split_directory, feat_name.format(x)))
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(acc_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]


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


def compile_train_graphs_func(directory, lang_directory, split_directory, job_name, debug=True):  # pragma: no cover
    fst_path = os.path.join(directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(directory, 'tree')
    mdl_path = os.path.join(directory, '0.mdl')
    if not os.path.exists(mdl_path):
        mdl_path = os.path.join(directory, 'final.mdl')

    log_path = os.path.join(directory, 'log', 'show_transition.log')
    transition_path = os.path.join(directory, 'transitions.txt')
    phones_file_path = os.path.join(lang_directory, 'phones.txt')

    triphones_file_path = os.path.join(directory, 'triphones.txt')
    if debug:
        with open(log_path, 'w', encoding='utf8') as logf:
            with open(transition_path, 'w', encoding='utf8') as f:
                subprocess.call([thirdparty_binary('show-transitions'), phones_file_path, mdl_path],
                                stdout=f, stderr=logf)
            parse_transitions(transition_path, triphones_file_path)
    log_path = os.path.join(directory, 'log', 'compile-graphs.0.{}.log'.format(job_name))

    if os.path.exists(triphones_file_path):
        phones_file_path = triphones_file_path
    words_file_path = os.path.join(lang_directory, 'words.txt')

    with open(os.path.join(split_directory, 'text.{}.int'.format(job_name)), 'r', encoding='utf8') as inf, \
            open(fst_path, 'wb') as outf, \
            open(log_path, 'w', encoding='utf8') as logf:
        proc = subprocess.Popen([thirdparty_binary('compile-train-graphs'),
                                 '--read-disambig-syms={}'.format(
                                     os.path.join(lang_directory, 'phones', 'disambig.int')),
                                 tree_path, mdl_path,
                                 os.path.join(lang_directory, 'L.fst'),
                                 "ark:-", "ark:-"],
                                stdin=inf, stdout=outf, stderr=logf)
        proc.communicate()

    if debug:
        utterances = []
        with open(os.path.join(split_directory, 'utt2spk.{}'.format(job_name)), 'r', encoding='utf8') as f:
            for line in f:
                utt = line.split()[0].strip()
                if not utt:
                    continue
                utterances.append(utt)

        with open(log_path, 'a', encoding='utf8') as logf:
            fst_ark_path = os.path.join(directory, 'fsts.{}.ark'.format(job_name))
            fst_scp_path = os.path.join(directory, 'fsts.{}.scp'.format(job_name))
            proc = subprocess.Popen([thirdparty_binary('fstcopy'),
                                     'ark:{}'.format(fst_path),
                                     'ark,scp:{},{}'.format(fst_ark_path, fst_scp_path)], stderr=logf)
            proc.communicate()

            temp_fst_path = os.path.join(directory, 'temp.fst.{}'.format(job_name))

            with open(fst_scp_path, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    utt = line.split()[0]

                    dot_path = os.path.join(directory, '{}.dot'.format(utt))
                    fst_proc = subprocess.Popen([thirdparty_binary('fstcopy'),
                                                 'scp:-',
                                                 'scp:echo {} {}|'.format(utt, temp_fst_path)],
                                                stdin=subprocess.PIPE, stderr=logf)
                    fst_proc.communicate(input=line.encode())

                    draw_proc = subprocess.Popen([thirdparty_binary('fstdraw'), '--portrait=true',
                                                  '--isymbols={}'.format(phones_file_path),
                                                  '--osymbols={}'.format(words_file_path), temp_fst_path, dot_path],
                                                 stderr=logf)
                    draw_proc.communicate()
                    try:
                        dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-O', dot_path], stderr=logf)
                        dot_proc.communicate()
                    except FileNotFoundError:
                        pass


def compile_train_graphs(directory, lang_directory, split_directory, num_jobs, debug=False):
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
    os.makedirs(os.path.join(directory, 'log'), exist_ok=True)
    jobs = [(directory, lang_directory, split_directory, x, debug)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(compile_train_graphs_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def mono_align_equal_func(mono_directory, split_directory, job_name, feat_path):  # pragma: no cover
    fst_path = os.path.join(mono_directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(mono_directory, 'tree')
    mdl_path = os.path.join(mono_directory, '0.mdl')
    directory = os.path.join(split_directory, str(job_name))
    log_path = os.path.join(mono_directory, 'log', 'align.0.{}.log'.format(job_name))
    ali_path = os.path.join(mono_directory, 'ali.{}'.format(job_name))
    acc_path = os.path.join(mono_directory, '0.{}.acc'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as logf:
        align_proc = subprocess.Popen([thirdparty_binary('align-equal-compiled'), "ark:" + fst_path,
                                       'scp:' + feat_path, 'ark:' + ali_path],
                                      stderr=logf)
        align_proc.communicate()
        stats_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), '--binary=true',
                                       mdl_path, 'scp:' + feat_path, 'ark:' + ali_path, acc_path],
                                      stdin=align_proc.stdout, stderr=logf)
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
    jobs = [(mono_directory, split_directory, x,
             os.path.join(split_directory, config.feature_file_base_name + '.{}.scp'.format(x)))
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(mono_align_equal_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def align_func(directory, iteration, job_name, mdl, config, feat_path, output_directory):  # pragma: no cover
    fst_path = os.path.join(directory, 'fsts.{}'.format(job_name))
    log_path = os.path.join(output_directory, 'log', 'align.{}.{}.log'.format(iteration, job_name))
    ali_path = os.path.join(output_directory, 'ali.{}'.format(job_name))
    score_path = os.path.join(output_directory, 'ali.{}.scores'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as logf:
        align_proc = subprocess.Popen([thirdparty_binary('gmm-align-compiled'),
                                       '--transition-scale={}'.format(config.transition_scale),
                                       '--acoustic-scale={}'.format(config.acoustic_scale),
                                       '--self-loop-scale={}'.format(config.self_loop_scale),
                                       '--beam={}'.format(config.beam),
                                       '--retry-beam={}'.format(config.retry_beam),
                                       '--careful=false',
                                       mdl,
                                       "ark:" + fst_path, "scp:" + feat_path, "ark,t:"+ali_path, "ark,t:"+score_path],
                                      stderr=logf)
        align_proc.communicate()


def align_no_pool(iteration, directory, split_directory, optional_silence, num_jobs, config, feature_name=None):
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('gmm-boost-silence'),
                                           config.boost_silence, optional_silence, make_path_safe(mdl_path))


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
    iteration : int
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
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('gmm-boost-silence'),
                                           config.boost_silence, optional_silence, make_path_safe(mdl_path))
    feat_name = config.feature_file_base_name
    feat_name += '.{}.scp'
    jobs = [(directory, iteration, x, mdl, config, os.path.join(split_directory, feat_name.format(x)), output_directory)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(align_func, args=i) for i in jobs]
        output = [p.get() for p in results]

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
    return unaligned


def compile_information(model_directory, corpus, num_jobs):
    log_dir = os.path.join(model_directory, 'log')

    jobs = [(log_dir, corpus, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(compile_information_func, args=i) for i in jobs]
        output = [p.get() for p in results]
    unaligned = {}
    for o in output:
        unaligned.update(o)
    return unaligned


def compute_alignment_improvement_func(iteration, config, model_directory, job_name):
    text_int_path = os.path.join(config.data_directory, 'text.{}.int'.format(job_name))
    log_path = os.path.join(model_directory, 'log', 'get_ctm.{}.{}.log'.format(iteration, job_name))
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, '{}.mdl'.format(iteration))
    phone_ctm_path = os.path.join(model_directory, 'phone.{}.{}.ctm'.format(iteration, job_name))
    if os.path.exists(phone_ctm_path):
        return

    frame_shift = config.feature_config.frame_shift / 1000
    with open(log_path, 'w', encoding='utf8') as logf:
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark:-'],
                                    stdout=subprocess.PIPE, stderr=logf)
        align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                       os.path.join(config.dictionary.phones_dir, 'word_boundary.int'), model_path,
                                       'ark:-', 'ark:-'],
                                      stdin=lin_proc.stdout, stderr=logf,
                                      stdout=subprocess.PIPE)
        phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                       'ark:-', "ark:-"],
                                      stdin=align_proc.stdout,
                                      stdout=subprocess.PIPE,
                                      stderr=logf)
        nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                       '--frame-shift={}'.format(frame_shift),
                                       "ark:-", phone_ctm_path],
                                      stdin=phone_proc.stdout,
                                      stderr=logf)
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
        time = Decimal('0.0')
        one_alignment = alignments_one[u]
        two_alignment = alignments_two[u]
        difference = 0
        while time < end:
            one_label = None
            two_label = None
            for b, e, l in one_alignment:
                if time < b:
                    continue
                if time >= e:
                    break
                one_label = l
            for b, e, l in two_alignment:
                if time < b:
                    continue
                if time >= e:
                    break
                two_label = l
            if one_label != two_label:
                difference += frame_shift
            time += frame_shift
        difference /= end
        differences.append(difference)
    if differences:
        mean_difference = statistics.mean(differences)
    else:
        mean_difference ='N/A'
    return utterances_aligned_diff, mean_difference


def compute_alignment_improvement(iteration, config, model_directory, num_jobs):
    jobs = [(iteration, config, model_directory, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        r = False
        try:
            results = [pool.apply_async(compute_alignment_improvement_func, args=i) for i in jobs]
            output = [p.get() for p in results]
        except OSError as e:
            if hasattr(e, 'errorno') and e.errorno == 24:
                r = True
            else:
                raise
    if r:
        raise (CorpusError(
            'There were too many files per speaker to process based on your OS settings. '
            'Please try to split your data into more speakers.'))
    alignment_diff_path = os.path.join(model_directory, 'train_change.csv')
    if iteration == 0 or iteration not in config.realignment_iterations:
        return
    ind = config.realignment_iterations.index(iteration)
    if ind != 0:
        previous_iteration = config.realignment_iterations[ind-1]
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


def ali_to_textgrid_func(align_config, model_directory, dictionary, corpus, job_name):  # pragma: no cover
    text_int_path = os.path.join(corpus.split_directory(), 'text.{}.int'.format(job_name))
    log_path = os.path.join(model_directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, 'final.mdl')
    aligned_path = os.path.join(model_directory, 'aligned.{}'.format(job_name))
    nbest_path = os.path.join(model_directory, 'nbest.{}'.format(job_name))
    word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(job_name))

    frame_shift = align_config.feature_config.frame_shift / 1000
    with open(log_path, 'w', encoding='utf8') as logf:
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark,t:'+nbest_path],
                                    stdout=subprocess.PIPE, stderr=logf)

        lin_proc.communicate()
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark:-'],
                                    stdout=subprocess.PIPE, stderr=logf)
        align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                       os.path.join(dictionary.phones_dir, 'word_boundary.int'), model_path,
                                       'ark:-', 'ark,t:' + aligned_path],
                                      stdin=lin_proc.stdout, stderr=logf)
        align_proc.communicate()

        subprocess.call([thirdparty_binary('nbest-to-ctm'),
                         '--frame-shift={}'.format(frame_shift),
                         'ark:' + aligned_path,
                         word_ctm_path],
                        stderr=logf)
        phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                       'ark:' + aligned_path, "ark:-"],
                                      stdout=subprocess.PIPE,
                                      stderr=logf)
        nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                       '--frame-shift={}'.format(frame_shift),
                                       "ark:-", phone_ctm_path],
                                      stdin=phone_proc.stdout,
                                      stderr=logf)
        nbest_proc.communicate()


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
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object that has information about pronunciations
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object that has information about the dataset
    num_jobs : int
        The number of processes to use in calculation

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)

    """
    jobs = [(align_config, model_directory, dictionary, corpus, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        r = False
        try:
            results = [pool.apply_async(ali_to_textgrid_func, args=i) for i in jobs]
            output = [p.get() for p in results]
        except OSError as e:
            if hasattr(e, 'errno') and e.errorno == 24:
                r = True
            else:
                raise
    if r:
        raise (CorpusError(
            'There were too many files per speaker to process based on your OS settings.  Please try to split your data into more speakers.'))
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
    ctm_to_textgrid(word_ctm, phone_ctm, output_directory, corpus, dictionary)


def tree_stats_func(directory, ci_phones, mdl, feat_path, ali_path, job_name):  # pragma: no cover
    context_opts = []
    log_path = os.path.join(directory, 'log', 'acc_tree.{}.log'.format(job_name))

    treeacc_path = os.path.join(directory, '{}.treeacc'.format(job_name))

    with open(log_path, 'w', encoding='utf8') as logf:
        subprocess.call([thirdparty_binary('acc-tree-stats')] + context_opts +
                        ['--ci-phones=' + ci_phones, mdl, "scp:" + feat_path,
                         "ark:" + ali_path,
                         treeacc_path], stderr=logf)


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
    fmllr : bool, optional
        Whether the current training session is using fMLLR (speaker-adaptation),
        defaults to False

    """
    feat_name = config.feature_file_base_name

    if '_fmllr' in feat_name:
        feat_name = feat_name.replace('_fmllr', '')

    feat_name += '.{}.scp'
    mdl_path = os.path.join(align_directory, 'final.mdl')

    jobs = [(directory, ci_phones, mdl_path,
             os.path.join(split_directory, feat_name.format(x)),
             os.path.join(align_directory, 'ali.{}'.format(x)), x)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(tree_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]

    tree_accs = [os.path.join(directory, '{}.treeacc'.format(x)) for x in range(num_jobs)]
    log_path = os.path.join(directory, 'log', 'sum_tree_acc.log')
    with open(log_path, 'w', encoding='utf8') as logf:
        subprocess.call([thirdparty_binary('sum-tree-stats'), os.path.join(directory, 'treeacc')] +
                        tree_accs, stderr=logf)
    # for f in tree_accs:
    #    os.remove(f)


def convert_alignments_func(directory, align_directory, job_name):  # pragma: no cover
    mdl_path = os.path.join(directory, '1.mdl')
    tree_path = os.path.join(directory, 'tree')
    ali_mdl_path = os.path.join(align_directory, 'final.mdl')
    ali_path = os.path.join(align_directory, 'ali.{}'.format(job_name))
    new_ali_path = os.path.join(directory, 'ali.{}'.format(job_name))

    log_path = os.path.join(directory, 'log', 'convert.{}.log'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as logf:
        subprocess.call([thirdparty_binary('convert-ali'), ali_mdl_path,
                         mdl_path, tree_path, "ark:" + ali_path,
                         "ark:" + new_ali_path], stderr=logf)


def convert_alignments(directory, align_directory, num_jobs):
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
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(convert_alignments_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def calc_fmllr_func(directory, split_directory, sil_phones, job_name, config, initial,
                    model_name='final'):  # pragma: no cover
    feat_scp = config.feature_config.feature_id + '.{}.scp'.format(job_name)
    base_scp = feat_scp.replace('_fmllr', '')
    if initial:
        feat_scp = base_scp
    feat_scp = os.path.join(split_directory, feat_scp)
    base_scp = os.path.join(split_directory, base_scp)

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
    with open(log_path, 'w', encoding='utf8') as logf:
        subprocess.call([thirdparty_binary('ali-to-post'),
                         "ark:" + ali_path, 'ark:' + post_path], stderr=logf)

        subprocess.call([thirdparty_binary('weight-silence-post'), '0.0',
                         sil_phones, mdl_path, 'ark:' + post_path,
                         'ark:' + weight_path], stderr=logf)

        subprocess.call([thirdparty_binary('gmm-est-fmllr'),
                         '--verbose=4',
                         '--fmllr-update-type={}'.format(config.fmllr_update_type),
                         '--spk2utt=ark:' + spk2utt_path, mdl_path, "scp:" + feat_scp,
                         'ark,s,cs:' + weight_path, 'ark:' + tmp_trans_path],
                        stderr=logf)

        if not initial:
            subprocess.call([thirdparty_binary('compose-transforms'),
                             '--b-is-affine=true',
                             'ark:' + tmp_trans_path, 'ark:' + trans_path,
                             'ark:' + cmp_trans_path], stderr=logf)
            os.remove(tmp_trans_path)
            os.remove(trans_path)
            os.rename(cmp_trans_path, trans_path)
        else:
            trans_path = tmp_trans_path
        utt2spk_path = os.path.join(config.corpus.split_directory(), 'utt2spk.{}'.format(job_name))
        base_scp = os.path.join(config.corpus.split_directory(), base_scp)
        feat_fmllr_scp_path = os.path.join(config.corpus.split_directory(),
                                           config.feature_config.feature_id + '.{}.scp'.format(job_name))
        feat_fmllr_ark_path = os.path.join(config.corpus.split_directory(),
                                           config.feature_config.feature_id + '.{}.ark'.format(job_name))
        subprocess.call([thirdparty_binary('transform-feats'),
                         '--utt2spk=ark:' + utt2spk_path,
                         'ark:' + trans_path, 'scp:' + base_scp,
                         'ark,scp:{},{}'.format(feat_fmllr_ark_path, feat_fmllr_scp_path)],
                        stderr=logf)


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
    iteration : int
        Specifies the current iteration, defaults to None

    """
    if iteration is None:
        if initial:
            model_name = '1'
        else:
            model_name = 'final'
    else:
        model_name = iteration
    jobs = [(directory, split_directory, sil_phones, x, config, initial, model_name)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(calc_fmllr_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def lda_acc_stats_func(directory, split_dir, align_directory, config, ci_phones, i):
    log_path = os.path.join(directory, 'log', 'ali_to_post.{}.log'.format(i))
    with open(log_path, 'w', encoding='utf8') as logf:
        spliced_feat_path = os.path.join(split_dir, config.feature_config.feature_id + '.{}.scp'.format(i))
        ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                             'ark:' + os.path.join(align_directory, 'ali.{}'.format(i)),
                                             'ark:-'],
                                            stderr=logf, stdout=subprocess.PIPE)
        weight_silence_post_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                     str(config.boost_silence), ci_phones,
                                                     os.path.join(align_directory, 'final.mdl'),
                                                     'ark:-', 'ark:-'],
                                                    stdin=ali_to_post_proc.stdout,
                                                    stderr=logf, stdout=subprocess.PIPE)
        acc_lda_post_proc = subprocess.Popen([thirdparty_binary('acc-lda'),
                                              '--rand-prune=' + str(config.random_prune),
                                              os.path.join(align_directory, 'final.mdl'),
                                              'scp:' + spliced_feat_path,
                                              'ark,s,cs:-',
                                              os.path.join(directory, 'lda.{}.acc'.format(i))],
                                             stdin=weight_silence_post_proc.stdout,
                                             stderr=logf)
        acc_lda_post_proc.communicate()


def lda_acc_stats(directory, split_dir, align_directory, config, ci_phones, num_jobs):
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
    jobs = [(directory, split_dir, align_directory, config, ci_phones, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(lda_acc_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]

    log_path = os.path.join(directory, 'log', 'lda_est.log')
    acc_list = []
    for x in range(num_jobs):
        acc_list.append(os.path.join(directory, 'lda.{}.acc'.format(x)))
    with open(log_path, 'w', encoding='utf8') as logf:
        est_lda_proc = subprocess.Popen([thirdparty_binary('est-lda'),
                                         '--write-full-matrix=' + os.path.join(directory, 'full.mat'),
                                         '--dim=' + str(config.lda_dimension),
                                         os.path.join(directory, 'lda.mat')] + acc_list,
                                        stderr=logf)
        est_lda_proc.communicate()
    shutil.copyfile(os.path.join(directory, 'lda.mat'), os.path.join(config.corpus.split_directory(), 'lda.mat'))
    config.feature_config.generate_features(config.corpus, overwrite=True)


def calc_lda_mllt_func(directory, split_directory, sil_phones, job_name, config,
                       initial,
                       model_name='final'):  # pragma: no cover
    log_path = os.path.join(directory, 'log', 'lda_mllt.{}.{}.log'.format(model_name, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    if not initial:
        mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    else:
        mdl_path = os.path.join(directory, '1.mdl')
        model_name = 1

    feat_path = os.path.join(split_directory, config.feature_config.feature_id + '.{}.scp'.format(job_name))
    post_path = os.path.join(directory, 'post.{}'.format(job_name))
    weight_path = os.path.join(directory, 'weight.{}'.format(job_name))

    # Estimating MLLT
    with open(log_path, 'a', encoding='utf8') as logf:
        subprocess.call([thirdparty_binary('ali-to-post'),
                         "ark:" + ali_path, 'ark:' + post_path], stderr=logf)

        subprocess.call([thirdparty_binary('weight-silence-post'), '0.0',
                         sil_phones, mdl_path, 'ark:' + post_path,
                         'ark:' + weight_path], stderr=logf)
        subprocess.call([thirdparty_binary('gmm-acc-mllt'),
                         '--rand-prune=' + str(config.random_prune),
                         mdl_path,
                         'scp:' + feat_path,
                         'ark:' + post_path,
                         os.path.join(directory, '{}.{}.macc'.format(model_name, job_name))],
                        stderr=logf)


def calc_lda_mllt(directory, split_directory, sil_phones, num_jobs, config, initial=False, iteration=None):
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
    split_directory : str
        Directory of training data split into the number of jobs
    fmllr_dir : str
        Directory of triphone FMLLR training
    sil_phones : str
        Colon-separated list of silence phones
    num_jobs : int
        The number of processes to use in calculation
    config : :class:`~aligner.config.LdaMlltConfig`
        Configuration object for training
    num_iters : int
        The number of iterations

    """
    if iteration is None:
        model_name = 'final'
    else:
        model_name = iteration
    jobs = [
        (directory, split_directory, sil_phones, x, config, initial, model_name)
        for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(calc_lda_mllt_func, args=i) for i in jobs]
        output = [p.get() for p in results]

    mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    log_path = os.path.join(directory, 'log', 'transform_means.{}.log'.format(model_name))
    previous_mat_path = os.path.join(directory, 'lda.mat')
    new_mat_path = os.path.join(directory, 'lda_new.mat')
    composed_path = os.path.join(directory, 'lda_composed.mat')
    with open(log_path, 'a', encoding='utf8') as logf:
        macc_list = []
        for x in range(num_jobs):
            macc_list.append(os.path.join(directory, '{}.{}.macc'.format(model_name, x)))
        subprocess.call([thirdparty_binary('est-mllt'),
                         new_mat_path]
                        + macc_list,
                        stderr=logf)
        subprocess.call([thirdparty_binary('gmm-transform-means'),
                         new_mat_path,
                         mdl_path, mdl_path],
                        stderr=logf)

        if os.path.exists(previous_mat_path):
            subprocess.call([thirdparty_binary('compose-transforms'),
                             new_mat_path,
                             previous_mat_path,
                             composed_path],
                            stderr=logf)
            os.remove(previous_mat_path)
            os.rename(composed_path, previous_mat_path)
        else:
            os.rename(new_mat_path, previous_mat_path)

    shutil.copyfile(os.path.join(directory, 'lda.mat'), os.path.join(config.corpus.split_directory(), 'lda.mat'))
    config.feature_config.generate_features(config.corpus,  overwrite=True)


def gmm_gselect_func(config, x):
    log_path = os.path.join(config.train_directory, 'log', 'gselect.{}.log'.format(x))
    feat_path = os.path.join(config.data_directory, config.feature_file_base_name + '.{}.scp'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config.subsample),
                                                 'scp:' + feat_path,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)

        gselect_proc = subprocess.Popen([thirdparty_binary('gmm-gselect'),
                                         '--n=' + str(config.num_gselect),
                                         os.path.join(config.train_directory, '1.dubm'),
                                         'ark:-',
                                         'ark:' + os.path.join(config.train_directory, 'gselect.{}'.format(x))],
                                        stdin=subsample_feats_proc.stdout,
                                        stderr=logf)
        gselect_proc.communicate()


def gmm_gselect(config, num_jobs):
    """
    Multiprocessing function that stores Gaussian selection indices on disk

    See:

    - http://kaldi-asr.org/doc/gmm-gselect_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_diag_ubm.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of Diagonal UBM training
    config : :class:`~aligner.config.DiagUbmConfig`
        Configuration object for training
    feats : str
        Path to LDA features
    num_jobs : int
        The number of processes to use in calculation

    """
    jobs = [(config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(gmm_gselect_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def acc_global_stats_func(config, x, iteration):
    log_path = os.path.join(config.train_directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    feat_path = os.path.join(config.data_directory, config.feature_file_base_name + '.{}.scp'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config.subsample),
                                                 'scp:' + feat_path,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
        gmm_global_acc_proc = subprocess.Popen([thirdparty_binary('gmm-global-acc-stats'),
                                                '--gselect=' + 'ark:' + os.path.join(config.train_directory,
                                                                                     'gselect.{}'.format(x)),
                                                os.path.join(config.train_directory, '{}.dubm'.format(iteration)),
                                                'ark:-',
                                                os.path.join(config.train_directory, '{}.{}.acc'.format(iteration, x))],
                                               stderr=logf,
                                               stdin=subsample_feats_proc.stdout)
        gmm_global_acc_proc.communicate()


def acc_global_stats(config, num_jobs, iteration):
    """
    Multiprocessing function that accumulates global GMM stats

    See:

    - http://kaldi-asr.org/doc/gmm-global-acc-stats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_diag_ubm.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of Diagonal UBM training
    config : :class:`~aligner.config.DiagUbmConfig`
        Configuration object for training
    feats : str
        Path to LDA features
    num_jobs : int
        The number of processes to use in calculation
    iteration : int
        Iteration to calculate stats for

    """
    jobs = [(config, x, iteration) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(acc_global_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def gauss_to_post_func(config, x):
    modified_posterior_scale = config.posterior_scale * config.subsample
    log_path = os.path.join(config.train_directory, 'log', 'post.{}.log'.format(x))
    feat_path = os.path.join(config.data_directory, config.feature_file_base_name + '.{}.scp'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config.subsample),
                                                 'scp:' + feat_path,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
        gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                     '--n=' + str(config.num_gselect),
                                                     '--min-post=' + str(config.min_post),
                                                     os.path.join(config.train_directory, 'final.dubm'),
                                                     'ark:-',
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stdin=subsample_feats_proc.stdout,
                                                    stderr=logf)
        scale_post_proc = subprocess.Popen([thirdparty_binary('scale-post'),
                                            'ark:-',
                                            str(modified_posterior_scale),
                                            'ark:' + os.path.join(config.train_directory, 'post.{}'.format(x))],
                                           stdin=gmm_global_get_post_proc.stdout,
                                           stderr=logf)
        scale_post_proc.communicate()


def gauss_to_post(config, num_jobs):
    """
    Multiprocessing function that does Gaussian selection and posterior extraction

    See:

    - http://kaldi-asr.org/doc/gmm-global-get-post_8cc.html
    - http://kaldi-asr.org/doc/scale-post_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of i-vector extractor training
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    diag_ubm_directory : str
        Directory of the diagonal UBM
    gmm_feats : str
        Path to features with online CMVN
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(gauss_to_post_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def acc_ivector_stats_func(config, x, iteration):
    log_path = os.path.join(config.train_directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    feat_path = os.path.join(config.data_directory, config.feature_config.feature_id + '.{}.scp'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config.subsample),
                                                 'scp:' + feat_path,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
        acc_stats_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-acc-stats'),
                                           os.path.join(config.train_directory, '{}.ie'.format(iteration)),
                                           'ark:-',
                                           'ark:' + os.path.join(config.train_directory, 'post.{}'.format(x)),
                                           os.path.join(config.train_directory, 'accinit.{}.{}'.format(iteration, x))],
                                          stdin=subsample_feats_proc.stdout,
                                          stderr=logf)
        acc_stats_proc.communicate()


def acc_ivector_stats(config, num_jobs, iteration):
    """
    Multiprocessing function that calculates i-vector extractor stats

    See:

    - http://kaldi-asr.org/doc/ivector-extractor-acc-stats_8cc.html
    - http://kaldi-asr.org/doc/ivector-extractor-sum-accs_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of i-vector extractor training
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    diag_ubm_directory : str
        Directory of the diagonal UBM
    feat_path : str
        Path to features without CMVN
    num_jobs : int
        The number of processes to use in calculation
    iteration : int
        Iteration to calculate stats for
    """
    jobs = [(config, x, iteration) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(acc_ivector_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]
    accinits = [os.path.join(config.train_directory, 'accinit.{}.{}'.format(iteration, j)) for j in range(num_jobs)]
    log_path = os.path.join(config.train_directory, 'log', 'sum_acc.{}.log'.format(iteration))
    with open(log_path, 'w', encoding='utf8') as logf:
        sum_accs_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-sum-accs'),
                                          '--parallel=true']
                                         + accinits
                                         + [os.path.join(config.train_directory, 'acc.{}'.format(iteration))],
                                         stderr=logf)

        sum_accs_proc.communicate()


def extract_ivectors_func(config, x):
    """

    Parameters
    ----------
    directory : str
        Directory of nnet training
    training_dir : str
        Directory of nnet training
    ieconf: str
        Path to ivector extractor configuration
    config : :class:`~aligner.trainers.IvectorExtractorTrainer`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """

    log_dir = os.path.join(config.align_directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    ivector_mdl = os.path.join(config.train_directory, 'final.ie')
    log_path = os.path.join(config.align_directory, 'log', 'extract_ivectors.{}.log'.format(x))
    feat_name = config.feature_file_base_name
    feat_name += '.{}.scp'
    # features_path = os.path.join(config.data_directory, 'features_for_ivector.{}.scp'.format(x))
    ivectors_path = os.path.join(config.train_directory, 'ivectors.{}'.format(x))
    post_path = os.path.join(config.align_directory, 'post.{}'.format(x))
    ali_path = os.path.join(config.align_directory, 'ali.{}'.format(x))
    weight_path = os.path.join(config.align_directory, 'weight.{}'.format(x))
    mdl_path = os.path.join(config.align_directory, 'final.mdl')
    gmm_feats = os.path.join(config.data_directory, feat_name.format(x))
    features_path = gmm_feats
    spk2utt_path = os.path.join(config.data_directory, 'spk2utt.{}'.format(x))
    sil_phones = config.dictionary.silence_csl

    silence_weight = 0.0
    posterior_scale = 0.1
    max_count = 100
    with open(log_path, 'w', encoding='utf8') as logf:
        ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                             'ark:' + ali_path, 'ark:-'],
                                            stderr=logf,
                                            stdout=subprocess.PIPE)
        weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                str(silence_weight),
                                                sil_phones,
                                                mdl_path,
                                                'ark:-', 'ark:-'],
                                               stderr=logf,
                                               stdin=ali_to_post_proc.stdout,
                                               stdout=subprocess.PIPE)
        post_to_weight_proc = subprocess.Popen([thirdparty_binary('post-to-weights'),
                                                'ark:-', 'ark:' + weight_path],
                                               stderr=logf,
                                               stdin=weight_silence_proc.stdout)
        post_to_weight_proc.communicate()

        gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                     '--n=' + str(config.num_gselect),
                                                     '--min-post=' + str(config.min_post),
                                                     os.path.join(config.train_directory, 'final.dubm'),
                                                     'scp:' + gmm_feats,
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=logf)
        extract_proc = subprocess.Popen([thirdparty_binary('ivector-extract'),
                                         '--acoustic-weight={}'.format(posterior_scale),
                                         '--compute-objf-change=true',
                                         '--max-count={}'.format(max_count),
                                         '--spk2utt=ark:' + spk2utt_path,
                                         ivector_mdl,
                                         'scp:' + features_path,
                                         'ark,s,cs:-',
                                         'ark,t:' + ivectors_path],
                                        stderr=logf,
                                        stdin=gmm_global_get_post_proc.stdout)
        extract_proc.communicate()
        utt_ivectors = []
        with open(ivectors_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                speaker = line[0]
                data = line[1:]
                for utt in config.corpus.speak_utt_mapping[speaker]:
                    utt_ivectors.append([utt] + data)

        with open(ivectors_path, 'w', newline='', encoding='utf8') as f:
            for u in utt_ivectors:
                f.write(' '.join(u))
                f.write('\n')

        feat_scp_path = os.path.join(config.data_directory, 'feats.{}.scp'.format(x))
        with open(os.devnull, 'w', encoding='utf8') as devnull:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'scp:' + feat_scp_path, '-'],
                                        stdout=subprocess.PIPE,
                                        stderr=devnull)
            stdout, stderr = dim_proc.communicate()
            feat_dim = int(stdout.decode('utf8').strip())

        ivector_ark_path = os.path.join(config.data_directory, 'ivector.{}.ark'.format(x))
        ivector_scp_path = os.path.join(config.data_directory, 'ivector.{}.scp'.format(x))
        append_proc = subprocess.Popen([thirdparty_binary('append-vector-to-feats'),
                                        'scp:' + feat_scp_path, 'ark:' + ivectors_path,
                                        'ark:-'],
                                       stderr=logf,
                                       stdout=subprocess.PIPE)
        select_proc = subprocess.Popen([thirdparty_binary('select-feats'),
                                        "{}-{}".format(feat_dim, feat_dim + config.ivector_dimension - 1),
                                        'ark:-',
                                        'ark,scp:{},{}'.format(ivector_ark_path, ivector_scp_path)],
                                       stderr=logf,
                                       stdin=append_proc.stdout)
        select_proc.communicate()


def extract_ivectors(config, num_jobs):
    """
    Multiprocessing function that extracts i-vectors.

    See:

    - http://kaldi-asr.org/doc/ivector-extract-online2_8cc.html
    - http://kaldi-asr.org/doc/copy-feats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/extract_ivectors_online.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    directory : str
        Directory of nnet training
    training_dir : str
        Directory of nnet training
    ieconf: str
        Path to ivector extractor configuration
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(extract_ivectors_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def get_egs_func(config, align_directory, valid_uttlist,
                 train_subset_uttlist,  x):
    # Create training examples
    ivectors_opt = []
    if config.feature_config.ivectors:
        ivector_dim = 100  # Make safe later
        ivectors_opt.append('--const-feat-dim={}'.format(ivector_dim))


    # Deal with ivector stuff
    log_path = os.path.join(config.train_directory, 'log', 'get_egs_feats.{}.log'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        # Gets "feats" (Kaldi)

        if not valid_uttlist:
            valid_uttlist = train_subset_uttlist
        base_feature_scp = os.path.join(config.data_directory, config.feature_file_base_name + '.{}.scp'.format(x))
        filtered = filter_scp(train_subset_uttlist,
                              os.path.join(config.data_directory, base_feature_scp))
        training_features_path = os.path.join(config.train_directory, 'features_for_nnet_train.{}.scp'.format(x))
        with open(training_features_path, 'w', encoding='utf8') as outf:
            for item in filtered:
                outf.write(item)

        # Gets "valid_feats" (Kaldi)
        filtered = filter_scp(valid_uttlist, os.path.join(config.train_directory, base_feature_scp))
        validation_features_path = os.path.join(config.train_directory, 'features_for_nnet_valid.{}.scp'.format(x))
        with open(validation_features_path, 'w', encoding='utf8') as outf:
            for item in filtered:
                outf.write(item)

    log_path = os.path.join(config.train_directory, 'log', 'ali_to_post.{}.log'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        ali_to_pdf_proc = subprocess.Popen([thirdparty_binary('ali-to-pdf'),
                                            os.path.join(align_directory, 'final.mdl'),
                                            'ark:' + os.path.join(align_directory, 'ali.{}'.format(x)),
                                            'ark:-'],
                                           stdout=subprocess.PIPE,
                                           stderr=logf)
        ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                             'ark:-', 'ark:-'],
                                            stdin=ali_to_pdf_proc.stdout,
                                            stderr=logf,
                                            stdout=subprocess.PIPE)

    log_path = os.path.join(config.train_directory, 'log', 'get_egs.{}.log'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        nnet_get_egs_proc = subprocess.Popen([thirdparty_binary('nnet-get-egs')]+ ivectors_opt +
                                              ['--left-context=' + str(config.feature_config.splice_left_context),
                                              '--right-context=' + str(config.feature_config.splice_right_context),
                                              'scp:' + training_features_path,
                                              'ark:-',
                                              'ark:-'],
                                             stdin=ali_to_post_proc.stdout,
                                             stdout=subprocess.PIPE,
                                             stderr=logf)
        nnet_copy_egs_proc = subprocess.Popen([thirdparty_binary('nnet-copy-egs'),
                                               'ark:-',
                                               'ark:' + os.path.join(config.egs_directory, 'egs_orig.{}'.format(x))],
                                              stdin=nnet_get_egs_proc.stdout,
                                              stderr=logf)
        nnet_copy_egs_proc.communicate()

    # Rearranging training examples
    log_path = os.path.join(config.train_directory, 'log', 'nnet_shuffle_egs.{}.log'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf:
        nnet_copy_egs_proc = subprocess.Popen([thirdparty_binary('nnet-copy-egs'),
                                               '--srand=' + str(x),
                                               'ark:' + os.path.join(config.egs_directory, 'egs_orig.{}'.format(x)),
                                               'ark:-'],
                                              stderr=logf,
                                              stdout=subprocess.PIPE)
        nnet_shuffle_egs_proc = subprocess.Popen([thirdparty_binary('nnet-shuffle-egs'),
                                                  '--srand=' + str(x),
                                                  'ark:-',
                                                  'ark:' + os.path.join(config.egs_directory, 'egs.{}'.format(x))],
                                                 stderr=logf,
                                                 stdin=nnet_copy_egs_proc.stdout)
        nnet_shuffle_egs_proc.communicate()


def get_egs(config, ali_dir, valid_uttlist,
            train_subset_uttlist):
    """
    Multiprocessing function that gets training examples for the neural net

    See:

    - http://kaldi-asr.org/doc/ali-to-pdf_8cc.html
    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/nnet-get-egs_8cc.html
    - http://kaldi-asr.org/doc/nnet-copy-egs_8cc.html
    - http://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/get_egs.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    nnet_dir : str
        Directory of nnet training
    egs_dir : str
        Directory where examples will go
    training_dir : str
        Directory of combined "split" files (e.g. utt2spk, cmvn.scp)
    split_dir : str
        Directory of training data split into the number of jobs
    ali_dir : str
        Directory of previous alignment
    ivector_dir : str
        Directory of the extracted i-vectors
    feats : str
        Path to features
    valid_uttlist : str
        Path to list of valid utterances
    train_subset_uttlist : str
        Path to subset list of training utterances
    config : :class:`~aligner.config.NnetBasicConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """

    jobs = [(config, ali_dir, valid_uttlist,
             train_subset_uttlist,  x) for x in range(config.corpus.num_jobs)]
    with mp.Pool(processes=config.corpus.num_jobs) as pool:
        results = [pool.apply_async(get_egs_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def get_lda_nnet_func(config, align_directory, x):
    sil_phones = config.dictionary.optional_silence_csl
    log_path = os.path.join(config.train_directory, 'log', 'lda_acc.{}.log'.format(x))

    base_feature_scp = os.path.join(config.data_directory, config.feature_file_base_name.replace('_ivectors', '') + '.{}.scp'.format(x))
    spliced_feature_ark = os.path.join(config.data_directory, 'features_for_nnet_lda.{}.ark'.format(x))
    spliced_feature_scp = os.path.join(config.data_directory, 'features_for_nnet_lda.{}.scp'.format(x))
    ivector_scp_path = os.path.join(config.data_directory, 'ivector.{}.scp'.format(x))
    ivector_period = 10
    with open(log_path, 'w', encoding='utf8') as logf:
        if config.feature_config.ivectors:
            splice_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                        '--left-context={}'.format(config.feature_config.splice_left_context),
                                        '--right-context={}'.format(config.feature_config.splice_right_context),
                                        'scp:' + base_feature_scp,
                                        'ark:-'],
                                       stdout=subprocess.PIPE,
                                       stderr=logf)
            paste_proc = subprocess.Popen([thirdparty_binary('paste-feats'),
                                           'ark:-',
                                           'scp:' + ivector_scp_path,
                                           'ark,scp:{},{}'.format(spliced_feature_ark, spliced_feature_scp)],
                                          stdin=splice_proc.stdout,
                                          stderr=logf)
            paste_proc.communicate()
            # Get i-vector dimension
            ivector_dim_path = os.path.join(config.train_directory, 'ivector_dim')
            with open(ivector_dim_path, 'w', encoding='utf8') as outf:
                dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                             'scp:' + ivector_scp_path,
                                             '-'],
                                            stderr=logf,
                                            stdout=outf)
        else:
            splice_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                        '--left-context={}'.format(config.feature_config.splice_left_context),
                                        '--right-context={}'.format(config.feature_config.splice_right_context),
                                        'scp:' + base_feature_scp,
                                           'ark,scp:{},{}'.format(spliced_feature_ark, spliced_feature_scp)],
                                       stderr=logf)
            splice_proc.communicate()

        feat_dim_path = os.path.join(config.train_directory, 'feat_dim')
        with open(feat_dim_path, 'w', encoding='utf8') as outf:
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'scp:' + base_feature_scp,
                                         '-'],
                                        stderr=logf,
                                        stdout=outf)

            ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                 'ark:' + os.path.join(align_directory, 'ali.{}'.format(x)),
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
            weight_silence_post_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                         '0.0',
                                                         sil_phones,
                                                         os.path.join(align_directory, 'final.mdl'),
                                                         'ark:-',
                                                         'ark:-'],
                                                        stdin=ali_to_post_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
            acc_lda_proc = subprocess.Popen([thirdparty_binary('acc-lda'),
                                             '--rand-prune={}'.format(config.lda_random_prune),
                                             os.path.join(align_directory, 'final.mdl'),
                                             'scp:' + spliced_feature_scp,
                                             'ark,s,cs:-',
                                             os.path.join(config.train_directory, 'lda.{}.acc'.format(x))],
                                            stdin=weight_silence_post_proc.stdout,
                                            stderr=logf)
            acc_lda_proc.communicate()


def get_lda_nnet(config, align_directory, num_jobs):
    """
    Multiprocessing function that extracts training examples and does LDA transformation

    See:

    - http://kaldi-asr.org/doc/apply-cmvn_8cc.html
    - http://kaldi-asr.org/doc/splice-feats_8cc.html
    - http://kaldi-asr.org/doc/subsample-feats_8cc.html
    - http://kaldi-asr.org/doc/ivector-randomize_8cc.html
    - http://kaldi-asr.org/doc/paste-feats_8cc.html
    - http://kaldi-asr.org/doc/ali-to-post_8cc.html
    - http://kaldi-asr.org/doc/weight-silence-post_8cc.html
    - http://kaldi-asr.org/doc/acc-lda_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/get_lda.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    nnet_dir : str
        Directory of nnet training
    ali_dir : str
        Directory of previous alignment
    ivector_dir : str
        Directory of the extracted i-vectors
    training_dir : str
        Directory of combined "split" files (e.g. utt2spk, cmvn.scp)
    split_dir : str
        Directory of training data split into the number of jobs
    feats : str
        Path to features
    sil_phones : str
        Colon-separated list of silence phones
    config : :class:`~aligner.config.NnetBasicConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """

    jobs = [(config, align_directory, x) for x in
            range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(get_lda_nnet_func, args=i) for i in jobs]
        output = [p.get() for p in results]
    log_path = os.path.join(config.train_directory, 'log', 'lda_matrix.log')
    with open(log_path, 'w', encoding='utf8') as logf:
        acc_files = [os.path.join(config.train_directory, 'lda.{}.acc'.format(x))
                     for x in range(config.corpus.num_jobs)]
        sum_lda_accs_proc = subprocess.Popen([thirdparty_binary('sum-lda-accs'),
                                              os.path.join(config.train_directory, 'lda.acc')]
                                             + acc_files,
                                             stderr=logf)
        sum_lda_accs_proc.communicate()

        lda_mat_proc = subprocess.Popen([thirdparty_binary('nnet-get-feature-transform'),
                                         '--dim={}'.format(config.lda_dimension),
                                         os.path.join(config.train_directory, 'nnet_lda.mat'),
                                         os.path.join(config.train_directory, 'lda.acc')],
                                        stderr=logf)
        lda_mat_proc.communicate()


def nnet_train_trans(nnet_dir, align_dir, prev_ali_path, num_jobs):
    """
    Multiprocessing function that trains transition prbabilities and sets priors.

    See:

    - http://kaldi-asr.org/doc/nnet-train-transitions_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/train_pnorm_multisplice.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    nnet_dir : str
        Directory of nnet training
    align_dir : str
        Directory of previous alignment
    prev_ali_path : str
        Path to previous alignment
    num_jobs : int
        The number of processes to use in calculation
    """
    log_path = os.path.join(nnet_dir, 'log', 'train_trans.log')
    with open(log_path, 'w', encoding='utf8') as logf:
        train_trans_proc = subprocess.Popen([thirdparty_binary('nnet-train-transitions'),
                                             os.path.join(nnet_dir, '0.mdl'),
                                             'ark:' + prev_ali_path,
                                             os.path.join(nnet_dir, '0.mdl')],
                                            stderr=logf)
        train_trans_proc.communicate()


def nnet_train_func(nnet_dir, egs_dir, mdl, i, x):
    log_path = os.path.join(nnet_dir, 'log', 'train.{}.{}.log'.format(i, x))
    with open(log_path, 'w', encoding='utf8') as logf:
        shuffle_proc = subprocess.Popen([thirdparty_binary('nnet-shuffle-egs'),
                                         '--srand={}'.format(i),
                                         'ark:' + os.path.join(egs_dir, 'egs.{}'.format(x)),
                                         'ark:-'],
                                        stdout=subprocess.PIPE,
                                        stderr=logf)
        train_proc = subprocess.Popen([thirdparty_binary('nnet-train-parallel'),
                                       # Leave off threads and minibatch params for now
                                       '--srand={}'.format(i),
                                       mdl,
                                       'ark:-',
                                       os.path.join(nnet_dir, '{}.{}.mdl'.format((i + 1), x))],
                                      stdin=shuffle_proc.stdout,
                                      stderr=logf)
        train_proc.communicate()


def nnet_train(nnet_dir, egs_dir, mdl, i, num_jobs):
    """
    Multiprocessing function that trains the neural net.

    See:

    - http://kaldi-asr.org/doc/nnet-shuffle-egs_8cc.html
    - http://kaldi-asr.org/doc/nnet-train-parallel_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/train_pnorm_multisplice.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    nnet_dir : str
        Directory of nnet training
    egs_dir : str
        Directory for training examples
    mdl : str
        Path to current model
    i : int
        Number of current iteration
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(nnet_dir, egs_dir, mdl, i, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(nnet_train_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def nnet_align_func(i, config, train_directory, align_directory, x):
    feat_name = config.feature_file_base_name + '.{}.scp'
    feat_path = os.path.join(config.data_directory, feat_name.format(x))
    fst_path = os.path.join(train_directory, 'fsts.{}'.format(x))
    log_path = os.path.join(align_directory, 'log', 'align.{}.{}.log'.format(i, x))
    ali_path = os.path.join(align_directory, 'ali.{}'.format(x))
    mdl_path = os.path.join(train_directory, '{}.mdl'.format(i))
    ivector_scp_path = os.path.join(config.data_directory, 'ivector.{}.scp'.format(x))
    with open(log_path, 'w', encoding='utf8') as logf, \
            open(ali_path, 'wb') as outf:
        paste_proc = subprocess.Popen([thirdparty_binary('paste-feats'),
                                       'scp:' + feat_path,
                                       'scp:' + ivector_scp_path,
                                       'ark:-'],
                                      stdout=subprocess.PIPE,
                                      stderr=logf)
        align_proc = subprocess.Popen([thirdparty_binary('nnet-align-compiled'),
                                       '--beam={}'.format(config.beam),
                                       '--retry-beam={}'.format(config.retry_beam),
                                       mdl_path,
                                       "ark:" + fst_path, "ark:-", "ark:-"],
                                      stdin=paste_proc.stdout,
                                      stderr=logf,
                                      stdout=outf)
        align_proc.communicate()


def nnet_align(i, config, train_directory, align_directory, num_jobs):
    """
    Multiprocessing function that generates an nnet alignment

    See:

    - http://kaldi-asr.org/doc/nnet-align-compiled_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/align.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    i : int
        Number of the current iteration
    nnet_dir : str
        Directory of nnet training
    optional_silence : str
        Colon-separated list of silence phones to boost
    num_jobs : int
        The number of processes to use in calculation
    """

    jobs = [(i, config, train_directory, align_directory, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(nnet_align_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def compute_prob_func(i, nnet_dir, egs_dir, model_path, x):
    log_path = os.path.join(nnet_dir, 'log', 'compute_prob_train.{}.{}.log'.format(i, x))
    with open(log_path, 'w', encoding='utf8') as logf:
        compute_prob_proc = subprocess.Popen([thirdparty_binary('nnet-compute-prob'),
                                              model_path,
                                              'ark:{}/egs.{}'.format(egs_dir, x)],
                                             stderr=logf)
        compute_prob_proc.communicate()


def compute_prob(i, nnet_dir, egs_dir, model_path, num_jobs):
    """
    Multiprocessing function that computes the current log probability of the iteration

    See:

    - http://kaldi-asr.org/doc/nnet-compute-prob_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/train_pnorm_multisplice2.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    i : int
        Number of the current iteration
    nnet_dir : str
        Directory of nnet training
    egs_dir : str
        Directory for training examples
    model_path : str
        Path to the current iteration's model
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(i, nnet_dir, egs_dir, model_path, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(compute_prob_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def ali_to_textgrid_kaldi_func(ali_directory, model_directory, lang_dir, split_data_dir, sym2int_script, oov,
                               job_name):  # pragma: no cover
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, 'final.mdl')
    aligned_path = os.path.join(model_directory, 'aligned.{}'.format(job_name))
    word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(job_name))
    phones_dir = os.path.join(lang_dir, 'phones')

    # Get integers
    log_path = os.path.join(ali_directory, 'log', 'sym2int.{}.log'.format(job_name))
    text_int_path = os.path.join(ali_directory, 'text.{}.int'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as logf, open(text_int_path, 'w', encoding='utf8') as outf:
        sym2int_proc = subprocess.Popen([sym2int_script,
                                         '--map-oov', oov, '-f', '2-',
                                         os.path.join(lang_dir, 'words.txt'),
                                         os.path.join(split_data_dir, str(job_name + 1), 'text')],
                                        stdout=outf, stderr=logf)
        sym2int_proc.communicate()

    frame_shift = 10 / 1000
    log_path = os.path.join(ali_directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as logf:
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark:-'],
                                    stdout=subprocess.PIPE, stderr=logf)
        align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                       os.path.join(phones_dir, 'word_boundary.int'), model_path,
                                       'ark:-', 'ark:' + aligned_path],
                                      stdin=lin_proc.stdout, stderr=logf)
        align_proc.communicate()

        subprocess.call([thirdparty_binary('nbest-to-ctm'),
                         '--frame-shift={}'.format(frame_shift),
                         'ark:' + aligned_path,
                         word_ctm_path],
                        stderr=logf)
        phone_proc = subprocess.Popen([thirdparty_binary('lattice-to-phone-lattice'), model_path,
                                       'ark:' + aligned_path, "ark:-"],
                                      stdout=subprocess.PIPE,
                                      stderr=logf)
        nbest_proc = subprocess.Popen([thirdparty_binary('nbest-to-ctm'),
                                       '--frame-shift={}'.format(frame_shift),
                                       "ark:-", phone_ctm_path],
                                      stdin=phone_proc.stdout,
                                      stderr=logf)
        nbest_proc.communicate()


def convert_ali_to_textgrids_kaldi(ali_directory, model_directory, lang_dir, split_data_dir, sym2int_script, oov,
                                   num_jobs):
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
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object that has information about pronunciations
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object that has information about the dataset
    num_jobs : int
        The number of processes to use in calculation

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)

    """
    jobs = [(ali_directory, model_directory, lang_dir, split_data_dir, sym2int_script, oov, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        r = False
        try:
            results = [pool.apply_async(ali_to_textgrid_kaldi_func, args=i) for i in jobs]
            output = [p.get() for p in results]
        except OSError as e:
            if hasattr(e, 'errno') and e.errorno == 24:
                r = True
            else:
                raise
    if r:
        raise (CorpusError(
            'There were too many files per speaker to process based on your OS settings.  Please try to split your data into more speakers.'))
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
    ctm_to_textgrid(word_ctm, phone_ctm, output_directory, corpus, dictionary)


def get_average_posteriors_func(i, nnet_dir, prev_egs_dir, config, x):
    log_path = os.path.join(nnet_dir, 'log', 'get_post.{}.{}.log'.format(i, x))
    with open(log_path, 'w', encoding='utf8') as logf:
        nnet_to_raw_nnet_proc = subprocess.Popen([thirdparty_binary('nnet-to-raw-nnet'),
                                                  os.path.join(nnet_dir, '{}.mdl'.format(i)),
                                                  os.path.join(nnet_dir, '{}_raw.mdl'.format(i))],
                                                 stderr=logf)
        nnet_to_raw_nnet_proc.communicate()
        nnet_subset_egs_proc = subprocess.Popen([thirdparty_binary('nnet-subset-egs'),
                                                 '--n={}'.format(config.prior_subset_size),
                                                 'ark:' + os.path.join(prev_egs_dir, 'egs.{}'.format(x)),
                                                 'ark:-'],
                                                stdout=subprocess.PIPE, stderr=logf)
        nnet_compute_from_egs_proc = subprocess.Popen([thirdparty_binary('nnet-compute-from-egs'),
                                                       os.path.join(nnet_dir, '{}_raw.mdl'.format(i)),
                                                       'ark:-', 'ark:-'],
                                                      stdin=nnet_subset_egs_proc.stdout,
                                                      stdout=subprocess.PIPE,
                                                      stderr=logf)
        matrix_sum_rows_proc = subprocess.Popen([thirdparty_binary('matrix-sum-rows'),
                                                 'ark:-', 'ark:-'],
                                                stdin=nnet_compute_from_egs_proc.stdout,
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
        vector_sum_proc = subprocess.Popen([thirdparty_binary('vector-sum'),
                                            'ark:-',
                                            os.path.join(nnet_dir, 'post.{}.{}.vec'.format(i, x))],
                                           stdin=matrix_sum_rows_proc.stdout,
                                           stderr=logf)
        vector_sum_proc.communicate()


def get_average_posteriors(i, nnet_dir, prev_egs_dir, config, num_jobs):
    """
    Multiprocessing function that gets average posterior for purposes of computing priors (for nnet)

    See:

    - http://kaldi-asr.org/doc/nnet-to-raw-nnet_8cc.html
    - http://kaldi-asr.org/doc/nnet-subset-egs_8cc.html
    - http://kaldi-asr.org/doc/nnet-compute-from-egs_8cc.html
    - http://kaldi-asr.org/doc/matrix-sum-rows_8cc.html
    - http://kaldi-asr.org/doc/vector-sum_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/train_pnorm_multisplice2.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    i : int
        Number of the current iteration
    nnet_dir : str
        Directory of nnet training
    prev_egs_dir : str
        Directory for training examples from the last iteration
    config : :class:`~aligner.config.NnetBasicConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(i, nnet_dir, prev_egs_dir, config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(get_average_posteriors_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def relabel_egs_func(i, nnet_dir, egs_in, alignments, egs_out, x):
    log_path = os.path.join(nnet_dir, 'log', 'relabel_egs.{}.log'.format(x))
    model_path = os.path.join(nnet_dir, '{}.mdl'.format(i))
    with open(log_path, 'w', encoding='utf8') as logf:
        ali_to_pdf_proc = subprocess.Popen([thirdparty_binary('ali-to-pdf'),
                                            model_path,
                                            'ark:' + alignments, 'ark:-'],
                                           stdout=subprocess.PIPE,
                                           # stdin=alignments,
                                           stderr=logf)
        nnet_relabel_egs_proc = subprocess.Popen([thirdparty_binary('nnet-relabel-egs'),
                                                  'ark:-',
                                                  'ark:' + os.path.join(egs_in, 'egs.{}'.format(x)),
                                                  'ark:' + os.path.join(egs_out, 'egs.{}'.format(x))],
                                                 stdin=ali_to_pdf_proc.stdout,
                                                 stderr=logf)
        nnet_relabel_egs_proc.communicate()


def relabel_egs(i, nnet_dir, egs_in, alignments, egs_out, num_jobs):
    """
    Multiprocessing function that relabels training examples
    
    See:

    - http://kaldi-asr.org/doc/ali-to-pdf_8cc.html
    - http://kaldi-asr.org/doc/nnet-relabel-egs_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/nnet2/relabel_egs2.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    i : int
        Number of the current iteration
    nnet_dir : str
        Directory of nnet training
    egs_in : str
        Path to examples to be relabeled
    egs_out : str
        Path to where relabeled examples will go
    num_jobs : int
        The number of processes to use in calculation
    """
    jobs = [(i, nnet_dir, egs_in, alignments, egs_out, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs, initializer=init, initargs=(os.environ.copy(),)) as pool:
        results = [pool.apply_async(relabel_egs_func, args=i) for i in jobs]
        output = [p.get() for p in results]
