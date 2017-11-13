import multiprocessing as mp
import subprocess
import os
import shutil
import re

from .helper import make_path_safe, thirdparty_binary

from .textgrid import ctm_to_textgrid, parse_ctm

from .config import *

from .exceptions import CorpusError


def mfcc_func(mfcc_directory, log_directory, job_name, mfcc_config_path):  # pragma: no cover
    raw_mfcc_path = os.path.join(mfcc_directory, 'raw_mfcc.{}.ark'.format(job_name))
    raw_scp_path = os.path.join(mfcc_directory, 'raw_mfcc.{}.scp'.format(job_name))
    log_path = os.path.join(log_directory, 'make_mfcc.{}.log'.format(job_name))
    segment_path = os.path.join(log_directory, 'segments.{}'.format(job_name))
    scp_path = os.path.join(log_directory, 'wav.{}.scp'.format(job_name))

    with open(log_path, 'w') as f:
        if os.path.exists(segment_path):
            seg_proc = subprocess.Popen([thirdparty_binary('extract-segments'),
                                         'scp,p:' + scp_path, segment_path, 'ark:-'],
                                        stdout=subprocess.PIPE, stderr=f)
            comp_proc = subprocess.Popen([thirdparty_binary('compute-mfcc-feats'), '--verbose=2',
                                          '--config=' + mfcc_config_path,
                                          'ark:-', 'ark:-'],
                                         stdout=subprocess.PIPE, stderr=f, stdin=seg_proc.stdout)
        else:

            comp_proc = subprocess.Popen([thirdparty_binary('compute-mfcc-feats'), '--verbose=2',
                                          '--config=' + mfcc_config_path,
                                          'scp,p:' + scp_path, 'ark:-'],
                                         stdout=subprocess.PIPE, stderr=f)
        copy_proc = subprocess.Popen([thirdparty_binary('copy-feats'),
                                      '--compress=true', 'ark:-',
                                      'ark,scp:{},{}'.format(raw_mfcc_path, raw_scp_path)],
                                     stdin=comp_proc.stdout, stderr=f)
        copy_proc.wait()


def mfcc(mfcc_directory, log_directory, num_jobs, mfcc_configs):
    """
    Multiprocessing function that converts wav files into MFCCs

    See http://kaldi-asr.org/doc/feat.html and
    http://kaldi-asr.org/doc/compute-mfcc-feats_8cc.html for more details on how
    MFCCs are computed.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/make_mfcc.sh
    for the bash script this function was based on.

    Parameters
    ----------
    mfcc_directory : str
        Directory to save MFCC feature matrices
    log_directory : str
        Directory to store log files
    num_jobs : int
        The number of processes to use in calculation
    mfcc_configs : list of :class:`~aligner.config.MfccConfig`
        Configuration object for generating MFCCs

    Raises
    ------
    CorpusError
        If the files per speaker exceeds the number of files that are
        allowed to be open on the computer (for Unix-based systems)
    """
    jobs = [(mfcc_directory, log_directory, x, mfcc_configs[x].path)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        r = False
        try:
            results = [pool.apply_async(mfcc_func, args=i) for i in jobs]
            output = [p.get() for p in results]
        except OSError as e:
            if e.errorno == 24:
                r = True
            else:
                raise
    if r:
        raise (CorpusError(
            'There were too many files per speaker to process based on your OS settings.  Please try to split your data into more speakers.'))


def acc_stats_func(directory, iteration, job_name, feat_path):  # pragma: no cover
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, job_name))
    model_path = os.path.join(directory, '{}.mdl'.format(iteration))
    next_model_path = os.path.join(directory, '{}.mdl'.format(iteration + 1))
    acc_path = os.path.join(directory, '{}.{}.acc'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        acc_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), model_path,
                                     "ark:" + feat_path, "ark,t:" + ali_path, acc_path],
                                    stderr=logf)
        acc_proc.communicate()


def acc_stats(iteration, directory, split_directory, num_jobs, fmllr=False, do_lda_mllt=None, feature_name=None):
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
    if feature_name == None:
        feat_name = 'cmvndeltafeats'
    else:
        feat_name = feature_name
    if fmllr:
        feat_name += '_fmllr'

    if do_lda_mllt == True:
        feat_name = 'cmvnsplicetransformfeats'

    #if lda_mllt:
    #    feat_name = 'cmvnsplicetransformfeats'
        #feat_name += '_lda_mllt'

    feat_name += '.{}'

    jobs = [(directory, iteration, x, os.path.join(split_directory, feat_name.format(x)))
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
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


def compile_train_graphs_func(directory, lang_directory, split_directory, job_name, debug=False):  # pragma: no cover
    fst_path = os.path.join(directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(directory, 'tree')
    mdl_path = os.path.join(directory, '0.mdl')

    log_path = os.path.join(directory, 'log', 'show_transition.log')
    transition_path = os.path.join(directory, 'transitions.txt')
    phones_file_path = os.path.join(lang_directory, 'phones.txt')

    triphones_file_path = os.path.join(directory, 'triphones.txt')
    if debug:
        with open(log_path, 'w') as logf:
            with open(transition_path, 'w', encoding='utf8') as f:
                subprocess.call([thirdparty_binary('show-transitions'), phones_file_path, mdl_path],
                                stdout=f, stderr=logf)
            parse_transitions(transition_path, triphones_file_path)
    log_path = os.path.join(directory, 'log', 'compile-graphs.0.{}.log'.format(job_name))

    if os.path.exists(triphones_file_path):
        phones_file_path = triphones_file_path
    words_file_path = os.path.join(lang_directory, 'words.txt')

    with open(os.path.join(split_directory, 'text.{}.int'.format(job_name)), 'r') as inf, \
            open(fst_path, 'wb') as outf, \
            open(log_path, 'w') as logf:
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

        with open(log_path, 'a') as logf:
            fst_ark_path = os.path.join(directory, 'fsts.{}.ark'.format(job_name))
            fst_scp_path = os.path.join(directory, 'fsts.{}.scp'.format(job_name))
            proc = subprocess.Popen([thirdparty_binary('fstcopy'),
                                     'ark:{}'.format(fst_path),
                                     'ark,scp:{},{}'.format(fst_ark_path, fst_scp_path)], stderr=logf)
            proc.communicate()

            temp_fst_path = os.path.join(directory, 'temp.fst.{}'.format(job_name))

            with open(fst_scp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    utt = line.split()[0]
                    print(utt)
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

    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(compile_train_graphs_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def mono_align_equal_func(mono_directory, split_directory, job_name, feat_path):  # pragma: no cover
    fst_path = os.path.join(mono_directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(mono_directory, 'tree')
    mdl_path = os.path.join(mono_directory, '0.mdl')
    directory = os.path.join(split_directory, str(job_name))
    log_path = os.path.join(mono_directory, 'log', 'align.0.{}.log'.format(job_name))
    ali_path = os.path.join(mono_directory, '0.{}.acc'.format(job_name))
    with open(log_path, 'w') as logf, \
            open(ali_path, 'wb') as outf:
        align_proc = subprocess.Popen([thirdparty_binary('align-equal-compiled'), "ark:" + fst_path,
                                       'ark:' + feat_path, 'ark,t:-'],
                                      stdout=subprocess.PIPE, stderr=logf)
        stats_proc = subprocess.Popen([thirdparty_binary('gmm-acc-stats-ali'), '--binary=true',
                                       mdl_path, 'ark:' + feat_path, 'ark:-', '-'],
                                      stdin=align_proc.stdout, stderr=logf, stdout=outf)
        stats_proc.communicate()


def mono_align_equal(mono_directory, split_directory, num_jobs):
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
    jobs = [(mono_directory, split_directory, x, os.path.join(split_directory, 'cmvndeltafeats.{}'.format(x)))
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(mono_align_equal_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def compile_utterance_train_graphs_func(directory, lang_directory, split_directory, job_name, debug=False):  # pragma: no cover
    disambig_int_path = os.path.join(lang_directory, 'phones', 'disambig.int')
    tree_path = os.path.join(directory, 'tree')
    mdl_path = os.path.join(directory, 'final.mdl')
    lexicon_fst_path = os.path.join(lang_directory, 'L_disambig.fst')
    fsts_path = os.path.join(split_directory, 'utt2fst.{}'.format(job_name))
    graphs_path = os.path.join(directory, 'utterance_graphs.{}.fst'.format(job_name))

    log_path = os.path.join(directory, 'log', 'compile-graphs-fst.0.{}.log'.format(job_name))

    with open(log_path, 'w') as logf, open(fsts_path, 'r', encoding='utf8') as f:
        proc = subprocess.Popen([thirdparty_binary('compile-train-graphs-fsts'),
                                 '--transition-scale=1.0', '--self-loop-scale=0.1',
                                 '--read-disambig-syms={}'.format(disambig_int_path),
                                 tree_path, mdl_path,
                                 lexicon_fst_path,
                                 "ark:-", "ark:" + graphs_path],
                                stdin=subprocess.PIPE, stderr=logf)
        group = []
        for line in f:
            group.append(line)
            if line.strip() == '':
                for l in group:
                    proc.stdin.write(l.encode('utf8'))
                group = []
                proc.stdin.flush()

        proc.communicate()


def test_utterances_func(directory, lang_directory, split_directory, job_name):  # pragma: no cover
    log_path = os.path.join(directory, 'log', 'decode.0.{}.log'.format(job_name))
    words_path = os.path.join(lang_directory, 'words.txt')
    mdl_path = os.path.join(directory, 'final.mdl')
    feat_path = os.path.join(split_directory, 'cmvndeltafeats.{}'.format(job_name))
    graphs_path = os.path.join(directory, 'utterance_graphs.{}.fst'.format(job_name))

    text_int_path = os.path.join(split_directory, 'text.{}.int'.format(job_name))
    edits_path = os.path.join(directory, 'edits.{}.txt'.format(job_name))
    out_int_path = os.path.join(directory, 'aligned.{}.int'.format(job_name))
    acoustic_scale = 0.1
    beam = 15.0
    lattice_beam = 8.0
    max_active = 750
    lat_path = os.path.join(directory, 'lat.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        latgen_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster', ),
                                        '--acoustic-scale={}'.format(acoustic_scale),
                                        '--beam={}'.format(beam),
                                        '--max-active={}'.format(max_active), '--lattice-beam={}'.format(lattice_beam),
                                        '--word-symbol-table=' + words_path,
                                        mdl_path, 'ark:' + graphs_path, 'ark:' + feat_path, 'ark:' + lat_path],
                                       stderr=logf)
        latgen_proc.communicate()

        oracle_proc = subprocess.Popen([thirdparty_binary('lattice-oracle'),
                                        'ark:' + lat_path, 'ark,t:' + text_int_path,
                                        'ark,t:' + out_int_path, 'ark,t:' + edits_path],
                                       stderr=logf)
        oracle_proc.communicate()


def test_utterances(aligner):
    print('Checking utterance transcriptions...')
    from alignment.sequence import Sequence
    from alignment.vocabulary import Vocabulary
    from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

    from .corpus import load_scp

    split_directory = aligner.corpus.split_directory
    model_directory = aligner.tri_directory
    lang_directory = aligner.dictionary.output_directory
    with mp.Pool(processes=aligner.num_jobs) as pool:
        jobs = [(model_directory, lang_directory, split_directory, x)
                for x in range(aligner.num_jobs)]
        results = [pool.apply_async(compile_utterance_train_graphs_func, args=i) for i in jobs]
        output = [p.get() for p in results]
        print('Utterance FSTs compiled!')
        print('Decoding utterances (this will take some time)...')
        results = [pool.apply_async(test_utterances_func, args=i) for i in jobs]
        output = [p.get() for p in results]
        print('Finished decoding utterances!')

    word_mapping = aligner.dictionary.reversed_word_mapping
    v = Vocabulary()
    errors = {}

    for job in range(aligner.num_jobs):
        text_path = os.path.join(split_directory, 'text.{}'.format(job))
        texts = load_scp(text_path)
        aligned_int = load_scp(os.path.join(model_directory, 'aligned.{}.int'.format(job)))
        with open(os.path.join(model_directory, 'aligned.{}'.format(job)), 'w') as outf:
            for utt, line in sorted(aligned_int.items()):
                text = []
                for t in line:
                    text.append(word_mapping[int(t)])
                outf.write('{} {}\n'.format(utt, ' '.join(text)))
                ref_text = texts[utt]
                if len(text) < len(ref_text) - 7:
                    insertions = [x for x in text if x not in ref_text]
                    deletions = [x for x in ref_text if x not in text]
                else:
                    aligned_seq = Sequence(text)
                    ref_seq = Sequence(ref_text)

                    alignedEncoded = v.encodeSequence(aligned_seq)
                    refEncoded = v.encodeSequence(ref_seq)
                    scoring = SimpleScoring(2, -1)
                    a = GlobalSequenceAligner(scoring, -2)
                    score, encodeds = a.align(refEncoded, alignedEncoded, backtrace=True)
                    insertions = []
                    deletions = []
                    for encoded in encodeds:
                        alignment = v.decodeSequenceAlignment(encoded)
                        for i, f in enumerate(alignment.first):
                            s = alignment.second[i]
                            if f == '-':
                                insertions.append(s)
                            if s == '-':
                                deletions.append(f)
                if insertions or deletions:
                    errors[utt] = (insertions, deletions, ref_text, text)
    if not errors:
        print('There were no utterances with transcription issues.')
        return True
    out_path = os.path.join(aligner.output_directory, 'transcription_problems.txt')
    with open(out_path, 'w') as problemf:
        problemf.write('Utterance\tInsertions\tDeletions\tReference\tDecoded\n')
        for utt, (insertions, deletions, ref_text, text) in sorted(errors.items(),
                                                                   key=lambda x: -1 * (len(x[1][1]) + len(x[1][2]))):
            problemf.write('{}\t{}\t{}\t{}\t{}\n'.format(utt, ', '.join(insertions), ', '.join(deletions),
                                                         ' '.join(ref_text), ' '.join(text)))
    print(
        'There were {} of {} utterances with at least one transcription issue. Please see the outputted csv file {}.'.format(
            len(errors), aligner.corpus.num_utterances, out_path))
    return False


def align_func(directory, iteration, job_name, mdl, config, feat_path):  # pragma: no cover
    fst_path = os.path.join(directory, 'fsts.{}'.format(job_name))
    log_path = os.path.join(directory, 'log', 'align.{}.{}.log'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    print("RUNNING")
    with open(log_path, 'w') as logf, \
            open(ali_path, 'wb') as outf:
        align_proc = subprocess.Popen([thirdparty_binary('gmm-align-compiled')] + config.scale_opts +
                                      ['--beam={}'.format(config.beam),
                                       #'--retry-beam={}'.format(config.beam * 4),
                                       '--retry-beam={}'.format(config.retry_beam),
                                       '--careful=false',
                                       mdl,
                                       "ark:" + fst_path, "ark:" + feat_path, "ark:-"],
                                      stderr=logf,
                                      stdout=outf)
        logf.write("hello")
        align_proc.communicate()

def align_no_pool(iteration, directory, split_directory, optional_silence, num_jobs, config, feature_name=None):
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('gmm-boost-silence'),
                                           config.boost_silence, optional_silence, make_path_safe(mdl_path))
    print("Safe mdl path:", make_path_safe(mdl_path))

    feat_name = feature_name + '.{}'

    for x in range(num_jobs):
        align_func(directory, iteration, x, mdl, config, os.path.join(split_directory, feat_name.format(x)))



def align(iteration, directory, split_directory, optional_silence, num_jobs, config, feature_name=None):
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
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('gmm-boost-silence'),
                                           config.boost_silence, optional_silence, make_path_safe(mdl_path))
    print("Safe mdl path:", make_path_safe(mdl_path))

    #if feature_name == None:
    feat_name = 'cmvndeltafeats'

    if config.do_lda_mllt:
        #feat_name += '_lda_mllt'
        #feat_name += '.{}_sub'

        feat_name = 'cmvnsplicetransformfeats.{}'
    else:
        feat_name += '.{}'
    jobs = [(directory, iteration, x, mdl, config, os.path.join(split_directory, feat_name.format(x)))
            for x in range(num_jobs)]
    print("JOBS FROM ALIGN:", jobs)

    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(align_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def ali_to_textgrid_func(output_directory, model_directory, dictionary, corpus, job_name):  # pragma: no cover
    text_int_path = os.path.join(corpus.split_directory, 'text.{}.int'.format(job_name))
    log_path = os.path.join(model_directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    ali_path = os.path.join(model_directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(model_directory, 'final.mdl')
    aligned_path = os.path.join(model_directory, 'aligned.{}'.format(job_name))
    word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(job_name))

    frame_shift = corpus.mfcc_configs[0].config_dict['frame-shift'] / 1000
    with open(log_path, 'w') as logf:
        lin_proc = subprocess.Popen([thirdparty_binary('linear-to-nbest'), "ark:" + ali_path,
                                     "ark:" + text_int_path,
                                     '', '', 'ark:-'],
                                    stdout=subprocess.PIPE, stderr=logf)
        align_proc = subprocess.Popen([thirdparty_binary('lattice-align-words'),
                                       os.path.join(dictionary.phones_dir, 'word_boundary.int'), model_path,
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


def convert_ali_to_textgrids(output_directory, model_directory, dictionary, corpus, num_jobs):
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
    jobs = [(output_directory, model_directory, dictionary, corpus, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs) as pool:
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
    print("TREE LOG PATH:", log_path)

    treeacc_path = os.path.join(directory, '{}.treeacc'.format(job_name))

    print("TREEACC PATH:", treeacc_path)
    print("TREE MDL:", mdl)
    print("TREE ALI PATH:", ali_path)
    print("TREE FEAT PATH:", feat_path)

    with open(log_path, 'w') as logf:
        subprocess.call([thirdparty_binary('acc-tree-stats')] + context_opts +
                        ['--ci-phones=' + ci_phones, mdl, "ark:" + feat_path,
                         "ark:" + ali_path,
                         treeacc_path], stderr=logf)



def tree_stats(directory, align_directory, split_directory, ci_phones, num_jobs, fmllr=False, lda_mllt=False, feature_name=None):
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
    if feature_name == None:
        feat_name = 'cmvndeltafeats'
    else:
        feat_name = feature_name

    if fmllr:
        feat_name += '_fmllr'

    #feat_name += '.{}'

    #if feature_name == None:
    feat_name += '.{}'
    #else:
    #    feat_name += '.{}_sub'

    print("feature name:", feature_name)
    mdl_path = os.path.join(align_directory, 'final.mdl')
    #if feature_name == None:
    #    print(":)")
    #    mdl_path = os.path.join(directory, 'final.mdl')
    #    align_directory = directory
    print("mdl path for tree:", mdl_path)
    jobs = [(directory, ci_phones, mdl_path,
             os.path.join(split_directory, feat_name.format(x)),
             os.path.join(align_directory, 'ali.{}'.format(x)), x)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(tree_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]

    tree_accs = [os.path.join(directory, '{}.treeacc'.format(x)) for x in range(num_jobs)]
    log_path = os.path.join(directory, 'log', 'sum_tree_acc.log')
    with open(log_path, 'w') as logf:
        subprocess.call([thirdparty_binary('sum-tree-stats'), os.path.join(directory, 'treeacc')] +
                        tree_accs, stderr=logf)
    #for f in tree_accs:
    #    os.remove(f)


def convert_alignments_func(directory, align_directory, job_name):  # pragma: no cover
    mdl_path = os.path.join(directory, '1.mdl')
    tree_path = os.path.join(directory, 'tree')
    ali_mdl_path = os.path.join(align_directory, 'final.mdl')
    ali_path = os.path.join(align_directory, 'ali.{}'.format(job_name))
    new_ali_path = os.path.join(directory, 'ali.{}'.format(job_name))

    log_path = os.path.join(directory, 'log', 'convert.{}.log'.format(job_name))
    with open(log_path, 'w') as logf:
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
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(convert_alignments_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def calc_fmllr_func(directory, split_directory, sil_phones, job_name, config, initial,
                    model_name='final'):  # pragma: no cover
    feat_path = os.path.join(split_directory, 'cmvndeltafeats')
    if not initial:
        feat_path += '_fmllr'
    feat_path += '.{}'.format(job_name)
    feat_fmllr_path = os.path.join(split_directory, 'cmvndeltafeats_fmllr.{}'.format(job_name))
    log_path = os.path.join(directory, 'log', 'fmllr.{}.log'.format(job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_name))
    utt2spk_path = os.path.join(split_directory, 'utt2spk.{}'.format(job_name))
    if not initial:
        tmp_trans_path = os.path.join(directory, 'trans.temp.{}'.format(job_name))
        trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
        cmp_trans_path = os.path.join(directory, 'trans.cmp.{}'.format(job_name))
    else:
        tmp_trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
    post_path = os.path.join(directory, 'post.{}'.format(job_name))
    weight_path = os.path.join(directory, 'weight.{}'.format(job_name))
    print("FROM CALC FMLLR, INITIAL?", initial)
    print("FROM CALC FMLLR, FEAT PATH:", feat_path)
    with open(log_path, 'w') as logf:
        subprocess.call([thirdparty_binary('ali-to-post'),
                         "ark:" + ali_path, 'ark:' + post_path], stderr=logf)

        subprocess.call([thirdparty_binary('weight-silence-post'), '0.0',
                         sil_phones, mdl_path, 'ark:' + post_path,
                         'ark:' + weight_path], stderr=logf)

        subprocess.call([thirdparty_binary('gmm-est-fmllr'),
                         '--verbose=4',
                         '--fmllr-update-type={}'.format(config.fmllr_update_type),
                         '--spk2utt=ark:' + spk2utt_path, mdl_path, "ark,s,cs:" + feat_path,
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
            feat_path = os.path.join(split_directory, 'cmvndeltafeats.{}'.format(job_name))
        else:
            trans_path = tmp_trans_path
        subprocess.call([thirdparty_binary('transform-feats'),
                         '--utt2spk=ark:' + utt2spk_path,
                         'ark:' + trans_path, 'ark:' + feat_path,
                         'ark:' + feat_fmllr_path],
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
        model_name = 'final'
    else:
        model_name = iteration
    jobs = [(directory, split_directory, sil_phones, x, config, initial, model_name)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(calc_fmllr_func, args=i) for i in jobs]
        output = [p.get() for p in results]


def calc_lda_mllt_func(directory, split_directory, fmllr_dir, sil_phones, num_jobs, job_name, config, num_iters, initial,
                    model_name='final', corpus=None):  # pragma: no cover
    feat_path = os.path.join(split_directory, 'cmvnsplicetransformfeats')
    if not initial:
        feat_path += '_lda_mllt'
    feat_path += '.{}'.format(job_name)
    feat_lda_mllt_path = os.path.join(split_directory, 'cmvnsplicetransformfeats_lda_mllt.{}'.format(job_name))
    log_path = os.path.join(directory, 'log', 'lda_mllt.{}.log'.format(job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    if not initial:
        mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    else:
        mdl_path = os.path.join(directory, '0.mdl')
        model_name = 0
        feat_path = os.path.join(split_directory, 'cmvnsplicetransformfeats.{}'.format(job_name))
    spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_name))
    utt2spk_path = os.path.join(split_directory, 'utt2spk.{}'.format(job_name))
    if not initial:
        tmp_trans_path = os.path.join(directory, 'trans.temp.{}'.format(job_name))
        trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
        cmp_trans_path = os.path.join(directory, 'trans.cmp.{}'.format(job_name))
    else:
        tmp_trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
    post_path = os.path.join(directory, 'post.{}'.format(job_name))
    weight_path = os.path.join(directory, 'weight.{}'.format(job_name))

    # Estimating MLLT
    with open(log_path, 'a') as logf:
        subprocess.call([thirdparty_binary('ali-to-post'),
                         "ark:" + ali_path, 'ark:' + post_path], stderr=logf)

        subprocess.call([thirdparty_binary('weight-silence-post'), '0.0',
                         sil_phones, mdl_path, 'ark:' + post_path,
                         'ark:' + weight_path], stderr=logf)
        if initial:
            feat_path = os.path.join(split_directory, 'cmvnsplicetransformfeats.{}'.format(job_name))
        if model_name == 1:
            subprocess.call([thirdparty_binary('gmm-acc-mllt'),
                             '--rand-prune=' + str(config.randprune),
                             mdl_path,
                             'ark:'+os.path.join(split_directory, 'cmvnsplicetransformfeats.{}'.format(job_name)),
                             'ark:'+post_path,
                             directory + '/{}.{}.macc'.format(model_name, job_name)],
                             stderr=logf)
        else:
            subprocess.call([thirdparty_binary('gmm-acc-mllt'),
                             '--rand-prune=' + str(config.randprune),
                             mdl_path,
                             'ark:'+feat_path,
                             'ark:'+post_path,
                             directory + '/{}.{}.macc'.format(model_name, job_name)],
                             stderr=logf)

        macc_list = []
        for j in range(int(model_name)+1):
            if j == range(int(model_name))[0]:
                continue
            macc_list.append(directory+'/{}.{}.macc'.format(j, job_name))
        subprocess.call([thirdparty_binary('est-mllt'),
                                           directory + '/{}.mat.new'.format(model_name)]
                                           + macc_list,
                                           stderr=logf)
    log_path = os.path.join(directory, 'log', 'transform_means.{}.log'.format(job_name))
    with open(log_path, 'a') as logf:
        subprocess.call([thirdparty_binary('gmm-transform-means'),
                                           directory + '/{}.mat.new'.format(model_name),
                                           mdl_path, mdl_path],
                                           stderr=logf)

        if not initial:
            subprocess.call([thirdparty_binary('compose-transforms'),
                            '--print-args=false',
                            directory + '/{}.mat.new'.format(model_name),
                            directory + '/{}.mat'.format(int(model_name)-1),
                            directory + '/{}.mat'.format(model_name)],
                            stderr=logf)
            logf.write("WRITING THIS MAT NOW:\n")
            logf.write(directory + '/{}.mat'.format(model_name))
            feat_path = os.path.join(split_directory, 'cmvnsplicetransformfeats.{}'.format(job_name))
        else:
            trans_path = tmp_trans_path

        corpus._norm_splice_transform_feats(directory, num=int(model_name))


def calc_lda_mllt(directory, split_directory, fmllr_dir, sil_phones, num_jobs, config, num_iters,
               initial=False, iteration=None, corpus=None):
    """
    Fill out docstring

    """
    if iteration is None:
        model_name = 'final'
    else:
        model_name = iteration
    jobs = [(directory, split_directory, fmllr_dir, sil_phones, num_jobs, x, config, num_iters, initial, model_name, corpus)
            for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(calc_lda_mllt_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def gmm_gselect_func(directory, config, feats, x):
    log_path = os.path.join(directory, 'log', 'gselect.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        subprocess.call([thirdparty_binary('gmm-gselect'),
                        '--n=' + str(config.num_gselect),
                        os.path.join(directory, '0.dubm'),
                        'ark:' + feats,
                        'ark:' + os.path.join(directory, 'gselect.{}'.format(x))],
                        stderr=logf)

def gmm_gselect(directory, config, feats, num_jobs):
    jobs = [(directory, config, feats, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(gmm_gselect_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def acc_global_stats_func(directory, config, feats, x, iteration):
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    with open(log_path, 'w') as logf:
        gmm_global_acc_proc = subprocess.Popen([thirdparty_binary('gmm-global-acc-stats'),
                                               '--gselect=' + 'ark:' + os.path.join(directory, 'gselect.{}'.format(x)),
                                               os.path.join(directory, '{}.dubm'.format(iteration)),
                                               'ark:' + feats,
                                               os.path.join(directory, '{}.{}.acc'.format(iteration, x))],
                                               stderr=logf)
        gmm_global_acc_proc.communicate()

def acc_global_stats(directory, config, feats, num_jobs, iteration):
    jobs = [(directory, config, feats, x, iteration) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(acc_global_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def gauss_to_post_func(directory, config, diag_ubm_directory, gmm_feats, x):
    modified_posterior_scale = config.posterior_scale * config.subsample
    log_path = os.path.join(directory, 'log', 'post.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                    '--n=' + str(config.num_gselect),
                                                    '--min-post=' + str(config.min_post),
                                                    os.path.join(diag_ubm_directory, 'final.dubm'),
                                                    'ark:' + gmm_feats,
                                                    'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=logf)
        scale_post_proc = subprocess.Popen([thirdparty_binary('scale-post'),
                                            'ark:-',
                                            str(modified_posterior_scale),
                                            'ark:' + os.path.join(directory, 'post.{}'.format(x))],
                                            stdin=gmm_global_get_post_proc.stdout,
                                            stderr=logf)
        scale_post_proc.communicate()

def gauss_to_post(directory, config, diag_ubm_directory, gmm_feats, num_jobs):
    jobs = [(directory, config, diag_ubm_directory, gmm_feats, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(gauss_to_post_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def acc_ivector_stats_func(directory, config, feat_path, num_jobs, x, iteration):
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    with open(log_path, 'w') as logf:
        # There is weird threading/array stuff here, so come back if necessary
        acc_stats_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-acc-stats'),
                                          os.path.join(directory, '{}.ie'.format(iteration)),
                                          'ark:' + feat_path,
                                          'ark:' + os.path.join(directory, 'post.{}'.format(x)),
                                          os.path.join(directory, 'accinit.{}.{}'.format(iteration, x))],
                                          stderr=logf)
        acc_stats_proc.communicate()

        accinits = []
        if x == 0:
            accinits.append(os.path.join(directory, 'accinit.{}.0'.format(iteration)))
        else:
            accinits = [os.path.join(directory, 'accinit.{}.{}'.format(iteration, j))
                         for j in range(x)]
        print("accinits:", accinits)
        sum_accs_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-sum-accs'),
                                         '--parallel=true']
                                         + accinits
                                         + [os.path.join(directory, 'acc.{}'.format(iteration))],
                                         stderr=logf)

        sum_accs_proc.communicate()


def acc_ivector_stats(directory, config, feat_path, num_jobs, iteration):
    jobs = [(directory, config, feat_path, num_jobs, x, iteration) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(acc_ivector_stats_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def extract_ivectors_func(directory, training_dir, ieconf, config, x):
    log_dir = os.path.join(directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(directory, 'log', 'extract_ivectors.{}.log'.format(x))
    ark_path = os.path.join(directory, 'ivector_online.{}.ark'.format(x))
    scp_path = os.path.join(directory, 'ivector_online.{}.scp'.format(x))
    with open(log_path, 'w') as logf:
        extract_proc = subprocess.Popen([thirdparty_binary('ivector-extract-online2'),
                                        '--config={}'.format(ieconf),
                                        'ark:' + os.path.join(training_dir, 'spk2utt'),
                                        'scp:' + os.path.join(training_dir, 'feats.scp'),
                                        'ark:-'],
                                        stdout=subprocess.PIPE,
                                        stderr=logf)
        copy_proc = subprocess.Popen([thirdparty_binary('copy-feats'),
                                     #'compress=true',
                                     'ark:-',
                                     'ark,scp:{},{}'.format(ark_path, scp_path)],
                                     stdin=extract_proc.stdout,
                                     stderr=logf)
        copy_proc.communicate()

def extract_ivectors(directory, training_dir, ieconf, config, num_jobs):
    jobs = [(directory, training_dir, ieconf, config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(extract_ivectors_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def get_egs_helper(nnet_dir, label, feats, ivector_period, to_filter, ivector_dir, ivector_randomize_prob, logf, x):
    new_feats = os.path.join(nnet_dir, '{}_helped.{}'.format(label, x))
    with open(new_feats, 'w') as outf:
        #filter_scp_path = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/utils/filter_scp.pl"
        filter_scp_path = "/data/acoles/acoles/kaldi/egs/wsj/s5/utils/filter_scp.pl"
        filter_proc = subprocess.Popen([filter_scp_path,
                                        to_filter,
                                        os.path.join(ivector_dir, 'ivector_online.scp')],
                                        stdout=subprocess.PIPE,
                                        stderr=logf)
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                '--n={}'.format(-10),
                                                'scp:-',
                                                'ark:-'],
                                                stdin=filter_proc.stdout,
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
        ivector_random_proc = subprocess.Popen([thirdparty_binary('ivector-randomize'),
                                                '--randomize-prob={}'.format(0),
                                                'ark:-',
                                                'ark:-'],
                                                stdin=subsample_feats_proc.stdout,
                                                stdout=subprocess.PIPE,
                                                stderr=logf)

        paste_feats_proc = subprocess.Popen([thirdparty_binary('paste-feats'),
                                            '--length-tolerance={}'.format(10),
                                            'ark:'+ feats,
                                            'ark:-',
                                            'ark:-'],
                                            stdin=ivector_random_proc.stdout,
                                            stdout=outf,
                                            stderr=logf)
        paste_feats_proc.communicate()

def get_egs_func(nnet_dir, egs_dir, training_dir, split_dir, ali_dir, ivector_dir, feats, valid_uttlist, train_subset_uttlist, config, x):
    # Create training examples
    iters_per_epoch = config.iters_per_epoch
    ivector_dim = 100 # Make safe later
    ivectors_opt = '--const-feat-dim={}'.format(ivector_dim)
    ivector_period = 3000
    ivector_randomize_prob = 0.0
    cmvn_opts = []

    # Deal with ivector stuff
    #filter_scp_path = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/utils/filter_scp.pl"
    filter_scp_path = "/data/acoles/acoles/kaldi/egs/wsj/s5/utils/filter_scp.pl"
    log_path = os.path.join(nnet_dir, 'log', 'get_egs_feats.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        # Gets "feats" (Kaldi)
        egs_feats = os.path.join(nnet_dir, 'egsfeats.{}'.format(x))
        with open(egs_feats, 'w') as outf:
            filter_proc = subprocess.Popen([filter_scp_path,
                                            valid_uttlist,
                                            os.path.join(split_dir, 'feats.{}.scp'.format(x))],
                                            stdout=subprocess.PIPE,
                                            stderr=logf)
            apply_cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn')]
                                                + cmvn_opts
                                                + ['--utt2spk=ark:{}'.format(os.path.join(split_dir, 'utt2spk.{}'.format(x))),
                                                'scp:' + os.path.join(split_dir, 'cmvn.{}.scp'.format(x)),
                                                'scp:-', 'ark:-'],
                                                stdin=filter_proc.stdout,
                                                stdout=outf,
                                                stderr=logf)
            apply_cmvn_proc.communicate()

        # Gets "valid_feats" (Kaldi)
        egs_valid_feats = os.path.join(nnet_dir, 'egsvalidfeats.{}'.format(x))
        with open(egs_valid_feats, 'w') as outf:
            filter_proc = subprocess.Popen([filter_scp_path,
                                            valid_uttlist,
                                            os.path.join(training_dir, 'feats.scp')],
                                            stdout=subprocess.PIPE,
                                            stderr=logf)
            apply_cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn')]
                                                + cmvn_opts
                                                + ['--utt2spk=ark:{}'.format(os.path.join(training_dir, 'utt2spk')),
                                                'scp:' + os.path.join(training_dir, 'cmvn.scp'),
                                                'scp:-', 'ark:-'],
                                                stdin=filter_proc.stdout,
                                                stdout=outf,
                                                stderr=logf)
            apply_cmvn_proc.communicate()

        """# Gets "train_subset_feats" (Kaldi)
        egs_train_subset_feats = os.path.join(nnet_dir, 'egstrainsubsetfeats.{}'.format(x))
        with open(egs_train_subset_feats, 'w') as outf:
            filter_proc = subprocess.Popen([filter_scp_path,
                                            train_subset_uttlist,
                                            os.path.join(training_dir, 'feats.scp')],
                                            stdout=subprocess.PIPE,
                                            stderr=logf)
            apply_cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn')]
                                                + cmvn_opts
                                                + ['--utt2spk=ark:{}'.format(os.path.join(training_dir, 'utt2spk')),
                                                'scp:' + os.path.join(training_dir, 'cmvn.scp'),
                                                'scp:-', 'ark:-'],
                                                stdin=filter_proc.stdout,
                                                stdout=outf,
                                                stderr=logf)
            apply_cmvn_proc.communicate()"""

        # Get final forms of these feats
        get_egs_helper(nnet_dir, 'egsfeats', egs_feats, ivector_period, os.path.join(split_dir, 'utt2spk.{}'.format(x)), ivector_dir, ivector_randomize_prob, logf, x)
        get_egs_helper(nnet_dir, 'egsvalidfeats', egs_valid_feats, ivector_period, valid_uttlist, ivector_dir, ivector_randomize_prob, logf, x)
        #get_egs_helper(nnet_dir, 'egstrainsubsetfeats', egs_feats, ivector_period, train_subset_uttlist, ivector_dir, ivector_randomize_prob, logf, x)

    # ---------------------------------

    log_path = os.path.join(nnet_dir, 'log', 'ali_to_post.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        ali_to_pdf_proc = subprocess.Popen([thirdparty_binary('ali-to-pdf'),
                                            os.path.join(ali_dir, 'final.mdl'),
                                            'ark:' + os.path.join(ali_dir, 'ali.{}'.format(x)),
                                            'ark:-'],
                                            stdout=subprocess.PIPE,
                                            stderr=logf)
        ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                            'ark:-', 'ark:-'],
                                            stdin=ali_to_pdf_proc.stdout,
                                            stderr=logf,
                                            stdout=subprocess.PIPE)

    log_path = os.path.join(nnet_dir, 'log', 'get_egs.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        nnet_get_egs_proc = subprocess.Popen([thirdparty_binary('nnet-get-egs'),
                                             ivectors_opt,
                                             '--left-context=' + str(config.splice_width),
                                             '--right-context=' + str(config.splice_width),
                                             'ark:' + os.path.join(nnet_dir, 'egsfeats_helped.{}'.format(x)),
                                             'ark:-',
                                             'ark:-'],
                                             stdin=ali_to_post_proc.stdout,
                                             stdout=subprocess.PIPE,
                                             stderr=logf)
        nnet_copy_egs_proc = subprocess.Popen([thirdparty_binary('nnet-copy-egs'),
                                              'ark:-',
                                              'ark:' + os.path.join(egs_dir, 'egs_orig.{}'.format(x))],
                                              stdin=nnet_get_egs_proc.stdout,
                                              stderr=logf)
        nnet_copy_egs_proc.communicate()

    # Rearranging training examples
    log_path = os.path.join(nnet_dir, 'log', 'nnet_copy_egs.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        nnet_copy_egs_proc = subprocess.Popen([thirdparty_binary('nnet-copy-egs'),
                                              '--srand=' + str(x),
                                              'ark:' + os.path.join(egs_dir, 'egs_orig.{}'.format(x)),
                                              'ark:' + os.path.join(egs_dir, 'egs_temp.{}'.format(x))],
                                              stderr=logf)
        nnet_copy_egs_proc.communicate()

    # Shuffling training examples
    log_path = os.path.join(nnet_dir, 'log', 'nnet_shuffle_egs.{}.log'.format(x))
    with open(log_path, 'w') as logf:
        nnet_shuffle_egs_proc = subprocess.Popen([thirdparty_binary('nnet-shuffle-egs'),
                                                 '--srand=' + str(x),
                                                 'ark:' + os.path.join(egs_dir, 'egs_temp.{}'.format(x)),
                                                 'ark:' + os.path.join(egs_dir, 'egs.{}'.format(x))],
                                                 stderr=logf)
        nnet_shuffle_egs_proc.communicate()

def get_egs(nnet_dir, egs_dir, training_dir, split_dir, ali_dir, ivector_dir, feats, valid_uttlist, train_subset_uttlist, config, num_jobs):
    jobs = [(nnet_dir, egs_dir, training_dir, split_dir, ali_dir, ivector_dir, feats, valid_uttlist, train_subset_uttlist, config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(get_egs_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def get_lda_nnet_func(nnet_dir, ali_dir, ivector_dir, training_dir, split_dir, feats, sil_phones, config, N, x):
    log_path = os.path.join(nnet_dir, 'log', 'lda_acc.{}.log'.format(x))
    splice_feats = os.path.join(nnet_dir, 'splicefeats.{}'.format(x))
    with open(log_path, 'w') as logf:
        with open(splice_feats, 'w') as outf:
            cmvn_opts = []

            apply_cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn')]
                                                + cmvn_opts
                                                + ['--utt2spk=ark:{}'.format(os.path.join(split_dir, 'utt2spk.{}'.format(x))),
                                                'scp:' + os.path.join(split_dir, 'cmvn.{}.scp'.format(x)),
                                                'scp:' + os.path.join(split_dir, 'feats.{}.scp'.format(x)),
                                                'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)

            splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats'),
                                                 '--left-context={}'.format(config.splice_width),
                                                 '--right-context={}'.format(config.splice_width),
                                                 'ark:-',
                                                 'ark:-'],
                                                 stdin=apply_cmvn_proc.stdout,
                                                 stdout=outf,
                                                 stderr=logf)
            splice_feats_proc.communicate()
            print("dim proc:")
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'ark:'+splice_feats,
                                         '-'],
                                        stderr=logf)
            print("done dim proc")

        # Add iVector functionality
        ivector_period = 3000
        new_splice_feats = os.path.join(nnet_dir, 'newsplicefeats.{}'.format(x))
        with open(new_splice_feats, 'w') as outf:
            #filter_scp_path = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/utils/filter_scp.pl"
            filter_scp_path = "/data/acoles/acoles/kaldi/egs/wsj/s5/utils/filter_scp.pl"
            filter_proc = subprocess.Popen([filter_scp_path,
                                            os.path.join(split_dir, 'utt2spk.{}'.format(x)),
                                            os.path.join(ivector_dir, 'ivector_online.scp')],
                                            stdout=subprocess.PIPE,
                                            stderr=logf)
            subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                    '--n={}'.format(-10),
                                                    'scp:-',
                                                    'ark:-'],
                                                    stdin=filter_proc.stdout,
                                                    stdout=subprocess.PIPE,
                                                    stderr=logf)
            ivector_random_proc = subprocess.Popen([thirdparty_binary('ivector-randomize'),
                                                    '--randomize-prob={}'.format(0),
                                                    'ark:-',
                                                    'ark:-'],
                                                    stdin=subsample_feats_proc.stdout,
                                                    stdout=subprocess.PIPE,
                                                    stderr=logf)

            paste_feats_proc = subprocess.Popen([thirdparty_binary('paste-feats'),
                                                '--length-tolerance={}'.format(10),
                                                'ark:'+ splice_feats,
                                                'ark:-',
                                                'ark:-'],
                                                stdin=ivector_random_proc.stdout,
                                                stdout=outf,
                                                stderr=logf)
            paste_feats_proc.communicate()

            splice_feats = new_splice_feats

        # Get iVector dimension
        ivector_dim_path = os.path.join(nnet_dir, 'ivector_dim')
        with open(ivector_dim_path, 'w') as outf:
            print("dim proc ivector:")
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'scp:' + os.path.join(ivector_dir, 'ivector_online.scp'),
                                         '-'],
                                        stderr=logf,
                                        stdout=outf)
            print("done dim proc ivector")


            ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                'ark:' + os.path.join(ali_dir, 'ali.{}'.format(x)),
                                                'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=logf)
            weight_silence_post_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                        '0.0',
                                                        sil_phones,
                                                        os.path.join(ali_dir, 'final.mdl'),
                                                        'ark:-',
                                                        'ark:-'],
                                                        stdin=ali_to_post_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
            acc_lda_proc = subprocess.Popen([thirdparty_binary('acc-lda'),
                                            '--rand-prune={}'.format(config.randprune),
                                            os.path.join(ali_dir, 'final.mdl'),
                                            'ark:' + splice_feats,
                                            'ark,s,cs:-',
                                            os.path.join(nnet_dir, 'lda.{}.acc'.format(x))],
                                            stdin=weight_silence_post_proc.stdout,
                                            stderr=logf)
            acc_lda_proc.communicate()

def get_lda_nnet(nnet_dir, ali_dir, ivector_dir, training_dir, split_dir, feats, sil_phones, config, num_jobs):
    num_feats = 10000
    N = num_feats/num_jobs
    jobs = [(nnet_dir, ali_dir, ivector_dir, training_dir, split_dir, feats, sil_phones, config, N, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(get_lda_nnet_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def nnet_train_trans_func(nnet_dir, align_dir, x):
    log_path = os.path.join(nnet_dir, 'log', 'train_trans{}.log'.format(x))
    with open(log_path, 'w') as logf:
        #ali_files = [os.path.join(align_dir 'ali.{}'.format(x))
        #             for x in range(self.num_jobs)]
        train_trans_proc = subprocess.Popen([thirdparty_binary('nnet-train-transitions'),
                                            os.path.join(nnet_dir, '0.mdl'),
                                            'ark:' + os.path.join(align_dir, 'ali.{}'.format(x)),
                                            os.path.join(nnet_dir, '0.mdl')],
                                            stderr=logf)
        train_trans_proc.communicate()

def nnet_train_trans(nnet_dir, align_dir, num_jobs):
    jobs = [(nnet_dir, align_dir, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(nnet_train_trans_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def nnet_train_func(nnet_dir, egs_dir, mdl, i, x):
    log_path = os.path.join(nnet_dir, 'log', 'train.{}.{}.log'.format(i, x))
    with open(log_path, 'w') as logf:
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
                                      os.path.join(nnet_dir, '{}.{}.mdl'.format((i+1), x))],
                                      stdin=shuffle_proc.stdout,
                                      stderr=logf)
        train_proc.communicate()

def nnet_train(nnet_dir, egs_dir, mdl, i, num_jobs):
    jobs = [(nnet_dir, egs_dir, mdl, i, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(nnet_train_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def nnet_get_align_feats_func(nnet_dir, split_dir, lda_dir, ivector_dir, config, x):
    log_path = os.path.join(nnet_dir, 'log', 'alignment_features{}.log'.format(x))
    with open(log_path, 'w') as logf:
        first_feats = os.path.join(nnet_dir, 'alignfeats_first.{}'.format(x))
        utt2spkpath = os.path.join(split_dir, 'utt2spk.{}'.format(x))
        cmvnpath = os.path.join(split_dir, 'cmvn.{}.scp'.format(x))
        featspath = os.path.join(split_dir, 'feats.{}.scp'.format(x))
        with open(first_feats, 'wb') as outf:
            cmvn_proc = subprocess.Popen([thirdparty_binary('apply-cmvn'),
                                          '--utt2spk=ark:' + utt2spkpath,
                                          'scp:' + cmvnpath,
                                          'scp:' + featspath,
                                          'ark:-'],
                                          stdout=outf,
                                          stderr=logf
                                         )
            cmvn_proc.communicate()
        print("dim proc cmvn:")
        dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                     'ark:'+first_feats,
                                     '-'],
                                    stderr=logf)

        new_feats = os.path.join(nnet_dir, 'alignfeats.{}'.format(x))
        with open(new_feats, 'w') as outf:
            #filter_scp_path = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/utils/filter_scp.pl"
            filter_scp_path = "/data/acoles/acoles/kaldi/egs/wsj/s5/utils/filter_scp.pl"
            filter_proc = subprocess.Popen([filter_scp_path,
                                            os.path.join(split_dir, 'utt2spk.{}'.format(x)),
                                            os.path.join(ivector_dir, 'ivector_online.scp')],
                                            stdout=subprocess.PIPE,
                                            stderr=logf)
            subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                    '--n={}'.format(-10),
                                                    'scp:-',
                                                    'ark:-'],
                                                    stdin=filter_proc.stdout,
                                                    stdout=subprocess.PIPE,
                                                    stderr=logf)
            ivector_random_proc = subprocess.Popen([thirdparty_binary('ivector-randomize'),
                                                    '--randomize-prob={}'.format(0),
                                                    'ark:-',
                                                    'ark:-'],
                                                    stdin=subsample_feats_proc.stdout,
                                                    stdout=subprocess.PIPE,
                                                    stderr=logf)

            paste_feats_proc = subprocess.Popen([thirdparty_binary('paste-feats'),
                                                '--length-tolerance={}'.format(10),
                                                'ark:'+ first_feats,
                                                'ark:-',
                                                'ark:-'],
                                                stdin=ivector_random_proc.stdout,
                                                stdout=outf,
                                                stderr=logf)
            paste_feats_proc.communicate()

            print("dim proc after:")
            dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                         'ark:'+new_feats,
                                         '-'],
                                        stderr=logf)
def nnet_get_align_feats(nnet_dir, split_dir, lda_dir, ivector_dir, config, num_jobs):
    jobs = [(nnet_dir, split_dir, lda_dir, ivector_dir, config, x) for x in range(num_jobs)]
    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(nnet_get_align_feats_func, args=i) for i in jobs]
        output = [p.get() for p in results]

def nnet_align_func(i, nnet_dir, mdl_path, config, x):
    feat_path = os.path.join(nnet_dir, 'alignfeats.{}'.format(x))
    fst_path = os.path.join(nnet_dir, 'fsts.{}'.format(x))
    log_path = os.path.join(nnet_dir, 'log', 'align.{}.{}.log'.format(i, x))
    ali_path = os.path.join(nnet_dir, 'ali.{}'.format(x))

    with open(log_path, 'w') as logf, \
            open(ali_path, 'wb') as outf:
        print("dim proc sanity check:")
        dim_proc = subprocess.Popen([thirdparty_binary('feat-to-dim'),
                                     'ark:'+feat_path,
                                     '-'],
                                    stderr=logf)
        dim_proc.communicate()
        align_proc = subprocess.Popen([thirdparty_binary('nnet-align-compiled'),
                                       '--beam={}'.format(config.beam),
                                       '--retry-beam={}'.format(config.retry_beam),
                                       mdl_path,
                                       "ark:" + fst_path, "ark:" + feat_path, "ark:-"],
                                      stderr=logf,
                                      stdout=outf)
        align_proc.communicate()

def nnet_align(i, nnet_dir, optional_silence, num_jobs, config, mdl=None):
    if mdl == None:
        mdl_path = os.path.join(nnet_dir, '{}.mdl'.format(i))
    else:
        mdl_path = mdl
        print("!!!")
    # No nnet equivalent to boost silence (yet?)
    mdl = "{} --boost={} {} {} - |".format(thirdparty_binary('nnet2-boost-silence'),
                                           config.boost_silence, optional_silence, make_path_safe(mdl_path))

    jobs = [(i, nnet_dir, mdl_path, config, x)
            for x in range(num_jobs)]

    with mp.Pool(processes=num_jobs) as pool:
        results = [pool.apply_async(nnet_align_func, args=i) for i in jobs]
        output = [p.get() for p in results]
