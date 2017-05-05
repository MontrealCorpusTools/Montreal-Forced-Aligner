import multiprocessing as mp
import subprocess
import os
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


def acc_stats(iteration, directory, split_directory, num_jobs, fmllr=False):
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
    feat_name = 'cmvndeltafeats'
    if fmllr:
        feat_name += '_fmllr'
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
    with open(log_path, 'w') as logf, \
            open(ali_path, 'wb') as outf:
        align_proc = subprocess.Popen([thirdparty_binary('gmm-align-compiled')] + config.scale_opts +
                                      ['--beam={}'.format(config.beam),
                                       '--retry-beam={}'.format(config.beam * 4),
                                       '--careful=false',
                                       mdl,
                                       "ark:" + fst_path, "ark:" + feat_path, "ark:-"],
                                      stderr=logf,
                                      stdout=outf)
        align_proc.communicate()


def align(iteration, directory, split_directory, optional_silence, num_jobs, config):
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

    feat_name = 'cmvndeltafeats'
    if config.do_fmllr:
        feat_name += '_fmllr'
    feat_name += '.{}'
    jobs = [(directory, iteration, x, mdl, config, os.path.join(split_directory, feat_name.format(x)))
            for x in range(num_jobs)]

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

    treeacc_path = os.path.join(directory, '{}.treeacc'.format(job_name))
    with open(log_path, 'w') as logf:
        subprocess.call([thirdparty_binary('acc-tree-stats')] + context_opts +
                        ['--ci-phones=' + ci_phones, mdl, "ark:" + feat_path,
                         "ark:" + ali_path,
                         treeacc_path], stderr=logf)


def tree_stats(directory, align_directory, split_directory, ci_phones, num_jobs, fmllr=False):
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
    feat_name = 'cmvndeltafeats'
    if fmllr:
        feat_name += '_fmllr'
    feat_name += '.{}'
    mdl_path = os.path.join(align_directory, 'final.mdl')
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
    for f in tree_accs:
        os.remove(f)


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
