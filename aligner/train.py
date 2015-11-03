import os
import subprocess
import re
import shutil

from .prep.helper import (load_scp, load_utt2spk, find_best_groupings,
                        utt2spk_to_spk2utt, save_scp, load_oov_int,
                        load_word_to_int, load_text)

from .multiprocessing import (align, mono_align_equal, compile_train_graphs,
                            acc_stats, tree_stats, convert_alignments,
                             convert_ali_to_textgrids, calc_fmllr)

from .align import align_si, align_fmllr

from .data_split import setup_splits, get_feat_dim

from .config import *

def get_num_gauss(mono_directory):
    with open(os.devnull, 'w') as devnull:
        proc = subprocess.Popen(['gmm-info','--print-args=false',
                    os.path.join(mono_directory, '0.mdl')],
                    stderr = devnull,
                    stdout = subprocess.PIPE)
        stdout, stderr = proc.communicate()
        num = stdout.decode('utf8')
        matches = re.search(r'gaussians (\d+)', num)
        num = int(matches.groups()[0])
    return num

def init_mono(split_directory, lang_directory, mono_directory, num_jobs):
    tree_path = os.path.join(mono_directory,'tree')
    mdl_path = os.path.join(mono_directory,'0.mdl')

    feat_dim = str(get_feat_dim(split_directory))
    directory = os.path.join(split_directory, '1')
    path = os.path.join(directory, 'cmvndeltafeats_sub')
    feat_path = os.path.join(directory, 'cmvndeltafeats')
    shared_phones_opt = "--shared-phones=" + os.path.join(lang_directory,'phones', 'sets.int')
    log_path = os.path.join(directory, 'log')
    with open(path, 'rb') as f, open(log_path, 'w') as logf:
        subprocess.call(['gmm-init-mono',shared_phones_opt,
                        "--train-feats=ark:-",
                        os.path.join(lang_directory,'topo'), feat_dim,
                        mdl_path,
                        tree_path],
                        stdin = f,
                        stderr = logf)
    num_gauss = get_num_gauss(mono_directory)
    compile_train_graphs(mono_directory, lang_directory, split_directory, num_jobs)
    mono_align_equal(mono_directory, lang_directory, split_directory, num_jobs)
    log_path = os.path.join(mono_directory, 'log', 'update.0.log')
    with open(log_path, 'w') as logf:
        acc_files = [os.path.join(mono_directory, '0.{}.acc'.format(x)) for x in range(1, num_jobs+1)]
        est_proc = subprocess.Popen(['gmm-est', '--min-gaussian-occupancy=3',
                '--mix-up={}'.format(num_gauss), '--power={}'.format(power),
                mdl_path, "gmm-sum-accs - {}|".format(' '.join(acc_files)),
                os.path.join(mono_directory,'1.mdl')],
                stderr = logf)
        est_proc.communicate()

def do_mono_training(directory, split_directory, lang_directory, num_jobs):
    num_gauss = get_num_gauss(directory)
    num_iters = mono_num_iters
    realign_iters = mono_realign_iters
    do_training(directory, split_directory, lang_directory,
                        num_gauss, num_jobs,
                    num_iters = num_iters, realign_iters = realign_iters)

def do_tri_training(directory, split_directory,
                    lang_directory, num_jobs, align_often = True):
    num_gauss = tri_num_gauss
    if align_often:
        num_iters = mono_num_iters
        realign_iters = mono_realign_iters
    else:
        num_iters = tri_num_iters
        realign_iters = tri_realign_iters
    do_training(directory, split_directory, lang_directory, num_gauss,
                num_jobs, num_iters = num_iters, realign_iters = realign_iters)

def do_tri_fmllr_training(directory, split_directory,
                        lang_directory, num_jobs, align_often = True):
    num_gauss = tri_num_gauss
    if align_often:
        num_iters = mono_num_iters
        realign_iters = mono_realign_iters
    else:
        num_iters = tri_num_iters
        realign_iters = tri_realign_iters
    do_training(directory, split_directory, lang_directory,
                    num_gauss, num_jobs, do_fmllr = True,
                    num_iters = num_iters, realign_iters = realign_iters)

def do_training(directory, split_directory, lang_directory,
                    num_gauss, num_jobs, do_fmllr = False,
                    num_iters = 40, realign_iters = None):
    if realign_iters is None:
        realign_iters = list(range(0, num_iters, 10))
    sil_phones = load_text(os.path.join(lang_directory, 'phones', 'silence.csl'))
    max_iter_inc = num_iters - 10
    inc_gauss = int((totgauss - num_gauss) / max_iter_inc)
    for i in range(1, num_iters):
        model_path = os.path.join(directory,'{}.mdl'.format(i))
        occs_path = os.path.join(directory, '{}.occs'.format(i+1))
        next_model_path = os.path.join(directory,'{}.mdl'.format(i+1))
        if os.path.exists(next_model_path):
            continue
        if i in realign_iters:
            align(i, directory, split_directory,
                        lang_directory, num_jobs, do_fmllr)
        if do_fmllr and i in fmllr_iters:
            calc_fmllr(directory, split_directory, sil_phones,
                    num_jobs, do_fmllr, i)


        acc_stats(i, directory, split_directory, num_jobs, do_fmllr)
        log_path = os.path.join(directory, 'log', 'update.{}.log'.format(i))
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(directory, '{}.{}.acc'.format(i, x)) for x in range(1, num_jobs+1)]
            est_proc = subprocess.Popen(['gmm-est', '--write-occs='+occs_path,
                    '--mix-up='+str(num_gauss), '--power='+str(power), model_path,
                    "gmm-sum-accs - {}|".format(' '.join(acc_files)), next_model_path],
                    stderr = logf)
            est_proc.communicate()
        if i < max_iter_inc:
            num_gauss += inc_gauss
    shutil.copy(os.path.join(directory,'{}.mdl'.format(num_iters)),
                    os.path.join(directory,'final.mdl'))
    shutil.copy(os.path.join(directory,'{}.occs'.format(num_iters)),
                    os.path.join(directory,'final.occs'))


def train_mono(data_directory, num_jobs = 4):
    lang_directory = os.path.join(data_directory, 'lang')
    train_directory = os.path.join(data_directory, 'train')
    mono_directory = os.path.join(data_directory, 'mono')
    final_mdl = os.path.join(mono_directory, 'final.mdl')
    split_directory = os.path.join(train_directory, 'split{}'.format(num_jobs))
    if os.path.exists(final_mdl):
        print('Monophone training already done, using previous final.mdl')
        return
    os.makedirs(os.path.join(mono_directory, 'log'), exist_ok = True)
    if not os.path.exists(split_directory):
        setup_splits(train_directory, split_directory, lang_directory, num_jobs)

    init_mono(split_directory, lang_directory, mono_directory, num_jobs)
    do_mono_training(mono_directory, split_directory, lang_directory, num_jobs)
    convert_ali_to_textgrids(mono_directory, lang_directory, split_directory, num_jobs)

def init_tri(split_directory, lang_directory, align_directory, tri_directory, cluster_thresh, num_jobs):
    context_opts = []
    ci_phones = load_text(os.path.join(lang_directory, 'phones', 'context_indep.csl'))

    tree_stats(tri_directory, align_directory, split_directory, ci_phones, num_jobs)
    log_path = os.path.join(tri_directory, 'log', 'questions.log')
    tree_path = os.path.join(tri_directory, 'tree')
    treeacc_path = os.path.join(tri_directory, 'treeacc')
    sets_int_path = os.path.join(lang_directory, 'phones', 'sets.int')
    roots_int_path = os.path.join(lang_directory, 'phones', 'roots.int')
    extra_question_int_path = os.path.join(lang_directory, 'phones', 'extra_questions.int')
    topo_path = os.path.join(lang_directory, 'topo')
    questions_path = os.path.join(tri_directory, 'questions.int')
    questions_qst_path = os.path.join(tri_directory, 'questions.qst')
    with open(log_path, 'w') as logf:
        subprocess.call(['cluster-phones'] + context_opts +
        [treeacc_path, sets_int_path, questions_path], stderr = logf)

    with open(extra_question_int_path, 'r') as inf, \
        open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

    log_path = os.path.join(tri_directory, 'log', 'compile_questions.log')
    with open(log_path, 'w') as logf:
        subprocess.call(['compile-questions'] + context_opts +
                [topo_path, questions_path, questions_qst_path],
                stderr = logf)

    log_path = os.path.join(tri_directory, 'log', 'build_tree.log')
    with open(log_path, 'w') as logf:
        subprocess.call(['build-tree'] + context_opts +
            ['--verbose=1', '--max-leaves={}'.format(tri_num_states), \
            '--cluster-thresh={}'.format(cluster_thresh),
             treeacc_path, roots_int_path, questions_qst_path,
             topo_path, tree_path], stderr = logf)

    log_path = os.path.join(tri_directory, 'log', 'init_model.log')
    occs_path = os.path.join(tri_directory, '0.occs')
    mdl_path = os.path.join(tri_directory, '0.mdl')
    with open(log_path, 'w') as logf:
        subprocess.call(['gmm-init-model',
        '--write-occs='+occs_path, tree_path, treeacc_path,
        topo_path, mdl_path], stderr = logf)

    log_path = os.path.join(tri_directory, 'log', 'mixup.log')
    with open(log_path, 'w') as logf:
        subprocess.call(['gmm-mixup', '--mix-up={}'.format(tri_num_gauss),
         mdl_path, occs_path, mdl_path], stderr = logf)
    os.remove(treeacc_path)

    compile_train_graphs(tri_directory, lang_directory, split_directory, num_jobs)
    os.rename(occs_path, os.path.join(tri_directory, '1.occs'))
    os.rename(mdl_path, os.path.join(tri_directory, '1.mdl'))

    convert_alignments(tri_directory, align_directory, num_jobs)

    if os.path.exists(os.path.join(align_directory, 'trans.1')):
        for i in range(1, num_jobs + 1):
            shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),
                    os.path.join(tri_directory, 'trans.{}'.format(i)))

def train_tri(data_directory, num_jobs = 4, cluster_thresh = 100):
    lang_directory = os.path.join(data_directory, 'lang')
    train_directory = os.path.join(data_directory, 'train')
    tri_directory = os.path.join(data_directory, 'tri')
    mono_directory = os.path.join(data_directory, 'mono')
    mono_ali_directory = os.path.join(data_directory, 'mono_ali')
    final_mdl = os.path.join(tri_directory, 'final.mdl')
    if os.path.exists(final_mdl):
        print('Triphone training already done, using previous final.mdl')
        return
    if not os.path.exists(mono_ali_directory):
        align_si(data_directory, mono_directory, mono_ali_directory, num_jobs)

    os.makedirs(os.path.join(tri_directory, 'log'), exist_ok = True)
    split_directory = os.path.join(train_directory, 'split{}'.format(num_jobs))
    if not os.path.exists(split_directory):
        setup_splits(train_directory, split_directory,
                                lang_directory, num_jobs)
    init_tri(split_directory, lang_directory, mono_ali_directory, tri_directory, cluster_thresh, num_jobs)
    do_tri_training(tri_directory, split_directory, lang_directory, num_jobs)
    convert_ali_to_textgrids(tri_directory, lang_directory, split_directory, num_jobs)

def train_tri_fmllr(data_directory, num_jobs = 4, cluster_thresh = 100):
    lang_directory = os.path.join(data_directory, 'lang')
    train_directory = os.path.join(data_directory, 'train')
    tri_directory = os.path.join(data_directory, 'tri')
    tri2_directory = os.path.join(data_directory, 'tri_fmllr')
    ali_directory = os.path.join(data_directory, 'tri_ali_fmllr')
    final_mdl = os.path.join(tri2_directory, 'final.mdl')
    if os.path.exists(final_mdl):
        print('Triphone FMLLR training already done, using previous final.mdl')
        return
    if not os.path.exists(ali_directory):
        align_fmllr(data_directory, tri_directory, ali_directory, num_jobs)

    os.makedirs(os.path.join(tri2_directory, 'log'), exist_ok = True)
    split_directory = os.path.join(train_directory, 'split{}'.format(num_jobs))
    if not os.path.exists(split_directory):
        setup_splits(train_directory, split_directory,
                                lang_directory, num_jobs)
    init_tri(split_directory, lang_directory, ali_directory, tri2_directory, cluster_thresh, num_jobs)
    do_tri_fmllr_training(tri2_directory, split_directory, lang_directory, num_jobs)
    convert_ali_to_textgrids(tri2_directory, lang_directory, split_directory, num_jobs)


def train_sgmm_sat():
    pass
