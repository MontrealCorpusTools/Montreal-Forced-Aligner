
import multiprocessing as mp
import subprocess
import os

from .prep.helper import load_text, load_word_to_int, load_phone_to_int, reverse_mapping

from .textgrid import ctm_to_textgrid

from .config import *

def mfcc_func(mfcc_directory, log_directory, job_name, mfcc_config_path):
    raw_mfcc_path = os.path.join(mfcc_directory, 'raw_mfcc.{}.ark'.format(job_name))
    raw_scp_path = os.path.join(mfcc_directory, 'raw_mfcc.{}.scp'.format(job_name))
    log_path = os.path.join(log_directory, 'make_mfcc.{}.log'.format(job_name))
    scp_path = os.path.join(log_directory, 'wav.{}.scp'.format(job_name))

    with open(log_path, 'w') as f:
        comp_proc = subprocess.Popen(['compute-mfcc-feats', '--verbose=2',
                    '--config=' + mfcc_config_path,
         'scp,p:'+scp_path, 'ark:-'], stdout = subprocess.PIPE,
         stderr = f)
        copy_proc = subprocess.Popen(['copy-feats',
            '--compress=true', 'ark:-',
            'ark,scp:{},{}'.format(raw_mfcc_path,raw_scp_path)],
            stdin = comp_proc.stdout, stderr = f)
        copy_proc.wait()

def mfcc(mfcc_directory, log_directory, num_jobs, mfcc_config_path):
    jobs = [ (mfcc_directory, log_directory, x, mfcc_config_path)
                for x in range(1, num_jobs + 1)]
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(mfcc_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def acc_stats_func(directory, iteration, job_name, feat_path):
    log_path = os.path.join(directory, 'log', 'acc.{}.{}.log'.format(iteration, job_name))
    model_path = os.path.join(directory,'{}.mdl'.format(iteration))
    next_model_path = os.path.join(directory,'{}.mdl'.format(iteration+1))
    acc_path = os.path.join(directory,'{}.{}.acc'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        acc_proc = subprocess.Popen(['gmm-acc-stats-ali', model_path,
             "ark:"+feat_path, "ark,t:" + ali_path,
          acc_path],
          stderr = logf)
        acc_proc.communicate()

def acc_stats(iteration, directory, split_directory, num_jobs, fmllr = False):
    feat_name = 'cmvndeltafeats'
    if fmllr:
        feat_name += '_fmllr'
    jobs = [ (directory, iteration, x, os.path.join(split_directory,str(x), feat_name))
                for x in range(1, num_jobs + 1)]
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(acc_stats_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def compile_train_graphs_func(directory, lang_directory, split_directory, job_name):
    fst_path = os.path.join(directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(directory,'tree')
    mdl_path = os.path.join(directory,'0.mdl')
    data_directory = os.path.join(split_directory, str(job_name))
    log_path = os.path.join(directory, 'log', 'compile-graphs.0.{}.log'.format(job_name))
    with open(os.path.join(data_directory,'text.int'), 'r') as inf, \
        open(fst_path, 'wb') as outf, \
        open(log_path, 'w') as logf:
        proc = subprocess.Popen(['compile-train-graphs',
                    tree_path, mdl_path,
                    os.path.join(lang_directory,'L.fst'),
                    "ark:-", "ark:-"],
                    stdin = inf, stdout = outf, stderr = logf)
        proc.communicate()

def compile_train_graphs(directory, lang_directory, split_directory, num_jobs):
    jobs = [ (directory, lang_directory, split_directory, x)
                for x in range(1, num_jobs + 1)]

    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(compile_train_graphs_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def mono_align_equal_func(mono_directory, lang_directory, split_directory, job_name, feat_path):
    fst_path = os.path.join(mono_directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(mono_directory,'tree')
    mdl_path = os.path.join(mono_directory,'0.mdl')
    directory = os.path.join(split_directory, str(job_name))
    log_path = os.path.join(mono_directory, 'log', 'align.0.{}.log'.format(job_name))
    with open(log_path, 'w') as logf:
        align_proc = subprocess.Popen(['align-equal-compiled', "ark:"+fst_path,
                    'ark:'+feat_path, 'ark,t:-'],stdout = subprocess.PIPE,
                    stderr = logf)
        stats_proc = subprocess.Popen(['gmm-acc-stats-ali', '--binary=true',
                mdl_path, 'ark:'+feat_path, 'ark:-',
                os.path.join(mono_directory,'0.{}.acc'.format(job_name))],
                stdin = align_proc.stdout,
                stderr = logf)
        stats_proc.communicate()

def mono_align_equal(mono_directory, lang_directory, split_directory, num_jobs):

    jobs = [ (mono_directory, lang_directory, split_directory, x, os.path.join(split_directory,str(x), 'cmvndeltafeats'))
                for x in range(1, num_jobs + 1)]

    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(mono_align_equal_func, args = i) for i in jobs]
        output = [p.get() for p in results]
    acc_stats(0, mono_directory, split_directory, num_jobs)

def align_func(directory, iteration, job_name, mdl, feat_path):
    if iteration == 1:
        beam = 6
    else:
        beam = 10
    fst_path = os.path.join(directory, 'fsts.{}'.format(job_name))
    log_path = os.path.join(directory, 'log', 'align.{}.{}.log'.format(iteration, job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        align_proc = subprocess.Popen(['gmm-align-compiled']+ scale_opts +
            ['--beam={}'.format(beam),
            '--retry-beam={}'.format(beam*4), '--careful=false', mdl,
        "ark:"+fst_path, "ark:"+feat_path, "ark,t:"+ ali_path],
        stderr = logf)
        align_proc.communicate()

def align(iteration, directory, split_directory, optional_silence, num_jobs, fmllr = False):
    mdl_path = os.path.join(directory, '{}.mdl'.format(iteration))
    mdl="gmm-boost-silence --boost={} {} {} - |".format(boost_silence, optional_silence, mdl_path)

    feat_name = 'cmvndeltafeats'
    if fmllr:
        feat_name += '_fmllr'

    jobs = [ (directory, iteration, x, mdl, os.path.join(split_directory,str(x), feat_name))
                for x in range(1, num_jobs + 1)]

    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(align_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def ali_to_textgrid_func(directory, lang_directory, split_directory, job_name, word_mapping, phone_mapping):
    text_int_path = os.path.join(split_directory, str(job_name), 'text.int')
    log_path = os.path.join(directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(directory, 'final.mdl')
    aligned_path = os.path.join(directory, 'aligned.{}'.format(job_name))
    word_ctm_path = os.path.join(directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(directory, 'phone_ctm.{}'.format(job_name))
    tg_path = os.path.join(directory, 'textgrids')
    os.makedirs(tg_path, exist_ok = True)
    with open(log_path, 'w') as logf:
        lin_proc = subprocess.Popen(['linear-to-nbest', "ark:"+ ali_path,
                      "ark:"+ text_int_path,
                      '', '', 'ark:-'],
                      stdout = subprocess.PIPE, stderr = logf)
        align_proc = subprocess.Popen(['lattice-align-words',
                        os.path.join(lang_directory, 'phones', 'word_boundary.int'), model_path,
                        'ark:-', 'ark:'+aligned_path],
                        stdin = lin_proc.stdout, stderr = logf)
        align_proc.communicate()

        subprocess.call(['nbest-to-ctm', 'ark:'+aligned_path,
                                word_ctm_path], stderr = logf)
        phone_proc = subprocess.Popen(['lattice-to-phone-lattice', model_path,
                    'ark:'+aligned_path, "ark:-"], stdout = subprocess.PIPE,
                    stderr = logf)
        nbest_proc = subprocess.Popen(['nbest-to-ctm', "ark:-", phone_ctm_path],
                        stdin = phone_proc.stdout, stderr = logf)
        nbest_proc.communicate()
        ctm_to_textgrid(word_ctm_path, phone_ctm_path, tg_path, word_mapping, phone_mapping)



def convert_ali_to_textgrids(directory, lang_directory, split_directory, num_jobs):

    word_mapping = reverse_mapping(load_word_to_int(lang_directory))
    phone_mapping = reverse_mapping(load_phone_to_int(lang_directory))
    jobs = [ (directory, lang_directory, split_directory, x, word_mapping, phone_mapping)
                for x in range(1, num_jobs + 1)]

    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(ali_to_textgrid_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def tree_stats_func(directory, ci_phones, mdl, feat_path, ali_path, job_name):
    context_opts = []
    log_path = os.path.join(directory, 'log', 'acc_tree.{}.log'.format(job_name))

    treeacc_path = os.path.join(directory, '{}.treeacc'.format(job_name))
    with open(log_path, 'w') as logf:
        subprocess.call(['acc-tree-stats'] + context_opts +
            ['--ci-phones='+ci_phones, mdl, "ark:" + feat_path,
            "ark:" + ali_path,
            treeacc_path], stderr = logf)

def tree_stats(directory, align_directory, split_directory,
            ci_phones, num_jobs, fmllr = False):
    feat_name = 'cmvndeltafeats'
    if fmllr:
        feat_name += '_fmllr'
    mdl_path = os.path.join(align_directory, 'final.mdl')
    jobs = [ (directory, ci_phones, mdl_path,
            os.path.join(split_directory,str(x), feat_name),
            os.path.join(align_directory, 'ali.{}'.format(x)),x)
                for x in range(1, num_jobs + 1)]
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(tree_stats_func, args = i) for i in jobs]
        output = [p.get() for p in results]

    tree_accs = [os.path.join(directory, '{}.treeacc'.format(x)) for x in range(1, num_jobs+1)]
    log_path = os.path.join(directory, 'log', 'sum_tree_acc.log')
    with open(log_path, 'w') as logf:
        subprocess.call(['sum-tree-stats', os.path.join(directory, 'treeacc')] +
                        tree_accs, stderr = logf)
    for f in tree_accs:
        os.remove(f)

def convert_alignments_func(directory, align_directory, job_name):

    mdl_path = os.path.join(directory, '1.mdl')
    tree_path = os.path.join(directory, 'tree')
    ali_mdl_path = os.path.join(align_directory, 'final.mdl')
    ali_path = os.path.join(align_directory, 'ali.{}'.format(job_name))
    new_ali_path = os.path.join(directory, 'ali.{}'.format(job_name))


    log_path = os.path.join(directory, 'log', 'convert.{}.log'.format(job_name))
    with open(log_path, 'w') as logf:
        subprocess.call(['convert-ali', ali_mdl_path,
            mdl_path, tree_path, "ark:"+ali_path,
            "ark:"+new_ali_path], stderr = logf)

def convert_alignments(directory, align_directory, num_jobs):

    jobs = [ (directory, align_directory, x)
                for x in range(1, num_jobs + 1)]
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(convert_alignments_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def calc_fmllr_func(directory, split_directory, sil_phones, job_name, fmllr, model_name = 'final'):
    sdir = os.path.join(split_directory, str(job_name))
    feat_path = os.path.join(sdir, 'cmvndeltafeats')
    if fmllr:
        feat_path += '_fmllr'
    feat_fmllr_path = os.path.join(sdir, 'cmvndeltafeats_fmllr')
    log_path = os.path.join(directory, 'log', 'fmllr.{}.log'.format(job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    mdl_path = os.path.join(directory, '{}.mdl'.format(model_name))
    spk2utt_path = os.path.join(sdir, 'spk2utt')
    utt2spk_path = os.path.join(sdir, 'utt2spk')
    if fmllr:
        tmp_trans_path = os.path.join(directory, 'trans.temp.{}'.format(job_name))
        trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
        cmp_trans_path = os.path.join(directory, 'trans.cmp.{}'.format(job_name))
    else:
        tmp_trans_path = os.path.join(directory, 'trans.{}'.format(job_name))
    post_path = os.path.join(directory, 'post.{}'.format(job_name))
    weight_path = os.path.join(directory, 'weight.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        subprocess.call(['ali-to-post',
        "ark:"+ ali_path, 'ark:'+ post_path], stderr = logf)

        subprocess.call(['weight-silence-post', '0.0',
                            sil_phones, mdl_path, 'ark:'+post_path,
                            'ark:'+weight_path], stderr = logf)

        subprocess.call(['gmm-est-fmllr',
                '--verbose=4',
                '--fmllr-update-type={}'.format(fmllr_update_type),
      '--spk2utt=ark:'+spk2utt_path, mdl_path ,"ark,s,cs:"+feat_path,
      'ark,s,cs:'+weight_path, 'ark:'+tmp_trans_path],
            stderr = logf)

        if fmllr:
            subprocess.call(['compose-transforms', '--b-is-affine=true',
            'ark:'+ tmp_trans_path, 'ark:'+ trans_path,
            'ark:'+ cmp_trans_path], stderr = logf)
            os.remove(tmp_trans_path)
            os.remove(trans_path)
            os.rename(cmp_trans_path, trans_path)
            feat_path = os.path.join(sdir, 'cmvndeltafeats')
        else:
            trans_path = tmp_trans_path
        subprocess.call(['transform-feats',
                '--utt2spk=ark:'+utt2spk_path,
                'ark:'+trans_path, 'ark:'+feat_path,
                'ark:'+feat_fmllr_path],
                stderr = logf)


def calc_fmllr(directory, split_directory, sil_phones, num_jobs,
            fmllr = False, iteration = None):
    if iteration is None:
        model_name = 'final'
    else:
        model_name = iteration
    jobs = [ (directory, split_directory, sil_phones, x, fmllr, model_name)
                for x in range(1, num_jobs + 1)]
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(calc_fmllr_func, args = i) for i in jobs]
        output = [p.get() for p in results]
