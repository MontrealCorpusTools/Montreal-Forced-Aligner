
import multiprocessing as mp
import subprocess
import os

from .prep.helper import load_text

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

def mono_realign_func(mono_directory, iteration, job_name, mdl, scale_opts, num_gauss, feat_path):
    if iteration == 1:
        beam = 6
    else:
        beam = 10
    fst_path = os.path.join(mono_directory, 'fsts.{}'.format(job_name))
    log_path = os.path.join(mono_directory, 'log', 'align.{}.{}.log'.format(iteration, job_name))
    ali_path = os.path.join(mono_directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        align_proc = subprocess.Popen(['gmm-align-compiled']+ scale_opts + \
            ['--beam={}'.format(beam),
            '--retry-beam={}'.format(beam*4), '--careful=false', mdl,
        "ark:"+fst_path, "ark:"+feat_path, "ark,t:"+ ali_path],
        stderr = logf)
        align_proc.communicate()

def mono_acc_stats_func(mono_directory, iteration, job_name, feat_path):
    log_path = os.path.join(mono_directory, 'log', 'acc.{}.{}.log'.format(iteration, job_name))
    model_path = os.path.join(mono_directory,'{}.mdl'.format(iteration))
    next_model_path = os.path.join(mono_directory,'{}.mdl'.format(iteration+1))
    acc_path = os.path.join(mono_directory,'{}.{}.acc'.format(iteration, job_name))
    ali_path = os.path.join(mono_directory, 'ali.{}'.format(job_name))
    with open(log_path, 'w') as logf:
        acc_proc = subprocess.Popen(['gmm-acc-stats-ali', model_path,
             "ark:"+feat_path, "ark,t:" + ali_path,
          acc_path],
          stderr = logf)
        acc_proc.communicate()

def mono_acc_stats(iteration, mono_directory, split_directory, num_jobs):
    jobs = [ (mono_directory, iteration, x, os.path.join(split_directory,str(x), 'cmvndeltafeats'))
                for x in range(1, num_jobs + 1)]
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(mono_acc_stats_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def mono_align_equal_func(mono_directory, lang_directory, split_directory, job_name, feat_path):
    fst_path = os.path.join(mono_directory, 'fsts.{}'.format(job_name))
    tree_path = os.path.join(mono_directory,'tree')
    mdl_path = os.path.join(mono_directory,'0.mdl')
    directory = os.path.join(split_directory, str(job_name))
    log_path = os.path.join(mono_directory, 'log', 'compile-graphs.0.{}.log'.format(job_name))
    with open(os.path.join(directory,'text.int'), 'r') as inf, \
        open(fst_path, 'wb') as outf, \
        open(log_path, 'w') as logf:
        proc = subprocess.Popen(['compile-train-graphs',
                    tree_path, mdl_path,
                    os.path.join(lang_directory,'L.fst'),
                    "ark:-", "ark:-"],
                    stdin = inf, stdout = outf, stderr = logf)
        proc.communicate()
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
    print(num_jobs)
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(mono_align_equal_func, args = i) for i in jobs]
        output = [p.get() for p in results]
    mono_acc_stats(0, mono_directory, split_directory, num_jobs)

def mono_realign(iteration, mono_directory, split_directory,
                lang_directory, scale_opts, num_jobs, boost_silence, num_gauss):
    optional_silence = load_text(os.path.join(lang_directory, 'phones', 'optional_silence.csl'))
    mdl_path = os.path.join(mono_directory, '{}.mdl'.format(iteration))
    mdl="gmm-boost-silence --boost={} {} {} - |".format(boost_silence, optional_silence, mdl_path)

    jobs = [ (mono_directory, iteration, x, mdl, scale_opts, num_gauss, os.path.join(split_directory,str(x), 'cmvndeltafeats'))
                for x in range(1, num_jobs + 1)]
    print(num_jobs)
    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(mono_realign_func, args = i) for i in jobs]
        output = [p.get() for p in results]

def ali_to_textgrid_func(directory, lang_directory, split_directory, job_name):
    text_int_path = os.path.join(split_directory, str(job_name), 'text.int')
    log_path = os.path.join(directory, 'log', 'get_ctm_align.{}.log'.format(job_name))
    ali_path = os.path.join(directory, 'ali.{}'.format(job_name))
    model_path = os.path.join(directory, 'final.mdl')
    aligned_path = os.path.join(directory, 'aligned.{}'.format(job_name))
    word_ctm_path = os.path.join(directory, 'word_ctm.{}'.format(job_name))
    phone_ctm_path = os.path.join(directory, 'phone_ctm.{}'.format(job_name))
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


def convert_ali_to_textgrids(directory, lang_directory, split_directory, num_jobs):

    jobs = [ (directory, lang_directory, split_directory, x)
                for x in range(1, num_jobs + 1)]

    with mp.Pool(processes = num_jobs) as pool:
        results = [pool.apply_async(ali_to_textgrid_func, args = i) for i in jobs]
        output = [p.get() for p in results]
