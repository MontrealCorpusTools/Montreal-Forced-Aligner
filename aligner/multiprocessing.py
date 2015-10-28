
import multiprocessing as mp
import subprocess
import os

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
