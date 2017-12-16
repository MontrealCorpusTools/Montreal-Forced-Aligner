import os
import re
from tqdm import tqdm
import subprocess
import shutil

from .base import BaseTrainer
from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               compile_information)


class MonophoneTrainer(BaseTrainer):
    '''
    Configuration class for monophone training


    Attributes
    ----------
    num_iterations : int
        Number of training iterations to perform, defaults to 40
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    gaussian_increment : int
        Last iter to increase #Gauss on, defaults to 30
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realignment_iterations : list
        List of iterations to perform alignment
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    '''

    def __init__(self):
        super(MonophoneTrainer, self).__init__()
        self.initial_gaussians = 135
        for i in range(1, self.num_iterations):
            if i <= int(self.num_iterations / 4):
                self.realignment_iterations.append(i)
            elif i <= int(self.num_iterations * 2 / 4):
                if i - self.realignment_iterations[-1] > 1:
                    self.realignment_iterations.append(i)
            else:
                if i - self.realignment_iterations[-1] > 2:
                    self.realignment_iterations.append(i)

    @property
    def train_type(self):
        return 'mono'

    @property
    def phone_type(self):
        return 'monophone'

    def get_num_gauss(self):
        '''
        Get the number of gaussians for a monophone model
        '''
        with open(os.devnull, 'w') as devnull:
            proc = subprocess.Popen([thirdparty_binary('gmm-info'),
                                     '--print-args=false',
                                     os.path.join(self.train_directory, '0.mdl')],
                                    stderr=devnull,
                                    stdout=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            num = stdout.decode('utf8')
            matches = re.search(r'gaussians (\d+)', num)
            num = int(matches.groups()[0])
        return num

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer=None):
        super(MonophoneTrainer, self).init_training(identifier, temporary_directory, corpus, dictionary, previous_trainer)

        tree_path = os.path.join(self.train_directory, 'tree')
        mdl_path = os.path.join(self.train_directory, '0.mdl')

        data_directory = corpus.split_directory(subset=self.subset)
        feat_dim = corpus.get_feat_dim()
        feat_path = os.path.join(data_directory, 'cmvndeltafeats.0')
        shared_phones_opt = "--shared-phones=" + os.path.join(dictionary.phones_dir, 'sets.int')
        log_path = os.path.join(self.log_directory, 'init.log')
        with open(feat_path, 'rb') as f, open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-mono'), shared_phones_opt,
                             "--train-feats=ark:-",
                             os.path.join(dictionary.output_directory, 'topo'),
                             feat_dim,
                             mdl_path,
                             tree_path],
                            stdin=f,
                            stderr=logf)
        num_gauss = self.get_num_gauss()
        self.initial_gaussians = num_gauss
        compile_train_graphs(self.train_directory, dictionary.output_directory,
                             data_directory, corpus.num_jobs)
        mono_align_equal(self.train_directory,
                         data_directory, corpus.num_jobs)
        log_path = os.path.join(self.train_directory, 'log', 'update.0.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(self.train_directory, '0.{}.acc'.format(x)) for x in range(corpus.num_jobs)]
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                         '--min-gaussian-occupancy=3',
                                         '--mix-up={}'.format(num_gauss), '--power={}'.format(self.power),
                                         mdl_path, "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                                     ' '.join(map(make_path_safe, acc_files))),
                                         os.path.join(self.train_directory, '1.mdl')],
                                        stderr=logf)
            est_proc.communicate()

