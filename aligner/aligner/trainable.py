
import os
import shutil
import subprocess
import re
from tqdm import tqdm

from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                            acc_stats, tree_stats, convert_alignments,
                             convert_ali_to_textgrids, calc_fmllr)

from ..exceptions import NoSuccessfulAlignments

from .base import BaseAligner

from ..archive import Archive

class TrainableAligner(BaseAligner):
    def save(self, path):
        directory, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        print(basename)
        archive = Archive.empty(basename)
        archive.add_triphone_model(self.tri_fmllr_directory)
        archive.add_dictionary(self.dictionary)
        os.makedirs(directory, exist_ok = True)
        basename, _ = os.path.splitext(directory)
        print(basename)
        print(archive.dump(basename))

    def do_tri_training(self):
        self.call_back('Beginning triphone training...')
        self._do_training(self.tri_directory, self.tri_config)

    def train_tri(self):
        if os.path.exists(self.tri_final_model_path):
            print('Triphone training already done, using previous final.mdl')
            return
        if not os.path.exists(self.mono_ali_directory):
            self.align_si()

        os.makedirs(os.path.join(self.tri_directory, 'log'), exist_ok = True)
        self.corpus.setup_splits(self.dictionary)
        self.init_tri(fmllr = False)
        self.do_tri_training()
        #convert_ali_to_textgrids(tri_directory, lang_directory, split_directory, num_jobs)

    def init_mono(self):
        log_dir = os.path.join(self.mono_directory, 'log')
        os.makedirs(log_dir, exist_ok =True)
        tree_path = os.path.join(self.mono_directory,'tree')
        mdl_path = os.path.join(self.mono_directory,'0.mdl')

        directory = self.corpus.split_directory
        feat_dim = self.corpus.get_feat_dim()
        path = os.path.join(directory, 'cmvndeltafeats.0_sub')
        feat_path = os.path.join(directory, 'cmvndeltafeats.0')
        shared_phones_opt = "--shared-phones=" + os.path.join(self.dictionary.phones_dir, 'sets.int')
        log_path = os.path.join(log_dir, 'log')
        with open(path, 'rb') as f, open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-mono'),shared_phones_opt,
                            "--train-feats=ark:-",
                            os.path.join(self.dictionary.output_directory,'topo'),
                            feat_dim,
                            mdl_path,
                            tree_path],
                            stdin = f,
                            stderr = logf)
        num_gauss = self.get_num_gauss_mono()
        compile_train_graphs(self.mono_directory, self.dictionary.output_directory,
                                self.corpus.split_directory, self.num_jobs)
        mono_align_equal(self.mono_directory, self.dictionary.output_directory,
                                self.corpus.split_directory, self.num_jobs)
        log_path = os.path.join(self.mono_directory, 'log', 'update.0.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(self.mono_directory, '0.{}.acc'.format(x)) for x in range(self.num_jobs)]
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                    '--min-gaussian-occupancy=3',
                    '--mix-up={}'.format(num_gauss), '--power={}'.format(self.mono_config.power),
                    mdl_path, "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                            ' '.join(map(make_path_safe, acc_files))),
                    os.path.join(self.mono_directory,'1.mdl')],
                    stderr = logf)
            est_proc.communicate()

    def do_mono_training(self):
        self.mono_config.num_gauss = self.get_num_gauss_mono()
        self.call_back('Beginning monophone training...')
        self._do_training(self.mono_directory, self.mono_config)

    def train_mono(self):
        final_mdl = os.path.join(self.mono_directory, 'final.mdl')
        split_directory = self.corpus.split_directory
        if os.path.exists(final_mdl):
            print('Monophone training already done, using previous final.mdl')
            return
        os.makedirs(os.path.join(self.mono_directory, 'log'), exist_ok = True)
        if not os.path.exists(split_directory):
            self.corpus.setup_splits(self.dictionary)

        self.init_mono()
        self.do_mono_training()
        #self.convert_ali_to_textgrids(mono_directory, lang_directory, split_directory, num_jobs)
