import os
import shutil
import subprocess
import re
import math
from tqdm import tqdm

from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (align, mono_align_equal, compile_train_graphs,
                               acc_stats, tree_stats, convert_alignments,
                               convert_ali_to_textgrids, calc_fmllr,
                               lda_acc_stats,
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train,
                               nnet_align, nnet_get_align_feats, extract_ivectors)

from ..exceptions import NoSuccessfulAlignments

from .base import BaseAligner

from ..models import AcousticModel


class TrainableAligner(BaseAligner):
    '''
    Aligner that aligns and trains acoustics models on a large dataset

    Parameters
    ----------
    corpus : :class:`~aligner.corpus.Corpus`
        Corpus object for the dataset
    dictionary : :class:`~aligner.dictionary.Dictionary`
        Dictionary object for the pronunciation dictionary
    output_directory : str
        Path to export aligned TextGrids
    temp_directory : str, optional
        Specifies the temporary directory root to save files need for Kaldi.
        If not specified, it will be set to ``~/Documents/MFA``
    num_jobs : int, optional
        Number of processes to use, defaults to 3
    call_back : callable, optional
        Specifies a call back function for alignment
    mono_params : :class:`~aligner.config.MonophoneConfig`, optional
        Monophone training parameters to use, if different from defaults
    tri_params : :class:`~aligner.config.TriphoneConfig`, optional
        Triphone training parameters to use, if different from defaults
    tri_fmllr_params : :class:`~aligner.config.TriphoneFmllrConfig`, optional
        Speaker-adapted triphone training parameters to use, if different from defaults
    '''

    def save(self, path):
        '''
        Output an acoustic model and dictionary to the specified path

        Parameters
        ----------
        path : str
            Path to save acoustic model and dictionary
        '''
        directory, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        acoustic_model = AcousticModel.empty(basename)
        acoustic_model.add_meta_file(self)
        #acoustic_model.add_triphone_model(self.tri_fmllr_directory)
        acoustic_model.add_triphone_fmllr_model(self.tri_fmllr_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        acoustic_model.dump(basename)
        print('Saved model to {}'.format(path))

    def _do_tri_training(self):
        self.call_back('Beginning triphone training...')
        self._do_training(self.tri_directory, self.tri_config)

    def train_tri(self):
        '''
        Perform triphone training
        '''
        # N.B.: Left commented out for development
        #if os.path.exists(self.tri_final_model_path):
        #    print('Triphone training already done, using previous final.mdl')
        #    return
        if not os.path.exists(self.mono_ali_directory):
            self._align_si()

        os.makedirs(os.path.join(self.tri_directory, 'log'), exist_ok=True)

        self._init_tri(fmllr=False)
        self._do_tri_training()

    def _init_mono(self):
        '''
        Initialize monophone training
        '''
        print("Initializing monophone training...")
        log_dir = os.path.join(self.mono_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        tree_path = os.path.join(self.mono_directory, 'tree')
        mdl_path = os.path.join(self.mono_directory, '0.mdl')

        directory = self.corpus.split_directory
        feat_dim = self.corpus.get_feat_dim()
        path = os.path.join(directory, 'cmvndeltafeats.0_sub')
        feat_path = os.path.join(directory, 'cmvndeltafeats.0')
        shared_phones_opt = "--shared-phones=" + os.path.join(self.dictionary.phones_dir, 'sets.int')
        log_path = os.path.join(log_dir, 'log')
        with open(path, 'rb') as f, open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-mono'), shared_phones_opt,
                             "--train-feats=ark:-",
                             os.path.join(self.dictionary.output_directory, 'topo'),
                             feat_dim,
                             mdl_path,
                             tree_path],
                            stdin=f,
                            stderr=logf)
        num_gauss = self.get_num_gauss_mono()
        compile_train_graphs(self.mono_directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)
        mono_align_equal(self.mono_directory,
                         self.corpus.split_directory, self.num_jobs)
        log_path = os.path.join(self.mono_directory, 'log', 'update.0.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(self.mono_directory, '0.{}.acc'.format(x)) for x in range(self.num_jobs)]
            est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                         '--min-gaussian-occupancy=3',
                                         '--mix-up={}'.format(num_gauss), '--power={}'.format(self.mono_config.power),
                                         mdl_path, "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                                     ' '.join(map(make_path_safe, acc_files))),
                                         os.path.join(self.mono_directory, '1.mdl')],
                                        stderr=logf)
            est_proc.communicate()

    def _do_mono_training(self):
        self.mono_config.initial_gauss_count = self.get_num_gauss_mono()
        self.call_back('Beginning monophone training...')
        self._do_training(self.mono_directory, self.mono_config)

    def train_mono(self):
        '''
        Perform monophone training
        '''
        final_mdl = os.path.join(self.mono_directory, 'final.mdl')
        # N.B.: Left commented out for development
        #if os.path.exists(final_mdl):
        #    print('Monophone training already done, using previous final.mdl')
        #    return
        os.makedirs(os.path.join(self.mono_directory, 'log'), exist_ok=True)

        self._init_mono()
        self._do_mono_training()

    # Beginning of nnet functions


    def _align_lda_mllt(self):
        '''
        Align the dataset using LDA + MLLT transforms
        '''
        log_dir = os.path.join(self.lda_mllt_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        feat_name = "cmvnsplicetransformfeats"
        model_directory = self.tri_fmllr_directory  # Get final.mdl from here
        output_directory = self.lda_mllt_ali_directory  # Alignments end up here
        self._align_si(fmllr=False, lda_mllt=True, feature_name=feat_name)
        sil_phones = self.dictionary.silence_csl

        log_dir = os.path.join(output_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)
        calc_lda_mllt(output_directory, self.corpus.split_directory,
                      self.tri_fmllr_directory,
                      sil_phones, self.num_jobs, self.lda_mllt_config,
                      self.lda_mllt_config.num_iters, initial=True)
        optional_silence = self.dictionary.optional_silence_csl
        align(0, model_directory, self.corpus.split_directory,
              optional_silence, self.num_jobs, self.lda_mllt_config, feature_name=feat_name)



    

    def train_diag_ubm(self):
        '''
        Train a diagonal UBM on the LDA + MLLT model
        '''
        # N.B.: Left commented out for development
        #if os.path.exists(self.diag_ubm_final_model_path):
        #    print('Diagonal UBM training already done; using previous model')
        #    return
        log_dir = os.path.join(self.diag_ubm_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        lda_mllt_path = self.lda_mllt_directory
        directory = self.diag_ubm_directory

        cmvn_path = os.path.join(train_dir, 'cmvn.scp')

        old_config = self.lda_mllt_config
        config = self.diag_ubm_config
        ci_phones = self.dictionary.silence_csl

        final_mat_path = os.path.join(lda_mllt_path, 'final.mat')

        # Create global_cmvn.stats
        log_path = os.path.join(directory, 'log', 'make_global_cmvn.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('matrix-sum'),
                            '--binary=false',
                            'scp:' + cmvn_path,
                             os.path.join(directory, 'global_cmvn.stats')],
                             stderr=logf)

        # Get all feats
        all_feats_path = os.path.join(split_dir, 'cmvnonlinesplicetransformfeats')
        log_path = os.path.join(split_dir, 'log', 'cmvnonlinesplicetransform.log')
        with open(log_path, 'w') as logf:
            with open(all_feats_path, 'wb') as outf:
                apply_cmvn_online_proc = subprocess.Popen([thirdparty_binary('apply-cmvn-online'),
                                                          #'--config=' +
                                                          # This^ makes reference to a config file
                                                          # in Kaldi, but it's empty there
                                                          os.path.join(directory, 'global_cmvn.stats'),
                                                          'scp:' + train_dir + '/feats.scp',
                                                          'ark:-'],
                                                          stdout=subprocess.PIPE,
                                                          stderr=logf)
                splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats')]
                                                     + config.splice_opts +
                                                     ['ark:-', 'ark:-'],
                                                     stdin=apply_cmvn_online_proc.stdout,
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                transform_feats_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                                        os.path.join(lda_mllt_path, 'final.mat'),
                                                        'ark:-', 'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=outf,
                                                        stderr=logf)
                transform_feats_proc.communicate()

        # Initialize model from E-M in memory
        num_gauss_init = int(config.initial_gauss_proportion * int(config.num_gauss))
        log_path = os.path.join(directory, 'log', 'gmm_init.log')
        with open(log_path, 'w') as logf:
            gmm_init_proc = subprocess.Popen([thirdparty_binary('gmm-global-init-from-feats'),
                                             '--num-threads=' + str(config.num_threads),
                                             '--num-frames=' + str(config.num_frames),
                                             '--num_gauss=' + str(config.num_gauss),
                                             '--num_gauss_init=' + str(num_gauss_init),
                                             '--num_iters=' + str(config.num_iters_init),
                                             'ark:' + all_feats_path,
                                             os.path.join(directory, '0.dubm')],
                                             stderr=logf)
            gmm_init_proc.communicate()

        # Get subset of all feats
        subsample_feats_path = os.path.join(split_dir, 'cmvnonlinesplicetransformsubsamplefeats')
        log_path = os.path.join(split_dir, 'log', 'cmvnonlinesplicetransformsubsample.log')
        with open(log_path, 'w') as logf:
            with open(all_feats_path, 'r') as inf, open(subsample_feats_path, 'wb') as outf:
                subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                        '--n=' + str(config.subsample),
                                                        'ark:-',
                                                        'ark:-'],
                                                        stdin=inf,
                                                        stdout=outf,
                                                        stderr=logf)
                subsample_feats_proc.communicate()


        # Store Gaussian selection indices on disk
        gmm_gselect(directory, config, subsample_feats_path, self.num_jobs)

        # Training
        for i in range(config.num_iters):
            # Accumulate stats
            acc_global_stats(directory, config, subsample_feats_path, self.num_jobs, i)

            # Don't remove low-count Gaussians till the last tier,
            # or gselect info won't be valid anymore
            if i < config.num_iters-1:
                opt = '--remove-low-count-gaussians=false'
            else:
                opt = '--remove-low-count-gaussians=' + str(config.remove_low_count_gaussians)

            log_path = os.path.join(directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.num_jobs)]
                gmm_global_est_proc = subprocess.Popen([thirdparty_binary('gmm-global-est'),
                                                        opt,
                                                        '--min-gaussian-weight=' + str(config.min_gaussian_weight),
                                                        os.path.join(directory, '{}.dubm'.format(i)),
                                                        "{} - {}|".format(thirdparty_binary('gmm-global-sum-accs'),
                                                                          ' '.join(map(make_path_safe, acc_files))),
                                                        os.path.join(directory, '{}.dubm'.format(i+1))],
                                                        stderr=logf)
                gmm_global_est_proc.communicate()

        # Move files
        shutil.copy(os.path.join(directory, '{}.dubm'.format(config.num_iters)),
                    os.path.join(directory, 'final.dubm'))

    def ivector_extractor(self):
        '''
        Train iVector extractor
        '''
        # N.B.: Left commented out for development
        #if os.path.exists(self.ivector_extractor_final_model_path):
        #    print('iVector training already done, using previous final.mdl')
        #    return
        os.makedirs(os.path.join(self.ivector_extractor_directory, 'log'), exist_ok=True)
        self._train_ivector_extractor()

    def _train_ivector_extractor(self):
        # N.B.: Left commented out for development
        #if os.path.exists(self.ivector_extractor_final_model_path):
        #    print('iVector extractor training already done, using previous final.ie')
        #    return

        log_dir = os.path.join(self.ivector_extractor_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        directory = self.ivector_extractor_directory
        split_dir = self.corpus.split_directory
        diag_ubm_path = self.diag_ubm_directory
        lda_mllt_path = self.lda_mllt_directory
        train_dir = self.corpus.output_directory
        config = self.ivector_extractor_config

        # Convert final.ubm to fgmm
        log_path = os.path.join(directory, 'log', 'global_to_fgmm.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-global-to-fgmm'),
                            os.path.join(diag_ubm_path, 'final.dubm'),
                            os.path.join(directory, '0.fgmm')],
                            stdout=subprocess.PIPE,
                            stderr=logf)

        # Initialize iVector extractor
        log_path = os.path.join(directory, 'log', 'init.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('ivector-extractor-init'),
                            '--ivector-dim=' + str(config.ivector_dim),
                            '--use-weights=false',
                            os.path.join(directory, '0.fgmm'),
                            os.path.join(directory, '0.ie')],
                            stderr=logf)

        # Get GMM feats with online CMVN
        gmm_feats_path = os.path.join(split_dir, 'ivectorgmmfeats')
        log_path = os.path.join(split_dir, 'log', 'ivectorgmmfeats.log')
        with open(log_path, 'w') as logf:
            with open(gmm_feats_path, 'wb') as outf:
                apply_cmvn_online_proc = subprocess.Popen([thirdparty_binary('apply-cmvn-online'),
                                                          #'--config=' +
                                                          # This^ makes reference to a config file
                                                          # in Kaldi, but it's empty there
                                                          os.path.join(diag_ubm_path, 'global_cmvn.stats'),
                                                          'scp:' + train_dir + '/feats.scp',
                                                          'ark:-'],
                                                          stdout=subprocess.PIPE,
                                                          stderr=logf)
                splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats')]
                                                     + config.splice_opts +
                                                     ['ark:-', 'ark:-'],
                                                     stdin=apply_cmvn_online_proc.stdout,
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                transform_feats_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                                        os.path.join(lda_mllt_path, 'final.mat'),
                                                        'ark:-', 'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
                subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                        '--n=' + str(config.subsample),
                                                        'ark:-', 'ark:-'],
                                                        stdin=transform_feats_proc.stdout,
                                                        stdout=outf,
                                                        stderr=logf)
                subsample_feats_proc.communicate()


        # Do Gaussian selection and posterior extraction
        gauss_to_post(directory, config, diag_ubm_path, gmm_feats_path, self.num_jobs)

        # Get GMM feats without online CMVN
        feats_path = os.path.join(split_dir, 'ivectorfeats')
        log_path = os.path.join(split_dir, 'log', 'ivectorfeats.log')
        with open(log_path, 'w') as logf:
            with open(feats_path, 'wb') as outf:
                splice_feats_proc = subprocess.Popen([thirdparty_binary('splice-feats')]
                                                     + config.splice_opts +
                                                     ['scp:' + os.path.join(train_dir, 'feats.scp'),
                                                     'ark:-'],
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                transform_feats_proc = subprocess.Popen([thirdparty_binary('transform-feats'),
                                                        os.path.join(lda_mllt_path, 'final.mat'),
                                                        'ark:-', 'ark:-'],
                                                        stdin=splice_feats_proc.stdout,
                                                        stdout=subprocess.PIPE,
                                                        stderr=logf)
                subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                        '--n=' + str(config.subsample),
                                                        'ark:-', 'ark:-'],
                                                        stdin=transform_feats_proc.stdout,
                                                        stdout=outf,
                                                        stderr=logf)
                subsample_feats_proc.communicate()

        # Training loop
        for i in range(config.num_iters):

            # Accumulate stats and sum
            acc_ivector_stats(directory, config, feats_path, self.num_jobs, i)

            # Est extractor
            log_path = os.path.join(directory, 'log', 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                extractor_est_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-est'),
                                                      os.path.join(directory, '{}.ie'.format(i)),
                                                      os.path.join(directory, 'acc.{}'.format(i)),
                                                      os.path.join(directory, '{}.ie'.format(i+1))],
                                                      stderr=logf)
                extractor_est_proc.communicate()
        # Rename to final
        shutil.copy(os.path.join(directory, '{}.ie'.format(config.num_iters)), os.path.join(directory, 'final.ie'))
