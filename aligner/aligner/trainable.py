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
    def _init_lda_mllt(self):
        '''
        Initialize LDA + MLLT training.
        '''
        config = self.lda_mllt_config
        directory = self.lda_mllt_directory
        align_directory = self.tri_fmllr_ali_directory  # The previous
        mdl_dir = self.tri_fmllr_directory
        # N.B.: Left commented out for development
        #if os.path.exists(os.path.join(directory, '1.mdl')):
        #    return
        print('Initializing LDA + MLLT training...')

        context_opts = []
        ci_phones = self.dictionary.silence_csl

        log_path = os.path.join(directory, 'log', 'questions.log')
        tree_path = os.path.join(directory, 'tree')
        treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        questions_path = os.path.join(directory, 'questions.int')
        questions_qst_path = os.path.join(directory, 'questions.qst')

        final_mdl_path = os.path.join(self.tri_fmllr_directory)

        # Accumulate LDA stats
        lda_acc_stats(directory, self.corpus.split_directory, align_directory, config, ci_phones, self.num_jobs)

        # Accumulating tree stats
        self.corpus._norm_splice_transform_feats(self.lda_mllt_directory)
        tree_stats(directory, align_directory, self.corpus.split_directory, ci_phones,
                   self.num_jobs, feature_name='cmvnsplicetransformfeats')

        # Getting questions for tree clustering
        log_path = os.path.join(directory, 'log', 'cluster_phones.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('cluster-phones')] + context_opts +
                            [treeacc_path, sets_int_path, questions_path], stderr=logf)

        with open(extra_question_int_path, 'r') as inf, \
                open(questions_path, 'a') as outf:
            for line in inf:
                outf.write(line)

        log_path = os.path.join(directory, 'log', 'compile_questions.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('compile-questions')] + context_opts +
                            [topo_path, questions_path, questions_qst_path],
                            stderr=logf)

        # Building the tree
        log_path = os.path.join(directory, 'log', 'build_tree.log')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('build-tree')] + context_opts +
                            ['--verbose=1', '--max-leaves={}'.format(config.initial_gauss_count),
                             '--cluster-thresh={}'.format(config.cluster_threshold),
                             treeacc_path, roots_int_path, questions_qst_path,
                             topo_path, tree_path], stderr=logf)

        # Initializing the model
        log_path = os.path.join(directory, 'log', 'init_model.log')
        occs_path = os.path.join(directory, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)
        shutil.copy(mdl_path, os.path.join(directory, '1.mdl'))
        shutil.copy(occs_path, os.path.join(directory, '1.occs'))

        convert_alignments(directory, align_directory, self.num_jobs)

        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        if os.path.exists(os.path.join(align_directory, 'trans.0')):            # ?
            for i in range(self.num_jobs):                                      # ?
                shutil.copy(os.path.join(align_directory, 'trans.{}'.format(i)),# ?
                            os.path.join(directory, 'trans.{}'.format(i)))      # ?

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

    def _do_lda_mllt_training(self):
        self.call_back('Beginning LDA + MLLT training...')
        self._do_training(self.lda_mllt_directory, self.lda_mllt_config)

    def train_lda_mllt(self):
        '''
        Perform LDA + MLLT training
        '''
        # N.B.: Left commented out for development
        #if os.path.exists(self.lda_mllt_final_model_path):
        #    print('LDA + MLLT training already done, using previous final.mdl')
        #    return

        # N.B: The function _align_lda_mllt() is half-developed, but there doesn't seem to
        # be a reason for it to actually ever be called (since people will always have
        # fmllr done immediately before in the pipeline. Can clean/delete later if determined
        # that we need to actually use it somewhere or not).
        #if not os.path.exists(self.lda_mllt_ali_directory):
        #    self._align_lda_mllt()
        #self._align_lda_mllt()  # half implemented, can come back later or make people run from fmllr

        os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)

        self._init_lda_mllt()
        self._do_lda_mllt_training()

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

    def _extract_ivectors(self):
        # N.B.: Left commented out for development
        #if os.path.exists(self.ivector_extractor_final_model_path):
        #    print('iVector extractor training already done, using previous final.ie')
        #    return

        log_dir = os.path.join(self.extracted_ivector_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        # N.B.: These paths are hacky and need to be correctly integrated
        directory = self.extracted_ivector_directory
        ivector_extractor_dir = "/data/acoles/acoles/Montreal-Forced-Aligner/ivector_extractor"
        #ivector_extractor_dir = "../../ivector_extractor"
        #ivector_extractor_dir = "/Users/mlml/Documents/GitHub/Montreal-Forced-Aligner/ivector_extractor"
        diag_ubm_path = "/data/acoles/acoles/Montreal-Forced-Aligner/diag_ubm"
        #diag_ubm_path = "../../diag_ubm"
        #diag_ubm_path = "/Users/mlml/Documents/GitHub/Montreal-Forced-Aligner/diag_ubm"
        lda_mllt_path = "/data/acoles/acoles/Montreal-Forced-Aligner/lda_mllt"
        #lda_mllt_path = "../../lda_mllt"
        #lda_mllt_path = "/Users/mlml/Documents/GitHub/Montreal-Forced-Aligner/lda_mllt"
        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        config = self.ivector_extractor_config
        training_directory = self.corpus.output_directory

        # Need to make a directory for corpus with just 2 utterances per speaker
        # (left commented out in case we ever decide to do this)
        #max2_dir = os.path.join(directory, 'max2')
        #os.makedirs(max2_dir, exist_ok=True)
        #mfa_working_dir = os.getcwd()
        #os.chdir("/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2")
        #opy_data_sh = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/steps/online/nnet2/copy_data_dir.sh"
        #log_path = os.path.join(directory, 'log', 'max2.log')
        #with open(log_path, 'w') as logf:
        #    command = [copy_data_sh, '--utts-per-spk-max', '2', train_dir, max2_dir]
        #    max2_proc = subprocess.Popen(command,
        #                                 stderr=logf)
        #    max2_proc.communicate()
        #os.chdir(mfa_working_dir)

        # Write a "cmvn config" file (this is blank in the actual kaldi code, but it needs the argument passed)
        cmvn_config = os.path.join(directory, 'online_cmvn.conf')
        with open(cmvn_config, 'w') as cconf:
            cconf.write("")

        # Write a "splice config" file
        splice_config = os.path.join(directory, 'splice.conf')
        with open(splice_config, 'w') as sconf:
            sconf.write(config.splice_opts[0])
            sconf.write('\n')
            sconf.write(config.splice_opts[1])

        # Write a "config" file to input to the extraction binary
        ext_config = os.path.join(directory, 'ivector_extractor.conf')
        with open(ext_config, 'w') as ieconf:
            ieconf.write('--cmvn-config={}\n'.format(cmvn_config))
            ieconf.write('--ivector-period={}\n'.format(config.ivector_period))
            ieconf.write('--splice-config={}\n'.format(splice_config))
            ieconf.write('--lda-matrix={}\n'.format(os.path.join(lda_mllt_path, 'final.mat')))
            ieconf.write('--global-cmvn-stats={}\n'.format(os.path.join(diag_ubm_path, 'global_cmvn.stats')))
            ieconf.write('--diag-ubm={}\n'.format(os.path.join(diag_ubm_path, 'final.dubm')))
            ieconf.write('--ivector-extractor={}\n'.format(os.path.join(ivector_extractor_dir, 'final.ie')))
            ieconf.write('--num-gselect={}\n'.format(config.num_gselect))
            ieconf.write('--min-post={}\n'.format(config.min_post))
            ieconf.write('--posterior-scale={}\n'.format(config.posterior_scale))
            ieconf.write('--max-remembered-frames=1000\n')
            ieconf.write('--max-count={}\n'.format(0))

        # Extract iVectors
        extract_ivectors(directory, training_directory, ext_config, config, self.num_jobs)

        # Combine iVectors across jobs
        file_list = []
        for j in range(self.num_jobs):
            file_list.append(os.path.join(directory, 'ivector_online.{}.scp'.format(j)))
        print("file list:", file_list)

        with open(os.path.join(directory, 'ivector_online.scp'), 'w') as outfile:
            for fname in file_list:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

    def train_nnet_basic(self):
        os.makedirs(os.path.join(self.nnet_basic_directory, 'log'), exist_ok=True)

        split_directory = self.corpus.split_directory
        config = self.nnet_basic_config
        tri_fmllr_config = self.tri_fmllr_config
        directory = self.nnet_basic_directory
        nnet_align_directory = self.nnet_basic_ali_directory
        align_directory = self.tri_fmllr_ali_directory
        lda_directory = self.lda_mllt_directory
        egs_directory = os.path.join(directory, 'egs')
        training_directory = self.corpus.output_directory

        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        L_fst_path = os.path.join(self.dictionary.output_directory, 'L.fst')
        ali_tree_path = os.path.join(align_directory, 'tree')
        shutil.copy(ali_tree_path, os.path.join(directory, 'tree'))

        mdl_path = os.path.join(align_directory, 'final.mdl')
        raw_feats = os.path.join(training_directory, 'feats.scp')

        tree_info_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                                          os.path.join(align_directory, 'tree')],
                                          stdout=subprocess.PIPE)
        tree_info = tree_info_proc.stdout.read()
        tree_info = tree_info.split()
        num_leaves = tree_info[1]
        num_leaves = num_leaves.decode("utf-8")

        lda_dim = 40 # Hard coded, could paramaterize this/make safer, but it's always 40 for LDA

        # Extract iVectors
        self._extract_ivectors()

        # Get LDA matrix
        fixed_ivector_dir = self.extracted_ivector_directory
        print("FIXED IVECTOR DIR:", fixed_ivector_dir)
        get_lda_nnet(directory, align_directory, fixed_ivector_dir, training_directory,
                     split_directory, raw_feats, self.dictionary.optional_silence_csl, config, self.num_jobs)

        log_path = os.path.join(directory, 'log', 'lda_matrix.log')
        with open(log_path, 'w') as logf:
            acc_files = [os.path.join(directory, 'lda.{}.acc'.format(x))
                         for x in range(self.num_jobs)]
            sum_lda_accs_proc = subprocess.Popen([thirdparty_binary('sum-lda-accs'),
                                                 os.path.join(directory, 'lda.acc')]
                                                 + acc_files,
                                                 stderr=logf)
            sum_lda_accs_proc.communicate()

            lda_mat_proc = subprocess.Popen([thirdparty_binary('nnet-get-feature-transform'),
                                            '--dim=' + str(lda_dim),
                                            os.path.join(directory, 'lda.mat'),
                                            os.path.join(directory, 'lda.acc')],
                                            stderr=logf)
            lda_mat_proc.communicate()
        lda_mat_path = os.path.join(directory, 'lda.mat')


        # Get examples for training
        os.makedirs(egs_directory, exist_ok=True)

        # # Get valid uttlist and train subset uttlist
        # (same issue with hacky paths throughout)
        #shuffle_list_path = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/utils/shuffle_list.pl"
        shuffle_list_path = "/data/acoles/acoles/kaldi/egs/wsj/s5/utils/shuffle_list.pl"
        #filter_scp_path = "/Users/mlml/Documents/Project/kaldi2/egs/wsj/s5/utils/filter_scp.pl"
        filter_scp_path = "/data/acoles/acoles/kaldi/egs/wsj/s5/utils/filter_scp.pl"
        valid_uttlist = os.path.join(directory, 'valid_uttlist')
        train_subset_uttlist = os.path.join(directory, 'train_subset_uttlist')
        training_feats = os.path.join(directory, 'nnet_training_feats')
        num_utts_subset = 300
        log_path = os.path.join(directory, 'log', 'training_egs_feats.log')
        with open(log_path, 'w') as logf:
            with open(valid_uttlist, 'w') as outf:
                valid_uttlist_proc = subprocess.Popen(['awk', '{print $1}',
                                                      os.path.join(training_directory, 'utt2spk')],
                                                      stdout=subprocess.PIPE,
                                                      stderr=logf)
                shuffle_list_proc = subprocess.Popen([shuffle_list_path],
                                                     stdin=valid_uttlist_proc.stdout,
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                head_proc = subprocess.Popen(['head', '-{}'.format(num_utts_subset)],
                                             stdin=shuffle_list_proc.stdout,
                                             stdout=outf,
                                             stderr=logf)
                head_proc.communicate()

            with open(train_subset_uttlist, 'w') as outf:
                awk_proc = subprocess.Popen(['awk', '{print $1}',
                                             os.path.join(training_directory, 'utt2spk')],
                                             stdout=subprocess.PIPE,
                                             stderr=logf)
                filter_scp_proc = subprocess.Popen([filter_scp_path,
                                                   '--exclude', valid_uttlist],
                                                   stdin=awk_proc.stdout,
                                                   stdout=subprocess.PIPE,
                                                   stderr=logf)
                shuffle_list_proc = subprocess.Popen([shuffle_list_path],
                                                     stdin=filter_scp_proc.stdout,
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                head_proc = subprocess.Popen(['head', '-{}'.format(num_utts_subset)],
                                             stdin=shuffle_list_proc.stdout,
                                             stdout=outf,
                                             stderr=logf)
                head_proc.communicate()

        get_egs(directory, egs_directory, training_directory, split_directory, align_directory,
                fixed_ivector_dir, training_feats, valid_uttlist,
                train_subset_uttlist, config, self.num_jobs)


        # Initialize neural net
        stddev = float(1.0/config.hidden_layer_dim**0.5)
        nnet_config_path = os.path.join(directory, 'nnet.config')
        hidden_config_path = os.path.join(directory, 'hidden.config')
        ivector_dim_path = os.path.join(directory, 'ivector_dim')
        with open(ivector_dim_path, 'r') as inf:
            ivector_dim = inf.read().strip()
        print("here is the ivector dim:", ivector_dim)
        feat_dim = 13 + int(ivector_dim)

        with open(nnet_config_path, 'w') as nc:
            print("feat dim:", feat_dim)
            nc.write('SpliceComponent input-dim={} left-context={} right-context={} const-component-dim={}\n'.format(feat_dim, config.splice_width, config.splice_width, ivector_dim))
            nc.write('FixedAffineComponent matrix={}\n'.format(lda_mat_path))
            nc.write('AffineComponent input-dim={} output-dim={} learning-rate={} param-stddev={} bias-stddev={}\n'.format(lda_dim, config.hidden_layer_dim, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('TanhComponent dim={}\n'.format(config.hidden_layer_dim))
            nc.write('AffineComponent input-dim={} output-dim={} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.hidden_layer_dim, num_leaves, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('SoftmaxComponent dim={}\n'.format(num_leaves))

        with open(hidden_config_path, 'w') as nc:
            nc.write('AffineComponent input-dim={} output-dim={} learning-rate={} param-stddev={} bias-stddev={}\n'.format(config.hidden_layer_dim, config.hidden_layer_dim, config.initial_learning_rate, stddev, config.bias_stddev))
            nc.write('TanhComponent dim={}\n'.format(config.hidden_layer_dim))

        log_path = os.path.join(directory, 'log', 'nnet_init.log')
        with open(log_path, 'w') as logf:
            nnet_am_init_proc = subprocess.Popen([thirdparty_binary('nnet-am-init'),
                                                 os.path.join(align_directory, 'tree'),
                                                 topo_path,
                                                 "{} {} -|".format(thirdparty_binary('nnet-init'),
                                                                   nnet_config_path),
                                                os.path.join(directory, '0.mdl')],
                                                stderr=logf)
            nnet_am_init_proc.communicate()

            nnet_am_info = subprocess.Popen([thirdparty_binary('nnet-am-info'),
                                            os.path.join(directory, '0.mdl')],
                                            stderr=logf)
            nnet_am_info.communicate()


        # Train transition probabilities and set priors
        nnet_train_trans(directory, align_directory, self.num_jobs)

        # Get iteration at which we will mix up
        num_tot_iters = config.num_epochs * config.iters_per_epoch
        finish_add_layers_iter = config.num_hidden_layers * config.add_layers_period
        first_modify_iter = finish_add_layers_iter + config.add_layers_period
        mix_up_iter = (num_tot_iters + finish_add_layers_iter)/2

        # Training loop
        for i in range(num_tot_iters):
            model_path = os.path.join(directory, '{}.mdl'.format(i))
            next_model_path = os.path.join(directory, '{}.mdl'.format(i + 1))

            # If it is NOT the first iteration,
            # AND we still have layers to add,
            # AND it's the right time to add a layer...
            if i > 0 and i <= ((config.num_hidden_layers-1)*config.add_layers_period) and ((i-1)%config.add_layers_period) == 0:
                # Add a new hidden layer
                mdl = os.path.join(directory, 'tmp{}.mdl'.format(i))
                log_path = os.path.join(directory, 'log', 'temp_mdl.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    with open(mdl, 'w') as outf:
                        tmp_mdl_init_proc = subprocess.Popen([thirdparty_binary('nnet-init'),
                                                            '--srand={}'.format(i),
                                                            os.path.join(directory, 'hidden.config'),
                                                            '-'],
                                                            stdout=subprocess.PIPE,
                                                            stderr=logf)
                        tmp_mdl_ins_proc = subprocess.Popen([thirdparty_binary('nnet-insert'),
                                                            os.path.join(directory, '{}.mdl'.format(i)),
                                                            '-', '-'],
                                                            stdin=tmp_mdl_init_proc.stdout,
                                                            stdout=outf,
                                                            stderr=logf)
                        tmp_mdl_ins_proc.communicate()

            # Otherwise just use the past model
            else:
                mdl = os.path.join(directory, '{}.mdl'.format(i))

            # Shuffle examples and train nets with SGD
            nnet_train(directory, egs_directory, mdl, i, self.num_jobs)

            # Get nnet list from the various jobs on this iteration
            nnets_list = [os.path.join(directory, '{}.{}.mdl'.format((i+1), x))
                         for x in range(self.num_jobs)]

            if (i+1) >= num_tot_iters:
                learning_rate = config.final_learning_rate
            else:
                learning_rate = config.initial_learning_rate * math.exp(i * math.log(config.final_learning_rate/config.initial_learning_rate)/num_tot_iters)

            log_path = os.path.join(directory, 'log', 'average.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                nnet_avg_proc = subprocess.Popen([thirdparty_binary('nnet-am-average')]
                                                 + nnets_list
                                                 + ['-'],
                                                 stdout=subprocess.PIPE,
                                                 stderr=logf)
                nnet_copy_proc = subprocess.Popen([thirdparty_binary('nnet-am-copy'),
                                                  '--learning-rate={}'.format(learning_rate),
                                                  '-',
                                                  os.path.join(directory, '{}.mdl'.format(i+1))],
                                                  stdin=nnet_avg_proc.stdout,
                                                  stderr=logf)
                nnet_copy_proc.communicate()

            # If it's the right time, do mixing up
            if config.mix_up > 0 and i == mix_up_iter:
                log_path = os.path.join(directory, 'log', 'mix_up.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_am_mixup_proc = subprocess.Popen([thirdparty_binary('nnet-am-mixup'),
                                                          '--min-count=10',
                                                          '--num-mixtures={}'.format(config.mix_up),
                                                          os.path.join(directory, '{}.mdl'.format(i+1)),
                                                          os.path.join(directory, '{}.mdl'.format(i+1))],
                                                          stderr=logf)
                    nnet_am_mixup_proc.communicate()

        # Rename the final model
        shutil.copy(os.path.join(directory, '{}.mdl'.format(num_tot_iters-1)), os.path.join(directory, 'final.mdl'))

        # Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(directory, self.dictionary.output_directory,
                             self.corpus.split_directory, self.num_jobs)

        # Get alignment feats
        nnet_get_align_feats(directory, self.corpus.split_directory, lda_directory, fixed_ivector_dir, config, self.num_jobs)

        # Do alignment
        nnet_align("final", directory,
              self.dictionary.optional_silence_csl,
              self.num_jobs, config, mdl=os.path.join(directory, 'final.mdl'))
