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
                               calc_lda_mllt, gmm_gselect, acc_global_stats,
                               gauss_to_post, acc_ivector_stats, get_egs,
                               get_lda_nnet, nnet_train_trans, nnet_train)

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
        #log_dir = os.path.join(self.lda_mllt_directory, 'log')
        #os.makedirs(log_dir, exist_ok=True)

        ##
        config = self.lda_mllt_config
        directory = self.lda_mllt_directory
        align_directory = self.tri_fmllr_ali_directory  # The previous
        #align_directory = self.tri_ali_directory
        mdl_dir = self.tri_fmllr_directory
        #mdl_dir = self.lda_mllt_directory
        #if os.path.exists(os.path.join(directory, '1.mdl')):
        #    return
        print('Initializing LDA + MLLT training...')

        #self.corpus._norm_splice_transform_feats(self.lda_mllt_directory)

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
        log_path = os.path.join(directory, 'log', 'ali_to_post.log')
        with open(log_path, 'w') as logf:
            for i in range(self.num_jobs):
                spliced_feat_path = os.path.join(self.corpus.split_directory, 'cmvnsplicefeats.{}'.format(i))
                ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                    #'ark:gunzip -c ' + align_directory +
                                                    #'/ali.{}.gz|'.format(i),
                                                    'ark:' + align_directory + '/ali.{}'.format(i),
                                                    'ark:-'],
                                                    stderr=logf, stdout=subprocess.PIPE)
                weight_silence_post_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                            str(config.boost_silence), ci_phones,
                                                            align_directory +'/final.mdl',
                                                            #final_mdl_path,
                                                            'ark:-', 'ark:-'],
                                                            stdin=ali_to_post_proc.stdout,
                                                            stderr=logf, stdout=subprocess.PIPE)
                acc_lda_post_proc = subprocess.Popen([thirdparty_binary('acc-lda'),
                                                    '--rand-prune=' + str(config.randprune),
                                                    align_directory + '/final.mdl',
                                                    #final_mdl_path,
                                                    'ark:'+spliced_feat_path, # Unsure about this
                                                    'ark,s,cs:-',
                                                    directory + '/lda.{}.acc'.format(i)],
                                                    stdin=weight_silence_post_proc.stdout,
                                                    stderr=logf)
                acc_lda_post_proc.communicate()

        log_path = os.path.join(directory, 'log', 'lda_est.log')
        with open(log_path, 'w') as logf:
            for i in range(self.num_jobs):
                est_lda_proc = subprocess.Popen([thirdparty_binary('est-lda'),
                                                 '--write-full-matrix=' + directory + '/full.mat',
                                                 '--dim=' + str(config.dim),
                                                 directory + '/0.mat',
                                                 directory + '/lda.{}.acc'.format(i)],
                                                 stderr=logf)
                est_lda_proc.communicate()

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
        #occs_path = os.path.join(mdl_dir, '0.occs')
        mdl_path = os.path.join(directory, '0.mdl')
        #mdl_path = os.path.join(mdl_dir, '0.mdl')
        with open(log_path, 'w') as logf:
            subprocess.call([thirdparty_binary('gmm-init-model'),
                             '--write-occs=' + occs_path, tree_path, treeacc_path,
                             topo_path, mdl_path], stderr=logf)
        #print("!!!!", mdl_path, os.path.exists(os.path.join(mdl_dir, '0.mdl')))
        #print("!!!!", os.path.exists(os.path.join(mdl_dir, '0.occs')))

        #compile_train_graphs(directory, self.dictionary.output_directory,
        #                     self.corpus.split_directory, self.num_jobs)

        #os.rename(occs_path, os.path.join(directory, '1.occs')) # ?
        #os.rename(mdl_path, os.path.join(directory, '1.mdl'))   # ?
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
        #model_directory = self.lda_mllt_directory
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
        #if os.path.exists(self.lda_mllt_final_model_path):
        #    print('LDA + MLLT training already done, using previous final.mdl')
        #    return

        #if not os.path.exists(self.lta_mllt_ali_directory):
        #    self._align_lda_mllt()
        #self._align_lda_mllt()  # NOT implemented, can come back later or make people run from fmllr

        os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)

        #self.corpus._norm_splice_transform_feats(self.lda_mllt_directory, num=0)
        self._init_lda_mllt()   # Implemented!
        #self.corpus._norm_splice_transform_feats(self.lda_mllt_directory, num=0)
        self._do_lda_mllt_training()    # Implemented!

    def train_diag_ubm(self):
        '''
        Train a diagonal UBM on the LDA + MLLT model
        '''
        #if os.path.exists(self.diag_ubm_final_model_path):  # What actually is this?
        #    print('Diagonal UBM training already done; using previous model')
        #    return
        log_dir = os.path.join(self.diag_ubm_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        split_dir = self.corpus.split_directory
        train_dir = self.corpus.output_directory
        lda_mllt_path = self.lda_mllt_directory
        directory = self.diag_ubm_directory

        #cmvn_path = os.path.join(split_dir, 'cmvn.{}.scp'.format(i))
        cmvn_path = os.path.join(train_dir, 'cmvn.scp')

        old_config = self.lda_mllt_config
        config = self.diag_ubm_config
        ci_phones = self.dictionary.silence_csl

        final_mat_path = os.path.join(lda_mllt_path, 'final.mat')

        # Beginning code: will likely need to be refactored

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
                                                        #all_feats_path,
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

    def ivectors(self):
        '''
        Train iVector extractor and extract iVectors
        '''
        os.makedirs(os.path.join(self.ivector_extractor_directory, 'log'), exist_ok=True)
        self._train_ivector_extractor()

        # YET TO BE IMPLEMENTED
        #os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)
        #self._extract_ivectors()

        #if os.path.exists(self.ivector_extractor_final_model_path):
        #    print('iVector training already done, using previous final.mdl')
        #    return

        #if not os.path.exists(self.lta_mllt_ali_directory):
        #    self._align_lda_mllt()
        #self._align_lda_mllt()  # NOT implemented, can come back later or make people run from fmllr

        #os.makedirs(os.path.join(self.lda_mllt_directory, 'log'), exist_ok=True)

    def _train_ivector_extractor(self):
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

            # Accumulate stats
            acc_ivector_stats(directory, config, feats_path, self.num_jobs, i)

    def train_nnet_basic(self):
        os.makedirs(os.path.join(self.nnet_basic_directory, 'log'), exist_ok=True)

        split_directory = self.corpus.split_directory
        config = self.nnet_basic_config
        tri_fmllr_config = self.tri_fmllr_config
        directory = self.nnet_basic_directory
        align_directory = self.tri_fmllr_ali_directory  # For now-- could change if the preprocessing changes
        lda_directory = self.lda_mllt_directory
        egs_directory = os.path.join(directory, 'egs')
        training_directory = self.corpus.output_directory

        #tree_path = os.path.join(directory, 'tree')
        #treeacc_path = os.path.join(directory, 'treeacc')
        sets_int_path = os.path.join(self.dictionary.phones_dir, 'sets.int')
        roots_int_path = os.path.join(self.dictionary.phones_dir, 'roots.int')
        extra_question_int_path = os.path.join(self.dictionary.phones_dir, 'extra_questions.int')
        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        #questions_path = os.path.join(directory, 'questions.int')
        #questions_qst_path = os.path.join(directory, 'questions.qst')
        L_fst_path = os.path.join(self.dictionary.output_directory, 'L.fst')
        ali_tree_path = os.path.join(align_directory, 'tree')
        mdl_path = os.path.join(align_directory, 'final.mdl')
        raw_feats = os.path.join(training_directory, 'feats.scp')

        #num_leaves = tri_fmllr_config.initial_gauss_count # Assuming max number of leaves is the same as number of leaves
        num_leaves = 93 # Hard coded, from dimensions of spliced features
        #feat_dim = self.corpus.get_feat_dim() # Unspliced
        feat_dim = 13 # Hard coded, from MFCC dimensions
        lda_dim = 40 # Hard coded, could paramaterize this/make safer

        # Get LDA matrix
        get_lda_nnet(directory, align_directory, raw_feats, config, self.num_jobs)

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
        training_feats = os.path.join(split_directory, 'nnet_training_feats')

        with open(training_feats, 'w') as outf:
            copy_feats_proc = subprocess.Popen([thirdparty_binary('copy-feats'),
                                                'scp:' + os.path.join(training_directory, 'feats.scp'),
                                                'ark:'],
                                                stdout=outf)
            copy_feats_proc.communicate()
        get_egs(directory, egs_directory, training_directory, align_directory, training_feats, config, self.num_jobs)

        # Initialize neural net
        stddev = float(1.0/config.hidden_layer_dim**0.5)
        nnet_config_path = os.path.join(directory, 'nnet.config')
        hidden_config_path = os.path.join(directory, 'hidden.config')

        with open(nnet_config_path, 'w') as nc:
            print("feat dim:", feat_dim)
            nc.write('SpliceComponent input-dim={} left-context={} right-context={}\n'.format(13, config.splice_width, config.splice_width))
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

        # Train transition probabilities and set priors
        nnet_train_trans(directory, align_directory, self.num_jobs)

        # Training loop
        num_tot_iters = config.num_epochs * config.iters_per_epoch
        for i in range(num_tot_iters):
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

        # Rename the final model
        shutil.copy(os.path.join(directory, '{}.mdl'.format(num_tot_iters)), os.path.join(directory, 'final.mdl'))
