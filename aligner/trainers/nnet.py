import os
import math
import glob
from tqdm import tqdm
import subprocess
import shutil
from random import shuffle

from .base import BaseTrainer
from ..models import AcousticModel
from ..helper import thirdparty_binary, filter_scp

from ..multiprocessing import (get_lda_nnet, get_egs, nnet_train_trans, compile_train_graphs,
                               nnet_align, nnet_train, relabel_egs, get_average_posteriors,
                               compute_alignment_improvement)


class NnetTrainer(BaseTrainer):
    """
    Configuration class for neural network training

    Attributes
    ----------
    num_epochs : int
        Number of epochs of training; number of iterations is worked out from this
    iters_per_epoch : int
        Number of iterations per epoch
    realign_times : int
        How many times to realign during training; this will equally space them over the iterations
    beam : int
        Default beam width for alignment
    retry_beam : int
        Beam width to fall back on if no alignment is produced
    initial_learning_rate : float
        The initial learning rate at the beginning of training
    final_learning_rate : float
        The final learning rate by the end of training
    pnorm_input_dim : int
        The input dimension of the pnorm component
    pnorm_output_dim : int
        The output dimension of the pnorm component
    p : int
        Pnorm parameter
    hidden_layer_dim : int
        Dimension of a hidden layer
    samples_per_iter : int
        Number of samples seen per job per each iteration; used when getting examples
    shuffle_buffer_size : int
        This "buffer_size" variable controls randomization of the samples on each iter.  You could set it to 0 or to a large value for complete randomization, but this would both consume memory and cause spikes in disk I/O.  Smaller is easier on disk and memory but less random.  It's not a huge deal though, as samples are anyway randomized right at the start. (the point of this is to get data in different minibatches on different iterations, since in the preconditioning method, 2 samples in the same minibatch can affect each others' gradients.
    add_layers_period : int
        Number of iterations between addition of a new layer
    num_hidden_layers : int
        Number of hidden layers
    randprune : float
        Speeds up LDA
    alpha : float
        Relates to preconditioning
    mix_up : int
        Number of components to mix up to
    prior_subset_size : int
        Number of samples per job for computing priors
    update_period : int
        How often the preconditioning subspace is updated
    num_samples_history : int
        Relates to online preconditioning
    preconditioning_rank_in : int
        Relates to online preconditioning
    preconditioning_rank_out : int
        Relates to online preconditioning
    """

    def __init__(self, default_feature_config):
        super(NnetTrainer, self).__init__(default_feature_config)
        self.realign_times = 0

        self.beam = 10
        self.retry_beam = 15000000

        self.initial_learning_rate = 0.32
        self.final_learning_rate = 0.032
        self.bias_stddev = 0.5

        self.pnorm_input_dim = 3000
        self.pnorm_output_dim = 300
        self.p = 2

        self.shrink_interval = 5
        self.shrink = True
        self.num_frames_shrink = 2000

        self.final_learning_rate_factor = 0.5
        self.hidden_layer_dim = 50

        self.samples_per_iter = 200000
        self.shuffle_buffer_size = 5000
        self.add_layers_period = 2
        self.num_hidden_layers = 3
        self.modify_learning_rates = False

        self.last_layer_factor = 0.1
        self.first_layer_factor = 1.0

        self.randprune = 4.0
        self.alpha = 4.0
        self.max_change = 10.0
        self.mix_up = 12000  # From run_nnet2.sh
        self.prior_subset_size = 10000
        self.boost_silence = 0.5

        self.update_period = 4
        self.num_samples_history = 2000
        self.max_change_per_sample = 0.075
        self.precondition_rank_in = 20
        self.precondition_rank_out = 80
        self.ivector_model_path = None
        self.lda_dimension = 40
        self.lda_random_prune = 4.0

        finish_add_layers_iter = self.num_hidden_layers * self.add_layers_period
        self.first_modify_iteration = finish_add_layers_iter + self.add_layers_period
        self.mix_up_iteration = (self.num_iterations + finish_add_layers_iter) / 2

        # Get iterations at which we will realign
        if self.realign_times != 0:
            step = int(self.num_iterations / self.realign_times)
            self.realignment_iterations = list(range(self.num_iterations, step))
        self.architecture = 'nnet'

    def compute_calculated_properties(self):
        finish_add_layers_iter = self.num_hidden_layers * self.add_layers_period
        self.first_modify_iteration = finish_add_layers_iter + self.add_layers_period
        self.mix_up_iteration = (self.num_iterations + finish_add_layers_iter) / 2

        # Get iterations at which we will realign
        if self.realign_times != 0:
            step = int(self.num_iterations / self.realign_times)
            self.realignment_iterations = list(range(self.num_iterations, step))

    @property
    def train_type(self):
        return 'nnet'

    @property
    def phone_type(self):
        return 'triphone'

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        self._setup_for_init(identifier, temporary_directory, corpus, dictionary)

        topo_path = os.path.join(self.dictionary.output_directory, 'topo')
        tree_info_proc = subprocess.Popen([thirdparty_binary('tree-info'),
                                           os.path.join(previous_trainer.align_directory, 'tree')],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tree_info = tree_info_proc.stdout.read()
        tree_info = tree_info.split()
        num_leaves = tree_info[1]
        ali_tree_path = os.path.join(previous_trainer.align_directory, 'tree')
        shutil.copy(ali_tree_path, os.path.join(self.train_directory, 'tree'))
        num_leaves = num_leaves.decode("utf-8")
        get_lda_nnet(self, previous_trainer.align_directory, self.corpus.num_jobs)

        lda_mat_path = os.path.join(self.train_directory, 'nnet_lda.mat')

        # Get examples for training
        os.makedirs(self.egs_directory, exist_ok=True)

        # # Get valid uttlist and train subset uttlist
        num_utts_subset = 300
        log_path = os.path.join(self.train_directory, 'log', 'training_egs_feats.log')

        utterances = self.corpus.utterances
        shuffle(utterances)
        train_subset_uttlist = utterances[:num_utts_subset]

        utterances = self.corpus.utterances
        # Filter by the scp list
        filtered = filter_scp(train_subset_uttlist, utterances, exclude=True)
        # Shuffle the list
        shuffle(filtered)
        # Take only the first num_utts_subset lines
        valid_uttlist = filtered[:num_utts_subset]

        get_egs(self,
                previous_trainer.align_directory,
                valid_uttlist,
                train_subset_uttlist)

        # Initialize neural net
        print('Initializing DNN training...')
        stddev = float(1.0 / self.pnorm_input_dim ** 0.5)
        online_preconditioning_opts = 'alpha={} num-samples-history={} update-period={} ' \
                                      'rank-in={} rank-out={} max-change-per-sample={}'.format(
            self.alpha, self.num_samples_history, self.update_period, self.precondition_rank_in,
            self.precondition_rank_out, self.max_change_per_sample)
        nnet_config_path = os.path.join(self.train_directory, 'nnet.config')
        hidden_config_path = os.path.join(self.train_directory, 'hidden.config')
        feat_dim_path = os.path.join(self.train_directory, 'feat_dim')
        with open(feat_dim_path, 'r') as inf:
            feat_dim = int(inf.read().strip())
        if self.feature_config.ivectors:
            ivector_dim_path = os.path.join(self.train_directory, 'ivector_dim')
            with open(ivector_dim_path, 'r') as inf:
                ivector_dim = int(inf.read().strip())
            feat_dim += ivector_dim

        with open(nnet_config_path, 'w', newline='') as nc:
            if self.feature_config.ivectors:
                nc.write('SpliceComponent input-dim={} left-context={} right-context={} const-component-dim={}\n'.format(
                    feat_dim, self.feature_config.splice_left_context, self.feature_config.splice_right_context,
                    ivector_dim))
            else:
                nc.write('SpliceComponent input-dim={} left-context={} right-context={}\n'.format(
                    feat_dim, self.feature_config.splice_left_context, self.feature_config.splice_right_context))
            nc.write('FixedAffineComponent matrix={}\n'.format(lda_mat_path))
            nc.write(
                'AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(
                    self.lda_dimension, self.pnorm_input_dim, online_preconditioning_opts, self.initial_learning_rate,
                    stddev,
                    self.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(self.pnorm_input_dim,
                                                                               self.pnorm_output_dim, self.p))
            nc.write('NormalizeComponent dim={}\n'.format(self.pnorm_output_dim))
            nc.write(
                'AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(
                    self.pnorm_output_dim, num_leaves, online_preconditioning_opts, self.initial_learning_rate,
                    stddev, self.bias_stddev))
            nc.write('SoftmaxComponent dim={}\n'.format(num_leaves))

        with open(hidden_config_path, 'w', newline='') as nc:
            nc.write(
                'AffineComponentPreconditionedOnline input-dim={} output-dim={} {} learning-rate={} param-stddev={} bias-stddev={}\n'.format(
                    self.pnorm_output_dim, self.pnorm_input_dim, online_preconditioning_opts,
                    self.initial_learning_rate, stddev, self.bias_stddev))
            nc.write('PnormComponent input-dim={} output-dim={} p={}\n'.format(self.pnorm_input_dim,
                                                                               self.pnorm_output_dim, self.p))
            nc.write('NormalizeComponent dim={}\n'.format(self.pnorm_output_dim))

        log_path = os.path.join(self.train_directory, 'log', 'nnet_init.log')
        nnet_info_path = os.path.join(self.train_directory, 'log', 'nnet_info.log')
        with open(log_path, 'w') as logf:
            with open(nnet_info_path, 'w') as outf:
                nnet_am_init_proc = subprocess.Popen([thirdparty_binary('nnet-am-init'),
                                                      os.path.join(previous_trainer.align_directory, 'tree'),
                                                      topo_path,
                                                      "{} {} -|".format(thirdparty_binary('nnet-init'),
                                                                        nnet_config_path),
                                                      os.path.join(self.train_directory, '0.mdl')],
                                                     stderr=logf)
                nnet_am_init_proc.communicate()

                nnet_am_info = subprocess.Popen([thirdparty_binary('nnet-am-info'),
                                                 os.path.join(self.train_directory, '0.mdl')],
                                                stdout=outf,
                                                stderr=logf)
                nnet_am_info.communicate()

        ali_files = glob.glob(os.path.join(previous_trainer.align_directory, 'ali.*'))
        prev_ali_path = os.path.join(self.train_directory, 'prev_ali')
        with open(prev_ali_path, 'wb') as outfile:
            for ali_file in ali_files:
                with open(os.path.join(previous_trainer.align_directory, ali_file), 'rb') as infile:
                    for line in infile:
                        outfile.write(line)
        nnet_train_trans(self.train_directory, previous_trainer.align_directory, prev_ali_path, self.corpus.num_jobs)
        #       Compile train graphs (gets fsts.{} for alignment)
        compile_train_graphs(self.train_directory, self.dictionary.output_directory,
                             self.data_directory, self.corpus.num_jobs)
        print('Initialization complete!')

    @property
    def egs_directory(self):
        return os.path.join(self.train_directory, 'egs')

    def train(self, call_back=None):
        egs_directory = self.egs_directory
        # Training loop
        if call_back == print:
            iters = tqdm(range(self.num_iterations))
        else:
            iters = range(self.num_iterations)
        for i in iters:
            model_path = os.path.join(self.train_directory, '{}.mdl'.format(i))
            next_model_path = os.path.join(self.train_directory, '{}.mdl'.format(i + 1))

            # Combine all examples (could integrate validation diagnostics, etc., later-- see egs functions)
            egs_files = []
            for file in os.listdir(egs_directory):
                if file.startswith('egs.'):
                    egs_files.append(file)
            with open(os.path.join(egs_directory, 'all_egs.egs'), 'wb') as outfile:
                for egs_file in egs_files:
                    with open(os.path.join(egs_directory, egs_file), 'rb') as infile:
                        for line in infile:
                            outfile.write(line)

            # Get accuracy rates for the current iteration (to pull out graph later)
            # compute_prob(i, directory, egs_directory, model_path, self.num_jobs)
            log_path = os.path.join(self.train_directory, 'log', 'compute_prob_train.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                compute_prob_proc = subprocess.Popen([thirdparty_binary('nnet-compute-prob'),
                                                      model_path,
                                                      'ark:'+os.path.join(egs_directory,'all_egs.egs')],
                                                     stdout=subprocess.PIPE,
                                                     stderr=logf)
                log_prob = compute_prob_proc.stdout.read().decode('utf-8').strip()
                compute_prob_proc.communicate()

            print("Iteration {} of {} \t\t Log-probability: {}".format(i + 1, self.num_iterations, log_prob))

            # Pull out and save graphs
            # This is not quite working when done automatically - to be worked out with unit testing.
            # get_accuracy_graph(os.path.join(directory, 'log'), os.path.join(directory, 'log'))

            # If it is NOT the first iteration,
            # AND we still have layers to add,
            # AND it's the right time to add a layer...
            if i > 0 and i <= ((self.num_hidden_layers - 1) * self.add_layers_period) and (
                    (i - 1) % self.add_layers_period) == 0:
                # Add a new hidden layer
                mdl = os.path.join(self.train_directory, 'tmp{}.mdl'.format(i))
                log_path = os.path.join(self.train_directory, 'log', 'temp_mdl.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    with open(mdl, 'w') as outf:
                        tmp_mdl_init_proc = subprocess.Popen([thirdparty_binary('nnet-init'),
                                                              '--srand={}'.format(i),
                                                              os.path.join(self.train_directory, 'hidden.config'),
                                                              '-'],
                                                             stdout=subprocess.PIPE,
                                                             stderr=logf)
                        tmp_mdl_ins_proc = subprocess.Popen([thirdparty_binary('nnet-insert'),
                                                             model_path,
                                                             '-', '-'],
                                                            stdin=tmp_mdl_init_proc.stdout,
                                                            stdout=outf,
                                                            stderr=logf)
                        tmp_mdl_ins_proc.communicate()

            # Otherwise just use the past model
            else:
                mdl = os.path.join(self.train_directory, '{}.mdl'.format(i))

            # Shuffle examples and train nets with SGD
            nnet_train(self.train_directory, egs_directory, mdl, i, self.corpus.num_jobs)

            # Get nnet list from the various jobs on this iteration
            nnets_list = [os.path.join(self.train_directory, '{}.{}.mdl'.format((i + 1), x))
                          for x in range(self.corpus.num_jobs)]

            if (i + 1) >= self.num_iterations:
                learning_rate = self.final_learning_rate
            else:
                learning_rate = self.initial_learning_rate * math.exp(
                    i * math.log(self.final_learning_rate / self.initial_learning_rate) / self.num_iterations)

            log_path = os.path.join(self.train_directory, 'log', 'average.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                nnet_avg_proc = subprocess.Popen([thirdparty_binary('nnet-am-average')]
                                                 + nnets_list
                                                 + ['-'],
                                                 stdout=subprocess.PIPE,
                                                 stderr=logf)
                nnet_copy_proc = subprocess.Popen([thirdparty_binary('nnet-am-copy'),
                                                   '--learning-rate={}'.format(learning_rate),
                                                   '-',
                                                   next_model_path],
                                                  stdin=nnet_avg_proc.stdout,
                                                  stderr=logf)
                nnet_copy_proc.communicate()
            if not os.path.exists(next_model_path):
                raise(Exception('There was an error training in iteration {}, please check the logs.'.format(i)))

            # If it's the right time, do mixing up
            if self.mix_up > 0 and i == self.mix_up_iteration:
                log_path = os.path.join(self.train_directory, 'log', 'mix_up.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_am_mixup_proc = subprocess.Popen([thirdparty_binary('nnet-am-mixup'),
                                                           '--min-count=10',
                                                           '--num-mixtures={}'.format(self.mix_up),
                                                           next_model_path,
                                                           next_model_path],
                                                          stderr=logf)
                    nnet_am_mixup_proc.communicate()

            # Realign if it's the right time
            if i in self.realignment_iterations:
                prev_egs_directory = egs_directory
                egs_directory = os.path.join(self.train_directory, 'egs_{}'.format(i))
                os.makedirs(egs_directory, exist_ok=True)

                #   Get average posterior for purposes of adjusting priors
                get_average_posteriors(i, self.train_directory, prev_egs_directory, self, self.corpus.num_jobs)
                log_path = os.path.join(self.train_directory, 'log', 'vector_sum_exterior.{}.log'.format(i))
                vectors_to_sum = glob.glob(os.path.join(self.train_directory, 'post.{}.*.vec'.format(i)))

                with open(log_path, 'w') as logf:
                    vector_sum_proc = subprocess.Popen([thirdparty_binary('vector-sum')]
                                                       + vectors_to_sum
                                                       + [os.path.join(self.train_directory, 'post.{}.vec'.format(i))
                                                          ],
                                                       stderr=logf)
                    vector_sum_proc.communicate()

                #   Readjust priors based on computed posteriors
                log_path = os.path.join(self.train_directory, 'log', 'adjust_priors.{}.log'.format(i))
                with open(log_path, 'w') as logf:
                    nnet_adjust_priors_proc = subprocess.Popen([thirdparty_binary('nnet-adjust-priors'),
                                                                os.path.join(self.train_directory, '{}.mdl'.format(i)),
                                                                os.path.join(self.train_directory,
                                                                             'post.{}.vec'.format(i)),
                                                                os.path.join(self.train_directory, '{}.mdl'.format(i))],
                                                               stderr=logf)
                    nnet_adjust_priors_proc.communicate()

                #   Realign:

                #       Do alignment
                nnet_align("final", self, self.train_directory, self.train_directory,
                           self.corpus.num_jobs)
                compute_alignment_improvement(i, self, self.train_directory, self.corpus.num_jobs)

                #     Finally, relabel the egs
                ali_files = glob.glob(os.path.join(self.train_directory, 'ali.*'))
                alignments = os.path.join(self.train_directory, 'alignments.')
                with open(alignments, 'wb') as outfile:
                    for ali_file in ali_files:
                        with open(os.path.join(self.train_directory, ali_file), 'rb') as infile:
                            for line in infile:
                                outfile.write(line)
                relabel_egs(i, self.train_directory, prev_egs_directory, alignments, egs_directory,
                            self.corpus.num_jobs)

        # Rename the final model
        shutil.copy(os.path.join(self.train_directory, '{}.mdl'.format(self.num_iterations - 1)),
                    os.path.join(self.train_directory, 'final.mdl'))

    def align(self, subset, call_back=None):

        shutil.copy(os.path.join(self.train_directory, 'tree'), self.align_directory)
        shutil.copyfile(os.path.join(self.train_directory, 'final.mdl'),
                        os.path.join(self.align_directory, 'final.mdl'))

        # Do alignment
        nnet_align("final", self, self.train_directory, self.align_directory,
                   self.corpus.num_jobs)
        self.save(os.path.join(self.align_directory, 'acoustic_model.zip'))
        self.export_textgrids()

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
        acoustic_model.add_model(self.train_directory)
        acoustic_model.add_lda_matrix(self.corpus.output_directory)
        if self.feature_config.ivectors:
            acoustic_model.add_ivector_model(self.corpus.ivector_directory)
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        acoustic_model.dump(basename)
