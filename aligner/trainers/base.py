import os
import re
from tqdm import tqdm
import subprocess
import shutil

from .. import __version__
from ..exceptions import TrainerError, NoSuccessfulAlignments
from ..helper import thirdparty_binary, make_path_safe

from ..multiprocessing import (align, acc_stats, convert_ali_to_textgrids, compute_alignment_improvement)

from ..models import AcousticModel
from ..features.config import FeatureConfig

class BaseTrainer(object):
    '''
    Configuration class for all trainings


    Attributes
    ----------
    num_iterations : int
        Number of training iterations to perform, defaults to 40
    transition_scale : float
        Scaling of transition costs in alignment, defaults to 1.0
    acoustic_scale : float
        Scaling of acoustic costs in alignment, defaults to 0.1
    self_loop_scale : float
        Scaling of self loop costs in alignment, defaults to 0.1
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realignment_iterations : list
        List of iterations to perform alignment
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    '''

    def __init__(self, default_feature_config):
        self.transition_scale = 1.0
        self.acoustic_scale = 0.1
        self.self_loop_scale = 0.1
        self.realignment_iterations = []
        self.num_iterations = 40
        self.beam = 10
        self.retry_beam = 40
        self.max_gaussians = 1000
        self.boost_silence = 1.0
        self.power = 0.25
        self.subset = None
        self.calc_pron_probs = False
        self.architecture = 'gmm-hmm'
        self.feature_config = FeatureConfig()
        self.feature_config.update(default_feature_config.params())
        self.initial_gaussians = None # Gets set later
        self.temp_directory = None
        self.identifier = None
        self.corpus = None
        self.data_directory = None
        self.dictionary = None
        self.debug =False

    def compute_calculated_properties(self):
        pass

    @property
    def feature_file_base_name(self):
        return self.feature_config.feature_id

    @property
    def train_directory(self):
        return os.path.join(self.temp_directory, self.identifier)

    @property
    def log_directory(self):
        return os.path.join(self.train_directory, 'log')

    @property
    def align_directory(self):
        return os.path.join(self.temp_directory, self.identifier + '_ali')

    @property
    def align_log_directory(self):
        return os.path.join(self.align_directory, 'log')

    @property
    def train_type(self):
        raise NotImplementedError

    @property
    def phone_type(self):
        raise NotImplementedError

    @property
    def final_gaussian_iteration(self):
        return self.num_iterations - 10

    @property
    def gaussian_increment(self):
        return int((self.max_gaussians - self.initial_gaussians) / self.final_gaussian_iteration)

    def update(self, data):
        for k, v in data.items():
            if k == 'features':
                self.feature_config.update(v)
            elif not hasattr(self, k):
                raise TrainerError('No field found for key {}'.format(k))
            else:
                setattr(self, k, v)
        self.compute_calculated_properties()

    def _setup_for_init(self, identifier, temporary_directory, corpus, dictionary):
        print('Initializing training for {}...'.format(identifier))
        self.temp_directory = temporary_directory
        self.identifier = identifier
        self.corpus = corpus
        self.dictionary = dictionary
        os.makedirs(self.train_directory, exist_ok=True)
        os.makedirs(self.align_directory, exist_ok=True)
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.align_log_directory, exist_ok=True)
        if self.subset is not None and self.subset > corpus.num_utterances:
            print('Warning: Subset specified is larger than the dataset, using full corpus for this training block.')
        self.data_directory = corpus.split_directory()
        self.feature_config.generate_features(self.corpus)
        if self.subset is not None:
            self.data_directory = corpus.subset_directory(self.subset, self.feature_config)

    def init_training(self, identifier, temporary_directory, corpus, dictionary, previous_trainer):
        raise NotImplementedError

    def parse_log_directory(self, directory, iteration, num_jobs, call_back):
        '''
        Parse error files and relate relevant information about unaligned files
        '''
        if call_back is None:
            return
        error_regex = re.compile(r'Did not successfully decode file (\w+),')
        too_little_data_regex = re.compile(
            r'Gaussian has too little data but not removing it because it is the last Gaussian')
        skipped_transition_regex = re.compile(r'(\d+) out of (\d+) transition-states skipped due to insufficient data')

        log_like_regex = re.compile(r'Overall avg like per frame = ([-0-9.]+|nan) over (\d+) frames')
        error_files = []
        for i in range(num_jobs):
            path = os.path.join(directory, 'align.{}.{}.log'.format(iteration - 1, i))
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                error_files.extend(error_regex.findall(f.read()))
        update_path = os.path.join(directory, 'update.{}.log'.format(iteration))
        #if os.path.exists(update_path):
        #    with open(update_path, 'r') as f:
        #        data = f.read()
        #        m = log_like_regex.search(data)
        #        if m is not None:
        #            log_like, tot_frames = m.groups()
        #            if log_like == 'nan':
        #                raise (NoSuccessfulAlignments('Could not align any files.  Too little data?'))
        #            log_like = float(log_like)
        #        skipped_transitions = skipped_transition_regex.search(data)
        #        skipped_transition = skipped_transitions.groups()
        #        num_too_little_data = len(too_little_data_regex.findall(data))
        #        call_back('missing data gaussians', num_too_little_data)
        return error_files#, log_like, num_too_little_data

    def get_unaligned_utterances(self):
        error_regex = re.compile(r'Did not successfully decode file (\w+),')
        error_files = []
        for i in range(self.corpus.num_jobs):
            path = os.path.join(self.align_directory, 'log', 'align.final.{}.log'.format(i))
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                error_files.extend(error_regex.findall(f.read()))
        return error_files


    def align(self, subset, call_back=None):
        align('final', self.train_directory, self.data_directory,
              self.dictionary.optional_silence_csl,
              self.corpus.num_jobs, self, self.align_directory)

        log_dir = os.path.join(self.align_directory, 'log')
        os.makedirs(log_dir, exist_ok=True)

        shutil.copy(os.path.join(self.train_directory, 'tree'), self.align_directory)
        shutil.copyfile(os.path.join(self.train_directory, 'final.mdl'),
                        os.path.join(self.align_directory, 'final.mdl'))

        shutil.copyfile(os.path.join(self.train_directory, 'final.occs'),
                        os.path.join(self.align_directory, 'final.occs'))
        self.save(os.path.join(self.align_directory, 'acoustic_model.zip'))
        self.export_textgrids()

    def train(self, call_back=None):
        final_mdl_path = os.path.join(self.train_directory, 'final.mdl')
        if os.path.exists(final_mdl_path):
            print('{} training already done, skipping.'.format(self.identifier))
            return
        num_gauss = self.initial_gaussians
        if call_back == print:
            iters = tqdm(range(1, self.num_iterations))
        else:
            iters = range(1, self.num_iterations)
        for i in iters:
            model_path = os.path.join(self.train_directory, '{}.mdl'.format(i))
            occs_path = os.path.join(self.train_directory, '{}.occs'.format(i + 1))
            next_model_path = os.path.join(self.train_directory, '{}.mdl'.format(i + 1))
            if os.path.exists(next_model_path):
                continue
            if i in self.realignment_iterations:
                align(i, self.train_directory, self.data_directory,
                      self.dictionary.optional_silence_csl,
                      self.corpus.num_jobs, self)
                if self.debug:
                    compute_alignment_improvement(i, self, self.train_directory, self.corpus.num_jobs)

            acc_stats(i, self.train_directory, self.data_directory, self.corpus.num_jobs, self)
            log_path = os.path.join(self.log_directory, 'update.{}.log'.format(i))
            with open(log_path, 'w') as logf:
                acc_files = [os.path.join(self.train_directory, '{}.{}.acc'.format(i, x))
                             for x in range(self.corpus.num_jobs)]
                est_proc = subprocess.Popen([thirdparty_binary('gmm-est'),
                                             '--write-occs=' + occs_path,
                                             '--mix-up=' + str(num_gauss), '--power=' + str(self.power),
                                             model_path,
                                             "{} - {}|".format(thirdparty_binary('gmm-sum-accs'),
                                                               ' '.join(map(make_path_safe, acc_files))),
                                             next_model_path],
                                            stderr=logf)
                est_proc.communicate()
            if not self.debug:
                for f in acc_files:
                    os.remove(f)
            if not os.path.exists(next_model_path):
                raise(Exception('There was an error training in iteration {}, please check the logs.'.format(i)))
            self.parse_log_directory(self.log_directory, i, self.corpus.num_jobs, call_back)
            if i < self.final_gaussian_iteration:
                num_gauss += self.gaussian_increment
        shutil.copy(os.path.join(self.train_directory, '{}.mdl'.format(self.num_iterations)),
                    final_mdl_path)
        shutil.copy(os.path.join(self.train_directory, '{}.occs'.format(self.num_iterations)),
                    os.path.join(self.train_directory, 'final.occs'))
        if not self.debug:
            for i in range(1, self.num_iterations):
                model_path = os.path.join(self.train_directory, '{}.mdl'.format(i))
                try:
                    os.remove(model_path)
                except FileNotFoundError:
                    pass
                try:
                    os.remove(os.path.join(self.train_directory, '{}.occs'.format(i)))
                except FileNotFoundError:
                    pass

    @property
    def meta(self):
        data = {'phones': sorted(self.dictionary.nonsil_phones),
                'version': __version__,
                'architecture': self.architecture,
                'features': self.feature_config.params(),
                }
        return data

    def export_textgrids(self):
        '''
        Export a TextGrid file for every sound file in the dataset
        '''
        ali_directory = self.align_directory

        convert_ali_to_textgrids(self, os.path.join(self.align_directory, 'textgrids'), self.align_directory, self.dictionary,
                                 self.corpus, self.corpus.num_jobs)

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
        os.makedirs(directory, exist_ok=True)
        basename, _ = os.path.splitext(path)
        acoustic_model.dump(basename)
