import os

TEMP_DIR = os.path.expanduser('~/Documents/MFA')


def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class MonophoneConfig(object):
    '''
    Configuration class for monophone training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    If ``align_often`` is True in the keyword arguments, ``realign_iters`` will be::

        [1, 5, 10, 15, 20, 25, 30, 35, 38]

    Otherwise, ``realign_iters`` will be::

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform, defaults to 40
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_iter_inc : int
        Last iter to increase #Gauss on, defaults to 30
    totgauss : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to False
    '''

    def __init__(self, **kwargs):
        self.num_iters = 40

        self.scale_opts = ['--transition-scale=1.0',
                           '--acoustic-scale=0.1',
                           '--self-loop-scale=0.1']
        self.beam = 10
        self.retry_beam = 40
        self.max_gauss_count = 1000
        self.boost_silence = 1.0
        if kwargs.get('align_often', False):
            self.realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14,
                                  16, 18, 20, 23, 26, 29, 32, 35, 38]
        else:
            self.realign_iters = [1, 5, 10, 15, 20, 25, 30, 35, 38]
        self.stage = -4
        self.power = 0.25

        self.do_fmllr = False

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def max_iter_inc(self):
        return self.num_iters - 10

    @property
    def inc_gauss_count(self):
        return int((self.max_gauss_count - self.initial_gauss_count) / self.max_iter_inc)


class TriphoneConfig(MonophoneConfig):
    '''
    Configuration class for triphone training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    If ``align_often`` is True in the keyword arguments, ``realign_iters`` will be::

        [1, 5, 10, 15, 20, 25, 30, 35, 38]

    Otherwise, ``realign_iters`` will be::

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform, defaults to 35
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_iter_inc : int
        Last iter to increase #Gauss on, defaults to 30
    totgauss : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to False
    num_states : int
        Number of states in the decision tree, defaults to 3100
    num_gauss : int
        Number of gaussians in the decision tree, defaults to 50000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    '''

    def __init__(self, **kwargs):
        defaults = {'num_iters': 35,
                    'initial_gauss_count': 3100,
                    'max_gauss_count': 50000,
                    'cluster_threshold': 100}
        defaults.update(kwargs)
        super(TriphoneConfig, self).__init__(**defaults)


class TriphoneFmllrConfig(TriphoneConfig):
    '''
    Configuration class for speaker-adapted triphone training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

    If ``align_often`` is True in the keyword arguments, ``realign_iters`` will be::

        [1, 5, 10, 15, 20, 25, 30, 35, 38]

    Otherwise, ``realign_iters`` will be::

        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]

    ``fmllr_iters`` defaults to::

        [2, 4, 6, 12]

    Attributes
    ----------
    num_iters : int
        Number of training iterations to perform, defaults to 35
    scale_opts : list
        Options for specifying scaling in alignment
    beam : int
        Default beam width for alignment, defaults = 10
    retry_beam : int
        Beam width to fall back on if no alignment is produced, defaults to 40
    max_iter_inc : int
        Last iter to increase #Gauss on, defaults to 30
    totgauss : int
        Total number of gaussians, defaults to 1000
    boost_silence : float
        Factor by which to boost silence likelihoods in alignment, defaults to 1.0
    realign_iters : list
        List of iterations to perform alignment
    stage : int
        Not used
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25
    do_fmllr : bool
        Specifies whether to do speaker adaptation, defaults to True
    num_states : int
        Number of states in the decision tree, defaults to 3100
    num_gauss : int
        Number of gaussians in the decision tree, defaults to 50000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to ``'full'``
    fmllr_iters : list
        List of iterations to perform fMLLR estimation
    fmllr_power : float
        Defaults to 0.2
    silence_weight : float
        Weight on silence in fMLLR estimation
    '''

    def __init__(self, align_often=True, **kwargs):
        defaults = {'do_fmllr': True,
                    'fmllr_update_type': 'full',
                    'fmllr_iters': [2, 4, 6, 12],
                    'fmllr_power': 0.2,
                    'silence_weight': 0.0}
        defaults.update(kwargs)
        super(TriphoneFmllrConfig, self).__init__(**defaults)


class MfccConfig(object):
    '''
    Class to store configuration information about MFCC generation

    The ``config_dict`` currently stores one key ``'use-energy'`` which
    defaults to False

    Parameters
    ----------
    output_directory : str
        Path to directory to save configuration files for Kaldi
    kwargs : dict, optional
        If specified, updates ``config_dict`` with this dictionary

    Attributes
    ----------
    config_dict : dict
        Dictionary of configuration parameters
    '''

    def __init__(self, output_directory, job=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.job = job
        self.config_dict = {'use-energy': False, 'frame-shift': 10}
        self.config_dict.update(kwargs)
        self.output_directory = output_directory
        self.write()

    def update(self, kwargs):
        '''
        Update configuration dictionary with new dictionary

        Parameters
        ----------
        kwargs : dict
            Dictionary of new parameter values
        '''
        self.config_dict.update(kwargs)
        self.write()

    @property
    def config_directory(self):
        path = os.path.join(self.output_directory, 'config')
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def path(self):
        if self.job is None:
            f = 'mfcc.conf'
        else:
            f = 'mfcc.{}.conf'.format(self.job)
        return os.path.join(self.config_directory, f)

    def write(self):
        '''
        Write configuration dictionary to a file for use in Kaldi binaries
        '''
        with open(self.path, 'w', encoding='utf8') as f:
            for k, v in self.config_dict.items():
                f.write('--{}={}\n'.format(k, make_safe(v)))
