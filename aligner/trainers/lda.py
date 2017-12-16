from .triphone import TriphoneTrainer


class LdaTrainer(TriphoneTrainer):
    '''

    Configuration class for LDA+MLLT training

    Scale options defaults to::

        ['--transition-scale=1.0', '--acoustic-scale=0.1', '--self-loop-scale=0.1']

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
    num_leaves : int
        Number of states in the decision tree, defaults to 1000
    max_gaussians : int
        Number of gaussians in the decision tree, defaults to 10000
    cluster_threshold : int
        For build-tree control final bottom-up clustering of leaves, defaults to 100
    lda_dimension : int
        Dimensionality of the LDA matrix
    mllt_iters : list
        List of iterations to perform MLLT estimation
    random_prune : float
        This is approximately the ratio by which we will speed up the
        LDA and MLLT calculations via randomized pruning
    '''

    def __init__(self):
        super(LdaTrainer, self).__init__()
        self.lda_dimension = 40
        self.mllt_iters = [2, 4, 6, 12]
        self.random_prune = 4.0
        self.left_context = 4
        self.right_context = 4

    @property
    def train_type(self):
        return 'lda'
