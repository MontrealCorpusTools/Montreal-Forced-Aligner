import os

def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)

class MonophoneConfig(object):
    def __init__(self, **kwargs):
        self.num_iters = 40

        self.scale_opts = ['--transition-scale=1.0',
                        '--acoustic-scale=0.1',
                        '--self-loop-scale=0.1']
        self.beam=10
        self.retry_beam=40
        self.max_iter_inc = 30
        self.totgauss = 1000
        self.boost_silence = 1.0
        if kwargs.get('align_often', False):
            self.realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14,
                                    16, 18, 20, 23, 26, 29, 32, 35, 38]
        else:
            self.realign_iters = [1, 5, 10, 15, 20, 25, 30, 35, 38]
        self.stage = -4
        self.power = 0.25

        self.do_fmllr = False

        for k,v in kwargs.items():
            setattr(self, k, v)

class TriphoneConfig(MonophoneConfig):
    def __init__(self, **kwargs):
        defaults = {'num_iters': 35,
        'num_states': 3100,
        'num_gauss': 50000,
        'cluster_threshold': 100,
        'silence_weight': 0.0}
        defaults.update(kwargs)
        super(TriphoneConfig, self).__init__(**defaults)

class TriphoneFmllrConfig(TriphoneConfig):
    def __init__(self, align_often = True, **kwargs):
        defaults = {'do_fmllr': True,
        'fmllr_update_type': 'full',
        'fmllr_iters': [2, 4, 6, 12],
        'fmllr_power': 0.2}
        defaults.update(kwargs)
        super(TriphoneFmllrConfig, self).__init__(**defaults)


class MfccConfig(object):
    def __init__(self, output_directory, kwargs = None):
        if kwargs is None:
            kwargs = {}
        self.config_dict = {'use-energy':False}
        self.config_dict.update(kwargs)
        self.output_directory = output_directory
        self.write()

    def update(self, kwargs):
        self.config_dict.update(kwargs)
        self.write()

    @property
    def config_directory(self):
        path = os.path.join(self.output_directory, 'config')
        os.makedirs(path, exist_ok = True)
        return path

    @property
    def path(self):
        return os.path.join(self.config_directory, 'mfcc.conf')

    def write(self):
        with open(self.path, 'w', encoding = 'utf8') as f:
            for k,v in self.config_dict.items():
                f.write('--{}={}\n'.format(k, make_safe(v)))
