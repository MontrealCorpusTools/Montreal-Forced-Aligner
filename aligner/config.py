import os

beam=10
retry_beam=40

mono_num_iters = 40

scale_opts = ['--transition-scale=1.0',
                '--acoustic-scale=0.1',
                '--self-loop-scale=0.1']

mono_max_iter_inc = 30
totgauss = 1000
boost_silence = 1.0
mono_realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]
stage = -4
power = 0.25

tri_num_iters = 35

tri_max_iter_inc = 25

tri_realign_iters = [10, 20, 30]

tri_num_states = 3100
tri_num_gauss = 50000

fmllr_update_type = 'full'
fmllr_iters=[2, 4, 6, 12]
silence_weight=0.0
fmllr_power = 0.2

def make_safe(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)

class MonophoneConfig(object):
    def __init__(self, output_directory, **kwargs):
        self.config_dict.update(kwargs)
        self.output_directory = output_directory
        self.num_iters = 40

        self.scale_opts = ['--transition-scale=1.0',
                        '--acoustic-scale=0.1',
                        '--self-loop-scale=0.1']
        self.beam=10
        self.retry_beam=40
        self.max_iter_inc = 30
        self.totgauss = 1000
        self.boost_silence = 1.0
        self.realign_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14,
                                    16, 18, 20, 23, 26, 29, 32, 35, 38]
        self.stage = -4
        self.power = 0.25

        for k,v in kwargs.items():
            setattr(self, k, v)

class MfccConfig(object):
    def __init__(self, output_directory, kwargs = None):
        if kwargs is None:
            kwargs = {}
        self.config_dict = {'use-energy':False}
        self.config_dict.update(kwargs)
        self.output_directory = output_directory
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
