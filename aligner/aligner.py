
import os
import shutil

temp_directory = os.path.normpath(os.path.expanduser('~/Documents/MFA'))

class BaseAligner(object):
    def __init__(self, source_directory, output_directory):

        self.source_directory = source_directory
        dirname = os.path.split(self.source_directory)[-1]
        self.output_directory = output_directory
        self.temp_directory = os.path.join(temp_directory, dirname)
        if os.path.exists(self.temp_directory): # Clean up from failed runs?
            shutil.rmtree(self.temp_directory)

        self.dict_dir = os.path.join(self.temp_directory, 'dict')
        self.lang_dir = os.path.join(self.temp_directory, 'lang')
        self.phones_dir = os.path.join(self.lang_dir, 'phones')
        self.mfcc_dir = os.path.join(self.temp_directory, 'mfcc')
        self.conf_dir = os.path.join(self.temp_directory, 'conf')
        self.train_dir = os.path.join(self.temp_directory, 'train')
        os.makedirs(self.dict_dir)
        os.makedirs(self.mfcc_dir)
        os.makedirs(self.conf_dir)
        os.makedirs(self.train_dir)
        os.makedirs(self.phones_dir)

    def data_prep(self):
        self.prep_config()
        self.prep_dict()

    def prep_mfcc(self):
        pass

    def prep_dict(self):
        prepare_dict(self)

    def prep_train(self):
        pass

    def prep_config(self):
        mfcc_config = os.path.join(self.conf_dir, 'mfcc.conf')
        with open(mfcc_config, 'w') as f:
            f.write('--use-energy=false   # only non-default option.')
