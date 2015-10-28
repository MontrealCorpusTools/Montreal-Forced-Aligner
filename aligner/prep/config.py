import os

def prepare_config(config_directory):
    os.makedirs(config_directory, exist_ok = True)

    mfcc_config = os.path.join(config_directory, 'mfcc.conf')
    with open(mfcc_config, 'w') as f:
        f.write('--use-energy=false   # only non-default option.')
