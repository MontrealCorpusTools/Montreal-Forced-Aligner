import shutil
import os
import sys
from tqdm import tqdm
from urllib.request import urlretrieve
import argparse


def tqdm_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = args.temp_directory
    if not args.temp_directory:
        temp_dir = base_dir
    os.makedirs(base_dir, exist_ok=True)
    if sys.platform == 'darwin':
        plat = 'macosx'
    elif sys.platform == 'win32':
        plat = 'win64'
    else:
        plat = 'linux'
    print('Downloading precompiled binaries for {}...'.format(plat))

    download_link = 'http://mlmlab.org/mfa/precompiled_binaries/mfa_thirdparty_{}.zip'.format(
        plat)
    path = os.path.join(temp_dir, '{}.zip'.format(plat))
    if args.redownload or not os.path.exists(path):
        with tqdm(unit='B', unit_scale=True, miniters=1) as t:
            filename, headers = urlretrieve(download_link, path, reporthook=tqdm_hook(t), data=None)
    shutil.unpack_archive(path, base_dir)
    if not args.keep:
        os.remove(path)
    if plat != 'win':
        import stat
        bin_dir = os.path.join(base_dir, 'bin')
        for f in os.listdir(bin_dir):
            if '.' in f:
                continue
            os.chmod(os.path.join(bin_dir, f), stat.S_IEXEC | stat.S_IWUSR | stat.S_IRUSR)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('temp_directory', nargs='?', default='', help='Full path to the directory to save to')
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--redownload', action='store_true')

    args = parser.parse_args()
    download(args)
