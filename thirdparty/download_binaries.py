import shutil
import os
import sys
from tqdm import tqdm
from urllib.request import urlretrieve


def tqdm_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'darwin':
        plat = 'macosx'
    elif sys.platform == 'win32':
        plat = 'win'
    else:
        plat = 'linux'
    print('Downloading precompiled binaries for {}...'.format(plat))

    download_link = 'http://mlmlab.org/mfa/precompiled_binaries/{}.zip'.format(
        plat)
    path = os.path.join(base_dir, '{}.zip'.format(plat))
    with tqdm(unit='B', unit_scale=True, miniters=1) as t:
        filename, headers = urlretrieve(download_link, path, reporthook=tqdm_hook(t), data=None)
    shutil.unpack_archive(filename, base_dir)
    os.remove(path)
    if plat != 'win':
        import stat
        bin_dir = os.path.join(base_dir, 'bin')
        for f in os.listdir(bin_dir):
            if '.' in f:
                continue
            os.chmod(os.path.join(bin_dir, f), stat.S_IEXEC|stat.S_IWUSR|stat.S_IRUSR)
    return True


if __name__ == '__main__':
    download()
