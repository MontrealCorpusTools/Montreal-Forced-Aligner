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

    download_link = 'https://montrealcorpustools.github.io/Montreal-Forced-Aligner/thirdparty_precompiled/{}.zip'.format(plat)
    path = os.path.join(base_dir, '{}.zip'.format(plat))
    with tqdm(unit='B', unit_scale=True, miniters=1) as t:
        filename, headers = urlretrieve(download_link, path, reporthook=tqdm_hook(t), data=None)
    shutil.unpack_archive(filename, base_dir)
    os.remove(path)
    return True

if __name__ == '__main__':
    download()
