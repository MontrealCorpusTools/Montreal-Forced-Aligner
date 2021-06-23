import shutil
import os
import sys
from tqdm import tqdm
from urllib.request import urlretrieve

from ..config import TEMP_DIR


def tqdm_hook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_binaries():
    base_dir = os.path.join(TEMP_DIR, 'thirdparty')
    os.makedirs(base_dir, exist_ok=True)
    if sys.platform == 'darwin':
        plat = 'macosx'
    elif sys.platform == 'win32':
        plat = 'win64'
    else:
        plat = 'linux'

    download_link = 'https://github.com/MontrealCorpusTools/mfa-models/raw/main/thirdparty/mfa_thirdparty_{}.zip'.format(
        plat)
    bin_dir = os.path.join(base_dir, 'bin')
    path = os.path.join(base_dir, '{}.zip'.format(plat))
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir, ignore_errors=True)
    with tqdm(unit='B', unit_scale=True, miniters=1) as t:
        filename, headers = urlretrieve(download_link, path, reporthook=tqdm_hook(t), data=None)
    shutil.unpack_archive(path, base_dir)
    os.remove(path)
    if plat != 'win':
        import stat
        bin_dir = os.path.join(base_dir, 'bin')
        for f in os.listdir(bin_dir):
            if '.' in f:
                continue
            os.chmod(os.path.join(bin_dir, f), stat.S_IEXEC | stat.S_IWUSR | stat.S_IRUSR)
    return True
