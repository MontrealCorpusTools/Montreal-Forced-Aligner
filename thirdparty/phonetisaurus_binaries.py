import sys
import shutil, os
import argparse
import subprocess
import re

if sys.platform == 'win32':
    exe_ext = '.exe'
elif sys.platform == 'darwin':
    exe_ext = ''
else:
    exe_ext = ''

included_filenames = ['phonetisaurus-align', 'phonetisaurus-arpa2wfst', 'phonetisaurus-g2pfst']

linux_libraries = []
included_libraries = {'linux': linux_libraries,
                      'win32': [],
                      'darwin': linux_libraries}

dylib_pattern = re.compile(r'\s*(.*)\s+\(')


def CollectBinaries(directory):
    outdirectory = os.path.dirname(os.path.realpath(__file__))
    bin_out = os.path.join(outdirectory, 'bin')
    os.makedirs(bin_out, exist_ok=True)

    if sys.platform == 'win32':
        src_dir = directory
    else:
        src_dir = os.path.join(directory, 'src')
    for root, dirs, files in os.walk(src_dir, followlinks=True):
        cur_dir = os.path.basename(root)
        for name in files:
            ext = os.path.splitext(name)
            (key, value) = ext
            bin_name = os.path.join(bin_out, name)
            if not os.path.exists(bin_name):
                if value == exe_ext:
                    if key not in included_filenames:
                        continue
                    shutil.copy(os.path.join(root, name), bin_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    directory = os.path.expanduser(args.dir)
    CollectBinaries(directory)
