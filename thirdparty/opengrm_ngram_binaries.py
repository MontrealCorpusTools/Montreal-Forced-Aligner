import sys
import shutil, os
import argparse
import subprocess
import re

if sys.platform == 'win32':
    exe_ext = '.exe'
    lib_ext = '.dll'
elif sys.platform == 'darwin':
    exe_ext = ''
    lib_ext = ['.dylib']
else:
    exe_ext = ''
    lib_ext = ['.so', '.so.1']

included_filenames = ['ngramcount', 'ngrammake', 'ngramsymbols', 'ngramprint']

linux_libraries = []
included_libraries = {'linux': linux_libraries,
                      'win32': [],
                      'darwin': ['libngram.2.dylib', 'libngramhist.2.dylib']}

dylib_pattern = re.compile(r'\s*(.*)\s+\(')


def CollectBinaries(directory):
    outdirectory = os.path.dirname(os.path.realpath(__file__))
    bin_out = os.path.join(outdirectory, 'bin')
    os.makedirs(bin_out, exist_ok=True)

    if sys.platform == 'win32':
        src_dir = directory
    else:
        src_dir = os.path.join(directory, 'src', 'bin', '.libs')
    for root, dirs, files in os.walk(src_dir, followlinks=True):
        cur_dir = os.path.basename(root)
        for name in files:
            ext = os.path.splitext(name)
            (key, value) = ext
            if value == exe_ext and key in included_filenames:
                bin_name = os.path.join(bin_out, name)
                shutil.copy(os.path.join(root, name), bin_out)
                if sys.platform == 'darwin':
                    p = subprocess.Popen(['otool', '-L', bin_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                    output, err = p.communicate()
                    rc = p.returncode
                    output = output.decode()
                    libs = dylib_pattern.findall(output)
                    for l in libs:
                        if l.startswith('/usr') and not l.startswith('/usr/local'):
                            continue
                        lib = os.path.basename(l)
                        subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, bin_name])
    if sys.platform == 'darwin':
        lib_dir = os.path.join(directory, 'src', 'lib', '.libs')
        for f in os.listdir(lib_dir):
            for lib in included_libraries[sys.platform]:
                if f == lib:
                    bin_name = os.path.join(bin_out, lib)
                    shutil.copyfile(os.path.join(lib_dir, f), bin_name)
                    if sys.platform == 'darwin':
                        p = subprocess.Popen(['otool', '-L', bin_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
                        output, err = p.communicate()
                        rc = p.returncode
                        output = output.decode()
                        libs = dylib_pattern.findall(output)
                        for l in libs:
                            if l.startswith('/usr') and not l.startswith('/usr/local'):
                                continue
                            lib = os.path.basename(l)
                            subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, bin_name])
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    directory = os.path.expanduser(args.dir)
    CollectBinaries(directory)
