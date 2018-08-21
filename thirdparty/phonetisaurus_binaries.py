import sys
import shutil, os, stat
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


def collect_binaries(directory):
    outdirectory = os.path.dirname(os.path.realpath(__file__))
    bin_out = os.path.join(outdirectory, 'bin')
    os.makedirs(bin_out, exist_ok=True)

    for root, dirs, files in os.walk(directory, followlinks=True):
        cur_dir = os.path.basename(root)
        for name in files:
            ext = os.path.splitext(name)
            (key, value) = ext
            out_path = os.path.join(bin_out, name)
            if value == exe_ext:
                if key not in included_filenames:
                    continue
                in_path = os.path.join(root, name)
                shutil.copyfile(in_path, out_path)
                st = os.stat(out_path)
                os.chmod(out_path, st.st_mode | stat.S_IEXEC)
                if sys.platform == 'darwin':
                    p = subprocess.Popen(['otool', '-L', out_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                    output, err = p.communicate()
                    rc = p.returncode
                    output = output.decode()
                    libs = dylib_pattern.findall(output)
                    for l in libs:
                        if l.startswith('/usr') and not l.startswith('/usr/local'):
                            continue
                        lib = os.path.basename(l)
                        subprocess.call(['install_name_tool', '-change', l, '@loader_path/' + lib, out_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    directory = os.path.expanduser(args.dir)
    collect_binaries(directory)
