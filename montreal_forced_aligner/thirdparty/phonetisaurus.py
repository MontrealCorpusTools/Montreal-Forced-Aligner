import sys
import shutil, os, stat
import subprocess
import re
from ..config import TEMP_DIR

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


def collect_phonetisaurus_binaries(directory):
    bin_out = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
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


def validate_phonetisaurus_binaries():
    bin_out = os.path.join(TEMP_DIR, 'thirdparty', 'bin')
    if not os.path.exists(bin_out):
        print('The folder {} does not exist'.format(bin_out))
        return False
    bin_files = os.listdir(bin_out)
    plat = sys.platform
    not_found = []
    for lib_file in included_libraries[plat]:
        if lib_file not in bin_files:
            not_found.append(lib_file)
    for bin_file in included_filenames:
        bin_file += exe_ext
        if bin_file not in bin_files:
            not_found.append(bin_file)
    if not_found:
        print('The following phonetisaurus binaries were not found in {}: {}'.format(bin_out, ', '.join(sorted(not_found))))
        return False
    print('All required phonetisaurus binaries were found!')
    return True
