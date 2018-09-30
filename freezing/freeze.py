import os
import sys
import shutil
import subprocess
from PyInstaller.__main__ import run as pyinstaller_run

if sys.platform == 'win32':
    exe_ext = '.exe'
else:
    exe_ext = ''

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dist_dir = os.path.join(root_dir, 'dist')

shutil.rmtree(dist_dir, ignore_errors=True)

# Generate executables

common_options = ['--clean', '-y',
                  '--additional-hooks-dir=' + os.path.join(root_dir, 'freezing', 'hooks'),
                  '--exclude-module=tkinter',
                  '--exclude-module=matplotlib',
                  '--exclude-module=pytz',
                  '--exclude-module=sphinx',
                  # '--exclude-module=numpy',
                  '--exclude-module=scipy']

executables = ['train_and_align', 'align',
               'generate_dictionary', 'train_g2p',
               'validate_dataset']

executable_template = os.path.join(root_dir, 'aligner', 'command_line', '{}.py')
for e in executables:
    script_name = executable_template.format(e)
    print(script_name)
    com = common_options + [script_name]
    pyinstaller_run(pyi_args=com)

mfa_root = os.path.join(dist_dir, 'montreal-forced-aligner')
os.makedirs(mfa_root)
bin_dir = os.path.join(mfa_root, 'bin')

if sys.platform == 'win32':
    for i, e in enumerate(executables):
        orig_dir = os.path.join(dist_dir, e)
        if i == 0:
            shutil.move(orig_dir, bin_dir)
            os.rename(os.path.join(bin_dir, e + exe_ext), os.path.join(bin_dir, 'mfa_' + e + exe_ext))
        else:
            shutil.move(os.path.join(orig_dir, e + exe_ext), os.path.join(bin_dir, 'mfa_' + e + exe_ext))
else:
    lib_dir = os.path.join(mfa_root, 'lib')
    os.makedirs(bin_dir)
    for i, e in enumerate(executables):
        orig_dir = os.path.join(dist_dir, e)
        if i == 0:
            shutil.move(orig_dir, lib_dir)
        else:
            shutil.move(os.path.join(orig_dir, e + exe_ext), os.path.join(lib_dir, e + exe_ext))
        lib_exe_path = os.path.join(lib_dir, e + exe_ext)
        os.symlink(os.path.relpath(lib_exe_path, bin_dir), os.path.join(bin_dir, 'mfa_' + e + exe_ext))

# Copy thirdparty binaries

orig_thirdparty_dir = os.path.join(root_dir, 'thirdparty', 'bin')

if sys.platform == 'win32':
    out_root_dir = os.path.join(root_dir, 'dist', 'montreal-forced-aligner', 'bin')
else:
    out_root_dir = os.path.join(root_dir, 'dist', 'montreal-forced-aligner', 'lib')

out_dir = os.path.join(out_root_dir, 'thirdparty', 'bin')

os.makedirs(out_dir, exist_ok=True)

for f in os.listdir(orig_thirdparty_dir):
    if f.startswith('libopenblas') and sys.platform != 'win32':
        for f2 in os.listdir(out_root_dir):
            if f2.startswith('libopenblas'):
                lib_exe_path = os.path.join(out_root_dir, f2)
                os.symlink(os.path.relpath(lib_exe_path, out_dir), os.path.join(out_dir, f))
                break
    else:
        shutil.copyfile(os.path.join(orig_thirdparty_dir, f), os.path.join(out_dir, f))
        shutil.copystat(os.path.join(orig_thirdparty_dir, f), os.path.join(out_dir, f))

# Copy pretrained models

pretrained_dir = os.path.join(mfa_root, 'pretrained_models')
pretrained_root_dir = os.path.join(root_dir, 'pretrained_models')
os.makedirs(pretrained_dir)

for f in os.listdir(pretrained_root_dir):
    if f.endswith('.zip'):
        shutil.copyfile(os.path.join(pretrained_root_dir, f), os.path.join(pretrained_dir, f))

# Create distributable archive

for d in executables:
    d = os.path.join(dist_dir, d)
    if os.path.exists(d):
        shutil.rmtree(d)

if sys.platform == 'win32':
    plat = 'win64'
elif sys.platform == 'darwin':
    plat = 'macosx'
else:
    plat = 'linux'

zip_path = os.path.join(dist_dir, 'montreal-forced-aligner_{}'.format(plat))

if sys.platform == 'linux':
    format = 'gztar'
else:
    format = 'zip'

if sys.platform == 'darwin':
    subprocess.run(['zip', '-y', '-r', 'montreal-forced-aligner_{}.zip'.format(plat), 'montreal-forced-aligner'], cwd=dist_dir)
else:
    shutil.make_archive(zip_path, format, dist_dir)
