import os
import sys
import shutil

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
orig_dir = os.path.join(root_dir, 'thirdparty', 'bin')

if sys.platform == 'win32':
    out_dir = os.path.join(root_dir, 'dist', 'montreal-forced-aligner', 'bin', 'thirdparty', 'bin')
else:
    out_dir = os.path.join(root_dir, 'dist', 'montreal-forced-aligner', 'lib', 'thirdparty', 'bin')

os.makedirs(out_dir, exist_ok=True)

for f in os.listdir(orig_dir):
    shutil.copyfile(os.path.join(orig_dir, f), os.path.join(out_dir, f))
    shutil.copystat(os.path.join(orig_dir, f), os.path.join(out_dir, f))
