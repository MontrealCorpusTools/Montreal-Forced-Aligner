import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs)

hiddenimports = ['six', 'packaging', 'packaging.version', 'packaging.specifiers',
                 'packaging.requirements', 'numpy']

if sys.platform == 'win32':
    hiddenimports += ['win32api']

datas = collect_data_files('aligner')
#thirdparty_dir = os.path.abspath(os.path.join('thirdparty', 'bin'))
#binaries = [(os.path.join(thirdparty_dir, x), 'thirdparty/bin')
#            for x in os.listdir(thirdparty_dir)
#            ]
