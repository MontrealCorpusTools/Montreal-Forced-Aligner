
import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs)

thirdparty_dir = os.path.abspath(os.path.join('thirdparty','bin'))
binaries = [( os.path.join(thirdparty_dir, x), 'thirdparty/bin') for x in os.listdir(thirdparty_dir)
    ]


