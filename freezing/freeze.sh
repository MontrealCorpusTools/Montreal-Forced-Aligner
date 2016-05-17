#!/bin/sh
if [ `uname` == Darwin ]; then

    pyinstaller -w --clean -n montreal-forced-aligner -y --additional-hooks-dir=freezing/hooks aligner/command_line/train_and_align.py

else

    pyinstaller -F --clean -n montreal-forced-aligner -y --additional-hooks-dir=freezing/hooks aligner/command_line/train_and_align.py

fi
