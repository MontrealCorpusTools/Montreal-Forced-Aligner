#!/bin/sh

rm -rf dist

pyinstaller --clean -y \
--additional-hooks-dir=freezing/hooks \
--exclude-module tkinter \
--exclude-module matplotlib \
--exclude-module pytz \
--exclude-module sphinx \
 aligner/command_line/train_and_align.py

pyinstaller --clean -y \
--additional-hooks-dir=freezing/hooks \
--exclude-module tkinter \
--exclude-module matplotlib \
--exclude-module pytz \
--exclude-module sphinx \
aligner/command_line/align.py

cd dist
mkdir montreal-forced-aligner

mv train_and_align/ montreal-forced-aligner/lib
mv align/align montreal-forced-aligner/lib/align

cd montreal-forced-aligner
mkdir bin
cd bin
ln -s ../lib/mfa_train_and_align train_and_align
ln -s ../lib/mfa_align align

cd ../..
zip -r montreal-forced-aligner.zip montreal-forced-aligner/

cd ..
