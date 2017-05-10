#!/bin/sh

rm -rf dist

pyinstaller --clean -y \
--additional-hooks-dir=freezing/hooks \
--exclude-module tkinter \
--exclude-module matplotlib \
--exclude-module pytz \
--exclude-module sphinx \
--exclude-module numpy \
--exclude-module scipy \
 aligner/command_line/train_and_align.py

pyinstaller --clean -y \
--additional-hooks-dir=freezing/hooks \
--exclude-module tkinter \
--exclude-module matplotlib \
--exclude-module pytz \
--exclude-module sphinx \
--exclude-module numpy \
--exclude-module scipy \
aligner/command_line/align.py

pyinstaller --clean -y \
--additional-hooks-dir=freezing/hooks \
--exclude-module tkinter \
--exclude-module matplotlib \
--exclude-module pytz \
--exclude-module sphinx \
--exclude-module numpy \
--exclude-module scipy \
aligner/command_line/generate_dictionary.py

pyinstaller --clean -y \
--additional-hooks-dir=freezing/hooks \
--exclude-module tkinter \
--exclude-module matplotlib \
--exclude-module pytz \
--exclude-module sphinx \
--exclude-module numpy \
--exclude-module scipy \
aligner/command_line/train_g2p.py

cd dist
mkdir montreal-forced-aligner

mv train_and_align/ montreal-forced-aligner/lib
mv align/align montreal-forced-aligner/lib/align
mv generate_dictionary/generate_dictionary montreal-forced-aligner/lib/generate_dictionary
mv train_g2p/train_g2p montreal-forced-aligner/lib/train_g2p

cd montreal-forced-aligner
mkdir bin
cd bin
ln -s ../lib/train_and_align mfa_train_and_align
ln -s ../lib/align mfa_align
ln -s ../lib/generate_dictionary mfa_generate_dictionary
ln -s ../lib/train_g2p mfa_train_g2p

cd ../..
cp -r ../pretrained_models montreal-forced-aligner/pretrained_models

python3 ../freezing/freeze_final.py

if [ `uname` == Darwin ]; then
zip -y -r montreal-forced-aligner.zip montreal-forced-aligner
else
tar -zcvf montreal-forced-aligner.tar.gz montreal-forced-aligner
fi

cd ..
