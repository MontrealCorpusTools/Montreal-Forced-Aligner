
rmdir /s /q dist

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
--exclude-module tkinter ^
--exclude-module matplotlib ^
--exclude-module pytz ^
--exclude-module sphinx ^
--exclude-module numpy ^
--exclude-module scipy ^
aligner/command_line/train_and_align.py

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
--exclude-module tkinter ^
--exclude-module matplotlib ^
--exclude-module pytz ^
--exclude-module sphinx ^
--exclude-module numpy ^
--exclude-module scipy ^
aligner/command_line/align.py

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
--exclude-module tkinter ^
--exclude-module matplotlib ^
--exclude-module pytz ^
--exclude-module sphinx ^
--exclude-module numpy ^
--exclude-module scipy ^
aligner/command_line/generate_dictionary.py

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
--exclude-module tkinter ^
--exclude-module matplotlib ^
--exclude-module pytz ^
--exclude-module sphinx ^
--exclude-module numpy ^
--exclude-module scipy ^
aligner/command_line/train_g2p.py


cd dist
mkdir montreal-forced-aligner

move train_and_align montreal-forced-aligner\bin
move montreal-forced-aligner\bin\train_and_align.exe montreal-forced-aligner\bin\mfa_train_and_align.exe
move align\align.exe montreal-forced-aligner\bin\mfa_align.exe
move generate_dictionary\generate_dictionary.exe montreal-forced-aligner\bin\mfa_generate_dictionary.exe
move train_g2p\train_g2p.exe montreal-forced-aligner\bin\mfa_train_g2p.exe

cd montreal-forced-aligner
mkdir pretrained_models

cd ..
copy ..\pretrained_models montreal-forced-aligner\pretrained_models

cd ..
