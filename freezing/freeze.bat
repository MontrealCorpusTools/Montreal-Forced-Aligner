
rmdir /s /q dist

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
--exclude-module tkinter ^
aligner/command_line/train_and_align.py

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
--exclude-module tkinter ^
aligner/command_line/align.py


cd dist
mkdir montreal-forced-aligner

move train_and_align montreal-forced-aligner\bin
move montreal-forced-aligner\bin\train_and_align.exe montreal-forced-aligner\bin\mfa_train_and_align.exe
move align\align.exe montreal-forced-aligner\bin\mfa_align.exe

cd montreal-forced-aligner
mkdir pretrained_models

cd ..
copy ..\pretrained_models montreal-forced-aligner\pretrained_models

cd ..
