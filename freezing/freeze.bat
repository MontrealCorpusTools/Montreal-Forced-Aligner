
pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
aligner/command_line/train_and_align.py

pyinstaller --clean -y ^
--additional-hooks-dir=freezing/hooks ^
aligner/command_line/align.py

