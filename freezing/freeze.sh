#!/bin/sh

pyinstaller -F --clean -n montreal-forced-aligner -y --additional-hooks-dir=freezing/hooks aligner/command_line/train_and_align.py
