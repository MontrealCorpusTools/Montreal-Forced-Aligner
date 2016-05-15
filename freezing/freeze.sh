#!/bin/sh

pyinstaller -w --clean -n montreal-forced-aligner --debug -y --additional-hooks-dir=freezing/hooks aligner/command_line/train_and_align.py
