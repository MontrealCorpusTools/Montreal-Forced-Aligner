#!/bin/sh


if [ `uname` == Darwin ]; then
	pyinstaller -w --clean -n montreal-forced-aligner --debug -y --additional-hooks-dir=freezing/hooks aligner/commandline.py

fi