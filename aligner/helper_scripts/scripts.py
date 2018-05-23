#!/usr/bin/python

import sys
from random import shuffle

if __name__ == '__main__':
	# Python for https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/shuffle_list.pl
	# Stdout will be the output, and piped into following functions.
	# Main idea: shuffle the lines.
	unshuffled = []
	for line in sys.stdin:
		unshuffled.append(line)
	shuffle(unshuffled)
	for line in unshuffled:
		print(line.strip())