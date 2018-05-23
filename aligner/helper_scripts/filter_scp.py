#!/usr/bin/python

import sys
import argparse

if __name__ == '__main__':
	# Python for https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/filter_scp.pl
	# Stdout will be the output, and piped into following functions.
	# Main idea: compare stdin (or optional scp argument) and valid_uttlist, and take the intersection or the difference.

	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--exclude', action='store_true', required=False)
	parser.add_argument('valid_uttlist', action='store', type=str)
	parser.add_argument('scp', nargs='?')
	args = parser.parse_args()

	input_lines = []
	if args.scp:
		with open(args.scp, 'r') as fp:
			input_lines = fp.readlines()
	else:
		for line in sys.stdin:
			input_lines.append(line)

	uttlist = []
	with open(args.valid_uttlist, 'r') as fp:
		uttlist_lines = fp.readlines()
		for line in uttlist_lines:
			utt_id = line.split()[0]
			uttlist.append(utt_id)

	checker = False
	not_excluded, excluded = [], []
	for utt_id in uttlist:
		for line in input_lines:
			if utt_id in line:
				checker = True
				if not args.exclude:
					not_excluded.append(line)
			elif checker == False and args.exclude:
				excluded.append(line)

	if not args.exclude:
		for line in not_excluded:
			print(line.strip())
	else:
		for line in excluded:
			print(line.strip())
