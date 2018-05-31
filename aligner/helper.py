import os
import sys
from collections import defaultdict, OrderedDict


def thirdparty_binary(binary_name):
    return binary_name


def make_path_safe(path):
    return '"{}"'.format(path)


def load_text(path):
    with open(path, 'r', encoding='utf8') as f:
        text = f.read().strip().lower()
    return text


def make_safe(element):
    if isinstance(element, list):
        return ' '.join(map(make_safe, element))
    return str(element)

def awk_like(path, column):
	# Grabs a column like bash awk. Columns are zero-indexed.
	col = []
	with open(path, 'r') as inf:
		f = inf.readlines()
		for line in f:
			fields = line.strip().split()
			col.append(fields[column])
	return col

def filter_scp(valid_uttlist, scp, exclude=False):
	# Modelled after https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/filter_scp.pl
	# Used in DNN recipes
	# Scp could be either a path or just the list

	# Get lines of scp file
	input_lines = []
	if not isinstance(scp, list) and os.path.exists(scp):
		# If path provided
		with open(scp, 'r') as fp:
			input_lines = fp.readlines()
	else:
		# If list provided
		input_lines = scp

	# Get lines of valid_uttlist in a list, and a list of utterance IDs.
	uttlist = []
	if os.path.exists(valid_uttlist):
		# If path provided
		with open(valid_uttlist, 'r') as fp:
			uttlist_lines = fp.readlines()
			for line in uttlist_lines:
				utt_id = line.split()[0]
				uttlist.append(utt_id)

	checker = False
	not_excluded, excluded = [], []
	for utt_id in uttlist:
		for line in input_lines:
			line_id = line.split()[0]
			if utt_id == line_id:
				checker = True
				if not exclude:
					not_excluded.append(line)
			elif checker == False and exclude:
				excluded.append(line)

	# Get rid of duplicates
	not_excluded = list(OrderedDict((x, True) for x in not_excluded).keys())
	excluded = list(OrderedDict((x, True) for x in excluded).keys())

	if not exclude:
		return not_excluded
	else:
		return excluded