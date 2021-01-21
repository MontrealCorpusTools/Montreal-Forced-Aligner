import os
import shutil
import numpy
from typing import Any, List, Tuple
import logging
import sys

from .exceptions import ThirdpartyError, KaldiProcessingError

Labels = List[Any]


def thirdparty_binary(binary_name):
    bin_path = shutil.which(binary_name)
    if bin_path is None:
        raise ThirdpartyError("Could not find '{}'.  Please ensure that you have downloaded the correct binaries.".format(binary_name))
    return bin_path


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


def output_mapping(mapping, path):
    with open(path, 'w', encoding='utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            f.write('{} {}\n'.format(k, v))


def save_scp(scp, path, sort=True, multiline=False):
    with open(path, 'w', encoding='utf8') as f:
        if sort:
            scp = sorted(scp)
        for line in scp:
            if multiline:
                f.write('{}\n{}\n'.format(make_safe(line[0]), make_safe(line[1])))
            else:
                f.write('{}\n'.format(' '.join(map(make_safe, line))))


def save_groups(groups, seg_dir, pattern, multiline=False):
    for i, g in enumerate(groups):
        path = os.path.join(seg_dir, pattern.format(i))
        save_scp(g, path, multiline=multiline)


def load_scp(path):
    """
    Load a Kaldi script file (.scp)

    See http://kaldi-asr.org/doc/io.html#io_sec_scp_details for more information

    Parameters
    ----------
    path : str
        Path to Kaldi script file

    Returns
    -------
    dict
        Dictionary where the keys are the first couple and the values are all
        other columns in the script file

    """
    scp = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line_list = line.split()
            key = line_list.pop(0)
            if len(line_list) == 1:
                value = line_list[0]
            else:
                value = [ x for x in line_list if x not in ['[', ']']]
            scp[key] = value
    return scp


def filter_scp(uttlist, scp, exclude=False):
    # Modelled after https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/filter_scp.pl
    # Used in DNN recipes
    # Scp could be either a path or just the list

    # Get lines of scp file
    if not isinstance(scp, list) and os.path.exists(scp):
        # If path provided
        with open(scp, 'r') as fp:
            input_lines = fp.readlines()
    else:
        # If list provided
        input_lines = scp

    # Get lines of valid_uttlist in a list, and a list of utterance IDs.
    uttlist = set(uttlist)
    filtered = []
    for line in input_lines:
        line_id = line.split()[0]
        if exclude:
            if line_id not in uttlist:
                filtered.append(line)
        else:
            if line_id in uttlist:
                filtered.append(line)
    return filtered


def edit_distance(x: Labels, y: Labels) -> int:
    # For a more expressive version of the same, see:
    #
    #     https://gist.github.com/kylebgorman/8034009
    idim = len(x) + 1
    jdim = len(y) + 1
    table = numpy.zeros((idim, jdim), dtype=numpy.uint8)
    table[1:, 0] = 1
    table[0, 1:] = 1
    for i in range(1, idim):
        for j in range(1, jdim):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                c1 = table[i - 1][j]
                c2 = table[i][j - 1]
                c3 = table[i - 1][j - 1]
                table[i][j] = min(c1, c2, c3) + 1
    return int(table[-1][-1])


def score(args: Tuple[Labels, Labels]) -> Tuple[int, int]:
    gold, hypo = args
    """Computes sufficient statistics for LER calculation."""
    edits = edit_distance(gold, hypo)
    return edits, len(gold)


def setup_logger(identifier, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, identifier + '.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    print(log_path)
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def parse_logs(log_directory):
    error_logs = []
    for name in os.listdir(log_directory):
        log_path = os.path.join(log_directory, name)
        with open(log_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if 'error while loading shared libraries: libopenblas.so.0' in line:
                    raise ThirdpartyError('There was a problem locating libopenblas.so.0. '
                                          'Try installing openblas via system package manager?')
                if line.startswith('ERROR') or line.startswith('ASSERTION_FAILED'):
                    error_logs.append(log_path)
                    break
    if error_logs:
        raise KaldiProcessingError(error_logs)


def log_kaldi_errors(error_logs, logger):
    logger.debug('There were {} kaldi processing files that had errors:'.format(len(error_logs)))
    for path in error_logs:
        logger.debug('')
        logger.debug(path)
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                logger.debug('\t' + line.strip())