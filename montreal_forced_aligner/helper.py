import os
import shutil
import numpy
from typing import Any, List, Tuple
import logging
import sys
import yaml
from .exceptions import ThirdpartyError, KaldiProcessingError

Labels = List[Any]


def thirdparty_binary(binary_name):
    bin_path = shutil.which(binary_name)
    if bin_path is None:
        if binary_name in ['fstcompile', 'fstarcsort', 'fstconvert'] and sys.platform != 'win32':
            raise ThirdpartyError("Could not find '{}'.  Please ensure that you are in an environment that has the "
                                  "openfst conda package installed, or that the openfst binaries are on your path "
                                  "if you compiled them yourself.".format(binary_name))
        else:
            raise ThirdpartyError("Could not find '{}'.  Please ensure that you have downloaded the "
                                  "correct binaries.".format(binary_name))
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

def make_scp_safe(string):

    return string.replace(' ', '_MFASPACE_')

def load_scp_safe(string):
    return string.replace('_MFASPACE_', ' ')

def output_mapping(mapping, path):
    with open(path, 'w', encoding='utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, (list, set, tuple)):
                v = ' '.join(map(str, v))
            else:
                v = make_scp_safe(v)
            f.write(f'{make_scp_safe(k)} {v}\n')


def save_scp(scp, path, sort=True, multiline=False):
    if sys.platform == 'win32':
        newline = ''
    else:
        newline = None
    if not scp:
        return
    with open(path, 'w', encoding='utf8', newline=newline) as f:
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
        if os.path.exists(path):
            continue
        save_scp(g, path, multiline=multiline)


def load_scp(path, data_type=str):
    """
    Load a Kaldi script file (.scp)

    See http://kaldi-asr.org/doc/io.html#io_sec_scp_details for more information

    Parameters
    ----------
    path : str
        Path to Kaldi script file
    data_type : type
        Type to coerce the data to

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
            key = load_scp_safe(line_list.pop(0))
            if len(line_list) == 1:
                value = data_type(line_list[0])
                if isinstance(value, str):
                    value = load_scp_safe(value)
            else:
                value = [ data_type(x) for x in line_list if x not in ['[', ']']]
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


def score(gold : Labels, hypo: (Labels, List)) -> Tuple[int, int]:
    """Computes sufficient statistics for LER calculation."""
    if isinstance(hypo, list):
        edits = 100000
        for h in hypo:
            e = edit_distance(gold, h)
            if e < edits:
                edits = e
            if not edits:
                break
    else:
        edits = edit_distance(gold, hypo)
    return edits, len(gold)


def setup_logger(identifier, output_directory, console_level='info'):
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.join(output_directory, identifier + '.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding='utf8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, console_level.upper()))
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_config(logger, config):
    stream = yaml.dump(config)
    logger.debug(stream)


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
                if 'GLIBC_2.27' in line or'GLIBCXX_3.4.20' in line:
                    raise ThirdpartyError('There was a problem with the version of system libraries that Kaldi was linked against. '
                                          'Try compiling Kaldi on your machine and collecting the binaries via '
                                          'the `mfa thirdparty kaldi` command.')
                if 'sox FAIL formats' in line:
                    f = line.split(' ')[-1]
                    raise ThirdpartyError('Your version of sox does not support the file format in your corpus. '
                                          'Try installing another version of sox with support for {}.'.format(f))
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