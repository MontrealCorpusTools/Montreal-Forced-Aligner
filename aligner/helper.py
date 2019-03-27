import os
import shutil


def thirdparty_binary(binary_name):
    return shutil.which(binary_name)


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
    '''
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

    '''
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
                value = line_list
            scp[key] = value
    return scp


def filter_scp(uttlist, scp, exclude=False):
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
