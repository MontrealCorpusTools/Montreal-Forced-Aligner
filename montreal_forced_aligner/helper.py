from __future__ import annotations
import os
import numpy
from typing import TYPE_CHECKING, Any, List, Tuple, Collection, Union, Optional, Type
if TYPE_CHECKING:
    from .config import ConfigDict
    from .corpus import CorpusMappingType, CorpusGroupedType, ScpType

import sys
import textwrap
from colorama import Fore, Style

Labels = List[Any]


class TerminalPrinter(object):
    def __init__(self):
        from .config import load_global_config
        c = load_global_config()
        self.colors = {}
        self.colors['bright'] = ''
        self.colors['green'] = ''
        self.colors['red'] = ''
        self.colors['blue'] = ''
        self.colors['cyan'] = ''
        self.colors['yellow'] = ''
        self.colors['reset'] = ''
        self.colors['normal'] = ''
        self.width = c['terminal_width']
        if c['terminal_colors']:
            self.colors['bright'] = Style.BRIGHT
            self.colors['green'] = Fore.GREEN
            self.colors['red'] = Fore.RED
            self.colors['blue'] = Fore.BLUE
            self.colors['cyan'] = Fore.CYAN
            self.colors['yellow'] = Fore.YELLOW
            self.colors['reset'] = Style.RESET_ALL
            self.colors['normal'] = Style.NORMAL

    def colorize(self, text: str, color: str) -> str:
        return f"{self.colors[color]}{text}{self.colors['reset']}"

    def print_block(self, block:dict, starting_level:int = 1) -> None:
        for k, v in block.items():
            value_color = None
            key_color = None
            value = ''
            if isinstance(k, tuple):
                k, key_color = k

            if isinstance(v, tuple):
                value, value_color = v
            elif not isinstance(v, dict):
                value = v
            self.print_information_line(k, value, key_color, value_color, starting_level)
            if isinstance(v, dict):
                self.print_block(v, starting_level=starting_level+1)
        print()

    def print_config(self, configuration: ConfigDict) -> None:
        for k, v in configuration.items():
            if 'name' in v:
                name = v['name']
                name_color = None
                if isinstance(name, tuple):
                    name, name_color = name
                self.print_information_line(k, name, value_color=name_color, level=0)
            if 'data' in v:
                self.print_block(v['data'])

    def print_information_line(self, key: str, value: Union[str, list, tuple, set, bool],
                               key_color: Optional[str]=None, value_color: Optional[str]=None, level: int=1) -> None:
        if key_color is None:
            key_color = 'bright'
        if value_color is None:
            value_color = 'yellow'
            if isinstance(value, bool):
                if value:
                    value_color = 'green'
                else:
                    value_color = 'red'
        if isinstance(value, (list, tuple, set)):
            value = comma_join([self.colorize(x, value_color) for x in sorted(value)])
        else:
            value = self.colorize(value, value_color)
        indent = ('  ' * level) + '-'
        subsequent_indent = ('  ' * (level+1))
        if key:
            key = f" {key}:"
            subsequent_indent += ' '*(len(key))
        wrapper = textwrap.TextWrapper(initial_indent=indent, subsequent_indent=subsequent_indent, width=self.width)
        print(wrapper.fill(f"{self.colorize(key, key_color)} {value}"))



def comma_join(sequence: Collection[Any]) -> str:
    if len(sequence) < 3:
        return ' and '.join(sequence)
    return f"{', '.join(sequence[:-1])}, and {sequence[-1]}"


def make_safe(element: Any) -> str:
    if isinstance(element, list):
        return ' '.join(map(make_safe, element))
    return str(element)

def make_scp_safe(string: str) -> str:
    return string.replace(' ', '_MFASPACE_')

def load_scp_safe(string: str) -> str:
    return string.replace('_MFASPACE_', ' ')

def output_mapping(mapping: CorpusMappingType, path: str, skip_safe:bool = False) -> None:
    if not mapping:
        return
    with open(path, 'w', encoding='utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, (list, set, tuple)):
                v = ' '.join(map(str, v))
            elif not skip_safe:
                v = make_scp_safe(v)
            f.write(f'{make_scp_safe(k)} {v}\n')


def save_scp(scp: ScpType, path: str, sort: Optional[bool]=True, multiline: Optional[bool]=False) -> None:
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
                f.write(f'{make_safe(line[0])}\n{make_safe(line[1])}\n')
            else:
                f.write(f"{' '.join(map(make_safe, line))}\n")


def save_groups(groups: CorpusGroupedType, seg_dir: str, pattern: str, multiline: Optional[bool]=False) -> None:
    for i, g in enumerate(groups):
        path = os.path.join(seg_dir, pattern.format(i))
        if os.path.exists(path):
            continue
        if not g:
            continue
        save_scp(g, path, multiline=multiline)


def load_scp(path: str, data_type: Optional[Type]=str) -> CorpusMappingType:
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


def filter_scp(uttlist: List[str], scp: Union[str, List[str]], exclude: Optional[bool]=False) -> List[str]:
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