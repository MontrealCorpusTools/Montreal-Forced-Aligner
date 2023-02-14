"""
Helper functions
================

"""
from __future__ import annotations

import functools
import itertools
import json
import logging
import re
import shutil
import sys
import typing
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import ansiwrap
import dataclassy
import numpy
import yaml
from Bio import pairwise2
from colorama import Fore, Style

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.textgrid import CtmInterval


__all__ = [
    "TerminalPrinter",
    "comma_join",
    "make_safe",
    "make_scp_safe",
    "load_scp",
    "load_scp_safe",
    "score_wer",
    "edit_distance",
    "output_mapping",
    "parse_old_features",
    "compare_labels",
    "overlap_scoring",
    "align_phones",
    "split_phone_position",
    "CustomFormatter",
    "configure_logger",
    "mfa_open",
    "load_configuration",
]


@contextmanager
def mfa_open(path, mode="r", encoding="utf8", newline=""):
    if "r" in mode:
        if "b" in mode:
            file = open(path, mode)
        else:
            file = open(path, mode, encoding=encoding)
    else:
        if "b" in mode:
            file = open(path, mode)
        else:
            file = open(path, mode, encoding=encoding, newline=newline)
    try:
        yield file
    finally:
        file.close()


def load_configuration(config_path: typing.Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration file

    Parameters
    ----------
    config_path: str or Path
        Path to yaml or json configuration file

    Returns
    -------
    dict[str, Any]
        Configuration dictionary
    """
    data = {}
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    with mfa_open(config_path, "r") as f:
        if config_path.suffix == ".yaml":
            data = yaml.load(f, Loader=yaml.SafeLoader)
        elif config_path.suffix == ".json":
            data = json.load(f)
    if not data:
        return {}
    return data


def split_phone_position(phone_label: str) -> List[str]:
    """
    Splits a phone label into its original phone and it's positional label

    Parameters
    ----------
    phone_label: str
        Phone label

    Returns
    -------
    List[str]
        Phone and position
    """
    phone = phone_label
    pos = None
    try:
        phone, pos = phone_label.rsplit("_", maxsplit=1)
    except ValueError:
        pass
    return phone, pos


def parse_old_features(config: MetaDict) -> MetaDict:
    """
    Backwards compatibility function to parse old feature configuration blocks

    Parameters
    ----------
    config: dict[str, Any]
        Configuration parameters

    Returns
    -------
    dict[str, Any]
        Up to date versions of feature blocks
    """
    feature_key_remapping = {
        "type": "feature_type",
        "deltas": "uses_deltas",
    }
    skip_keys = ["lda", "fmllr"]
    if "features" in config:
        for key in skip_keys:
            if key in config["features"]:
                del config["features"][key]
        for key, new_key in feature_key_remapping.items():
            if key in config["features"]:

                config["features"][new_key] = config["features"][key]
                del config["features"][key]
    else:
        for key in skip_keys:
            if key in config:
                del config[key]
        for key, new_key in feature_key_remapping.items():
            if key in config:
                config[new_key] = config[key]
                del config[key]
    return config


def configure_logger(identifier: str, log_file: Optional[Path] = None) -> None:
    """
    Configure logging for the given identifier

    Parameters
    ----------
    identifier: str
        Logger identifier
    log_file: str
        Path to file to write all messages to
    quiet: bool
        Flag for whether logger should write to stdout
    verbose: bool
        Flag for writing debug level information to stdout
    """
    from montreal_forced_aligner.config import MfaConfiguration

    config = MfaConfiguration()
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif not config.current_profile.quiet:
        handler = logging.StreamHandler(sys.stdout)
        if config.current_profile.verbose:
            handler.setLevel(logging.DEBUG)
            logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG)
            logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter class for MFA to highlight messages and incorporate terminal options from
    the global configuration
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from montreal_forced_aligner.config import GLOBAL_CONFIG

        use_colors = GLOBAL_CONFIG.terminal_colors
        red = ""
        green = ""
        yellow = ""
        blue = ""
        reset = ""
        if use_colors:
            red = Fore.RED
            green = Fore.GREEN
            yellow = Fore.YELLOW
            blue = Fore.CYAN
            reset = Style.RESET_ALL

        self.FORMATS = {
            logging.DEBUG: (f"{blue}DEBUG{reset} - ", "%(message)s"),
            logging.INFO: (f"{green}INFO{reset} - ", "%(message)s"),
            logging.WARNING: (f"{yellow}WARNING{reset} - ", "%(message)s"),
            logging.ERROR: (f"{red}ERROR{reset} - ", "%(message)s"),
            logging.CRITICAL: (f"{red}CRITICAL{reset} - ", "%(message)s"),
        }

    def format(self, record: logging.LogRecord):
        """
        Format a given log message

        Parameters
        ----------
        record: logging.LogRecord
            Log record to format

        Returns
        -------
        str
            Formatted log message
        """
        log_fmt = self.FORMATS.get(record.levelno)
        return ansiwrap.fill(
            record.getMessage(),
            initial_indent=log_fmt[0],
            subsequent_indent=" " * len(log_fmt[0]),
            width=shutil.get_terminal_size().columns,
        )


class TerminalPrinter:
    """
    Helper class to output colorized text

    Parameters
    ----------
    print_function: Callable, optional
        Function to print information, defaults to :func:`print`

    Attributes
    ----------
    colors: dict[str, str]
        Mapping of color names to terminal codes in colorama (or empty strings
        if the global terminal_colors flag is set to False)
    """

    def __init__(self, print_function: typing.Callable = None):
        if print_function is not None:
            self.print_function = print_function
        else:
            self.print_function = print
        from montreal_forced_aligner.config import GLOBAL_CONFIG

        self.colors = {}
        self.colors["bright"] = ""
        self.colors["green"] = ""
        self.colors["red"] = ""
        self.colors["blue"] = ""
        self.colors["cyan"] = ""
        self.colors["yellow"] = ""
        self.colors["reset"] = ""
        self.colors["normal"] = ""
        self.indent_level = 0
        self.indent_size = 2
        if GLOBAL_CONFIG.terminal_colors:
            self.colors["bright"] = Style.BRIGHT
            self.colors["green"] = Fore.GREEN
            self.colors["red"] = Fore.RED
            self.colors["blue"] = Fore.BLUE
            self.colors["cyan"] = Fore.CYAN
            self.colors["yellow"] = Fore.YELLOW
            self.colors["reset"] = Style.RESET_ALL
            self.colors["normal"] = Style.NORMAL

    def error_text(self, text: Any) -> str:
        """
        Highlight text as an error

        Parameters
        ----------
        text: Any
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return self.colorize(text, "red")

    def emphasized_text(self, text: Any) -> str:
        """
        Highlight text as emphasis

        Parameters
        ----------
        text: Any
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return self.colorize(text, "bright")

    def pass_text(self, text: Any) -> str:
        """
        Highlight text as good

        Parameters
        ----------
        text: Any
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return self.colorize(text, "green")

    def warning_text(self, text: Any) -> str:
        """
        Highlight text as a warning

        Parameters
        ----------
        text: Any
            Text to highlight

        Returns
        -------
        str
            Highlighted text
        """
        return self.colorize(text, "yellow")

    @property
    def indent_string(self) -> str:
        """Indent string to use in formatting the output messages"""
        return " " * self.indent_size * self.indent_level

    def print_header(self, header: str) -> None:
        """
        Print a section header

        Parameters
        ----------
        header: str
            Section header string
        """
        self.indent_level = 0
        self.print_function("")
        underline = "*" * len(header)
        self.print_function(self.colorize(underline, "bright"))
        self.print_function(self.colorize(header, "bright"))
        self.print_function(self.colorize(underline, "bright"))
        self.print_function("")
        self.indent_level += 1

    def print_sub_header(self, header: str) -> None:
        """
        Print a subsection header

        Parameters
        ----------
        header: str
            Subsection header string
        """
        underline = "=" * len(header)
        self.print_function(self.indent_string + self.colorize(header, "bright"))
        self.print_function(self.indent_string + self.colorize(underline, "bright"))
        self.print_function("")
        self.indent_level += 1

    def print_end_section(self) -> None:
        """Mark the end of a section"""
        self.indent_level -= 1
        self.print_function("")

    def format_info_lines(self, lines: Union[list[str], str]) -> List[str]:
        """
        Format lines

        Parameters
        ----------
        lines: Union[list[str], str
            Lines to format

        Returns
        -------
        str
            Formatted string
        """
        if isinstance(lines, str):
            lines = [lines]

        for i, line in enumerate(lines):
            lines[i] = ansiwrap.fill(
                line,
                initial_indent=self.indent_string,
                subsequent_indent=" " * self.indent_size * (self.indent_level + 1),
                width=shutil.get_terminal_size().columns,
                break_on_hyphens=False,
                break_long_words=False,
                drop_whitespace=False,
            )
        return lines

    def print_info_lines(self, lines: Union[list[str], str]) -> None:
        """
        Print formatted information lines

        Parameters
        ----------
        lines: Union[list[str], str
            Lines to format
        """
        if isinstance(lines, str):
            lines = [lines]
        lines = self.format_info_lines(lines)
        for line in lines:
            self.print_function(line)

    def print_green_stat(self, stat: Any, text: str) -> None:
        """
        Print a statistic in green

        Parameters
        ----------
        stat: Any
            Statistic to print
        text: str
            Other text to follow statistic
        """
        self.print_function(self.indent_string + f"{self.colorize(stat, 'green')} {text}")

    def print_yellow_stat(self, stat, text) -> None:
        """
        Print a statistic in yellow

        Parameters
        ----------
        stat: Any
            Statistic to print
        text: str
            Other text to follow statistic
        """
        self.print_function(self.indent_string + f"{self.colorize(stat, 'yellow')} {text}")

    def print_red_stat(self, stat, text) -> None:
        """
        Print a statistic in red

        Parameters
        ----------
        stat: Any
            Statistic to print
        text: str
            Other text to follow statistic
        """
        self.print_function(self.indent_string + f"{self.colorize(stat, 'red')} {text}")

    def colorize(self, text: Any, color: str) -> str:
        """
        Colorize a string

        Parameters
        ----------
        text: Any
            Text to colorize
        color: str
            Colorama code or empty string to wrap the text

        Returns
        -------
        str
            Colorized string
        """
        return f"{self.colors[color]}{text}{self.colors['reset']}"

    def print_block(self, block: dict, starting_level: int = 1) -> None:
        """
        Print a configuration block

        Parameters
        ----------
        block: dict
            Configuration options to output
        starting_level: int
            Starting indentation level
        """
        for k, v in block.items():
            value_color = None
            key_color = None
            value = ""
            if isinstance(k, tuple):
                k, key_color = k

            if isinstance(v, tuple):
                value, value_color = v
            elif not isinstance(v, dict):
                value = v
            self.print_information_line(k, value, key_color, value_color, starting_level)
            if isinstance(v, dict):
                self.print_block(v, starting_level=starting_level + 1)
        self.print_function("")

    def print_config(self, configuration: MetaDict) -> None:
        """
        Pretty print a configuration

        Parameters
        ----------
        configuration: dict[str, Any]
            Configuration to print
        """
        for k, v in configuration.items():
            if "name" in v:
                name = v["name"]
                name_color = None
                if isinstance(name, tuple):
                    name, name_color = name
                self.print_information_line(k, name, value_color=name_color, level=0)
            if "data" in v:
                self.print_block(v["data"])

    def print_information_line(
        self,
        key: str,
        value: Any,
        key_color: Optional[str] = None,
        value_color: Optional[str] = None,
        level: int = 1,
    ) -> None:
        """
        Pretty print a given configuration line

        Parameters
        ----------
        key: str
            Configuration key
        value: Any
            Configuration value
        key_color: str
            Key color
        value_color: str
            Value color
        level: int
            Indentation level
        """
        if key_color is None:
            key_color = "bright"
        if value_color is None:
            value_color = "cyan"
            if isinstance(value, bool):
                if value:
                    value_color = "green"
                else:
                    value_color = "red"
        if isinstance(value, (list, tuple, set)):
            value = comma_join([self.colorize(x, value_color) for x in sorted(value)])
        else:
            value = self.colorize(str(value), value_color)
        indent = ("  " * level) + "-"
        subsequent_indent = "  " * (level + 1)
        if key:
            key = f" {key}:"
            subsequent_indent += " " * (len(key))

        self.print_function(
            ansiwrap.fill(
                f"{self.colorize(key, key_color)} {value}",
                width=shutil.get_terminal_size().columns,
                initial_indent=indent,
                subsequent_indent=subsequent_indent,
            )
        )


def comma_join(sequence: List[Any]) -> str:
    """
    Helper function to combine a list into a human-readable expression with commas and a
    final "and" separator

    Parameters
    ----------
    sequence: list[Any]
        Items to join together into a list

    Returns
    -------
    str
        Joined items in list format
    """
    if len(sequence) < 3:
        return " and ".join(sequence)
    return f"{', '.join(sequence[:-1])}, and {sequence[-1]}"


def make_re_character_set_safe(
    characters: typing.Collection[str], extra_strings: Optional[List[str]] = None
) -> str:
    """
    Construct a character set string for use in regex, escaping necessary characters and
    moving "-" to the initial position

    Parameters
    ----------
    characters: Collection[str]
        Characters to compile
    extra_strings: list[str], optional
        Optional other strings to put in the character class

    Returns
    -------
    str
        Character set specifier for re functions
    """
    characters = sorted(characters)
    extra = ""
    if "-" in characters:
        extra = "-"
        characters = [x for x in characters if x != "-"]
    if extra_strings:
        extra += "".join(extra_strings)
    return f"[{extra}{re.escape(''.join(characters))}]"


def make_safe(element: Any) -> str:
    """
    Helper function to make an element a string

    Parameters
    ----------
    element: Any
        Element to recursively turn into a string

    Returns
    -------
    str
        All elements combined into a string
    """
    if isinstance(element, list):
        return " ".join(map(make_safe, element))
    return str(element)


def make_scp_safe(string: str) -> str:
    """
    Helper function to make a string safe for saving in Kaldi scp files.  They use space as a delimiter, so
    any spaces in the string will be converted to "_MFASPACE_" to preserve them

    Parameters
    ----------
    string: str
        Text to escape

    Returns
    -------
    str
        Escaped text
    """
    return str(string).replace(" ", "_MFASPACE_")


def load_scp_safe(string: str) -> str:
    """
    Helper function to load previously made safe text.  All instances of "_MFASPACE_" will be converted to a
    regular space character

    Parameters
    ----------
    string: str
        String to convert

    Returns
    -------
    str
        Converted string
    """
    return string.replace("_MFASPACE_", " ")


def output_mapping(mapping: Dict[str, Any], path: str, skip_safe: bool = False) -> None:
    """
    Helper function to save mapping information (i.e., utt2spk) in Kaldi scp format

    CorpusMappingType is either a dictionary of key to value for
    one-to-one mapping case and a dictionary of key to list of values for one-to-many case.

    Parameters
    ----------
    mapping: dict[str, Any]
        Mapping to output
    path: str
        Path to save mapping
    skip_safe: bool, optional
        Flag for whether to skip over making a string safe
    """
    if not mapping:
        return
    with mfa_open(path, "w") as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, (list, set, tuple)):
                v = " ".join(map(str, v))
            elif not skip_safe:
                v = make_scp_safe(v)
            f.write(f"{make_scp_safe(k)} {v}\n")


def load_scp(path: str, data_type: Optional[Type] = str) -> Dict[str, Any]:
    """
    Load a Kaldi script file (.scp)

    Scp files in Kaldi can either be one-to-one or one-to-many, with the first element separated by
    whitespace as the key and the remaining whitespace-delimited elements the values.

    Returns a dictionary of key to value for
    one-to-one mapping case and a dictionary of key to list of values for one-to-many case.

    See Also
    --------
    :kaldi_docs:`io#io_sec_scp_details`
        For more information on the SCP format

    Parameters
    ----------
    path : str
        Path to Kaldi script file
    data_type : type
        Type to coerce the data to

    Returns
    -------
    dict[str, Any]
        Dictionary where the keys are the first column and the values are all
        other columns in the scp file

    """
    scp = {}
    with mfa_open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            line_list = line.split()
            key = load_scp_safe(line_list.pop(0))
            if len(line_list) == 1:
                value = data_type(line_list[0])
                if isinstance(value, str):
                    value = load_scp_safe(value)
            else:
                value = [data_type(x) for x in line_list if x not in ["[", "]"]]
            scp[key] = value
    return scp


def edit_distance(x: List[str], y: List[str]) -> int:
    """
    Compute edit distance between two sets of labels

    See Also
    --------
    `https://gist.github.com/kylebgorman/8034009 <https://gist.github.com/kylebgorman/8034009>`_
         For a more expressive version of this function

    Parameters
    ----------
    x: list[str]
        First sequence to compare
    y: list[str]
        Second sequence to compare

    Returns
    -------
    int
        Edit distance
    """
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


def score_g2p(gold: List[str], hypo: List[str]) -> Tuple[int, int]:
    """
    Computes sufficient statistics for LER calculation.

    Parameters
    ----------
    gold: WordData
        The reference labels
    hypo: WordData
        The hypothesized labels

    Returns
    -------
    int
        Edit distance
    int
        Length of the gold labels
    """
    for h in hypo:
        if h in gold:
            return 0, len(h)
    edits = 100000
    best_length = 100000
    for (g, h) in itertools.product(gold, hypo):
        e = edit_distance(g.split(), h.split())
        if e < edits:
            edits = e
            best_length = len(g)
        if not edits:
            best_length = len(g)
            break
    return edits, best_length


def score_wer(gold: List[str], hypo: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes word error rate and character error rate for a transcription

    Parameters
    ----------
    gold: list[str]
        The reference words
    hypo: list[str]
        The hypothesized words

    Returns
    -------
    int
        Word Edit distance
    int
        Length of the gold words labels
    int
        Character edit distance
    int
        Length of the gold characters
    """
    word_edits = edit_distance(gold, hypo)
    character_gold = list("".join(gold))
    character_hypo = list("".join(hypo))
    character_edits = edit_distance(character_gold, character_hypo)
    return word_edits, len(gold), character_edits, len(character_gold)


def compare_labels(
    ref: str, test: str, silence_phone: str, mapping: Optional[Dict[str, str]] = None
) -> int:
    """

    Parameters
    ----------
    ref: str
    test: str
    mapping: Optional[dict[str, str]]

    Returns
    -------
    int
        0 if labels match or they're in mapping, 2 otherwise
    """
    if ref == test:
        return 0
    if ref == silence_phone or test == silence_phone:
        return 10
    if mapping is not None and test in mapping:
        if isinstance(mapping[test], str):
            if mapping[test] == ref:
                return 0
        elif ref in mapping[test]:
            return 0
    ref = ref.lower()
    test = test.lower()
    if ref == test:
        return 0
    return 2


def overlap_scoring(
    first_element: CtmInterval,
    second_element: CtmInterval,
    silence_phone: str,
    mapping: Optional[Dict[str, str]] = None,
) -> float:
    r"""
    Method to calculate overlap scoring

    .. math::

       Score = -(\lvert begin_{1} - begin_{2} \rvert + \lvert end_{1} - end_{2} \rvert + \begin{cases}
                0, & if label_{1} = label_{2} \\
                2, & otherwise
                \end{cases})

    See Also
    --------
    `Blog post <https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html>`_
        For a detailed example that using this metric

    Parameters
    ----------
    first_element: :class:`~montreal_forced_aligner.data.CtmInterval`
        First CTM interval to compare
    second_element: :class:`~montreal_forced_aligner.data.CtmInterval`
        Second CTM interval
    mapping: Optional[dict[str, str]]
        Optional mapping of phones to treat as matches even if they have different symbols

    Returns
    -------
    float
        Score calculated as the negative sum of the absolute different in begin timestamps, absolute difference in end
        timestamps and the label score
    """
    begin_diff = abs(first_element.begin - second_element.begin)
    end_diff = abs(first_element.end - second_element.end)
    label_diff = compare_labels(first_element.label, second_element.label, silence_phone, mapping)
    return -1 * (begin_diff + end_diff + label_diff)


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON serialization"""

    def default(self, o: Any) -> Any:
        """Get the dictionary of a dataclass"""
        if dataclassy.functions.is_dataclass_instance(o):
            return dataclassy.asdict(o)
        if isinstance(o, set):
            return list(o)
        return dataclassy.asdict(o)


def align_phones(
    ref: List[CtmInterval],
    test: List[CtmInterval],
    silence_phone: str,
    ignored_phones: typing.Set[str] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Align phones based on how much they overlap and their phone label, with the ability to specify a custom mapping for
    different phone labels to be scored as if they're the same phone

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    silence_phone: str
        Silence phone (these are ignored in the final calculation)
    custom_mapping: dict[str, str], optional
        Mapping of phones to treat as matches even if they have different symbols

    Returns
    -------
    float
        Score based on the average amount of overlap in phone intervals
    float
        Phone error rate
    """

    if ignored_phones is None:
        ignored_phones = set()
    if custom_mapping is None:
        score_func = functools.partial(overlap_scoring, silence_phone=silence_phone)
    else:
        score_func = functools.partial(
            overlap_scoring, silence_phone=silence_phone, mapping=custom_mapping
        )

    alignments = pairwise2.align.globalcs(
        ref, test, score_func, -5, -5, gap_char=["-"], one_alignment_only=True
    )
    overlap_count = 0
    overlap_sum = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-":
                if sb.label not in ignored_phones:
                    num_insertions += 1
                else:
                    continue
            elif sb == "-":
                if sa.label not in ignored_phones:
                    num_deletions += 1
                else:
                    continue
            else:
                if sa.label in ignored_phones:
                    continue
                overlap_sum += (abs(sa.begin - sb.begin) + abs(sa.end - sb.end)) / 2
                overlap_count += 1
                if compare_labels(sa.label, sb.label, silence_phone, mapping=custom_mapping) > 0:
                    num_substitutions += 1
    if debug:
        import logging

        logger = logging.getLogger("mfa")
        logger.debug(pairwise2.format_alignment(*alignments[0]))
    if overlap_count:
        score = overlap_sum / overlap_count
    else:
        score = None
    phone_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    return score, phone_error_rate


def format_probability(probability_value: float) -> float:
    """Format a probability to have two decimal places and be between 0.01 and 0.99"""
    return min(max(round(probability_value, 2), 0.01), 0.99)


def format_correction(correction_value: float, positive_only=True) -> float:
    """Format a probability correction value to have two decimal places and be  greater than 0.01"""
    correction_value = round(correction_value, 2)
    if correction_value <= 0 and positive_only:
        correction_value = 0.01
    return correction_value
