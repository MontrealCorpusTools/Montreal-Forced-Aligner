"""
Helper functions
================

"""
from __future__ import annotations

import itertools
import json
import logging
import re
import typing
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import dataclassy
import numpy
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict


__all__ = [
    "comma_join",
    "make_safe",
    "make_scp_safe",
    "load_scp",
    "load_scp_safe",
    "score_wer",
    "score_g2p",
    "edit_distance",
    "output_mapping",
    "parse_old_features",
    "make_re_character_set_safe",
    "split_phone_position",
    "configure_logger",
    "mfa_open",
    "load_configuration",
    "format_correction",
    "format_probability",
    "load_evaluation_mapping",
]


console = Console(
    theme=Theme(
        {
            "logging.level.debug": "cyan",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
        }
    ),
    stderr=True,
)


@contextmanager
def mfa_open(
    path: typing.Union[Path, str],
    mode: str = "r",
    encoding: str = "utf8",
    newline: typing.Optional[str] = "",
):
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


def load_configuration(config_path: typing.Union[str, Path]) -> typing.Dict[str, typing.Any]:
    """
    Load a configuration file

    Parameters
    ----------
    config_path: :class:`~pathlib.Path`
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
            data = yaml.load(f, Loader=yaml.Loader)
        elif config_path.suffix == ".json":
            data = json.load(f)
    if not data:
        return {}
    return data


def split_phone_position(phone_label: str) -> typing.List[str]:
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


def configure_logger(identifier: str, log_file: typing.Optional[Path] = None) -> None:
    """
    Configure logging for the given identifier

    Parameters
    ----------
    identifier: str
        Logger identifier
    log_file: str
        Path to file to write all messages to
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
    if not config.current_profile.quiet:
        handler = RichHandler(
            rich_tracebacks=True, log_time_format="", console=console, show_path=False
        )
        if config.current_profile.verbose:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def comma_join(sequence: typing.List[typing.Any]) -> str:
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
    characters: typing.Collection[str], extra_strings: typing.Optional[typing.List[str]] = None
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


def make_safe(element: typing.Any) -> str:
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


def output_mapping(
    mapping: typing.Dict[str, typing.Any], path: Path, skip_safe: bool = False
) -> None:
    """
    Helper function to save mapping information (i.e., utt2spk) in Kaldi scp format

    CorpusMappingType is either a dictionary of key to value for
    one-to-one mapping case and a dictionary of key to list of values for one-to-many case.

    Parameters
    ----------
    mapping: dict[str, Any]
        Mapping to output
    path: :class:`~pathlib.Path`
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


def load_scp(
    path: Path, data_type: typing.Optional[typing.Type] = str
) -> typing.Dict[str, typing.Any]:
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
    path : :class:`~pathlib.Path`
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


def edit_distance(x: typing.List[str], y: typing.List[str]) -> int:
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


def score_g2p(gold: typing.List[str], hypo: typing.List[str]) -> typing.Tuple[int, int]:
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
    for g, h in itertools.product(gold, hypo):
        e = edit_distance(g.split(), h.split())
        if e < edits:
            edits = e
            best_length = len(g)
        if not edits:
            best_length = len(g)
            break
    return edits, best_length


def score_wer(
    gold: typing.List[str], hypo: typing.List[str], filter_brackets=True
) -> typing.Tuple[int, int, int, int]:
    """
    Computes word error rate and character error rate for a transcription

    Parameters
    ----------
    gold: list[str]
        The reference words
    hypo: list[str]
        The hypothesized words
    filter_brackets : bool
        Flag for whether to ignore bracketed words

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
    if filter_brackets:
        gold = [x for x in gold if not any(x.startswith(y) for y in "[<{")]
        hypo = [x for x in hypo if not any(x.startswith(y) for y in "[<{")]
    word_edits = edit_distance(gold, hypo)
    character_gold = list("".join(gold))
    character_hypo = list("".join(hypo))
    character_edits = edit_distance(character_gold, character_hypo)
    return word_edits, len(gold), character_edits, len(character_gold)


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON serialization"""

    def default(self, o: typing.Any) -> typing.Any:
        """Get the dictionary of a dataclass"""
        if dataclassy.functions.is_dataclass_instance(o):
            return dataclassy.asdict(o)
        if isinstance(o, set):
            return list(o)
        return dataclassy.asdict(o)


def load_evaluation_mapping(custom_mapping_path):
    with mfa_open(custom_mapping_path, "r") as f:
        mapping = yaml.load(f, Loader=yaml.Loader)
    for k, v in mapping.items():
        if isinstance(v, str):
            mapping[k] = {v}
        else:
            mapping[k] = set(v)
    return mapping


def format_probability(probability_value: float) -> float:
    """Format a probability to have two decimal places and be between 0.01 and 0.99"""
    return min(max(round(probability_value, 2), 0.01), 0.99)


def format_correction(correction_value: float, positive_only=True) -> float:
    """Format a probability correction value to have two decimal places and be  greater than 0.01"""
    correction_value = round(correction_value, 2)
    if correction_value <= 0 and positive_only:
        correction_value = 0.01
    return correction_value
