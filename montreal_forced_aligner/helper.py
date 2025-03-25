"""
Helper functions
================

"""
from __future__ import annotations

import collections
import functools
import itertools
import json
import logging
import re
import typing
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import dataclassy
import numpy
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from Bio import pairwise2

if TYPE_CHECKING:
    from kalpy.fstext.lexicon import LexiconCompiler

    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.data import CtmInterval


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
    "compare_labels",
    "overlap_scoring",
    "make_re_character_set_safe",
    "align_phones",
    "split_phone_position",
    "align_pronunciations",
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


def output_mapping(mapping: Dict[str, Any], path: Path, skip_safe: bool = False) -> None:
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


def load_scp(path: Path, data_type: Optional[Type] = str) -> Dict[str, Any]:
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
    for g, h in itertools.product(gold, hypo):
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


def align_pronunciations(
    ref_text: typing.List[str],
    pronunciations: typing.List[str],
    oov_phone: str,
    silence_phone: str,
    silence_word: str,
    word_pronunciations: typing.Dict[str, typing.Set[str]],
):
    def score_function(ref: str, pron: typing.List[str]):
        if not word_pronunciations:
            return 0
        if ref in word_pronunciations and pron in word_pronunciations[ref]:
            return 0
        if pron == oov_phone:
            return 0
        return -2

    alignments = pairwise2.align.globalcs(
        ref_text,
        pronunciations,
        score_function,
        -1 if word_pronunciations else -5,
        -1 if word_pronunciations else -5,
        gap_char=["-"],
        one_alignment_only=True,
    )
    transformed_pronunciations = []
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-" and sb == silence_phone:
                sa = silence_word
            if "-" in (sa, sb):
                continue
            transformed_pronunciations.append((sa, sb.split()))
    return transformed_pronunciations


def load_evaluation_mapping(custom_mapping_path):
    with mfa_open(custom_mapping_path, "r") as f:
        mapping = yaml.load(f, Loader=yaml.Loader)
    for k, v in mapping.items():
        if isinstance(v, str):
            mapping[k] = {v}
        else:
            mapping[k] = set(v)
    return mapping


def fix_many_to_one_alignments(alignments, custom_mapping):
    test_keys = set(x for x in custom_mapping.keys() if " " in x)
    ref_keys = set()
    for val in custom_mapping.values():
        ref_keys.update(x for x in val if " " in x)
    new_ref = []
    new_test = []
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if i != 0:
                prev_sa = a.seqA[i - 1]
                prev_sb = a.seqB[i - 1]
                ref_key = " ".join(x.label for x in [prev_sa, sa] if x != "-")
                test_key = " ".join(x.label for x in [prev_sb, sb] if x != "-")
                if (
                    ref_key in ref_keys
                    and test_key in custom_mapping
                    and ref_key in custom_mapping[test_key]
                ):
                    new_ref[-1].label = ref_key
                    new_ref[-1].end = sa.end
                    if sb != "-":
                        new_test.append(sb)
                    continue
                if (
                    test_key in test_keys
                    and test_key in custom_mapping
                    and ref_key in custom_mapping[test_key]
                ):
                    new_test[-1].label = test_key
                    new_test[-1].end = sb.end
                    if sa != "-":
                        new_ref.append(sa)
                    continue
            if sa != "-":
                new_ref.append(sa)
            if sb != "-":
                new_test.append(sb)
        return new_ref, new_test


def align_phones(
    ref: List[CtmInterval],
    test: List[CtmInterval],
    silence_phone: str,
    ignored_phones: typing.Set[str] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    debug: bool = False,
) -> Tuple[float, float, Dict[Tuple[str, str], int]]:
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
    ignored_phones: set[str], optional
        Phones that should be ignored in score calculations (silence phone is automatically added)
    custom_mapping: dict[str, str], optional
        Mapping of phones to treat as matches even if they have different symbols
    debug: bool, optional
        Flag for logging extra information about alignments

    Returns
    -------
    float
        Score based on the average amount of overlap in phone intervals
    float
        Phone error rate
    dict[tuple[str, str], int]
        Dictionary of error pairs with their counts
    """

    if ignored_phones is None:
        ignored_phones = set()
    if not isinstance(ignored_phones, set):
        ignored_phones = set(ignored_phones)
    if custom_mapping is None:
        score_func = functools.partial(overlap_scoring, silence_phone=silence_phone)
    else:
        score_func = functools.partial(
            overlap_scoring, silence_phone=silence_phone, mapping=custom_mapping
        )

    alignments = pairwise2.align.globalcs(
        ref, test, score_func, -2, -2, gap_char=["-"], one_alignment_only=True
    )
    if custom_mapping is not None:
        ref, test = fix_many_to_one_alignments(alignments, custom_mapping)
        alignments = pairwise2.align.globalcs(
            ref, test, score_func, -2, -2, gap_char=["-"], one_alignment_only=True
        )
    overlap_count = 0
    overlap_sum = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0
    errors = collections.Counter()
    ignored_phones.add(silence_phone)
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-":
                if sb.label not in ignored_phones:
                    errors[(sa, sb.label)] += 1
                    num_insertions += 1
                else:
                    continue
            elif sb == "-":
                if sa.label not in ignored_phones:
                    errors[(sa.label, sb)] += 1
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
                    errors[(sa.label, sb.label)] += 1
    if overlap_count:
        score = overlap_sum / overlap_count
    else:
        score = None
    phone_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    if debug:
        import logging

        logger = logging.getLogger("mfa")
        logger.debug(
            f"{pairwise2.format_alignment(*alignments[0])}\nScore: {score}\nPER: {phone_error_rate}\nErrors: {errors}"
        )
    return score, phone_error_rate, errors


def fix_unk_words(
    ref: List[str],
    test: List[CtmInterval],
    lexicon_compiler: LexiconCompiler,
) -> Tuple[float, float, Dict[Tuple[str, str], int]]:
    """
    Align phones based on how much they overlap and their phone label, with the ability to specify a custom mapping for
    different phone labels to be scored as if they're the same phone

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    lexicon_compiler: LexiconCompiler
        Lexicon compiler to use for evaluating the identity of OOV items

    Returns
    -------
    float
        Extra duration of new words
    float
        Word error rate
    float
        Aligned duration of found words
    """

    from kalpy.gmm.data import WordCtmInterval

    def score_func(ref, test):
        ref_label = ref
        if isinstance(ref_label, WordCtmInterval):
            ref_label = ref_label.label
        test_label = test
        if isinstance(test_label, WordCtmInterval):
            test_label = test_label.label
        if ref_label == test_label:
            return 0
        if (
            test_label == lexicon_compiler.silence_word
            or ref_label == lexicon_compiler.silence_word
        ):
            return -10
        if lexicon_compiler.to_int(ref_label) == lexicon_compiler.to_int(test_label):
            return 0
        return -2

    alignments = pairwise2.align.globalcs(
        ref, test, score_func, -2, -2, gap_char=["-"], one_alignment_only=True
    )
    output_ctm = []
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-":
                output_ctm.append(sb)
            elif sb == "-":
                continue
            else:
                if sa != sb.label and sb.label == lexicon_compiler.oov_word:
                    sb.label = sa
                output_ctm.append(sb)
    return output_ctm


def align_words(
    ref: List[str],
    test: List[CtmInterval],
    silence_word: str,
    ignored_words: typing.Set[str] = None,
    debug: bool = False,
) -> Tuple[float, float, Dict[Tuple[str, str], int]]:
    """
    Align phones based on how much they overlap and their phone label, with the ability to specify a custom mapping for
    different phone labels to be scored as if they're the same phone

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    silence_word: str
        Silence word (these are ignored in the final calculation)
    ignored_words: set[str], optional
        Words that should be ignored in score calculations (silence phone is automatically added)
    debug: bool, optional
        Flag for logging extra information about alignments

    Returns
    -------
    float
        Extra duration of new words
    float
        Word error rate
    float
        Aligned duration of found words
    """

    from montreal_forced_aligner.data import CtmInterval

    if ignored_words is None:
        ignored_words = set()
    if not isinstance(ignored_words, set):
        ignored_words = set(ignored_words)

    def score_func(ref, test):
        ref_label = ref
        if isinstance(ref_label, CtmInterval):
            ref_label = ref_label.label
        test_label = test
        if isinstance(test_label, CtmInterval):
            test_label = test_label.label
        if ref_label == test_label:
            return 0
        if test_label == silence_word or ref_label == silence_word:
            return -10
        return -2

    alignments = pairwise2.align.globalcs(
        ref, test, score_func, -2, -2, gap_char=["-"], one_alignment_only=True
    )
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0

    ignored_words.add(silence_word)
    extra_duration = 0
    aligned_duration = 0
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-":
                if sb.label not in ignored_words:
                    num_insertions += 1
                    extra_duration += sb.end - sb.begin
                else:
                    continue
            elif sb == "-":
                if sa not in ignored_words:
                    num_deletions += 1
                else:
                    continue
            else:
                if sa in ignored_words:
                    continue
                if sa != sb.label:
                    num_substitutions += 1
                else:
                    aligned_duration += sb.end - sb.begin
    word_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    if debug:
        import logging

        logger = logging.getLogger("mfa")
        logger.debug(
            f"{pairwise2.format_alignment(*alignments[0])}\nExtra word duration: {extra_duration}\nWER: {word_error_rate}"
        )
    return extra_duration, word_error_rate, aligned_duration


def format_probability(probability_value: float) -> float:
    """Format a probability to have two decimal places and be between 0.01 and 0.99"""
    return min(max(round(probability_value, 2), 0.01), 0.99)


def format_correction(correction_value: float, positive_only=True) -> float:
    """Format a probability correction value to have two decimal places and be  greater than 0.01"""
    correction_value = round(correction_value, 2)
    if correction_value <= 0 and positive_only:
        correction_value = 0.01
    return correction_value
