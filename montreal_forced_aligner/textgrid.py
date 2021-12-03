"""
Textgrid utilities
==================

"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from praatio import textgrid as tgio

from .data import CtmInterval

if TYPE_CHECKING:
    from .abc import ReversedMappingType
    from .alignment.base import CorpusAligner
    from .corpus.classes import File, Speaker
    from .dictionary import DictionaryData

__all__ = [
    "process_ctm_line",
    "parse_from_word",
    "parse_from_phone",
    "parse_from_word_no_cleanup",
    "generate_tiers",
    "export_textgrid",
    "ctm_to_textgrid",
    "output_textgrid_writing_errors",
]


def process_ctm_line(line: str) -> CtmInterval:
    """
    Helper function for parsing a line of CTM file to construct a CTMInterval

    Parameters
    ----------
    line: str
        Input string

    Returns
    -------
    :class:`~montreal_forced_aligner.data.CtmInterval`
        Extracted data from the line
    """
    line = line.split(" ")
    utt = line[0]
    if len(line) == 5:
        begin = round(float(line[2]), 4)
        duration = float(line[3])
        end = round(begin + duration, 4)
        label = line[4]
    else:
        begin = round(float(line[1]), 4)
        duration = float(line[2])
        end = round(begin + duration, 4)
        label = line[3]
    return CtmInterval(begin, end, label, utt)


def parse_from_word(
    ctm_labels: list[CtmInterval], text: list[str], dictionary_data: DictionaryData
) -> list[CtmInterval]:
    """
    Parse CTM intervals into the corresponding text for an utterance

    Parameters
    ----------
    ctm_labels: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        CTM intervals
    text: list[str]
        The original text that was to be aligned
    dictionary_data: :class:`~montreal_forced_aligner.dictionary.DictionaryData`
        Dictionary data necessary for splitting subwords

    Returns
    -------
    list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        Correct intervals with subwords merged back into their original text
    """
    cur_ind = 0
    actual_labels = []
    utterance = None
    for word in text:
        ints = dictionary_data.to_int(word)
        b = 1000000
        e = -1
        for i in ints:
            cur = ctm_labels[cur_ind]
            if utterance is None:
                utterance = cur.utterance
            if i == int(cur.label):
                if cur.begin < b:
                    b = cur.begin
                if cur.end > e:
                    e = cur.end
            cur_ind += 1
        lab = CtmInterval(b, e, word, utterance)
        actual_labels.append(lab)
    return actual_labels


def parse_from_word_no_cleanup(
    ctm_labels: list[CtmInterval], reversed_word_mapping: ReversedMappingType
) -> list[CtmInterval]:
    """
    Assume that subwords in the CTM files are desired, so just does a reverse look up to get the sub word
    text

    Parameters
    ----------
    ctm_labels: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of :class:`~montreal_forced_aligner.data.CtmInterval` to convert
    reversed_word_mapping: dict[int, str]
        Look up for Kaldi word IDs to convert them back to text

    Returns
    -------
    list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        Parsed intervals with text rather than integer IDs
    """
    for ctm_interval in ctm_labels:
        label = reversed_word_mapping[int(ctm_interval.label)]
        ctm_interval.label = label
    return ctm_labels


def parse_from_phone(
    ctm_labels: list[CtmInterval],
    reversed_phone_mapping: ReversedMappingType,
    positions: list[str],
) -> list[CtmInterval]:
    """
    Parse CtmIntervals to original phone transcriptions

    Parameters
    ----------
    ctm_labels: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of :class:`~montreal_forced_aligner.data.CtmInterval` to convert
    reversed_phone_mapping: dict[int, str]
        Mapping to convert phone IDs to phone labels
    positions: list[str]
        List of word positions to account for

    Returns
    -------
    list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        Parsed intervals with phone labels rather than IDs
    """
    for ctm_interval in ctm_labels:
        label = reversed_phone_mapping[int(ctm_interval.label)]
        for p in positions:
            if label.endswith(p):
                label = label[: -1 * len(p)]
        ctm_interval.label = label
    return ctm_labels


def output_textgrid_writing_errors(output_directory: str, export_errors: dict[str, str]) -> None:
    """
    Output any errors that were encountered in writing TextGrids

    Parameters
    ----------
    output_directory: str
        Directory to save TextGrids files
    export_errors: dict[str, str]
        Dictionary of errors encountered
    """
    error_log = os.path.join(output_directory, "output_errors.txt")
    if os.path.exists(error_log):
        os.remove(error_log)
    for file_name, result in export_errors.items():
        if not os.path.exists(error_log):
            with open(error_log, "w", encoding="utf8") as f:
                f.write(
                    "The following exceptions were encountered during the output of the alignments to TextGrids:\n\n"
                )
        with open(error_log, "a", encoding="utf8") as f:
            f.write(f"{file_name}:\n")
            f.write(f"{result}\n\n")


def generate_tiers(
    file: File, cleanup_textgrids: Optional[bool] = True
) -> dict[Speaker, dict[str, list[CtmInterval]]]:
    """
    Generate TextGrid tiers for a given File

    Parameters
    ----------
    file: File
        File to generate TextGrid tiers
    cleanup_textgrids: bool, optional
        Flag for whether the TextGrids should be cleaned up

    Returns
    -------
    dict[Speaker, dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]
        Tier information per speaker, with :class:`~montreal_forced_aligner.data.CtmInterval` split by "phones" and "words"
    """
    output = {}

    for u in file.utterances.values():
        if not u.word_labels:
            continue
        speaker = u.speaker
        dictionary_data = speaker.dictionary_data

        words = []
        phones = []
        if dictionary_data.multilingual_ipa and cleanup_textgrids:
            phone_ind = 0
            for interval in u.word_labels:
                end = interval.end
                word = interval.label
                subwords = dictionary_data.lookup(
                    word,
                )
                subwords = [
                    x if x in dictionary_data.words_mapping else dictionary_data.oov_word
                    for x in subwords
                ]
                subprons = [dictionary_data.words[x] for x in subwords]
                cur_phones = []
                while u.phone_labels[phone_ind].end <= end:
                    p = u.phone_labels[phone_ind]
                    if p.label in dictionary_data.silence_phones:
                        phone_ind += 1
                        continue
                    cur_phones.append(p)
                    phone_ind += 1
                    if phone_ind > len(u.phone_labels) - 1:
                        break
                phones.extend(dictionary_data.map_to_original_pronunciation(cur_phones, subprons))
                if not word:
                    continue

                words.append(interval)
        else:
            for interval in u.word_labels:
                words.append(interval)
            for interval in u.phone_labels:
                if interval.label in dictionary_data.silence_phones and cleanup_textgrids:
                    continue
                phones.append(interval)
        if speaker not in output:
            output[speaker] = {"words": words, "phones": phones}
        else:
            output[speaker]["words"].extend(words)
            output[speaker]["phones"].extend(phones)
    return output


def export_textgrid(
    file: File,
    output_path: str,
    speaker_data: dict[Speaker, dict[str, list[CtmInterval]]],
    frame_shift: int,
    first_file_write: Optional[bool] = True,
) -> None:
    """
    Export aligned file to TextGrid

    Parameters
    ----------
    file: File
        File object to export
    output_path: str
        Output path of the file
    speaker_data: dict[Speaker, dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]
        Per speaker, per word/phone :class:`~montreal_forced_aligner.data.CtmInterval`
    frame_shift: int
        Frame shift of features, in ms
    first_file_write: bool, optional
        Flag for whether the file should be created from scratch or appended to if it
        has been modified by another export process
    """
    if frame_shift > 1:
        frame_shift = round(frame_shift / 1000, 4)
    if first_file_write:
        # Create initial textgrid
        tg = tgio.Textgrid()
        tg.minTimestamp = 0
        tg.maxTimestamp = file.duration

        if len(file.speaker_ordering) > 1:
            for speaker in file.speaker_ordering:
                word_tier_name = f"{speaker} - words"
                phone_tier_name = f"{speaker} - phones"

                word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=file.duration)
                phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=file.duration)
                tg.addTier(word_tier)
                tg.addTier(phone_tier)
        else:
            word_tier_name = "words"
            phone_tier_name = "phones"
            word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=file.duration)
            phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=file.duration)
            tg.addTier(word_tier)
            tg.addTier(phone_tier)
    else:
        # Use existing
        tg = tgio.openTextgrid(output_path, includeEmptyIntervals=False)

        word_tier_name = "words"
        phone_tier_name = "phones"
        word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=file.duration)
        phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=file.duration)
        tg.addTier(word_tier)
        tg.addTier(phone_tier)
    for speaker, data in speaker_data.items():
        words = data["words"]
        phones = data["phones"]
        tg_words = []
        tg_phones = []
        for w in words:
            if file.duration - w.end < frame_shift:  # Fix rounding issues
                w.end = file.duration
            tg_words.append(w.to_tg_interval())
        for p in phones:
            if file.duration - p.end < frame_shift:  # Fix rounding issues
                p.end = file.duration
            tg_phones.append(p.to_tg_interval())

        if len(file.speaker_ordering) > 1:
            word_tier_name = f"{speaker} - words"
            phone_tier_name = f"{speaker} - phones"
        else:
            word_tier_name = "words"
            phone_tier_name = "phones"
        word_tier = tgio.IntervalTier(word_tier_name, tg_words, minT=0, maxT=file.duration)
        phone_tier = tgio.IntervalTier(phone_tier_name, tg_phones, minT=0, maxT=file.duration)
        tg.replaceTier(word_tier_name, word_tier)
        tg.replaceTier(phone_tier_name, phone_tier)

    tg.save(output_path, includeBlankSpaces=True, format="long_textgrid", reportingMode="error")


def ctm_to_textgrid(file: File, aligner: CorpusAligner, first_file_write=True) -> None:
    """
    Export a File to TextGrid

    Parameters
    ----------
    file: File
        File to export
    aligner: CorpusAligner
        Aligner used to generate the alignments
    first_file_write: bool, optional
        Flag for whether this is the first time touching this file
    """
    data = generate_tiers(file, cleanup_textgrids=aligner.cleanup_textgrids)

    backup_output_directory = None
    if not aligner.overwrite:
        backup_output_directory = aligner.backup_output_directory
        os.makedirs(backup_output_directory, exist_ok=True)
    output_path = file.construct_output_path(aligner.textgrid_output, backup_output_directory)
    export_textgrid(file, output_path, data, aligner.frame_shift, first_file_write)
