"""
Textgrid utilities
==================

"""
from __future__ import annotations

import os
import re
import typing
from typing import Dict, List

from praatio import textgrid as tgio

from montreal_forced_aligner.data import CtmInterval
from montreal_forced_aligner.exceptions import TextGridParseError

__all__ = [
    "process_ctm_line",
    "export_textgrid",
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


def output_textgrid_writing_errors(output_directory: str, export_errors: Dict[str, str]) -> None:
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


def parse_aligned_textgrid(
    path: str, root_speaker: typing.Optional[str] = None
) -> Dict[str, List[CtmInterval]]:
    tg = tgio.openTextgrid(path, includeEmptyIntervals=False, reportingMode="silence")
    data = {}
    num_tiers = len(tg.tierNameList)
    if num_tiers == 0:
        raise TextGridParseError(path, "Number of tiers parsed was zero")
    phone_tier_pattern = re.compile(r"(.*) ?- ?phones")
    for tier_name in tg.tierNameList:
        ti = tg.tierDict[tier_name]
        if not isinstance(ti, tgio.IntervalTier):
            continue
        if "phones" not in tier_name:
            continue
        m = phone_tier_pattern.match(tier_name)
        if m:
            speaker_name = m.groups()[0]
        elif root_speaker:
            speaker_name = root_speaker
        else:
            speaker_name = ""
        if speaker_name not in data:
            data[speaker_name] = []
        for begin, end, text in ti.entryList:
            text = text.lower().strip()
            if not text:
                continue
            begin, end = round(begin, 4), round(end, 4)
            if end - begin < 0.01:
                continue
            interval = CtmInterval(begin, end, text, speaker_name)
            data[speaker_name].append(interval)
    return data


def export_textgrid(
    speaker_data: Dict[str, Dict[str, List[CtmInterval]]],
    output_path: str,
    duration: float,
    frame_shift: int,
) -> None:
    """
    Export aligned file to TextGrid

    Parameters
    ----------
    speaker_data: dict[Speaker, dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]
        Per speaker, per word/phone :class:`~montreal_forced_aligner.data.CtmInterval`
    output_path: str
        Output path of the file
    duration: float
        Duration of the file
    frame_shift: int
        Frame shift of features, in ms
    """
    if frame_shift > 1:
        frame_shift = round(frame_shift / 1000, 4)
    # Create initial textgrid
    tg = tgio.Textgrid()
    tg.minTimestamp = 0
    tg.maxTimestamp = duration

    if len(speaker_data) > 1:
        for speaker in speaker_data:
            word_tier_name = f"{speaker} - words"
            phone_tier_name = f"{speaker} - phones"

            word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=duration)
            phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=duration)
            tg.addTier(word_tier)
            tg.addTier(phone_tier)
    else:
        word_tier_name = "words"
        phone_tier_name = "phones"
        word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=duration)
        phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=duration)
        tg.addTier(word_tier)
        tg.addTier(phone_tier)

    for speaker, data in speaker_data.items():
        words = data["words"]
        phones = data["phones"]
        tg_words = []
        tg_phones = []
        for w in words:
            if duration - w.end < frame_shift:  # Fix rounding issues
                w.end = duration
            tg_words.append(w.to_tg_interval())
        for p in phones:
            if duration - p.end < frame_shift:  # Fix rounding issues
                p.end = duration
            tg_phones.append(p.to_tg_interval())

        if len(speaker_data) > 1:
            word_tier_name = f"{speaker} - words"
            phone_tier_name = f"{speaker} - phones"
        else:
            word_tier_name = "words"
            phone_tier_name = "phones"
        word_tier = tgio.IntervalTier(word_tier_name, tg_words, minT=0, maxT=duration)
        phone_tier = tgio.IntervalTier(phone_tier_name, tg_phones, minT=0, maxT=duration)
        tg.replaceTier(word_tier_name, word_tier)
        tg.replaceTier(phone_tier_name, phone_tier)

    tg.save(output_path, includeBlankSpaces=True, format="long_textgrid", reportingMode="error")
