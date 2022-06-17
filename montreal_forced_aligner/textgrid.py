"""
Textgrid utilities
==================

"""
from __future__ import annotations

import csv
import json
import os
import re
import typing
from typing import Dict, List

from praatio import textgrid as tgio
from praatio.data_classes.interval_tier import Interval

from montreal_forced_aligner.data import CtmInterval, TextFileType
from montreal_forced_aligner.exceptions import AlignmentExportError, TextGridParseError

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
    utt = int(line[0].split("-")[-1])
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


def output_textgrid_writing_errors(
    output_directory: str, export_errors: Dict[str, AlignmentExportError]
) -> None:
    """
    Output any errors that were encountered in writing TextGrids

    Parameters
    ----------
    output_directory: str
        Directory to save TextGrids files
    export_errors: dict[str, :class:`~montreal_forced_aligner.exceptions.AlignmentExportError`]
        Dictionary of errors encountered
    """
    error_log = os.path.join(output_directory, "output_errors.txt")
    if os.path.exists(error_log):
        os.remove(error_log)
    for result in export_errors.values():
        if not os.path.exists(error_log):
            with open(error_log, "w", encoding="utf8") as f:
                f.write(
                    "The following exceptions were encountered during the output of the alignments to TextGrids:\n\n"
                )
        with open(error_log, "a", encoding="utf8") as f:
            f.write(f"{str(result)}\n\n")


def parse_aligned_textgrid(
    path: str, root_speaker: typing.Optional[str] = None
) -> Dict[str, List[CtmInterval]]:
    """
    Load a TextGrid as a dictionary of speaker's phone tiers

    Parameters
    ----------
    path: str
        TextGrid file to parse
    root_speaker: str, optional
        Optional speaker if the TextGrid has no speaker information

    Returns
    -------
    dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]
        Parsed phone tier
    """
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
            speaker_name = m.groups()[0].strip()
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
            interval = CtmInterval(begin, end, text, 0)
            data[speaker_name].append(interval)
    return data


def export_textgrid(
    speaker_data: Dict[str, Dict[str, List[CtmInterval]]],
    output_path: str,
    duration: float,
    frame_shift: int,
    output_format: str = TextFileType.TEXTGRID.value,
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
    output_format: str, optional
        Output format, one of: "long_textgrid" (default), "short_textgrid", "json", or "csv"
    """
    if frame_shift > 1:
        frame_shift = round(frame_shift / 1000, 4)
    has_data = False
    if output_format == "csv":
        csv_data = []
        for speaker, data in speaker_data.items():
            for annotation_type, intervals in data.items():
                if len(intervals):
                    has_data = True
                for a in intervals:
                    if duration - a.end < (frame_shift * 2):  # Fix rounding issues
                        a.end = duration
                    csv_data.append(
                        {
                            "Begin": a.begin,
                            "End": a.end,
                            "Label": a.label,
                            "Type": annotation_type,
                            "Speaker": speaker,
                        }
                    )
        if has_data:
            with open(output_path, "w", encoding="utf8", newline=None) as f:
                writer = csv.DictWriter(f, fieldnames=["Begin", "End", "Label", "Type", "Speaker"])
                writer.writeheader()
                for line in csv_data:
                    writer.writerow(line)
    elif output_format == "json":
        json_data = {"start": 0, "end": duration, "tiers": {}}
        for speaker, data in speaker_data.items():
            for annotation_type, intervals in data.items():
                if len(speaker_data) > 1:
                    tier_name = f"{speaker} - {annotation_type}"
                else:
                    tier_name = annotation_type
                if tier_name not in json_data["tiers"]:
                    json_data["tiers"][tier_name] = {"type": "interval", "entries": []}
                if len(intervals):
                    has_data = True
                for a in intervals:
                    if duration - a.end < (frame_shift * 2):  # Fix rounding issues
                        a.end = duration
                    json_data["tiers"][tier_name]["entries"].append([a.begin, a.end, a.label])
        if has_data:
            with open(output_path, "w", encoding="utf8") as f:
                json.dump(json_data, f)
    else:
        # Create initial textgrid
        tg = tgio.Textgrid()
        tg.minTimestamp = 0
        tg.maxTimestamp = duration
        for speaker, data in speaker_data.items():
            for annotation_type, intervals in data.items():
                if len(intervals):
                    has_data = True
                if len(speaker_data) > 1:
                    tier_name = f"{speaker} - {annotation_type}"
                else:
                    tier_name = annotation_type
                if tier_name not in tg.tierNameList:
                    tg.addTier(tgio.IntervalTier(tier_name, [], minT=0, maxT=duration))
                for a in intervals:
                    if duration - a.end < (frame_shift * 2):  # Fix rounding issues
                        a.end = duration
                    tg.tierDict[tier_name].entryList.append(a.to_tg_interval())
        if has_data:
            for tier in tg.tierDict.values():
                if len(tier.entryList) > 0 and tier.entryList[-1][1] > tg.maxTimestamp:
                    tier.entryList[-1] = Interval(
                        tier.entryList[-1].start, tg.maxTimestamp, tier.entryList[-1].label
                    )
            tg.save(
                output_path, includeBlankSpaces=True, format=output_format, reportingMode="error"
            )
