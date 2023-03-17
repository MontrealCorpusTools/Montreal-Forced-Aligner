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
from pathlib import Path
from typing import Dict, List

from praatio import textgrid as tgio
from praatio.data_classes.interval_tier import Interval
from sqlalchemy.orm import Session, joinedload, selectinload

from montreal_forced_aligner.data import (
    CtmInterval,
    TextFileType,
    TextgridFormats,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    PhoneInterval,
    Speaker,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignmentExportError, TextGridParseError
from montreal_forced_aligner.helper import mfa_open

__all__ = [
    "process_ctm_line",
    "export_textgrid",
    "construct_output_tiers",
    "construct_output_path",
    "output_textgrid_writing_errors",
]


def process_ctm_line(
    line: str, reversed_phone_mapping: Dict[int, int], raw_id=False
) -> typing.Tuple[int, CtmInterval]:
    """
    Helper function for parsing a line of CTM file to construct a CTMInterval

    CTM format is:

    utt_id channel_num start_time phone_dur phone_id [confidence]

    Parameters
    ----------
    line: str
        Input string
    reversed_phone_mapping: dict[int, str]
        Mapping from integer IDs to phone labels

    Returns
    -------
    :class:`~montreal_forced_aligner.data.CtmInterval`
        Extracted data from the line
    """
    line = line.split()
    utt = line[0]
    if not raw_id:
        utt = int(line[0].split("-")[-1])
    begin = round(float(line[2]), 4)
    duration = float(line[3])
    end = round(begin + duration, 4)
    label = line[4]
    conf = None
    if len(line) > 5:
        conf = round(float(line[5]), 4)

    label = reversed_phone_mapping[int(label)]
    return utt, CtmInterval(begin, end, label, confidence=conf)


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
            with mfa_open(error_log, "w") as f:
                f.write(
                    "The following exceptions were encountered during the output of the alignments to TextGrids:\n\n"
                )
        with mfa_open(error_log, "a") as f:
            f.write(f"{str(result)}\n\n")


def parse_aligned_textgrid(
    path: Path, root_speaker: typing.Optional[str] = None
) -> Dict[str, List[CtmInterval]]:
    """
    Load a TextGrid as a dictionary of speaker's phone tiers

    Parameters
    ----------
    path: :class:`~pathlib.Path`
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
    num_tiers = len(tg.tiers)
    if num_tiers == 0:
        raise TextGridParseError(path, "Number of tiers parsed was zero")
    phone_tier_pattern = re.compile(r"(.*) ?- ?phones")
    for tier_name in tg.tierNames:
        ti = tg._tierDict[tier_name]
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
        for begin, end, text in ti.entries:
            text = text.lower().strip()
            if not text:
                continue
            begin, end = round(begin, 4), round(end, 4)
            if end - begin < 0.01:
                continue
            interval = CtmInterval(begin, end, text)
            data[speaker_name].append(interval)
    return data


def construct_output_tiers(
    session: Session,
    file_id: int,
    workflow: CorpusWorkflow,
    cleanup_textgrids: bool,
    clitic_marker: str,
    include_original_text: bool,
) -> Dict[str, Dict[str, List[CtmInterval]]]:
    """
    Construct aligned output tiers for a file

    Parameters
    ----------
    session: Session
        SqlAlchemy session
    file_id: int
        Integer ID for the file

    Returns
    -------
    Dict[str, Dict[str,List[CtmInterval]]]
        Aligned tiers
    """
    utterances = (
        session.query(Utterance)
        .options(
            joinedload(Utterance.speaker, innerjoin=True).load_only(Speaker.name),
        )
        .filter(Utterance.file_id == file_id)
    )
    data = {}
    for utt in utterances:
        word_intervals = (
            session.query(WordInterval, Word)
            .join(WordInterval.word)
            .filter(WordInterval.utterance_id == utt.id)
            .filter(WordInterval.workflow_id == workflow.id)
            .options(
                selectinload(WordInterval.phone_intervals).joinedload(
                    PhoneInterval.phone, innerjoin=True
                )
            )
            .order_by(WordInterval.begin)
        )
        if cleanup_textgrids:
            word_intervals = word_intervals.filter(Word.word_type != WordType.silence)
        if utt.speaker.name not in data:
            data[utt.speaker.name] = {"words": [], "phones": []}
            if include_original_text:
                data[utt.speaker.name]["utterances"] = []
        actual_words = utt.normalized_text.split()
        if include_original_text:
            data[utt.speaker.name]["utterances"].append(CtmInterval(utt.begin, utt.end, utt.text))
        for i, (wi, w) in enumerate(word_intervals.all()):
            if len(wi.phone_intervals) == 0:
                continue
            label = w.word
            if cleanup_textgrids:
                if (
                    w.word_type is WordType.oov
                    and workflow.workflow_type is WorkflowType.alignment
                ):
                    label = actual_words[i]
                if (
                    data[utt.speaker.name]["words"]
                    and clitic_marker
                    and (
                        data[utt.speaker.name]["words"][-1].label.endswith(clitic_marker)
                        or label.startswith(clitic_marker)
                    )
                ):
                    data[utt.speaker.name]["words"][-1].end = wi.end
                    data[utt.speaker.name]["words"][-1].label += label

                    for pi in sorted(wi.phone_intervals, key=lambda x: x.begin):
                        data[utt.speaker.name]["phones"].append(
                            CtmInterval(pi.begin, pi.end, pi.phone.phone)
                        )
                    continue

            data[utt.speaker.name]["words"].append(CtmInterval(wi.begin, wi.end, label))

            for pi in wi.phone_intervals:
                data[utt.speaker.name]["phones"].append(
                    CtmInterval(pi.begin, pi.end, pi.phone.phone)
                )
    return data


def construct_output_path(
    name: str,
    relative_path: Path,
    output_directory: Path,
    input_path: Path = None,
    output_format: str = TextgridFormats.SHORT_TEXTGRID,
) -> Path:
    """
    Construct an output path

    Returns
    -------
    Path
        Output path
    """
    if isinstance(output_directory, str):
        output_directory = Path(output_directory)
    if output_format.upper() == "LAB":
        extension = ".lab"
    elif output_format.upper() == "JSON":
        extension = ".json"
    elif output_format.upper() == "CSV":
        extension = ".csv"
    else:
        extension = ".TextGrid"
    if relative_path:
        relative = output_directory.joinpath(relative_path)
    else:
        relative = output_directory
    output_path = relative.joinpath(name + extension)
    if output_path == input_path:
        output_path = relative.joinpath(name + "_aligned" + extension)
    os.makedirs(relative, exist_ok=True)
    relative.mkdir(parents=True, exist_ok=True)
    return output_path


def export_textgrid(
    speaker_data: Dict[str, Dict[str, List[CtmInterval]]],
    output_path: Path,
    duration: float,
    frame_shift: float,
    output_format: str = TextFileType.TEXTGRID.value,
) -> None:
    """
    Export aligned file to TextGrid

    Parameters
    ----------
    speaker_data: dict[Speaker, dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]
        Per speaker, per word/phone :class:`~montreal_forced_aligner.data.CtmInterval`
    output_path: :class:`~pathlib.Path`
        Output path of the file
    duration: float
        Duration of the file
    frame_shift: float
        Frame shift of features, in seconds
    output_format: str, optional
        Output format, one of: "long_textgrid" (default), "short_textgrid", "json", or "csv"
    """
    has_data = False
    duration = round(duration, 6)
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
            with mfa_open(output_path, "w") as f:
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
            with mfa_open(output_path, "w") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
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
                if tier_name not in tg.tierNames:
                    tg.addTier(tgio.IntervalTier(tier_name, [], minT=0, maxT=duration))
                tier = tg.getTier(tier_name)
                for i, a in enumerate(sorted(intervals, key=lambda x: x.begin)):
                    if duration - a.end < (frame_shift * 2):  # Fix rounding issues
                        a.end = duration
                    if i > 0 and a.to_tg_interval().start > tier.entries[-1].end:
                        a.begin = tier.entries[-1].end
                    tier.insertEntry(a.to_tg_interval(duration))
        if has_data:
            for tier in tg.tiers:
                if len(tier.entries) > 0 and tier.entries[-1][1] > tg.maxTimestamp:
                    tier.insertEntry(
                        Interval(tier.entries[-1].start, tg.maxTimestamp, tier.entries[-1].label),
                        collisionMode="replace",
                    )
            tg.save(
                str(output_path),
                includeBlankSpaces=True,
                format=output_format,
                reportingMode="error",
            )
