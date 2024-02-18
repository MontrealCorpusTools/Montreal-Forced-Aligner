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
from praatio.utilities import utils as tgio_utils
from sqlalchemy.orm import Session

from montreal_forced_aligner.data import (
    CtmInterval,
    PhoneType,
    TextFileType,
    TextgridFormats,
    WordType,
)
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    Phone,
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


class Textgrid(tgio.Textgrid):
    def save(
        self,
        fn: str,
        format: typing.Literal["short_textgrid", "long_textgrid", "json", "textgrid_json"],
        includeBlankSpaces: bool,
        minTimestamp: typing.Optional[float] = None,
        maxTimestamp: typing.Optional[float] = None,
        minimumIntervalLength: float = None,
        reportingMode: typing.Literal["silence", "warning", "error"] = "warning",
    ) -> None:
        """Save the current textgrid to a file

        Args:
            fn: the fullpath filename of the output
            format: one of ['short_textgrid', 'long_textgrid', 'json', 'textgrid_json']
                'short_textgrid' and 'long_textgrid' are both used by praat
                'json' and 'textgrid_json' are two json variants. 'json' cannot represent
                tiers with different min and max timestamps than the textgrid.
            includeBlankSpaces: if True, blank sections in interval
                tiers will be filled in with an empty interval
                (with a label of ""). If you are unsure, True is recommended
                as Praat needs blanks to render textgrids properly.
            minTimestamp: the minTimestamp of the saved Textgrid;
                if None, use whatever is defined in the Textgrid object.
                If minTimestamp is larger than timestamps in your textgrid,
                an exception will be thrown.
            maxTimestamp: the maxTimestamp of the saved Textgrid;
                if None, use whatever is defined in the Textgrid object.
                If maxTimestamp is smaller than timestamps in your textgrid,
                an exception will be thrown.
            minimumIntervalLength: any labeled intervals smaller
                than this will be removed, useful for removing ultrashort
                or fragmented intervals; if None, don't remove any.
                Removed intervals are merged (without their label) into
                adjacent entries.
            reportingMode: one of "silence", "warning", or "error". This flag
                determines the behavior if there is a size difference between the
                maxTimestamp in the tier and the current textgrid.

        Returns:
            a string representation of the textgrid
        """

        with mfa_open(fn, mode="w") as fd:
            if format == TextgridFormats.LONG_TEXTGRID:
                fd.write('File type = "ooTextFile"\n')
                fd.write('Object class = "TextGrid"\n\n')

                tab = " " * 4

                # Header
                fd.write(f"xmin = {self.minTimestamp} \n")
                fd.write(f"xmax = {self.maxTimestamp} \n")
                fd.write("tiers? <exists> \n")
                fd.write(f"size = {len(self._tierDict)} \n")
                fd.write("item []: \n")

                for tierNum, (name, tier) in enumerate(self._tierDict.items()):
                    # Interval header
                    tier_name = tgio_utils.escapeQuotes(name)
                    fd.write(tab + f"item [{tierNum + 1}]:\n")
                    fd.write(tab * 2 + f'class = "{tier.tierType}" \n')
                    fd.write(tab * 2 + f'name = "{tier_name}" \n')
                    fd.write(tab * 2 + f"xmin = {self.minTimestamp} \n")
                    fd.write(tab * 2 + f"xmax = {self.maxTimestamp} \n")

                    fd.write(tab * 2 + f"intervals: size = {len(tier._entries)} \n")
                    interval_index = 1
                    if includeBlankSpaces and tier._entries:
                        if tier._entries[0][0] > 0.001:
                            fd.write(
                                f"{tab * 2}intervals [{interval_index}]:\n"
                                f"{tab * 3}xmin = 0.0 \n"
                                f"{tab * 3}xmax = {tier._entries[0][0]} \n"
                                f'{tab * 3}text = "" \n'
                            )
                            interval_index += 1

                    for i, entry in enumerate(tier._entries):
                        start, end, label = entry
                        if (
                            includeBlankSpaces
                            and i > 0
                            and start - tier._entries[i - 1][1] > 0.001
                        ):
                            fd.write(
                                f"{tab * 2}intervals [{interval_index}]:\n"
                                f"{tab * 3}xmin = {tier._entries[i-1][1]} \n"
                                f"{tab * 3}xmax = {start} \n"
                                f'{tab * 3}text = "" \n'
                            )
                            interval_index += 1
                        fd.write(
                            f"{tab * 2}intervals [{interval_index}]:\n"
                            f"{tab * 3}xmin = {start} \n"
                            f"{tab * 3}xmax = {end} \n"
                            f'{tab * 3}text = "{tgio_utils.escapeQuotes(label)}" \n'
                        )
                        interval_index += 1
                    if includeBlankSpaces and tier._entries:
                        if self.maxTimestamp - tier._entries[-1][1] > 0.001:
                            fd.write(
                                f"{tab * 2}intervals [{interval_index}]:\n"
                                f"{tab * 3}xmin = {tier._entries[-1][1]} \n"
                                f"{tab * 3}xmax = {self.maxTimestamp} \n"
                                f'{tab * 3}text = "" \n'
                            )
                            interval_index += 1
            elif format == TextgridFormats.SHORT_TEXTGRID:
                # Header
                fd.write('File type = "ooTextFile"\n')
                fd.write('Object class = "TextGrid"\n\n')
                fd.write(f"{self.minTimestamp}\n{self.maxTimestamp}\n")
                fd.write(f"<exists>\n{len(self._tierDict)}\n")
                for name, tier in self._tierDict.items():
                    tier_name = tgio_utils.escapeQuotes(name)
                    c = tier.tierType
                    fd.write(f'"{c}"\n')
                    fd.write(f'"{tier_name}"\n')
                    fd.write(f"{self.minTimestamp}\n{self.maxTimestamp}\n{len(tier._entries)}\n")

                    if includeBlankSpaces and tier._entries:
                        if tier._entries[0][0] > 0.001:
                            fd.write(f'0.0\n{tier._entries[0][0]}\n""\n')
                    for entry in tier._entries:
                        start, end, label = entry
                        label = tgio_utils.escapeQuotes(label)

                        fd.write(f'{start}\n{end}\n"{label}"\n')
                    if includeBlankSpaces and tier._entries:
                        if self.maxTimestamp - tier._entries[-1][1] > 0.001:
                            fd.write(f'{tier._entries[-1][1]}\n{self.maxTimestamp}\n""\n')


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
    data = {}
    phone_intervals = (
        session.query(PhoneInterval.begin, PhoneInterval.end, Phone.phone, Speaker.name)
        .join(PhoneInterval.phone)
        .join(PhoneInterval.utterance)
        .join(Utterance.speaker)
        .filter(Utterance.file_id == file_id)
        .filter(PhoneInterval.workflow_id == workflow.id)
        .filter(PhoneInterval.duration > 0)
        .order_by(PhoneInterval.begin)
    )
    word_intervals = (
        session.query(WordInterval.begin, WordInterval.end, Word.word, Speaker.name)
        .join(WordInterval.word)
        .join(WordInterval.utterance)
        .join(Utterance.speaker)
        .filter(Utterance.file_id == file_id)
        .filter(WordInterval.workflow_id == workflow.id)
        .filter(WordInterval.duration > 0)
        .order_by(WordInterval.begin)
    )
    if cleanup_textgrids:
        phone_intervals = phone_intervals.filter(Phone.phone_type != PhoneType.silence)
        word_intervals = word_intervals.filter(Word.word_type != WordType.silence)

    for w_begin, w_end, w, speaker_name in word_intervals:
        if speaker_name not in data:
            data[speaker_name] = {"words": [], "phones": []}
            if include_original_text:
                data[speaker_name]["utterances"] = []
        if (
            data[speaker_name]["words"]
            and w_begin - data[speaker_name]["words"][-1].end < 0.02
            and clitic_marker
            and (
                data[speaker_name]["words"][-1].label.endswith(clitic_marker)
                or w.startswith(clitic_marker)
            )
        ):
            data[speaker_name]["words"][-1].end = w_end
            data[speaker_name]["words"][-1].label += w

        else:
            data[speaker_name]["words"].append(CtmInterval(w_begin, w_end, w))

    if include_original_text:
        utterances = (
            session.query(Utterance.begin, Utterance.end, Utterance.text, Speaker.name)
            .join(Utterance.speaker)
            .filter(Utterance.file_id == file_id)
        )
        for utt_begin, utt_end, utt_text, speaker_name in utterances:
            data[speaker_name]["utterances"].append(CtmInterval(utt_begin, utt_end, utt_text))

    for p_begin, p_end, phone, speaker_name in phone_intervals:
        data[speaker_name]["phones"].append(CtmInterval(p_begin, p_end, phone))
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
        tg = Textgrid()
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
                    if i == len(intervals) - 1 and duration - a.end < (
                        frame_shift * 2
                    ):  # Fix rounding issues
                        a.end = duration
                    tg_interval = a.to_tg_interval()
                    if i > 0 and tier._entries[-1].end > tg_interval.start:
                        a.begin = tier._entries[-1].end
                        tg_interval = a.to_tg_interval()
                    tier._entries.append(tg_interval)
        if has_data:
            for tier in tg.tiers:
                if len(tier._entries) > 0 and tier._entries[-1][1] > tg.maxTimestamp:
                    tier.insertEntry(
                        Interval(
                            tier._entries[-1].start, tg.maxTimestamp, tier._entries[-1].label
                        ),
                        collisionMode="replace",
                    )
            tg.save(
                str(output_path),
                includeBlankSpaces=True,
                format=output_format,
                minimumIntervalLength=None,
                reportingMode="silence",
            )
