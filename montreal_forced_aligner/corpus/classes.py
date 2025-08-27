"""Class definitions for Speakers, Files, Utterances and Jobs"""
from __future__ import annotations

import os
import typing
import unicodedata
from typing import TYPE_CHECKING, Optional, Union

from kalpy.gmm.data import CtmInterval
from praatio import textgrid

from montreal_forced_aligner.corpus.helper import get_wav_info, load_text
from montreal_forced_aligner.data import SoundFileInformation, TextFileType
from montreal_forced_aligner.exceptions import TextParseError
from montreal_forced_aligner.textgrid import load_textgrid

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclassy import dataclass

__all__ = ["FileData", "UtteranceData"]


@dataclass(slots=True)
class FileData:
    """
    Data class for file information

    Parameters
    ----------
    name: str
        File name
    wav_path: str, optional
        Path to sound file
    text_path: str, optional
        Path to sound file
    relative_path: str
        Path relative to corpus root directory
    wav_info: dict[str, Any]
        Information dictionary about the sound file
    speaker_ordering: list[str]
        List of speakers in the file
    utterances: list[:class:`~montreal_forced_aligner.corpus.classes.UtteranceData`]
        Utterance data for the file
    """

    name: str
    wav_path: typing.Optional[str]
    text_path: typing.Optional[str]
    text_type: TextFileType
    relative_path: str
    wav_info: SoundFileInformation = None
    speaker_ordering: typing.List[str] = []
    utterances: typing.List[UtteranceData] = []

    @classmethod
    def parse_file(
        cls,
        file_name: str,
        wav_path: Optional[str],
        text_path: Optional[str],
        relative_path: str,
        speaker_characters: Union[int, str],
    ):
        """
        Parse a collection of sound file and transcription file into a File

        Parameters
        ----------
        file_name: str
            File identifier
        wav_path: str
            Full sound file path
        text_path: str
            Full transcription path
        relative_path: str
            Relative path from the corpus directory root
        speaker_characters: int, optional
            Number of characters in the file name to specify the speaker

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.classes.FileData`
            Parsed file
        """
        text_type: TextFileType = TextFileType.NONE
        if text_path is not None:
            if str(text_path).lower().endswith(".textgrid"):
                text_type = TextFileType.TEXTGRID
            else:
                text_type = TextFileType.LAB
        file = FileData(
            file_name, wav_path, text_path, relative_path=relative_path, text_type=text_type
        )
        if wav_path is not None:
            root = os.path.dirname(wav_path)
            file.wav_info = get_wav_info(wav_path)
        else:
            root = os.path.dirname(text_path)
        if not speaker_characters:
            speaker_name = os.path.basename(root)
        elif isinstance(speaker_characters, int):
            speaker_name = file_name[:speaker_characters]
        elif speaker_characters == "prosodylab":
            speaker_name = file_name.split("_")[1]
        else:
            speaker_name = file_name
        file.load_text(speaker_name, ensure_root_speaker=bool(speaker_characters))
        return file

    def load_text(self, root_speaker: str, ensure_root_speaker: bool = False) -> None:
        """
        Load the transcription text from the text_file of the object

        Parameters
        ----------
        root_speaker: str
            Speaker derived from the root directory, ignored for TextGrid tiers that have a speaker label
        ensure_root_speaker: bool, defaults to False
            Flag for ensuring that root_speaker is used as the speaker name
        """
        if self.text_type == TextFileType.LAB:
            try:
                text = load_text(self.text_path)
                text = unicodedata.normalize("NFKC", text)
            except UnicodeDecodeError:
                raise TextParseError(self.text_path)
            if self.wav_info is None:
                end = -1
            else:
                end = self.wav_info.duration
            utterance = UtteranceData(
                speaker_name=root_speaker,
                file_name=self.name,
                text=text,
                begin=0,
                channel=0,
                end=end,
            )
            self.utterances.append(utterance)
            self.speaker_ordering.append(root_speaker)
        elif self.text_type == TextFileType.TEXTGRID:
            tg = load_textgrid(self.text_path)
            num_tiers = len(tg.tiers)
            num_channels = 1
            if self.wav_info is not None:
                duration = self.wav_info.duration
                num_channels = self.wav_info.num_channels
            else:
                duration = tg.maxTimestamp
            if root_speaker:
                self.speaker_ordering.append(root_speaker)
            phone_data = {}
            word_data = {}
            for i, tier_name in enumerate(tg.tierNames):
                if tier_name.endswith("phones"):
                    if " - " in tier_name:
                        speaker_name = tier_name.split(" - ")[0]
                    else:
                        speaker_name = root_speaker
                    if speaker_name not in phone_data:
                        phone_data[speaker_name] = []
                    ti = tg._tierDict[tier_name]
                    for begin, end, text in ti.entries:
                        text = text.strip()
                        if not text:
                            continue
                        begin, end = round(begin, 4), round(end, 4)
                        if begin >= duration:
                            continue
                        end = min(end, duration)
                        phone_data[speaker_name].append(CtmInterval(begin, end, text))
                elif tier_name.endswith("words"):
                    if " - " in tier_name:
                        speaker_name = tier_name.split(" - ")[0]
                    else:
                        speaker_name = root_speaker
                    if speaker_name not in word_data:
                        word_data[speaker_name] = []
                    ti = tg._tierDict[tier_name]
                    for begin, end, text in ti.entries:
                        text = text.strip()
                        if not text:
                            continue
                        begin, end = round(begin, 4), round(end, 4)
                        if begin >= duration:
                            continue
                        end = min(end, duration)
                        word_data[speaker_name].append(CtmInterval(begin, end, text))
            for i, tier_name in enumerate(tg.tierNames):
                if tier_name.lower() == "notes":
                    continue
                if tier_name.endswith("words") or tier_name.endswith("phones"):
                    if root_speaker not in self.speaker_ordering:
                        self.speaker_ordering.append(root_speaker)
                    continue
                ti = tg._tierDict[tier_name]
                if not isinstance(ti, textgrid.IntervalTier):
                    continue
                if not ensure_root_speaker:
                    speaker_name = tier_name.strip()
                else:
                    speaker_name = root_speaker
                self.speaker_ordering.append(speaker_name)
                channel = 0
                if num_channels == 2 and i >= num_tiers / 2:
                    channel = 1
                for begin, end, text in ti.entries:
                    text = text.strip()
                    if not text:
                        continue
                    text = unicodedata.normalize("NFKC", text)
                    begin, end = round(begin, 4), round(end, 4)
                    if begin >= duration:
                        continue
                    end = min(end, duration)
                    utt = UtteranceData(
                        speaker_name=speaker_name,
                        file_name=self.name,
                        begin=begin,
                        end=end,
                        text=text,
                        channel=channel,
                    )
                    if not utt.text:
                        continue
                    if speaker_name in phone_data:
                        for pi in phone_data[speaker_name]:
                            if pi.begin < utt.begin:
                                continue
                            if pi.end > utt.end:
                                break
                            if (
                                len(utt.phone_intervals) > 0
                                and pi.begin != utt.phone_intervals[-1].end
                            ):
                                utt.phone_intervals.append(
                                    CtmInterval(utt.phone_intervals[-1].end, pi.begin, "sil")
                                )
                            utt.phone_intervals.append(pi)
                        utt.manual_alignments = len(utt.phone_intervals) > 0
                        if utt.manual_alignments:
                            if utt.begin != utt.phone_intervals[0].begin:
                                if (
                                    abs(utt.phone_intervals[0].begin - utt.begin) < 0.02
                                    or utt.phone_intervals[0].label == "sil"
                                ):
                                    utt.phone_intervals[0].begin = utt.begin
                                else:
                                    utt.phone_intervals.insert(
                                        0,
                                        CtmInterval(
                                            utt.begin, utt.phone_intervals[0].begin, "sil"
                                        ),
                                    )
                            if utt.phone_intervals[-1].end != utt.end:
                                if (
                                    abs(utt.phone_intervals[-1].end - utt.end) < 0.02
                                    or utt.phone_intervals[-1].label == "sil"
                                ):
                                    utt.phone_intervals[-1].end = utt.end
                                else:
                                    utt.phone_intervals.append(
                                        CtmInterval(utt.phone_intervals[-1].end, utt.end, "sil")
                                    )
                    if speaker_name in word_data:
                        for wi in word_data[speaker_name]:
                            if wi.begin < utt.begin:
                                continue
                            if wi.end > utt.end:
                                break
                            if (
                                len(utt.word_intervals) > 0
                                and wi.begin != utt.word_intervals[-1].end
                            ):
                                utt.word_intervals.append(
                                    CtmInterval(utt.word_intervals[-1].end, wi.begin, "<eps>")
                                )
                            utt.word_intervals.append(wi)
                        if utt.manual_alignments and len(utt.word_intervals) > 0:
                            if utt.begin != utt.word_intervals[0].begin:
                                utt.word_intervals.insert(
                                    0, CtmInterval(utt.begin, utt.word_intervals[0].begin, "<eps>")
                                )
                            if utt.word_intervals[-1].end != utt.end:
                                utt.word_intervals.append(
                                    CtmInterval(utt.word_intervals[-1].end, utt.end, "<eps>")
                                )
                    self.utterances.append(utt)
        else:
            if self.wav_info is not None:
                duration = self.wav_info.duration
            else:
                duration = 1
            utt = UtteranceData(
                speaker_name=root_speaker,
                file_name=self.name,
                begin=0,
                channel=0,
                end=duration,
            )
            self.utterances.append(utt)
            self.speaker_ordering.append(root_speaker)


@dataclass(slots=True)
class UtteranceData:
    """
    Data class for utterance information

    Parameters
    ----------
    speaker_name: str
        Speaker name
    file_name: str
        File name
    begin: float, optional
        Begin timestamp
    end: float, optional
        End timestamp
    channel: int, optional
        Sound file channel
    text: str, optional
        Utterance text
    oovs: set[str]
        Set of words not found in a look up
    """

    speaker_name: str
    file_name: str
    begin: float
    end: float
    channel: int = 0
    manual_alignments: bool = False
    text: str = ""
    normalized_text: str = ""
    normalized_character_text: str = ""
    oovs: str = ""
    phone_intervals: typing.List = []
    word_intervals: typing.List = []
