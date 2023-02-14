"""Class definitions for Speakers, Files, Utterances and Jobs"""
from __future__ import annotations

import os
import sys
import traceback
import typing
from typing import TYPE_CHECKING, Optional, Union

from praatio import textgrid

from montreal_forced_aligner.corpus.helper import get_wav_info, load_text
from montreal_forced_aligner.data import SoundFileInformation, TextFileType
from montreal_forced_aligner.exceptions import TextGridParseError, TextParseError

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
        enforce_sample_rate: Optional[int] = None,
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
        sanitize_function: Callable, optional
            Function to sanitize words and strip punctuation

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.classes.FileData`
            Parsed file
        """
        text_type: TextFileType = TextFileType.NONE
        if text_path is not None:
            if text_path.lower().endswith(".textgrid"):
                text_type = TextFileType.TEXTGRID
            else:
                text_type = TextFileType.LAB
        file = FileData(
            file_name, wav_path, text_path, relative_path=relative_path, text_type=text_type
        )
        if wav_path is not None:
            root = os.path.dirname(wav_path)
            file.wav_info = get_wav_info(
                wav_path,
                enforce_mono=file.text_type is TextFileType.LAB,
                enforce_sample_rate=enforce_sample_rate,
            )
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
        root_speaker = None
        if speaker_characters or file.text_type != TextFileType.TEXTGRID:
            root_speaker = speaker_name
        file.load_text(
            root_speaker=root_speaker,
        )
        return file

    def load_text(
        self,
        root_speaker: Optional[str] = None,
    ) -> None:
        """
        Load the transcription text from the text_file of the object

        Parameters
        ----------
        root_speaker: str, optional
            Speaker derived from the root directory, ignored for TextGrids
        """
        if self.text_type == TextFileType.LAB:
            try:
                text = load_text(self.text_path)
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
            try:
                tg = textgrid.openTextgrid(self.text_path, includeEmptyIntervals=False)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                raise TextGridParseError(
                    self.text_path,
                    "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
                )

            num_tiers = len(tg.tierNames)
            if num_tiers == 0:
                raise TextGridParseError(self.text_path, "Number of tiers parsed was zero")
            num_channels = 1
            if self.wav_info is not None:
                duration = self.wav_info.duration
                num_channels = self.wav_info.num_channels
            else:
                duration = tg.maxTimestamp
            for i, tier_name in enumerate(tg.tierNames):
                ti = tg._tierDict[tier_name]
                if tier_name.lower() == "notes":
                    continue
                if not isinstance(ti, textgrid.IntervalTier):
                    continue
                if not root_speaker:
                    speaker_name = tier_name.strip()
                    self.speaker_ordering.append(speaker_name)
                else:
                    speaker_name = root_speaker
                channel = 0
                if num_channels == 2 and i >= num_tiers / 2:
                    channel = 1
                for begin, end, text in ti.entries:
                    text = text.strip()
                    if not text:
                        continue
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
    text: str = ""
    normalized_text: str = ""
    normalized_character_text: str = ""
    oovs: str = ""
