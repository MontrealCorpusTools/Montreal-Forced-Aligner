"""Helper functions for corpus parsing and loading"""
from __future__ import annotations

import typing
from pathlib import Path

import soundfile

from montreal_forced_aligner.data import FileExtensions, SoundFileInformation
from montreal_forced_aligner.helper import mfa_open

SoundFileInfoDict = typing.Dict[str, typing.Union[int, float, str]]

supported_audio_extensions = {
    ".flac",
    ".ogg",
    ".aiff",
    ".mp3",
    ".opus",
    "flac",
    "ogg",
    "aiff",
    "mp3",
    "opus",
}

__all__ = ["load_text", "find_exts", "get_wav_info"]


def load_text(path: str) -> str:
    """
    Load a text file

    Parameters
    ----------
    path: str
        Text file to load

    Returns
    -------
    str
        Orthographic text of the file
    """
    with mfa_open(path, "r") as f:
        text = f.read().strip()
    return text


def find_exts(files: typing.List[str]) -> FileExtensions:
    """
    Find and group sound file extensions and transcription file extensions

    Parameters
    ----------
    files: List
        List of filename

    Returns
    -------
    :class:`~montreal_forced_aligner.data.FileExtensions`
        Data class for files found
    """
    exts = FileExtensions(set(), {}, {}, {}, {})
    for full_filename in files:
        if full_filename.startswith("."):  # Ignore hidden files
            continue
        try:
            filename, fext = full_filename.rsplit(".", maxsplit=1)
        except ValueError:
            continue
        fext = fext.lower()
        if fext == "wav":
            exts.wav_files[filename] = full_filename
        elif fext == "lab":
            exts.lab_files[filename] = full_filename
        elif (
            fext == "txt" and filename not in exts.lab_files
        ):  # .lab files have higher priority than .txt files
            exts.lab_files[filename] = full_filename
        elif fext == "textgrid":
            exts.textgrid_files[filename] = full_filename
        elif fext in supported_audio_extensions:
            exts.other_audio_files[filename] = full_filename
        exts.identifiers.add(filename)
    return exts


def get_wav_info(
    file_path: str, enforce_mono: bool = False, enforce_sample_rate: typing.Optional[int] = None
) -> SoundFileInformation:
    """
    Get sound file information

    Parameters
    ----------
    file_path: str
        Sound file path
    enforce_mono: bool
        Flag for whether to ensure that stereo files have the first channel extracted when processing
        them in Kaldi
    enforce_sample_rate: int, optional
        Sampling rate to enforce when sending data to Kaldi

    Returns
    -------
    :class:`~montreal_forced_aligner.data.SoundFileInformation`
        Sound information for format, sampling rate, duration, and number of channels
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    format = file_path.suffix.lower()
    with soundfile.SoundFile(file_path) as inf:
        frames = inf.frames
        sample_rate = inf.samplerate
        duration = frames / sample_rate
        num_channels = inf.channels

    return SoundFileInformation(format, sample_rate, duration, num_channels)
