"""Helper functions for corpus parsing and loading"""
from __future__ import annotations

import os
import shutil
import subprocess
import typing

import soundfile

from montreal_forced_aligner.data import FileExtensions, SoundFileInformation
from montreal_forced_aligner.exceptions import SoxError

SoundFileInfoDict = typing.Dict[str, typing.Union[int, float, str]]

supported_audio_extensions = [".flac", ".ogg", ".aiff", ".mp3"]

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
    with open(path, "r", encoding="utf8") as f:
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
    exts = FileExtensions([], {}, {}, {}, {})
    for full_filename in files:
        filename, fext = os.path.splitext(full_filename)
        fext = fext.lower()
        if fext == ".wav":
            exts.wav_files[filename] = full_filename
        elif fext == ".lab":
            exts.lab_files[filename] = full_filename
        elif (
            fext == ".txt" and filename not in exts.lab_files
        ):  # .lab files have higher priority than .txt files
            exts.lab_files[filename] = full_filename
        elif fext == ".textgrid":
            exts.textgrid_files[filename] = full_filename
        elif fext in supported_audio_extensions and shutil.which("sox") is not None:
            exts.other_audio_files[filename] = full_filename
        if filename not in exts.identifiers:
            exts.identifiers.append(filename)
    return exts


def get_wav_info(file_path: str) -> SoundFileInformation:
    """
    Get sound file information

    Parameters
    ----------
    file_path: str
        Sound file path

    Returns
    -------
    :class:`~montreal_forced_aligner.data.SoundFileInformation`
        Sound information for format, duration, number of channels, bit depth, and
        sox_string for use in Kaldi feature extraction if necessary
    """
    if file_path.endswith(".mp3"):
        if not shutil.which("soxi"):
            raise SoxError("No sox found")
        sox_proc = subprocess.Popen(
            ["soxi", "-D", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        if stderr.startswith("soxi FAIL formats"):
            raise SoxError("No support for mp3 in sox")
        duration = float(stdout.strip())
        format = "MP3"
        sox_proc = subprocess.Popen(
            ["soxi", "-r", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        sample_rate = int(stdout.strip())
        sox_proc = subprocess.Popen(
            ["soxi", "-c", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        num_channels = int(stdout.strip())
        sox_proc = subprocess.Popen(
            ["soxi", "-p", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        bit_depth = int(stdout.strip())
        use_sox = True
    else:
        with soundfile.SoundFile(file_path) as inf:
            subtype = inf.subtype
            if subtype == "FLOAT":
                bit_depth = 32
            else:
                bit_depth = int(subtype.split("_")[-1])
            frames = inf.frames
            sample_rate = inf.samplerate
            duration = frames / sample_rate
            num_channels = inf.channels
            format = inf.format
        use_sox = False
        if bit_depth != 16:
            use_sox = True
        if format != "WAV":
            use_sox = True
        if not subtype.startswith("PCM"):
            use_sox = True
    sox_string = ""
    if use_sox:
        sox_string = f"sox {file_path} -t wav -b 16 - |"

    return SoundFileInformation(format, sample_rate, duration, num_channels, bit_depth, sox_string)
