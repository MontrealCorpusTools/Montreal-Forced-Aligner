"""Helper functions for corpus parsing and loading"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Optional, Union

import soundfile

from montreal_forced_aligner.dictionary.mixins import SanitizeFunction
from montreal_forced_aligner.exceptions import SoxError

SoundFileInfoDict = dict[str, Union[int, float, str]]

supported_audio_extensions = [".flac", ".ogg", ".aiff", ".mp3"]

__all__ = ["load_text", "parse_transcription", "find_exts", "get_wav_info"]


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
        text = f.read().strip().lower()
    return text


def parse_transcription(text: str, sanitize_function=Optional[SanitizeFunction]) -> list[str]:
    """
    Parse an orthographic transcription given punctuation and clitic markers

    Parameters
    ----------
    text: str
        Orthographic text to parse
    sanitize_function: :class:`~montreal_forced_aligner.dictionary.mixins.SanitizeFunction`, optional
        Function to sanitize words and strip punctuation

    Returns
    -------
    List
        Parsed orthographic transcript
    """
    if sanitize_function is not None:
        words = [
            sanitize_function(w)
            for w in text.split()
            if w not in sanitize_function.clitic_markers + sanitize_function.compound_markers
        ]
    else:
        words = text.split()
    return words


def find_exts(
    files: list[str],
) -> tuple[list[str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """
    Find and group sound file extensions and transcription file extensions

    Parameters
    ----------
    files: List
        List of filename

    Returns
    -------
    list[str]
        File name identifiers
    dict[str, str]
        Wav files
    dict[str, str]
        Lab and text files
    dict[str, str]
        TextGrid files
    dict[str, str]
        Other audio files (flac, mp3, etc)
    """
    wav_files = {}
    other_audio_files = {}
    lab_files = {}
    textgrid_files = {}
    identifiers = []
    for full_filename in files:
        filename, fext = os.path.splitext(full_filename)
        fext = fext.lower()
        if fext == ".wav":
            wav_files[filename] = full_filename
        elif fext == ".lab":
            lab_files[filename] = full_filename
        elif (
            fext == ".txt" and filename not in lab_files
        ):  # .lab files have higher priority than .txt files
            lab_files[filename] = full_filename
        elif fext == ".textgrid":
            textgrid_files[filename] = full_filename
        elif fext in supported_audio_extensions and shutil.which("sox") is not None:
            other_audio_files[filename] = full_filename
        if filename not in identifiers:
            identifiers.append(filename)
    return identifiers, wav_files, lab_files, textgrid_files, other_audio_files


def get_wav_info(file_path: str) -> dict[str, Any]:
    """
    Get sound file information

    Parameters
    ----------
    file_path: str
        Sound file path

    Returns
    -------
    dict[str, Any]
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
        return_dict = {"duration": float(stdout.strip()), "format": "MP3"}
        sox_proc = subprocess.Popen(
            ["soxi", "-r", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        return_dict["sample_rate"] = int(stdout.strip())
        sox_proc = subprocess.Popen(
            ["soxi", "-c", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        return_dict["num_channels"] = int(stdout.strip())
        sox_proc = subprocess.Popen(
            ["soxi", "-p", file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        stdout, stderr = sox_proc.communicate()
        return_dict["bit_depth"] = int(stdout.strip())
        use_sox = True
    else:
        with soundfile.SoundFile(file_path, "r") as inf:
            subtype = inf.subtype
            if subtype == "FLOAT":
                bit_depth = 32
            else:
                bit_depth = int(subtype.split("_")[-1])
            frames = inf.frames
            sr = inf.samplerate
            duration = frames / sr
            return_dict = {
                "num_channels": inf.channels,
                "type": inf.subtype,
                "bit_depth": bit_depth,
                "sample_rate": sr,
                "duration": duration,
                "format": inf.format,
            }
        use_sox = False
        if bit_depth != 16:
            use_sox = True
        if return_dict["format"] != "WAV":
            use_sox = True
        if not subtype.startswith("PCM"):
            use_sox = True
    return_dict["sox_string"] = ""
    if use_sox:
        return_dict["sox_string"] = f"sox {file_path} -t wav -b 16 - |"
    return return_dict
