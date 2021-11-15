"""Helper functions for corpus parsing and loading"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..config.dictionary_config import DictionaryConfig

import soundfile

from ..exceptions import SoxError

SoundFileInfoDict = Dict[str, Union[int, float, str]]

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


def parse_transcription(
    text: str, dictionary_config: Optional[DictionaryConfig] = None
) -> List[str]:
    """
    Parse an orthographic transcription given punctuation and clitic markers

    Parameters
    ----------
    text: str
        Orthographic text to parse
    dictionary_config: Optional[DictionaryConfig]
        Characters to treat as punctuation

    Returns
    -------
    List
        Parsed orthographic transcript
    """
    if dictionary_config is not None:
        words = [dictionary_config.sanitize(x) for x in text.split()]
        words = [
            x
            for x in words
            if x
            and x not in dictionary_config.clitic_markers
            and x not in dictionary_config.compound_markers
        ]
    else:
        words = text.split()
    return words


def find_exts(
    files: List[str],
) -> Tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Find and group sound file extensions and transcription file extensions

    Parameters
    ----------
    files: List
        List of filename

    Returns
    -------
    List[str]
        File name identifiers
    Dict[str, str]
        Wav files
    Dict[str, str]
        Lab and text files
    Dict[str, str]
        TextGrid files
    Dict[str, str]
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


def get_wav_info(file_path: str, sample_rate: int = 16000) -> dict:
    """
    Get sound file information

    Parameters
    ----------
    file_path: str
        Sound file path
    sample_rate: int
        Default sample rate

    Returns
    -------
    Dict
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
        return_dict["sox_string"] = f"sox {file_path} -t wav -b 16 -r {sample_rate} - |"
    return return_dict
