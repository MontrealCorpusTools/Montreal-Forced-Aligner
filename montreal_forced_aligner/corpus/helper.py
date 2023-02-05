"""Helper functions for corpus parsing and loading"""
from __future__ import annotations

import datetime
import subprocess
import typing

import soundfile

from montreal_forced_aligner.data import FileExtensions, SoundFileInformation
from montreal_forced_aligner.exceptions import SoundFileError
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
        Sound information for format, duration, number of channels, bit depth, and
        sox_string for use in Kaldi feature extraction if necessary
    """
    _, format = file_path.rsplit(".", maxsplit=1)
    format = format.lower()
    num_channels = 0
    sample_rate = 0
    duration = 0
    sox_string = ""
    if format in {"mp3", "opus"}:
        if format == "mp3":
            sox_proc = subprocess.Popen(
                ["soxi", f"{file_path}"], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
            stdout, stderr = sox_proc.communicate()
            if stderr:
                raise SoundFileError(file_path, stderr)
            for line in stdout.splitlines():
                if line.startswith("Channels"):
                    num_channels = int(line.split(":")[-1].strip())
                elif line.startswith("Sample Rate"):
                    sample_rate = int(line.split(":")[-1].strip())
                elif line.startswith("Duration"):
                    duration_string = line.split(":", maxsplit=1)[-1].split("=")[0].strip()
                    duration = (
                        datetime.datetime.strptime(duration_string, "%H:%M:%S.%f")
                        - datetime.datetime(1900, 1, 1)
                    ).total_seconds()
                    break
            sample_rate_string = ""
            if enforce_sample_rate is not None:
                sample_rate_string = f" -r {enforce_sample_rate}"
            sox_string = f'sox "{file_path}" -t wav -b 16{sample_rate_string} - |'
        else:  # Fall back use ffmpeg if sox doesn't support the format
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-hide_banner",
                    "-show_entries",
                    "stream=duration,channels,sample_rate",
                    "-of",
                    "default=noprint_wrappers=1",
                    "-i",
                    file_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = ffmpeg_proc.communicate()
            if stderr:
                raise SoundFileError(file_path, stderr)
            for line in stdout.splitlines():
                try:
                    key, value = line.strip().split("=")
                    if key == "duration":
                        duration = float(value)
                    elif key == "sample_rate":
                        sample_rate = int(value)
                    else:
                        num_channels = int(value)
                except ValueError:
                    pass
            mono_string = ""
            sample_rate_string = ""
            if num_channels > 1 and enforce_mono:
                mono_string = ' -af "pan=mono|FC=FL"'
            if enforce_sample_rate is not None:
                sample_rate_string = f" -ar {enforce_sample_rate}"
            sox_string = f'ffmpeg -nostdin -hide_banner -loglevel error -nostats -i "{file_path}" -acodec pcm_s16le -f wav{mono_string}{sample_rate_string} - |'
    else:
        use_sox = False
        with soundfile.SoundFile(file_path) as inf:
            frames = inf.frames
            sample_rate = inf.samplerate
            duration = frames / sample_rate
            num_channels = inf.channels
            try:
                bit_depth = int(inf.subtype.split("_")[-1])
                if bit_depth != 16:
                    use_sox = True
            except Exception:
                use_sox = True
        sample_rate_string = ""
        if enforce_sample_rate is not None:
            sample_rate_string = f" -r {enforce_sample_rate}"
        if format != "wav":
            use_sox = True
        if num_channels > 1 and enforce_mono:
            use_sox = True
        elif enforce_sample_rate is not None and sample_rate != enforce_sample_rate:
            use_sox = True
        if num_channels > 1 and enforce_mono:
            sox_string = f'sox "{file_path}" -t wav -b 16{sample_rate_string} - remix 1  |'
        elif use_sox:
            sox_string = f'sox "{file_path}" -t wav -b 16{sample_rate_string} - |'

    return SoundFileInformation(format, sample_rate, duration, num_channels, sox_string)
