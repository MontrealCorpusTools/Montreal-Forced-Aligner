from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Union

import os
import subprocess
import shutil
import soundfile

from ..dictionary import sanitize

from ..exceptions import SoxError

SoundFileInfoDict = Dict[str, Union[int, float, str]]


supported_audio_extensions = ['.flac', '.ogg', '.aiff', '.mp3']



def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf8') as f:
        text = f.read().strip().lower()
    return text


def parse_transcription(text: str, punctuation: Optional[str]=None, clitic_markers: Optional[str]=None) -> List[str]:
    words = [sanitize(x, punctuation, clitic_markers) for x in text.split()]
    words = [x for x in words if x not in ['', '-', "'"]]
    return words



def find_exts(files: List[str]) -> Tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    wav_files = {}
    other_audio_files = {}
    lab_files = {}
    textgrid_files = {}
    identifiers = []
    for full_filename in files:
        filename, fext = os.path.splitext(full_filename)
        fext = fext.lower()
        if fext == '.wav':
            wav_files[filename] = full_filename
        elif fext == '.lab':
            lab_files[filename] = full_filename
        elif fext == '.txt' and filename not in lab_files:  # .lab files have higher priority than .txt files
            lab_files[filename] = full_filename
        elif fext == '.textgrid':
            textgrid_files[filename] = full_filename
        elif fext in supported_audio_extensions and shutil.which('sox') is not None:
            other_audio_files[filename] = full_filename
        if filename not in identifiers:
            identifiers.append(filename)
    return identifiers, wav_files, lab_files, textgrid_files, other_audio_files


def get_wav_info(file_path: str, sample_rate: int=16000) -> dict:
    if file_path.endswith('.mp3'):
        if not shutil.which('soxi'):
            raise SoxError('No sox found')
        sox_proc = subprocess.Popen(['soxi', '-D', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        if stderr.startswith('soxi FAIL formats'):
            raise SoxError('No support for mp3 in sox')
        return_dict = {'duration': float(stdout.strip()), 'format': 'MP3'}
        sox_proc = subprocess.Popen(['soxi', '-r', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        return_dict['sample_rate'] = int(stdout.strip())
        sox_proc = subprocess.Popen(['soxi', '-c', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        return_dict['num_channels'] = int(stdout.strip())
        sox_proc = subprocess.Popen(['soxi', '-p', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = sox_proc.communicate()
        return_dict['bit_depth'] = int(stdout.strip())
        use_sox = True
    else:
        with soundfile.SoundFile(file_path, 'r') as inf:
            subtype = inf.subtype
            if subtype == 'FLOAT':
                bit_depth = 32
            else:
                bit_depth = int(subtype.split('_')[-1])
            frames = inf.frames
            sr = inf.samplerate
            duration = frames / sr
            return_dict = {'num_channels': inf.channels, 'type': inf.subtype, 'bit_depth': bit_depth,
                           'sample_rate': sr, 'duration': duration, 'format': inf.format}
        use_sox = False
        if bit_depth != 16:
            use_sox = True
        if return_dict['format'] != 'WAV':
            use_sox = True
        if not subtype.startswith('PCM'):
            use_sox = True
    return_dict['sox_string'] = ''
    if use_sox:
        return_dict['sox_string'] = 'sox {} -t wav -b 16 -r {} - |'.format(file_path, sample_rate)
    return return_dict