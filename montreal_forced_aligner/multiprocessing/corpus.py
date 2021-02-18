import multiprocessing as mp
from queue import Empty
import traceback
import sys
import os
from textgrid import TextGrid, IntervalTier

from ..helper import parse_logs, thirdparty_binary, make_path_safe, load_text

from ..dictionary import sanitize

from ..exceptions import SampleRateError, CorpusError, WavReadError, SampleRateMismatchError, \
    BitDepthError, TextParseError, TextGridParseError

from ..corpus.base import extract_temp_channels, get_wav_info


def parse_transcription(text):
    words = [sanitize(x) for x in text.split()]
    words = [x for x in words if x not in ['', '-', "'"]]
    return words


def parse_wav_file(utt_name, wav_path, lab_path, relative_path, speaker_characters, temp_directory=None):
    root = os.path.dirname(wav_path)
    try:
        wav_info = get_wav_info(wav_path)
        sr = wav_info['sample_rate']
    except Exception:
        raise WavReadError(wav_path)
    if sr < 16000:
        raise SampleRateError(wav_path)
    bit_depth = wav_info['bit_depth']
    if bit_depth != 16:
        raise BitDepthError(wav_path)
    if not speaker_characters:
        speaker_name = os.path.basename(root)
    elif isinstance(speaker_characters, int):
        speaker_name = utt_name[:speaker_characters]
    elif speaker_characters == 'prosodylab':
        speaker_name = utt_name.split('_')[1]
    else:
        speaker_name = utt_name
    speaker_name = speaker_name.strip().replace(' ', '_')
    utt_name = utt_name.strip().replace(' ', '_')
    return {'utt_name': utt_name, 'speaker_name': speaker_name, 'wav_path': wav_path,
            'wav_info': wav_info, 'relative_path': relative_path}


def parse_lab_file(utt_name, wav_path, lab_path, relative_path, speaker_characters, temp_directory=None):
    root = os.path.dirname(wav_path)
    try:
        wav_info = get_wav_info(wav_path)
        sr = wav_info['sample_rate']
    except Exception:
        raise WavReadError(wav_path)
    if sr < 16000:
        raise SampleRateError(wav_path)
    bit_depth = wav_info['bit_depth']
    if bit_depth != 16:
        raise BitDepthError(wav_path)
    try:
        text = load_text(lab_path)
    except UnicodeDecodeError:
        raise TextParseError(lab_path)
    words = parse_transcription(text)
    if not words:
        raise TextParseError(lab_path)
    if not speaker_characters:
        speaker_name = os.path.basename(root)
    elif isinstance(speaker_characters, int):
        speaker_name = utt_name[:speaker_characters]
    elif speaker_characters == 'prosodylab':
        speaker_name = utt_name.split('_')[1]
    else:
        speaker_name = utt_name
    speaker_name = speaker_name.strip().replace(' ', '_')
    utt_name = utt_name.strip().replace(' ', '_')
    if not utt_name.startswith(speaker_name):
        utt_name = speaker_name + '_' + utt_name  # Fix for some Kaldi issues in needing sorting by speaker

    return {'utt_name': utt_name, 'speaker_name': speaker_name, 'text_file': lab_path, 'wav_path':wav_path,
            'words': ' '.join(words), 'wav_info': wav_info, 'relative_path': relative_path}

def parse_textgrid_file(recording_name, wav_path, textgrid_path, relative_path, speaker_characters, temp_directory):
    file_name = recording_name
    try:
        wav_info = get_wav_info(wav_path)
        sr = wav_info['sample_rate']
    except Exception:
        raise WavReadError(wav_path)
    if sr < 16000:
        raise SampleRateError(wav_path)
    bit_depth = wav_info['bit_depth']
    wav_max_time = wav_info['duration']
    if bit_depth != 16:
        raise BitDepthError(wav_path)
    tg = TextGrid()
    try:
        tg.read(textgrid_path)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        raise TextGridParseError(textgrid_path, '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    n_channels = wav_info['num_channels']
    num_tiers = len(tg.tiers)
    if n_channels == 2:
        a_name = file_name + "_channel1"
        b_name = file_name + "_channel2"

        a_path, b_path = extract_temp_channels(wav_path, temp_directory)
    elif n_channels > 2:
        raise (Exception('More than two channels'))
    speaker_ordering = []
    if speaker_characters:
        if isinstance(speaker_characters, int):
            speaker_name = file_name[:speaker_characters]
        elif speaker_characters == 'prosodylab':
            speaker_name = file_name.split('_')[1]
        else:
            speaker_name = file_name
        speaker_name = speaker_name.strip().replace(' ', '_')
        speaker_ordering.append(speaker_name)
    segments = {}
    utt_wav_mapping = {}
    text_mapping = {}
    utt_text_file_mapping = {}
    utt_speak_mapping = {}
    speak_utt_mapping = {}
    utt_file_mapping = {}
    file_utt_mapping = {file_name: []}
    for i, ti in enumerate(tg.tiers):
        if ti.name.lower() == 'notes':
            continue
        if not isinstance(ti, IntervalTier):
            continue
        if not speaker_characters:
            speaker_name = ti.name.strip().replace(' ', '_')
            speaker_ordering.append(speaker_name)
        for interval in ti:
            text = interval.mark.lower().strip()
            words = parse_transcription(text)
            if not words:
                continue
            begin, end = round(interval.minTime, 4), round(interval.maxTime, 4)
            if end > wav_max_time:
                end = wav_max_time
            utt_name = '{}_{}_{}_{}'.format(speaker_name, file_name, begin, end)
            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
            if n_channels == 1:
                segments[utt_name] = '{} {} {}'.format(file_name, begin, end)
                utt_wav_mapping[file_name] = wav_path
            else:
                if i < num_tiers / 2:
                    utt_name += '_channel1'
                    segments[utt_name] = '{} {} {}'.format(a_name, begin, end)
                    utt_wav_mapping[a_name] = a_path
                else:
                    utt_name += '_channel2'
                    segments[utt_name] = '{} {} {}'.format(b_name, begin, end)
                    utt_wav_mapping[b_name] = b_path
            text_mapping[utt_name] = ' '.join(words)
            utt_text_file_mapping[utt_name] = textgrid_path
            utt_speak_mapping[utt_name] = speaker_name
            utt_file_mapping[utt_name] = file_name
            file_utt_mapping[file_name].append(utt_name)
            if speaker_name not in speak_utt_mapping:
                speak_utt_mapping[speaker_name] = []
            speak_utt_mapping[speaker_name].append(utt_name)
    file_names = [file_name]
    if n_channels == 2:
        file_names.append(a_name)
        file_names.append(b_name)
    return {'text_file': textgrid_path, 'wav_path':wav_path, 'wav_info': wav_info, 'segments': segments,
            'utt_wav_mapping': utt_wav_mapping, 'text_mapping': text_mapping,
            'utt_text_file_mapping': utt_text_file_mapping, 'utt_speak_mapping': utt_speak_mapping,
            'speak_utt_mapping': speak_utt_mapping, 'speaker_ordering': speaker_ordering,
            'file_names': file_names, 'relative_path': relative_path, 'recording_name': recording_name,
            'file_utt_mapping': file_utt_mapping, 'utt_file_mapping': utt_file_mapping
            }


class CorpusProcessWorker(mp.Process):
    def __init__(self, job_q, return_dict, return_q, stopped, initializing=True):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.initializing = initializing

    def run(self):
        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty as error:
                if self.initializing and not self.stopped.stop_check():
                    continue
                else:
                    break
            self.initializing = False
            self.job_q.task_done()
            wav_path = arguments[1]
            transcription_path = arguments[2]

            try:
                if transcription_path is None:
                    info = parse_wav_file(*arguments)
                elif transcription_path.lower().endswith('.textgrid'):
                    info = parse_textgrid_file(*arguments)
                else:
                    info = parse_lab_file(*arguments)
                self.return_q.put(info)
            except WavReadError:
                if 'wav_read_errors' not in self.return_dict:
                    self.return_dict['wav_read_errors'] = []
                self.return_dict['wav_read_errors'].append(wav_path)
            except SampleRateError:
                if 'unsupported_sample_rate' not in self.return_dict:
                    self.return_dict['unsupported_sample_rate'] = []
                self.return_dict['unsupported_sample_rate'].append(wav_path)
            except BitDepthError:
                if 'unsupported_bit_depths' not in self.return_dict:
                    self.return_dict['unsupported_bit_depths'] = []
                self.return_dict['unsupported_bit_depths'].append(wav_path)
            except TextParseError:
                if 'decode_error_files' not in self.return_dict:
                    self.return_dict['decode_error_files'] = []
                self.return_dict['decode_error_files'].append(transcription_path)
            except TextGridParseError as e:
                if 'textgrid_read_errors' not in self.return_dict:
                    self.return_dict['textgrid_read_errors'] = {}
                self.return_dict['textgrid_read_errors'][transcription_path] = e.error
            except Exception as e:
                self.stopped.stop()
                self.return_dict['error'] = arguments, Exception(traceback.format_exception(*sys.exc_info()))
        return