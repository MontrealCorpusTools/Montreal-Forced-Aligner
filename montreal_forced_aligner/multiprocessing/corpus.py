import multiprocessing as mp
from queue import Empty
import traceback
import sys
import os
import time
from praatio import textgrid

from ..helper import load_text

from ..dictionary import sanitize

from ..exceptions import SampleRateError, WavReadError, \
    TextParseError, TextGridParseError

from ..corpus.base import get_wav_info


def parse_transcription(text, punctuation=None, clitic_markers=None):
    words = [sanitize(x, punctuation, clitic_markers) for x in text.split()]
    words = [x for x in words if x not in ['', '-', "'"]]
    return words


def parse_wav_file(utt_name, wav_path, lab_path, relative_path, speaker_characters, sample_rate=16000):
    root = os.path.dirname(wav_path)
    wav_info = get_wav_info(wav_path, sample_rate=sample_rate)
    if not speaker_characters:
        speaker_name = os.path.basename(root)
    elif isinstance(speaker_characters, int):
        speaker_name = utt_name[:speaker_characters]
    elif speaker_characters == 'prosodylab':
        speaker_name = utt_name.split('_')[1]
    else:
        speaker_name = utt_name
    speaker_name = speaker_name.strip().replace(' ', '_')

    new_utt_name = utt_name.strip().replace(' ', '_space_')
    return_dict = {'utt_name': new_utt_name, 'speaker_name': speaker_name, 'wav_path': wav_path,
            'wav_info': wav_info, 'relative_path': relative_path, 'file_name': utt_name}
    if 'sox_string' in wav_info:
        return_dict['sox_string'] = wav_info['sox_string']
    return return_dict


def parse_lab_file(utt_name, wav_path, lab_path, relative_path, speaker_characters, sample_rate=16000, punctuation=None,
                   clitic_markers=None):
    root = os.path.dirname(wav_path)
    wav_info = get_wav_info(wav_path, sample_rate=sample_rate)
    try:
        text = load_text(lab_path)
    except UnicodeDecodeError:
        raise TextParseError(lab_path)
    words = parse_transcription(text, punctuation, clitic_markers)
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

    new_utt_name = utt_name.strip().replace(' ', '_space_')

    if not new_utt_name.startswith(speaker_name):
        new_utt_name = speaker_name + '_' + new_utt_name  # Fix for some Kaldi issues in needing sorting by speaker

    new_utt_name = new_utt_name.replace('_', '-')
    return_dict = {'utt_name': new_utt_name, 'speaker_name': speaker_name, 'text_file': lab_path, 'wav_path': wav_path,
                   'words': ' '.join(words), 'wav_info': wav_info, 'relative_path': relative_path, 'file_name': utt_name}
    if 'sox_string' in wav_info:
        return_dict['sox_string'] = wav_info['sox_string']
    return return_dict


def parse_textgrid_file(recording_name, wav_path, textgrid_path, relative_path, speaker_characters, sample_rate=16000,
                        punctuation=None, clitic_markers=None, stop_check=None):
    file_name = recording_name
    wav_info = get_wav_info(wav_path, sample_rate=sample_rate)
    wav_max_time = wav_info['duration']
    try:
        tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        raise TextGridParseError(textgrid_path,
                                 '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    if stop_check is not None and stop_check.stop_check():
        return
    n_channels = wav_info['num_channels']
    num_tiers = len(tg.tierNameList)
    if num_tiers == 0:
        raise TextGridParseError(textgrid_path, 'Number of tiers parsed was zero')
    if n_channels > 2:
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
    sox_string_mappings = {}
    file_utt_mapping = {file_name: []}
    for i, tier_name in enumerate(tg.tierNameList):
        ti = tg.tierDict[tier_name]
        if tier_name.lower() == 'notes':
            continue
        if not isinstance(ti, textgrid.IntervalTier):
            continue
        if not speaker_characters:
            speaker_name = tier_name.strip().replace(' ', '_')
            speaker_ordering.append(speaker_name)
        for begin, end, text in ti.entryList:
            if stop_check is not None and stop_check.stop_check():
                return
            text = text.lower().strip()
            words = parse_transcription(text, punctuation, clitic_markers)
            if not words:
                continue
            begin, end = round(begin, 4), round(end, 4)
            end = min(end, wav_max_time)
            utt_name = '{}_{}_{}_{}'.format(file_name, speaker_name, begin, end)
            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
            utt_name = utt_name.replace('_', '-')
            utt_wav_mapping[file_name] = wav_path
            if n_channels == 1:
                segments[utt_name] = {'file_name': file_name, 'begin': begin, 'end': end, 'channel': 0}
            else:
                if i < num_tiers / 2:
                    segments[utt_name] = {'file_name': file_name, 'begin': begin, 'end': end, 'channel': 0}
                else:
                    segments[utt_name] = {'file_name': file_name, 'begin': begin, 'end': end, 'channel': 1}
            text_mapping[utt_name] = ' '.join(words)
            utt_text_file_mapping[file_name] = textgrid_path
            utt_speak_mapping[utt_name] = speaker_name
            utt_file_mapping[utt_name] = file_name
            file_utt_mapping[file_name].append(utt_name)
            if speaker_name not in speak_utt_mapping:
                speak_utt_mapping[speaker_name] = []
            speak_utt_mapping[speaker_name].append(utt_name)
    file_names = [file_name]
    return_dict = {'text_file': textgrid_path, 'wav_path': wav_path, 'wav_info': wav_info, 'segments': segments,
                   'utt_wav_mapping': utt_wav_mapping, 'text_mapping': text_mapping,
                   'utt_text_file_mapping': utt_text_file_mapping, 'utt_speak_mapping': utt_speak_mapping,
                   'speak_utt_mapping': speak_utt_mapping, 'speaker_ordering': speaker_ordering,
                   'file_names': file_names, 'relative_path': relative_path, 'recording_name': recording_name,
                   'file_utt_mapping': file_utt_mapping, 'utt_file_mapping': utt_file_mapping
                   }
    if 'sox_string' in wav_info:
        sox_string_mappings[file_name] = wav_info['sox_string']
        return_dict['sox_string'] = sox_string_mappings
    return return_dict


class CorpusProcessWorker(mp.Process):
    def __init__(self, job_q, return_dict, return_q, stopped, finished_adding):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding

    def run(self):
        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty as error:
                if self.finished_adding.stop_check():
                    break
                continue
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            wav_path = arguments[1]
            transcription_path = arguments[2]

            try:
                if transcription_path is None:
                    info = parse_wav_file(*arguments)
                elif transcription_path.lower().endswith('.textgrid'):
                    info = parse_textgrid_file(*arguments, stop_check=self.stopped)
                else:
                    info = parse_lab_file(*arguments)
                self.return_q.put(info)
            except WavReadError:
                self.return_dict['wav_read_errors'].append(wav_path)
            except SampleRateError:
                self.return_dict['unsupported_sample_rate'].append(wav_path)
            except TextParseError:
                self.return_dict['decode_error_files'].append(transcription_path)
            except TextGridParseError as e:
                self.return_dict['textgrid_read_errors'][transcription_path] = e
            except Exception as e:
                self.stopped.stop()
                self.return_dict['error'] = arguments, Exception(traceback.format_exception(*sys.exc_info()))
        return
