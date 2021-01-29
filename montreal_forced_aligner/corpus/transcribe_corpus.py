import os
import sys
import traceback
import time
from collections import defaultdict
from textgrid import TextGrid, IntervalTier

from .base import BaseCorpus, get_wav_info, find_exts, extract_temp_channels
from ..helper import save_groups, load_scp, save_scp

from ..exceptions import SampleRateError, CorpusError
from ..multiprocessing import segment_vad
import multiprocessing as mp
from queue import Empty
from ..multiprocessing.helper import Stopped
from ..multiprocessing.corpus import CorpusProcessWorker, parse_transcription, parse_lab_file, parse_textgrid_file


class TranscribeCorpus(BaseCorpus):
    def __init__(self, directory, output_directory,
                 speaker_characters=0,
                 num_jobs=3, debug=False, logger=None, use_mp=True):
        super(TranscribeCorpus, self).__init__(directory, output_directory,
                                               speaker_characters,
                                               num_jobs, debug, logger, use_mp)
        self.subsegments = {}
        self.vad_segments = {}
        self.subsegment_mapping = defaultdict(list)

        loaded = self._load_from_temp()
        if not loaded:
            if self.use_mp:
                self._load_from_source_mp()
            else:
                self._load_from_source()
        self.check_warnings()
        self.find_best_groupings()

    def _load_from_temp(self):
        begin_time = time.time()
        feat_path = os.path.join(self.output_directory, 'feats.scp')
        if not os.path.exists(feat_path):
            return False
        cmvn_path = os.path.join(self.output_directory, 'cmvn.scp')
        if not os.path.exists(cmvn_path):
            return False
        utt2spk_path = os.path.join(self.output_directory, 'utt2spk')
        if not os.path.exists(utt2spk_path):
            return False
        spk2utt_path = os.path.join(self.output_directory, 'spk2utt')
        if not os.path.exists(spk2utt_path):
            return False
        sr_path = os.path.join(self.output_directory, 'sr.scp')
        if not os.path.exists(sr_path):
            return False
        wav_path = os.path.join(self.output_directory, 'wav.scp')
        if not os.path.exists(wav_path):
            return False
        file_directory_path = os.path.join(self.output_directory, 'file_directory.scp')
        if not os.path.exists(file_directory_path):
            return False
        wav_info_path = os.path.join(self.output_directory, 'wav_info.scp')
        if not os.path.exists(wav_info_path):
            return False
        self.feat_mapping = load_scp(feat_path)
        self.cmvn_mapping = load_scp(cmvn_path)
        self.utt_speak_mapping = load_scp(utt2spk_path)
        self.speak_utt_mapping = load_scp(spk2utt_path)
        self.sample_rates = {int(k): v for k,v in load_scp(sr_path).items()}
        self.utt_wav_mapping = load_scp(wav_path)
        self.wav_info = load_scp(wav_info_path, float)
        self.file_directory_mapping = load_scp(file_directory_path)
        segments_path = os.path.join(self.output_directory, 'segments.scp')
        if os.path.exists(segments_path):
            self.segments = load_scp(segments_path)
        speaker_ordering_path = os.path.join(self.output_directory, 'speaker_ordering.scp')
        if os.path.exists(speaker_ordering_path):
            self.speaker_ordering = load_scp(speaker_ordering_path)
        text_path = os.path.join(self.output_directory, 'text')
        if os.path.exists(text_path):
            self.text_mapping = load_scp(text_path)
            for utt, text in self.text_mapping.items():
                self.text_mapping[utt] = ' '.join(text)
        self.logger.debug('Loaded from corpus_data temp directory in {} seconds'.format(time.time()-begin_time))
        return True

    def _load_from_source_mp(self):
        begin_time = time.time()
        manager = mp.Manager()
        job_queue = manager.Queue()
        return_queue = manager.Queue()
        return_dict = manager.dict()
        stopped = Stopped()

        procs = []
        for i in range(self.num_jobs):
            p = CorpusProcessWorker(job_queue, return_dict, return_queue, stopped)
            procs.append(p)
            p.start()

        for root, dirs, files in os.walk(self.directory, followlinks=True):
            wav_files, lab_files, textgrid_files = find_exts(files)
            relative_path = root.replace(self.directory, '').lstrip('/').lstrip('\\')
            for file_name, f in wav_files.items():
                wav_path = os.path.join(root, f)

                if file_name in textgrid_files:
                    tg_name = textgrid_files[file_name]
                    transcription_path = os.path.join(root, tg_name)
                else:
                    transcription_path = None
                job_queue.put((file_name, wav_path, transcription_path, relative_path, self.speaker_characters, self.temp_directory))
        job_queue.join()

        for p in procs:
            p.join()

        while True:
            try:
                info = return_queue.get(timeout=1)
            except Empty:
                break
            if 'segments' not in info:  # didn't have a textgrid file
                utt_name = info['utt_name']
                speaker_name = info['speaker_name']
                wav_info = info['wav_info']
                sr = wav_info['sample_rate']
                if utt_name in self.utt_wav_mapping:
                    ind = 0
                    fixed_utt_name = utt_name
                    while fixed_utt_name not in self.utt_wav_mapping:
                        ind += 1
                        fixed_utt_name = utt_name + '_{}'.format(ind)
                    utt_name = fixed_utt_name
                file_name = utt_name
                self.speak_utt_mapping[speaker_name].append(utt_name)
                self.utt_wav_mapping[utt_name] = info['wav_path']
                self.sample_rates[sr].add(speaker_name)
                self.utt_speak_mapping[utt_name] = speaker_name
                self.file_directory_mapping[utt_name] = info['relative_path']
            else:
                wav_info = info['wav_info']
                sr = wav_info['sample_rate']
                file_name = info['recording_name']
                self.wav_files.append(file_name)
                self.speaker_ordering[file_name] = info['speaker_ordering']
                for s in info['speaker_ordering']:
                    self.sample_rates[sr].add(s)
                self.segments.update(info['segments'])
                self.utt_wav_mapping.update(info['utt_wav_mapping'])
                for utt, words in info['text_mapping'].items():
                    self.text_mapping[utt] = ' '.join(words)
                self.utt_speak_mapping.update(info['utt_speak_mapping'])
                for speak, utts in info['speak_utt_mapping'].items():
                    if speak not in self.speak_utt_mapping:
                        self.speak_utt_mapping[speak] = utts
                    else:
                        self.speak_utt_mapping[speak].extend(utts)
                for fn in info['file_names']:
                    self.file_directory_mapping[fn] = info['relative_path']

            self.wav_info[file_name] = [wav_info['num_channels'], wav_info['sample_rate'], wav_info['duration']]

        for k in ['wav_read_errors', 'unsupported_sample_rate', 'unsupported_bit_depths',
                  'decode_error_files', 'textgrid_read_errors']:
            if hasattr(self, k):
                if k in return_dict:
                    if k == 'textgrid_read_errors':
                        getattr(self, k).update(return_dict[k])
                    else:
                        setattr(self, k, return_dict[k])
        self.logger.debug('Parsed corpus directory with {} jobs in {} seconds'.format(self.num_jobs, time.time()-begin_time))

    def _load_from_source(self):
        for root, dirs, files in os.walk(self.directory, followlinks=True):
            wav_files, lab_files, textgrid_files = find_exts(files)
            for file_name, f in wav_files.items():
                wav_path = os.path.join(root, f)
                try:
                    wav_info = get_wav_info(wav_path)
                    sr = wav_info['sample_rate']
                except Exception:
                    self.wav_read_errors.append(wav_path)
                    continue
                bit_depth = wav_info['bit_depth']
                wav_max_time = wav_info['duration']
                if bit_depth != 16:
                    self.unsupported_bit_depths.append(wav_path)
                    continue
                if sr < 16000:
                    self.unsupported_sample_rate.append(wav_path)
                if self.speaker_directories:
                    speaker_name = os.path.basename(root)
                else:
                    if isinstance(self.speaker_characters, int):
                        speaker_name = f[:self.speaker_characters]
                    elif self.speaker_characters == 'prosodylab':
                        speaker_name = f.split('_')[1]
                    else:
                        speaker_name = f
                speaker_name = speaker_name.strip().replace(' ', '_')
                utt_name = file_name
                if utt_name in self.utt_wav_mapping:
                    ind = 0
                    fixed_utt_name = utt_name
                    while fixed_utt_name not in self.utt_wav_mapping:
                        ind += 1
                        fixed_utt_name = utt_name + '_{}'.format(ind)
                    utt_name = fixed_utt_name
                utt_name = utt_name.strip().replace(' ', '_')
                self.utt_wav_mapping[utt_name] = wav_path
                self.speak_utt_mapping[speaker_name].append(utt_name)
                self.sample_rates[sr].add(speaker_name)
                self.utt_speak_mapping[utt_name] = speaker_name
                self.file_directory_mapping[utt_name] = root.replace(self.directory, '').lstrip('/').lstrip('\\')

                if file_name in textgrid_files:
                    tg_name = textgrid_files[file_name]
                    tg_path = os.path.join(root, tg_name)
                    tg = TextGrid()
                    try:
                        tg.read(tg_path)
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        self.textgrid_read_errors[tg_path] = '\n'.join(
                            traceback.format_exception(exc_type, exc_value, exc_traceback))
                    n_channels = wav_info['num_channels']
                    num_tiers = len(tg.tiers)
                    if n_channels == 2:
                        a_name = file_name + "_A"
                        b_name = file_name + "_B"

                        a_path, b_path = extract_temp_channels(wav_path, self.temp_directory)
                    elif n_channels > 2:
                        raise (Exception('More than two channels'))
                    self.speaker_ordering[file_name] = []
                    if not self.speaker_directories:
                        if isinstance(self.speaker_characters, int):
                            speaker_name = f[:self.speaker_characters]
                        elif self.speaker_characters == 'prosodylab':
                            speaker_name = f.split('_')[1]
                        else:
                            speaker_name = f
                        speaker_name = speaker_name.strip().replace(' ', '_')
                        self.speaker_ordering[file_name].append(speaker_name)
                    for i, ti in enumerate(tg.tiers):
                        if ti.name.lower() == 'notes':
                            continue
                        if not isinstance(ti, IntervalTier):
                            continue
                        if self.speaker_directories:
                            speaker_name = ti.name.strip().replace(' ', '_')
                            self.speaker_ordering[file_name].append(speaker_name)
                        self.sample_rates[sr].add(speaker_name)
                        for interval in ti:
                            text = interval.mark.lower().strip()
                            if not text:
                                continue

                            begin, end = round(interval.minTime, 4), round(interval.maxTime, 4)
                            if end > wav_max_time:
                                end = wav_max_time
                            utt_name = '{}_{}_{}_{}'.format(speaker_name, file_name, begin, end)
                            utt_name = utt_name.strip().replace(' ', '_').replace('.', '_')
                            if n_channels == 1:
                                if self.feat_mapping and utt_name not in self.feat_mapping:
                                    self.ignored_utterances.append(utt_name)
                                self.segments[utt_name] = '{} {} {}'.format(file_name, begin, end)
                                self.utt_wav_mapping[file_name] = wav_path
                            else:
                                if i < num_tiers / 2:
                                    utt_name += '_A'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                    self.segments[utt_name] = '{} {} {}'.format(a_name, begin, end)
                                    self.utt_wav_mapping[a_name] = a_path
                                else:
                                    utt_name += '_B'
                                    if self.feat_mapping and utt_name not in self.feat_mapping:
                                        self.ignored_utterances.append(utt_name)
                                    self.segments[utt_name] = '{} {} {}'.format(b_name, begin, end)
                                    self.utt_wav_mapping[b_name] = b_path
                            self.text_mapping[utt_name] = text
                            self.utt_speak_mapping[utt_name] = speaker_name
                            self.speak_utt_mapping[speaker_name].append(utt_name)

    def check_warnings(self):
        bad_speakers = []
        for speaker in self.speak_utt_mapping.keys():
            count = 0
            for k, v in self.sample_rates.items():
                if speaker in v:
                    count += 1
            if count > 1:
                bad_speakers.append(speaker)
        if bad_speakers:
            msg = 'The following speakers had multiple speaking rates: {}. ' \
                  'Please make sure that each speaker has a consistent sampling rate.'.format(', '.join(bad_speakers))
            self.logger.error(msg)
            raise (SampleRateError(msg))

        if len(self.speak_utt_mapping) < self.num_jobs:
            self.num_jobs = len(self.speak_utt_mapping)
        if self.num_jobs < len(self.sample_rates.keys()):
            self.num_jobs = len(self.sample_rates.keys())
            msg = 'The number of jobs was set to {}, due to the different sample rates in the dataset. ' \
                  'If you would like to use fewer parallel jobs, ' \
                  'please resample all wav files to the same sample rate.'.format(self.num_jobs)
            print('WARNING: ' + msg)
            self.logger.warning(msg)

    def initialize_corpus(self, dictionary=None):
        if not self.utt_wav_mapping:
            raise CorpusError('There were no wav files found for transcribing this corpus. Please validate the corpus.')
        split_dir = self.split_directory()
        self.write()
        if not os.path.exists(split_dir):
            self.split()
        self.figure_utterance_lengths()

    @property
    def ivector_directory(self):
        return os.path.join(self.output_directory, 'ivectors')

    @property
    def grouped_subsegments(self):
        output = []
        for g in self.groups:
            output_g = []
            for u in g:
                for new_utt in self.subsegment_mapping[u]:
                    try:
                        output_g.append([new_utt, ' '.join([self.subsegments[new_utt]['recording'],
                                                            self.subsegments[new_utt]['abs_start'],
                                                            self.subsegments[new_utt]['abs_end']]
                                                           )])
                    except KeyError:
                        pass
            output.append(output_g)
        return output

    def create_subsegments(self, feature_config, max_segment_duration=30, overlap_duration=5, max_remaining_duration=10,
                           constant_duration=False):
        frame_shift = feature_config.frame_shift / 1000
        split_dir = self.split_directory()
        subsegment_dir = os.path.join(split_dir, 'subsegments')
        os.makedirs(subsegment_dir, exist_ok=True)
        if constant_duration:
            dur_threshold = max_segment_duration
        else:
            dur_threshold = max_segment_duration + max_remaining_duration
        for job_name, vad_segs in self.vad_segments.items():
            self.subsegments[job_name] = {}
            max_frames_path = os.path.join(split_dir, 'utt2num_frames.{}'.format(job_name))
            vad_scp_path = os.path.join(split_dir, 'vad.{}.scp'.format(job_name))
            vads = load_scp(vad_scp_path)
            max_frames = load_scp(max_frames_path)
            subseg_feats = {}
            subseg_vads = {}
            subseg_spk2utt = defaultdict(list)
            subseg_utt2spk = {}
            subseg_reco2utt = defaultdict(list)
            for utt_id, parts in vad_segs.items():
                utt_max_frames = int(max_frames[parts[0]])
                utt_feats = self.feat_mapping[parts[0]]
                speak = self.utt_speak_mapping[parts[0]]

                utt_vads = vads[parts[0]]
                start_time = float(parts[1])
                end_time = float(parts[2])

                dur = end_time - start_time
                start = start_time
                while dur > dur_threshold:
                    end = start + max_segment_duration
                    start_relative = start - start_time
                    end_relative = end - start_time
                    new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                        utt_id=utt_id, s=int(100 * start_relative),
                        e=int(100 * end_relative))
                    # print("{new_utt} {utt_id} {s:.3f} {e:.3f}".format(
                    #    new_utt=new_utt, utt_id=utt_id, s=start_relative,
                    #    e=start_relative + max_segment_duration))
                    start += max_segment_duration - overlap_duration
                    dur -= max_segment_duration - overlap_duration
                    self.subsegments[job_name][new_utt] = {'recording': parts[0], 'abs_start': start, 'abs_end': end,
                                                           'rel_start': start_relative,
                                                           'rel_end': start_relative + max_segment_duration}
                    start_frame = int(start / frame_shift)
                    end_frame = int(end / frame_shift)
                    if end_frame > utt_max_frames:
                        end_frame = utt_max_frames
                    end_frame -= 1
                    subseg_spk2utt[speak].append(new_utt)
                    subseg_reco2utt[parts[0]].append(new_utt)
                    subseg_utt2spk[new_utt] = speak
                    subseg_feats[new_utt] = '{}[{}:{}]'.format(utt_feats, start_frame, end_frame)
                    subseg_vads[new_utt] = '{}[{}:{}]'.format(utt_vads, start_frame, end_frame)

                if constant_duration:
                    if dur < 0:
                        continue
                    if dur < max_remaining_duration:
                        start = max(end_time - max_segment_duration, start_time)
                    end = min(start + max_segment_duration, end_time)
                else:
                    end = end_time
                start_relative = start - start_time
                end_relative = end - start_time
                new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                    utt_id=utt_id, s=int(round(100 * (start - start_time))),
                    e=int(round(100 * (end - start_time))))
                # print("{new_utt} {utt_id} {s:.3f} {e:.3f}".format(
                #    new_utt=new_utt, utt_id=utt_id, s=start - start_time,
                #    e=end - start_time))
                self.subsegments[job_name][new_utt] = {'recording': parts[0], 'abs_start': start, 'abs_end': end,
                                                       'rel_start': start_relative, 'rel_end': end_relative}
                start_frame = int(start / frame_shift)
                end_frame = int(end / frame_shift)
                if end_frame > utt_max_frames:
                    end_frame = utt_max_frames
                end_frame -= 1
                subseg_spk2utt[speak].append(new_utt)
                subseg_reco2utt[parts[0]].append(new_utt)
                subseg_utt2spk[new_utt] = speak
                subseg_feats[new_utt] = '{}[{}:{}]'.format(utt_feats, start_frame, end_frame)
                subseg_vads[new_utt] = '{}[{}:{}]'.format(utt_vads, start_frame, end_frame)
            save_scp(((k, '{} {:.3f} {:.3f}'.format(v['recording'], v['abs_start'], v['abs_end'])) for k, v in self.subsegments[job_name].items()),
                     os.path.join(subsegment_dir, 'subsegments.{}.scp'.format(job_name)))
            save_scp(((k, v) for k, v in subseg_feats.items()), os.path.join(subsegment_dir, 'feats.{}.scp'.format(job_name)))
            save_scp(((k, v) for k, v in subseg_vads.items()), os.path.join(subsegment_dir, 'vad.{}.scp'.format(job_name)))
            save_scp(((k, v) for k, v in subseg_utt2spk.items()), os.path.join(subsegment_dir, 'utt2spk.{}'.format(job_name)))
            save_scp(((k, ' '.join(v)) for k, v in subseg_spk2utt.items()), os.path.join(subsegment_dir, 'spk2utt.{}'.format(job_name)))
            save_scp(((k, ' '.join(v)) for k, v in subseg_reco2utt.items()), os.path.join(subsegment_dir, 'reco2utt.{}'.format(job_name)))
        # Combine files
        for t in ['reco2utt', 'spk2utt']:
            with open(os.path.join(subsegment_dir, t), 'w', encoding='utf8') as out_f:
                for i in range(self.num_jobs):
                    with open(os.path.join(subsegment_dir, '{}.{}'.format(t,i)), 'r', encoding='utf8') as in_f:
                        for line in in_f:
                            out_f.write(line)

    def create_vad_segments(self, feature_config):
        segment_vad(self, feature_config)
        directory = self.split_directory()
        self.vad_segments = {}
        for i in range(self.num_jobs):
            vad_segments_path = os.path.join(directory, 'vad_segments.{}.scp'.format(i))
            self.vad_segments[i] = load_scp(vad_segments_path)
