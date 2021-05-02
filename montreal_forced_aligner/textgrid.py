import os
import sys
import traceback
from decimal import Decimal
from praatio import tgio


def parse_ctm(ctm_path, corpus, dictionary, mode='word'):
    if mode == 'word':
        mapping = dictionary.reversed_word_mapping
    elif mode == 'phone':
        mapping = dictionary.reversed_phone_mapping
    file_dict = {}
    with open(ctm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split(' ')
            utt = line[0]
            begin = Decimal(line[2])
            duration = Decimal(line[3])
            end = begin + duration
            label = line[4]
            speaker = corpus.utt_speak_mapping[utt]
            if utt in corpus.segments:
                seg = corpus.segments[utt]
                filename = seg['file_name']
                utt_begin = seg['begin']
                utt_begin = Decimal(utt_begin)
                begin += utt_begin
                end += utt_begin
            else:
                filename = utt

            try:
                label = mapping[int(label)]
            except KeyError:
                pass
            if mode == 'phone':
                for p in dictionary.positions:
                    if label.endswith(p):
                        label = label[:-1 * len(p)]
            if filename not in file_dict:
                file_dict[filename] = {}
            if speaker not in file_dict[filename]:
                file_dict[filename][speaker] = []
            file_dict[filename][speaker].append([begin, end, label])

    # Sort by begins
    for k, v in file_dict.items():
        for k2, v2 in v.items():
            file_dict[k][k2] = sorted(v2)

    return file_dict


def ctm_to_textgrid(word_ctm, phone_ctm, out_directory, corpus, dictionary, frame_shift=0.01):
    textgrid_write_errors = {}
    frameshift = Decimal(str(frame_shift))
    if not os.path.exists(out_directory):
        os.makedirs(out_directory, exist_ok=True)
    if not corpus.segments:
        for i, (k, v) in enumerate(sorted(word_ctm.items())):
            max_time = Decimal(str(corpus.get_wav_duration(k)))
            speaker = list(v.keys())[0]
            v = list(v.values())[0]
            try:
                tg = tgio.Textgrid()
                tg.minTimestamp = 0
                tg.maxTimestamp = max_time
                phone_tier_len = len(phone_ctm[k][speaker])
                words = []
                phones = []
                for interval in v:
                    if max_time - interval[1] < frameshift:  # Fix rounding issues
                        interval[1] = max_time
                    words.append(interval)
                for j, interval in enumerate(phone_ctm[k][speaker]):
                    if j == phone_tier_len - 1:  # sync last phone boundary to end of audio file
                        interval[1] = max_time
                    phones.append(interval)
                word_tier = tgio.IntervalTier('words', words, minT=0, maxT=max_time)
                phone_tier = tgio.IntervalTier('phones', phones, minT=0, maxT=max_time)
                tg.addTier(word_tier)
                tg.addTier(phone_tier)
                relative = corpus.file_directory_mapping[k]
                if relative:
                    speaker_directory = os.path.join(out_directory, relative)
                else:
                    speaker_directory = out_directory
                os.makedirs(speaker_directory, exist_ok=True)
                if k.startswith(speaker) and speaker in k.split('_')[1:]:  # deal with prosodylab speaker prefixing
                    k = '_'.join(k.split('_')[1:])
                out_path = os.path.join(speaker_directory, k + '.TextGrid')
                tg.save(out_path, useShortForm=False)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_write_errors[k] = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    else:
        silences = {dictionary.optional_silence, dictionary.nonoptional_silence}
        for i, filename in enumerate(sorted(word_ctm.keys())):
            max_time = corpus.get_wav_duration(filename)
            try:
                try:
                    speaker_directory = os.path.join(out_directory, corpus.file_directory_mapping[filename])
                except KeyError:
                    speaker_directory = out_directory
                tg = tgio.Textgrid()
                tg.minTimestamp = 0
                tg.maxTimestamp = max_time
                for speaker in corpus.speaker_ordering[filename]:
                    word_tier_name = '{} - words'.format(speaker)
                    phone_tier_name = '{} - phones'.format(speaker)
                    words = []
                    phones = []
                    for interval in word_ctm[filename][speaker]:
                        if max_time - interval[1] < frameshift:  # Fix rounding issues
                            interval[1] = max_time
                        words.append(interval)
                    for p in phone_ctm[filename][speaker]:
                        if len(phones) > 0 and phones[-1][-1] in silences and p[2] in silences:
                            phones[-1][1] = p[1]
                        else:
                            if len(phones) > 0 and p[2] in silences and p[0] < phones[-1][1]:
                                p = phones[-1][1], p[1], p[2]
                            elif len(phones) > 0 and p[2] not in silences and p[0] < phones[-1][1] and \
                                    phones[-1][2] in silences:
                                phones[-1][1] = p[0]
                            phones.append(p)
                    word_tier = tgio.IntervalTier(word_tier_name, words, minT=0, maxT=max_time)
                    phone_tier = tgio.IntervalTier(phone_tier_name, phones, minT=0, maxT=max_time)
                    tg.addTier(word_tier)
                    tg.addTier(phone_tier)
                tg.save(os.path.join(speaker_directory, filename + '.TextGrid'), useShortForm=False)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_write_errors[filename] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))
    if textgrid_write_errors:
        error_log = os.path.join(out_directory, 'output_errors.txt')
        with open(error_log, 'w', encoding='utf8') as f:
            f.write('The following exceptions were encountered during the ouput of the alignments to TextGrids:\n\n')
            for k, v in textgrid_write_errors.items():
                f.write('{}:\n'.format(k))
                f.write('{}\n\n'.format(v))
