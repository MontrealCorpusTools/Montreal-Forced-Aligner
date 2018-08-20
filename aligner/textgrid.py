import os
import sys
import traceback
from decimal import Decimal
from collections import defaultdict
from textgrid import TextGrid, IntervalTier


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
            if corpus.segments:
                filename = corpus.segments[utt]
                filename, utt_begin, utt_end = filename.split(' ')
                utt_begin = Decimal(utt_begin)
                if filename.endswith('_A') or filename.endswith('_B'):
                    filename = filename[:-2]
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


def ctm_to_textgrid(word_ctm, phone_ctm, out_directory, corpus, dictionary, frameshift=0.01):
    textgrid_write_errors = {}
    frameshift = Decimal(str(frameshift))
    if not os.path.exists(out_directory):
        os.makedirs(out_directory, exist_ok=True)
    if not corpus.segments:
        for i, (k, v) in enumerate(sorted(word_ctm.items())):
            maxtime = Decimal(str(corpus.get_wav_duration(k)))
            speaker = list(v.keys())[0]
            v = list(v.values())[0]
            try:
                tg = TextGrid(maxTime=maxtime)
                wordtier = IntervalTier(name='words', maxTime=maxtime)
                phonetier = IntervalTier(name='phones', maxTime=maxtime)
                for interval in v:
                    if maxtime - interval[1] < frameshift:  # Fix rounding issues
                        interval[1] = maxtime
                    wordtier.add(*interval)
                for interval in phone_ctm[k][speaker]:
                    if maxtime - interval[1] < frameshift:
                        interval[1] = maxtime
                    phonetier.add(*interval)
                tg.append(wordtier)
                tg.append(phonetier)
                relative = corpus.file_directory_mapping[k]
                if relative:
                    speaker_directory = os.path.join(out_directory, relative)
                else:
                    speaker_directory = out_directory
                os.makedirs(speaker_directory, exist_ok=True)
                outpath = os.path.join(speaker_directory, k + '.TextGrid')
                tg.write(outpath)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_write_errors[k] = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    else:
        silences = {dictionary.optional_silence, dictionary.nonoptional_silence}
        for i, (filename, speaker_dict) in enumerate(sorted(word_ctm.items())):
            maxtime = corpus.get_wav_duration(filename)
            try:
                speaker_directory = os.path.join(out_directory, corpus.file_directory_mapping[filename])
                tg = TextGrid(maxTime=maxtime)
                for speaker in corpus.speaker_ordering[filename]:
                    words = speaker_dict[speaker]
                    word_tier_name = '{} - words'.format(speaker)
                    phone_tier_name = '{} - phones'.format(speaker)
                    word_tier = IntervalTier(name=word_tier_name, maxTime=maxtime)
                    phone_tier = IntervalTier(name=phone_tier_name, maxTime=maxtime)
                    for w in words:
                        word_tier.add(*w)
                    for p in phone_ctm[filename][speaker]:
                        if len(phone_tier) > 0 and phone_tier[-1].mark in silences and p[2] in silences:
                            phone_tier[-1].maxTime = p[1]
                        else:
                            if len(phone_tier) > 0 and p[2] in silences and p[0] < phone_tier[-1].maxTime:
                                p = phone_tier[-1].maxTime, p[1], p[2]
                            elif len(phone_tier) > 0 and p[2] not in silences and p[0] < phone_tier[-1].maxTime and \
                                            phone_tier[-1].mark in silences:
                                phone_tier[-1].maxTime = p[0]
                            phone_tier.add(*p)
                    tg.append(word_tier)
                    tg.append(phone_tier)
                tg.write(os.path.join(speaker_directory, filename + '.TextGrid'))
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_write_errors[filename] = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    if textgrid_write_errors:
        error_log = os.path.join(out_directory, 'output_errors.txt')
        with open(error_log, 'w', encoding='utf8') as f:
            f.write('The following exceptions were encountered during the ouput of the alignments to TextGrids:\n\n')
            for k,v in textgrid_write_errors.items():
                f.write('{}:\n'.format(k))
                f.write('{}\n\n'.format(v))