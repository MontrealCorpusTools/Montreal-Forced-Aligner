
import os
import sys
from collections import defaultdict
from textgrid import TextGrid, IntervalTier

def parse_ctm(ctm_path, dictionary, mode = 'word'):
    if mode == 'word':
        mapping = dictionary.reversed_word_mapping
    elif mode == 'phone':
        mapping = dictionary.reversed_phone_mapping

    file_dict = defaultdict(list)
    with open(ctm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split(' ')
            filename = line[0]
            begin = float(line[2])
            duration = float(line[3])
            end = round(begin + duration, 2)
            label = line[4]
            try:
                label = mapping[int(label)]
            except KeyError:
                pass
            if mode == 'phone':
                for p in dictionary.positions:
                    if label.endswith(p):
                        label = label[:-1 * len(p)]
            file_dict[filename].append([begin, end, label])
    return file_dict

def ctm_to_textgrid(word_ctm, phone_ctm, out_directory, corpus):

    if not os.path.exists(out_directory):
        os.makedirs(out_directory, exist_ok=True)
    if not corpus.segments:
        for i,(k,v) in enumerate(word_ctm.items()):
            maxtime = corpus.get_wav_duration(k)
            tg = TextGrid(maxTime = maxtime)
            wordtier = IntervalTier(name = 'words', maxTime = maxtime)
            phonetier = IntervalTier(name = 'phones', maxTime = maxtime)
            for interval in v:
                wordtier.add(*interval)
            for interval in phone_ctm[k]:
                phonetier.add(*interval)
            tg.append(wordtier)
            tg.append(phonetier)
            if corpus.speaker_directories:
                speaker_directory = os.path.join(out_directory, corpus.utt_speak_mapping[k])
            else:
                speaker_directory = out_directory
            if not os.path.exists(speaker_directory):
                os.makedirs(speaker_directory, exist_ok=True)
            outpath = os.path.join(speaker_directory, k + '.TextGrid')
            try:
                tg.write(outpath)
            except ValueError as e:
                print('Could not write textgrid for {}'.format(k))
                print(e)
    else:
        tgs = {}
        for i,(k,v) in enumerate(word_ctm.items()):
            rec = corpus.segments[k]
            rec, begin, end = rec.split(' ')
            maxtime = corpus.get_wav_duration(k)
            if rec not in tgs:
                tgs[rec] = TextGrid(maxTime = maxtime)
            tg = tgs[rec]
            begin = float(begin)
            speaker = corpus.utt_speak_mapping[k]
            word_tier_name = '{} - words'.format(speaker)
            phone_tier_name = '{} - phones'.format(speaker)
            wordtier = tg.getFirst(word_tier_name)
            if wordtier is None:
                wordtier = IntervalTier(name = word_tier_name, maxTime = maxtime)
                tg.append(wordtier)
            phonetier = tg.getFirst(phone_tier_name)
            if phonetier is None:
                phonetier = IntervalTier(name = phone_tier_name, maxTime = maxtime)
                tg.append(phonetier)
            for interval in v:
                interval = interval[0] + begin, interval[1] + begin, interval[2]
                wordtier.add(*interval)
            for interval in phone_ctm[k]:
                interval = interval[0] + begin, interval[1] + begin, interval[2]
                phonetier.add(*interval)
        for k,v in tgs.items():
            outpath = os.path.join(out_directory, k + '.TextGrid')
            try:
                v.write(outpath)
            except ValueError as e:
                print('Could not write textgrid for {}'.format(k))
                print(e)


