
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

def find_max(input):
    return max(x[1] for x in input)

def ctm_to_textgrid(word_ctm_path, phone_ctm_path, out_directory,
                    dictionary):
    if not os.path.exists(word_ctm_path):
        return
    current = None
    word_dict = parse_ctm(word_ctm_path, dictionary, mode = 'word')
    phone_dict = parse_ctm(phone_ctm_path, dictionary, mode = 'phone')
    num_files = len(word_dict)
    for i,(k,v) in enumerate(word_dict.items()):
        maxtime = find_max(v+phone_dict[k])
        tg = TextGrid(maxTime = maxtime)
        wordtier = IntervalTier(name = 'words', maxTime = maxtime)
        phonetier = IntervalTier(name = 'phones', maxTime = maxtime)
        for interval in v:
            wordtier.add(*interval)
        for interval in phone_dict[k]:
            phonetier.add(*interval)
        tg.append(wordtier)
        tg.append(phonetier)
        outpath = os.path.join(out_directory, k + '.TextGrid')
        tg.write(outpath)
