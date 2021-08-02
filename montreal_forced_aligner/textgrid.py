import os
import sys
import traceback
import re
from praatio import tgio


def parse_ctm(ctm_path, corpus, dictionary, mode='word'):
    if mode == 'word':
        mapping = dictionary.reversed_word_mapping
    else:
        mapping = dictionary.reversed_phone_mapping
    file_dict = {}
    cur_utt = None
    text = None
    text_ind = 0
    current_labels = []
    with open(ctm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split(' ')
            utt = line[0]
            begin = round(float(line[2]), 4)
            duration = float(line[3])
            end = round(begin + duration, 4)
            label = line[4]
            if dictionary.has_multiple:
                d = dictionary.get_dictionary(speaker)
            else:
                d = dictionary
            if mode == 'word':
                if utt != cur_utt:
                    if cur_utt != None:
                        cur_ind = 0
                        actual_labels = []
                        for word in text:

                            ints = d.to_int(word)
                            b = 1000000
                            e = -1
                            for i in ints:
                                cur = current_labels[cur_ind]
                                if i == int(cur[2]):
                                    if cur[0] < b:
                                        b = cur[0]
                                    if cur[1] > e:
                                        e = cur[1]
                                cur_ind += 1
                            if b == 1000000 or e == -1:
                                print('UGH')
                                print(cur_utt)
                                print(text)
                                print (word)
                                print(ints)
                                print(current_labels[cur_ind])
                                print(current_labels)
                                print(dictionary._lookup(word))
                                print(dictionary.split_clitics(word))
                                print(word in dictionary.words_mapping)
                                print(dictionary.clitic_markers)
                                print(dictionary.clitic_set)
                                print(re.split(r'[{}]'.format(dictionary.clitic_markers), word, maxsplit=1))
                                initial, final = re.split(r'[{}]'.format(dictionary.clitic_markers), word, maxsplit=1)
                                if any(x in final for x in dictionary.clitic_markers):
                                    final = dictionary.split_clitics(final)
                                else:
                                    final = [final]
                                for clitic in dictionary.clitic_markers:
                                    if initial + clitic in dictionary.clitic_set:
                                        print('FOUND INITIAL')
                                        to_return = [initial + clitic] + final
                                        initial = initial + clitic

                                    elif clitic+ final[0] in dictionary.clitic_set:
                                        print('FOUND FINAL')
                                        to_return = [initial] + final
                                        final[0] = clitic + final[0]
                                print(initial, final, to_return)
                                print(dictionary.words[word])
                                raise Exception()
                            lab = [b, e, word]
                            actual_labels.append(lab)
                        file_dict[filename][speaker].extend(actual_labels)
                    speaker = corpus.utt_speak_mapping[utt]
                    if utt in corpus.segments:
                        seg = corpus.segments[utt]
                        filename = seg['file_name']
                        utt_begin = seg['begin']
                        begin += utt_begin
                        end += utt_begin
                    else:
                        filename = utt
                    if filename not in file_dict:
                        file_dict[filename] = {}
                    if speaker not in file_dict[filename]:
                        file_dict[filename][speaker] = []
                    cur_utt = utt
                    text = corpus.text_mapping[utt].split()
                    current_labels = []

                current_labels.append([begin, end, label])
            else:
                speaker = corpus.utt_speak_mapping[utt]
                if utt in corpus.segments:
                    seg = corpus.segments[utt]
                    filename = seg['file_name']
                    utt_begin = seg['begin']
                    begin += utt_begin
                    end += utt_begin
                else:
                    filename = utt
                if filename not in file_dict:
                    file_dict[filename] = {}
                if speaker not in file_dict[filename]:
                    file_dict[filename][speaker] = []
                mapping = d.reversed_phone_mapping
                label = mapping[int(label)]
                for p in dictionary.positions:
                    if label.endswith(p):
                        label = label[:-1 * len(p)]
                file_dict[filename][speaker].append([begin, end, label])
    if mode == 'word' and current_labels:

        cur_ind = 0
        actual_labels = []
        for word in text:

            ints = d.to_int(word)
            b = 1000000
            e = -1
            for i in ints:
                cur = current_labels[cur_ind]
                if i == int(cur[2]):
                    if cur[0] < b:
                        b = cur[0]
                    if cur[1] > e:
                        e = cur[1]
                cur_ind += 1
            lab = [b, e, word]
            actual_labels.append(lab)
        file_dict[filename][speaker].extend(actual_labels)
    # Sort by begins
    for k, v in file_dict.items():
        for k2, v2 in v.items():
            file_dict[k][k2] = sorted(v2)
    return file_dict

def map_to_original_pronunciation(phones, subpronunciations, strip_diacritics, digraphs):
    transcription = tuple(x[2] for x in phones)
    new_phones = []
    pron = None
    mapping_ind = 0
    for pronunciations in subpronunciations:
        if mapping_ind >= len(phones):
            break
        for p in pronunciations:
            if ('original_pronunciation' in p and transcription == p['pronunciation'] == p['original_pronunciation'])\
                    or (transcription == p['pronunciation'] and 'original_pronunciation' not in p) :
                new_phones.extend(phones)
                mapping_ind += len(phones)
                break
            if p['pronunciation'] == transcription and pron is None:
                pron = p
        if mapping_ind >= len(phones):
            break
        if not pron:
            new_phones.extend(phones)
            mapping_ind += len(phones)
            break
        p = pron
        if 'original_pronunciation' not in p or p['pronunciation'] == p['original_pronunciation']:
            new_phones.extend(phones)
            mapping_ind += len(phones)
            break
        for p in p['original_pronunciation']:
            if p == phones[mapping_ind][2]:
                new_phones.append(phones[mapping_ind])
            else:
                modded_phone = p
                new_p = phones[mapping_ind][2]
                for diacritic in strip_diacritics:
                    modded_phone = modded_phone.replace(diacritic, '')
                if modded_phone == new_p:
                    phones[mapping_ind][2] = p
                    new_phones.append(phones[mapping_ind])
                elif mapping_ind != len(phones) - 1:
                    new_p = phones[mapping_ind][2] + phones[mapping_ind + 1][2]
                    if modded_phone == new_p:
                        new_phones.append([phones[mapping_ind][0],
                                           phones[mapping_ind + 1][1],
                                           new_p])
                        mapping_ind += 1
            mapping_ind += 1
    return new_phones


def ctm_to_textgrid(word_ctm, phone_ctm, out_directory, corpus, dictionary, frame_shift=0.01):
    from .dictionary import MultispeakerDictionary
    textgrid_write_errors = {}
    if not os.path.exists(out_directory):
        os.makedirs(out_directory, exist_ok=True)
    silences = {dictionary.optional_silence, dictionary.nonoptional_silence}
    if not corpus.segments:
        for i, (k, v) in enumerate(sorted(word_ctm.items())):
            max_time = round(corpus.get_wav_duration(k), 4)
            speaker = list(v.keys())[0]
            if isinstance(dictionary, MultispeakerDictionary):
                d = dictionary.get_dictionary(speaker)
            else:
                d = dictionary
            v = list(v.values())[0]
            try:
                tg = tgio.Textgrid()
                tg.minTimestamp = 0
                tg.maxTimestamp = max_time
                words = []
                phones = []
                if dictionary.multilingual_ipa:
                    phone_ind = 0
                    for interval in v:
                        if max_time - interval[1] < frame_shift:  # Fix rounding issues
                            interval[1] = max_time
                        end = interval[1]
                        word = interval[2]
                        subwords = d._lookup(word)
                        subwords = [x if x in d.words_mapping else d.oov_code for x in subwords ]
                        subprons = [d.words[x] for x in subwords]
                        cur_phones = []
                        while phone_ctm[k][speaker][phone_ind][1] <= end:
                            p = phone_ctm[k][speaker][phone_ind]
                            if max_time - p[1] < frame_shift:  # Fix rounding issues
                                p[1] = max_time
                            if p[2] in silences:
                                phone_ind += 1
                                continue
                            cur_phones.append(p)
                            phone_ind += 1
                            if phone_ind > len(phone_ctm[k][speaker]) - 1:
                                break
                        phones.extend(map_to_original_pronunciation(cur_phones, subprons, dictionary.strip_diacritics, dictionary.digraphs))
                        if not word:
                            continue

                        words.append(interval)
                else:
                    for interval in v:
                        if max_time - interval[1] < frame_shift:  # Fix rounding issues
                            interval[1] = max_time
                        words.append(interval)
                    for j, interval in enumerate(phone_ctm[k][speaker]):
                        if max_time - interval[1] < frame_shift:  # Fix rounding issues
                            interval[1] = max_time
                        if interval[2] in silences:
                            continue
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
                if k in corpus.file_name_mapping:
                    output_name = corpus.file_name_mapping[k]
                else:
                    if k.startswith(speaker) and speaker in k.split('_')[1:]:  # deal with prosodylab speaker prefixing
                        k = '_'.join(k.split('_')[1:])
                    output_name = k
                out_path = os.path.join(speaker_directory, output_name + '.TextGrid')
                tg.save(out_path, useShortForm=False)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_write_errors[k] = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    else:
        for i, filename in enumerate(sorted(word_ctm.keys())):
            max_time = corpus.get_wav_duration(filename)
            try:
                try:
                    if not corpus.file_directory_mapping[filename]:
                        raise KeyError
                    speaker_directory = os.path.join(out_directory, corpus.file_directory_mapping[filename])
                except KeyError:
                    speaker_directory = out_directory
                tg = tgio.Textgrid()
                tg.minTimestamp = 0
                tg.maxTimestamp = max_time
                for speaker in corpus.speaker_ordering[filename]:
                    if isinstance(dictionary, MultispeakerDictionary):
                        d = dictionary.get_dictionary(speaker)
                    else:
                        d = dictionary
                    word_tier_name = '{} - words'.format(speaker)
                    phone_tier_name = '{} - phones'.format(speaker)
                    words = []
                    phones = []
                    if dictionary.multilingual_ipa:
                        phone_ind = 0
                        phone_tier_len = len(phone_ctm[filename][speaker])
                        for interval in word_ctm[filename][speaker]:
                            if max_time - interval[1] < frame_shift:  # Fix rounding issues
                                interval[1] = max_time
                            end = interval[1]
                            word = interval[2]
                            cur_phones = []
                            while phone_ind <= len(phone_ctm[filename][speaker]) - 1 and phone_ctm[filename][speaker][phone_ind][1] <= end:
                                p = phone_ctm[filename][speaker][phone_ind]
                                if max_time - p[1] < frame_shift:  # Fix rounding issues
                                    p[1] = max_time
                                if p[2] in silences:
                                    phone_ind += 1
                                    continue
                                cur_phones.append(p)
                                phone_ind += 1
                            phones.extend(map_to_original_pronunciation(cur_phones, d.words[word], dictionary.strip_diacritics, dictionary.digraphs))

                            words.append(interval)

                    else:
                        for interval in word_ctm[filename][speaker]:
                            if max_time - interval[1] < frame_shift:  # Fix rounding issues
                                interval[1] = max_time
                            words.append(interval)
                        for p in phone_ctm[filename][speaker]:
                            if max_time - p[1] < frame_shift:  # Fix rounding issues
                                p[1] = max_time
                            if p[2] in silences:
                                continue
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
    error_log = os.path.join(out_directory, 'output_errors.txt')
    if os.path.exists(error_log):
        os.remove(error_log)
    if textgrid_write_errors:
        with open(error_log, 'w', encoding='utf8') as f:
            f.write('The following exceptions were encountered during the ouput of the alignments to TextGrids:\n\n')
            for k, v in textgrid_write_errors.items():
                f.write('{}:\n'.format(k))
                f.write('{}\n\n'.format(v))

