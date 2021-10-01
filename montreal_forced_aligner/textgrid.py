import os
import sys
import traceback
import re
from praatio import textgrid as tgio


def split_clitics(item, words_mapping, clitic_set, clitic_markers, compound_markers):
    if item in words_mapping:
        return [item]
    if any(x in item for x in compound_markers):
        s = re.split(r'[{}]'.format(compound_markers), item)
        if any(x in item for x in clitic_markers):
            new_s = []
            for seg in s:
                if any(x in seg for x in clitic_markers):
                    new_s.extend(split_clitics(seg, words_mapping, clitic_set, clitic_markers, compound_markers))
                else:
                    new_s.append(seg)
            s = new_s
        return s
    if any(x in item and not item.endswith(x) and not item.startswith(x) for x in clitic_markers):
        initial, final = re.split(r'[{}]'.format(clitic_markers), item, maxsplit=1)
        if any(x in final for x in clitic_markers):
            final = split_clitics(final, words_mapping, clitic_set, clitic_markers, compound_markers)
        else:
            final = [final]
        for clitic in clitic_markers:
            if initial + clitic in clitic_set:
                return [initial + clitic] + final
            elif clitic + final[0] in clitic_set:
                final[0] = clitic + final[0]
                return [initial] + final
    return [item]

def _lookup(item, words_mapping, punctuation, clitic_set, clitic_markers, compound_markers):
    from montreal_forced_aligner.dictionary import sanitize
    if item in words_mapping:
        return [item]
    sanitized = sanitize(item, punctuation, clitic_markers)
    if sanitized in words_mapping:
        return [sanitized]
    split = split_clitics(sanitized, words_mapping, clitic_set, clitic_markers, compound_markers)
    oov_count = sum(1 for x in split if x not in words_mapping)

    if oov_count < len(split):  # Only returned split item if it gains us any transcribed speech
        return split
    return [sanitized]

def to_int(item, words_mapping, punctuation, clitic_set, clitic_markers, compound_markers, oov_int):
    """
    Convert a given word into its integer id
    """
    if item == '':
        return []
    sanitized = _lookup(item, words_mapping, punctuation, clitic_set, clitic_markers, compound_markers)
    text_int = []
    for item in sanitized:
        if not item:
            continue
        if item not in words_mapping:
            text_int.append(oov_int)
        else:
            text_int.append(words_mapping[item])
    return text_int


def parse_from_word(ctm_labels, text, words_mapping, punctuation, clitic_set, clitic_markers, compound_markers, oov_int):
    cur_ind = 0
    actual_labels = []
    for word in text:
        ints = to_int(word, words_mapping, punctuation, clitic_set, clitic_markers, compound_markers, oov_int)
        b = 1000000
        e = -1
        for i in ints:
            cur = ctm_labels[cur_ind]
            i_begin, i_end, lab = cur
            if i == int(lab):
                if i_begin < b:
                    b = i_begin
                if i_end > e:
                    e = i_end
            cur_ind += 1
        lab = [b, e, word]
        actual_labels.append(lab)
    return actual_labels


def parse_from_word_no_cleanup(ctm_labels, reversed_word_mapping):
    actual_labels = []
    for begin, end, label in ctm_labels:
        label = reversed_word_mapping[int(label)]
        actual_labels.append((begin, end, label))
    return actual_labels


def parse_from_phone(ctm_labels, reversed_phone_mapping, positions):
    actual_labels = []
    for begin, end, label in ctm_labels:
        label = reversed_phone_mapping[int(label)]
        for p in positions:
            if label.endswith(p):
                label = label[:-1 * len(p)]
        actual_labels.append((begin, end, label))
    return actual_labels


def map_to_original_pronunciation(phones, subpronunciations, strip_diacritics):
    transcription = tuple(x[2] for x in phones)
    new_phones = []
    mapping_ind = 0
    transcription_ind = 0
    for pronunciations in subpronunciations:
        pron = None
        if mapping_ind >= len(phones):
            break
        for p in pronunciations:
            if ('original_pronunciation' in p and transcription == p['pronunciation'] == p['original_pronunciation'])\
                    or (transcription == p['pronunciation'] and 'original_pronunciation' not in p) :
                new_phones.extend(phones)
                mapping_ind += len(phones)
                break
            if p['pronunciation'] == transcription[transcription_ind: transcription_ind+len(p['pronunciation'])] \
                    and pron is None:
                pron = p
        if mapping_ind >= len(phones):
            break
        if not pron:
            new_phones.extend(phones)
            mapping_ind += len(phones)
            break
        to_extend = phones[transcription_ind: transcription_ind+len(pron['pronunciation'])]
        transcription_ind += len(pron['pronunciation'])
        p = pron
        if 'original_pronunciation' not in p or p['pronunciation'] == p['original_pronunciation']:

            new_phones.extend(to_extend)
            mapping_ind += len(to_extend)
            break
        for pi in p['original_pronunciation']:
            if pi == phones[mapping_ind][2]:
                new_phones.append(phones[mapping_ind])
            else:
                modded_phone = pi
                new_p = phones[mapping_ind][2]
                for diacritic in strip_diacritics:
                    modded_phone = modded_phone.replace(diacritic, '')
                if modded_phone == new_p:
                    phones[mapping_ind] = (phones[mapping_ind][0], phones[mapping_ind][1], pi)
                    new_phones.append(phones[mapping_ind])
                elif mapping_ind != len(phones) - 1:
                    new_p = phones[mapping_ind][2] + phones[mapping_ind + 1][2]
                    if modded_phone == new_p:
                        new_phones.append([phones[mapping_ind][0], phones[mapping_ind + 1][1], new_p])
                        mapping_ind += 1
            mapping_ind += 1
    return new_phones


def ctms_to_textgrids_non_mp(align_config, output_directory, model_directory, dictionary, corpus, num_jobs, cleanup_textgrids=True):
    frame_shift = align_config.feature_config.frame_shift / 1000

    if dictionary.has_multiple:
        words_mapping = {}
        words = {}
        for name, d in dictionary.dictionary_mapping.items():
            words_mapping[name] = d.words_mapping
            words[name] = d.words
        speaker_mapping = dictionary.speaker_mapping
    else:
        words_mapping = dictionary.words_mapping
        revered_word_mapping = dictionary.reversed_word_mapping
        words = dictionary.words
        speaker_mapping = None

    backup_output_directory = None
    if not align_config.overwrite:
        backup_output_directory = os.path.join(model_directory, 'textgrids')
        os.makedirs(backup_output_directory, exist_ok=True)

    def process_current_word_labels(utterance_id):
        if utterance_id in corpus.segments:
            seg = corpus.segments[utterance_id]
            file_name = seg['file_name']
        else:
            file_name = utterance_id
        speaker = corpus.utt_speak_mapping[utterance_id]
        text = corpus.text_mapping[utterance_id].split()
        if dictionary.has_multiple:
            d = dictionary.get_dictionary(speaker)
            inner_words_mapping = d.words_mapping
            oov_int = d.oov_int
        else:
            inner_words_mapping = dictionary.words_mapping
            oov_int = dictionary.oov_int
        if cleanup_textgrids:
            actual_labels = parse_from_word(current_labels, text, inner_words_mapping, dictionary.punctuation,
                                            dictionary.clitic_set,
                                            dictionary.clitic_markers, dictionary.compound_markers, oov_int)
        else:
            actual_labels = parse_from_word_no_cleanup(current_labels, revered_word_mapping)
        if file_name not in word_data:
            word_data[file_name] = {}
        if speaker not in word_data[file_name]:
            word_data[file_name][speaker] = []
        word_data[file_name][speaker].extend(actual_labels)

    def process_current_phone_labels(utterance_id):
        if utterance_id in corpus.segments:
            seg = corpus.segments[utterance_id]
            file_name = seg['file_name']
        else:
            file_name = utterance_id
        speaker = corpus.utt_speak_mapping[utterance_id]

        if dictionary.has_multiple:
            d = dictionary.get_dictionary(speaker)
            reversed_phone_mapping = d.reversed_phone_mapping
        else:
            reversed_phone_mapping = dictionary.reversed_phone_mapping

        actual_labels = parse_from_phone(current_labels, reversed_phone_mapping, dictionary.positions)

        if file_name not in phone_data:
            phone_data[file_name] = {}
        if speaker not in phone_data[file_name]:
            phone_data[file_name][speaker] = []
        phone_data[file_name][speaker].extend(actual_labels)

    export_errors = {}
    wav_durations = corpus.file_durations
    for i in range(num_jobs):
        word_data = {}
        phone_data = {}
        corpus.logger.debug(f'Parsing ctms for job {i}...')
        word_ctm_path = os.path.join(model_directory, 'word_ctm.{}'.format(i))
        cur_utt = None
        current_labels = []

        with open(word_ctm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(' ')
                utt = line[0]
                if cur_utt is None:
                    cur_utt = utt
                if utt in corpus.segments:
                    seg = corpus.segments[utt]
                    file_name = seg['file_name']
                    utt_begin = seg['begin']
                else:
                    utt_begin = 0
                    file_name = utt
                if file_name not in wav_durations:
                    wav_durations[file_name] = corpus.get_wav_duration(utt)
                begin = round(float(line[2]), 4)
                duration = float(line[3])
                end = round(begin + duration, 4)
                label = line[4]
                if utt != cur_utt:
                    process_current_word_labels(cur_utt)
                    cur_utt = utt
                    current_labels = []

                begin += utt_begin
                end += utt_begin
                current_labels.append([begin, end, label])
        if current_labels:
            process_current_word_labels(cur_utt)
        cur_file = None
        cur_utt = None
        current_labels = []
        phone_ctm_path = os.path.join(model_directory, 'phone_ctm.{}'.format(i))
        with open(phone_ctm_path, 'r') as f:
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
                if cur_utt is None:
                    cur_utt = utt
                if utt in corpus.segments:
                    seg = corpus.segments[utt]
                    file_name = seg['file_name']
                    utt_begin = seg['begin']
                else:
                    utt_begin = 0
                    file_name = utt
                if cur_file is None:
                    cur_file = file_name
                if utt != cur_utt and cur_utt is not None:
                    process_current_phone_labels(cur_utt)
                    cur_utt = utt
                    current_labels = []

                begin += utt_begin
                end += utt_begin
                current_labels.append([begin, end, label])
        if current_labels:
            process_current_phone_labels(cur_utt)

        corpus.logger.debug(f'Generating TextGrids for job {i}...')
        processed_files = set()
        for file_name in word_data.keys():
            word_ctm = word_data[file_name]
            phone_ctm = phone_data[file_name]
            overwrite = True
            if file_name in processed_files:
                overwrite = False
            try:
                ctm_to_textgrid(file_name, word_ctm, phone_ctm, output_directory, dictionary.silences, wav_durations, dictionary.multilingual_ipa,
                     frame_shift, words_mapping, speaker_mapping,
                     dictionary.punctuation, dictionary.clitic_set, dictionary.clitic_markers, dictionary.compound_markers, dictionary.oov_code, words,
                     dictionary.strip_diacritics, corpus.file_directory_mapping, corpus.file_name_mapping, corpus.speaker_ordering, overwrite, backup_output_directory)
                processed_files.add(file_name)
            except Exception as e:
                if align_config.debug:
                    raise
                exc_type, exc_value, exc_traceback = sys.exc_info()
                export_errors[file_name] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))
    output_textgrid_writing_errors(output_directory, export_errors)

def output_textgrid_writing_errors(output_directory, written_textgrids):
    error_log = os.path.join(output_directory, 'output_errors.txt')
    if os.path.exists(error_log):
        os.remove(error_log)
    for file_name, result in written_textgrids.items():
        if not os.path.exists(error_log):
            with open(error_log, 'w', encoding='utf8') as f:
                f.write(
                    'The following exceptions were encountered during the output of the alignments to TextGrids:\n\n')
        with open(error_log, 'a', encoding='utf8') as f:
            f.write('{}:\n'.format(file_name))
            f.write('{}\n\n'.format(result))

def generate_tiers(word_ctm, phone_ctm, silences, multilingual_ipa,
                 words_mapping, speaker_mapping,
                 punctuation, clitic_set, clitic_markers, compound_markers, oov_code, words,
                 strip_diacritics, cleanup_textgrids=True):
    output = {}
    for speaker in word_ctm:
        words_mapping = words_mapping
        dictionary_words = words
        if speaker_mapping is not None:
            if speaker not in speaker_mapping:
                dict_speaker = 'default'
            else:
                dict_speaker = speaker
            words_mapping = words_mapping[speaker_mapping[dict_speaker]]
            dictionary_words = words[speaker_mapping[dict_speaker]]
        words = []
        phones = []
        if multilingual_ipa and cleanup_textgrids:
            phone_ind = 0
            for interval in word_ctm[speaker]:
                end = interval[1]
                word = interval[2]
                subwords = _lookup(word, words_mapping, punctuation, clitic_set,
                                   clitic_markers, compound_markers)
                subwords = [x if x in words_mapping else oov_code for x in subwords]
                subprons = [dictionary_words[x] for x in subwords]
                cur_phones = []
                while phone_ctm[speaker][phone_ind][1] <= end:
                    p = phone_ctm[speaker][phone_ind]
                    if p[2] in silences:
                        phone_ind += 1
                        continue
                    cur_phones.append(p)
                    phone_ind += 1
                    if phone_ind > len(phone_ctm[speaker]) - 1:
                        break
                phones.extend(map_to_original_pronunciation(cur_phones, subprons,
                                                            strip_diacritics))
                if not word:
                    continue

                words.append(interval)
        else:
            for interval in word_ctm[speaker]:
                words.append(interval)
            for j, interval in enumerate(phone_ctm[speaker]):
                if interval[2] in silences and cleanup_textgrids:
                    continue
                phones.append(interval)
        output[speaker] = {'words': words, 'phones': phones}
    return output

def construct_output_path(file_name, out_directory, file_directory_mapping, file_name_mapping,
                          speaker=None, backup_output_directory=None):
    output_name = file_name
    if file_name in file_name_mapping:
        output_name = file_name_mapping[output_name]
    elif speaker is not None:
        if output_name.startswith(speaker) and speaker in output_name.split('_')[1:]:
            # deal with prosodylab speaker prefixing
            output_name = '_'.join(output_name.split('_')[1:])
    try:
        if not file_directory_mapping[output_name]:
            raise KeyError
        speaker_directory = os.path.join(out_directory, file_directory_mapping[output_name])
    except KeyError:
        speaker_directory = out_directory
    os.makedirs(speaker_directory, exist_ok=True)
    tg_path = os.path.join(speaker_directory, output_name + '.TextGrid')
    if backup_output_directory is not None and os.path.exists(tg_path):
        tg_path = tg_path.replace(out_directory, backup_output_directory)
        os.makedirs(os.path.dirname(tg_path), exist_ok=True)
    return output_name, tg_path

def export_textgrid(file_name, output_path, speaker_data, max_time,
                 frame_shift, speaker_ordering,overwrite=True):

    if overwrite:
        # Create initial textgrid
        tg = tgio.Textgrid()
        tg.minTimestamp = 0
        tg.maxTimestamp = max_time

        if speaker_ordering and file_name in speaker_ordering:
            speakers = speaker_ordering[file_name]
            for speaker in speakers:
                word_tier_name = f'{speaker} - words'
                phone_tier_name = f'{speaker} - phones'

                word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=max_time)
                phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=max_time)
                tg.addTier(word_tier)
                tg.addTier(phone_tier)
        else:
            word_tier_name = 'words'
            phone_tier_name = 'phones'
            word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=max_time)
            phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=max_time)
            tg.addTier(word_tier)
            tg.addTier(phone_tier)
    else:
        # Use existing
        tg = tgio.openTextgrid(output_path, includeEmptyIntervals=False)

        word_tier_name = 'words'
        phone_tier_name = 'phones'
        word_tier = tgio.IntervalTier(word_tier_name, [], minT=0, maxT=max_time)
        phone_tier = tgio.IntervalTier(phone_tier_name, [], minT=0, maxT=max_time)
        tg.addTier(word_tier)
        tg.addTier(phone_tier)
    for speaker, data in speaker_data.items():
        words = data['words']
        phones = data['phones']
        for w in words:
            if max_time - w[1] < frame_shift:  # Fix rounding issues
                w[1] = max_time
        for p in phones:
            if max_time - p[1] < frame_shift:  # Fix rounding issues
                p[1] = max_time

        if speaker_ordering and file_name in speaker_ordering:
            word_tier_name = f'{speaker} - words'
            phone_tier_name = f'{speaker} - phones'
        else:
            word_tier_name = 'words'
            phone_tier_name = 'phones'
        word_tier = tgio.IntervalTier(word_tier_name, words, minT=0, maxT=max_time)
        phone_tier = tgio.IntervalTier(phone_tier_name, phones, minT=0, maxT=max_time)
        tg.replaceTier(word_tier_name, word_tier)
        tg.replaceTier(phone_tier_name, phone_tier)

    tg.save(output_path, includeBlankSpaces=True, format="long_textgrid", reportingMode='error')

def ctm_to_textgrid(file_name, word_ctm, phone_ctm, out_directory, silences, wav_durations, multilingual_ipa,
                 frame_shift, words_mapping, speaker_mapping,
                 punctuation, clitic_set, clitic_markers, compound_markers, oov_code, words,
                 strip_diacritics, file_directory_mapping, file_name_mapping, speaker_ordering, overwrite=True,
                    backup_output_directory=None):
    data = generate_tiers(word_ctm, phone_ctm, silences, multilingual_ipa,
                 words_mapping, speaker_mapping,
                 punctuation, clitic_set, clitic_markers, compound_markers, oov_code, words,
                 strip_diacritics)
    speaker = None
    if len(data) == 1:
        speaker = next(iter(data))
    output_name, output_path = construct_output_path(file_name, out_directory, file_directory_mapping, file_name_mapping,
                          speaker, backup_output_directory)
    max_time = round(wav_durations[output_name], 4)
    export_textgrid(file_name, output_path, data, max_time,
                     frame_shift,  speaker_ordering, overwrite)

