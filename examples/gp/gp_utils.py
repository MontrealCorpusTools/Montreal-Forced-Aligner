
import os
import shutil
import re

lang_encodings = {
                'AR': 'iso-8859-1',
                'BG': 'utf8',
                'CH': 'gb2312',
                'WU': 'gb2312',
                'CR': 'iso-8859-2',
                'CZ': 'iso-8859-2',
                'FR': 'iso-8859-1',
                'GE': 'iso-8859-1',
                'HA': 'utf8',
                'JA': '',
                'KO': 'korean',
                'RU': 'koi8-r',
                'PO': 'iso-8859-1',
                'PL': 'utf8',
                'SP': 'iso-8859-1',
                'SW': 'iso-8859-1',
                'TA': '',
                'TH': 'utf8',
                'TU': 'iso-8859-9',
                'VN': 'utf8'
                }

def parse_rmn_file(path, output_dir, lang_code, wav_files):
    file_line_pattern = re.compile('^;\s+(\d+)\s*:$')
    speaker_line_pattern = re.compile('^;SprecherID\s(\d{3})$')
    speaker = None
    current = None
    with open(path, 'r') as f:
        for line in f:
            if speaker is None:
                speaker_match = speaker_line_pattern.match(line)
                if speaker_match is None:
                    raise(Exception('The rmn file did not start with the speaker id.'))
                speaker = speaker_match.groups()[0]
            line = line.strip()
            file_match = file_line_pattern.match(line)
            if file_match is not None:
                current = file_match.groups()[0]
            elif current is not None:
                name = '{}_{}'.format(speaker, current)
                if not name.startswith(lang_code):
                    name = lang_code + name
                if name not in wav_files:
                    continue
                lab_path = os.path.join(output_dir, name + '.lab')
                with open(lab_path, 'w') as fw:
                    fw.write(line)

def parse_trl_file(path, output_dir, lang_code, wav_files):
    file_line_pattern = re.compile('^;\s+(\d+)\s*:$')
    speaker_line_pattern = re.compile('^;SprecherID\s((\w{2})?\d{2,3}).*$')
    speaker = None
    current = None
    with open(path, 'r', encoding = lang_encodings[lang_code]) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if speaker is None:
                speaker_match = speaker_line_pattern.match(line)
                if speaker_match is None:
                    raise(Exception('The file \'{}\' did not start with the speaker id.'.format(path)))
                speaker = speaker_match.groups()[0]
                if len(speaker) == 2:
                    speaker = '0' + speaker
            file_match = file_line_pattern.match(line)
            if file_match is not None:
                current = file_match.groups()[0]
            elif current is not None:
                name = '{}_{}'.format(speaker, current)
                if not name.startswith(lang_code):
                    name = lang_code + name
                if name not in wav_files:
                    continue
                lab_path = os.path.join(output_dir, name+'.lab')
                with open(lab_path, 'w', encoding = 'utf8') as fw:
                    fw.write(line.lower())

def copy_wav_files(in_dir, out_dir):
    wave_files = [f for f in os.listdir(in_dir) if f.lower().endswith('.wav')]
    if len(wave_files) == 0:
        raise(Exception('No wav files found.'))

    for f in wave_files:
        base = os.path.splitext(f)[0]
        while '.' in base:
            base = os.path.splitext(base)[0]

        shutil.copy(os.path.join(in_dir, f), os.path.join(out_dir, base + '.wav'))

def get_utterances_with_wavs(speaker_dir):
    wav_files = []
    for x in os.listdir(speaker_dir):
        if not x.endswith('.wav'):
            continue
        base = os.path.splitext(x)[0]
        while '.' in base:
            base = os.path.splitext(base)[0]
        wav_files.append(base)
    return wav_files

def globalphone_prep(source_dir, data_dir, lang_code):
    files_dir = os.path.join(data_dir, 'files')
    if os.path.exists(files_dir):
        print('Using existing data directory.')
        return
    print('Creating a consolidated data directory...')
    os.makedirs(files_dir, exist_ok = True)
    rmn_dir = os.path.join(source_dir, 'rmn')
    trl_dir = os.path.join(source_dir, 'trl')
    adc_dir = os.path.join(source_dir, 'adc')

    for speaker_id in sorted(os.listdir(adc_dir)):
        speaker_dir = os.path.join(adc_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        wav_files = get_utterances_with_wavs(speaker_dir)
        output_speaker_dir = os.path.join(files_dir, speaker_id)
        os.makedirs(output_speaker_dir, exist_ok = True)
        if lang_code in ['CH', 'WU']:
            rmn_path = os.path.join(rmn_dir, '{}{}.rmn'.format(lang_code, speaker_id))
            parse_rmn_file(rmn_path, output_speaker_dir, lang_code, wav_files)
        else:
            trl_path = os.path.join(trl_dir, '{}{}.trl'.format(lang_code, speaker_id))
            parse_trl_file(trl_path, output_speaker_dir, lang_code, wav_files)
        copy_wav_files(speaker_dir, output_speaker_dir)
    print('Done!')

def globalphone_dict_prep(path, data_dir, lang_code):
    dict_dir = os.path.join(data_dir, 'dict')
    if  os.path.exists(dict_dir):
        print('Using existing dictionary.')
        return
    print('Preparing dictionary...')
    os.makedirs(dict_dir, exist_ok = True)

    extra_questions_path = os.path.join(dict_dir, 'extra_questions.txt')
    lexicon_path = os.path.join(dict_dir, 'lexicon.txt')
    lexicon_nosil_path = os.path.join(dict_dir, 'lexicon_nosil.txt')
    lexiconp_path = os.path.join(dict_dir, 'lexiconp.txt')
    nonsilence_phones_path = os.path.join(dict_dir, 'nonsilence_phones.txt')
    optional_sil_path = os.path.join(dict_dir, 'optional_silence.txt')
    sil_phone_path = os.path.join(dict_dir, 'silence_phones.txt')

    with open(lexicon_path, 'w', encoding = 'utf8') as f:
        f.write('!SIL\tsil\n<unk>\tspn\n')
    with open(sil_phone_path, 'w', encoding = 'utf8') as f:
        f.write('sil\nspn')
    with open(extra_questions_path, 'w', encoding = 'utf8') as f:
        f.write('sil spn\n')
    with open(optional_sil_path, 'w', encoding = 'utf8') as f:
        f.write('sil')

    nonsil = set()

    phone_cleanup_pattern = re.compile(r'(M_|\{| WB\}|\})')
    word_cleanup_pattern = re.compile(r'\(\d+\)')
    line_break_pattern = re.compile(r'\}\s+')
    word_pattern = re.compile(r'^{([^{}]+)\s+')
    words = []
    with open(path, 'r', encoding = 'utf8') as f:
        try:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                try:
                    word, phones = line_break_pattern.split(line, maxsplit=1)
                except ValueError:
                    raise(Exception('There was a problem with the line \'{}\'.'.format(line)))
                if 'SIL' in phones:
                    continue
                word = word[1:].strip()
                if '{' in word:
                    word = word_pattern.match(line)
                    word = word.groups()[0]
                    phones = word_pattern.sub('',line)
                word = word_cleanup_pattern.sub('', word)
                phones = phone_cleanup_pattern.sub('', phones).strip()
                matches = phones.split()
                nonsil.update(matches)
                words.append((word.lower(), ' '.join(matches)))
        except UnicodeDecodeError:
            s = f.readline()
            print(repr(s))
            print(f.readline())
            raise(Exception)

    with open(lexicon_path, 'a', encoding = 'utf8') as lf, \
        open(lexicon_nosil_path, 'w', encoding = 'utf8') as lnsf:
        for w in words:
            outline = '{}\t{}\n'.format(*w)
            lf.write(outline)
            lnsf.write(outline)

    nonsil = sorted(nonsil)
    with open(nonsilence_phones_path, 'w', encoding = 'utf8') as f:
        f.write('\n'.join(nonsil))
    with open(extra_questions_path, 'a', encoding = 'utf8') as f:
        f.write(' '.join(nonsil))
    print('Done!')
