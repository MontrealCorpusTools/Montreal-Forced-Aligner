
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
                'KO': '',
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

source_dir = r'D:\Data\GlobalPhone\Russian'

lang_code = 'RU'

data_dir = os.path.join(source_dir, 'kaldi_align_data')

dict_path = os.path.join(source_dir, 'Russian_Dict', 'Russian-GPDict.txt')

lm_path = os.path.join(source_dir, 'Russian_languageModel', 'RU.3gram.lm.gz')

temp_dir = r'D:\temp\GP\Russian'

os.makedirs(temp_dir, exist_ok = True)

def parse_rmn_file(path, output_dir, lang_code):
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
                lab_path = os.path.join(output_dir, '{}{}_{}.lab'.format(lang_code, speaker, current))
                with open(lab_path, 'w') as fw:
                    fw.write(line)

def parse_trl_file(path, output_dir, lang_code):
    file_line_pattern = re.compile('^;\s+(\d+)\s*:$')
    speaker_line_pattern = re.compile('^;SprecherID\s(\d{3})$')
    speaker = None
    current = None
    with open(path, 'r', encoding = lang_encodings[lang_code]) as f:
        for line in f:
            if speaker is None:
                speaker_match = speaker_line_pattern.match(line)
                if speaker_match is None:
                    raise(Exception('The trl file did not start with the speaker id.'))
                speaker = speaker_match.groups()[0]
            line = line.strip()
            file_match = file_line_pattern.match(line)
            if file_match is not None:
                current = file_match.groups()[0]
            elif current is not None:
                lab_path = os.path.join(output_dir, '{}{}_{}.lab'.format(lang_code, speaker, current))
                with open(lab_path, 'w', encoding = 'utf8') as fw:
                    fw.write(line.lower())

def copy_wav_files(in_dir, out_dir):
    wave_files = [f for f in os.listdir(in_dir) if f.lower().endswith('.wav')]
    if len(wave_files) == 0:
        raise(Exception('No wav files found.'))

    for f in wave_files:
        shutil.copy(os.path.join(in_dir, f), os.path.join(out_dir, f))

def globalphone_prep(source_dir, data_dir):
    files_dir = os.path.join(data_dir, 'files')
    os.makedirs(files_dir, exist_ok = True)
    rmn_dir = os.path.join(source_dir, 'rmn')
    trl_dir = os.path.join(source_dir, 'trl')
    adc_dir = os.path.join(source_dir, 'adc')

    for speaker_id in sorted(os.listdir(adc_dir)):
        speaker_dir = os.path.join(adc_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        output_speaker_dir = os.path.join(files_dir, speaker_id)
        os.makedirs(output_speaker_dir, exist_ok = True)
        #rmn_path = os.path.join(rmn_dir, '{}{}.rmn'.format(lang_code, speaker_id))
        #parse_rmn_file(rmn_path, output_speaker_dir, lang_code)
        trl_path = os.path.join(trl_dir, '{}{}.trl'.format(lang_code, speaker_id))
        parse_trl_file(trl_path, output_speaker_dir, lang_code)
        copy_wav_files(speaker_dir, output_speaker_dir)

def globalphone_dict_prep(path, data_dir):
    dict_dir = os.path.join(data_dir, 'dict')
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

    phone_pattern = re.compile('M_(?P<phone>[a-zA-Z0-9]+)')
    words = []
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            word, phones = line.split('} ', maxsplit=1)
            word = word[1:]
            matches = phone_pattern.findall(phones)
            if not matches:
                continue
            nonsil.update(matches)
            words.append((word.lower(), ' '.join(matches)))

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


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        print('Creating a consolidated data directory...')
        globalphone_prep(source_dir, data_dir)
        print('Done!')
    else:
        print('Using existing data directory.')
    globalphone_dict_prep(dict_path, data_dir)
    prepare_lang(data_dir)
