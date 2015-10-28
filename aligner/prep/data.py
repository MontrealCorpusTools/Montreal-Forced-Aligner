from collections import defaultdict
import os

from .helper import load_text

def output_mapping(mapping, path):
    with open(path, 'w', encoding = 'utf8') as f:
        for k in sorted(mapping.keys()):
            v = mapping[k]
            if isinstance(v, list):
                v = ' '.join(v)
            f.write('{} {}\n'.format(k, v))

def prepare_train_data(source_directory, train_directory):
    os.makedirs(train_directory, exist_ok = True)
    speaker_dirs = os.listdir(source_directory)
    speak_utt_mapping = defaultdict(list)
    utt_speak_mapping = {}
    utt_wav_mapping = {}
    text_mapping = {}
    for speaker_id in speaker_dirs:
        speaker_dir = os.path.join(source_directory, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue

        for f in os.listdir(speaker_dir):
            if not f.endswith('.lab'):
                continue
            utt_name = os.path.splitext(f)[0]
            path = os.path.join(speaker_dir, f)
            wav_path = path.replace('.lab', '.wav')
            text_mapping[utt_name] = load_text(path)
            speak_utt_mapping[speaker_id].append(utt_name)
            utt_wav_mapping[utt_name] = wav_path
            utt_speak_mapping[utt_name] = speaker_id

    spk2utt = os.path.join(train_directory, 'spk2utt')
    output_mapping(speak_utt_mapping, spk2utt)

    utt2spk = os.path.join(train_directory, 'utt2spk')
    output_mapping(utt_speak_mapping, utt2spk)

    text = os.path.join(train_directory, 'text')
    output_mapping(text_mapping, text)

    wavscp = os.path.join(train_directory, 'wav.scp')
    output_mapping(utt_wav_mapping, wavscp)
