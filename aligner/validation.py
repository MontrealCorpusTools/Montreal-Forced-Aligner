import os

from .prep.helper import load_word_to_int, load_scp

def validate_training_directory(data_directory, fail_ok = False):
    log_directory = os.path.join(data_directory, 'log')
    os.makedirs(log_directory, exist_ok = True)
    lang_directory = os.path.join(data_directory, 'lang')
    train_directory = os.path.join(data_directory, 'train')
    text_path = os.path.join(train_directory, 'text')

    words = load_word_to_int(lang_directory)

    text = load_scp(text_path)

    log_path = os.path.join(log_directory, 'training_validation.log')

    errors = 0

    with open(log_path, 'w', encoding = 'utf8') as logf:
        for line in text:
            utt = line.pop(0)
            missing = []
            for w in line:
                if w not in words.keys():
                    missing.append(w)
            if missing:
                errors += 1
                logf.write('Utterance {} missing: {}\n'.format(utt, ', '.join(missing)))

    if errors:
        message = 'There were {} utterances with words missing from the dictionary. Please check the log file.'.format(errors)
        if fail_ok:
            print(message)
        else:
            raise(Exception(message))
