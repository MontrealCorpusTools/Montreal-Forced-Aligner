
import os
from collections import Counter
import math
import subprocess
import shutil

from .helper import load_phone_to_int, load_word_to_int, load_text

def prepare_dict(aligner):
    dict_directory = aligner.dict_dir
    lang_directory = aligner.lang_dir
    phone_directory = aligner.phones_dir
    validate_dict_dir(aligner)

    if aligner.position_dependent_phones:
        make_position_dependent(aligner)

    make_phone_sets(aligner)
    phone_mapping = create_phone_symbol_table(aligner)
    word_mapping = create_word_file(aligner)

    create_word_boundaries(aligner)
    prepare_align_dict(aligner)
    make_lexicon_fst(aligner)

    write_oov(aligner, word_mapping)
    create_int_files(aligner, phone_mapping)

def create_int_files(aligner, phone_mapping):
    phone_directory = aligner.phones_dir
    sil = load_silence_phones(phone_directory)
    nonsil = load_nonsilence_phones(phone_directory)
    optsil = load_optional_silence(phone_directory)
    disambig = load_disambig(phone_directory)
    context_indep = load_context_indep(phone_directory)

    sil_int_path = os.path.join(phone_directory, 'silence.int')
    sil_csl_path = os.path.join(phone_directory, 'silence.csl')
    save_int(sil_int_path, sil, phone_mapping)
    save_csl(sil_csl_path, sil, phone_mapping)

    nonsil_int_path = os.path.join(phone_directory, 'nonsilence.int')
    nonsil_csl_path = os.path.join(phone_directory, 'nonsilence.csl')
    save_int(nonsil_int_path, nonsil, phone_mapping)
    save_csl(nonsil_csl_path, nonsil, phone_mapping)

    optsil_int_path = os.path.join(phone_directory, 'optional_silence.int')
    optsil_csl_path = os.path.join(phone_directory, 'optional_silence.csl')
    save_int(optsil_int_path, [optsil], phone_mapping)
    save_csl(optsil_csl_path, [optsil], phone_mapping)

    disambig_int_path = os.path.join(phone_directory, 'disambig.int')
    disambig_csl_path = os.path.join(phone_directory, 'disambig.csl')
    save_int(disambig_int_path, disambig, phone_mapping)
    save_csl(disambig_csl_path, disambig, phone_mapping)

    context_indep_int_path = os.path.join(phone_directory, 'context_indep.int')
    context_indep_csl_path = os.path.join(phone_directory, 'context_indep.csl')
    save_int(context_indep_int_path, context_indep, phone_mapping)
    save_csl(context_indep_csl_path, context_indep, phone_mapping)

    convert_to_int(os.path.join(phone_directory, 'sets.txt'), phone_mapping)
    convert_to_int(os.path.join(phone_directory, 'roots.txt'), phone_mapping)
    convert_to_int(os.path.join(phone_directory, 'extra_questions.txt'), phone_mapping)
    convert_to_int(os.path.join(phone_directory, 'word_boundary.txt'), phone_mapping)

    sil_phones = [str(phone_mapping[x]) for x in sil]
    nonsil_phones = [str(phone_mapping[x]) for x in nonsil]
    generate_topo(aligner, sil_phones, nonsil_phones)



def convert_to_int(text_path, mapping):
    int_path = text_path.replace('.txt', '.int')
    with open(text_path, 'r') as inf, open(int_path, 'w') as outf:
        for line in inf:
            line = line.strip()
            if line == '':
                continue
            line = line.split(' ')
            newline = []
            for x in line:
                try:
                    newline.append(str(mapping[x]))
                except KeyError:
                    newline.append(x)
            line = ' '.join(newline)
            outf.write('{}\n'.format(line))

def save_int(path, to_save, mapping):
    with open(path, 'w') as f:
        for line in to_save:
            f.write('{}\n'.format(mapping[line]))

def save_csl(path, to_save, mapping):
    with open(path, 'w') as f:
        f.write(':'.join(str(mapping[x]) for x in to_save))


def validate_dict_dir(aligner):
    lexiconp_path = os.path.join(aligner.dict_dir, 'lexiconp.txt')
    lexicon_path = os.path.join(aligner.dict_dir, 'lexicon.txt')
    with open(lexiconp_path, 'w', encoding = 'utf8') as outf, \
        open(lexicon_path, 'r', encoding = 'utf8') as inf:
            for line in inf:
                line = line.strip()
                try:
                    word, pron = line.split('\t')
                except ValueError:
                    print(line)
                    raise
                line = '\t'.join([word, '1.0', pron])
                outf.write(line + '\n')

def make_position_dependent(aligner):
    lexiconp_path = os.path.join(aligner.dict_dir, 'lexiconp.txt')
    lexicon_temp = os.path.join(aligner.dict_dir, 'temp.txt')
    with open(lexiconp_path, 'r', encoding = 'utf8') as inf, \
        open(lexicon_temp, 'w', encoding = 'utf8') as outf:
        for line in inf:
            line = line.strip()
            w, p, phones = line.split('\t')
            phones = phones.split(' ')
            if len(phones) == 1:
                phones[0] += '_S'
            else:
                for i in range(len(phones)):
                    if i == 0:
                        phones[i] += '_B'
                    elif i == len(phones) - 1:
                        phones[i] += '_E'
                    else:
                        phones[i] += '_I'
            phones = ' '.join(phones)
            line = '\t'.join([w, p, phones])
            outf.write(line + '\n')
    os.remove(lexiconp_path)
    os.rename(lexicon_temp, lexiconp_path)

    phone_map_file = os.path.join(aligner.dict_dir, 'phone_map.txt')
    silence_file = os.path.join(aligner.dict_dir, 'silence_phones.txt')
    nonsilence_file = os.path.join(aligner.dict_dir, 'nonsilence_phones.txt')
    with open(phone_map_file, 'w', encoding = 'utf8') as outf:
        with open(silence_file, 'r', encoding = 'utf8') as inf:
            for line in inf:
                line = line.strip()
                new_phones = [line+x for x in ['', ''] + positions]
                outf.write(' '.join(new_phones) + '\n')

        with open(nonsilence_file, 'r', encoding = 'utf8') as inf:
            for line in inf:
                line = line.strip()
                new_phones = [line+x for x in [''] + positions]
                outf.write(' '.join(new_phones) + '\n')


def load_phones(path):
    phones = []
    with open(path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            phones.append(line)
    return phones

def load_nonsilence_phones(directory):
    path = os.path.join(directory, 'nonsilence_phones.txt')
    if not os.path.exists(path):
        path = os.path.join(directory, 'nonsilence.txt')
    return load_phones(path)

def load_silence_phones(directory):
    path = os.path.join(directory, 'silence_phones.txt')
    if not os.path.exists(path):
        path = os.path.join(directory, 'silence.txt')
    return load_phones(path)

def load_context_indep(directory):
    path = os.path.join(directory, 'context_indep.txt')
    return load_phones(path)

def load_disambig(directory):
    path = os.path.join(directory, 'disambig.txt')
    return load_phones(path)

def load_optional_silence(directory):
    path = os.path.join(directory, 'optional_silence.txt')
    with open(path, 'r', encoding = 'utf8') as f:
        phone = f.read().strip()
    return phone

def disambiguate_lexicon(data_directory):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')
    lexiconp_path = os.path.join(dict_directory, 'lexiconp.txt')
    lexicon_disamb_path = os.path.join(dict_directory, 'lexiconp_disambig.txt')
    c = Counter()

    with open(lexiconp_path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            phones = line[-1]
            c.update([phones])
    current = {}
    with open(lexiconp_path, 'r', encoding = 'utf8') as inf, \
            open(lexicon_disamb_path, 'w', encoding = 'utf8') as outf:
        for line in inf:
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            phones = line[-1]
            count = c[phones]
            if count > 1:
                if phones not in current:
                    current[phones] = 0
                current[phones] += 1
                phones += ' #{}'.format(current[phones])
                line[-1] = phones
            outf.write('\t'.join(line) + '\n')

    max_disamb = max(current.values()) + 1
    with open(os.path.join(dict_directory, 'lex_ndisambig'), 'w') as f:
        f.write(str(max_disamb))

    with open(os.path.join(phone_directory, 'disambig.txt'), 'w') as f:
        for i in range(max_disamb + 1):
            f.write('#' + str(i) + '\n')



def prepare_align_dict(aligner):
    dict_directory = aligner.dict_dir
    lang_directory = aligner.lang_dir
    phone_directory = aligner.phones_dir

    lexiconp_path = os.path.join(dict_directory, 'lexiconp.txt')
    lexicon_ali_path = os.path.join(phone_directory, 'align_lexicon.txt')
    lexicon_ali_int_path = os.path.join(phone_directory, 'align_lexicon.int')

    opt_sil = load_optional_silence(dict_directory)

    words = set(['<eps> {}'.format(opt_sil)])

    with open(lexiconp_path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            w, p, phones = line.split('\t')
            words.add('{} {}'.format(w, phones))

    phone_mapping = load_phone_to_int(lang_directory)
    word_mapping = load_word_to_int(lang_directory)
    with open(lexicon_ali_path, 'w', encoding = 'utf8') as f, \
        open(lexicon_ali_int_path, 'w', encoding = 'utf8') as f2:
        for w in sorted(words):
            f.write('{}\n'.format(w))
            symbols = w.split(' ')
            word_int = word_mapping[symbols[0]]
            phone_ints = [phone_mapping[x] for x in symbols[1:]]
            phones = ' '.join(phone_ints)
            f2.write('{} {} {}\n'.format(word_int, word_int, phones))

def lexicon_to_fst(lexicon_path, lexicon_fst_path,
                    pronunciation_probabilities, sil_prob, silphone):
    if sil_prob == 0:
        loopstate = 0
        nextstate = 1
        with open(lexicon_path, 'r', encoding = 'utf8') as f, \
            open(lexicon_fst_path, 'w', encoding = 'utf8') as outf:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                elements = line.split('\t')

                w = elements.pop(0)
                if not pronunciation_probabilities:
                    pron_cost = 0
                else:
                    pron_prob = elements.pop(0)
                    pron_cost = -1 * math.log(float(pron_prob))

                pron_cost_string = ''
                if pron_cost != 0:
                    pron_cost_string = '\t{}'.pron_cost

                s = loopstate
                word_or_eps = w
                elements = elements[-1].split(' ')
                while len(elements) > 0:
                    p = elements.pop(0)
                    if len(elements) > 0:
                        ns = nextstate
                        nextstate += 1
                    else:
                        ns = loopstate
                    outf.write('\t'.join(map(str,[s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                    word_or_eps = '<eps>'
                    pron_cost_string = ""
                    s = ns
            outf.write("{}\t{}\n".format(loopstate, 0))
    else:
        def is_sil(element):
            return element == silphone
        silcost = -1 * math.log(sil_prob);
        nosilcost = -1 * math.log(1.0 - sil_prob)
        startstate = 0
        loopstate = 1
        silstate = 2
        with open(lexicon_path, 'r', encoding = 'utf8') as f, \
            open(lexicon_fst_path, 'w', encoding = 'utf8') as outf:
            outf.write('\t'.join(map(str,[startstate, loopstate, '<eps>', '<eps>', nosilcost])) + '\n')

            outf.write('\t'.join(map(str,[startstate, loopstate, silphone, '<eps>',silcost]))+"\n")
            outf.write('\t'.join(map(str,[silstate, loopstate, silphone, '<eps>']))+"\n")
            nextstate = 3

            for line in f:
                line = line.strip()
                if line == '':
                    continue
                elements = line.split('\t')

                w = elements.pop(0)
                if not pronunciation_probabilities:
                    pron_cost = 0
                else:
                    pron_prob = elements.pop(0)
                    pron_cost = -1 * math.log(float(pron_prob))

                pron_cost_string = ''
                if pron_cost != 0:
                    pron_cost_string = '\t{}'.pron_cost

                s = loopstate
                word_or_eps = w
                elements = elements[-1].split(' ')
                while len(elements) > 0:
                    p = elements.pop(0)
                    if len(elements) > 0:
                        ns = nextstate
                        nextstate += 1
                        outf.write('\t'.join(map(str,[s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                        word_or_eps = '<eps>'
                        pron_cost_string = ""
                        pron_cost = 0.0
                        s = ns
                    else:
                        if not is_sil(p):
                            local_nosilcost = nosilcost + pron_cost
                            local_silcost = silcost + pron_cost;
                            outf.write('\t'.join(map(str,[s, loopstate, p, word_or_eps, local_nosilcost]))+"\n")
                            outf.write('\t'.join(map(str,[s, silstate, p, word_or_eps, local_silcost]))+"\n")
                        else:
                            outf.write('\t'.join(map(str,[s, loopstate, p, word_or_eps]))+pron_cost_string+"\n")
            outf.write("{}\t{}\n".format(loopstate, 0))

def make_lexicon_fst(aligner):
    dict_directory = aligner.dict_dir
    lang_directory = aligner.lang_dir
    phone_directory = aligner.phones_dir

    log = os.path.join(data_directory, 'lexicon_fst.log')

    silphone = load_optional_silence(phone_directory)
    phone_mapping = load_phone_to_int(lang_directory)
    word_mapping = load_word_to_int(lang_directory)

    lexicon_path = os.path.join(dict_directory, 'lexiconp.txt')

    lexicon_fst_path = os.path.join(lang_directory, 'lexicon.text.fst')
    lexicon_to_fst(lexicon_path, lexicon_fst_path,
            aligner.pronunciation_probabilities, aligner.sil_prob, silphone)

    phones_file_path = os.path.join(lang_directory, 'phones.txt')
    words_file_path = os.path.join(lang_directory, 'words.txt')


    output_fst = os.path.join(lang_directory, 'L.fst')
    temp_fst_path = os.path.join(lang_directory, 'temp.fst')
    subprocess.call(['fstcompile', '--isymbols={}'.format(phones_file_path),
                    '--osymbols={}'.format(words_file_path),
                    '--keep_isymbols=false','--keep_osymbols=false',
                    lexicon_fst_path, temp_fst_path])

    subprocess.call(['fstarcsort', '--sort_type=olabel',
                temp_fst_path, output_fst])
