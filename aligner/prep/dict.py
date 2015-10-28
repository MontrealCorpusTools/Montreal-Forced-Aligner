
import os
from collections import Counter
import math
import subprocess
import shutil

from .helper import load_phone_to_int, load_word_to_int

def prepare_dict(data_directory, oov_code = "<unk>",
                position_dependent_phones = True,
                num_sil_states = 5,
                num_nonsil_states = 3,
                share_silence_phones = False,
                sil_prob = 0.5,
                reverse = False):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')
    validate_dict_dir(dict_directory)

    if position_dependent_phones:
        make_position_dependent(dict_directory)

    os.makedirs(phone_directory, exist_ok = True)
    make_phone_sets(data_directory, share_silence_phones)
    disambiguate_lexicon(data_directory)
    phone_mapping = create_phone_symbol_table(data_directory)
    word_mapping = create_word_file(data_directory)

    create_word_boundaries(data_directory, position_dependent_phones)
    prepare_align_dict(data_directory)
    make_lexicon_fst(data_directory)

    write_oov(oov_code, lang_directory, word_mapping)
    create_int_files(lang_directory, phone_mapping, num_sil_states, num_nonsil_states)

def create_int_files(lang_directory, phone_mapping,
                    num_sil_states, num_nonsil_states):
    phone_directory = os.path.join(lang_directory, 'phones')
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
    convert_to_int(os.path.join(phone_directory, 'word_boundary.txt'), phone_mapping)
    convert_to_int(os.path.join(phone_directory, 'word_boundary.txt'), phone_mapping)

    sil_phones = [str(phone_mapping[x]) for x in sil]
    nonsil_phones = [str(phone_mapping[x]) for x in nonsil]
    generate_topo(lang_directory, sil_phones, nonsil_phones, num_sil_states, num_nonsil_states)

def generate_topo(lang_directory, sil_phones, nonsil_phones,
                num_sil_states = 5, num_nonsil_states = 3):
    filepath = os.path.join(lang_directory, 'topo')
    template = '<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>'
    with open(filepath, 'w') as f:
        f.write('<Topology>\n')
        f.write("<TopologyEntry>\n")
        f.write("<ForPhones>\n")
        f.write("{}\n".format(' '.join(nonsil_phones)))
        f.write("</ForPhones>\n")
        states = [template.format(cur_state = x, next_state = x + 1)
                    for x in range(num_nonsil_states)]
        f.write('\n'.join(states))
        f.write("\n<State> {} </State>\n".format(num_nonsil_states))
        f.write("</TopologyEntry>\n")

        f.write('<Topology>\n')
        f.write("<TopologyEntry>\n")
        f.write("<ForPhones>\n")
        f.write("{}\n".format(' '.join(sil_phones)))
        f.write("</ForPhones>\n")
        states = [template.format(cur_state = x, next_state = x + 1)
                    for x in range(num_sil_states)]
        f.write('\n'.join(states))
        f.write("\n<State> {} </State>\n".format(num_sil_states))
        f.write("</TopologyEntry>\n")
        f.write("</Topology>\n")



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

def write_oov(oov_code, lang_directory, word_mapping):
    with open(os.path.join(lang_directory,'oov.txt'),'w') as f:
        f.write(oov_code)
    save_int(os.path.join(lang_directory,'oov.int'), [oov_code], word_mapping)


def validate_dict_dir(dict_directory):
    lexiconp_path = os.path.join(dict_directory, 'lexiconp.txt')
    lexicon_path = os.path.join(dict_directory, 'lexicon.txt')
    with open(lexiconp_path, 'w', encoding = 'utf8') as outf, \
        open(lexicon_path, 'r', encoding = 'utf8') as inf:
            for line in inf:
                line = line.strip()
                word, pron = line.split('\t')
                line = '\t'.join([word, '1.0', pron])
                outf.write(line + '\n')

def make_position_dependent(dict_directory):
    lexiconp_path = os.path.join(dict_directory, 'lexiconp.txt')
    lexicon_temp = os.path.join(dict_directory, 'temp.txt')
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

    phone_map_file = os.path.join(dict_directory, 'phone_map.txt')
    silence_file = os.path.join(dict_directory, 'silence_phones.txt')
    nonsilence_file = os.path.join(dict_directory, 'nonsilence_phones.txt')
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

positions = ["_B", "_E", "_I", "_S"]

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

def make_phone_sets(data_directory, shared_silence_phones):
    sharesplit = ['shared', 'split']
    if shared_silence_phones:
        sil_sharesplit = ['not-shared', 'not-split']
    else:
        sil_sharesplit = sharesplit
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')

    silence_phones = load_silence_phones(dict_directory)
    nonsilence_phones = load_nonsilence_phones(dict_directory)

    sets_file = os.path.join(phone_directory, 'sets.txt')
    roots_file = os.path.join(phone_directory, 'roots.txt')

    phone_silence = os.path.join(phone_directory, 'silence.txt')
    phone_nonsilence = os.path.join(phone_directory, 'nonsilence.txt')

    phone_map_file = os.path.join(dict_directory, 'phone_map.txt')
    phone_map = {}
    with open(phone_map_file, 'r', encoding = 'utf8') as inf:
        for i, line in enumerate(inf):
            line = line.strip()
            line = line.split(' ')
            phone_map[line[0]] = line[1:]


    with open(sets_file, 'w', encoding = 'utf8') as setf, \
                open(roots_file, 'w', encoding = 'utf8') as rootf:

        #process silence phones
        with open(phone_silence, 'w', encoding = 'utf8') as silf:
            for i, line in enumerate(silence_phones):
                line = line.strip()
                mapped = phone_map[line]
                setf.write(' '.join(mapped) + '\n')
                for item in mapped:
                    silf.write(item + '\n')
                if i == 0:
                    line = sil_sharesplit + mapped
                else:
                    line = sharesplit + mapped
                rootf.write(' '.join(line) + '\n')

        #process nonsilence phones
        with open(phone_nonsilence, 'w', encoding = 'utf8') as nonsilf:
            for line in nonsilence_phones:
                line = line.strip()
                mapped = phone_map[line]
                setf.write(' '.join(mapped) + '\n')
                for item in mapped:
                    nonsilf.write(item + '\n')
                line = sharesplit + mapped
                rootf.write(' '.join(line) + '\n')

    shutil.copy(os.path.join(dict_directory, 'optional_silence.txt'),
                os.path.join(phone_directory, 'optional_silence.txt'))
    shutil.copy(phone_silence, os.path.join(phone_directory, 'context_indep.txt'))
    dict_extra = os.path.join(dict_directory, 'extra_questions.txt')
    phone_extra = os.path.join(phone_directory, 'extra_questions.txt')
    with open(dict_extra, 'r', encoding = 'utf8') as inf, \
        open(phone_extra, 'w', encoding = 'utf8') as outf:
        for line in inf:
            line = line.strip()
            if line == '':
                continue
            line = line.split()
            for i in range(len(line)):
                line[i] = ' '.join(phone_map[line[i]])
            outf.write(' '.join(line) + '\n')
        for p in positions:
            line = [x + p for x in nonsilence_phones]
            outf.write(' '.join(line) + '\n')
        for p in [''] + positions:
            line = [x + p for x in silence_phones]
            outf.write(' '.join(line) + '\n')

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

def create_phone_symbol_table(data_directory):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')


    outfile = os.path.join(lang_directory, 'phones.txt')

    silence_phones = load_silence_phones(phone_directory)
    nonsilence_phones = load_nonsilence_phones(phone_directory)
    disambig = load_disambig(phone_directory)

    mapping = {}

    with open(outfile, 'w', encoding = 'utf8') as f:
        for i, p in enumerate(['eps'] + silence_phones + nonsilence_phones + disambig):
            f.write(' '.join([p, str(i)]) + '\n')
            mapping[p] = str(i)
    return mapping

def create_word_boundaries(data_directory, position_dependent_phones):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')

    silence_phones = load_silence_phones(phone_directory)
    nonsilence_phones = load_nonsilence_phones(phone_directory)

    boundary_path = os.path.join(phone_directory, 'word_boundary.txt')

    with open(boundary_path, 'w', encoding = 'utf8') as f:
        if position_dependent_phones:
            for p in silence_phones + nonsilence_phones:
                cat = 'nonword'
                if p.endswith('_B'):
                    cat = 'begin'
                elif p.endswith('_S'):
                    cat = 'singleton'
                elif p.endswith('_I'):
                    cat = 'internal'
                elif p.endswith('_E'):
                    cat = 'end'
                f.write(' '.join([p, cat])+'\n')

def create_word_file(data_directory):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')
    lexiconp_path = os.path.join(dict_directory, 'lexiconp.txt')
    words_path = os.path.join(lang_directory, 'words.txt')
    words = set()
    mapping = {}
    with open(lexiconp_path, 'r', encoding = 'utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            words.add(line[0])

    with open(words_path, 'w', encoding = 'utf8') as f:
        i = 0
        f.write('{} {}\n'.format('<eps>', i))
        mapping['<eps>'] = i
        for w in sorted(words):
            i += 1
            f.write('{} {}\n'.format(w, i))
            mapping[w] = i

        f.write('{} {}\n'.format('#0', i + 1))
        mapping['#0'] = i + 1
        f.write('{} {}\n'.format('<s>', i + 2))
        mapping['<s>'] = i + 2
        f.write('{} {}\n'.format('<\s>', i + 3))
        mapping['<\s>'] = i + 3
    return mapping

def prepare_align_dict(data_directory):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')

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
                    pronunciation_probabilities, sil_prob):
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

def make_lexicon_fst_disambig(data_directory, pronunciation_probabilities = True,
                    sil_prob = 0.5):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')

    silphone = load_optional_silence(phone_directory)
    phone_mapping = load_phone_to_int(lang_directory)
    word_mapping = load_word_to_int(lang_directory)

    lexicon_disamb_path = os.path.join(dict_directory, 'lexiconp_disambig.txt')

    lexicon_fst_path = os.path.join(lang_directory, 'lexicon_disambig.text.fst')
    lexicon_to_fst(lexicon_disamb_path, lexicon_fst_path, pronunciation_probabilities, sil_prob)
    temp_fst_path = os.path.join(lang_directory, 'temp.fst')
    phones_file_path = os.path.join(lang_directory, 'phones.txt')
    words_file_path = os.path.join(lang_directory, 'words.txt')
    phone_disambig_symbol = phone_mapping['#0']
    phone_disambig_symbol_path = os.path.join(phone_directory, 'phone_disambig_symbol')
    with open(phone_disambig_symbol_path, 'w') as f:
        f.write(str(phone_disambig_symbol))
    word_disambig_symbol = word_mapping['#0']
    word_disambig_symbol_path = os.path.join(phone_directory, 'word_disambig_symbol')
    with open(word_disambig_symbol_path, 'w') as f:
        f.write(str(word_disambig_symbol))


    output_fst = os.path.join(lang_directory, 'L_disambig.fst')
    comp_proc = subprocess.Popen(['fstcompile', '--isymbols={}'.format(phones_file_path),
                    '--osymbols={}'.format(words_file_path),
                    '--keep_isymbols=false','--keep_osymbols=false',
                    lexicon_fst_path], stdout = subprocess.PIPE)
    selfloop_proc = subprocess.Popen(['fstaddselfloops',
        phone_disambig_symbol_path, word_disambig_symbol_path],
        stdin = comp_proc.stdout, stdout = subprocess.PIPE)
    with open(output_fst, 'wb') as f:
        sort_proc = subprocess.Popen(['fstarcsort', '--sort_type=olabel',
                     output_fst], stdin = selfloop_proc.stdout,
                     stdout = f)
        sort_proc.wait()

def make_lexicon_fst(data_directory, pronunciation_probabilities = True,
                    sil_prob = 0.5):
    dict_directory = os.path.join(data_directory, 'dict')
    lang_directory = os.path.join(data_directory, 'lang')
    phone_directory = os.path.join(lang_directory, 'phones')

    log = os.path.join(data_directory, 'lexicon_fst.log')

    silphone = load_optional_silence(phone_directory)
    phone_mapping = load_phone_to_int(lang_directory)
    word_mapping = load_word_to_int(lang_directory)

    lexicon_path = os.path.join(dict_directory, 'lexiconp.txt')

    lexicon_fst_path = os.path.join(lang_directory, 'lexicon.text.fst')
    lexicon_to_fst(lexicon_path, lexicon_fst_path, pronunciation_probabilities, sil_prob)

    phones_file_path = os.path.join(lang_directory, 'phones.txt')
    words_file_path = os.path.join(lang_directory, 'words.txt')


    output_fst = os.path.join(lang_directory, 'L.fst')
    #with open(log, 'w') as logf:
        #with open(lexicon_fst_path,'r') as inf:
            #comp_proc = subprocess.Popen(['fstcompile', '--isymbols={}'.format(phones_file_path),
                            #'--osymbols={}'.format(words_file_path),
                            #'--keep_isymbols=false','--keep_osymbols=false'
                            #], stdout = subprocess.PIPE,
                            #stdin = inf,
                            #stderr = logf)
        #with open(output_fst, 'wb') as f:
            #sort_proc = subprocess.Popen(['fstarcsort', '--sort_type=olabel',
                         #output_fst], stdin = comp_proc.stdout,
                         #stdout = f,
                        #stderr = logf)
            #sort_proc.wait()
    temp_fst_path = os.path.join(lang_directory, 'temp.fst')
    subprocess.call(['fstcompile', '--isymbols={}'.format(phones_file_path),
                    '--osymbols={}'.format(words_file_path),
                    '--keep_isymbols=false','--keep_osymbols=false',
                    lexicon_fst_path, temp_fst_path])

    subprocess.call(['fstarcsort', '--sort_type=olabel',
                temp_fst_path, output_fst])
