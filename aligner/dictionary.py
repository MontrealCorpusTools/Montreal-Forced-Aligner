import os
import shutil
import math
import subprocess
import re
from collections import defaultdict

from .helper import thirdparty_binary

def compile_graphemes(graphemes):
    if '-' in graphemes:
        base = r'^\W*([-{}]+)\W*'
    else:
        base = r'^\W*([{}]+)\W*'
    string = ''.join(x for x in graphemes if x != '-')
    return re.compile(base.format(string))

class Dictionary(object):
    topo_template = '<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>'
    topo_sil_template = '<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>'
    topo_transition_template = '<Transition> {} {}'
    positions = ["_B", "_E", "_I", "_S"]
    clitic_markers = ["'", '-']
    @staticmethod
    def read(filename):
        pass

    def __init__(self, input_path, output_directory, oov_code = '<unk>',
                    position_dependent_phones = True, num_sil_states = 5,
                    num_nonsil_states = 3, shared_silence_phones = False,
                    pronunciation_probabilities = True,
                    sil_prob = 0.5):
        self.output_directory = os.path.join(output_directory, 'dictionary')
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.position_dependent_phones = position_dependent_phones
        self.pronunciation_probabilities = pronunciation_probabilities

        self.words = defaultdict(list)
        self.nonsil_phones = set()
        self.sil_phones = set(['sil', 'spn'])
        self.optional_silence = 'sil'
        self.disambig = set()
        self.graphemes = set()
        with open(input_path, 'r', encoding = 'utf8') as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                word = line.pop(0).lower()
                if word in ['!sil', oov_code]:
                    continue
                self.graphemes.update(word)
                pron = line
                self.words[word].append(pron)
                self.nonsil_phones.update(pron)
        self.word_pattern = compile_graphemes(self.graphemes)
        self.words['!SIL'].append(['sil'])
        self.words[self.oov_code].append(['spn'])
        self.phone_mapping = {}
        i = 0
        self.phone_mapping['<eps>'] = i
        if self.position_dependent_phones:
            for p in self.positional_sil_phones:
                i += 1
                self.phone_mapping[p] = i
            for p in self.positional_nonsil_phones:
                i += 1
                self.phone_mapping[p] = i
        else:
            for p in sorted(self.sil_phones):
                i += 1
                self.phone_mapping[p] = i
            for p in sorted(self.nonsil_phones):
                i += 1
                self.phone_mapping[p] = i
        for p in sorted(self.disambig):
            i += 1
            self.phone_mapping[p] = i

        self.words_mapping = {}
        i = 0
        self.words_mapping['<eps>'] = i
        for w in sorted(self.words.keys()):
            i += 1
            self.words_mapping[w] = i

        self.words_mapping['#0'] = i + 1
        self.words_mapping['<s>'] = i + 2
        self.words_mapping['</s>'] = i + 3

        self.oovs_found = set()

    def to_int(self, item):
        m = self.word_pattern.match(item)
        if m is None:
            return None
        item = m.groups()[0]
        if item not in self.words_mapping:
            self.oovs_found.add(item)
            return self.oov_int
        return self.words_mapping[item]

    def save_oovs_found(self, directory):
        with open(os.path.join(directory, 'oovs_found.txt'), 'w', encoding = 'utf8') as f:
            for oov in sorted(self.oovs_found):
                f.write(oov + '\n')
        self.oovs_found = set()

    def separate_clitics(self, item):
        vocab = []
        chars = list(item)
        count = 0
        for i in chars:
            if i in self.clitic_markers:
                count = count + 1
        if item not in self.words:
            for i in range(count):
                for punc in chars:
                    if punc in self.clitic_markers:
                        idx = chars.index(punc)
                        option1withpunc = ''.join(chars[:idx+1])
                        option1nopunc = ''.join(chars[:idx])
                        option2withpunc = ''.join(chars[idx:])
                        option2nopunc = ''.join(chars[idx+1:])
                        if option1withpunc in self.words:
                            vocab.append(option1withpunc)
                            if option2nopunc in self.words:
                                vocab.append(option2nopunc)
                            elif all(x not in list(option2nopunc) for x in self.clitic_markers):
                                vocab.append(option2nopunc)
                        else:
                            vocab.append(option1nopunc)
                            if option2withpunc in self.words:
                                vocab.append(option2withpunc)
                            elif option2nopunc in self.words:
                                vocab.append(option2nopunc)
                            elif all(x not in list(option2nopunc) for x in self.clitic_markers):
                                vocab.append(option2nopunc)
                        chars = list(option2nopunc)
        else:
            return [item]
        if vocab == []:
            return [item]
        elif len(vocab) > 0:
            unk = []
            for i in vocab:
                if i not in self.words:
                    unk.append(i)
            if len(unk) == count + 1:
                return [item]
            return vocab

    @property
    def reversed_word_mapping(self):
        mapping = {}
        for k,v in self.words_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def reversed_phone_mapping(self):
        mapping = {}
        for k,v in self.phone_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def oov_int(self):
        return self.words_mapping[self.oov_code]

    @property
    def positional_sil_phones(self):
        sil_phones = []
        for p in sorted(self.sil_phones):
            sil_phones.append(p)
            for pos in self.positions:
                sil_phones.append(p+pos)
        return sil_phones

    @property
    def positional_nonsil_phones(self):
        nonsil_phones = []
        for p in sorted(self.nonsil_phones):
            for pos in self.positions:
                nonsil_phones.append(p+pos)
        return nonsil_phones

    @property
    def optional_silence_csl(self):
        return '{}'.format(self.phone_mapping[self.optional_silence])

    @property
    def silence_csl(self):
        if self.position_dependent_phones:
            return ':'.join(map(str,(self.phone_mapping[x] for x in self.positional_sil_phones)))
        else:
            return ':'.join(map(str,(self.phone_mapping[x] for x in self.sil_phones)))

    @property
    def phones_dir(self):
        return os.path.join(self.output_directory, 'phones')

    @property
    def phones(self):
        return self.sil_phones & self.nonsil_phones

    def write(self):
        print('Creating dictionary information...')
        if not os.path.exists(self.phones_dir):
            os.makedirs(self.phones_dir, exist_ok = True)
        #self._write_lexicon()
        #self._write_lexiconp()

        self._write_graphemes()
        self._write_phone_map_file()
        self._write_phone_sets()
        self._write_phone_symbol_table()
        self._write_topo()
        self._write_word_boundaries()
        self._write_extra_questions()
        self._write_word_file()
        self._write_fst_text()
        self._write_fst_binary()

    def cleanup(self):
        os.remove(os.path.join(self.output_directory, 'temp.fst'))
        os.remove(os.path.join(self.output_directory, 'lexicon.text.fst'))

    def _write_graphemes(self):
        outfile = os.path.join(self.output_directory, 'graphemes.txt')
        with open(outfile, 'w', encoding = 'utf8') as f:
            for char in sorted(self.graphemes):
                f.write(char + '\n')

    def _write_lexicon(self):
        outfile = os.path.join(self.output_directory, 'lexicon.txt')
        with open(outfile, 'w', encoding = 'utf8') as f:
            for w in sorted(self.words.keys()):
                for p in sorted(self.words[w]):
                    phones = [x for x in p]
                    if self.position_dependent_phones:
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
                    f.write('{}\t{}\n'.format(w, phones))

    def _write_lexiconp(self):
        outfile = os.path.join(self.output_directory, 'lexiconp.txt')
        with open(outfile, 'w', encoding = 'utf8') as f:
            for w in sorted(self.words.keys()):
                for p in sorted(self.words[w]):
                    phones = [x for x in p]
                    if self.position_dependent_phones:
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
                    p = 1.0
                    f.write('{}\t{}\t{}\n'.format(w, p, phones))

    def _write_phone_map_file(self):
        outfile = os.path.join(self.output_directory, 'phone_map.txt')
        with open(outfile, 'w', encoding = 'utf8') as f:
            for sp in self.sil_phones:
                if self.position_dependent_phones:
                    new_phones = [sp+x for x in ['', ''] + self.positions]
                else:
                    new_phones = [sp]
                f.write(' '.join(new_phones) + '\n')
            for nsp in self.nonsil_phones:
                if self.position_dependent_phones:
                    new_phones = [nsp+x for x in [''] + self.positions]
                else:
                    new_phones = [nsp]
                f.write(' '.join(new_phones) + '\n')

    def _write_phone_symbol_table(self):
        outfile = os.path.join(self.output_directory, 'phones.txt')
        with open(outfile, 'w', encoding = 'utf8') as f:
            for p, i in sorted(self.phone_mapping.items(), key = lambda x: x[1]):
                f.write('{} {}\n'.format(p, i))

    def _write_word_boundaries(self):
        boundary_path = os.path.join(self.output_directory, 'phones', 'word_boundary.txt')
        boundary_int_path = os.path.join(self.output_directory, 'phones', 'word_boundary.int')
        with open(boundary_path, 'w', encoding = 'utf8') as f, \
            open(boundary_int_path, 'w', encoding ='utf8') as intf:
            if self.position_dependent_phones:
                for p in sorted(self.phone_mapping.keys(),
                            key = lambda x: self.phone_mapping[x]):
                    if p == '<eps>':
                        continue
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
                    intf.write(' '.join([str(self.phone_mapping[p]), cat])+'\n')

    def _write_word_file(self):
        words_path = os.path.join(self.output_directory, 'words.txt')

        with open(words_path, 'w', encoding = 'utf8') as f:
            for w, i in sorted(self.words_mapping.items(), key = lambda x: x[1]):
                f.write('{} {}\n'.format(w, i))

    def _write_topo(self):
        filepath = os.path.join(self.output_directory, 'topo')
        sil_transp = 1 / (self.num_sil_states - 1)
        sil_transp = 1 / (self.num_sil_states - 1)
        initial_transition = [self.topo_transition_template.format(x, sil_transp)
                                for x in range(self.num_sil_states - 1)]
        middle_transition = [self.topo_transition_template.format(x, sil_transp)
                                for x in range(1, self.num_sil_states)]
        final_transition = [self.topo_transition_template.format(self.num_sil_states - 1, 0.75),
                                self.topo_transition_template.format(self.num_sil_states, 0.25)]
        with open(filepath, 'w') as f:
            f.write('<Topology>\n')
            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            if self.position_dependent_phones:
                phones = self.positional_nonsil_phones
            else:
                phones = sorted(self.nonsil_phones)
            f.write("{}\n".format(' '.join(str(self.phone_mapping[x]) for x in phones)))
            f.write("</ForPhones>\n")
            states = [self.topo_template.format(cur_state = x, next_state = x + 1)
                        for x in range(self.num_nonsil_states)]
            f.write('\n'.join(states))
            f.write("\n<State> {} </State>\n".format(self.num_nonsil_states))
            f.write("</TopologyEntry>\n")

            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            if self.position_dependent_phones:
                phones = self.positional_sil_phones
            else:
                phones = self.sil_phones
            f.write("{}\n".format(' '.join(str(self.phone_mapping[x]) for x in phones)))
            f.write("</ForPhones>\n")
            states = []
            for i in range(self.num_sil_states):
                if i == 0:
                    transition = ' '.join(initial_transition)
                elif i == self.num_sil_states - 1:
                    transition = ' '.join(final_transition)
                else:
                    transition = ' '.join(middle_transition)
                states.append(self.topo_sil_template.format(cur_state = i,
                                                transitions = transition))
            f.write('\n'.join(states))
            f.write("\n<State> {} </State>\n".format(self.num_sil_states))
            f.write("</TopologyEntry>\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self):
        sharesplit = ['shared', 'split']
        if self.shared_silence_phones:
            sil_sharesplit = ['not-shared', 'not-split']
        else:
            sil_sharesplit = sharesplit

        sets_file = os.path.join(self.output_directory, 'phones', 'sets.txt')
        roots_file = os.path.join(self.output_directory, 'phones', 'roots.txt')

        sets_int_file = os.path.join(self.output_directory, 'phones', 'sets.int')
        roots_int_file = os.path.join(self.output_directory, 'phones', 'roots.int')


        with open(sets_file, 'w', encoding = 'utf8') as setf, \
                    open(roots_file, 'w', encoding = 'utf8') as rootf,\
                    open(sets_int_file, 'w', encoding = 'utf8') as setintf, \
                    open(roots_int_file, 'w', encoding = 'utf8') as rootintf:

            #process silence phones
            for i, sp in enumerate(self.sil_phones):
                if self.position_dependent_phones:
                    mapped = [sp+x for x in [''] + self.positions]
                else:
                    mapped = [sp]
                setf.write(' '.join(mapped) + '\n')
                setintf.write(' '.join(map(str, (self.phone_mapping[x] for x in mapped))) + '\n')
                if i == 0:
                    line = sil_sharesplit + mapped
                    lineint = sil_sharesplit + [self.phone_mapping[x] for x in mapped]
                else:
                    line = sharesplit + mapped
                    lineint = sharesplit + [self.phone_mapping[x] for x in mapped]
                rootf.write(' '.join(line) + '\n')
                rootintf.write(' '.join(map(str, lineint)) + '\n')

            #process nonsilence phones
            for nsp in sorted(self.nonsil_phones):
                if self.position_dependent_phones:
                    mapped = [nsp+x for x in  self.positions]
                else:
                    mapped = [nsp]
                setf.write(' '.join(mapped) + '\n')
                setintf.write(' '.join(map(str, (self.phone_mapping[x] for x in mapped))) + '\n')
                line = sharesplit + mapped
                lineint = sharesplit + [self.phone_mapping[x] for x in mapped]
                rootf.write(' '.join(line) + '\n')
                rootintf.write(' '.join(map(str, lineint)) + '\n')

    def _write_extra_questions(self):
        phone_extra = os.path.join(self.phones_dir, 'extra_questions.txt')
        phone_extra_int = os.path.join(self.phones_dir, 'extra_questions.int')
        with open(phone_extra, 'w', encoding = 'utf8') as outf, \
            open(phone_extra_int, 'w', encoding = 'utf8') as intf:
            if self.position_dependent_phones:
                sils = sorted(self.positional_sil_phones)
            else:
                sils = sorted(self.sil_phones)
            outf.write(' '.join(sils) + '\n')
            intf.write(' '.join(map(str, (self.phone_mapping[x] for x in sils))) + '\n')

            if self.position_dependent_phones:
                nonsils = sorted(self.positional_nonsil_phones)
            else:
                nonsils = sorted(self.nonsil_phones)
            outf.write(' '.join(nonsils) + '\n')
            intf.write(' '.join(map(str, (self.phone_mapping[x] for x in nonsils))) + '\n')

            for p in self.positions:
                line = [x + p for x in sorted(self.nonsil_phones)]
                outf.write(' '.join(line) + '\n')
                intf.write(' '.join(map(str, (self.phone_mapping[x] for x in line))) + '\n')
            for p in [''] + self.positions:
                line = [x + p for x in sorted(self.sil_phones)]
                outf.write(' '.join(line) + '\n')
                intf.write(' '.join(map(str, (self.phone_mapping[x] for x in line))) + '\n')

    def _write_fst_binary(self):

        lexicon_fst_path = os.path.join(self.output_directory, 'lexicon.text.fst')

        phones_file_path = os.path.join(self.output_directory, 'phones.txt')
        words_file_path = os.path.join(self.output_directory, 'words.txt')


        output_fst = os.path.join(self.output_directory, 'L.fst')
        temp_fst_path = os.path.join(self.output_directory, 'temp.fst')
        subprocess.call([thirdparty_binary('fstcompile'), '--isymbols={}'.format(phones_file_path),
                        '--osymbols={}'.format(words_file_path),
                        '--keep_isymbols=false','--keep_osymbols=false',
                        lexicon_fst_path, temp_fst_path])

        subprocess.call([thirdparty_binary('fstarcsort'), '--sort_type=olabel',
                    temp_fst_path, output_fst])

    def _write_fst_text(self):
        lexicon_fst_path = os.path.join(self.output_directory, 'lexicon.text.fst')
        if self.sil_prob != 0:
            silphone = self.optional_silence
            def is_sil(element):
                return element == silphone
            silcost = -1 * math.log(self.sil_prob);
            nosilcost = -1 * math.log(1.0 - self.sil_prob)
            startstate = 0
            loopstate = 1
            silstate = 2
        else:
            loopstate = 0
            nextstate = 1

        with open(lexicon_fst_path, 'w', encoding = 'utf8') as outf:
            if self.sil_prob > 0:

                outf.write('\t'.join(map(str,[startstate, loopstate, '<eps>', '<eps>', nosilcost])) + '\n')

                outf.write('\t'.join(map(str,[startstate, loopstate, silphone, '<eps>',silcost]))+"\n")
                outf.write('\t'.join(map(str,[silstate, loopstate, silphone, '<eps>']))+"\n")
                nextstate = 3
            for w in sorted(self.words.keys()):
                for phones in sorted(self.words[w]):
                    phones = [x for x in phones]
                    if self.position_dependent_phones:
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
                    if not self.pronunciation_probabilities:
                        pron_cost = 0
                    else:
                        p = 1.0
                        pron_cost = -1 * math.log(p)

                    pron_cost_string = ''
                    if pron_cost != 0:
                        pron_cost_string = '\t{}'.pron_cost

                    s = loopstate
                    word_or_eps = w
                    while len(phones) > 0:
                        p = phones.pop(0)
                        if len(phones) > 0:
                            ns = nextstate
                            nextstate += 1
                            outf.write('\t'.join(map(str,[s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                            word_or_eps = '<eps>'
                            pron_cost_string = ""
                            pron_cost = 0.0
                            s = ns
                        elif self.sil_prob == 0:
                            ns = loopstate
                            outf.write('\t'.join(map(str,[s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                            word_or_eps = '<eps>'
                            pron_cost_string = ""
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


class OrthographicDictionary(Dictionary):

    def __init__(self, input_dict, output_directory, oov_code = '<unk>',
                    position_dependent_phones = True, num_sil_states = 5,
                    num_nonsil_states = 3, shared_silence_phones = False,
                    pronunciation_probabilities = True,
                    sil_prob = 0.5):
        self.output_directory = os.path.join(output_directory, 'dictionary')
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.position_dependent_phones = position_dependent_phones
        self.pronunciation_probabilities = pronunciation_probabilities

        self.words = defaultdict(list)
        self.nonsil_phones = set()
        self.sil_phones = set(['sil', 'spn'])
        self.optional_silence = 'sil'
        self.disambig = set()
        self.graphemes = set()
        for w in input_dict:
            self.graphemes.update(w)
            pron = input_dict[w]
            self.words[w].append(pron)
            self.nonsil_phones.update(pron)
        self.word_pattern = compile_graphemes(self.graphemes)
        self.words['!SIL'].append(['sil'])
        self.words[self.oov_code].append(['spn'])
        self.phone_mapping = {}
        i = 0
        self.phone_mapping['<eps>'] = i
        if self.position_dependent_phones:
            for p in self.positional_sil_phones:
                i += 1
                self.phone_mapping[p] = i
            for p in self.positional_nonsil_phones:
                i += 1
                self.phone_mapping[p] = i
        else:
            for p in sorted(self.sil_phones):
                i += 1
                self.phone_mapping[p] = i
            for p in sorted(self.nonsil_phones):
                i += 1
                self.phone_mapping[p] = i
        for p in sorted(self.disambig):
            i += 1
            self.phone_mapping[p] = i

        self.words_mapping = {}
        i = 0
        self.words_mapping['<eps>'] = i
        for w in sorted(self.words.keys()):
            i += 1
            self.words_mapping[w] = i

        self.words_mapping['#0'] = i + 1
        self.words_mapping['<s>'] = i + 2
        self.words_mapping['</s>'] = i + 3

        self.oovs_found = set()
