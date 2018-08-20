import os
import math
import subprocess
import re
from collections import defaultdict, Counter

from .helper import thirdparty_binary
from .exceptions import DictionaryPathError, DictionaryFileError, DictionaryError


def compile_graphemes(graphemes):
    if '-' in graphemes:
        base = r'^\W*([-{}]+)\W*'
    else:
        base = r'^\W*([{}]+)\W*'
    graphemes = list(graphemes)
    for i in range(len(graphemes)):
        if graphemes[i] == ']':
            graphemes[i] = r'\]'
    string = ''.join(x for x in graphemes if x != '-')
    try:
        return re.compile(base.format(string))
    except Exception:
        print(graphemes)
        raise


brackets = [('[', ']'), ('{', '}'), ('<', '>'), ('(', ')')]


def check_bracketed(word):
    for b in brackets:
        if word[0] == b[0] and word[-1] == b[-1]:
            return True
    return False


def sanitize(item):
    if not item:
        return item
    for b in brackets:
        if item[0] == b[0] and item[-1] == b[1]:
            return item
    # Clitic markers are "-" and "'"
    sanitized = re.sub(r"^[^-\w']+", '', item)
    sanitized = re.sub(r"[^-\w']+$", '', sanitized)
    return sanitized


def sanitize_clitics(item):
    if not item:
        return item
    for b in brackets:
        if item[0] == b[0] and item[-1] == b[1]:
            return item
    # Clitic markers are "-" and "'"
    sanitized = re.sub(r"^\W+", '', item)
    sanitized = re.sub(r"\W+$", '', sanitized)
    return sanitized


class Dictionary(object):
    """
    Class containing information about a pronunciation dictionary

    Parameters
    ----------
    input_path : str
        Path to an input pronunciation dictionary
    output_directory : str
        Path to a directory to store files for Kaldi
    oov_code : str, optional
        What to label words not in the dictionary, defaults to ``'<unk>'``
    position_dependent_phones : bool, optional
        Specifies whether phones should be represented as dependent on their
        position in the word (beginning, middle or end), defaults to True
    num_sil_states : int, optional
        Number of states to use for silence phones, defaults to 5
    num_nonsil_states : int, optional
        Number of states to use for non-silence phones, defaults to 3
    shared_silence_phones : bool, optional
        Specify whether to share states across all silence phones, defaults
        to True
    pronunciation probabilities : bool, optional
        Specifies whether to model different pronunciation probabilities
        or to treat each entry as a separate word, defaults to True
    sil_prob : float, optional
        Probability of optional silences following words, defaults to 0.5
    """

    topo_template = '<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>'
    topo_sil_template = '<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>'
    topo_transition_template = '<Transition> {} {}'
    positions = ["_B", "_E", "_I", "_S"]
    clitic_markers = ["'", '-']

    def __init__(self, input_path, output_directory, oov_code='<unk>',
                 position_dependent_phones=True, num_sil_states=5,
                 num_nonsil_states=3, shared_silence_phones=True,
                 sil_prob=0.5, word_set=None, debug=False):
        if not os.path.exists(input_path):
            raise (DictionaryPathError(input_path))
        if not os.path.isfile(input_path):
            raise (DictionaryFileError(input_path))
        self.input_path = input_path
        self.debug = debug
        self.output_directory = os.path.join(output_directory, 'dictionary')
        self.num_sil_states = num_sil_states
        self.num_nonsil_states = num_nonsil_states
        self.shared_silence_phones = shared_silence_phones
        self.sil_prob = sil_prob
        self.oov_code = oov_code
        self.position_dependent_phones = position_dependent_phones

        self.words = defaultdict(list)
        self.nonsil_phones = set()
        self.sil_phones = {'sp', 'spn', 'sil'}
        self.optional_silence = 'sp'
        self.nonoptional_silence = 'sil'
        self.graphemes = set()
        if word_set is not None:
            word_set = {sanitize(x) for x in word_set}
        self.words['!sil'].append((('sp',), 1))
        self.words[self.oov_code].append((('spn',), 1))
        self.pronunciation_probabilities = True
        with open(input_path, 'r', encoding='utf8') as inf:
            for i, line in enumerate(inf):
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                word = line.pop(0).lower()
                if not line:
                    raise DictionaryError('Line {} of {} does not have a pronunciation.'.format(i, input_path))
                if word in ['!sil', oov_code]:
                    continue
                if word_set is not None and sanitize(word) not in word_set:
                    continue
                self.graphemes.update(word)
                try:
                    prob = float(line[0])
                    line = line[1:]
                except ValueError:
                    prob = None
                    self.pronunciation_probabilities = False
                pron = tuple(line)
                if not any(x in self.sil_phones for x in pron):
                    self.nonsil_phones.update(pron)
                if word in self.words and pron in set(x[0] for x in self.words[word]):
                    continue
                self.words[word].append((pron, prob))
        if not self.graphemes:
            raise DictionaryFileError('No words were found in the dictionary path {}'.format(input_path))
        self.word_pattern = compile_graphemes(self.graphemes)
        self.phone_mapping = {}
        self.words_mapping = {}

    def generate_mappings(self):
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

        self.words_mapping = {}
        i = 0
        self.words_mapping['<eps>'] = i
        for w in sorted(self.words.keys()):
            i += 1
            self.words_mapping[w] = i

        self.words_mapping['#0'] = i + 1
        self.words_mapping['<s>'] = i + 2
        self.words_mapping['</s>'] = i + 3

        self.oovs_found = Counter()
        self.add_disambiguation()

    def add_disambiguation(self):
        subsequences = set()
        pronunciation_counts = defaultdict(int)

        for w, prons in self.words.items():
            for p in prons:
                pronunciation_counts[p[0]] += 1
                pron = [x for x in p[0]][:-1]
                while pron:
                    subsequences.add(tuple(p))
                    pron = pron[:-1]
        last_used = defaultdict(int)
        for w, prons in sorted(self.words.items()):
            new_prons = []
            for p in prons:
                if pronunciation_counts[p[0]] == 1 and not p[0] in subsequences:
                    disambig = None
                else:
                    pron = p[0]
                    last_used[pron] += 1
                    disambig = last_used[pron]
                new_prons.append((p[0], p[1], disambig))
            self.words[w] = new_prons
        if last_used:
            self.max_disambig = max(last_used.values())
        else:
            self.max_disambig = 0
        self.disambig = set('#{}'.format(x) for x in range(self.max_disambig + 1))
        i = max(self.phone_mapping.values())
        for p in sorted(self.disambig):
            i += 1
            self.phone_mapping[p] = i

    def create_utterance_fst(self, text, frequent_words):
        num_words = len(text)
        word_probs = Counter(text)
        word_probs = {k: v / num_words for k, v in word_probs.items()}
        word_probs.update(frequent_words)
        text = ''
        for k, v in word_probs.items():
            cost = -1 * math.log(v)
            text += '0 0 {w} {w} {cost}\n'.format(w=self.to_int(k), cost=cost)
        text += '0 {}\n'.format(-1 * math.log(1 / num_words))
        return text

    def to_int(self, item):
        """
        Convert a given word into its integer id
        """
        if item == '':
            return None
        item = self._lookup(item)
        if item not in self.words_mapping:
            self.oovs_found.update([item])
            return self.oov_int
        return self.words_mapping[item]

    def save_oovs_found(self, directory):
        """
        Save all out of vocabulary items to a file in the specified directory

        Parameters
        ----------
        directory : str
            Path to directory to save ``oovs_found.txt``
        """
        with open(os.path.join(directory, 'oovs_found.txt'), 'w', encoding='utf8') as f, \
                open(os.path.join(directory, 'oov_counts.txt'), 'w', encoding='utf8') as cf:
            for oov in sorted(self.oovs_found.keys(), key=lambda x: (-self.oovs_found[x], x)):
                f.write(oov + '\n')
                cf.write('{}\t{}\n'.format(oov, self.oovs_found[oov]))
        self.oovs_found = set()

    def _lookup(self, item):
        if item in self.words_mapping:
            return item
        sanitized = sanitize(item)
        if sanitized in self.words_mapping:
            return sanitized
        sanitized = sanitize_clitics(item)
        if sanitized in self.words_mapping:
            return sanitized
        return item

    def separate_clitics(self, item):
        """Separates words with apostrophes or hyphens if the subparts are in the lexicon.

        Checks whether the text on either side of an apostrophe or hyphen is in the dictionary. If so,
        splits the word. If neither part is in the dictionary, returns the word without splitting it.

        Parameters
        ----------
        item : string
            Lexical item

        Returns
        -------
        vocab_items: list
            List containing all words after any splits due to apostrophes or hyphens

        """
        unit_re = re.compile(r'^(\[.*\]|\{.*\}|<.*>)$')
        if unit_re.match(item) is not None:
            return [item]
        lookup = self._lookup(item)

        if lookup not in self.words_mapping:
            item = sanitize(item)
            vocab = []
            chars = list(item)
            count = 0
            for i in chars:
                if i in self.clitic_markers:
                    count += 1
            for i in range(count):
                for punc in chars:
                    if punc in self.clitic_markers:
                        idx = chars.index(punc)
                        option1withpunc = ''.join(chars[:idx + 1])
                        option1nopunc = ''.join(chars[:idx])
                        option2withpunc = ''.join(chars[idx:])
                        option2nopunc = ''.join(chars[idx + 1:])
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
            return [lookup]
        if not vocab:
            return [lookup]
        else:
            unk = []
            for i in vocab:
                if i not in self.words:
                    unk.append(i)
            if len(unk) == count + 1:
                return [lookup]
            return vocab

    @property
    def reversed_word_mapping(self):
        """
        A mapping of integer ids to words
        """
        mapping = {}
        for k, v in self.words_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def reversed_phone_mapping(self):
        """
        A mapping of integer ids to phones
        """
        mapping = {}
        for k, v in self.phone_mapping.items():
            mapping[v] = k
        return mapping

    @property
    def oov_int(self):
        """
        The integer id for out of vocabulary items
        """
        return self.words_mapping[self.oov_code]

    @property
    def positional_sil_phones(self):
        """
        List of silence phones with positions
        """
        sil_phones = []
        for p in sorted(self.sil_phones):
            sil_phones.append(p)
            for pos in self.positions:
                sil_phones.append(p + pos)
        return sil_phones

    @property
    def positional_nonsil_phones(self):
        """
        List of non-silence phones with positions
        """
        nonsil_phones = []
        for p in sorted(self.nonsil_phones):
            for pos in self.positions:
                nonsil_phones.append(p + pos)
        return nonsil_phones

    @property
    def optional_silence_csl(self):
        """
        Phone id of the optional silence phone
        """
        return '{}'.format(self.phone_mapping[self.optional_silence])

    @property
    def silence_csl(self):
        """
        A colon-separated list (as a string) of silence phone ids
        """
        if self.position_dependent_phones:
            return ':'.join(map(str, (self.phone_mapping[x] for x in self.positional_sil_phones)))
        else:
            return ':'.join(map(str, (self.phone_mapping[x] for x in self.sil_phones)))

    @property
    def phones_dir(self):
        """
        Directory to store information Kaldi needs about phones
        """
        return os.path.join(self.output_directory, 'phones')

    @property
    def phones(self):
        """
        The set of all phones (silence and non-silence)
        """
        return self.sil_phones | self.nonsil_phones

    def write(self):
        """
        Write the files necessary for Kaldi
        """
        print('Creating dictionary information...')
        os.makedirs(self.phones_dir, exist_ok=True)
        self.generate_mappings()
        self._write_graphemes()
        self._write_phone_map_file()
        self._write_phone_sets()
        self._write_phone_symbol_table()
        self._write_disambig()
        self._write_topo()
        self._write_word_boundaries()
        self._write_extra_questions()
        self._write_word_file()
        self._write_fst_text()
        self._write_fst_text(disambig=True)
        self._write_fst_binary()
        self._write_fst_binary(disambig=True)
        # self.cleanup()

    def cleanup(self):
        """
        Clean up temporary files in the output directory
        """
        os.remove(os.path.join(self.output_directory, 'temp.fst'))
        os.remove(os.path.join(self.output_directory, 'lexicon.text.fst'))

    def _write_graphemes(self):
        outfile = os.path.join(self.output_directory, 'graphemes.txt')
        with open(outfile, 'w', encoding='utf8') as f:
            for char in sorted(self.graphemes):
                f.write(char + '\n')

    def export_lexicon(self, path, disambig=False, probability=False):
        with open(path, 'w', encoding='utf8') as f:
            for w in sorted(self.words.keys()):
                for p in sorted(self.words[w]):
                    phones = ' '.join(p[0])
                    if disambig and p[2] is not None:
                        phones += ' #{}'.format(p[2])
                    if probability:
                        f.write('{}\t{}\t{}\n'.format(w, p[1], phones))
                    else:
                        f.write('{}\t{}\n'.format(w, phones))

    def _write_phone_map_file(self):
        outfile = os.path.join(self.output_directory, 'phone_map.txt')
        with open(outfile, 'w', encoding='utf8') as f:
            for sp in self.sil_phones:
                if self.position_dependent_phones:
                    new_phones = [sp + x for x in ['', ''] + self.positions]
                else:
                    new_phones = [sp]
                f.write(' '.join(new_phones) + '\n')
            for nsp in self.nonsil_phones:
                if self.position_dependent_phones:
                    new_phones = [nsp + x for x in [''] + self.positions]
                else:
                    new_phones = [nsp]
                f.write(' '.join(new_phones) + '\n')

    def _write_phone_symbol_table(self):
        outfile = os.path.join(self.output_directory, 'phones.txt')
        with open(outfile, 'w', encoding='utf8') as f:
            for p, i in sorted(self.phone_mapping.items(), key=lambda x: x[1]):
                f.write('{} {}\n'.format(p, i))

    def _write_word_boundaries(self):
        boundary_path = os.path.join(self.output_directory, 'phones', 'word_boundary.txt')
        boundary_int_path = os.path.join(self.output_directory, 'phones', 'word_boundary.int')
        with open(boundary_path, 'w', encoding='utf8') as f, \
                open(boundary_int_path, 'w', encoding='utf8') as intf:
            if self.position_dependent_phones:
                for p in sorted(self.phone_mapping.keys(), key=lambda x: self.phone_mapping[x]):
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
                    f.write(' '.join([p, cat]) + '\n')
                    intf.write(' '.join([str(self.phone_mapping[p]), cat]) + '\n')

    def _write_word_file(self):
        words_path = os.path.join(self.output_directory, 'words.txt')

        with open(words_path, 'w', encoding='utf8') as f:
            for w, i in sorted(self.words_mapping.items(), key=lambda x: x[1]):
                f.write('{} {}\n'.format(w, i))

    def _write_topo(self):
        filepath = os.path.join(self.output_directory, 'topo')
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
            states = [self.topo_template.format(cur_state=x, next_state=x + 1)
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
                states.append(self.topo_sil_template.format(cur_state=i, transitions=transition))
            f.write('\n'.join(states))
            f.write("\n<State> {} </State>\n".format(self.num_sil_states))
            f.write("</TopologyEntry>\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self):
        sharesplit = ['shared', 'split']
        if not self.shared_silence_phones:
            sil_sharesplit = ['not-shared', 'not-split']
        else:
            sil_sharesplit = sharesplit

        sets_file = os.path.join(self.output_directory, 'phones', 'sets.txt')
        roots_file = os.path.join(self.output_directory, 'phones', 'roots.txt')

        sets_int_file = os.path.join(self.output_directory, 'phones', 'sets.int')
        roots_int_file = os.path.join(self.output_directory, 'phones', 'roots.int')

        with open(sets_file, 'w', encoding='utf8') as setf, \
                open(roots_file, 'w', encoding='utf8') as rootf, \
                open(sets_int_file, 'w', encoding='utf8') as setintf, \
                open(roots_int_file, 'w', encoding='utf8') as rootintf:

            # process silence phones
            for i, sp in enumerate(self.sil_phones):
                if self.position_dependent_phones:
                    mapped = [sp + x for x in [''] + self.positions]
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

            # process nonsilence phones
            for nsp in sorted(self.nonsil_phones):
                if self.position_dependent_phones:
                    mapped = [nsp + x for x in self.positions]
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
        with open(phone_extra, 'w', encoding='utf8') as outf, \
                open(phone_extra_int, 'w', encoding='utf8') as intf:
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
            if self.position_dependent_phones:
                for p in self.positions:
                    line = [x + p for x in sorted(self.nonsil_phones)]
                    outf.write(' '.join(line) + '\n')
                    intf.write(' '.join(map(str, (self.phone_mapping[x] for x in line))) + '\n')
                for p in [''] + self.positions:
                    line = [x + p for x in sorted(self.sil_phones)]
                    outf.write(' '.join(line) + '\n')
                    intf.write(' '.join(map(str, (self.phone_mapping[x] for x in line))) + '\n')

    def _write_disambig(self):
        disambig = os.path.join(self.phones_dir, 'disambig.txt')
        disambig_int = os.path.join(self.phones_dir, 'disambig.int')
        with open(disambig, 'w', encoding='utf8') as outf, \
                open(disambig_int, 'w', encoding='utf8') as intf:
            for d in sorted(self.disambig):
                outf.write('{}\n'.format(d))
                intf.write('{}\n'.format(self.phone_mapping[d]))

    def _write_fst_binary(self, disambig=False):
        if disambig:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon_disambig.text.fst')
            output_fst = os.path.join(self.output_directory, 'L_disambig.fst')
        else:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon.text.fst')
            output_fst = os.path.join(self.output_directory, 'L.fst')

        phones_file_path = os.path.join(self.output_directory, 'phones.txt')
        words_file_path = os.path.join(self.output_directory, 'words.txt')

        log_path = os.path.join(self.output_directory, 'fst.log')
        temp_fst_path = os.path.join(self.output_directory, 'temp.fst')
        subprocess.call([thirdparty_binary('fstcompile'), '--isymbols={}'.format(phones_file_path),
                         '--osymbols={}'.format(words_file_path),
                         '--keep_isymbols=false', '--keep_osymbols=false',
                         lexicon_fst_path, temp_fst_path])

        subprocess.call([thirdparty_binary('fstarcsort'), '--sort_type=olabel',
                         temp_fst_path, output_fst])
        if self.debug:
            dot_path = os.path.join(self.output_directory, 'L.dot')
            with open(log_path, 'w') as logf:
                draw_proc = subprocess.Popen([thirdparty_binary('fstdraw'), '--portrait=true',
                                              '--isymbols={}'.format(phones_file_path),
                                              '--osymbols={}'.format(words_file_path), output_fst, dot_path],
                                             stderr=logf)
                draw_proc.communicate()
                dot_proc = subprocess.Popen([thirdparty_binary('dot'), '-Tpdf', '-O', dot_path], stderr=logf)
                dot_proc.communicate()

    def _write_fst_text(self, disambig=False):
        if disambig:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon_disambig.text.fst')
        else:
            lexicon_fst_path = os.path.join(self.output_directory, 'lexicon.text.fst')
        if self.sil_prob != 0:
            silphone = self.optional_silence
            nonoptsil = self.nonoptional_silence

            def is_sil(element):
                return element in [silphone, silphone + '_S']

            silcost = -1 * math.log(self.sil_prob)
            nosilcost = -1 * math.log(1.0 - self.sil_prob)
            startstate = 0
            loopstate = 1
            silstate = 2
        else:
            loopstate = 0
            nextstate = 1

        with open(lexicon_fst_path, 'w', encoding='utf8') as outf:
            if self.sil_prob != 0:
                outf.write('\t'.join(map(str, [startstate, loopstate, '<eps>', '<eps>', nosilcost])) + '\n')

                outf.write('\t'.join(map(str, [startstate, loopstate, nonoptsil, '<eps>', silcost])) + "\n")
                outf.write('\t'.join(map(str, [silstate, loopstate, silphone, '<eps>'])) + "\n")
                nextstate = 3
            for w in sorted(self.words.keys()):
                for phones, prob, disambig_symbol in sorted(self.words[w]):
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
                        if prob is None:
                            prob = 1.0
                        pron_cost = -1 * math.log(prob)

                    pron_cost_string = ''
                    if pron_cost != 0:
                        pron_cost_string = '\t{}'.format(pron_cost)

                    s = loopstate
                    word_or_eps = w
                    local_nosilcost = nosilcost + pron_cost
                    local_silcost = silcost + pron_cost
                    while len(phones) > 0:
                        p = phones.pop(0)
                        if len(phones) > 0 or (disambig and disambig_symbol is not None):
                            ns = nextstate
                            nextstate += 1
                            outf.write('\t'.join(map(str, [s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                            word_or_eps = '<eps>'
                            pron_cost_string = ""
                            pron_cost = 0.0
                            s = ns
                        elif self.sil_prob == 0:
                            ns = loopstate
                            outf.write('\t'.join(map(str, [s, ns, p, word_or_eps])) + pron_cost_string + '\n')
                            word_or_eps = '<eps>'
                            pron_cost_string = ""
                            s = ns
                        else:
                            outf.write('\t'.join(map(str, [s, loopstate, p, word_or_eps, local_nosilcost])) + "\n")
                            outf.write('\t'.join(map(str, [s, silstate, p, word_or_eps, local_silcost])) + "\n")
                    if disambig and disambig_symbol is not None:
                        outf.write('\t'.join(map(str, [s, loopstate, '#{}'.format(disambig_symbol), word_or_eps,
                                                       local_nosilcost])) + "\n")
                        outf.write('\t'.join(
                            map(str, [s, silstate, '#{}'.format(disambig_symbol), word_or_eps, local_silcost])) + "\n")

            outf.write("{}\t{}\n".format(loopstate, 0))


class OrthographicDictionary(Dictionary):
    def __init__(self, input_dict, output_directory, oov_code='<unk>',
                 position_dependent_phones=True, num_sil_states=5,
                 num_nonsil_states=3, shared_silence_phones=False,
                 pronunciation_probabilities=True,
                 sil_prob=0.5, debug=False):
        self.debug = debug
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
        self.sil_phones = {'sp', 'spn', 'sil'}
        self.optional_silence = 'sp'
        self.nonoptional_silence = 'sil'
        self.graphemes = set()
        for w in input_dict:
            self.graphemes.update(w)
            pron = tuple(input_dict[w])
            self.words[w].append((pron, None))
            self.nonsil_phones.update(pron)
        self.word_pattern = compile_graphemes(self.graphemes)
        self.words['!SIL'].append((('sil',), None))
        self.words[self.oov_code].append((('spn',), None))
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

        self.words_mapping = {}
        i = 0
        self.words_mapping['<eps>'] = i
        for w in sorted(self.words.keys()):
            i += 1
            self.words_mapping[w] = i

        self.words_mapping['#0'] = i + 1
        self.words_mapping['<s>'] = i + 2
        self.words_mapping['</s>'] = i + 3

        self.oovs_found = Counter()
        self.add_disambiguation()
