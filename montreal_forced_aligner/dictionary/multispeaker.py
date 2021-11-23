"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Collection, Dict, Optional, Union

from ..abc import TemporaryDirectoryMixin
from ..models import DictionaryModel
from .base_dictionary import PronunciationDictionary
from .mixins import DictionaryMixin

if TYPE_CHECKING:

    from ..corpus.classes import Speaker


__all__ = [
    "MultispeakerDictionaryMixin",
]


topo_template = "<State> {cur_state} <PdfClass> {cur_state} <Transition> {cur_state} 0.75 <Transition> {next_state} 0.25 </State>"
topo_sil_template = "<State> {cur_state} <PdfClass> {cur_state} {transitions} </State>"
topo_transition_template = "<Transition> {} {}"


class MultispeakerDictionaryMixin(DictionaryMixin, TemporaryDirectoryMixin):
    """
    Mixin class containing information about a pronunciation dictionary with different dictionaries per speaker

    Parameters
    ----------
    dictionary_path : str
        Dictionary path

    Attributes
    ----------
    dictionary_model: DictionaryModel
        Dictionary model
    speaker_mapping: dict[str, str]
        Mapping of speaker names to dictionary names
    dictionary_mapping: dict[str, PronunciationDictionary]
        Mapping of dictionary names to PronunciationDictionary
    """

    def __init__(self, dictionary_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.dictionary_model = DictionaryModel(dictionary_path)
        self.speaker_mapping = {}
        self.dictionary_mapping = {}

    def dictionary_setup(self):
        """Setup the dictionary for processing"""
        for speaker, dictionary in self.dictionary_model.load_dictionary_paths().items():
            self.speaker_mapping[speaker] = dictionary.name
            if dictionary.name not in self.dictionary_mapping:
                self.dictionary_mapping[dictionary.name] = PronunciationDictionary(
                    dictionary_path=dictionary.path,
                    temporary_directory=self.dictionary_output_directory,
                    **self.dictionary_options,
                )
                self.non_silence_phones.update(
                    self.dictionary_mapping[dictionary.name].non_silence_phones
                )
        for dictionary in self.dictionary_mapping.values():
            dictionary.non_silence_phones = self.non_silence_phones

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return self.dictionary_model.name

    @property
    def disambiguation_symbols_txt_path(self):
        """Path to the file containing phone disambiguation symbols"""
        return os.path.join(self.phones_dir, "disambiguation_symbols.txt")

    @property
    def disambiguation_symbols_int_path(self):
        """Path to the file containing integer IDs for phone disambiguation symbols"""
        return os.path.join(self.phones_dir, "disambiguation_symbols.int")

    @property
    def phones_dir(self) -> str:
        """Directory for storing phone information"""
        return os.path.join(self.dictionary_output_directory, "phones")

    @property
    def topo_path(self) -> str:
        """Path to the dictionary's topology file"""
        return os.path.join(self.phones_dir, "topo")

    def calculate_oovs_found(self) -> None:
        """Sum the counts of oovs found in pronunciation dictionaries"""
        for dictionary in self.dictionary_mapping.values():
            self.oovs_found.update(dictionary.oovs_found)

    @property
    def default_dictionary(self) -> PronunciationDictionary:
        """Default PronunciationDictionary"""
        return self.get_dictionary("default")

    def get_dictionary_name(self, speaker: Union[str, Speaker]) -> str:
        """
        Get the dictionary name for a given speaker

        Parameters
        ----------
        speaker: Union[Speaker, str]
            Speaker to look up

        Returns
        -------
        str
            PronunciationDictionary name for the speaker
        """
        if not isinstance(speaker, str):
            speaker = speaker.name
        if speaker not in self.speaker_mapping:
            return self.speaker_mapping["default"]
        return self.speaker_mapping[speaker]

    def get_dictionary(self, speaker: Union[Speaker, str]) -> PronunciationDictionary:
        """
        Get a dictionary for a given speaker

        Parameters
        ----------
        speaker: Union[Speaker, str]
            Speaker to look up

        Returns
        -------
        :class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`
            PronunciationDictionary for the speaker
        """
        return self.dictionary_mapping[self.get_dictionary_name(speaker)]

    def write_lexicon_information(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write all child dictionaries to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag to use disambiguation symbols in the output
        """
        os.makedirs(self.phones_dir, exist_ok=True)
        for d in self.dictionary_mapping.values():
            d.generate_mappings()
            if d.max_disambiguation_symbol > self.max_disambiguation_symbol:
                self.max_disambiguation_symbol = d.max_disambiguation_symbol
        self._write_word_boundaries()
        self._write_phone_map_file()
        self._write_phone_sets()
        self._write_phone_symbol_table()
        self._write_disambig()
        self._write_topo()
        self._write_extra_questions()
        for d in self.dictionary_mapping.values():
            d.write(write_disambiguation, multispeaker_dictionary=self)

    def _write_word_boundaries(self) -> None:
        """
        Write the word boundaries file to the temporary directory
        """
        boundary_path = os.path.join(
            self.dictionary_output_directory, "phones", "word_boundary.txt"
        )
        boundary_int_path = os.path.join(
            self.dictionary_output_directory, "phones", "word_boundary.int"
        )
        with open(boundary_path, "w", encoding="utf8") as f, open(
            boundary_int_path, "w", encoding="utf8"
        ) as intf:
            if self.position_dependent_phones:
                for p in sorted(self.phone_mapping.keys(), key=lambda x: self.phone_mapping[x]):
                    if p == "<eps>" or p.startswith("#"):
                        continue
                    cat = "nonword"
                    if p.endswith("_B"):
                        cat = "begin"
                    elif p.endswith("_S"):
                        cat = "singleton"
                    elif p.endswith("_I"):
                        cat = "internal"
                    elif p.endswith("_E"):
                        cat = "end"
                    f.write(" ".join([p, cat]) + "\n")
                    intf.write(" ".join([str(self.phone_mapping[p]), cat]) + "\n")

    def _write_topo(self) -> None:
        """
        Write the topo file to the temporary directory
        """
        sil_transp = 1 / (self.num_silence_states - 1)
        initial_transition = [
            topo_transition_template.format(x, sil_transp)
            for x in range(self.num_silence_states - 1)
        ]
        middle_transition = [
            topo_transition_template.format(x, sil_transp)
            for x in range(1, self.num_silence_states)
        ]
        final_transition = [
            topo_transition_template.format(self.num_silence_states - 1, 0.75),
            topo_transition_template.format(self.num_silence_states, 0.25),
        ]
        with open(self.topo_path, "w") as f:
            f.write("<Topology>\n")
            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")
            phones = self.kaldi_non_silence_phones
            f.write(f"{' '.join(str(self.phone_mapping[x]) for x in phones)}\n")
            f.write("</ForPhones>\n")
            states = [
                topo_template.format(cur_state=x, next_state=x + 1)
                for x in range(self.num_non_silence_states)
            ]
            f.write("\n".join(states))
            f.write(f"\n<State> {self.num_non_silence_states} </State>\n")
            f.write("</TopologyEntry>\n")

            f.write("<TopologyEntry>\n")
            f.write("<ForPhones>\n")

            phones = self.kaldi_silence_phones
            f.write(f"{' '.join(str(self.phone_mapping[x]) for x in phones)}\n")
            f.write("</ForPhones>\n")
            states = []
            for i in range(self.num_silence_states):
                if i == 0:
                    transition = " ".join(initial_transition)
                elif i == self.num_silence_states - 1:
                    transition = " ".join(final_transition)
                else:
                    transition = " ".join(middle_transition)
                states.append(topo_sil_template.format(cur_state=i, transitions=transition))
            f.write("\n".join(states))
            f.write(f"\n<State> {self.num_silence_states} </State>\n")
            f.write("</TopologyEntry>\n")
            f.write("</Topology>\n")

    def _write_phone_sets(self) -> None:
        """
        Write phone symbol sets to the temporary directory
        """
        sharesplit = ["shared", "split"]
        if not self.shared_silence_phones:
            sil_sharesplit = ["not-shared", "not-split"]
        else:
            sil_sharesplit = sharesplit

        sets_file = os.path.join(self.dictionary_output_directory, "phones", "sets.txt")
        roots_file = os.path.join(self.dictionary_output_directory, "phones", "roots.txt")

        sets_int_file = os.path.join(self.dictionary_output_directory, "phones", "sets.int")
        roots_int_file = os.path.join(self.dictionary_output_directory, "phones", "roots.int")

        with open(sets_file, "w", encoding="utf8") as setf, open(
            roots_file, "w", encoding="utf8"
        ) as rootf, open(sets_int_file, "w", encoding="utf8") as setintf, open(
            roots_int_file, "w", encoding="utf8"
        ) as rootintf:

            # process silence phones
            for i, sp in enumerate(self.silence_phones):
                if self.position_dependent_phones:
                    mapped = [sp + x for x in [""] + self.positions]
                else:
                    mapped = [sp]
                setf.write(" ".join(mapped) + "\n")
                setintf.write(" ".join(map(str, (self.phone_mapping[x] for x in mapped))) + "\n")
                if i == 0:
                    line = sil_sharesplit + mapped
                    lineint = sil_sharesplit + [str(self.phone_mapping[x]) for x in mapped]
                else:
                    line = sharesplit + mapped
                    lineint = sharesplit + [str(self.phone_mapping[x]) for x in mapped]
                rootf.write(" ".join(line) + "\n")
                rootintf.write(" ".join(lineint) + "\n")

            # process nonsilence phones
            for nsp in sorted(self.non_silence_phones):
                if self.position_dependent_phones:
                    mapped = [nsp + x for x in self.positions]
                else:
                    mapped = [nsp]
                setf.write(" ".join(mapped) + "\n")
                setintf.write(" ".join(map(str, (self.phone_mapping[x] for x in mapped))) + "\n")
                line = sharesplit + mapped
                lineint = sharesplit + [str(self.phone_mapping[x]) for x in mapped]
                rootf.write(" ".join(line) + "\n")
                rootintf.write(" ".join(lineint) + "\n")

    def _write_extra_questions(self) -> None:
        """
        Write extra questions symbols to the temporary directory
        """
        phone_extra = os.path.join(self.phones_dir, "extra_questions.txt")
        phone_extra_int = os.path.join(self.phones_dir, "extra_questions.int")
        with open(phone_extra, "w", encoding="utf8") as outf, open(
            phone_extra_int, "w", encoding="utf8"
        ) as intf:
            silences = self.kaldi_silence_phones
            outf.write(" ".join(silences) + "\n")
            intf.write(" ".join(str(self.phone_mapping[x]) for x in silences) + "\n")

            non_silences = self.kaldi_non_silence_phones
            outf.write(" ".join(non_silences) + "\n")
            intf.write(" ".join(str(self.phone_mapping[x]) for x in non_silences) + "\n")
            if self.position_dependent_phones:
                for p in self.positions:
                    line = [x + p for x in sorted(self.non_silence_phones)]
                    outf.write(" ".join(line) + "\n")
                    intf.write(" ".join(str(self.phone_mapping[x]) for x in line) + "\n")
                for p in [""] + self.positions:
                    line = [x + p for x in sorted(self.silence_phones)]
                    outf.write(" ".join(line) + "\n")
                    intf.write(" ".join(str(self.phone_mapping[x]) for x in line) + "\n")

    def _write_disambig(self) -> None:
        """
        Write disambiguation symbols to the temporary directory
        """
        disambig = self.disambiguation_symbols_txt_path
        disambig_int = self.disambiguation_symbols_int_path
        with open(disambig, "w", encoding="utf8") as outf, open(
            disambig_int, "w", encoding="utf8"
        ) as intf:
            for d in sorted(self.disambiguation_symbols, key=lambda x: self.phone_mapping[x]):
                outf.write(f"{d}\n")
                intf.write(f"{self.phone_mapping[d]}\n")

    def _write_phone_map_file(self) -> None:
        """
        Write the phone map to the temporary directory
        """
        outfile = os.path.join(self.dictionary_output_directory, "phone_map.txt")
        with open(outfile, "w", encoding="utf8") as f:
            for sp in self.silence_phones:
                if self.position_dependent_phones:
                    new_phones = [sp + x for x in ["", ""] + self.positions]
                else:
                    new_phones = [sp]
                f.write(" ".join(new_phones) + "\n")
            for nsp in self.non_silence_phones:
                if self.position_dependent_phones:
                    new_phones = [nsp + x for x in [""] + self.positions]
                else:
                    new_phones = [nsp]
                f.write(" ".join(new_phones) + "\n")

    @property
    def phone_symbol_table_path(self):
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.dictionary_output_directory, "phones.txt")

    def _write_phone_symbol_table(self) -> None:
        """
        Write the phone mapping to the temporary directory
        """
        with open(self.phone_symbol_table_path, "w", encoding="utf8") as f:
            for p, i in sorted(self.phone_mapping.items(), key=lambda x: x[1]):
                f.write(f"{p} {i}\n")

    def set_lexicon_word_set(self, word_set: Collection[str]) -> None:
        """
        Limit output to a subset of overall words

        Parameters
        ----------
        word_set: Collection[str]
            Word set to limit generated files to
        """
        for d in self.dictionary_mapping.values():
            d.set_lexicon_word_set(word_set)

    @property
    def output_paths(self) -> Dict[str, str]:
        """
        Mapping of output directory for child directories
        """
        return {d.name: d.dictionary_output_directory for d in self.dictionary_mapping.values()}


class MultispeakerDictionary(MultispeakerDictionaryMixin):
    """
    Class for processing multi- and single-speaker pronunciation dictionaries
    """

    @property
    def data_source_identifier(self) -> str:
        """Name of the dictionary"""
        return f"{self.name}"

    @property
    def identifier(self) -> str:
        """Name of the dictionary"""
        return f"{self.data_source_identifier}"

    @property
    def output_directory(self) -> str:
        """Root temporary directory to store all dictionary information"""
        return os.path.join(self.temporary_directory, self.identifier)
