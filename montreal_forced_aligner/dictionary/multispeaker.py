"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import abc
import collections
import logging
import os
import re
import typing
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import pynini
import pywrapfst
import sqlalchemy.orm.session
import yaml
from kalpy.fstext.lexicon import G2PCompiler, LexiconCompiler
from kalpy.fstext.lexicon import Pronunciation as KalpyPronunciation
from sqlalchemy.orm import selectinload
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.data import PhoneSetType, PhoneType, WordType
from montreal_forced_aligner.db import (
    Corpus,
    Dialect,
    DictBundle,
    Dictionary,
    Grapheme,
    Phone,
    PhonologicalRule,
    Pronunciation,
    RuleApplication,
    Speaker,
    Utterance,
    Word,
    bulk_update,
)
from montreal_forced_aligner.dictionary.mixins import TemporaryDictionaryMixin
from montreal_forced_aligner.exceptions import (
    DictionaryError,
    DictionaryFileError,
    PhoneGroupTopologyMismatchError,
)
from montreal_forced_aligner.helper import comma_join, mfa_open, split_phone_position
from montreal_forced_aligner.models import DictionaryModel
from montreal_forced_aligner.utils import parse_dictionary_file

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.models import AcousticModel

__all__ = [
    "MultispeakerDictionaryMixin",
    "MultispeakerDictionary",
]

logger = logging.getLogger("mfa")


class MultispeakerDictionaryMixin(TemporaryDictionaryMixin, metaclass=abc.ABCMeta):
    """
    Mixin class containing information about a pronunciation dictionary with different dictionaries per speaker

    Parameters
    ----------
    dictionary_path : str
        Dictionary path
    kwargs : kwargs
        Extra parameters to passed to parent classes (see below)

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters


    Attributes
    ----------
    dictionary_model: :class:`~montreal_forced_aligner.models.DictionaryModel`
        Dictionary model
    dictionary_lookup: dict[str, int]
        Mapping of dictionary names to ids
    """

    def __init__(
        self,
        dictionary_path: typing.Union[str, Path] = None,
        rules_path: typing.Union[str, Path] = None,
        phone_groups_path: typing.Union[str, Path] = None,
        topology_path: typing.Union[str, Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dictionary_model = None
        if dictionary_path is not None:
            self.dictionary_model = DictionaryModel(
                dictionary_path, phone_set_type=self.phone_set_type
            )
        self._num_dictionaries = None
        self.dictionary_lookup = {}
        self._phone_mapping = None
        self._phone_table = None
        self._grapheme_mapping = None
        self._words_mappings = {}
        self._speaker_mapping = {}
        self._default_dictionary_id = None
        self._dictionary_base_names = None
        self.clitic_marker = None
        self.use_g2p = False
        if isinstance(rules_path, str):
            rules_path = Path(rules_path)
        if isinstance(phone_groups_path, str):
            phone_groups_path = Path(phone_groups_path)
        if isinstance(topology_path, str):
            topology_path = Path(topology_path)
        self.rules_path = rules_path
        self.phone_groups_path = phone_groups_path
        self.topology_path = topology_path
        self.lexicon_compilers: typing.Dict[int, typing.Union[LexiconCompiler, G2PCompiler]] = {}
        self._tokenizers = {}

    @property
    def tokenizers(self):
        from montreal_forced_aligner.tokenization.simple import SimpleTokenizer

        if not self._tokenizers:
            with self.session() as session:
                grapheme_set = set()
                grapheme_query = session.query(Grapheme.grapheme)
                for (g,) in grapheme_query:
                    grapheme_set.add(g)
                dictionaries = session.query(Dictionary)
                for d in dictionaries:
                    clitic_set = set(
                        x[0]
                        for x in session.query(Word.word)
                        .filter(Word.word_type == WordType.clitic)
                        .filter(Word.dictionary_id == d.id)
                    )
                    self._tokenizers[d.id] = SimpleTokenizer(
                        word_table=d.word_table,
                        word_break_markers=self.word_break_markers,
                        punctuation=self.punctuation,
                        clitic_markers=self.clitic_markers,
                        compound_markers=self.compound_markers,
                        brackets=self.brackets,
                        laughter_word=self.laughter_word,
                        oov_word=self.oov_word,
                        bracketed_word=self.bracketed_word,
                        cutoff_word=self.cutoff_word,
                        ignore_case=self.ignore_case,
                        use_g2p=self.use_g2p or getattr(self, "g2p_model", None) is not None,
                        clitic_set=clitic_set,
                        grapheme_set=grapheme_set,
                    )
        return self._tokenizers

    def load_phone_groups(self) -> None:
        """
        Load phone groups from the dictionary's groups file path
        """
        if self.phone_groups_path is not None and self.phone_groups_path.exists():
            with mfa_open(self.phone_groups_path) as f:
                self._phone_groups = yaml.load(f, Loader=yaml.Loader)
                if isinstance(self._phone_groups, list):
                    self._phone_groups = {k: v for k, v in enumerate(self._phone_groups)}
                for k, v in self._phone_groups.items():
                    if not v:
                        continue
                    self._phone_groups[k] = sorted(
                        set(x for x in v if x in self.non_silence_phones)
                    )
            errors = []
            for phone_group in self._phone_groups.values():
                topos = set()
                for phone in phone_group:
                    min_states = 1
                    max_states = self.num_non_silence_states
                    if phone in self._topologies:
                        min_states = self._topologies[phone].get("min_states", 1)
                        max_states = self._topologies[phone].get(
                            "max_states", self.num_non_silence_states
                        )
                    topos.add((min_states, max_states))
                if len(topos) > 1:
                    errors.append((phone_group, topos))
            if errors:
                raise PhoneGroupTopologyMismatchError(
                    errors, self.phone_groups_path, self.topology_path
                )

            found_phones = set()
            for phones in self._phone_groups.values():
                found_phones.update(phones)
            missing_phones = self.non_silence_phones - found_phones
            if missing_phones:
                logger.warning(
                    f"The following phones were missing from the phone group: "
                    f"{comma_join(sorted(missing_phones))}"
                )
            else:
                logger.debug("All phones were included in phone groups")

    def load_phone_topologies(self) -> None:
        """
        Load phone topologies from the dictionary's groups file path
        """
        if self.topology_path is not None and self.topology_path.exists():
            with mfa_open(self.topology_path) as f:
                self._topologies = {
                    k: v
                    for k, v in yaml.load(f, Loader=yaml.Loader).items()
                    if k in self.non_silence_phones
                }
            found_phones = set(self._topologies.keys())
            missing_phones = self.non_silence_phones - found_phones
            if missing_phones:
                logger.debug(
                    f"The following phones will use the default topology (min states = 1, max states = {self.num_non_silence_states}): "
                    f"{comma_join(sorted(missing_phones))}"
                )
            else:
                logger.debug("All phones were included in topology config")
            logger.debug("The following phones will use custom topologies: ")
            for k, v in self._topologies.items():
                min_states = self._topologies[k].get("min_states", 1)
                max_states = self._topologies[k].get("max_states", self.num_non_silence_states)
                logger.debug(f"{k}: min states = {min_states}, max states = {max_states}")
                assert min_states <= max_states

    @property
    def speaker_mapping(self) -> typing.Dict[str, int]:
        """Mapping of speakers to dictionaries"""
        if not self._speaker_mapping:
            with self.session() as session:
                self._speaker_mapping = {
                    x[0]: x[1] for x in session.query(Speaker.name, Speaker.dictionary_id)
                }
        return self._speaker_mapping

    def get_dict_id_for_speaker(self, speaker_name: str) -> int:
        """
        Get the dictionary id of the speaker

        Parameters
        ----------
        speaker_name: str
            Speaker to look up

        Returns
        -------
        int
            Dictionary id
        """
        if speaker_name not in self.speaker_mapping:
            return self._default_dictionary_id
        return self.speaker_mapping[speaker_name]

    @property
    def dictionary_base_names(self) -> Dict[int, str]:
        """Mapping of base file names for pronunciation dictionaries"""
        if self._dictionary_base_names is None:
            with self.session() as session:
                dictionaries = session.query(Dictionary.id, Dictionary.name).all()
                self._dictionary_base_names = {}
                for d_id, d_name in dictionaries:
                    base_name = d_name
                    if any(d_name == x[1] and d_id != x[0] for x in dictionaries):
                        base_name += f"_{d_id}"
                    self._dictionary_base_names[d_id] = base_name
        return self._dictionary_base_names

    def word_mapping(self, dictionary_id: int = None) -> Dict[str, int]:
        """
        Get the word mapping for a specified dictionary id

        Parameters
        ----------
        dictionary_id: int, optional
            Database ID for dictionary, defaults to 1

        Returns
        -------
        dict[str, int]
            Mapping from words to their integer IDs for Kaldi processing
        """
        if dictionary_id is None:
            dictionary_id = self._default_dictionary_id
        if dictionary_id not in self._words_mappings:
            with self.session() as session:
                d = session.get(Dictionary, dictionary_id)
                self._words_mappings[dictionary_id] = d.word_mapping
        return self._words_mappings[dictionary_id]

    def reversed_word_mapping(self, dictionary_id: int = 1) -> Dict[int, str]:
        """
        Get the reversed word mapping for a specified dictionary id

        Parameters
        ----------
        dictionary_id: int, optional
            Database ID for dictionary, defaults to 1

        Returns
        -------
        dict[int, str]
            Mapping from integer IDs to words for Kaldi processing
        """
        mapping = {}
        for k, v in self.word_mapping(dictionary_id).items():
            mapping[v] = k
        return mapping

    @property
    def num_dictionaries(self) -> int:
        """Number of pronunciation dictionaries"""
        return len(self.dictionary_lookup)

    def load_alignment_lexicon_compilers(self):
        with self.session() as session:
            for d in session.query(Dictionary):
                self.lexicon_compilers[d.id] = LexiconCompiler(
                    disambiguation=False,
                    silence_probability=self.silence_probability,
                    initial_silence_probability=self.initial_silence_probability,
                    final_silence_correction=self.final_silence_correction,
                    final_non_silence_correction=self.final_non_silence_correction,
                    silence_word=self.silence_word,
                    oov_word=self.oov_word,
                    silence_phone=self.optional_silence_phone,
                    oov_phone=self.oov_phone,
                    position_dependent_phones=self.position_dependent_phones,
                    ignore_case=self.ignore_case,
                    phones=self.non_silence_phones,
                )
                if d.lexicon_fst_path.exists():
                    self.lexicon_compilers[d.id].load_l_from_file(d.lexicon_fst_path)
                elif d.lexicon_disambig_fst_path.exists():
                    self.lexicon_compilers[d.id].load_l_from_file(d.lexicon_disambig_fst_path)
                if d.align_lexicon_path.exists():
                    self.lexicon_compilers[d.id].load_l_align_from_file(d.align_lexicon_path)
                elif d.align_lexicon_disambig_path.exists():
                    self.lexicon_compilers[d.id].load_l_align_from_file(
                        d.align_lexicon_disambig_path
                    )
                self.lexicon_compilers[d.id].word_table = d.word_table
                self.lexicon_compilers[d.id].phone_table = d.phone_table

    def dictionary_setup(self) -> Tuple[typing.Set[str], collections.Counter]:
        """Set up the dictionary for processing"""
        self.initialize_database()
        if self.use_g2p:
            return set(), collections.Counter()
        if getattr(self, "text_normalized", False):
            logger.debug(
                "Skipping setup from original dictionary file due to text already normalized."
            )
            return
        auto_set = {PhoneSetType.AUTO, PhoneSetType.UNKNOWN, "AUTO", "UNKNOWN"}
        if not isinstance(self.phone_set_type, PhoneSetType):
            self.phone_set_type = PhoneSetType[self.phone_set_type]

        os.makedirs(self.dictionary_output_directory, exist_ok=True)
        pretrained = False
        if self.non_silence_phones:
            pretrained = True

        self._speaker_ids = getattr(self, "_speaker_ids", {})
        self._current_speaker_index = getattr(self, "_current_speaker_index", 1)
        dictionary_id_cache = {}
        dialect_id_cache = {}
        with self.session() as session:
            self.non_silence_phones.update(
                x
                for x, in session.query(Phone.phone).filter(
                    Phone.phone_type == PhoneType.non_silence
                )
            )
            for speaker_id, speaker_name in session.query(Speaker.id, Speaker.name):
                self._speaker_ids[speaker_name] = speaker_id
                if speaker_id >= self._current_speaker_index:
                    self._current_speaker_index = speaker_id + 1
            for (
                dictionary_id,
                dict_name,
                default,
                max_disambiguation_symbol,
                path,
            ) in session.query(
                Dictionary.id,
                Dictionary.name,
                Dictionary.default,
                Dictionary.max_disambiguation_symbol,
                Dictionary.path,
            ):
                dictionary_id_cache[path] = dictionary_id
                self.dictionary_lookup[dict_name] = dictionary_id
                self.max_disambiguation_symbol = max(
                    self.max_disambiguation_symbol, max_disambiguation_symbol
                )
                if default:
                    self._default_dictionary_id = dictionary_id

            word_primary_key = 1
            pronunciation_primary_key = 1
            word_objs = []
            pron_objs = []
            speaker_objs = []
            phone_counts = collections.Counter()
            graphemes = set(self.clitic_markers + self.compound_markers)
            clitic_cleanup_regex = None
            has_nonnative_speakers = False
            self._words_mappings = {}
            current_mapping_id = 0
            if len(self.clitic_markers) >= 1:
                other_clitic_markers = self.clitic_markers[1:]
                if other_clitic_markers:
                    extra = ""
                    if "-" in other_clitic_markers:
                        extra = "-"
                        other_clitic_markers = [x for x in other_clitic_markers if x != "-"]
                    clitic_cleanup_regex = re.compile(rf'[{extra}{"".join(other_clitic_markers)}]')
                self.clitic_marker = self.clitic_markers[0]
            for (
                dictionary_model,
                speakers,
            ) in self.dictionary_model.load_dictionary_paths().values():
                if dictionary_model == "nonnative":
                    if not has_nonnative_speakers:
                        dialect = "nonnative"
                        if dialect not in dialect_id_cache:
                            dialect_obj = Dialect(name=dialect)
                            session.add(dialect_obj)
                            session.flush()
                            dialect_id_cache[dialect] = dialect_obj.id
                        dictionary = Dictionary(
                            name=dialect,
                            dialect_id=dialect_id_cache[dialect],
                            path="",
                            phone_set_type=self.phone_set_type,
                            root_temp_directory=self.dictionary_output_directory,
                            position_dependent_phones=self.position_dependent_phones,
                            clitic_marker=self.clitic_marker
                            if self.clitic_marker is not None
                            else "",
                            default=False,
                            use_g2p=False,
                            max_disambiguation_symbol=0,
                            silence_word=self.silence_word,
                            oov_word=self.oov_word,
                            bracketed_word=self.bracketed_word,
                            cutoff_word=self.cutoff_word,
                            laughter_word=self.laughter_word,
                            optional_silence_phone=self.optional_silence_phone,
                            oov_phone=self.oov_phone,
                        )
                        session.add(dictionary)
                        session.flush()
                        dictionary_id_cache[dialect] = dictionary.id
                    for speaker in speakers:
                        if speaker not in self._speaker_ids:
                            speaker_objs.append(
                                {
                                    "id": self._current_speaker_index,
                                    "name": speaker,
                                    "dictionary_id": dictionary_id_cache[dialect],
                                }
                            )
                            self._speaker_ids[speaker] = self._current_speaker_index
                            self._current_speaker_index += 1
                    has_nonnative_speakers = True
                    continue
                if dictionary_model.path not in dictionary_id_cache and not self.use_g2p:
                    word_cache = {}
                    pronunciation_cache = set()
                    if self.phone_set_type not in auto_set:
                        if (
                            self.phone_set_type != dictionary_model.phone_set_type
                            and dictionary_model.phone_set_type not in auto_set
                        ):
                            raise DictionaryError(
                                f"Mismatch found in phone sets: {self.phone_set_type} vs {dictionary_model.phone_set_type}"
                            )
                    else:
                        self.phone_set_type = dictionary_model.phone_set_type
                    dialect = None
                    if "_mfa" in dictionary_model.name:
                        name_parts = dictionary_model.name.split("_")
                        dialect = "_".join(name_parts[1:-1])
                    if not dialect:
                        dialect = "us"
                    if dialect not in dialect_id_cache:
                        dialect_obj = Dialect(name=dialect)
                        session.add(dialect_obj)
                        session.flush()
                        dialect_id_cache[dialect] = dialect_obj.id
                    dialect_id = dialect_id_cache[dialect]
                    dictionary = Dictionary(
                        name=dictionary_model.name,
                        dialect_id=dialect_id,
                        path=dictionary_model.path,
                        phone_set_type=self.phone_set_type,
                        root_temp_directory=self.dictionary_output_directory,
                        position_dependent_phones=self.position_dependent_phones,
                        clitic_marker=self.clitic_marker if self.clitic_marker is not None else "",
                        default="default" in speakers,
                        use_g2p=False,
                        max_disambiguation_symbol=0,
                        silence_word=self.silence_word,
                        oov_word=self.oov_word,
                        bracketed_word=self.bracketed_word,
                        cutoff_word=self.cutoff_word,
                        laughter_word=self.laughter_word,
                        optional_silence_phone=self.optional_silence_phone,
                        oov_phone=self.oov_phone,
                    )
                    session.add(dictionary)
                    session.flush()
                    dictionary_id_cache[dictionary_model.path] = dictionary.id
                    if dictionary.default:
                        self._default_dictionary_id = dictionary.id
                    if self.silence_word not in self._words_mappings:
                        word_objs.append(
                            {
                                "id": word_primary_key,
                                "mapping_id": current_mapping_id,
                                "word": self.silence_word,
                                "word_type": WordType.silence,
                                "dictionary_id": dictionary.id,
                            }
                        )
                        self._words_mappings[self.silence_word] = current_mapping_id
                        current_mapping_id += 1
                        word_primary_key += 1

                    pron_objs.append(
                        {
                            "id": pronunciation_primary_key,
                            "pronunciation": self.optional_silence_phone,
                            "probability": 1.0,
                            "disambiguation": None,
                            "silence_after_probability": None,
                            "silence_before_correction": None,
                            "non_silence_before_correction": None,
                            "word_id": word_primary_key,
                        }
                    )
                    pronunciation_primary_key += 1

                    special_words = {
                        self.oov_word: WordType.oov,
                        self.bracketed_word: WordType.bracketed,
                        self.cutoff_word: WordType.cutoff,
                        self.laughter_word: WordType.laughter,
                    }
                    specials_found = set()
                    if not os.path.exists(dictionary_model.path):
                        raise DictionaryFileError(dictionary_model.path)
                    for (
                        word,
                        pron,
                        prob,
                        silence_after_prob,
                        silence_before_correct,
                        non_silence_before_correct,
                    ) in parse_dictionary_file(dictionary_model.path):
                        if self.ignore_case:
                            word = word.lower()
                        if " " in word:
                            logger.debug(f'Skipping "{word}" for containing whitespace.')
                            continue
                        if clitic_cleanup_regex is not None:
                            word = clitic_cleanup_regex.sub(self.clitic_marker, word)
                        if word in self.specials_set:
                            continue
                        if word not in special_words:
                            if getattr(self, "unicode_decomposition", False):
                                characters = unicodedata.normalize("NFKD", word)
                            else:
                                characters = unicodedata.normalize("NFKC", word)
                            graphemes.update(characters)
                        if pretrained:
                            difference = set(pron) - self.non_silence_phones - self.silence_phones
                            if difference:
                                self.excluded_phones.update(difference)
                                self.excluded_pronunciation_count += 1
                                difference = " ".join(sorted(difference))
                                logger.debug(
                                    f'Skipping "{word}" for containing a phone not in the acoustic model: {difference}.'
                                )
                                continue
                        if word not in word_cache:
                            if pron == (self.optional_silence_phone,):
                                wt = WordType.silence
                            elif dictionary.clitic_marker is not None and (
                                word.startswith(dictionary.clitic_marker)
                                or word.endswith(dictionary.clitic_marker)
                            ):
                                wt = WordType.clitic
                            else:
                                wt = WordType.speech
                                for special_w, special_wt in special_words.items():
                                    if word == special_w:
                                        wt = special_wt
                                        specials_found.add(special_w)
                                        break
                            if word not in self._words_mappings:
                                self._words_mappings[word] = current_mapping_id
                                current_mapping_id += 1
                            word_objs.append(
                                {
                                    "id": word_primary_key,
                                    "mapping_id": self._words_mappings[word],
                                    "word": word,
                                    "word_type": wt,
                                    "dictionary_id": dictionary.id,
                                }
                            )
                            word_cache[word] = word_primary_key
                            word_primary_key += 1
                        pron_string = " ".join(pron)
                        if (word, pron_string) in pronunciation_cache:
                            continue

                        if not pretrained and word_objs[word_cache[word] - 1]["word_type"] in {
                            WordType.speech,
                            WordType.clitic,
                        }:
                            self.non_silence_phones.update(pron)

                        pron_objs.append(
                            {
                                "id": pronunciation_primary_key,
                                "pronunciation": pron_string,
                                "probability": prob,
                                "disambiguation": None,
                                "silence_after_probability": silence_after_prob,
                                "silence_before_correction": silence_before_correct,
                                "non_silence_before_correction": non_silence_before_correct,
                                "word_id": word_cache[word],
                            }
                        )
                        self.non_silence_phones.update(pron)

                        pronunciation_primary_key += 1
                        pronunciation_cache.add((word, pron_string))
                        phone_counts.update(pron)

                    for word, wt in special_words.items():
                        if word in specials_found:
                            continue
                        if word not in self._words_mappings:
                            self._words_mappings[word] = current_mapping_id
                            current_mapping_id += 1
                        word_objs.append(
                            {
                                "id": word_primary_key,
                                "mapping_id": self._words_mappings[word],
                                "word": word,
                                "word_type": wt,
                                "dictionary_id": dictionary.id,
                            }
                        )
                        pron_objs.append(
                            {
                                "id": pronunciation_primary_key,
                                "pronunciation": self.oov_phone,
                                "probability": 1.0,
                                "disambiguation": None,
                                "silence_after_probability": None,
                                "silence_before_correction": None,
                                "non_silence_before_correction": None,
                                "word_id": word_primary_key,
                            }
                        )
                        pronunciation_primary_key += 1
                        word_primary_key += 1
                    for word in ["#0", "<s>", "</s>"]:
                        if word not in self._words_mappings:
                            self._words_mappings[word] = current_mapping_id
                            current_mapping_id += 1
                        word_objs.append(
                            {
                                "id": word_primary_key,
                                "word": word,
                                "dictionary_id": dictionary.id,
                                "mapping_id": self._words_mappings[word],
                                "word_type": WordType.disambiguation,
                            }
                        )
                        word_primary_key += 1

                    if not graphemes:
                        raise DictionaryFileError(
                            f"No words were found in the dictionary path {dictionary_model.path}"
                        )
                    self.dictionary_lookup[dictionary.name] = dictionary.id
                    dictionary_id_cache[dictionary_model.path] = dictionary.id
                for speaker in speakers:
                    if speaker != "default":
                        if speaker not in self._speaker_ids:
                            speaker_objs.append(
                                {
                                    "id": self._current_speaker_index,
                                    "name": speaker,
                                    "dictionary_id": dictionary_id_cache[dictionary_model.path],
                                }
                            )
                            self._speaker_ids[speaker] = self._current_speaker_index
                            self._current_speaker_index += 1
                session.commit()

            self.non_silence_phones -= self.silence_phones
            grapheme_objs = []
            if graphemes:
                i = 0
                session.query(Grapheme).delete()
                session.commit()
                special_graphemes = [self.silence_word, "<space>"]
                special_graphemes.append(self.bracketed_word)
                special_graphemes.append(self.cutoff_word)
                special_graphemes.append(self.laughter_word)
                for g in special_graphemes:
                    grapheme_objs.append(
                        {
                            "id": i + 1,
                            "mapping_id": i,
                            "grapheme": g,
                        }
                    )
                    i += 1
                for g in sorted(graphemes) + [self.oov_word, "#0", "<s>", "</s>"]:
                    grapheme_objs.append(
                        {
                            "id": i + 1,
                            "mapping_id": i,
                            "grapheme": g,
                        }
                    )
                    i += 1
        with self.session() as session:
            with session.bind.begin() as conn:
                if word_objs:
                    conn.execute(sqlalchemy.insert(Word.__table__), word_objs)
                if pron_objs:
                    conn.execute(sqlalchemy.insert(Pronunciation.__table__), pron_objs)
                if speaker_objs:
                    conn.execute(sqlalchemy.insert(Speaker.__table__), speaker_objs)
                if grapheme_objs:
                    conn.execute(sqlalchemy.insert(Grapheme.__table__), grapheme_objs)

                session.commit()
        self.create_default_dictionary()
        if has_nonnative_speakers:
            self.create_nonnative_dictionary()
        if pron_objs:
            self.calculate_disambiguation()
        self.load_phonological_rules()
        self.calculate_phone_mapping()
        self.load_phone_topologies()
        self.load_phone_groups()
        return graphemes, phone_counts

    def create_default_dictionary(self):
        if self._default_dictionary_id is not None:  # Already a default dictionary
            return
        with self.session() as session:
            dictionary = session.query(Dictionary).filter_by(default=True).first()
            if dictionary is not None:  # Already a default dictionary
                self._default_dictionary_id = dictionary.id
                return
            dialect = session.query(Dialect).filter_by(name="unknown").first()
            if dialect is None:
                dialect = Dialect(name="unknown")
                session.add(dialect)
                session.flush()
            dictionary = Dictionary(
                name="default",
                dialect_id=dialect.id,
                path="",
                phone_set_type=self.phone_set_type,
                root_temp_directory=self.dictionary_output_directory,
                position_dependent_phones=self.position_dependent_phones,
                clitic_marker=self.clitic_marker if self.clitic_marker is not None else "",
                default=True,
                use_g2p=False,
                max_disambiguation_symbol=0,
                silence_word=self.silence_word,
                oov_word=self.oov_word,
                bracketed_word=self.bracketed_word,
                cutoff_word=self.cutoff_word,
                laughter_word=self.laughter_word,
                optional_silence_phone=self.optional_silence_phone,
                oov_phone=self.oov_phone,
            )
            session.add(dictionary)
            session.commit()

            self._default_dictionary_id = dictionary.id
            self.dictionary_lookup[dictionary.name] = dictionary.id

    def create_nonnative_dictionary(self):
        with self.session() as session:
            dictionary = session.query(Dictionary).filter_by(name="nonnative").first()
            word_objs = []
            pron_objs = []

            self.dictionary_lookup[dictionary.name] = dictionary.id
            word_primary_key = self.get_next_primary_key(Word)
            pronunciation_primary_key = self.get_next_primary_key(Pronunciation)
            word_cache = {}
            query = (
                session.query(
                    Word.word,
                    Word.mapping_id,
                    Word.word_type,
                    Pronunciation.pronunciation,
                )
                .join(Word.pronunciations, isouter=True)
                .distinct()
                .order_by(Word.mapping_id)
            )
            for word, mapping_id, word_type, pronunciation in query:
                if word not in word_cache:
                    word_objs.append(
                        {
                            "id": word_primary_key,
                            "mapping_id": mapping_id,
                            "word": word,
                            "word_type": word_type,
                            "dictionary_id": dictionary.id,
                        }
                    )
                    word_cache[word] = word_primary_key
                    word_primary_key += 1
                    if pronunciation is not None:
                        pron_objs.append(
                            {
                                "id": pronunciation_primary_key,
                                "pronunciation": pronunciation,
                                "probability": None,
                                "disambiguation": None,
                                "silence_after_probability": None,
                                "silence_before_correction": None,
                                "non_silence_before_correction": None,
                                "word_id": word_cache[word],
                            }
                        )
                        pronunciation_primary_key += 1
        with self.session() as session:
            with session.bind.begin() as conn:
                if word_objs:
                    conn.execute(sqlalchemy.insert(Word.__table__), word_objs)
                if pron_objs:
                    conn.execute(sqlalchemy.insert(Pronunciation.__table__), pron_objs)
                session.commit()

    def calculate_disambiguation(self) -> None:
        """Calculate the number of disambiguation symbols necessary for the dictionary"""
        with self.session() as session:
            dictionaries = session.query(Dictionary)
            update_pron_objs = []
            for d in dictionaries:
                if d.name == "default":
                    continue
                subsequences = set()
                words = (
                    session.query(Word)
                    .filter(Word.dictionary_id == d.id)
                    .filter(Word.included == True)  # noqa
                    .options(selectinload(Word.pronunciations))
                ).all()
                for w in words:
                    for p in w.pronunciations:
                        pron = p.pronunciation.split()
                        while len(pron) > 0:
                            subsequences.add(tuple(pron))
                            pron = pron[:-1]
                last_used = collections.defaultdict(int)
                for w in words:
                    for p in w.pronunciations:
                        pron = p.pronunciation
                        pron = tuple(pron.split())
                        if pron in subsequences:
                            last_used[pron] += 1

                            update_pron_objs.append(
                                {"id": p.id, "disambiguation": last_used[pron]}
                            )

                if last_used:
                    d.max_disambiguation_symbol = max(
                        d.max_disambiguation_symbol, max(last_used.values())
                    )
                self.max_disambiguation_symbol = max(
                    self.max_disambiguation_symbol, d.max_disambiguation_symbol
                )
            if update_pron_objs:
                bulk_update(session, Pronunciation, update_pron_objs)
                session.commit()

    def load_phonological_rules(self) -> None:
        if not self.rules_path or not self.rules_path.exists():
            return
        with mfa_open(self.rules_path) as f:
            rule_data = yaml.load(f, Loader=yaml.Loader)
        with self.session() as session:
            if session.query(PhonologicalRule).first() is not None:
                return
            dialect_ids = {d.name: d.id for d in session.query(Dialect).all()}
            for rule in rule_data["rules"]:
                for d_id in dialect_ids.values():
                    r = PhonologicalRule(dialect_id=d_id, **rule)
                    session.add(r)
                    if r.replacement:
                        self.non_silence_phones.update(r.replacement.split())
            for k, v in rule_data.get("dialects", {}).items():
                d_id = dialect_ids.get(k, None)
                if not d_id:
                    continue
                for rule in v:
                    r = PhonologicalRule(dialect_id=d_id, **rule)
                    session.add(r)
                    if r.replacement:
                        self.non_silence_phones.update(r.replacement.split())
            session.commit()

    def apply_phonological_rules(self) -> None:
        """Apply any phonological rules specified in the rules file path"""
        with self.session() as session:
            num_words = (
                session.query(Word)
                .filter(Word.included == True)  # noqa
                .filter(Word.word_type.in_(WordType.speech_types()))
                .count()
            )
            if session.query(PhonologicalRule).first() is None:
                return
            logger.info("Applying phonological rules...")
            with tqdm(total=num_words, disable=config.QUIET) as pbar:
                new_pron_objs = []
                rule_application_objs = []
                pronunciation_primary_key = session.query(
                    sqlalchemy.func.max(Pronunciation.id)
                ).scalar()
                if not pronunciation_primary_key:
                    pronunciation_primary_key = 0
                pronunciation_primary_key += 1
                dictionaries = session.query(Dictionary)
                for d in dictionaries:
                    if d.rules_applied:
                        continue
                    words = (
                        session.query(Word)
                        .filter(Word.dictionary_id == d.id)
                        .filter(Word.included == True)  # noqa
                        .filter(Word.word_type.in_(WordType.speech_types()))
                        .options(selectinload(Word.pronunciations))
                    )
                    rules = (
                        session.query(PhonologicalRule)
                        .filter(PhonologicalRule.dialect_id == d.dialect_id)
                        .all()
                    )
                    for w in words:
                        pbar.update(1)
                        variant_to_rule_mapping = collections.defaultdict(set)
                        variant_mapping = {}
                        existing_prons = {p.pronunciation for p in w.pronunciations}
                        for p in w.pronunciations:
                            new_variants = [p.pronunciation]

                            variant_index = 0
                            while True:
                                s = new_variants[variant_index]
                                for r in rules:
                                    n = r.apply_rule(s)
                                    if any(x not in self.non_silence_phones for x in n.split()):
                                        continue
                                    if n and n not in existing_prons and n not in new_variants:
                                        new_pron_objs.append(
                                            {
                                                "id": pronunciation_primary_key,
                                                "pronunciation": n,
                                                "probability": None,
                                                "disambiguation": None,
                                                "silence_after_probability": None,
                                                "silence_before_correction": None,
                                                "non_silence_before_correction": None,
                                                "generated_by_rule": True,
                                                "word_id": w.id,
                                            }
                                        )
                                        new_variants.append(n)
                                        existing_prons.add(n)
                                        variant_mapping[n] = pronunciation_primary_key
                                        variant_to_rule_mapping[n].update(
                                            variant_to_rule_mapping[s]
                                        )
                                        variant_to_rule_mapping[n].add(r)
                                        pronunciation_primary_key += 1
                                variant_index += 1

                                if variant_index >= len(new_variants):
                                    break
                        for v, rs in variant_to_rule_mapping.items():
                            for r in rs:
                                rule_application_objs.append(
                                    {"rule_id": r.id, "pronunciation_id": variant_mapping[v]}
                                )
                if new_pron_objs:
                    session.execute(sqlalchemy.insert(Pronunciation.__table__), new_pron_objs)
                d.rules_applied = True
                if rule_application_objs:
                    session.execute(
                        sqlalchemy.insert(RuleApplication.__table__), rule_application_objs
                    )
                    session.commit()

    def calculate_phone_mapping(self) -> None:
        """Calculate the necessary phones and add phone objects to the database"""
        with self.session() as session:
            try:
                session.query(Phone).delete()
            except Exception:
                return
            acoustic_model: AcousticModel = getattr(self, "acoustic_model", None)
            phone_objs = []
            existing_phones = {"<eps>"}
            i = 1
            if acoustic_model is not None and "phone_mapping" in acoustic_model.meta:
                for p, v in acoustic_model.meta["phone_mapping"].items():
                    phone_type = PhoneType.non_silence
                    if p in self.kaldi_silence_phones:
                        phone_type = PhoneType.silence
                        if self.oov_phone in p:
                            phone_type = PhoneType.oov
                    original_phone = split_phone_position(p)[0]
                    phone_objs.append(
                        {
                            "id": i,
                            "mapping_id": v,
                            "phone": original_phone,
                            "kaldi_label": p,
                            "phone_type": phone_type,
                            "count": 0,
                        }
                    )
                    i += 1
            else:
                rules = session.query(PhonologicalRule)
                for r in rules:
                    if not r.replacement:
                        continue
                    self.non_silence_phones.update(r.replacement.split())
                phone_objs.append(
                    {
                        "id": i,
                        "mapping_id": i - 1,
                        "phone": "<eps>",
                        "kaldi_label": "<eps>",
                        "phone_type": PhoneType.silence,
                        "count": 0,
                    }
                )
                i += 1
                for p in self.kaldi_silence_phones:
                    if p in existing_phones:
                        continue
                    phone = p
                    position = None
                    if self.position_dependent_phones:
                        phone, position = split_phone_position(p)
                    if p == self.oov_phone:
                        phone_type = PhoneType.oov
                    else:
                        phone_type = PhoneType.silence
                    phone_objs.append(
                        {
                            "id": i,
                            "mapping_id": i - 1,
                            "phone": phone,
                            "position": position,
                            "kaldi_label": p,
                            "phone_type": phone_type,
                            "count": 0,
                        }
                    )
                    i += 1
                    existing_phones.add(p)
                for p in self.kaldi_non_silence_phones:
                    if p in existing_phones:
                        continue
                    phone = p
                    position = None
                    if self.position_dependent_phones:
                        phone, position = split_phone_position(p)
                    phone_objs.append(
                        {
                            "id": i,
                            "mapping_id": i - 1,
                            "phone": phone,
                            "position": position,
                            "kaldi_label": p,
                            "phone_type": PhoneType.non_silence,
                            "count": 0,
                        }
                    )
                    i += 1
                    existing_phones.add(p)
            for x in range(self.max_disambiguation_symbol + 3):
                p = f"#{x}"
                if p in existing_phones:
                    continue
                self.disambiguation_symbols.add(p)
                phone_objs.append(
                    {
                        "id": i,
                        "mapping_id": i - 1,
                        "phone": p,
                        "kaldi_label": p,
                        "phone_type": PhoneType.disambiguation,
                        "count": 0,
                    }
                )
                i += 1
                existing_phones.add(p)
            if phone_objs:
                session.execute(sqlalchemy.insert(Phone.__table__), phone_objs)
                session.commit()

    def compile_phone_group_lexicon_fst(self):
        grouped_phones = {**self.kaldi_grouped_phones}

        grouped_phones["silence"] = [self.optional_silence_phone]
        grouped_phones["unknown"] = [self.oov_phone]
        if self.position_dependent_phones:
            grouped_phones["silence"] += [self.optional_silence_phone + x for x in self.positions]
            grouped_phones["unknown"] += [self.oov_phone + x for x in self.positions]
        group_table = pywrapfst.SymbolTable()
        group_table.add_symbol("<eps>")
        for k in sorted(grouped_phones.keys()):
            group_table.add_symbol(k)
        fst = pynini.Fst()
        initial_state = fst.add_state()
        fst.set_start(initial_state)
        fst.set_final(initial_state, 0)
        processed = set()
        phone_to_group_mapping = {}
        for k, group in grouped_phones.items():
            for p in group:
                phone_to_group_mapping[p] = k
                fst.add_arc(
                    initial_state,
                    pywrapfst.Arc(self.phone_table.find(p), group_table.find(k), 0, initial_state),
                )
            processed.update(group)
        for i in range(self.phone_table.num_symbols()):
            phone = self.phone_table.find(i)
            if phone in processed:
                continue
            group_symbol = group_table.add_symbol(phone)
            fst.add_arc(initial_state, pywrapfst.Arc(i, group_symbol, 0, initial_state))
        fst.arcsort("olabel")
        return fst, group_table, phone_to_group_mapping

    def export_trained_rules(self, output_directory: str) -> None:
        """
        Export rules with pronunciation and silence probabilities calculated to an output directory

        Parameters
        ----------
        output_directory: str
            Directory for export
        """
        with self.session() as session:
            rules = (
                session.query(PhonologicalRule)
                .filter(PhonologicalRule.probability != None)  # noqa
                .all()
            )
            if rules:
                output_rules_path = os.path.join(output_directory, "rules.yaml")
                dialectal_rules = {"rules": []}
                for r in rules:
                    d = r.to_json()
                    dialect = d.pop("dialect")
                    if dialect is None:
                        dialectal_rules["rules"].append(d)
                    else:
                        dialect = dialect.name
                        if "dialects" not in dialectal_rules:
                            dialectal_rules["dialects"] = {}
                        if dialect not in dialectal_rules["dialects"]:
                            dialectal_rules["dialects"][dialect] = []
                        dialectal_rules["dialects"][dialect].append(d)
                with mfa_open(output_rules_path, "w") as f:
                    yaml.dump(dict(dialectal_rules), f, Dumper=yaml.Dumper, allow_unicode=True)

    def add_words(
        self, new_word_data: typing.List[typing.Dict[str, typing.Any]], dictionary_id: int = None
    ) -> None:
        """
        Add word data to a dictionary in the form exported from
        :meth:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin.words_for_export`

        Parameters
        ----------
        new_word_data: list[dict[str,Any]]
            Word data to add
        dictionary_id: int, optional
            Dictionary id to add words, defaults to the default dictionary
        """
        if dictionary_id is None:
            dictionary_id = self._default_dictionary_id
        word_mapping = {}
        pronunciation_mapping = []
        word_index = self.get_next_primary_key(Word)
        pronunciation_index = self.get_next_primary_key(Pronunciation)
        with self.session() as session:
            word_mapping_index = (
                session.query(sqlalchemy.func.max(Word.mapping_id))
                .filter(Word.dictionary_id == dictionary_id)
                .scalar()
                + 1
            )
            for data in new_word_data:
                word = data["word"]
                if word in self.word_mapping(dictionary_id):
                    continue
                if word not in word_mapping:
                    word_mapping[word] = {
                        "id": word_index,
                        "mapping_id": word_mapping_index,
                        "word": word,
                        "word_type": WordType.speech,
                        "count": 0,
                        "dictionary_id": dictionary_id,
                    }
                    word_index += 1
                    word_mapping_index += 1
                phones = data["pronunciation"]
                d = {
                    "id": pronunciation_index,
                    "word_id": word_mapping[word]["id"],
                    "pronunciation": phones,
                }
                pronunciation_index += 1
                if "probability" in data and data["probability"] is not None:
                    d["probability"] = data["probability"]
                    d["silence_after_probability"] = data["silence_after_probability"]
                    d["silence_before_correction"] = data["silence_before_correction"]
                    d["non_silence_before_correction"] = data["non_silence_before_correction"]

                pronunciation_mapping.append(d)
            self._num_speech_words = None
            session.bulk_insert_mappings(Word, list(word_mapping.values()))
            session.flush()
            session.bulk_insert_mappings(Pronunciation, pronunciation_mapping)
            session.commit()

    def words_for_export(
        self,
        dictionary_id: int = None,
        write_disambiguation: typing.Optional[bool] = False,
        probability: typing.Optional[bool] = False,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Generate exportable pronunciations

        Parameters
        ----------
        dictionary_id: int, optional
            Dictionary id to export, defaults to the default dictionary
        write_disambiguation: bool, optional
            Flag for whether to include disambiguation information
        probability: bool, optional
            Flag for whether to include probabilities

        Returns
        -------
        list[dict[str,Any]]
            List of pronunciations as dictionaries
        """
        if dictionary_id is None:
            dictionary_id = self._default_dictionary_id
        with self.session() as session:
            columns = [Word.word, Pronunciation.pronunciation]
            if write_disambiguation:
                columns.append(Pronunciation.disambiguation)
            if probability:
                columns.append(Pronunciation.probability)
                columns.append(Pronunciation.silence_after_probability)
                columns.append(Pronunciation.silence_before_correction)
                columns.append(Pronunciation.non_silence_before_correction)
            bn = DictBundle("pronunciation_data", *columns)
            pronunciations = (
                session.query(bn)
                .join(Pronunciation.word)
                .filter(
                    Word.dictionary_id == dictionary_id,
                    sqlalchemy.or_(
                        Word.word_type.in_(
                            [WordType.speech, WordType.clitic, WordType.interjection]
                        ),
                        Word.word.in_(
                            [
                                self.oov_word,
                                self.bracketed_word,
                                self.cutoff_word,
                                self.laughter_word,
                            ]
                        ),
                    ),
                )
                .order_by(Word.word)
            )
            data = [row for row, in pronunciations]
        return data

    def export_lexicon(
        self,
        dictionary_id: int,
        path: Path,
        write_disambiguation: typing.Optional[bool] = False,
        probability: typing.Optional[bool] = False,
    ) -> None:
        """
        Export pronunciation dictionary to a text file

        Parameters
        ----------
        path: :class:`~pathlib.Path`
            Path to save dictionary
        write_disambiguation: bool, optional
            Flag for whether to include disambiguation information
        probability: bool, optional
            Flag for whether to include probabilities
        """
        with mfa_open(path, "w") as f:
            for data in self.words_for_export(dictionary_id, write_disambiguation, probability):
                phones = data["pronunciation"]
                if write_disambiguation and data["disambiguation"] is not None:
                    phones += f" #{data['disambiguation']}"
                probability_string = ""
                if probability and data["probability"] is not None:
                    probability_string = f"{data['probability']}"

                    extra_probs = [
                        data["silence_after_probability"],
                        data["silence_before_correction"],
                        data["non_silence_before_correction"],
                    ]
                    if all(x is None for x in extra_probs):
                        continue
                    for x in extra_probs:
                        if x is None:
                            continue
                        probability_string += f"\t{x}"
                if probability_string:
                    f.write(f"{data['word']}\t{probability_string}\t{phones}\n")
                else:
                    f.write(f"{data['word']}\t{phones}\n")

    @property
    def phone_disambig_path(self) -> str:
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_dir, "phone_disambig.txt")

    @property
    def phone_mapping(self) -> Dict[str, int]:
        """Mapping of phone symbols to integer IDs for Kaldi processing"""
        if self._phone_mapping is None:
            with self.session() as session:
                phones = (
                    session.query(Phone.kaldi_label, Phone.mapping_id).order_by(Phone.id).all()
                )
                self._phone_mapping = {x[0]: x[1] for x in phones}
        return self._phone_mapping

    @property
    def phone_table(self) -> pywrapfst.SymbolTable:
        """Mapping of phone symbols to integer IDs for Kaldi processing"""
        if self._phone_table is None:
            self._phone_table = pywrapfst.SymbolTable()
            with self.session() as session:
                phones = (
                    session.query(Phone.kaldi_label, Phone.mapping_id).order_by(Phone.id).all()
                )
                for p, m_id in phones:
                    self._phone_table.add_symbol(p, m_id)
        return self._phone_table

    @property
    def grapheme_mapping(self) -> Dict[str, int]:
        """Mapping of phone symbols to integer IDs for Kaldi processing"""
        if self._grapheme_mapping is None:
            with self.session() as session:
                graphemes = (
                    session.query(Grapheme.grapheme, Grapheme.mapping_id)
                    .order_by(Grapheme.id)
                    .all()
                )
                self._grapheme_mapping = {x[0]: x[1] for x in graphemes}
        return self._grapheme_mapping

    @property
    def reversed_grapheme_mapping(self) -> Dict[int, str]:
        """
        A mapping of integer ids to graphemes
        """
        mapping = {}
        for k, v in self.grapheme_mapping.items():
            mapping[v] = k
        return mapping

    def _write_grapheme_symbol_table(self) -> None:
        """
        Write the phone mapping to the temporary directory
        """
        with mfa_open(self.grapheme_symbol_table_path, "w") as f:
            for p, i in self.grapheme_mapping.items():
                f.write(f"{p} {i}\n")

    def _write_extra_questions(self) -> None:
        """
        Write extra questions symbols to the temporary directory
        """
        phone_extra = os.path.join(self.phones_dir, "extra_questions.txt")
        phone_extra_int = os.path.join(self.phones_dir, "extra_questions.int")
        with mfa_open(phone_extra, "w") as outf, mfa_open(phone_extra_int, "w") as intf:
            for v in self.extra_questions_mapping.values():
                if not v:
                    continue
                outf.write(f"{' '.join(v)}\n")
                intf.write(f"{' '.join(str(self.phone_mapping[x]) for x in v)}\n")

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return self.dictionary_model.name

    def save_oovs_found(self, directory: str) -> None:
        """
        Save all out of vocabulary items to a file in the specified directory

        Parameters
        ----------
        directory : str
            Path to directory to save ``oovs_found.txt``
        """
        with self.session() as session:
            for dict_id, base_name in self.dictionary_base_names.items():
                with mfa_open(
                    os.path.join(directory, f"oovs_found_{base_name}.txt"),
                    "w",
                    encoding="utf8",
                ) as f, mfa_open(
                    os.path.join(directory, f"oov_counts_{base_name}.txt"),
                    "w",
                    encoding="utf8",
                ) as cf:
                    oovs = (
                        session.query(Word.word, Word.count)
                        .filter(
                            Word.dictionary_id == dict_id,
                            Word.word_type == WordType.oov,
                            Word.word != self.oov_word,
                        )
                        .order_by(sqlalchemy.desc(Word.count))
                    )
                    for word, count in oovs:
                        f.write(word + "\n")
                        cf.write(f"{word}\t{count}\n")

    def find_all_cutoffs(self) -> None:
        """Find all instances of cutoff words followed by actual words"""
        with self.session() as session:
            c = session.query(Corpus).first()
            if c.cutoffs_found:
                logger.debug("Cutoffs already found")
                return
            logger.info("Finding all cutoffs...")
            initial_brackets = re.escape("".join(x[0] for x in self.brackets))
            final_brackets = re.escape("".join(x[1] for x in self.brackets))
            cutoff_identifier = re.sub(
                rf"[{initial_brackets}{final_brackets}]", "", self.cutoff_word
            )
            max_pron_id = session.query(sqlalchemy.func.max(Pronunciation.id)).scalar()
            max_word_id = session.query(sqlalchemy.func.max(Word.id)).scalar()
            for d_id in self.dictionary_lookup.values():
                pronunciation_mapping = collections.defaultdict(set)
                new_word_mapping = {}
                new_pronunciation_mapping = []
                max_id = (
                    session.query(sqlalchemy.func.max(Word.mapping_id))
                    .join(Word.dictionary)
                    .filter(Dictionary.id == d_id)
                ).first()[0]
                words = (
                    session.query(Word.word, Pronunciation.pronunciation)
                    .join(Pronunciation.word)
                    .filter(
                        Word.dictionary_id == d_id,
                        Word.count > 1,
                        Word.word_type == WordType.speech,
                    )
                )
                for w, pron in words:
                    pronunciation_mapping[w].add(pron)
                    new_word = f"{self.cutoff_word[:-1]}-{w}{self.cutoff_word[-1]}"
                    if new_word not in new_word_mapping:
                        max_word_id += 1
                        max_id += 1
                        max_pron_id += 1
                        pronunciation_mapping[new_word].add(self.oov_phone)
                        new_pronunciation_mapping.append(
                            {
                                "id": max_pron_id,
                                "pronunciation": self.oov_phone,
                                "word_id": max_word_id,
                            }
                        )
                        new_word_mapping[new_word] = {
                            "id": max_word_id,
                            "word": new_word,
                            "dictionary_id": d_id,
                            "mapping_id": max_id,
                            "word_type": WordType.cutoff,
                            "count": 0,
                            "included": False,
                        }
                    p = pron.split()
                    for pi in range(len(p)):
                        new_p = " ".join(p[: pi + 1])

                        if new_p in pronunciation_mapping[new_word]:
                            continue
                        pronunciation_mapping[new_word].add(new_p)
                        max_pron_id += 1
                        new_pronunciation_mapping.append(
                            {
                                "id": max_pron_id,
                                "pronunciation": new_p,
                                "word_id": new_word_mapping[new_word]["id"],
                            }
                        )
                utterances = (
                    session.query(
                        Utterance.id,
                        Utterance.normalized_text,
                    )
                    .join(Utterance.speaker)
                    .filter(
                        Speaker.dictionary_id == d_id,
                        Utterance.normalized_text.regexp_match(
                            f"[{initial_brackets}]{cutoff_identifier}"
                        ),
                    )
                )
                for u_id, normalized_text in utterances:
                    text = normalized_text.split()
                    for i, word in enumerate(text):
                        if not word.startswith(f"{self.cutoff_word[:-1]}-"):
                            continue
                        if word not in new_word_mapping:
                            continue
                        new_word_mapping[word]["count"] += 1
                        new_word_mapping[word]["included"] = True
                session.bulk_insert_mappings(
                    Word, new_word_mapping.values(), return_defaults=False, render_nulls=True
                )
                session.bulk_insert_mappings(
                    Pronunciation,
                    new_pronunciation_mapping,
                    return_defaults=False,
                    render_nulls=True,
                )
                session.flush()
            session.query(Corpus).update({"cutoffs_found": True})
            session.commit()
            oov_count_threshold = getattr(self, "oov_count_threshold", 0)
            if oov_count_threshold > 0:
                session.query(Word).filter(Word.word_type == WordType.cutoff).filter(
                    Word.count <= oov_count_threshold
                ).update({Word.included: False})
                session.commit()
        self._words_mappings = {}

    def write_lexicon_information(self, write_disambiguation: Optional[bool] = False) -> None:
        """
        Write all child dictionaries to the temporary directory

        Parameters
        ----------
        write_disambiguation: bool, optional
            Flag to use disambiguation symbols in the output
        """
        os.makedirs(self.phones_dir, exist_ok=True)
        if self.use_cutoff_model:
            self.find_all_cutoffs()
        self.phone_table.write_text(self.phone_symbol_table_path)
        self._write_grapheme_symbol_table()
        self._write_disambig()
        if self.position_dependent_phones:
            self._write_word_boundaries()

        with self.session() as session:
            dictionaries: typing.List[Dictionary] = session.query(Dictionary)
            for d in dictionaries:
                d.temp_directory.mkdir(parents=True, exist_ok=True)
                if self.use_g2p:
                    fst = pynini.Fst.read(d.lexicon_fst_path)
                    align_fst = pynini.Fst.read(d.align_lexicon_path)
                    grapheme_table = pywrapfst.SymbolTable.read_text(d.grapheme_symbol_table_path)
                    self.lexicon_compilers[d.id] = G2PCompiler(
                        fst,
                        grapheme_table,
                        self.phone_table,
                        align_fst=align_fst,
                        silence_phone=self.optional_silence_phone,
                    )

                else:
                    d.words_symbol_path.unlink(missing_ok=True)
                    self.lexicon_compilers[d.id] = self.build_lexicon_compiler(
                        d.id, disambiguation=write_disambiguation
                    )
                    self.lexicon_compilers[d.id].word_table.write_text(d.words_symbol_path)

    def load_lexicon_compilers(self):
        with self.session() as session:
            self.lexicon_compilers = {}
            dictionaries = session.query(Dictionary)
            for d in dictionaries:
                self.lexicon_compilers[d.id] = d.lexicon_compiler

    def build_lexicon_compiler(
        self,
        dictionary_id: int,
        acoustic_model: AcousticModel = None,
        disambiguation: bool = False,
    ):
        with self.session() as session:
            d = session.get(Dictionary, dictionary_id)
            if acoustic_model is None:
                lexicon_compiler = LexiconCompiler(
                    disambiguation=disambiguation,
                    silence_probability=self.silence_probability,
                    initial_silence_probability=self.initial_silence_probability,
                    final_silence_correction=self.final_silence_correction,
                    final_non_silence_correction=self.final_non_silence_correction,
                    silence_word=self.silence_word,
                    oov_word=self.oov_word,
                    silence_phone=self.optional_silence_phone,
                    oov_phone=self.oov_phone,
                    position_dependent_phones=self.position_dependent_phones,
                    ignore_case=self.ignore_case,
                    phones=self.non_silence_phones,
                )
                lexicon_compiler.phone_table = self.phone_table
            else:
                lexicon_compiler = acoustic_model.lexicon_compiler
                lexicon_compiler.disambiguation = disambiguation
            if d.name != "default":
                query = (
                    session.query(
                        Word.mapping_id,
                        Word.word,
                        Pronunciation.pronunciation,
                        Pronunciation.probability,
                        Pronunciation.silence_after_probability,
                        Pronunciation.silence_before_correction,
                        Pronunciation.non_silence_before_correction,
                    )
                    .join(Pronunciation.word)
                    .filter(Word.dictionary_id == d.id)
                    .filter(Word.included == True)  # noqa
                    .filter(Word.word_type != WordType.silence)
                    .order_by(Word.word)
                )
            else:
                query = (
                    session.query(
                        Word.mapping_id,
                        Word.word,
                        Pronunciation.pronunciation,
                        sqlalchemy.func.avg(Pronunciation.probability),
                        sqlalchemy.func.avg(Pronunciation.silence_after_probability),
                        sqlalchemy.func.avg(Pronunciation.silence_before_correction),
                        sqlalchemy.func.avg(Pronunciation.non_silence_before_correction),
                    )
                    .join(Pronunciation.word)
                    .filter(Word.included == True)  # noqa
                    .filter(Word.word_type != WordType.silence)
                    .group_by(Word.mapping_id, Word.word, Pronunciation.pronunciation)
                    .order_by(Word.mapping_id)
                )
            lexicon_compiler.word_table = d.word_table
            for (
                mapping_id,
                word,
                pronunciation,
                probability,
                silence_after_probability,
                silence_before_correction,
                non_silence_before_correction,
            ) in query:
                phones = pronunciation.split()
                if self.position_dependent_phones:
                    if any(not lexicon_compiler.phone_table.member(x + "_S") for x in phones):
                        continue
                else:
                    if any(not lexicon_compiler.phone_table.member(x) for x in phones):
                        continue
                if not lexicon_compiler.word_table.member(word):
                    lexicon_compiler.word_table.add_symbol(word, mapping_id)
                lexicon_compiler.pronunciations.append(
                    KalpyPronunciation(
                        word,
                        pronunciation,
                        probability,
                        silence_after_probability,
                        silence_before_correction,
                        non_silence_before_correction,
                        None,
                    )
                )
            if not lexicon_compiler.pronunciations:
                raise DictionaryError("Lexicon compiler did not have any pronunciations.")
            lexicon_compiler.compute_disambiguation_symbols()

            lexicon_compiler.create_fsts()
            if disambiguation:
                lexicon_compiler.align_fst.write(d.align_lexicon_disambig_path)
                lexicon_compiler.fst.write(d.lexicon_disambig_fst_path)
            else:
                lexicon_compiler.align_fst.write(d.align_lexicon_path)
                lexicon_compiler.fst.write(d.lexicon_fst_path)
            lexicon_compiler.clear()
        return lexicon_compiler

    def write_training_information(self) -> None:
        """Write phone information needed for training"""
        self._write_topo()
        self._write_phone_sets()
        self._write_extra_questions()


class MultispeakerDictionary(MultispeakerDictionaryMixin):
    """
    Class for processing multi- and single-speaker pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
    """

    @property
    def actual_words(self) -> typing.List[str]:
        with self.session() as session:
            actual_words = [
                x[0]
                for x in session.query(Word.word).filter(
                    Word.word_type.in_(WordType.speech_types())
                )
            ]
        return actual_words

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
        return config.TEMPORARY_DIRECTORY.joinpath(self.identifier)
