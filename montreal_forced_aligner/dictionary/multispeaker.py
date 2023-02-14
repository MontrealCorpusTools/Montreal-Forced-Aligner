"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import abc
import collections
import logging
import math
import os
import re
import subprocess
import typing
from pathlib import Path
from typing import Dict, Optional, Tuple

import pynini
import pywrapfst
import sqlalchemy.orm.session
import tqdm
import yaml
from sqlalchemy.orm import selectinload

from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import PhoneType, WordType
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
    KaldiProcessingError,
)
from montreal_forced_aligner.helper import mfa_open, split_phone_position
from montreal_forced_aligner.models import DictionaryModel, PhoneSetType
from montreal_forced_aligner.utils import parse_dictionary_file, thirdparty_binary

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
        self.rules_path = rules_path
        self.phone_groups_path = phone_groups_path

    def load_phone_groups(self) -> None:
        """
        Load phone groups from the dictionary's groups file path
        """
        if self.phone_groups_path is not None and self.phone_groups_path.exists():
            with mfa_open(self.phone_groups_path) as f:
                self._phone_groups = yaml.safe_load(f)
                if isinstance(self._phone_groups, list):
                    self._phone_groups = {k: v for k, v in enumerate(self._phone_groups)}
                for k, v in self._phone_groups.items():
                    self._phone_groups[k] = [x for x in v if x in self.non_silence_phones]

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

    def word_mapping(self, dictionary_id: int = 1) -> Dict[str, int]:
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
        if dictionary_id not in self._words_mappings:
            self._words_mappings[dictionary_id] = {}
            with self.session() as session:
                words = session.query(Word.word, Word.mapping_id).filter(
                    Word.dictionary_id == dictionary_id
                )
                for w, index in words:
                    self._words_mappings[dictionary_id][w] = index
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

    def dictionary_setup(self) -> Tuple[typing.Set[str], collections.Counter]:
        """Set up the dictionary for processing"""
        self.initialize_database()
        if self.use_g2p:
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
                if dictionary_model.path not in dictionary_id_cache and not self.use_g2p:
                    word_cache = {}
                    pronunciation_cache = set()
                    subsequences = set()
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
                        laughter_word=self.laughter_word,
                        optional_silence_phone=self.optional_silence_phone,
                        oov_phone=self.oov_phone,
                    )
                    session.add(dictionary)
                    session.flush()
                    dictionary_id_cache[dictionary_model.path] = dictionary.id
                    if dictionary.default:
                        self._default_dictionary_id = dictionary.id
                    self._words_mappings[dictionary.id] = {}
                    current_index = 0
                    word_objs.append(
                        {
                            "id": word_primary_key,
                            "mapping_id": current_index,
                            "word": self.silence_word,
                            "word_type": WordType.silence,
                            "dictionary_id": dictionary.id,
                        }
                    )
                    self._words_mappings[dictionary.id][self.silence_word] = current_index
                    current_index += 1

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
                            "base_pronunciation_id": pronunciation_primary_key,
                        }
                    )
                    word_primary_key += 1
                    pronunciation_primary_key += 1

                    special_words = {self.oov_word: WordType.oov}
                    special_words[self.bracketed_word] = WordType.bracketed
                    special_words[self.laughter_word] = WordType.laughter
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
                        characters = list(word)
                        if word not in special_words:
                            graphemes.update(characters)
                        if pretrained:
                            difference = set(pron) - self.non_silence_phones - self.silence_phones
                            if difference:
                                self.excluded_phones.update(difference)
                                self.excluded_pronunciation_count += 1
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
                            word_objs.append(
                                {
                                    "id": word_primary_key,
                                    "mapping_id": current_index,
                                    "word": word,
                                    "word_type": wt,
                                    "dictionary_id": dictionary.id,
                                }
                            )
                            self._words_mappings[dictionary.id][word] = current_index
                            current_index += 1
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
                        base_pronunciation_key = pronunciation_primary_key

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
                                "base_pronunciation_id": base_pronunciation_key,
                            }
                        )
                        self.non_silence_phones.update(pron)

                        pronunciation_primary_key += 1
                        pronunciation_cache.add((word, pron_string))
                        phone_counts.update(pron)
                        pron = pron[:-1]
                        while pron:
                            subsequences.add(tuple(pron))
                            pron = pron[:-1]

                    for w, wt in special_words.items():
                        if w in specials_found:
                            continue
                        word_objs.append(
                            {
                                "id": word_primary_key,
                                "mapping_id": current_index,
                                "word": w,
                                "word_type": wt,
                                "dictionary_id": dictionary.id,
                            }
                        )
                        self._words_mappings[dictionary.id][w] = current_index
                        current_index += 1
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
                                "base_pronunciation_id": pronunciation_primary_key,
                            }
                        )
                        pronunciation_primary_key += 1
                        word_primary_key += 1
                    for s in ["#0", "<s>", "</s>"]:
                        word_objs.append(
                            {
                                "id": word_primary_key,
                                "word": s,
                                "dictionary_id": dictionary.id,
                                "mapping_id": current_index,
                                "word_type": WordType.disambiguation,
                            }
                        )
                        self._words_mappings[dictionary.id][s] = current_index
                        word_primary_key += 1
                        current_index += 1

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
        if pron_objs:
            self.apply_phonological_rules()
            self.calculate_disambiguation()
        self.calculate_phone_mapping()
        self.load_phone_groups()
        return graphemes, phone_counts

    def calculate_disambiguation(self) -> None:
        """Calculate the number of disambiguation symbols necessary for the dictionary"""
        with self.session() as session:
            dictionaries = session.query(Dictionary)
            update_pron_objs = []
            for d in dictionaries:
                subsequences = set()
                words = (
                    session.query(Word)
                    .filter(Word.dictionary_id == d.id)
                    .options(selectinload(Word.pronunciations))
                )
                for w in words:
                    for p in w.pronunciations:

                        pron = p.pronunciation.split()
                        while pron:
                            subsequences.add(tuple(pron))
                            pron = pron[:-1]
                last_used = collections.defaultdict(int)
                for p_id, pron in (
                    session.query(Pronunciation.id, Pronunciation.pronunciation)
                    .join(Pronunciation.word)
                    .filter(Word.dictionary_id == d.id)
                ):
                    pron = tuple(pron.split())
                    if pron in subsequences:
                        last_used[pron] += 1

                        update_pron_objs.append({"id": p_id, "disambiguation": last_used[pron]})

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

    def apply_phonological_rules(self) -> None:
        """Apply any phonological rules specified in the rules file path"""
        # Set up phonological rules
        if not self.rules_path or not self.rules_path.exists():
            return
        with mfa_open(self.rules_path) as f:
            rule_data = yaml.safe_load(f)
        with self.session() as session:
            num_words = session.query(Word).count()
            logger.info("Applying phonological rules...")
            with tqdm.tqdm(total=num_words, disable=GLOBAL_CONFIG.quiet) as pbar:
                new_pron_objs = []
                rule_application_objs = []
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
                session.flush()
                pronunciation_primary_key = session.query(
                    sqlalchemy.func.max(Pronunciation.id)
                ).scalar()
                if not pronunciation_primary_key:
                    pronunciation_primary_key = 0
                pronunciation_primary_key += 1
                dictionaries = session.query(Dictionary)
                for d in dictionaries:
                    words = (
                        session.query(Word)
                        .filter(Word.dictionary_id == d.id)
                        .filter(Word.word_type.in_([WordType.clitic, WordType.speech]))
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
                            base_id = p.id
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
                                                "word_id": w.id,
                                                "base_pronunciation_id": base_id,
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
            i = 1
            for r in session.query(PhonologicalRule):
                if not r.replacement:
                    continue
                self.non_silence_phones.update(r.replacement.split())
            phone_objs = []
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
            existing_phones = {"<eps>"}
            i += 1
            for p in self.kaldi_silence_phones:
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
                        "phone_type": PhoneType.silence,
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

    def _write_probabilistic_fst_text(
        self,
        session: sqlalchemy.orm.session.Session,
        dictionary: Dictionary,
        silence_disambiguation_symbol=None,
        path: typing.Optional[str] = None,
        alignment: bool = False,
    ) -> None:
        """
        Write the L.fst or L_disambig.fst text file to the temporary directory

        Parameters
        ----------
        session: :class:`~sqlalchemy.orm.session.Session`
            Session to use for querying
        dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
            Dictionary for generating L.fst
        silence_disambiguation_symbol: str, optional
            Symbol to use for disambiguating silence for L_disambig.fst
        path: str, optional
            Full path to write L.fst to
        alignment: bool
            Flag for whether the FST will be used to align lattices
        """
        base_ext = ".text_fst"
        disambiguation = False
        if silence_disambiguation_symbol is not None:
            base_ext = ".disambig_text_fst"
            disambiguation = True
        if path is not None:
            path = path.replace(".fst", base_ext)
        else:
            path = os.path.join(dictionary.temp_directory, "lexicon" + base_ext)
        start_state = 0
        non_silence_state = 1  # Also loop state
        silence_state = 2
        next_state = 3
        if silence_disambiguation_symbol is None:
            silence_disambiguation_symbol = "<eps>"

        initial_silence_cost = -1 * math.log(self.initial_silence_probability)
        initial_non_silence_cost = -1 * math.log(1.0 - (self.initial_silence_probability))
        if self.final_silence_correction is None or self.final_non_silence_correction is None:
            final_silence_cost = "0"
            final_non_silence_cost = "0"
        else:
            final_silence_cost = str(-math.log(self.final_silence_correction))
            final_non_silence_cost = str(-math.log(self.final_non_silence_correction))
        base_silence_following_cost = -math.log(self.silence_probability)
        base_non_silence_following_cost = -math.log(1 - self.silence_probability)
        with mfa_open(path, "w") as outf:
            outf.write(
                f"{start_state}\t{non_silence_state}\t{silence_disambiguation_symbol}\t{self.silence_word}\t{initial_non_silence_cost}\n"
            )  # initial no silence

            outf.write(
                f"{start_state}\t{silence_state}\t{self.optional_silence_phone}\t{self.silence_word}\t{initial_silence_cost}\n"
            )  # initial silence
            silence_query = (
                session.query(Word.word)
                .filter(Word.word_type == WordType.silence)
                .filter(Word.dictionary_id == dictionary.id)
            )
            for (word,) in silence_query:
                if word == self.silence_word:
                    continue
                outf.write(
                    f"{non_silence_state}\t{non_silence_state}\t{self.optional_silence_phone}\t{word}\t0.0\n"
                )
                outf.write(
                    f"{start_state}\t{non_silence_state}\t{self.optional_silence_phone}\t{word}\t0.0\n"
                )  # initial silence
            columns = [
                Word.word,
                Pronunciation.pronunciation,
                Pronunciation.probability,
                Pronunciation.silence_after_probability,
                Pronunciation.silence_before_correction,
                Pronunciation.non_silence_before_correction,
            ]

            if disambiguation:
                columns.append(Pronunciation.disambiguation)
            bn = DictBundle("pronunciation_data", *columns)
            pronunciation_query = (
                session.query(bn)
                .join(Pronunciation.word)
                .filter(Word.dictionary_id == dictionary.id)
                .filter(Word.word_type != WordType.oov)
                .filter(Word.count > 0)
                .filter(Word.word_type != WordType.silence)
            )
            for row in pronunciation_query:
                data = row.pronunciation_data
                phones = data["pronunciation"].split()
                probability = data["probability"]
                silence_before_cost = 0.0
                non_silence_before_cost = 0.0
                silence_following_cost = base_silence_following_cost
                non_silence_following_cost = base_non_silence_following_cost

                silence_after_probability = data.get("silence_after_probability", None)
                if silence_after_probability is not None:
                    silence_following_cost = -math.log(silence_after_probability)
                    non_silence_following_cost = -math.log(1 - (silence_after_probability))

                silence_before_correction = data.get("silence_before_correction", None)
                if silence_before_correction is not None:
                    silence_before_cost = -math.log(silence_before_correction)

                non_silence_before_correction = data.get("non_silence_before_correction", None)
                if non_silence_before_correction is not None:
                    non_silence_before_cost = -math.log(non_silence_before_correction)
                if self.position_dependent_phones:
                    if len(phones) == 1:
                        phones[0] += "_S"
                    else:
                        phones[0] += "_B"
                        phones[-1] += "_E"
                        for i in range(1, len(phones) - 1):
                            phones[i] += "_I"
                if probability is None:
                    probability = 1
                elif probability < 0.01:
                    probability = 0.01  # Dithering to ensure low probability entries
                pron_cost = abs(math.log(probability))
                if disambiguation and data["disambiguation"] is not None:
                    phones += [f"#{data['disambiguation']}"]
                if alignment:
                    phones = ["#1"] + phones + ["#2"]
                new_state = next_state
                outf.write(
                    f"{non_silence_state}\t{new_state}\t{phones[0]}\t{data['word']}\t{pron_cost+non_silence_before_cost}\n"
                )
                outf.write(
                    f"{silence_state}\t{new_state}\t{phones[0]}\t{data['word']}\t{pron_cost+silence_before_cost}\n"
                )

                next_state += 1
                current_state = new_state
                for i in range(1, len(phones)):
                    new_state = next_state
                    next_state += 1
                    outf.write(f"{current_state}\t{new_state}\t{phones[i]}\t<eps>\n")
                    current_state = new_state
                outf.write(
                    f"{current_state}\t{non_silence_state}\t{silence_disambiguation_symbol}\t<eps>\t{non_silence_following_cost}\n"
                )
                outf.write(
                    f"{current_state}\t{silence_state}\t{self.optional_silence_phone}\t<eps>\t{silence_following_cost}\n"
                )
            oov_pron = (
                session.query(Pronunciation)
                .join(Pronunciation.word)
                .filter(Word.word == self.oov_word)
                .first()
            )
            if not disambiguation:
                oovs = (
                    session.query(Word.word)
                    .filter(Word.word_type == WordType.oov, Word.dictionary_id == dictionary.id)
                    .filter(sqlalchemy.or_(Word.count > 0, Word.word.in_(self.specials_set)))
                )
            else:
                oovs = session.query(Word.word).filter(Word.word == self.oov_word)

            phones = [self.oov_phone]
            if self.position_dependent_phones:
                phones[0] += "_S"
            if alignment:
                phones = ["#1"] + phones + ["#2"]
            for (w,) in oovs:
                silence_before_cost = 0.0
                non_silence_before_cost = 0.0
                silence_following_cost = base_silence_following_cost
                non_silence_following_cost = base_non_silence_following_cost

                silence_after_probability = oov_pron.silence_after_probability
                if silence_after_probability is not None:
                    silence_following_cost = -math.log(silence_after_probability)
                    non_silence_following_cost = -math.log(1 - (silence_after_probability))

                silence_before_correction = oov_pron.silence_before_correction
                if silence_before_correction is not None:
                    silence_before_cost = -math.log(silence_before_correction)

                non_silence_before_correction = oov_pron.non_silence_before_correction
                if non_silence_before_correction is not None:
                    non_silence_before_cost = -math.log(non_silence_before_correction)
                pron_cost = 0.0
                new_state = next_state
                outf.write(
                    f"{non_silence_state}\t{new_state}\t{phones[0]}\t{w}\t{pron_cost+non_silence_before_cost}\n"
                )
                outf.write(
                    f"{silence_state}\t{new_state}\t{phones[0]}\t{w}\t{pron_cost+silence_before_cost}\n"
                )

                next_state += 1
                current_state = new_state
                for i in range(1, len(phones)):
                    new_state = next_state
                    next_state += 1
                    outf.write(f"{current_state}\t{new_state}\t{phones[i]}\t<eps>\n")
                    current_state = new_state
                outf.write(
                    f"{current_state}\t{non_silence_state}\t{silence_disambiguation_symbol}\t<eps>\t{non_silence_following_cost}\n"
                )
                outf.write(
                    f"{current_state}\t{silence_state}\t{self.optional_silence_phone}\t<eps>\t{silence_following_cost}\n"
                )

            outf.write(f"{silence_state}\t{final_silence_cost}\n")
            outf.write(f"{non_silence_state}\t{final_non_silence_cost}\n")

    def _write_align_lexicon(
        self,
        session: sqlalchemy.orm.Session,
        dictionary: Dictionary,
        silence_disambiguation_symbol=None,
    ) -> None:
        """
        Write an alignment FST for use by :kaldi_src:`phones-to-prons` to extract pronunciations

        Parameters
        ----------
        session: sqlalchemy.orm.Session
            Database session
        dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
            Dictionary object for align lexicon
        """
        fst = pynini.Fst()
        phone_symbol_table = pywrapfst.SymbolTable.read_text(self.phone_symbol_table_path)
        word_symbol_table = pywrapfst.SymbolTable.read_text(dictionary.words_symbol_path)
        start_state = fst.add_state()
        loop_state = fst.add_state()
        sil_state = fst.add_state()
        next_state = fst.add_state()
        fst.set_start(start_state)
        silence_words = {self.silence_word}
        silence_query = (
            session.query(Word.word)
            .filter(Word.word_type == WordType.silence)
            .filter(Word.dictionary_id == dictionary.id)
        )
        for (word,) in silence_query:
            silence_words.add(word)
        word_eps_symbol = word_symbol_table.find("<eps>")
        phone_eps_symbol = phone_symbol_table.find("<eps>")
        sil_cost = -math.log(0.5)
        non_sil_cost = sil_cost
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                phone_eps_symbol,
                word_eps_symbol,
                pywrapfst.Weight(fst.weight_type(), non_sil_cost),
                loop_state,
            ),
        )
        fst.add_arc(
            start_state,
            pywrapfst.Arc(
                phone_eps_symbol,
                word_eps_symbol,
                pywrapfst.Weight(fst.weight_type(), sil_cost),
                sil_state,
            ),
        )
        for silence in silence_words:
            w_s = word_symbol_table.find(silence)
            fst.add_arc(
                sil_state,
                pywrapfst.Arc(
                    phone_symbol_table.find(self.optional_silence_phone),
                    w_s,
                    pywrapfst.Weight.one(fst.weight_type()),
                    loop_state,
                ),
            )

        oovs = session.query(Word.word).filter(
            Word.word_type == WordType.oov,
            sqlalchemy.or_(Word.count > 0, Word.word.in_(self.specials_set)),
        )
        for (w,) in oovs:
            pron = [self.oov_phone]
            if self.position_dependent_phones:
                pron[0] += "_S"
            pron = ["#1"] + pron + ["#2"]
            current_state = loop_state
            for i in range(len(pron) - 1):
                p_s = phone_symbol_table.find(pron[i])
                if i == 0:
                    w_s = word_symbol_table.find(w)
                else:
                    w_s = word_eps_symbol
                fst.add_arc(
                    current_state,
                    pywrapfst.Arc(p_s, w_s, pywrapfst.Weight.one(fst.weight_type()), next_state),
                )
                current_state = next_state
                next_state = fst.add_state()
            i = len(pron) - 1
            if i >= 0:
                p_s = phone_symbol_table.find(pron[i])
            else:
                p_s = phone_eps_symbol
            if i <= 0:
                w_s = word_symbol_table.find(w)
            else:
                w_s = word_eps_symbol
            fst.add_arc(
                current_state,
                pywrapfst.Arc(
                    p_s, w_s, pywrapfst.Weight(fst.weight_type(), non_sil_cost), loop_state
                ),
            )
            fst.add_arc(
                current_state,
                pywrapfst.Arc(p_s, w_s, pywrapfst.Weight(fst.weight_type(), sil_cost), sil_state),
            )
        pronunciation_query = (
            session.query(Word.word, Pronunciation.pronunciation)
            .join(Pronunciation.word)
            .filter(Word.dictionary_id == dictionary.id)
            .filter(
                Word.word_type != WordType.silence, Word.word_type != WordType.oov, Word.count > 0
            )
        )
        for w, pron in pronunciation_query:
            pron = pron.split()
            if self.position_dependent_phones:
                if pron[0] != self.optional_silence_phone:
                    if len(pron) == 1:
                        pron[0] += "_S"
                    else:
                        pron[0] += "_B"
                        pron[-1] += "_E"
                        for i in range(1, len(pron) - 1):
                            pron[i] += "_I"
            pron = ["#1"] + pron + ["#2"]
            current_state = loop_state
            for i in range(len(pron) - 1):
                p_s = phone_symbol_table.find(pron[i])
                if i == 0:
                    w_s = word_symbol_table.find(w)
                else:
                    w_s = word_eps_symbol
                fst.add_arc(
                    current_state,
                    pywrapfst.Arc(p_s, w_s, pywrapfst.Weight.one(fst.weight_type()), next_state),
                )
                current_state = next_state
                next_state = fst.add_state()
            i = len(pron) - 1
            if i >= 0:
                p_s = phone_symbol_table.find(pron[i])
            else:
                p_s = phone_eps_symbol
            if i <= 0:
                w_s = word_symbol_table.find(w)
            else:
                w_s = word_eps_symbol
            fst.add_arc(
                current_state,
                pywrapfst.Arc(
                    p_s, w_s, pywrapfst.Weight(fst.weight_type(), non_sil_cost), loop_state
                ),
            )
            fst.add_arc(
                current_state,
                pywrapfst.Arc(p_s, w_s, pywrapfst.Weight(fst.weight_type(), sil_cost), sil_state),
            )
        fst.delete_states([next_state])
        fst.set_final(loop_state, pywrapfst.Weight.one(fst.weight_type()))
        fst.arcsort("olabel")
        fst.write(dictionary.align_lexicon_path)

        with mfa_open(dictionary.align_lexicon_int_path, "w") as f:
            pronunciation_query = (
                session.query(Word.mapping_id, Pronunciation.pronunciation)
                .join(Pronunciation.word)
                .filter(Word.dictionary_id == dictionary.id)
                .order_by(Word.mapping_id)
            )
            for m_id, pron in pronunciation_query:
                pron = pron.split()
                if self.position_dependent_phones:
                    if pron[0] != self.optional_silence_phone:
                        if len(pron) == 1:
                            pron[0] += "_S"
                        else:
                            pron[0] += "_B"
                            pron[-1] += "_E"
                            for i in range(1, len(pron) - 1):
                                pron[i] += "_I"

                pron_string = " ".join(map(str, [self.phone_mapping[x] for x in pron]))
                f.write(f"{m_id} {m_id} {pron_string}\n")

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
                    yaml.safe_dump(dict(dialectal_rules), f, allow_unicode=True)

    def export_lexicon(
        self,
        dictionary_id: int,
        path: str,
        write_disambiguation: typing.Optional[bool] = False,
        probability: typing.Optional[bool] = False,
    ) -> None:
        """
        Export pronunciation dictionary to a text file

        Parameters
        ----------
        path: str
            Path to save dictionary
        write_disambiguation: bool, optional
            Flag for whether to include disambiguation information
        probability: bool, optional
            Flag for whether to include probabilities
        silence_probabilities: bool, optional
            Flag for whether to include per pronunciation silence probabilities, only valid
            when ``probability`` is set to True
        """
        with mfa_open(path, "w") as f, self.session() as session:
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
                    Word.word_type.in_([WordType.speech, WordType.clitic]),
                )
                .order_by(Word.word)
            )
            for row in pronunciations:
                data = row.pronunciation_data
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

    def _write_fst_binary(
        self,
        dictionary: Dictionary,
        write_disambiguation: bool = False,
        path: typing.Optional[str] = None,
    ) -> None:
        """
        Write the binary fst file to the temporary directory

        See Also
        --------
        :kaldi_src:`fstaddselfloops`
            Relevant Kaldi binary
        :openfst_src:`fstcompile`
            Relevant OpenFst binary
        :openfst_src:`fstarcsort`
            Relevant OpenFst binary

        Parameters
        ----------
        dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
            Dictionary object
        write_disambiguation: bool, optional
            Flag for including disambiguation symbols
        path: str, optional
            Full path to write compiled L.fst to
        """
        text_ext = ".text_fst"
        binary_ext = ".fst"
        word_disambig_path = os.path.join(dictionary.temp_directory, "word_disambig.txt")
        with mfa_open(word_disambig_path, "w") as f:
            f.write(str(self.word_mapping(dictionary.id)["#0"]))
        if write_disambiguation:
            text_ext = ".disambig_text_fst"
            binary_ext = ".disambig_fst"
        if path is not None:
            text_path = path.replace(".fst", text_ext)
            binary_path = path.replace(".fst", binary_ext)
        else:
            text_path = os.path.join(dictionary.temp_directory, "lexicon" + text_ext)
            binary_path = os.path.join(dictionary.temp_directory, "L" + binary_ext)

        words_file_path = os.path.join(dictionary.temp_directory, "words.txt")

        log_path = os.path.join(dictionary.temp_directory, os.path.basename(binary_path) + ".log")
        with mfa_open(log_path, "w") as log_file:
            log_file.write(f"Phone isymbols: {self.phone_symbol_table_path}\n")
            log_file.write(f"Word osymbols: {words_file_path}\n")
            com = [
                thirdparty_binary("fstcompile"),
                f"--isymbols={self.phone_symbol_table_path}",
                f"--osymbols={words_file_path}",
                "--keep_isymbols=false",
                "--keep_osymbols=false",
                "--keep_state_numbering=true",
                text_path,
            ]
            log_file.write(f"{' '.join(com)}\n")
            log_file.flush()
            compile_proc = subprocess.Popen(
                com,
                stderr=log_file,
                stdout=subprocess.PIPE,
            )
            if write_disambiguation:
                com = [
                    thirdparty_binary("fstaddselfloops"),
                    self.phone_disambig_path,
                    word_disambig_path,
                ]
                log_file.write(f"{' '.join(com)}\n")
                log_file.flush()
                selfloop_proc = subprocess.Popen(
                    com,
                    stdin=compile_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                )
                com = [
                    thirdparty_binary("fstarcsort"),
                    "--sort_type=olabel",
                    "-",
                    binary_path,
                ]
                log_file.write(f"{' '.join(com)}\n")
                log_file.flush()
                arc_sort_proc = subprocess.Popen(
                    com,
                    stdin=selfloop_proc.stdout,
                    stderr=log_file,
                )
            else:
                com = [
                    thirdparty_binary("fstarcsort"),
                    "--sort_type=olabel",
                    "-",
                    binary_path,
                ]
                log_file.write(f"{' '.join(com)}\n")
                log_file.flush()
                arc_sort_proc = subprocess.Popen(
                    com,
                    stdin=compile_proc.stdout,
                    stderr=log_file,
                )
            arc_sort_proc.communicate()
            if arc_sort_proc.returncode != 0:
                raise KaldiProcessingError([log_path])

    def actual_words(
        self, session: sqlalchemy.orm.session.Session, dictionary_id=None
    ) -> typing.Dict[str, Word]:
        """Words in the dictionary stripping out Kaldi's internal words"""
        words = (
            session.query(Word)
            .options(selectinload(Word.pronunciations))
            .filter(Word.word_type == WordType.speech)
        )
        if dictionary_id is not None:
            words = words.filter(Word.dictionary_id == dictionary_id)
        return {x.word: x for x in words}

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

    def lookup_grapheme(self, grapheme: str) -> int:
        """
        Look up grapheme in the dictionary's mapping

        Parameters
        ----------
        grapheme: str
            Grapheme

        Returns
        -------
        int
            Integer ID for the grapheme
        """
        if grapheme in self.grapheme_mapping:
            return self.grapheme_mapping[grapheme]
        return self.grapheme_mapping[self.oov_word]

    @property
    def reversed_grapheme_mapping(self) -> Dict[int, str]:
        """
        A mapping of integer ids to graphemes
        """
        mapping = {}
        for k, v in self.grapheme_mapping.items():
            mapping[v] = k
        return mapping

    def _write_phone_symbol_table(self) -> None:
        """
        Write the phone mapping to the temporary directory
        """
        with mfa_open(self.phone_symbol_table_path, "w") as f:
            for p, i in self.phone_mapping.items():
                f.write(f"{p} {i}\n")

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
        logger.info("Finding all cutoffs...")
        with self.session() as session:
            c = session.query(Corpus).first()
            if c.cutoffs_found:
                return
            initial_brackets = re.escape("".join(x[0] for x in self.brackets))
            final_brackets = re.escape("".join(x[1] for x in self.brackets))
            pronunciation_mapping = {}
            word_mapping = {}
            max_ids = collections.defaultdict(int)
            max_pron_id = session.query(sqlalchemy.func.max(Pronunciation.id)).scalar()
            max_word_id = session.query(sqlalchemy.func.max(Word.id)).scalar()
            for d_id, max_id in (
                session.query(Dictionary.id, sqlalchemy.func.max(Word.mapping_id))
                .join(Word.dictionary)
                .group_by(Dictionary.id)
            ):
                max_ids[d_id] = max_id
            for d_id in self.dictionary_lookup.values():
                pronunciation_mapping[d_id] = collections.defaultdict(list)
                word_mapping[d_id] = {}
                words = (
                    session.query(Word.mapping_id, Word.word, Pronunciation.pronunciation)
                    .join(Pronunciation.word)
                    .filter(Word.dictionary_id == d_id)
                )
                for m_id, w, pron in words:
                    pronunciation_mapping[d_id][w].append(pron)
                    word_mapping[d_id][w] = m_id
            new_word_mapping = []
            new_pronunciation_mapping = []
            utterances = (
                session.query(
                    Utterance.id,
                    Speaker.dictionary_id,
                    Utterance.normalized_text,
                )
                .join(Utterance.speaker)
                .filter(
                    Utterance.normalized_text.regexp_match(f"[{initial_brackets}](cutoff|hes)")
                )
            )
            utterance_mapping = []
            for u_id, dict_id, normalized_text in utterances:
                text = normalized_text.split()
                modified = False
                for i, word in enumerate(text):
                    m = re.match(
                        f"^[{initial_brackets}](cutoff|hes)([-_](?P<word>[^{final_brackets}]))?[{final_brackets}]$",
                        word,
                    )
                    if not m:
                        continue
                    next_word = None
                    try:
                        next_word = m.group("word")
                        if next_word not in word_mapping[dict_id]:
                            raise ValueError
                    except Exception:
                        if i != len(text) - 1:
                            next_word = text[i + 1]
                    if (
                        next_word is None
                        or next_word not in pronunciation_mapping[dict_id]
                        or self.oov_phone in pronunciation_mapping[dict_id][next_word]
                        or self.optional_silence_phone in pronunciation_mapping[dict_id][next_word]
                    ):
                        continue
                    new_word = f"<cutoff-{next_word}>"
                    if new_word not in word_mapping[dict_id]:
                        max_word_id += 1
                        max_ids[dict_id] += 1
                        max_pron_id += 1
                        new_pronunciation_mapping.append(
                            {
                                "id": max_pron_id,
                                "base_pronunciation_id": max_pron_id,
                                "pronunciation": self.oov_phone,
                                "word_id": max_word_id,
                            }
                        )
                        prons = pronunciation_mapping[dict_id][next_word]
                        pronunciation_mapping[dict_id][new_word] = []
                        for p in prons:
                            p = p.split()
                            for pi in range(len(p)):
                                new_p = " ".join(p[: pi + 1])
                                if new_p in pronunciation_mapping[dict_id][new_word]:
                                    continue
                                pronunciation_mapping[dict_id][new_word].append(new_p)
                                max_pron_id += 1
                                new_pronunciation_mapping.append(
                                    {
                                        "id": max_pron_id,
                                        "pronunciation": new_p,
                                        "word_id": max_word_id,
                                        "base_pronunciation_id": max_pron_id,
                                    }
                                )
                        new_word_mapping.append(
                            {
                                "id": max_word_id,
                                "word": new_word,
                                "dictionary_id": dict_id,
                                "mapping_id": max_ids[dict_id],
                                "word_type": WordType.cutoff,
                            }
                        )
                        word_mapping[dict_id][new_word] = max_ids[dict_id]
                    text[i] = new_word
                    modified = True
                if modified:
                    utterance_mapping.append(
                        {
                            "id": u_id,
                            "normalized_text": " ".join(text),
                        }
                    )
            session.bulk_insert_mappings(
                Word, new_word_mapping, return_defaults=False, render_nulls=True
            )
            session.bulk_insert_mappings(
                Pronunciation, new_pronunciation_mapping, return_defaults=False, render_nulls=True
            )
            bulk_update(session, Utterance, utterance_mapping)
            session.query(Corpus).update({"cutoffs_found": True})
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
        self._write_phone_symbol_table()
        self._write_grapheme_symbol_table()
        self._write_disambig()
        if self.position_dependent_phones:
            self._write_word_boundaries()
        if self.use_g2p:
            return
        silence_disambiguation_symbol = None
        if write_disambiguation:
            silence_disambiguation_symbol = self.silence_disambiguation_symbol

        with self.session() as session:
            dictionaries: typing.List[Dictionary] = session.query(Dictionary)
            for d in dictionaries:
                os.makedirs(d.temp_directory, exist_ok=True)
                self._write_word_file(d)
                self._write_probabilistic_fst_text(session, d, silence_disambiguation_symbol)
                self._write_fst_binary(
                    d, write_disambiguation=silence_disambiguation_symbol is not None
                )
                self._write_align_lexicon(session, d, silence_disambiguation_symbol)

    def write_training_information(self) -> None:
        """Write phone information needed for training"""
        self._write_topo()
        self._write_phone_sets()
        self._write_extra_questions()

    def _write_word_file(self, dictionary: Dictionary) -> None:
        """
        Write the word mapping to the temporary directory
        """
        self._words_mappings = {}
        with mfa_open(dictionary.words_symbol_path, "w") as f, self.session() as session:
            words = (
                session.query(Word.word, Word.mapping_id)
                .filter(Word.dictionary_id == dictionary.id)
                .order_by(Word.mapping_id)
            )
            for w, i in words:
                f.write(f"{w} {i}\n")


class MultispeakerDictionary(MultispeakerDictionaryMixin):
    """
    Class for processing multi- and single-speaker pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.dictionary.multispeaker.MultispeakerDictionaryMixin`
        For dictionary parsing parameters
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
        return os.path.join(GLOBAL_CONFIG.temporary_directory, self.identifier)
