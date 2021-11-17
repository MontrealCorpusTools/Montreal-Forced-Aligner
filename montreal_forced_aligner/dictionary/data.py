"""Pronunciation dictionaries for use in alignment and transcription"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from ..data import CtmInterval

if TYPE_CHECKING:
    IpaType = Optional[List[str]]
    PunctuationType = Optional[str]
    from ..abc import DictionaryEntryType, MappingType, ReversedMappingType, WordsType
    from ..config.dictionary_config import DictionaryConfig
    from ..data import CtmType

__all__ = [
    "DictionaryData",
]


@dataclass
class DictionaryData:
    """
    Information required for parsing Kaldi-internal ids to text
    """

    dictionary_config: DictionaryConfig
    words_mapping: MappingType
    reversed_words_mapping: ReversedMappingType
    reversed_phone_mapping: ReversedMappingType
    words: WordsType

    @property
    def oov_int(self):
        return self.words_mapping[self.dictionary_config.oov_word]

    def split_clitics(
        self,
        item: str,
    ) -> List[str]:
        """
        Split a word into subwords based on dictionary information

        Parameters
        ----------
        item: str
            Word to split

        Returns
        -------
        List[str]
            List of subwords
        """
        if item in self.words:
            return [item]
        if any(x in item for x in self.dictionary_config.compound_markers):
            s = re.split(rf"[{''.join(self.dictionary_config.compound_markers)}]", item)
            if any(x in item for x in self.dictionary_config.clitic_markers):
                new_s = []
                for seg in s:
                    if any(x in seg for x in self.dictionary_config.clitic_markers):
                        new_s.extend(self.split_clitics(seg))
                    else:
                        new_s.append(seg)
                s = new_s
            return s
        if any(
            x in item and not item.endswith(x) and not item.startswith(x)
            for x in self.dictionary_config.clitic_markers
        ):
            initial, final = re.split(
                rf"[{''.join(self.dictionary_config.clitic_markers)}]", item, maxsplit=1
            )
            if any(x in final for x in self.dictionary_config.clitic_markers):
                final = self.split_clitics(final)
            else:
                final = [final]
            for clitic in self.dictionary_config.clitic_markers:
                if initial + clitic in self.dictionary_config.clitic_set:
                    return [initial + clitic] + final
                elif clitic + final[0] in self.dictionary_config.clitic_set:
                    final[0] = clitic + final[0]
                    return [initial] + final
        return [item]

    def lookup(
        self,
        item: str,
    ) -> List[str]:
        """
        Look up a word and return the list of sub words if necessary
        taking into account clitic and compound markers

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        List[str]
            List of subwords that are in the dictionary
        """

        if item in self.words:
            return [item]
        sanitized = self.dictionary_config.sanitize(item)
        if sanitized in self.words:
            return [sanitized]
        split = self.split_clitics(sanitized)
        oov_count = sum(1 for x in split if x not in self.words)

        if oov_count < len(
            split
        ):  # Only returned split item if it gains us any transcribed speech
            return split
        return [sanitized]

    def to_int(
        self,
        item: str,
    ) -> List[int]:
        """
        Convert a given word into integer IDs

        Parameters
        ----------
        item: str
            Word to look up

        Returns
        -------
        List[int]
            List of integer IDs corresponding to each subword
        """
        if item == "":
            return []
        sanitized = self.lookup(item)
        text_int = []
        for item in sanitized:
            if not item:
                continue
            if item not in self.words_mapping:
                text_int.append(self.oov_int)
            else:
                text_int.append(self.words_mapping[item])
        return text_int

    def check_word(self, item: str) -> bool:
        """
        Check whether a word is in the dictionary, takes into account sanitization and
        clitic and compound markers

        Parameters
        ----------
        item: str
            Word to check

        Returns
        -------
        bool
            True if the look up would not result in an OOV item
        """
        if item == "":
            return False
        if item in self.words:
            return True
        sanitized = self.dictionary_config.sanitize(item)
        if sanitized in self.words:
            return True

        sanitized = self.split_clitics(sanitized)
        if all(s in self.words for s in sanitized):
            return True
        return False

    def map_to_original_pronunciation(
        self, phones: CtmType, subpronunciations: List[DictionaryEntryType]
    ) -> CtmType:
        """
        Convert phone transcriptions from multilingual IPA mode to their original IPA transcription

        Parameters
        ----------
        phones: List[CtmInterval]
            List of aligned phones
        subpronunciations: List[DictionaryEntryType]
            Pronunciations of each sub word to reconstruct the transcriptions

        Returns
        -------
        List[CtmInterval]
            Intervals with their original IPA pronunciation rather than the internal simplified form
        """
        transcription = tuple(x.label for x in phones)
        new_phones = []
        mapping_ind = 0
        transcription_ind = 0
        for pronunciations in subpronunciations:
            pron = None
            if mapping_ind >= len(phones):
                break
            for p in pronunciations:
                if (
                    "original_pronunciation" in p
                    and transcription == p["pronunciation"] == p["original_pronunciation"]
                ) or (transcription == p["pronunciation"] and "original_pronunciation" not in p):
                    new_phones.extend(phones)
                    mapping_ind += len(phones)
                    break
                if (
                    p["pronunciation"]
                    == transcription[
                        transcription_ind : transcription_ind + len(p["pronunciation"])
                    ]
                    and pron is None
                ):
                    pron = p
            if mapping_ind >= len(phones):
                break
            if not pron:
                new_phones.extend(phones)
                mapping_ind += len(phones)
                break
            to_extend = phones[transcription_ind : transcription_ind + len(pron["pronunciation"])]
            transcription_ind += len(pron["pronunciation"])
            p = pron
            if (
                "original_pronunciation" not in p
                or p["pronunciation"] == p["original_pronunciation"]
            ):
                new_phones.extend(to_extend)
                mapping_ind += len(to_extend)
                break
            for pi in p["original_pronunciation"]:
                if pi == phones[mapping_ind].label:
                    new_phones.append(phones[mapping_ind])
                else:
                    modded_phone = pi
                    new_p = phones[mapping_ind].label
                    for diacritic in self.dictionary_config.strip_diacritics:
                        modded_phone = modded_phone.replace(diacritic, "")
                    if modded_phone == new_p:
                        phones[mapping_ind].label = pi
                        new_phones.append(phones[mapping_ind])
                    elif mapping_ind != len(phones) - 1:
                        new_p = phones[mapping_ind].label + phones[mapping_ind + 1].label
                        if modded_phone == new_p:
                            new_phones.append(
                                CtmInterval(
                                    phones[mapping_ind].begin,
                                    phones[mapping_ind + 1].end,
                                    new_p,
                                    phones[mapping_ind].utterance,
                                )
                            )
                            mapping_ind += 1
                mapping_ind += 1
        return new_phones
