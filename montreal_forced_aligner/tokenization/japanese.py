from __future__ import annotations

import pathlib
import re

try:
    import sudachipy

    JA_AVAILABLE = True
except ImportError:
    JA_AVAILABLE = False
    sudachipy = None


class JapaneseTokenizer:
    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case
        resource_dir = pathlib.Path(__file__).parent.joinpath("resources")
        config_path = str(resource_dir.joinpath("japanese", "sudachi_config.json"))
        try:
            self.tokenizer = sudachipy.Dictionary(dict="full", config_path=config_path).create(
                mode=sudachipy.SplitMode.B
            )
        except ModuleNotFoundError:
            try:
                self.tokenizer = sudachipy.Dictionary(dict="core", config_path=config_path).create(
                    mode=sudachipy.SplitMode.B
                )
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install a Japanese dictionary via `conda install -c conda-forge sudachidict-core` or install `sudachidict-full`, `sudachidict-core`, or `sudachidict-small` from pip"
                )
        self.morphemes = self.tokenizer.tokenize("")

    def __call__(self, text):
        self.tokenizer.tokenize(text, out=self.morphemes)
        new_text = []
        pronunciations = []
        conjugations = []
        verb_types = []
        dictionary_words = []
        original_pos_tags = []
        pos_tags = []
        for morph in self.morphemes:
            normalized = morph.surface()
            if (
                morph.part_of_speech()[0] == "補助記号"
                and normalized
                and not re.match(r"[-_<({\[>)}\]]+", normalized)
            ):
                continue
            pos = morph.part_of_speech()[4]
            pronunciation = ""
            if morph.part_of_speech()[0] != "補助記号":
                pronunciation = morph.reading_form()
            conjugation = morph.part_of_speech()[5]
            verb_type = morph.part_of_speech()[4]
            dictionary_word = morph.normalized_form()
            if pos in {"*", "一般"}:
                pos = morph.part_of_speech()[2]
            if pos in {"*", "一般"}:
                pos = morph.part_of_speech()[1]
            if pos in {"*", "一般"}:
                pos = morph.part_of_speech()[0]
            if pos == "名詞的":
                pos = morph.part_of_speech()[0]
            original_pos_tags.append(morph.part_of_speech())
            join = False
            m = re.search(r"[]})>][<({[]", normalized)
            if new_text and m:
                new_text[-1] += normalized[: m.start() + 1]
                normalized = normalized[m.end() - 1 :]

            elif new_text and re.match(r"^[<({\[].*", new_text[-1]):
                join = True
            elif new_text and re.match(r".*[-_]$", new_text[-1]):
                join = True
            elif new_text and re.match(r".*[>)}\]]$", normalized):
                join = True
            elif new_text and re.match(r"^[-_].*", normalized):
                join = True
            # elif pos_tags and pos_tags[-1] == '接頭辞':
            #    join = True
            # elif pos_tags and main_pos == '接尾辞':
            #    join = True
            # elif pos_tags and pos_tags[-1].split('+')[-1] == '数詞' and pos == '数詞' and normalized in {
            #    '十', '千', '万', '百', '億', '兆'
            # } and new_text[-1][-1] not in {'十', '千', '万', '百', '億', '兆'}:
            #    join = True
            elif pos_tags and pos_tags[-1].split("+")[-1] == "数詞" and pos in {"助数詞", "助数詞可能"}:
                join = True
            elif (
                conjugations
                and conjugations[-1].split("+")[-1].split("-")[0] in {"語幹", "連用形"}
                and normalized == "そう"
                and pos in {"助動詞語幹"}
            ):
                join = True
            elif (
                conjugations
                and conjugations[-1].split("+")[-1].split("-")[0] in {"連用形"}
                and dictionary_word in {"とる", "とく"}
            ):
                join = True
            elif (
                conjugations
                and conjugations[-1].split("+")[-1] in {"連用形-一般"}
                and normalized == "易い"
                and pos in {"形容詞"}
            ):
                join = True
            elif (
                conjugations
                and conjugations[-1].split("+")[-1] in {"連用形-一般"}
                and normalized == "なさい"
            ):
                join = True
            elif (
                conjugations
                and conjugations[-1].split("+")[-1] in {"連用形-促音便"}
                and dictionary_word == "ちゃう"
                and pos in {"下一段-ア行"}
            ):
                join = True
            elif new_text and new_text[-1] == "す" and normalized == "べき":
                join = True
            elif new_text and normalized == "る" and dictionary_word == "居る":
                join = True
            elif new_text and normalized == "的" and pos == "形状詞的":
                join = True
            elif (
                pos_tags
                and pos_tags[-1] in {"形容詞"}
                and conjugations[-1].split("+")[-1] in {"語幹-一般"}
                and conjugation in {"終止形-一般"}
            ):
                join = True
            elif (
                conjugations
                and conjugations[-1].split("+")[-1] in {"連用形-一般"}
                and dictionary_word in {"がる", "たがる"}
            ):
                join = True
            elif (
                pos_tags
                and pos_tags[-1] in {"五段-バ行", "五段-マ行"}
                and conjugations[-1] in {"連用形-撥音便"}
                and any(normalized.startswith(x) for x in {"だ", "で"})
            ):
                join = True
            elif (
                pos
                in {
                    "助動詞-マス",
                    "助動詞-ナイ",
                    "助動詞-ヘン",
                    "助動詞-ヌ",
                    "助動詞-タ",
                    "助動詞-レル",
                    "助動詞-タイ",
                    "接尾辞",
                }
                and new_text
            ):
                join = True
            elif (
                normalized in {"と", "て"}
                and new_text
                and pos_tags[-1] == "副詞"
                and any(new_text[-1].endswith(x) for x in ["っ", "ッ"])
            ):
                join = True
            elif new_text and new_text[-1] == "で" and normalized in {"は", "も"}:
                join = True
            elif new_text and new_text[-1] == "と" and normalized in {"か"}:
                join = True
            elif (
                conjugations
                and conjugations[-1] in {"連用形-イ音便"}
                and pos == "五段-カ行"
                and dictionary_word in {"とく"}
                and new_text
                and any(new_text[-1].endswith(x) for x in ["っ", "い"])
            ):
                join = True
            elif new_text and pos == "接続助詞" and normalized == "で" and pos_tags[-1].endswith("-ナイ"):
                join = True
            elif (
                conjugations
                and conjugations[-1] in {"意志推量形"}
                and normalized == "か"
                and new_text
                and new_text[-1].endswith("っ")
            ):
                join = True
            elif (
                (
                    pos in {"接続助詞", "文語助動詞-リ", "副助詞"}
                    and normalized
                    in {"ば", "て", "で", "る", "たり", "だり", "たれ", "たれ", "ったら", "たら", "ちゃ"}
                )
                and (
                    new_text
                    and (
                        pos_tags[-1].split("+")[-1]
                        in {
                            "サ行変格",
                            "カ行変格",
                            "形容詞",
                            "助動詞-ナイ",
                            "助動詞-レル",
                            "助動詞-マス",
                            "助動詞-ダ",
                            "助動詞-デス",
                            "サ行変格+下一段-サ行",
                        }
                        or pos_tags[-1].split("+")[-1].split("-")[0] in {"下一段", "上一段", "五段"}
                    )
                )
                and "終止形" not in conjugations[-1]
            ):
                join = True
            elif (
                new_text
                and (
                    pos_tags[-1].split("+")[-1]
                    in {
                        "サ行変格",
                        "カ行変格",
                        # '形容詞',
                        "助動詞-レル",
                        "助動詞-ダ",
                    }
                    or pos_tags[-1].split("+")[-1].split("-")[0] in {"下一段", "上一段", "五段"}
                )
                and (dictionary_word in {"せる", "させる"} or normalized in {"てる", "て", "てて", "たり"})
            ):
                join = True
            elif (
                new_text
                and (
                    pos_tags[-1].split("+")[-1]
                    in {
                        "サ行変格",
                        "カ行変格",
                        "形容詞",
                        "助動詞-レル",
                    }
                    or pos_tags[-1].split("+")[-1].split("-")[0] in {"下一段", "上一段", "五段"}
                )
                and (
                    pos in {"五段-ワア行"} and any(normalized.startswith(x) for x in ["ちま", "ちゃ", "じゃ"])
                )
            ):
                join = True
            if new_text and any(new_text[-1].endswith(x) for x in {">", ")", "}", "]"}):
                join = False
            if join:
                new_text[-1] += normalized
                pronunciations[-1] += pronunciation
                dictionary_words[-1] += dictionary_word
                if pos_tags[-1] != pos:
                    pos_tags[-1] += "+" + pos
                if verb_types[-1] != verb_type:
                    verb_types[-1] += "+" + verb_type
                if conjugations[-1] != conjugation:
                    conjugations[-1] += "+" + conjugation
                continue
            new_text.append(normalized)
            pronunciations.append(pronunciation)
            conjugations.append(conjugation)
            verb_types.append(verb_type)
            dictionary_words.append(dictionary_word)
            pos_tags.append(pos)
        if " " in new_text:
            space_indices = [i for i, x in enumerate(new_text) if x == " "]
            new_text = [x for i, x in enumerate(new_text) if i not in space_indices]
            pronunciations = [x for i, x in enumerate(pronunciations) if i not in space_indices]
        new_text = " ".join(new_text)
        pronunciations = " ".join(pronunciations)
        if self.ignore_case:
            new_text = new_text.lower()
            pronunciations = pronunciations.lower()
        return new_text, pronunciations


def ja_spacy(ignore_case: bool = True):
    if not JA_AVAILABLE:
        raise ImportError(
            "Please install Japanese support via `conda install -c conda-forge spacy sudachipy sudachidict-core`"
        )
    return JapaneseTokenizer(ignore_case)
