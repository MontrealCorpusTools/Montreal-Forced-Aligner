from __future__ import annotations

import os
import subprocess

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

from montreal_forced_aligner.data import Language
from montreal_forced_aligner.tokenization.english import en_spacy
from montreal_forced_aligner.tokenization.japanese import ja_spacy

language_model_mapping = {
    # Use small models optimized for CPU because tokenizer accuracy does not depend on model
    Language.catalan: "ca_core_news_sm",
    Language.chinese: "zh_core_web_sm",
    Language.croatian: "hr_core_news_sm",
    Language.danish: "da_core_news_sm",
    Language.dutch: "nl_core_news_sm",
    Language.english: "en_core_web_sm",
    Language.finnish: "fi_core_news_sm",
    Language.french: "fr_core_news_sm",
    Language.german: "de_core_news_sm",
    Language.greek: "el_core_news_sm",
    Language.italian: "it_core_news_sm",
    Language.japanese: "ja_core_news_sm",
    Language.korean: "ko_core_news_sm",
    Language.lithuanian: "lt_core_news_sm",
    Language.macedonian: "mk_core_news_sm",
    Language.multilingual: "xx_sent_ud_sm",
    Language.norwegian: "nb_core_news_sm",
    Language.polish: "pl_core_news_sm",
    Language.portuguese: "pt_core_news_sm",
    Language.romanian: "ro_core_news_sm",
    Language.russian: "ru_core_news_sm",
    Language.slovenian: "sl_core_news_sm",
    Language.spanish: "es_core_news_sm",
    Language.swedish: "sv_core_news_sm",
    Language.ukrainian: "uk_core_news_sm",
}


def generate_language_tokenizer(language: Language):
    if not SPACY_AVAILABLE:
        raise ImportError("Please install spacy via `conda install spacy`")
    if language is Language.english:
        return en_spacy()
    elif language is Language.japanese:
        return ja_spacy()
    name = language_model_mapping[language]
    try:
        nlp = spacy.load(name)
    except OSError:
        subprocess.call(["python", "-m", "spacy", "download", name], env=os.environ)
        nlp = spacy.load(name)
    return nlp
