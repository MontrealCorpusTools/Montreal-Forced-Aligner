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
from montreal_forced_aligner.tokenization.chinese import ZH_AVAILABLE, zh_spacy
from montreal_forced_aligner.tokenization.english import en_spacy
from montreal_forced_aligner.tokenization.japanese import JA_AVAILABLE, ja_spacy
from montreal_forced_aligner.tokenization.korean import KO_AVAILABLE, ko_spacy
from montreal_forced_aligner.tokenization.thai import TH_AVAILABLE, th_spacy

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


def generate_language_tokenizer(language: Language, ignore_case: bool = True):
    if language is Language.english:
        if not SPACY_AVAILABLE:
            raise ImportError("Please install spacy via `conda install spacy`")
        return en_spacy(ignore_case)
    elif language is Language.japanese:
        return ja_spacy(ignore_case)
    elif language is Language.korean:
        return ko_spacy(ignore_case)
    elif language is Language.chinese:
        return zh_spacy(ignore_case)
    elif language is Language.thai:
        return th_spacy(ignore_case)
    if not SPACY_AVAILABLE:
        raise ImportError("Please install spacy via `conda install spacy`")
    name = language_model_mapping[language]
    nlp = spacy.load(name)
    return nlp


def check_language_tokenizer_availability(language: Language):
    if language is Language.japanese:
        if not JA_AVAILABLE:
            raise ImportError(
                "Please install Japanese support via `conda install -c conda-forge spacy sudachipy sudachidict-core`"
            )
    elif language is Language.korean:
        if not KO_AVAILABLE:
            raise ImportError(
                "Please install Korean support via `pip install python-mecab-ko jamo`"
            )
    elif language is Language.chinese:
        if not ZH_AVAILABLE:
            raise ImportError(
                "Please install Chinese tokenization support via `pip install spacy-pkuseg dragonmapper hanziconv`"
            )
        import spacy_pkuseg

        spacy_pkuseg.pkuseg(postag=True)
    elif language is Language.thai:
        if not TH_AVAILABLE:
            raise ImportError(
                "Please install Thai tokenization support via `pip install pythainlp`"
            )
    else:
        if not SPACY_AVAILABLE:
            raise ImportError("Please install spacy via `conda install spacy`")
        if language not in language_model_mapping:
            raise ImportError(f"Language '{language}' not yet currently supported in spacy.")
        name = language_model_mapping[language]
        try:
            _ = spacy.load(name)
        except OSError:
            subprocess.check_call(["python", "-m", "spacy", "download", name], env=os.environ)
