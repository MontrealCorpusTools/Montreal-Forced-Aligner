"""Classes for calculating alignments online"""
from __future__ import annotations

import typing

import sqlalchemy.orm
from kalpy.aligner import KalpyAligner
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.lexicon import Pronunciation as KalpyPronunciation
from kalpy.gmm.data import HierarchicalCtm
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner.data import Language, WordType
from montreal_forced_aligner.db import (
    Phone,
    PhoneInterval,
    Pronunciation,
    Utterance,
    Word,
    WordInterval,
    get_next_primary_key,
)
from montreal_forced_aligner.models import AcousticModel, G2PModel


def tokenize_utterance_text(
    text,
    lexicon_compiler: LexiconCompiler,
    tokenizer=None,
    g2p_model: G2PModel = None,
    language: Language = Language.unknown,
):
    if tokenizer is None:
        return text.lower()
    if language is Language.unknown:
        normalized_text, _, oovs = tokenizer(text)
        if g2p_model is not None:
            for w in oovs:
                if not lexicon_compiler.word_table.member(w):
                    pron = g2p_model.rewriter(w)
                    if pron:
                        lexicon_compiler.add_pronunciation(
                            KalpyPronunciation(w, pron[0], None, None, None, None, None)
                        )

    else:
        normalized_text, pronunciation_form = tokenizer(text)
        if not pronunciation_form:
            pronunciation_form = text
        g2p_cache = {}
        if g2p_model is not None:
            for norm_w, w in zip(text.split(), pronunciation_form.split()):
                if w not in g2p_cache:
                    pron = g2p_model.rewriter(w)
                    if not pron:
                        continue
                    g2p_cache[w] = pron[0]
                if w in g2p_cache and not lexicon_compiler.word_table.member(norm_w):
                    lexicon_compiler.add_pronunciation(
                        KalpyPronunciation(norm_w, g2p_cache[w], None, None, None, None, None)
                    )
    return normalized_text


def align_utterance_online(
    acoustic_model: AcousticModel,
    utterance: KalpyUtterance,
    lexicon_compiler: LexiconCompiler,
    tokenizer=None,
    g2p_model: G2PModel = None,
    beam: int = 10,
    retry_beam: int = 40,
    transition_scale: float = 1.0,
    acoustic_scale: float = 0.1,
    self_loop_scale: float = 0.1,
    boost_silence: float = 1.0,
    careful: bool = False,
) -> HierarchicalCtm:
    utterance.transcript = tokenize_utterance_text(
        utterance.transcript,
        lexicon_compiler,
        lexicon_compiler,
        tokenizer,
        g2p_model,
        language=acoustic_model.language,
    )
    kalpy_aligner = KalpyAligner(
        acoustic_model,
        lexicon_compiler,
        beam=beam,
        retry_beam=retry_beam,
        transition_scale=transition_scale,
        acoustic_scale=acoustic_scale,
        self_loop_scale=self_loop_scale,
        boost_silence=boost_silence,
        careful=careful,
    )
    ctm = kalpy_aligner.align_utterance(utterance)
    ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
    return ctm


def update_utterance_intervals(
    session: sqlalchemy.orm.Session,
    utterance: typing.Union[int, Utterance],
    ctm: HierarchicalCtm,
):
    if isinstance(utterance, int):
        utterance = session.get(Utterance, utterance)
    max_phone_interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
    if max_phone_interval_id is None:
        max_phone_interval_id = 0
    max_word_interval_id = session.query(sqlalchemy.func.max(WordInterval.id)).scalar()
    if max_word_interval_id is None:
        max_word_interval_id = 0
    mapping_id = session.query(sqlalchemy.func.max(Word.mapping_id)).scalar()
    if mapping_id is None:
        mapping_id = -1
    mapping_id += 1
    word_index = get_next_primary_key(session, Word)
    new_phone_interval_mappings = []
    new_word_interval_mappings = []
    words = (
        session.query(Word.word, Word.id)
        .filter(Word.dictionary_id == utterance.speaker.dictionary_id)
        .filter(Word.word.in_(utterance.normalized_text.split()))
    )
    phone_to_phone_id = {}
    word_mapping = {}
    pronunciation_mapping = {}
    ds = session.query(Phone.id, Phone.mapping_id).all()
    for p_id, mapping_id in ds:
        phone_to_phone_id[mapping_id] = p_id
    new_words = []
    for w, w_id in words:
        word_mapping[w] = w_id
    pronunciations = (
        session.query(Word.word, Pronunciation.pronunciation, Pronunciation.id)
        .join(Pronunciation.word)
        .filter(Word.dictionary_id == utterance.speaker.dictionary_id)
        .filter(Word.word.in_(utterance.normalized_text.split()))
    )
    for w, pron, p_id in pronunciations:
        pronunciation_mapping[(w, pron)] = p_id
    for word_interval in ctm.word_intervals:
        if word_interval.label not in word_mapping:
            new_words.append(
                {
                    "id": word_index,
                    "mapping_id": mapping_id,
                    "word": word_interval.label,
                    "dictionary_id": 1,
                    "word_type": WordType.oov,
                }
            )
            word_mapping[word_interval.label] = word_index
            word_id = word_index
        else:
            word_id = word_mapping[word_interval.label]
        max_word_interval_id += 1
        pronunciation_id = pronunciation_mapping.get(
            (word_interval.label, word_interval.pronunciation), None
        )

        new_word_interval_mappings.append(
            {
                "id": max_word_interval_id,
                "begin": word_interval.begin,
                "end": word_interval.end,
                "word_id": word_id,
                "pronunciation_id": pronunciation_id,
                "utterance_id": utterance.id,
            }
        )
        for interval in word_interval.phones:
            max_phone_interval_id += 1
            new_phone_interval_mappings.append(
                {
                    "id": max_phone_interval_id,
                    "begin": interval.begin,
                    "end": interval.end,
                    "phone_id": phone_to_phone_id[interval.symbol],
                    "utterance_id": utterance.id,
                    "word_interval_id": max_word_interval_id,
                    "phone_goodness": interval.confidence if interval.confidence else 0.0,
                }
            )
    session.query(Utterance).filter(Utterance.id == utterance.id).update(
        {Utterance.alignment_log_likelihood: ctm.likelihood}
    )
    session.query(PhoneInterval).filter(PhoneInterval.utterance_id == utterance.id).delete(
        synchronize_session=False
    )
    session.flush()
    session.query(WordInterval).filter(WordInterval.utterance_id == utterance.id).delete(
        synchronize_session=False
    )
    session.flush()
    if new_words:
        session.bulk_insert_mappings(Word, new_words, return_defaults=False, render_nulls=True)
        session.flush()
    if new_word_interval_mappings:
        session.bulk_insert_mappings(
            WordInterval, new_word_interval_mappings, return_defaults=False, render_nulls=True
        )
        session.flush()
        session.bulk_insert_mappings(
            PhoneInterval,
            new_phone_interval_mappings,
            return_defaults=False,
            render_nulls=True,
        )
    session.commit()
