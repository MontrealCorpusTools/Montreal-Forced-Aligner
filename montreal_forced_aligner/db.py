"""Database classes"""
from __future__ import annotations

import logging
import os
import re
import typing

import librosa
import numpy as np
import sqlalchemy
from pgvector.sqlalchemy import Vector
from praatio import textgrid
from praatio.utilities.constants import Interval
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Bundle, declarative_base, relationship

from montreal_forced_aligner.config import IVECTOR_DIMENSION, PLDA_DIMENSION, XVECTOR_DIMENSION
from montreal_forced_aligner.data import (
    CtmInterval,
    PhoneSetType,
    PhoneType,
    TextFileType,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.helper import mfa_open

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.corpus.classes import UtteranceData

logger = logging.getLogger("mfa")

__all__ = [
    "Corpus",
    "CorpusWorkflow",
    "PhonologicalRule",
    "RuleApplication",
    "DictBundle",
    "Dictionary",
    "Word",
    "Phone",
    "Pronunciation",
    "File",
    "TextFile",
    "SoundFile",
    "Speaker",
    "SpeakerOrdering",
    "Utterance",
    "PhoneInterval",
    "WordInterval",
    "M2MSymbol",
    "Job",
    "Word2Job",
    "M2M2Job",
    "Grapheme",
    "MfaSqlBase",
    "bulk_update",
]

MfaSqlBase = declarative_base()


def bulk_update(
    session: sqlalchemy.orm.Session,
    table: MfaSqlBase,
    values: typing.List[typing.Dict[str, typing.Any]],
    id_field=None,
) -> None:
    """
    Perform a bulk update of a database.

    Parameters
    ----------
    session: :class:`sqlalchemy.orm.Session`
        SqlAlchemy session to use
    table: :class:`~montreal_forced_aligner.db.MfaSqlBase`
        Table to update
    values: list[dict[str, Any]]
        List of field-value dictionaries to insert
    id_field: str, optional
        Optional specifier of the primary key field
    """
    if len(values) == 0:
        return
    if id_field is None:
        id_field = "id"

    column_names = [x for x in values[0].keys()]
    columns = [getattr(table, x)._copy() for x in column_names if x != id_field]
    sql_column_names = [f'"{x}"' for x in column_names if x != id_field]
    session.execute(sqlalchemy.text(f"ALTER TABLE {table.__tablename__} DISABLE TRIGGER all"))
    session.commit()
    with session.begin_nested():
        temp_table = sqlalchemy.Table(
            f"temp_{table.__tablename__}",
            MfaSqlBase.metadata,
            sqlalchemy.Column(id_field, sqlalchemy.Integer, primary_key=True),
            *columns,
            prefixes=["TEMPORARY"],
            extend_existing=True,
        )
        create_statement = str(
            sqlalchemy.schema.CreateTable(temp_table).compile(session.get_bind())
        )
        session.execute(sqlalchemy.text(create_statement))
        session.execute(temp_table.insert(), values)

        set_statements = []
        for c in sql_column_names:
            set_statements.append(f""" {c} = b.{c}""")
        set_statements = ",\n".join(set_statements)
        sql = f"""
        UPDATE {table.__tablename__}
        SET
            {set_statements}
        FROM temp_{table.__tablename__} AS b
        WHERE {table.__tablename__}.{id_field}=b.{id_field};
        """
        session.execute(sqlalchemy.text(sql))

        # drop temp table
        session.execute(sqlalchemy.text(f"DROP TABLE temp_{table.__tablename__}"))
    session.execute(sqlalchemy.text(f"ALTER TABLE {table.__tablename__} ENABLE TRIGGER all"))
    session.commit()
    session.execute(sqlalchemy.text("DISCARD TEMP"))
    MfaSqlBase.metadata.remove(temp_table)


Dictionary2Job = sqlalchemy.Table(
    "dictionary_job",
    MfaSqlBase.metadata,
    Column("dictionary_id", ForeignKey("dictionary.id"), primary_key=True),
    Column("job_id", ForeignKey("job.id"), primary_key=True),
)

SpeakerOrdering = sqlalchemy.Table(
    "speaker_ordering",
    MfaSqlBase.metadata,
    Column("speaker_id", ForeignKey("speaker.id"), primary_key=True),
    Column("file_id", ForeignKey("file.id"), primary_key=True),
    Column("index", Integer, primary_key=True),
)


class DictBundle(Bundle):
    """
    SqlAlchemy custom Bundle class for loading variable column counts
    """

    def create_row_processor(self, query, procs, labels):
        """Override create_row_processor to return values as dictionaries"""

        def proc(row):
            return dict(zip(labels, (proc(row) for proc in procs)))

        return proc


class Corpus(MfaSqlBase):
    """
    Database class for storing information about a corpus

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Corpus name
    imported: bool
        Flag for whether the corpus has been imported
    features_generated: bool
        Flag for whether features have been generated
    alignment_done: bool
        Flag for whether alignment has successfully completed
    transcription_done: bool
        Flag for whether transcription has successfully completed
    alignment_evaluation_done: bool
        Flag for whether alignment evaluation has successfully completed
    has_reference_alignments: bool
        Flag for whether reference alignments have been imported
    """

    __tablename__ = "corpus"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    path = Column(String, unique=True, nullable=False)
    imported = Column(Boolean, default=False)
    text_normalized = Column(Boolean, default=False)
    cutoffs_found = Column(Boolean, default=False)
    features_generated = Column(Boolean, default=False)
    vad_calculated = Column(Boolean, default=False)
    ivectors_calculated = Column(Boolean, default=False)
    plda_calculated = Column(Boolean, default=False)
    xvectors_loaded = Column(Boolean, default=False)
    alignment_done = Column(Boolean, default=False)
    transcription_done = Column(Boolean, default=False)
    alignment_evaluation_done = Column(Boolean, default=False)
    has_reference_alignments = Column(Boolean, default=False)
    has_sound_files = Column(Boolean, default=False)
    has_text_files = Column(Boolean, default=False)
    num_jobs = Column(Integer, default=0)

    current_subset = Column(Integer, default=0)
    data_directory = Column(String, nullable=False)

    jobs = relationship("Job", back_populates="corpus")

    @property
    def split_directory(self):
        return os.path.join(self.data_directory, f"split{self.num_jobs}")

    @property
    def current_subset_directory(self):
        if not self.current_subset:
            return self.split_directory
        return os.path.join(self.data_directory, f"subset_{self.current_subset}")

    @property
    def speaker_ivector_column(self):
        if self.plda_calculated:
            return Speaker.plda_vector
        elif self.xvectors_loaded:
            return Speaker.xvector
        return Speaker.ivector

    @property
    def utterance_ivector_column(self):
        if self.plda_calculated:
            return Utterance.plda_vector
        elif self.xvectors_loaded:
            return Utterance.xvector
        return Utterance.ivector


class Dialect(MfaSqlBase):
    """
    Database class for storing information about a dialect

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Dialect name
    """

    __tablename__ = "dialect"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)

    dictionaries = relationship("Dictionary", back_populates="dialect")
    rules = relationship("PhonologicalRule", back_populates="dialect")


class Dictionary(MfaSqlBase):
    """
    Database class for storing information about a pronunciation dictionary

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Dictionary name
    dialect: str
        Dialect of dictionary if dictionary name is in MFA format
    path: str
        Path to the dictionary
    phone_set_type: :class:`~montreal_forced_aligner.data.PhoneSetType`
        Phone set
    bracket_regex: str
        Regular expression for detecting bracketed words
    laughter_regex: str
        Regular expression for detecting laughter words
    position_dependent_phones: bool
        Flag for whether phones have word-position flags
    default: bool
        Flag for whether this dictionary is the default one
    clitic_marker: str
        Character marking clitics
    silence_word: str
        Symbol for silence
    optional_silence_phone: str
        Symbol for silence phone
    oov_word: str
        Symbol for unknown words
    bracketed_word: str
        Symbol for bracketed words (cutoffs, hesitations, etc)
    laughter_word: str
        Symbol for laughter words
    max_disambiguation_symbol: int
        Highest disambiguation index required
    silence_probability: float
        Probability of inserting non-initial optional silence
    initial_silence_probability: float
        Probability of inserting initial silence
    final_silence_correction: float
        Correction factor on having final silence
    final_non_silence_correction: float
        Correction factor on having final non-silence
    """

    __tablename__ = "dictionary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    path = Column(String, unique=True)
    phone_set_type = Column(Enum(PhoneSetType), nullable=True)
    root_temp_directory = Column(String, nullable=True)
    clitic_cleanup_regex = Column(String, nullable=True)
    bracket_regex = Column(String, nullable=True)
    laughter_regex = Column(String, nullable=True)
    position_dependent_phones = Column(Boolean, nullable=True)
    default = Column(Boolean, default=False, nullable=False)
    clitic_marker = Column(String(1), nullable=True)
    silence_word = Column(String, nullable=True)
    optional_silence_phone = Column(String, nullable=True)
    oov_word = Column(String, nullable=True)
    oov_phone = Column(String, nullable=True)
    bracketed_word = Column(String, nullable=True)
    laughter_word = Column(String, nullable=True)

    use_g2p = Column(Boolean, nullable=False, default=False)
    max_disambiguation_symbol = Column(Integer, default=0, nullable=False)
    silence_probability = Column(Float, default=0.5, nullable=False)
    initial_silence_probability = Column(Float, default=0.5, nullable=False)
    final_silence_correction = Column(Float, nullable=True)
    final_non_silence_correction = Column(Float, nullable=True)

    dialect_id = Column(Integer, ForeignKey("dialect.id"), index=True, nullable=True)
    dialect = relationship("Dialect", back_populates="dictionaries")

    words = relationship(
        "Word",
        back_populates="dictionary",
        order_by="Word.mapping_id",
        collection_class=ordering_list("mapping_id"),
        cascade="all, delete",
    )
    speakers = relationship(
        "Speaker",
        back_populates="dictionary",
        cascade="all, delete-orphan",
    )

    jobs = relationship(
        "Job",
        secondary=Dictionary2Job,
        back_populates="dictionaries",
    )

    @property
    def special_set(self) -> typing.Set[str]:
        return {
            "<s>",
            "</s>",
            self.silence_word,
            self.oov_word,
            self.bracketed_word,
            self.laughter_word,
        }

    @property
    def clitic_set(self) -> typing.Set[str]:
        """Set of clitic words"""
        return {x.word for x in self.words if x.word_type is WordType.clitic}

    @property
    def word_boundary_int_path(self) -> str:
        """Path to the word boundary integer IDs"""
        return os.path.join(self.phones_directory, "word_boundary.int")

    @property
    def disambiguation_symbols_int_path(self) -> str:
        """Path to the word boundary integer IDs"""
        return os.path.join(self.phones_directory, "disambiguation_symbols.int")

    @property
    def phones_directory(self) -> str:
        """
        Phones directory
        """
        return os.path.join(str(self.root_temp_directory), "phones")

    @property
    def phone_symbol_table_path(self) -> str:
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_directory, "phones.txt")

    @property
    def grapheme_symbol_table_path(self) -> str:
        """Path to file containing grapheme symbols and their integer IDs"""
        return os.path.join(self.phones_directory, "graphemes.txt")

    @property
    def phone_disambig_path(self) -> str:
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_directory, "phone_disambig.txt")

    @property
    def temp_directory(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(str(self.root_temp_directory), f"{self.id}_{self.name}")

    @property
    def lexicon_disambig_fst_path(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.temp_directory, "L.disambig_fst")

    @property
    def align_lexicon_path(self) -> str:
        """
        Path of lexicon file to use for aligning lattices
        """
        return os.path.join(self.temp_directory, "align_lexicon.fst")

    @property
    def align_lexicon_disambig_path(self) -> str:
        """
        Path of lexicon file to use for aligning lattices
        """
        return os.path.join(self.temp_directory, "align_lexicon.disambig_fst")

    @property
    def align_lexicon_int_path(self) -> str:
        """
        Path of lexicon file to use for aligning lattices
        """
        return os.path.join(self.temp_directory, "align_lexicon.int")

    @property
    def lexicon_fst_path(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.temp_directory, "L.fst")

    @property
    def words_symbol_path(self) -> str:
        """
        Path of word to int mapping file for the dictionary
        """
        return os.path.join(self.temp_directory, "words.txt")

    @property
    def data_source_identifier(self) -> str:
        """Dictionary name"""
        return f"{self.id}_{self.name}"

    @property
    def identifier(self) -> str:
        """Dictionary name"""
        return f"{self.data_source_identifier}"

    @property
    def silence_probability_info(self) -> typing.Dict[str, float]:
        """Dictionary of silence information"""
        return {
            "silence_probability": self.silence_probability,
            "initial_silence_probability": self.initial_silence_probability,
            "final_silence_correction": self.final_silence_correction,
            "final_non_silence_correction": self.final_non_silence_correction,
        }


class Phone(MfaSqlBase):
    """
    Database class for storing phones and their integer IDs

    Parameters
    ----------
    id: int
        Primary key
    mapping_id: int
        Integer ID of the phone for Kaldi processing
    phone: str
        Phone label
    phone_type: :class:`~montreal_forced_aligner.data.PhoneType`
        Type of phone
    """

    __tablename__ = "phone"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mapping_id = Column(Integer, nullable=False, unique=True)
    phone = Column(String(10), nullable=False)
    kaldi_label = Column(String(10), unique=True, nullable=False)
    position = Column(String(2), nullable=True)
    phone_type = Column(Enum(PhoneType), nullable=False, index=True)

    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="phone",
        order_by="PhoneInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
        passive_deletes=True,
    )


class Grapheme(MfaSqlBase):
    """
    Database class for storing phones and their integer IDs

    Parameters
    ----------
    id: int
        Primary key
    mapping_id: int
        Integer ID of the phone for Kaldi processing
    grapheme: str
        Phone label
    """

    __tablename__ = "grapheme"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mapping_id = Column(Integer, nullable=False, unique=True)
    grapheme = Column(String(25), unique=True, nullable=False)


class Word(MfaSqlBase):
    """
    Database class for storing words, their integer IDs, and pronunciation information

    Parameters
    ----------
    id: int
        Primary key
    mapping_id: int
        Integer ID of the word for Kaldi processing
    word: str
        Word label
    count: int
        Count frequency of word in the corpus
    word_type: :class:`~montreal_forced_aligner.data.WordType`
        Type of word
    dictionary_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Dictionary`
    dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
        Pronunciation dictionary that the word belongs to
    """

    __tablename__ = "word"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mapping_id = Column(Integer, nullable=False, index=True)
    word = Column(String, nullable=False, index=True)
    count = Column(Integer, default=0, nullable=False, index=True)
    word_type = Column(Enum(WordType), nullable=False, index=True)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"), nullable=False, index=True)
    dictionary = relationship("Dictionary", back_populates="words")
    pronunciations = relationship(
        "Pronunciation", back_populates="word", cascade="all, delete", passive_deletes=True
    )

    job = relationship(
        "Word2Job",
        back_populates="word",
        uselist=False,
        cascade="all, delete",
    )
    word_intervals = relationship(
        "WordInterval",
        back_populates="word",
        order_by="WordInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )

    __table_args__ = (
        sqlalchemy.Index("dictionary_word_type_index", "dictionary_id", "word_type"),
        sqlalchemy.Index("word_dictionary_index", "word", "dictionary_id"),
    )


class Pronunciation(MfaSqlBase):
    """
    Database class for storing information about a pronunciation

    Parameters
    ----------
    id: int
        Primary key
    pronunciation: str
        Space-delimited pronunciation
    probability: float
        Probability of the pronunciation
    silence_after_probability: float
        Probability of silence following the pronunciation
    silence_before_correction: float
        Correction factor for silence before the pronunciation
    non_silence_before_correction: float
        Correction factor for non-silence before the pronunciation
    word_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Word`
    word: :class:`~montreal_forced_aligner.db.Word`
        Word for the pronunciation
    """

    __tablename__ = "pronunciation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pronunciation = Column(String, nullable=False)
    probability = Column(Float, nullable=True)
    disambiguation = Column(Integer, nullable=True)
    silence_after_probability = Column(Float, nullable=True)
    silence_before_correction = Column(Float, nullable=True)
    non_silence_before_correction = Column(Float, nullable=True)

    count = Column(Integer, nullable=False, default=0)
    silence_following_count = Column(Integer, nullable=True)
    non_silence_following_count = Column(Integer, nullable=True)

    word_id = Column(
        Integer, ForeignKey("word.id", ondelete="CASCADE"), nullable=False, index=True
    )
    word = relationship("Word", back_populates="pronunciations")

    base_pronunciation_id = Column(
        Integer, ForeignKey("pronunciation.id"), nullable=False, index=True
    )
    variants = relationship(
        "Pronunciation", backref=sqlalchemy.orm.backref("base_pronunciation", remote_side=[id])
    )

    rules = relationship(
        "RuleApplication",
        back_populates="pronunciation",
        cascade="all, delete",
    )

    word_intervals = relationship(
        "WordInterval",
        back_populates="pronunciation",
        order_by="WordInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )


class PhonologicalRule(MfaSqlBase):
    """
    Database class for storing information about a phonological rule

    Parameters
    ----------
    id: int
        Primary key
    segment: str
        Segment to replace
    preceding_context: str
        Context before segment to match
    following_context: str
        Context after segment to match
    replacement: str
        Replacement of segment
    probability: float
        Probability of the rule application
    silence_after_probability: float
        Probability of silence following forms with rule application
    silence_before_correction: float
        Correction factor for silence before forms with rule application
    non_silence_before_correction: float
        Correction factor for non-silence before forms with rule application
    pronunciations: list[:class:`~montreal_forced_aligner.db.RuleApplication`]
        List of rule applications
    """

    __tablename__ = "phonological_rule"

    id = Column(Integer, primary_key=True, autoincrement=True)

    segment = Column(String, nullable=False, index=True)
    preceding_context = Column(String, nullable=False, index=True)
    following_context = Column(String, nullable=False, index=True)
    replacement = Column(String, nullable=False)

    probability = Column(Float, nullable=True)
    silence_after_probability = Column(Float, nullable=True)
    silence_before_correction = Column(Float, nullable=True)
    non_silence_before_correction = Column(Float, nullable=True)

    dialect_id = Column(Integer, ForeignKey("dialect.id"), index=True, nullable=False)
    dialect = relationship("Dialect", back_populates="rules")

    pronunciations = relationship(
        "RuleApplication",
        back_populates="rule",
        cascade="all, delete",
    )

    def __hash__(self):
        return hash(
            (self.segment, self.preceding_context, self.following_context, self.replacement)
        )

    def to_json(self) -> typing.Dict[str, typing.Any]:
        """
        Serializes the rule for export

        Returns
        -------
        dict[str, Any]
            Serialized rule
        """
        return {
            "segment": self.segment,
            "dialect": self.dialect,
            "preceding_context": self.preceding_context,
            "following_context": self.following_context,
            "replacement": self.replacement,
            "probability": self.probability,
            "silence_after_probability": self.silence_after_probability,
            "silence_before_correction": self.silence_before_correction,
            "non_silence_before_correction": self.non_silence_before_correction,
        }

    @property
    def match_regex(self):
        """Regular expression of the rule"""
        components = []
        initial = False
        final = False
        preceding = self.preceding_context
        following = self.following_context
        if preceding.startswith("^"):
            initial = True
            preceding = preceding.replace("^", "").strip()
        if following.endswith("$"):
            final = True
            following = following.replace("$", "").strip()
        if preceding:

            components.append(rf"(?P<preceding>{preceding})")
        if self.segment:
            components.append(rf"(?P<segment>{self.segment})")
        if following:
            components.append(rf"(?P<following>{following})")
        pattern = " ".join(components)
        if initial:
            pattern = "^" + pattern
        else:
            pattern = r"(?:^|(?<=\s))" + pattern
        if final:
            pattern += "$"
        else:
            pattern += r"(?:$|(?=\s))"
        return re.compile(pattern, flags=re.UNICODE)

    def __str__(self):
        from_components = []
        to_components = []
        initial = False
        final = False
        preceding = self.preceding_context
        following = self.following_context
        if preceding.startswith("^"):
            initial = True
            preceding = preceding.replace("^", "").strip()
        if following.endswith("$"):
            final = True
            following = following.replace("$", "").strip()
        if preceding:
            from_components.append(preceding)
            to_components.append(preceding)
        if self.segment:
            from_components.append(self.segment)
        if self.replacement:
            to_components.append(self.replacement)
        if following:
            from_components.append(following)
            to_components.append(following)

        from_string = " ".join(from_components)
        to_string = " ".join(to_components)
        if initial:
            from_string = "^" + from_string
        if final:
            from_string += "$"
        return f"<PhonologicalRule {self.id} for Dialect {self.dialect_id}: {from_string} -> {to_string}>"

    def apply_rule(self, pronunciation: str) -> str:
        """
        Apply the rule on a pronunciation by replacing any matching segments with the replacement

        Parameters
        ----------
        pronunciation: str
            Pronunciation to apply rule

        Returns
        -------
        str
            Pronunciation with rule applied
        """
        preceding = self.preceding_context
        following = self.following_context
        if preceding.startswith("^"):
            preceding = preceding.replace("^", "").strip()
        if following.startswith("$"):
            following = following.replace("$", "").strip()
        components = []
        if preceding:
            components.append(r"\g<preceding>")
        if self.replacement:
            components.append(self.replacement)
        if following:
            components.append(r"\g<following>")
        return self.match_regex.sub(" ".join(components), pronunciation).strip()


class RuleApplication(MfaSqlBase):
    """
    Database class for mapping rules to generated pronunciations

    Parameters
    ----------
    pronunciation_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Pronunciation`
    rule_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.PhonologicalRule`
    pronunciation: :class:`~montreal_forced_aligner.db.Pronunciation`
        Pronunciation
    rule: :class:`~montreal_forced_aligner.db.PhonologicalRule`
        Rule applied
    """

    __tablename__ = "rule_applications"
    pronunciation_id = Column(ForeignKey("pronunciation.id", ondelete="CASCADE"), primary_key=True)
    rule_id = Column(ForeignKey("phonological_rule.id", ondelete="CASCADE"), primary_key=True)

    pronunciation = relationship("Pronunciation", back_populates="rules")

    rule = relationship("PhonologicalRule", back_populates="pronunciations")


class Speaker(MfaSqlBase):
    """
    Database class for storing information about speakers

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Name of the speaker
    cmvn: str
        File index for the speaker's CMVN stats
    dictionary_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Dictionary`
    dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
        Pronunciation dictionary that the speaker uses
    utterances: list[:class:`~montreal_forced_aligner.db.Utterance`]
        Utterances for the speaker
    files: list[:class:`~montreal_forced_aligner.db.File`]
        Files that the speaker spoke in
    """

    __tablename__ = "speaker"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    cmvn = Column(String)
    min_f0 = Column(Float, nullable=True)
    max_f0 = Column(Float, nullable=True)
    ivector = Column(Vector(IVECTOR_DIMENSION), nullable=True)
    plda_vector = Column(Vector(PLDA_DIMENSION), nullable=True)
    xvector = Column(Vector(XVECTOR_DIMENSION), nullable=True)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"), nullable=True, index=True)
    dictionary = relationship("Dictionary", back_populates="speakers")
    utterances = relationship("Utterance", back_populates="speaker")
    files = relationship("File", secondary=SpeakerOrdering, back_populates="speakers")

    __table_args__ = (
        sqlalchemy.Index(
            "speaker_ivector_index",
            "ivector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"ivector": "vector_cosine_ops"},
        ),
        sqlalchemy.Index(
            "speaker_xvector_index",
            "xvector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"xvector": "vector_cosine_ops"},
        ),
        sqlalchemy.Index(
            "speaker_plda_vector_index",
            "plda_vector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"plda_vector": "vector_cosine_ops"},
        ),
    )


class File(MfaSqlBase):
    """
    Database class for storing information about files in the corpus

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Base name of the file
    relative_path: str
        Path of the file relative to the root corpus directory
    modified: bool
        Flag for whether the file has been changed in the database for exporting
    text_file: :class:`~montreal_forced_aligner.db.TextFile`
        TextFile object with information about the transcript of a file
    sound_file: :class:`~montreal_forced_aligner.db.SoundFile`
        SoundFile object with information about the audio of a file
    speakers: list[:class:`~montreal_forced_aligner.db.Speaker`]
        Speakers in the file ordered by their index
    utterances: list[:class:`~montreal_forced_aligner.db.Utterance`]
        Utterances in the file
    """

    __tablename__ = "file"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)
    relative_path = Column(String, nullable=False)
    modified = Column(Boolean, nullable=False, default=False, index=True)
    speakers = relationship(
        "Speaker",
        secondary=SpeakerOrdering,
        back_populates="files",
        order_by=SpeakerOrdering.c.index,
    )
    text_file = relationship(
        "TextFile", back_populates="file", uselist=False, cascade="all, delete"
    )
    sound_file = relationship(
        "SoundFile", back_populates="file", uselist=False, cascade="all, delete"
    )
    utterances = relationship(
        "Utterance",
        back_populates="file",
        order_by="Utterance.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )

    @property
    def num_speakers(self) -> int:
        """Number of speakers in the file"""
        return len(self.speakers)

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the file"""
        return len(self.utterances)

    @property
    def duration(self) -> float:
        """Duration of the associated sound file"""
        return self.sound_file.duration

    @property
    def num_channels(self) -> int:
        """Number of channels of the associated sound file"""
        return self.sound_file.num_channels

    @property
    def sample_rate(self) -> int:
        """Sample rate of the associated sound file"""
        return self.sound_file.sample_rate

    def save(
        self,
        output_directory,
        output_format: typing.Optional[str] = None,
        save_transcription: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Output File to TextGrid or lab.  If ``text_type`` is not specified, the original file type will be used,
        but if there was no text file for the file, it will guess lab format if there is only one utterance, otherwise
        it will output a TextGrid file.

        Parameters
        ----------
        output_directory: str
            Directory to output file, if None, then it will overwrite the original file
        output_format: str, optional
            Text type to save as, if not provided, it will use either the original file type or guess the file type
        save_transcription: bool
            Flag for whether the hypothesized transcription text should be saved instead of the default text
        """
        from montreal_forced_aligner.alignment.multiprocessing import construct_output_path

        utterance_count = len(self.utterances)
        if output_format is None:  # Saving directly
            if (
                utterance_count == 1
                and self.utterances[0].begin == 0
                and self.utterances[0].end == self.duration
            ):
                output_format = TextFileType.LAB.value
            else:
                output_format = TextFileType.TEXTGRID.value
        output_path = construct_output_path(
            self.name, self.relative_path, output_directory, output_format=output_format
        )
        if overwrite:
            if self.text_file is None:
                self.text_file = TextFile(
                    file_id=self.id, text_file_path=output_path, file_type=output_format
                )
            if output_path != self.text_file.text_file_path and os.path.exists(
                self.text_file.text_file_path
            ):
                os.remove(self.text_file.text_file_path)
            self.text_file.file_type = output_format
            self.text_file.text_file_path = output_path
        if output_format == TextFileType.LAB.value:
            if (
                utterance_count == 0
                and os.path.exists(self.text_file.text_file_path)
                and not save_transcription
            ):
                os.remove(self.text_file.text_file_path)
                return
            elif utterance_count == 0:
                return
            for u in self.utterances:
                if save_transcription:
                    with mfa_open(output_path, "w") as f:
                        f.write(u.transcription_text if u.transcription_text else "")
                elif u.text:
                    with mfa_open(output_path, "w") as f:
                        f.write(u.text)
            return
        elif output_format == TextFileType.TEXTGRID.value:
            max_time = self.sound_file.duration
            tiers = {}
            for speaker in self.speakers:
                tiers[speaker.name] = textgrid.IntervalTier(
                    speaker.name, [], minT=0, maxT=max_time
                )

            tg = textgrid.Textgrid()
            tg.maxTimestamp = max_time
            for utterance in self.utterances:
                if utterance.speaker.name not in tiers:
                    tiers[utterance.speaker.name] = textgrid.IntervalTier(
                        utterance.speaker.name, [], minT=0, maxT=max_time
                    )
                if save_transcription:
                    tiers[utterance.speaker.name].insertEntry(
                        Interval(
                            start=utterance.begin,
                            end=utterance.end,
                            label=utterance.transcription_text
                            if utterance.transcription_text
                            else "",
                        )
                    )
                else:
                    if tiers[utterance.speaker.name].entries:
                        if tiers[utterance.speaker.name].entries[-1].end > utterance.begin:
                            utterance.begin = tiers[utterance.speaker.name].entries[-1].end
                    if utterance.end > self.duration:
                        utterance.end = self.duration
                    tiers[utterance.speaker.name].insertEntry(
                        Interval(
                            start=utterance.begin, end=utterance.end, label=utterance.text.strip()
                        )
                    )
            for t in tiers.values():
                tg.addTier(t)
            tg.save(output_path, includeBlankSpaces=True, format=output_format)

    def construct_transcription_tiers(
        self, original_text=False
    ) -> typing.Dict[str, typing.Dict[str, typing.List[CtmInterval]]]:
        """
        Construct output transcription tiers for the file

        Returns
        -------
        dict[str, dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]]
            Tier dictionary of utterance transcriptions
        """
        data = {}
        for u in self.utterances:
            speaker_name = u.speaker_name
            if speaker_name not in data:
                data[speaker_name] = {}
            if original_text:
                label = u.text
                key = "text"
            else:
                label = u.transcription_text
                key = "transcription"
            if not label:
                label = ""
            if key not in data[speaker_name]:
                data[speaker_name][key] = []
            data[speaker_name][key].append(CtmInterval(u.begin, u.end, label))
        return data


class SoundFile(MfaSqlBase):
    """

    Database class for storing information about sound files

    Parameters
    ----------
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.File`
    file: :class:`~montreal_forced_aligner.db.File`
        Root file
    sound_file_path: str
        Path to the audio file
    format: str
        Format of the audio file (flac, wav, mp3, etc)
    sample_rate: int
        Sample rate of the audio file
    duration: float
        Duration of audio file
    num_channels: int
        Number of channels in the audio file
    sox_string: str
        String that Kaldi will use to process the sound file
    """

    __tablename__ = "sound_file"

    file_id = Column(ForeignKey("file.id"), primary_key=True)
    file = relationship("File", back_populates="sound_file")
    sound_file_path = Column(String, nullable=False)
    format = Column(String, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    num_channels = Column(Integer, nullable=False)
    sox_string = Column(String)

    def normalized_waveform(
        self, begin: float = 0, end: typing.Optional[float] = None
    ) -> typing.Tuple[np.array, np.array]:
        """
        Load a normalized waveform for acoustic processing/visualization

        Parameters
        ----------
        begin: float, optional
            Starting time point to return, defaults to 0
        end: float, optional
            Ending time point to return, defaults to the end of the file

        Returns
        -------
        numpy.array
            Time points
        numpy.array
            Sample values
        """
        if end is None or end > self.duration:
            end = self.duration

        y, _ = librosa.load(
            self.sound_file_path, sr=None, mono=False, offset=begin, duration=end - begin
        )
        if len(y.shape) > 1 and y.shape[0] == 2:
            y /= np.max(np.abs(y))
            num_steps = y.shape[1]
        else:
            y /= np.max(np.abs(y), axis=0)
            num_steps = y.shape[0]
        y[np.isnan(y)] = 0
        x = np.linspace(start=begin, stop=end, num=num_steps)
        return x, y


class TextFile(MfaSqlBase):
    """
    Database class for storing information about transcription files

    Parameters
    ----------
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.File`
    file: :class:`~montreal_forced_aligner.db.File`
        Root file
    text_file_path: str
        Path to the transcription file
    file_type: str
        Type of the transcription file (lab, TextGrid, etc)
    """

    __tablename__ = "text_file"

    file_id = Column(ForeignKey("file.id"), primary_key=True)
    file = relationship("File", back_populates="text_file")
    text_file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)


class Utterance(MfaSqlBase):
    """

    Database class for storing information about utterances

    Parameters
    ----------
    id: int
        Primary key
    begin: float
        Beginning timestamp of the utterance
    end: float
        Ending timestamp of the utterance, -1 if there is no audio file
    duration: float
        Duration of the utterance
    channel: int
        Channel of the utterance in the audio file
    num_frames: int
        Number of feature frames extracted
    text: str
        Input text for the utterance
    oovs: str
        Space-delimited list of items that were not found in the speaker's pronunciation dictionary
    normalized_text: str
        Normalized text for the utterance, after removing case and punctuation, and splitting up compounds and clitics if the whole word is not
        found in the speaker's pronunciation dictionary
    features:str
        File index for generated features
    in_subset: bool
        Flag for whether to use this utterance in the current training subset
    ignored: bool
        Flag for if the utterance is ignored due to lacking features
    alignment_log_likelihood: float
        Log likelihood for the alignment of the utterance
    phone_error_rate: float
        Phone error rate for alignment evaluation
    alignment_score: float
        Alignment score from alignment evaluation
    word_error_rate: float
        Word error rate for transcription evaluation
    character_error_rate: float
        Character error rate for transcription evaluation
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.File`
    speaker_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Speaker`
    file: :class:`~montreal_forced_aligner.db.File`
        File object that the utterance is from
    speaker: :class:`~montreal_forced_aligner.db.Speaker`
        Speaker object of the utterance
    phone_intervals: list[:class:`~montreal_forced_aligner.db.PhoneInterval`]
        Reference phone intervals
    word_intervals: list[:class:`~montreal_forced_aligner.db.WordInterval`]
        Aligned word intervals
    job_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Job`
    job: :class:`~montreal_forced_aligner.db.Job`
        Job that processes the utterance
    """

    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    duration = Column(Float, sqlalchemy.Computed('"end" - "begin"'), index=True)
    channel = Column(Integer, nullable=False)
    num_frames = Column(Integer)
    text = Column(String)
    oovs = Column(String)
    normalized_text = Column(String)
    normalized_character_text = Column(String)
    transcription_text = Column(String)
    features = Column(String)
    ivector_ark = Column(String)
    vad_ark = Column(String)
    in_subset = Column(Boolean, nullable=False, default=False, index=True)
    ignored = Column(Boolean, nullable=False, default=False, index=True)
    alignment_log_likelihood = Column(Float)
    phone_error_rate = Column(Float)
    alignment_score = Column(Float)
    word_error_rate = Column(Float)
    character_error_rate = Column(Float)
    ivector = Column(Vector(IVECTOR_DIMENSION), nullable=True)
    plda_vector = Column(Vector(PLDA_DIMENSION), nullable=True)
    xvector = Column(Vector(XVECTOR_DIMENSION), nullable=True)
    file_id = Column(Integer, ForeignKey("file.id"), index=True, nullable=False)
    speaker_id = Column(Integer, ForeignKey("speaker.id"), index=True, nullable=False)
    kaldi_id = Column(
        String,
        sqlalchemy.Computed("CAST(speaker_id AS text)|| '-' ||CAST(id AS text)"),
        unique=True,
        index=True,
    )
    job_id = Column(Integer, ForeignKey("job.id"), index=True, nullable=True)
    file = relationship("File", back_populates="utterances")
    speaker = relationship("Speaker", back_populates="utterances")
    job = relationship("Job", back_populates="utterances")
    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="utterance",
        order_by="PhoneInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )
    word_intervals = relationship(
        "WordInterval",
        back_populates="utterance",
        order_by="WordInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )

    __table_args__ = (
        sqlalchemy.Index(
            "utterance_position_index", "file_id", "speaker_id", "begin", "end", "channel"
        ),
        sqlalchemy.Index(
            "utterance_text_idx",
            "text",
            postgresql_ops={"text": "gin_trgm_ops"},
            postgresql_using="gin",
        ),
        sqlalchemy.Index(
            "utterance_ivector_index",
            "ivector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"ivector": "vector_cosine_ops"},
        ),
        sqlalchemy.Index(
            "utterance_xvector_index",
            "xvector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"xvector": "vector_cosine_ops"},
        ),
        sqlalchemy.Index(
            "utterance_plda_vector_index",
            "plda_vector",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"plda_vector": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        """String representation of the utterance object"""
        return f"<Utterance in {self.file_name} by {self.speaker_name} from {self.begin} to {self.end}>"

    def phone_intervals_for_workflow(self, workflow_id: int) -> typing.List[CtmInterval]:
        """
        Extract phone intervals for a given :class:`~montreal_forced_aligner.db.CorpusWorkflow`

        Parameters
        ----------
        workflow_id: int
            Integer ID for :class:`~montreal_forced_aligner.db.CorpusWorkflow`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.data.CtmInterval`]
            List of phone intervals
        """
        return [x.as_ctm() for x in self.phone_intervals if x.workflow_id == workflow_id]

    def word_intervals_for_workflow(self, workflow_id: int) -> typing.List[CtmInterval]:
        """
        Extract word intervals for a given :class:`~montreal_forced_aligner.db.CorpusWorkflow`

        Parameters
        ----------
        workflow_id: int
            Integer ID for :class:`~montreal_forced_aligner.db.CorpusWorkflow`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.data.CtmInterval`]
            List of word intervals
        """
        return [x.as_ctm() for x in self.word_intervals if x.workflow_id == workflow_id]

    @property
    def reference_phone_intervals(self) -> typing.List[CtmInterval]:
        """
        Phone intervals from :attr:`montreal_forced_aligner.data.WorkflowType.reference`
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if x.workflow.workflow_type is WorkflowType.reference
        ]

    @property
    def aligned_phone_intervals(self) -> typing.List[CtmInterval]:
        """
        Phone intervals from :attr:`montreal_forced_aligner.data.WorkflowType.alignment`
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if x.workflow.workflow_type is WorkflowType.alignment
        ]

    @property
    def aligned_word_intervals(self) -> typing.List[CtmInterval]:
        """
        Word intervals from :attr:`montreal_forced_aligner.data.WorkflowType.alignment`
        """
        return [
            x.as_ctm()
            for x in self.word_intervals
            if x.workflow.workflow_type is WorkflowType.alignment
        ]

    @property
    def transcribed_phone_intervals(self) -> typing.List[CtmInterval]:
        """
        Phone intervals from :attr:`montreal_forced_aligner.data.WorkflowType.transcription`
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if x.workflow.workflow_type is WorkflowType.transcription
        ]

    @property
    def transcribed_word_intervals(self) -> typing.List[CtmInterval]:
        """
        Word intervals from :attr:`montreal_forced_aligner.data.WorkflowType.transcription`
        """
        return [
            x.as_ctm()
            for x in self.word_intervals
            if x.workflow.workflow_type is WorkflowType.transcription
        ]

    @property
    def per_speaker_transcribed_phone_intervals(self) -> typing.List[CtmInterval]:
        """
        Phone intervals from :attr:`montreal_forced_aligner.data.WorkflowType.per_speaker_transcription`
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if x.workflow.workflow_type is WorkflowType.per_speaker_transcription
        ]

    @property
    def per_speaker_transcribed_word_intervals(self) -> typing.List[CtmInterval]:
        """
        Word intervals from :attr:`montreal_forced_aligner.data.WorkflowType.per_speaker_transcription`
        """
        return [
            x.as_ctm()
            for x in self.word_intervals
            if x.workflow.workflow_type is WorkflowType.per_speaker_transcription
        ]

    @property
    def phone_transcribed_phone_intervals(self) -> typing.List[CtmInterval]:
        """
        Phone intervals from :attr:`montreal_forced_aligner.data.WorkflowType.phone_transcription`
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if x.workflow.workflow_type is WorkflowType.phone_transcription
        ]

    @property
    def file_name(self) -> str:
        """Name of the utterance's file"""
        return self.file.name

    @property
    def speaker_name(self) -> str:
        """Name of the utterance's speaker"""
        return self.speaker.name

    def to_data(self) -> UtteranceData:
        """
        Construct an UtteranceData object that can be used in multiprocessing

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`
            Data for the utterance
        """
        from montreal_forced_aligner.corpus.classes import UtteranceData

        if self.normalized_text is None:
            self.normalized_text = ""
        return UtteranceData(
            self.speaker_name,
            self.file_name,
            self.begin,
            self.end,
            self.channel,
            self.text,
            self.normalized_text.split(),
            set(self.oovs.split()),
        )

    @classmethod
    def from_data(cls, data: UtteranceData, file: File, speaker: int, frame_shift: int = None):
        """
        Generate an utterance object from :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`

        Parameters
        ----------
        data: :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`
            Data for the utterance
        file: :class:`~montreal_forced_aligner.db.File`
            File database object for the utterance
        speaker: :class:`~montreal_forced_aligner.db.Speaker`
            Speaker database object for the utterance
        frame_shift: int, optional
            Frame shift in ms to use for calculating the number of frames in the utterance

        Returns
        -------
        :class:`~montreal_forced_aligner.db.Utterance`
            Utterance object
        """
        if not isinstance(speaker, int):
            speaker = speaker.id
        num_frames = None
        if frame_shift is not None:
            num_frames = int((data.end - data.begin) / round(frame_shift / 1000, 4))

        return Utterance(
            begin=data.begin,
            end=data.end,
            channel=data.channel,
            oovs=" ".join(sorted(data.oovs)),
            normalized_text=" ".join(data.normalized_text),
            text=data.text,
            num_frames=num_frames,
            file_id=file.id,
            speaker_id=speaker,
        )


class CorpusWorkflow(MfaSqlBase):
    """

    Database class for storing information about a particular workflow (alignment, transcription, etc)

    Parameters
    ----------
    id: int
        Primary key
    workflow_type: :class:`~montreal_forced_aligner.data.WorkflowType`
        Workflow type
    time_stamp: :class:`datetime.datetime`
        Time stamp for the workflow run
    score: float
        Log likelihood or other score for the workflow run
    phone_intervals: list[:class:`~montreal_forced_aligner.db.PhoneInterval`]
        Phone intervals linked to the workflow
    word_intervals: list[:class:`~montreal_forced_aligner.db.WordInterval`]
        Word intervals linked to the workflow
    """

    __tablename__ = "corpus_workflow"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    workflow_type = Column(Enum(WorkflowType), nullable=False, index=True)
    working_directory = Column(String, nullable=False)
    time_stamp = Column(DateTime, nullable=False, server_default=sqlalchemy.func.now(), index=True)
    current = Column(Boolean, nullable=False, default=False, index=True)
    done = Column(Boolean, nullable=False, default=False, index=True)
    dirty = Column(Boolean, nullable=False, default=False, index=True)
    alignments_collected = Column(Boolean, nullable=False, default=False, index=True)
    score = Column(Float, nullable=True)

    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="workflow",
        order_by="PhoneInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )

    word_intervals = relationship(
        "WordInterval",
        back_populates="workflow",
        order_by="WordInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )

    @property
    def lda_mat_path(self) -> str:
        return os.path.join(self.working_directory, "lda.mat")


class PhoneInterval(MfaSqlBase):
    """

    Database class for storing information about aligned phone intervals

    Parameters
    ----------
    id: int
        Primary key
    begin: float
        Beginning timestamp of the interval
    end: float
        Ending timestamp of the interval
    phone_goodness: float
        Confidence score, log-likelihood, etc for the phone interval
    phone_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Phone`
    phone: :class:`~montreal_forced_aligner.db.Phone`
        Phone of the interval
    utterance_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.db.Utterance`
        Utterance of the interval
    word_interval_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.WordInterval`
    word_interval: :class:`~montreal_forced_aligner.db.WordInterval`
        Word interval that is associated with the phone interval
    workflow_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.CorpusWorkflow`
    workflow: :class:`~montreal_forced_aligner.db.CorpusWorkflow`
        Workflow that generated the phone interval
    """

    __tablename__ = "phone_interval"

    id = Column(Integer, primary_key=True, autoincrement=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    phone_goodness = Column(Float, nullable=True)

    phone_id = Column(
        Integer, ForeignKey("phone.id", ondelete="CASCADE"), index=True, nullable=False
    )
    phone = relationship("Phone", back_populates="phone_intervals")

    word_interval_id = Column(
        Integer, ForeignKey("word_interval.id", ondelete="CASCADE"), index=True, nullable=True
    )
    word_interval = relationship("WordInterval", back_populates="phone_intervals")

    utterance_id = Column(
        Integer, ForeignKey("utterance.id", ondelete="CASCADE"), index=True, nullable=False
    )
    utterance = relationship("Utterance", back_populates="phone_intervals")

    workflow_id = Column(
        Integer, ForeignKey("corpus_workflow.id", ondelete="CASCADE"), index=True, nullable=False
    )
    workflow = relationship("CorpusWorkflow", back_populates="phone_intervals")

    __table_args__ = (
        sqlalchemy.Index("phone_utterance_workflow_index", "utterance_id", "workflow_id"),
    )

    def __repr__(self):
        return f"<PhoneInterval {self.kaldi_label} ({self.workflow.workflow_type}) from {self.begin}-{self.end} for utterance {self.utterance_id}>"

    @classmethod
    def from_ctm(
        self, interval: CtmInterval, utterance: Utterance, workflow_id: int
    ) -> PhoneInterval:
        """
        Construct a PhoneInterval from a CtmInterval object

        Parameters
        ----------
        interval: :class:`~montreal_forced_aligner.data.CtmInterval`
            CtmInterval containing data for the phone interval
        utterance: :class:`~montreal_forced_aligner.db.Utterance`
            Utterance object that the phone interval belongs to
        workflow_id: int
            Integer id for the workflow that generated the phone interval

        Returns
        -------
        :class:`~montreal_forced_aligner.db.PhoneInterval`
            Phone interval object
        """
        return PhoneInterval(
            begin=interval.begin,
            end=interval.end,
            label=interval.label,
            utterance=utterance,
            workflow_id=workflow_id,
        )

    def as_ctm(self) -> CtmInterval:
        """
        Generate a CtmInterval from the database object

        Returns
        -------
        :class:`~montreal_forced_aligner.data.CtmInterval`
            CTM interval object
        """
        return CtmInterval(self.begin, self.end, self.phone.phone, confidence=self.phone_goodness)


class WordInterval(MfaSqlBase):
    """

    Database class for storing information about aligned word intervals

    Parameters
    ----------
    id: int
        Primary key
    begin: float
        Beginning timestamp of the interval
    end: float
        Ending timestamp of the interval
    word_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Word`
    word: :class:`~montreal_forced_aligner.db.Word`
        Word of the interval
    pronunciation_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Pronunciation`
    pronunciation: :class:`~montreal_forced_aligner.db.Pronunciation`
        Pronunciation of the word
    utterance_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.db.Utterance`
        Utterance of the interval
    workflow_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.CorpusWorkflow`
    workflow: :class:`~montreal_forced_aligner.db.CorpusWorkflow`
        Workflow that generated the interval
    phone_intervals: list[:class:`~montreal_forced_aligner.db.PhoneInterval`]
        Phone intervals for the word interval
    """

    __tablename__ = "word_interval"

    id = Column(Integer, primary_key=True, autoincrement=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)

    utterance_id = Column(
        Integer, ForeignKey("utterance.id", ondelete="CASCADE"), index=True, nullable=False
    )
    utterance = relationship("Utterance", back_populates="word_intervals")

    word_id = Column(
        Integer, ForeignKey("word.id", ondelete="CASCADE"), index=True, nullable=False
    )
    word = relationship("Word", back_populates="word_intervals")

    pronunciation_id = Column(Integer, ForeignKey("pronunciation.id"), index=True, nullable=True)
    pronunciation = relationship("Pronunciation", back_populates="word_intervals")

    workflow_id = Column(
        Integer, ForeignKey("corpus_workflow.id", ondelete="CASCADE"), index=True, nullable=False
    )
    workflow = relationship("CorpusWorkflow", back_populates="word_intervals")

    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="word_interval",
        order_by="PhoneInterval.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete",
    )

    __table_args__ = (
        sqlalchemy.Index("word_utterance_workflow_index", "utterance_id", "workflow_id"),
    )

    @classmethod
    def from_ctm(
        self, interval: CtmInterval, utterance: Utterance, workflow_id: int
    ) -> WordInterval:
        """
        Construct a WordInterval from a CtmInterval object

        Parameters
        ----------
        interval: :class:`~montreal_forced_aligner.data.CtmInterval`
            CtmInterval containing data for the word interval
        utterance: :class:`~montreal_forced_aligner.db.Utterance`
            Utterance object that the word interval belongs to
        workflow_id: int
            Integer id for the workflow that generated the phone interval

        Returns
        -------
        :class:`~montreal_forced_aligner.db.WordInterval`
            Word interval object
        """
        return WordInterval(
            begin=interval.begin,
            end=interval.end,
            label=interval.label,
            utterance=utterance,
            workflow_id=workflow_id,
        )

    def as_ctm(self) -> CtmInterval:
        """
        Generate a CtmInterval from the database object

        Returns
        -------
        :class:`~montreal_forced_aligner.data.CtmInterval`
            CTM interval object
        """
        return CtmInterval(self.begin, self.end, self.word.word)


class Job(MfaSqlBase):
    """
    Database class for storing information about multiprocessing jobs

    Parameters
    ----------
    id: int
        Primary key
    corpus_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Corpus`
    corpus: :class:`~montreal_forced_aligner.db.Corpus`
        Corpus
    utterances: list[:class:`~montreal_forced_aligner.db.Utterance`]
        Utterances associated with the job
    symbols: list[:class:`~montreal_forced_aligner.db.M2M2Job`]
        Symbols associated with the job in training phonetisaurus models
    words: list[:class:`~montreal_forced_aligner.db.Word2Job`]
        Words associated with the job in training phonetisaurus models
    """

    __tablename__ = "job"

    id = Column(Integer, primary_key=True, autoincrement=True)

    corpus_id = Column(Integer, ForeignKey("corpus.id"), index=True, nullable=True)
    corpus = relationship("Corpus", back_populates="jobs")
    utterances = relationship("Utterance", back_populates="job")

    symbols = relationship(
        "M2M2Job",
        back_populates="job",
    )

    words = relationship(
        "Word2Job",
        back_populates="job",
    )

    dictionaries = relationship(
        "Dictionary",
        secondary=Dictionary2Job,
        back_populates="jobs",
    )

    def __str__(self):
        return f"<Job {self.id}>"

    @property
    def has_dictionaries(self) -> bool:
        return len(self.dictionaries) > 0

    @property
    def dictionary_ids(self) -> typing.List[int]:
        return [x.id for x in self.dictionaries]

    @property
    def wav_scp_path(self):
        return self.construct_path(self.corpus.split_directory, "wav", "scp")

    @property
    def segments_scp_path(self):
        return self.construct_path(self.corpus.split_directory, "segments", "scp")

    @property
    def feats_scp_path(self):
        return self.construct_path(self.corpus.split_directory, "feats", "scp")

    @property
    def feats_ark_path(self):
        return self.construct_path(self.corpus.split_directory, "feats", "ark")

    @property
    def per_dictionary_feats_scp_paths(self):
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(
                self.corpus.current_subset_directory, "feats", "scp", d.id
            )
        return paths

    @property
    def per_dictionary_utt2spk_scp_paths(self):
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(
                self.corpus.current_subset_directory, "utt2spk", "scp", d.id
            )
        return paths

    @property
    def per_dictionary_spk2utt_scp_paths(self):
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(
                self.corpus.current_subset_directory, "spk2utt", "scp", d.id
            )
        return paths

    @property
    def per_dictionary_cmvn_scp_paths(self):
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(
                self.corpus.current_subset_directory, "cmvn", "scp", d.id
            )
        return paths

    @property
    def per_dictionary_text_int_scp_paths(self):
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(
                self.corpus.current_subset_directory, "text", "int.scp", d.id
            )
        return paths

    def construct_path(
        self, directory: str, identifier: str, extension: str, dictionary_id: int = None
    ) -> str:
        """
        Helper function for constructing dictionary-dependent paths for the Job

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        str
            Path
        """
        if dictionary_id is None:
            return os.path.join(directory, f"{identifier}.{self.id}.{extension}")
        return os.path.join(directory, f"{identifier}.{dictionary_id}.{self.id}.{extension}")

    def construct_path_dictionary(self, directory: str, identifier: str, extension: str):
        paths = {}
        for d_id in self.dictionary_ids:
            paths[d_id] = self.construct_path(directory, identifier, extension, d_id)
        return paths

    def construct_dictionary_dependent_paths(
        self, directory: str, identifier: str, extension: str
    ) -> typing.Dict[int, str]:
        """
        Helper function for constructing paths that depend only on the dictionaries of the job, and not the job name itself.
        These paths should be merged with all other jobs to get a full set of dictionary paths.
        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark
        Returns
        -------
        dict[int, str]
            Path for each dictionary
        """
        output = {}
        for dict_id in self.dictionary_ids:
            output[dict_id] = os.path.join(directory, f"{identifier}.{dict_id}.{extension}")
        return output

    def construct_online_feature_proc_string(self):
        feat_path = self.construct_path(self.corpus.current_subset_directory, "feats", "scp")
        return f'ark,s,cs:add-deltas scp,s,cs:"{feat_path}" ark:- |'

    def construct_feature_proc_string(
        self,
        working_directory,
        dictionary_id,
        uses_splices: bool,
        splice_left_context: int,
        splice_right_context: int,
        uses_speaker_adaptation: bool = False,
    ) -> str:
        """
        Constructs a feature processing string to supply to Kaldi binaries, taking into account corpus features and the
        current working directory of the aligner (whether fMLLR or LDA transforms should be used, etc).

        Parameters
        ----------
        uses_speaker_adaptation: bool
            Flag for whether features should be speaker-independent regardless of the presence of fMLLR transforms

        Returns
        -------
        dict[int, dict[str, str]]
            Feature strings per job
        """
        lda_mat_path = None
        fmllr_trans_path = None
        feat_path = self.construct_path(
            self.corpus.current_subset_directory, "feats", "scp", dictionary_id=dictionary_id
        )
        if working_directory is not None:
            lda_mat_path = os.path.join(working_directory, "lda.mat")
            if not os.path.exists(lda_mat_path):
                lda_mat_path = None
            fmllr_trans_path = self.construct_path(
                working_directory, "trans", "ark", dictionary_id
            )

            if not os.path.exists(fmllr_trans_path):
                fmllr_trans_path = None
        utt2spk_path = self.construct_path(
            self.corpus.current_subset_directory, "utt2spk", "scp", dictionary_id
        )
        feats = "ark,s,cs:"

        if lda_mat_path is not None:
            feats += f'splice-feats --left-context={splice_left_context} --right-context={splice_right_context} scp,s,cs:"{feat_path}" ark:- |'
            feats += f' transform-feats "{lda_mat_path}" ark:- ark:- |'
        elif uses_splices:
            feats += f'splice-feats --left-context={splice_left_context} --right-context={splice_right_context} scp,s,cs:"{feat_path}" ark:- |'
        else:
            feats += f'add-deltas scp,s,cs:"{feat_path}" ark:- |'
        if fmllr_trans_path is not None and uses_speaker_adaptation:
            feats += f' transform-feats --utt2spk=ark:"{utt2spk_path}" ark:"{fmllr_trans_path}" ark:- ark:- |'

        return feats


class M2MSymbol(MfaSqlBase):
    """

    Database class for storing information many to many G2P training information

    Parameters
    ----------
    id: int
        Primary key
    symbol: str
        Symbol
    total_order: int
        Summed order of graphemes and phones
    max_order: int
        Maximum order between graphemes and phones
    grapheme_order: int
        Grapheme order
    phone_order: int
        Phone order
    weight: float
        Weight of arcs
    jobs: list[:class:`~montreal_forced_aligner.db.M2M2Job`]
        Jobs that use this symbol
    """

    __tablename__ = "m2m_symbol"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True, unique=True)
    total_order = Column(Integer, nullable=False)
    max_order = Column(Integer, nullable=False)
    grapheme_order = Column(Integer, nullable=False)
    phone_order = Column(Integer, nullable=False)
    weight = Column(Float, nullable=False)

    jobs = relationship(
        "M2M2Job",
        back_populates="m2m_symbol",
    )


class M2M2Job(MfaSqlBase):
    """
    Mapping class between :class:`~montreal_forced_aligner.db.M2MSymbol`
    and :class:`~montreal_forced_aligner.db.Job`

    Parameters
    ----------
    m2m_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.M2MSymbol`
    job_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Job`
    m2m_symbol: :class:`~montreal_forced_aligner.db.M2MSymbol`
        M2MSymbol object
    job: :class:`~montreal_forced_aligner.db.Job`
        Job object
    """

    __tablename__ = "m2m_job"
    m2m_id = Column(ForeignKey("m2m_symbol.id"), primary_key=True)
    job_id = Column(ForeignKey("job.id"), primary_key=True)
    m2m_symbol = relationship("M2MSymbol", back_populates="jobs")
    job = relationship("Job", back_populates="symbols")


class Word2Job(MfaSqlBase):
    """
    Mapping class between :class:`~montreal_forced_aligner.db.Word`
    and :class:`~montreal_forced_aligner.db.Job`

    Parameters
    ----------
    word_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.M2MSymbol`
    job_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Job`
    word: :class:`~montreal_forced_aligner.db.Word`
        Word object
    job: :class:`~montreal_forced_aligner.db.Job`
        Job object
    """

    __tablename__ = "word_job"

    word_id = Column(ForeignKey("word.id"), primary_key=True)
    job_id = Column(ForeignKey("job.id"), primary_key=True)
    training = Column(Boolean, index=True)
    word = relationship("Word", back_populates="job")
    job = relationship("Job", back_populates="words")
