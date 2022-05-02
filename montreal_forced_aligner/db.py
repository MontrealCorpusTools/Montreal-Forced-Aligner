"""Database classes"""
from __future__ import annotations

import os
import typing

import librosa
import numpy as np
import sqlalchemy
from praatio import textgrid
from praatio.utilities.constants import Interval, TextgridFormats
from sqlalchemy import Boolean, Column, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import Bundle, column_property, declarative_base, relationship

from montreal_forced_aligner.data import (
    CtmInterval,
    PhoneSetType,
    PhoneType,
    TextFileType,
    WordType,
)

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.corpus.classes import UtteranceData

__all__ = [
    "DictBundle",
    "Dictionary",
    "Word",
    "OovWord",
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
    "ReferencePhoneInterval",
    "MfaSqlBase",
]

MfaSqlBase = declarative_base()


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

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    imported = Column(Boolean, default=False)
    features_generated = Column(Boolean, default=False)
    alignment_done = Column(Boolean, default=False)
    transcription_done = Column(Boolean, default=False)
    alignment_evaluation_done = Column(Boolean, default=False)
    has_reference_alignments = Column(Boolean, default=False)


class Dictionary(MfaSqlBase):
    """
    Database class for storing information about a pronunciation dictionary

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Dictionary name
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

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    path = Column(String, unique=True, nullable=False)
    phone_set_type = Column(Enum(PhoneSetType))
    root_temp_directory = Column(String, nullable=False)
    bracket_regex = Column(String, nullable=True)
    laughter_regex = Column(String, nullable=True)
    position_dependent_phones = Column(Boolean, nullable=False)
    default = Column(Boolean, default=False, nullable=False)
    clitic_marker = Column(String(1), nullable=True)
    silence_word = Column(String, nullable=False)
    optional_silence_phone = Column(String, nullable=False)
    oov_word = Column(String, nullable=False)
    bracketed_word = Column(String, nullable=False)
    laughter_word = Column(String, nullable=False)

    max_disambiguation_symbol = Column(Integer, default=0, nullable=False)
    silence_probability = Column(Float, default=0.5, nullable=False)
    initial_silence_probability = Column(Float, default=0.5, nullable=False)
    final_silence_correction = Column(Float, nullable=True)
    final_non_silence_correction = Column(Float, nullable=True)

    words = relationship(
        "Word",
        back_populates="dictionary",
        order_by="Word.mapping_id",
        collection_class=ordering_list("mapping_id"),
        cascade="all, delete-orphan",
    )
    oov_words = relationship(
        "OovWord",
        back_populates="dictionary",
        order_by="OovWord.count.desc()",
        collection_class=ordering_list("count", ordering_func=lambda x: -x),
        cascade="all, delete-orphan",
    )
    speakers = relationship("Speaker", back_populates="dictionary")

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
        return os.path.join(self.root_temp_directory, "phones")

    @property
    def phone_symbol_table_path(self):
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_directory, "phones.txt")

    @property
    def phone_disambig_path(self):
        """Path to file containing phone symbols and their integer IDs"""
        return os.path.join(self.phones_directory, "phone_disambig.txt")

    @property
    def temp_directory(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.root_temp_directory, f"{self.id}_{self.name}")

    @property
    def lexicon_disambig_fst_path(self) -> str:
        """
        Path of disambiguated lexicon fst (L.fst)
        """
        return os.path.join(self.temp_directory, "L.disambig_fst")

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
    def output_directory(self) -> str:
        """Temporary directory for the dictionary"""
        return os.path.join(self.temporary_directory, self.identifier)

    def write(
        self,
        silence_disambiguation_symbol=None,
        debug=False,
    ) -> None:
        """
        Write the files necessary for Kaldi

        Parameters
        ----------
        silence_disambiguation_symbol: str, optional
            Symbol to use as silence disambiguation
        debug: bool, optional
            Flag for whether to keep temporary files, defaults to False
        """
        os.makedirs(self.temp_directory, exist_ok=True)
        if debug:
            self.export_lexicon(os.path.join(self.temp_directory, "lexicon.txt"))
        self._write_word_file()
        self._write_probabilistic_fst_text(silence_disambiguation_symbol)
        self._write_fst_binary(write_disambiguation=silence_disambiguation_symbol is not None)
        if not debug:
            self.cleanup()

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

    id = Column(Integer, primary_key=True)
    mapping_id = Column(Integer, nullable=False, unique=True)
    phone = Column(String(10), unique=True, nullable=False)
    phone_type = Column(Enum(PhoneType), nullable=False, index=True)


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

    id = Column(Integer, primary_key=True)
    mapping_id = Column(Integer, nullable=False, index=True)
    word = Column(String, nullable=False, index=True)
    count = Column(Integer, default=0, nullable=False)
    word_type = Column(Enum(WordType), nullable=False, index=True)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"), nullable=False, index=True)
    dictionary: Dictionary = relationship("Dictionary", back_populates="words")
    pronunciations = relationship("Pronunciation", back_populates="word")

    __table_args__ = (
        sqlalchemy.Index("dictionary_word_type_index", "dictionary_id", "word_type"),
    )


class OovWord(MfaSqlBase):
    """
    Database class for storing words, their integer IDs, and pronunciation information

    Parameters
    ----------
    id: int
        Primary key
    word: str
        Word label
    count: int
        Count frequency of word in the corpus
    dictionary_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Dictionary`
    dictionary: :class:`~montreal_forced_aligner.db.Dictionary`
        Pronunciation dictionary that the word belongs to
    """

    __tablename__ = "oov_word"

    id = Column(Integer, primary_key=True)
    word = Column(String, nullable=False, index=True)
    count = Column(Integer, default=0, nullable=False)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"), nullable=False, index=True)
    dictionary: Dictionary = relationship("Dictionary", back_populates="oov_words")


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

    id = Column(Integer, primary_key=True)
    pronunciation = Column(String, nullable=False)
    probability = Column(Float, nullable=True)
    disambiguation = Column(Integer, nullable=True)
    silence_after_probability = Column(Float, nullable=True)
    silence_before_correction = Column(Float, nullable=True)
    non_silence_before_correction = Column(Float, nullable=True)
    word_id = Column(Integer, ForeignKey("word.id"), nullable=False, index=True)
    word: Word = relationship("Word", back_populates="pronunciations")


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
    job_id: int
        Multiprocessing job ID for the speaker
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

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    cmvn = Column(String)
    job_id = Column(Integer, index=True)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"), nullable=True, index=True)
    dictionary: Dictionary = relationship("Dictionary", back_populates="speakers")
    utterances = relationship("Utterance", back_populates="speaker")
    files = relationship("SpeakerOrdering", back_populates="speaker")

    __table_args__ = (sqlalchemy.Index("job_dictionary_index", "job_id", "dictionary_id"),)


class SpeakerOrdering(MfaSqlBase):
    """
    Mapping class between :class:`~montreal_forced_aligner.db.Speaker`
    and :class:`~montreal_forced_aligner.db.File` that preserves the order of tiers

    Parameters
    ----------
    speaker_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Speaker`
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.File`
    index: int
        Position of speaker in the input TextGrid
    speaker: :class:`~montreal_forced_aligner.db.Speaker`
        Speaker object
    file: :class:`~montreal_forced_aligner.db.File`
        File object
    """

    __tablename__ = "speaker_ordering"
    speaker_id = Column(ForeignKey("speaker.id"), primary_key=True)
    file_id = Column(ForeignKey("file.id"), primary_key=True)
    index = Column(Integer)
    speaker: Speaker = relationship("Speaker", back_populates="files")
    file: File = relationship("File", back_populates="speakers")


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
    speakers: list[:class:`~montreal_forced_aligner.db.SpeakerOrdering`]
        Speakers in the file ordered by their index
    utterances: list[:class:`~montreal_forced_aligner.db.Utterance`]
        Utterances in the file
    """

    __tablename__ = "file"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    relative_path = Column(String, nullable=False)
    modified = Column(Boolean, nullable=False, default=False)
    speakers = relationship(
        "SpeakerOrdering",
        back_populates="file",
        order_by="SpeakerOrdering.index",
        collection_class=ordering_list("index"),
        cascade="all, delete-orphan",
    )
    text_file: TextFile = relationship(
        "TextFile", back_populates="file", uselist=False, cascade="all, delete-orphan"
    )
    sound_file: SoundFile = relationship(
        "SoundFile", back_populates="file", uselist=False, cascade="all, delete-orphan"
    )
    utterances = relationship(
        "Utterance",
        back_populates="file",
        order_by="Utterance.begin",
        collection_class=ordering_list("begin"),
        cascade="all, delete-orphan",
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
        output_directory: typing.Optional[str] = None,
        output_format: typing.Optional[str] = None,
        save_transcription: bool = False,
    ) -> None:
        """
        Output File to TextGrid or lab.  If ``text_type`` is not specified, the original file type will be used,
        but if there was no text file for the file, it will guess lab format if there is only one utterance, otherwise
        it will output a TextGrid file.

        Parameters
        ----------
        output_directory: str, optional
            Directory to output file, if None, then it will overwrite the original file
        output_format: str, optional
            Text type to save as, if not provided, it will use either the original file type or guess the file type
        save_transcription: bool
            Flag for whether the hypothesized transcription text should be saved instead of the default text
        """
        utterance_count = len(self.utterances)
        overwrite = output_format is None
        if overwrite:  # Saving directly
            if (
                utterance_count == 1
                and self.utterances[0].begin == 0
                and self.utterances[0].end == self.duration
            ):
                output_format = TextFileType.LAB.value
            else:
                output_format = TextFileType.TEXTGRID.value
        output_path = self.construct_output_path(output_directory, output_format=output_format)
        if overwrite:
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
                    with open(output_path, "w", encoding="utf8") as f:
                        f.write(u.transcription_text if u.transcription_text else "")
                elif u.text:
                    with open(output_path, "w", encoding="utf8") as f:
                        f.write(u.text)
            return
        elif output_format == TextFileType.TEXTGRID.value:
            max_time = self.sound_file.duration
            tiers = {}
            for speaker in self.speakers:
                tiers[speaker.speaker.name] = textgrid.IntervalTier(
                    speaker.speaker.name, [], minT=0, maxT=max_time
                )

            tg = textgrid.Textgrid()
            tg.maxTimestamp = max_time
            for utterance in self.utterances:

                if save_transcription:
                    tiers[utterance.speaker.name].entryList.append(
                        Interval(
                            start=utterance.begin,
                            end=utterance.end,
                            label=utterance.transcription_text
                            if utterance.transcription_text
                            else "",
                        )
                    )
                else:
                    if tiers[utterance.speaker.name].entryList:
                        if tiers[utterance.speaker.name].entryList[-1].end > utterance.begin:
                            utterance.begin = tiers[utterance.speaker.name].entryList[-1].end
                    if utterance.end > self.duration:
                        utterance.end = self.duration
                    tiers[utterance.speaker.name].entryList.append(
                        Interval(
                            start=utterance.begin, end=utterance.end, label=utterance.text.strip()
                        )
                    )
            for t in tiers.values():
                tg.addTier(t)
            tg.save(output_path, includeBlankSpaces=True, format=output_format)

    def construct_transcription_tiers(self) -> typing.Dict[str, typing.List[CtmInterval]]:
        """
        Construct output transcription tiers for the file

        Returns
        -------
        dict[str, list[:class:`~montreal_forced_aligner.data.CtmInterval`]]
            Tier dictionary of utterance transcriptions
        """
        data = {}
        for u in self.utterances:
            speaker_name = ""
            for speaker in self.speakers:
                if u.speaker_id == speaker.speaker.id:
                    speaker_name = speaker.speaker.name
                    break
            if speaker_name not in data:
                data[speaker_name] = []
            label = u.transcription_text
            if not label:
                label = ""
            data[speaker_name].append(CtmInterval(u.begin, u.end, label, u.id))
        return data

    def construct_output_tiers(
        self,
    ) -> typing.Dict[str, typing.Dict[str, typing.List[CtmInterval]]]:
        """
        Construct aligned output tiers for a file

        Returns
        -------
        dict[str, dict[str, list[CtmInterval]]]
            Per-speaker aligned "words" and "phones" tiers
        """
        data = {}
        for utt in self.utterances:
            if utt.speaker.name not in data:
                data[utt.speaker.name] = {"words": [], "phones": []}
            for wi in utt.word_intervals:
                data[utt.speaker.name]["words"].append(
                    CtmInterval(wi.begin, wi.end, wi.label, utt.id)
                )

            for pi in utt.phone_intervals:
                data[utt.speaker.name]["phones"].append(
                    CtmInterval(pi.begin, pi.end, pi.label, utt.id)
                )
        return data

    def construct_output_path(
        self,
        output_directory: typing.Optional[str] = None,
        output_format: str = TextgridFormats.SHORT_TEXTGRID,
    ) -> str:
        """
        Construct the output path for the File

        Parameters
        ----------
        output_directory: str, optional
            Directory to output to, if None, it will overwrite the original file
        output_format: str
            File format to save in, one of ``lab``, ``long_textgrid``, ``short_textgrid`` (the default), or ``json``

        Returns
        -------
        str
            Output path
        """
        if output_format.upper() == "LAB":
            extension = ".lab"
        elif output_format.upper() == "JSON":
            extension = ".json"
        else:
            extension = ".TextGrid"
        if output_directory is None:
            if self.text_file is None or not self.text_file.text_file_path.endswith(extension):
                return os.path.splitext(self.sound_file.sound_file_path)[0] + extension
            return self.text_file.text_file_path
        if self.relative_path:
            relative = os.path.join(output_directory, self.relative_path)
        else:
            relative = output_directory
        output_path = os.path.join(relative, self.name + extension)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path


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
    file: File = relationship("File", back_populates="sound_file")
    sound_file_path = Column(String, nullable=False)
    format = Column(String, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    num_channels = Column(Integer, nullable=False)
    sox_string = Column(String)

    waveform: np.array

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
            y /= np.max(np.abs(y), axis=0)
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
    file: File = relationship("File", back_populates="text_file")
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
    normalized_text_int: str
        Space-delimited list of the normalized text converted to integer IDs for use in Kaldi programs
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
        Aligned phone intervals
    reference_phone_intervals: list[:class:`~montreal_forced_aligner.db.ReferencePhoneInterval`]
        Reference phone intervals
    word_intervals: list[:class:`~montreal_forced_aligner.db.WordInterval`]
        Aligned word intervals

    """

    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    duration = Column(Float, nullable=False, index=True)
    channel = Column(Integer, nullable=False)
    num_frames = Column(Integer)
    original_text = Column(String)
    text = Column(String)
    oovs = Column(String)
    normalized_text = Column(String)
    transcription_text = Column(String)
    normalized_text_int = Column(String)
    features = Column(String)
    in_subset = Column(Boolean, nullable=False, default=False, index=True)
    ignored = Column(Boolean, nullable=False, default=False, index=True)
    alignment_log_likelihood = Column(Float)
    phone_error_rate = Column(Float)
    alignment_score = Column(Float)
    word_error_rate = Column(Float)
    character_error_rate = Column(Float)
    file_id = Column(Integer, ForeignKey("file.id"), index=True, nullable=False)
    speaker_id = Column(Integer, ForeignKey("speaker.id"), index=True, nullable=False)
    file: File = relationship("File", back_populates="utterances")
    speaker: Speaker = relationship("Speaker", back_populates="utterances")
    phone_intervals = relationship(
        "PhoneInterval",
        back_populates="utterance",
        order_by="PhoneInterval.begin",
        collection_class=ordering_list("begin"),
    )
    reference_phone_intervals = relationship(
        "ReferencePhoneInterval",
        back_populates="utterance",
        order_by="ReferencePhoneInterval.begin",
        collection_class=ordering_list("begin"),
    )
    word_intervals = relationship(
        "WordInterval",
        back_populates="utterance",
        order_by="WordInterval.begin",
        collection_class=ordering_list("begin"),
    )
    kaldi_id = column_property(
        speaker_id.cast(sqlalchemy.VARCHAR) + "-" + id.cast(sqlalchemy.VARCHAR)
    )

    __table_args__ = (
        sqlalchemy.Index(
            "utterance_position_index", "file_id", "speaker_id", "begin", "end", "channel"
        ),
    )

    def __repr__(self):
        """String representation of the utterance object"""
        return f"<Utterance in {self.file_name} by {self.speaker_name} from {self.begin} to {self.end}>"

    @property
    def file_name(self):
        """Name of the utterance's file"""
        return self.file.name

    @property
    def speaker_name(self):
        """Name of the utterance's speaker"""
        return self.speaker.name

    def to_data(self):
        """
        Construct an UtteranceData object that can be used in multiprocessing

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`
            Data for the utterance
        """
        from montreal_forced_aligner.corpus.classes import UtteranceData

        return UtteranceData(
            self.speaker_name,
            self.file_name,
            self.begin,
            self.end,
            self.channel,
            self.text,
            self.normalized_text.split(),
            self.normalized_text_int.split(),
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
            duration=data.end - data.begin,
            channel=data.channel,
            oovs=" ".join(sorted(data.oovs)),
            normalized_text=" ".join(data.normalized_text),
            text=data.text,
            normalized_text_int=" ".join(str(x) for x in data.normalized_text_int),
            num_frames=num_frames,
            file=file,
            speaker_id=speaker,
        )


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
    label: str
        Text label of the interval
    utterance_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "phone_interval"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    utterance_id = Column(Integer, ForeignKey("utterance.id"), index=True, nullable=False)
    utterance: Utterance = relationship("Utterance", back_populates="phone_intervals")

    @classmethod
    def from_ctm(self, interval: CtmInterval, utterance: Utterance) -> PhoneInterval:
        """
        Construct a PhoneInterval from a CtmInterval object

        Parameters
        ----------
        interval: :class:`~montreal_forced_aligner.data.CtmInterval`
            CtmInterval containing data for the phone interval
        utterance: :class:`~montreal_forced_aligner.db.Utterance`
            Utterance object that the phone interval belongs to

        Returns
        -------
        :class:`~montreal_forced_aligner.db.PhoneInterval`
            Phone interval object
        """
        return PhoneInterval(
            begin=interval.begin, end=interval.end, label=interval.label, utterance=utterance
        )

    def as_ctm(self) -> CtmInterval:
        """
        Generate a CtmInterval from the database object

        Returns
        -------
        :class:`~montreal_forced_aligner.data.CtmInterval`
            CTM interval object
        """
        return CtmInterval(self.begin, self.end, self.label, self.utterance_id)


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
    label: str
        Text label of the interval
    utterance_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "word_interval"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    utterance_id = Column(Integer, ForeignKey("utterance.id"), index=True, nullable=False)
    utterance: Utterance = relationship("Utterance", back_populates="word_intervals")

    @classmethod
    def from_ctm(self, interval: CtmInterval, utterance: Utterance) -> WordInterval:
        """
        Construct a WordInterval from a CtmInterval object

        Parameters
        ----------
        interval: :class:`~montreal_forced_aligner.data.CtmInterval`
            CtmInterval containing data for the word interval
        utterance: :class:`~montreal_forced_aligner.db.Utterance`
            Utterance object that the word interval belongs to

        Returns
        -------
        :class:`~montreal_forced_aligner.db.WordInterval`
            Word interval object
        """
        return WordInterval(
            begin=interval.begin, end=interval.end, label=interval.label, utterance=utterance
        )

    def as_ctm(self) -> CtmInterval:
        """
        Generate a CtmInterval from the database object

        Returns
        -------
        :class:`~montreal_forced_aligner.data.CtmInterval`
            CTM interval object
        """
        return CtmInterval(self.begin, self.end, self.label, self.utterance_id)


class ReferencePhoneInterval(MfaSqlBase):
    """

    Database class for storing information about reference phone intervals

    Parameters
    ----------
    id: int
        Primary key
    begin: float
        Beginning timestamp of the interval
    end: float
        Ending timestamp of the interval
    label: str
        Text label of the interval
    utterance_id: int
        Foreign key to :class:`~montreal_forced_aligner.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "reference_phone_interval"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    utterance_id = Column(Integer, ForeignKey("utterance.id"), index=True, nullable=False)
    utterance: Utterance = relationship("Utterance", back_populates="reference_phone_intervals")

    def as_ctm(self) -> CtmInterval:
        """
        Generate a CtmInterval from the database object

        Returns
        -------
        :class:`~montreal_forced_aligner.data.CtmInterval`
            CTM interval object
        """
        return CtmInterval(self.begin, self.end, self.label, self.utterance_id)
