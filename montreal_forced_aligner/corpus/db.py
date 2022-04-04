"""Database classes"""
from __future__ import annotations

import os
import typing

import librosa
import numpy as np
import sqlalchemy
from praatio import textgrid
from praatio.utilities.constants import Interval, TextgridFormats
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import column_property, declarative_base, relationship

from montreal_forced_aligner.data import CtmInterval, TextFileType

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.corpus.classes import UtteranceData

Base = declarative_base()

__all__ = [
    "Dictionary",
    "Word",
    "Phone",
    "File",
    "TextFile",
    "SoundFile",
    "Speaker",
    "SpeakerOrdering",
    "Utterance",
    "PhoneInterval",
    "WordInterval",
    "ReferencePhoneInterval",
]


class Corpus(Base):
    __tablename__ = "corpus"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    imported = Column(Boolean, default=False)
    features_generated = Column(Boolean, default=False)
    alignment_done = Column(Boolean, default=False)
    transcription_done = Column(Boolean, default=False)
    alignment_evaluation_done = Column(Boolean, default=False)
    has_reference_alignments = Column(Boolean, default=False)


class Dictionary(Base):
    """
    Database class for storing information about a
    :class:`~montreal_forced_aligner.dictionary.pronunciation.PronunciationDictionary`

    Parameters
    ----------
    id: int
        Primary key
    name: str
        Name of the dictionary
    phones_directory: str
        Directory containing saved files related to phone information
    lexicon_fst_path: str
        Path to the dictionary's lexicon fst
    lexicon_disambig_fst_path: str
        Path to the dictionary's disambiguated lexicon fst
    words_path: str
        Path to the dictionary's word symbol table
    word_boundary_int_path: str
        Path to the dictionary's file containing integer IDs for word boundary phones
    silence_word: str
        Optional silence word
    optional_silence_phone: str
        Optional silence phone
    oov_word: str
        Word representing out of vocabulary items
    bracketed_word: str
        Word representing out of vocabulary items surrounded by brackets (cutoffs, laughter, etc)
    position_dependent_phones: bool
        Flag for using position dependent phones
    clitic_marker: str
        Character that splits words into roots and clitics
    words: Collection[:class:`~montreal_forced_aligner.corpus.db.Word`]
        Words in the dictionary
    speakers: Collection[:class:`~montreal_forced_aligner.corpus.db.Speaker`]
        Speakers in the corpus that use the dictionary
    """

    __tablename__ = "dictionary"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    phones_directory = Column(String, nullable=False)
    lexicon_fst_path = Column(String, nullable=False)
    lexicon_disambig_fst_path = Column(String, nullable=False)
    words_path = Column(String, nullable=False)
    word_boundary_int_path = Column(String, nullable=False)
    silence_word = Column(String(10), nullable=False)
    optional_silence_phone = Column(String(15), nullable=False)
    oov_word = Column(String(15), nullable=False)
    bracketed_word = Column(String(15), nullable=False)
    position_dependent_phones = Column(Boolean, nullable=False)
    clitic_marker = Column(String(1), nullable=True)
    words = relationship("Word", back_populates="dictionary")
    speakers = relationship("Speaker", back_populates="dictionary")


class Phone(Base):
    """
    Database class for storing phones and their integer IDs

    Parameters
    ----------
    id: int
        Integer ID of the phone
    phone: str
        Phone label
    """

    __tablename__ = "phone"

    id = Column(Integer, primary_key=True)
    phone = Column(String(10), unique=True, nullable=False)


class Word(Base):
    """
    Database class for storing words, their integer IDs, and pronunciation information

    Parameters
    ----------
    id: int
        Integer ID of the phone
    phone: str
        Phone label
    pronunciations: str
        String of pronunciation variants delimited by ";"`" (phones in the pronunciations are delimited with " ")
    dictionary_id: int
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Dictionary`
    dictionary: :class:`~montreal_forced_aligner.corpus.db.Dictionary`
        Pronunciation dictionary that the word belongs to
    """

    __tablename__ = "word"

    id = Column(Integer, primary_key=True)
    word = Column(String, nullable=False, index=True)
    pronunciations = Column(String, nullable=False)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"), primary_key=True)
    dictionary: Dictionary = relationship("Dictionary", back_populates="words")


class Speaker(Base):
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
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Dictionary`
    dictionary: :class:`~montreal_forced_aligner.corpus.db.Dictionary`
        Pronunciation dictionary that the speaker uses
    utterances: Collection[:class:`~montreal_forced_aligner.corpus.db.Utterance`]
        Utterances for the speaker
    files: Collection[:class:`~montreal_forced_aligner.corpus.db.File`]
        Files that the speaker spoke in
    """

    __tablename__ = "speaker"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    cmvn = Column(String)
    job_id = Column(Integer)
    dictionary_id = Column(Integer, ForeignKey("dictionary.id"))
    dictionary: Dictionary = relationship("Dictionary", back_populates="speakers")
    utterances = relationship("Utterance", back_populates="speaker")
    files = relationship("SpeakerOrdering", back_populates="speaker")


class SpeakerOrdering(Base):
    """
    Mapping class between :class:`~montreal_forced_aligner.corpus.db.Speaker`
    and :class:`~montreal_forced_aligner.corpus.db.File` that preserves the order of tiers

    Parameters
    ----------
    speaker_id: int
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Speaker`
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.File`
    index: int
        Position of speaker in the input TextGrid
    speaker: :class:`~montreal_forced_aligner.corpus.db.Speaker`
        Speaker object
    file: :class:`~montreal_forced_aligner.corpus.db.File`
        File object
    """

    __tablename__ = "speaker_ordering"
    speaker_id = Column(ForeignKey("speaker.id"), primary_key=True)
    file_id = Column(ForeignKey("file.id"), primary_key=True)
    index = Column(Integer)
    speaker: Speaker = relationship("Speaker", back_populates="files")
    file: File = relationship("File", back_populates="speakers")


class File(Base):
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
    text_file: :class:`~montreal_forced_aligner.corpus.db.TextFile`
        TextFile object with information about the transcript of a file
    sound_file: :class:`~montreal_forced_aligner.corpus.db.SoundFile`
        SoundFile object with information about the audio of a file
    speakers: Collection[:class:`~montreal_forced_aligner.corpus.db.SpeakerOrdering`]
        Speakers in the file ordered by their index
    utterances: Collection[:class:`~montreal_forced_aligner.corpus.db.Utterance`]
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
        return len(self.speakers)

    @property
    def num_utterances(self) -> int:
        return len(self.utterances)

    @property
    def duration(self) -> float:
        return self.sound_file.duration

    @property
    def num_channels(self) -> int:
        return self.sound_file.num_channels

    @property
    def sample_rate(self) -> int:
        return self.sound_file.sample_rate

    def has_speaker(self, speaker_name):
        for s in self.speakers:
            if s.speaker.name == speaker_name:
                return True
        return False

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
        if output_format is None:
            output_format = self.text_file.file_type
            if not output_format or output_format == TextFileType.NONE.value:
                if utterance_count == 1:
                    output_format = TextFileType.LAB.value
                else:
                    output_format = TextFileType.TEXTGRID.value
        output_path = self.construct_output_path(output_directory, output_format=output_format)
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
            output_path = self.construct_output_path(output_directory, output_format)
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
                    tiers[utterance.speaker.name].entryList.append(
                        Interval(start=utterance.begin, end=utterance.end, label=utterance.text)
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
            if self.text_file is None:
                return os.path.splitext(self.sound_file.sound_file_path)[0] + extension
            return self.text_file.text_file_path
        if self.relative_path:
            relative = os.path.join(output_directory, self.relative_path)
        else:
            relative = output_directory
        output_path = os.path.join(relative, self.name + extension)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path


class SoundFile(Base):
    """

    Database class for storing information about sound files

    Parameters
    ----------
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.File`
    file: :class:`~montreal_forced_aligner.corpus.db.File`
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


class TextFile(Base):
    """
    Database class for storing information about transcription files

    Parameters
    ----------
    file_id: int
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.File`
    file: :class:`~montreal_forced_aligner.corpus.db.File`
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


class Utterance(Base):
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
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.File`
    speaker_id: int
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Speaker`
    file: :class:`~montreal_forced_aligner.corpus.db.File`
        File object that the utterance is from
    speaker: :class:`~montreal_forced_aligner.corpus.db.Speaker`
        Speaker object of the utterance
    phone_intervals: Collection[:class:`~montreal_forced_aligner.corpus.db.PhoneInterval`]
        Aligned phone intervals
    reference_phone_intervals: Collection[:class:`~montreal_forced_aligner.corpus.db.ReferencePhoneInterval`]
        Reference phone intervals
    word_intervals: Collection[:class:`~montreal_forced_aligner.corpus.db.WordInterval`]
        Aligned word intervals

    """

    __tablename__ = "utterance"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    duration = Column(Float, nullable=False, index=True)
    channel = Column(Integer, nullable=False)
    num_frames = Column(Integer)
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
    file_id = Column(Integer, ForeignKey("file.id"), nullable=False)
    speaker_id = Column(Integer, ForeignKey("speaker.id"), nullable=False)
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
        return f"<Utterance in {self.file_name} by {self.speaker_name} from {self.begin} to {self.end}>"

    @property
    def file_name(self):
        return self.file.name

    @property
    def speaker_name(self):
        return self.speaker.name

    def to_data(self):
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
    def from_data(cls, data: UtteranceData, file: File, speaker: Speaker, frame_shift: int = None):
        """
        Generate an utterance object from :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`

        Parameters
        ----------
        data: :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`
            Data for the utterance
        file: :class:`~montreal_forced_aligner.corpus.db.File`
            File database object for the utterance
        speaker: :class:`~montreal_forced_aligner.corpus.db.Speaker`
            Speaker database object for the utterance
        frame_shift: int, optional
            Frame shift in ms to use for calculating the number of frames in the utterance

        Returns
        -------
        :class:`~montreal_forced_aligner.corpus.db.Utterance`
            Utterance object
        """
        if isinstance(speaker, Speaker):
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


class PhoneInterval(Base):
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
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.corpus.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "phone_interval"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    utterance_id = Column(Integer, ForeignKey("utterance.id"), nullable=False)
    utterance: Utterance = relationship("Utterance", back_populates="phone_intervals")

    @classmethod
    def from_ctm(self, interval: CtmInterval, utterance: Utterance) -> PhoneInterval:
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


class WordInterval(Base):
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
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.corpus.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "word_interval"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    utterance_id = Column(Integer, ForeignKey("utterance.id"), nullable=False)
    utterance: Utterance = relationship("Utterance", back_populates="word_intervals")

    @classmethod
    def from_ctm(self, interval: CtmInterval, utterance: Utterance) -> WordInterval:
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


class ReferencePhoneInterval(Base):
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
        Foreign key to :class:`~montreal_forced_aligner.corpus.db.Utterance`
    utterance: :class:`~montreal_forced_aligner.corpus.db.Utterance`
        Utterance of the interval
    """

    __tablename__ = "reference_phone_interval"

    id = Column(Integer, primary_key=True)
    begin = Column(Float, nullable=False, index=True)
    end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    utterance_id = Column(Integer, ForeignKey("utterance.id"), nullable=False)
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
