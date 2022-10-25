"""Class definitions for corpora"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import typing
from abc import ABCMeta, abstractmethod
from collections import Counter
from queue import Empty

import sqlalchemy.engine
import tqdm
from sqlalchemy.orm import Session, joinedload, selectinload

from montreal_forced_aligner.abc import DatabaseMixin, MfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.classes import FileData, UtteranceData
from montreal_forced_aligner.corpus.multiprocessing import Job, NormalizationWorker, normalize
from montreal_forced_aligner.data import DatabaseImportData, TextFileType, WordType, WorkflowType
from montreal_forced_aligner.db import Corpus, CorpusWorkflow, Dialect, Dictionary, File
from montreal_forced_aligner.db import Job as JobTable
from montreal_forced_aligner.db import (
    OovWord,
    SoundFile,
    Speaker,
    SpeakerOrdering,
    TextFile,
    Utterance,
    Word,
    bulk_update,
)
from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionaryMixin
from montreal_forced_aligner.exceptions import CorpusError
from montreal_forced_aligner.helper import output_mapping
from montreal_forced_aligner.utils import Stopped

__all__ = ["CorpusMixin"]


class CorpusMixin(MfaWorker, DatabaseMixin, metaclass=ABCMeta):
    """
    Mixin class for processing corpora

    Notes
    -----
    Using characters in files to specify speakers is generally finicky and leads to errors, so I would not
    recommend using it.  Additionally, consider it deprecated and could be removed in future versions

    Parameters
    ----------
    corpus_directory: str
        Path to corpus
    speaker_characters: int or str, optional
        Number of characters in the file name to specify the speaker
    ignore_speakers: bool
        Flag for whether to discard any parsed speaker information during top-level worker's processing

    See Also
    --------
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters

    Attributes
    ----------
    jobs: list[:class:`~montreal_forced_aligner.corpus.multiprocessing.Job`]
        List of jobs for processing the corpus and splitting speakers
    word_counts: Counter
        Counts of words in the corpus
    stopped: Stopped
        Stop check for loading the corpus
    decode_error_files: list[str]
        List of text files that could not be loaded with utf8
    textgrid_read_errors: list[str]
        List of TextGrid files that had an error in loading
    """

    def __init__(
        self,
        corpus_directory: str,
        speaker_characters: typing.Union[int, str] = 0,
        ignore_speakers: bool = False,
        **kwargs,
    ):
        if not os.path.exists(corpus_directory):
            raise CorpusError(f"The directory '{corpus_directory}' does not exist.")
        if not os.path.isdir(corpus_directory):
            raise CorpusError(
                f"The specified path for the corpus ({corpus_directory}) is not a directory."
            )
        self._speaker_ids = {}
        self.corpus_directory = corpus_directory
        self.speaker_characters = speaker_characters
        self.ignore_speakers = ignore_speakers
        self.word_counts = Counter()
        self.stopped = Stopped()
        self.decode_error_files = []
        self.textgrid_read_errors = []
        self.jobs: typing.List[Job] = []
        self._num_speakers = None
        self._num_utterances = None
        self._num_files = None
        super().__init__(**kwargs)
        os.makedirs(self.corpus_output_directory, exist_ok=True)
        self.imported = False
        self.text_normalized = False
        self._current_speaker_index = 1
        self._current_file_index = 1
        self._speaker_ids = {}
        self._word_set = []

    def inspect_database(self) -> None:
        """Check if a database file exists and create the necessary metadata"""
        self.initialize_database()
        with self.session() as session:
            corpus = session.query(Corpus).first()
            if corpus:
                self.imported = corpus.imported
                self.text_normalized = corpus.text_normalized
            else:
                session.add(Corpus(name=self.data_source_identifier))
                session.commit()

    def get_utterances(
        self,
        id: typing.Optional[int] = None,
        file: typing.Optional[typing.Union[str, int]] = None,
        speaker: typing.Optional[typing.Union[str, int]] = None,
        begin: typing.Optional[float] = None,
        end: typing.Optional[float] = None,
        session: Session = None,
    ):
        """
        Get a file from search parameters

        Parameters
        ----------
        id: int
            Integer ID to look up
        file: str or int
            File name or ID to look up
        speaker: str or int
            Speaker name or ID to look up
        begin: float
            Begin timestamp to look up
        end: float
            Ending timestamp to look up

        Returns
        -------
        :class:`~montreal_forced_aligner.db.Utterance`
            Utterance match
        """
        if session is None:
            session = self.session()
        if id is not None:
            utterance = session.query(Utterance).get(id)
            if not utterance:
                raise Exception(f"Could not find utterance with id of {id}")
            return utterance
        else:
            utterance = session.query(Utterance)
            if file is not None:
                utterance = utterance.join(Utterance.file)
                if isinstance(file, int):
                    utterance = utterance.filter(File.id == file)
                else:
                    utterance = utterance.filter(File.name == file)
            if speaker is not None:
                utterance = utterance.join(Utterance.speaker)
                if isinstance(speaker, int):
                    utterance = utterance.filter(Speaker.id == speaker)
                else:
                    utterance = utterance.filter(Speaker.name == speaker)
            if begin is not None:
                utterance = utterance.filter(Utterance.begin == begin)
            if end is not None:
                utterance = utterance.filter(Utterance.end == end)
            utterance = utterance.all()
            return list(utterance)

    def get_file(
        self, id: typing.Optional[int] = None, name=None, session: Session = None
    ) -> File:
        """
        Get a file from search parameters

        Parameters
        ----------
        id: int
            Integer ID to look up
        name: str
            File name to look up

        Returns
        -------
        :class:`~montreal_forced_aligner.db.File`
            File match
        """
        if session is None:
            session = self.session()
        file = session.query(File).options(
            selectinload(File.utterances).joinedload(Utterance.speaker, innerjoin=True),
            joinedload(File.sound_file, innerjoin=True),
            joinedload(File.text_file, innerjoin=True),
            selectinload(File.speakers).joinedload(SpeakerOrdering.speaker, innerjoin=True),
        )
        if id is not None:
            file = file.get(id)
            if not file:
                raise Exception(f"Could not find utterance with id of {id}")
            return file
        else:
            file = file.filter(File.name == name).first()
            if not file:
                raise Exception(f"Could not find utterance with name of {name}")
            return file

    @property
    def corpus_meta(self) -> typing.Dict[str, typing.Any]:
        """Corpus metadata"""
        return {}

    @property
    def features_log_directory(self) -> str:
        """Feature log directory"""
        return os.path.join(self.split_directory, "log")

    @property
    def split_directory(self) -> str:
        """Directory used to store information split by job"""
        return os.path.join(self.corpus_output_directory, f"split{GLOBAL_CONFIG.num_jobs}")

    def _write_spk2utt(self) -> None:
        """Write spk2utt scp file for Kaldi"""
        data = {}
        utt2spk_data = {}
        with self.session() as session:
            utterances = session.query(Utterance.kaldi_id, Utterance.speaker_id).order_by(
                Utterance.kaldi_id
            )

            for utt_id, speaker_id in utterances:
                if speaker_id not in data:
                    data[speaker_id] = []
                data[speaker_id].append(utt_id)
                utt2spk_data[utt_id] = speaker_id

        output_mapping(utt2spk_data, os.path.join(self.corpus_output_directory, "utt2spk.scp"))
        output_mapping(data, os.path.join(self.corpus_output_directory, "spk2utt.scp"))

    def create_corpus_split(self) -> None:
        """Create split directory and output information from Jobs"""
        split_dir = self.split_directory
        os.makedirs(os.path.join(split_dir, "log"), exist_ok=True)
        with self.session() as session:
            for job in self.jobs:
                job.output_to_directory(split_dir, session)

    @property
    def corpus_word_set(self) -> typing.List[str]:
        """Set of words used in the corpus"""
        if not self._word_set:
            with self.session() as session:
                self._word_set = [
                    x[0]
                    for x in session.query(Word.word).filter(Word.count > 0).order_by(Word.word)
                ]
        return self._word_set

    def add_utterance(self, utterance: UtteranceData, session: Session = None) -> Utterance:
        """
        Add an utterance to the corpus

        Parameters
        ----------
        utterance: :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`
            Utterance to add
        """
        close = False
        if session is None:
            session = self.session()
            close = True

        speaker_obj = session.query(Speaker).filter_by(name=utterance.speaker_name).first()
        if not speaker_obj:
            dictionary = None
            if hasattr(self, "get_dictionary"):
                dictionary = (
                    session.query(Dictionary)
                    .filter_by(name=self.get_dictionary(utterance.speaker_name).name)
                    .first()
                )
            speaker_obj = Speaker(name=utterance.speaker_name, dictionary=dictionary)
            session.add(speaker_obj)
            self._speaker_ids[utterance.speaker_name] = speaker_obj
        else:
            self._speaker_ids[utterance.speaker_name] = speaker_obj
        file_obj = session.query(File).filter_by(name=utterance.file_name).first()
        u = Utterance.from_data(
            utterance, file_obj, speaker_obj, frame_shift=getattr(self, "frame_shift", None)
        )
        session.add(u)
        if close:
            session.commit()
            session.close()
        return u

    def delete_utterance(self, utterance_id: int, session: Session = None) -> None:
        """
        Delete an utterance from the corpus

        Parameters
        ----------
        utterance_id: int
            Utterance to delete
        """
        close = False
        if session is None:
            session = self.session()
            close = True

        session.query(Utterance).filter(Utterance.id == utterance_id).delete()
        session.commit()
        if close:
            session.close()

    def speakers(self, session: Session = None) -> sqlalchemy.orm.Query:
        """
        Get all speakers in the corpus

        Parameters
        ----------
        session: sqlalchemy.orm.Session, optional
           Session to use in querying

        Returns
        -------
        sqlalchemy.orm.Query
            Speaker query
        """
        close = False
        if session is None:
            session = self.session()
            close = True
        speakers = session.query(Speaker).options(
            selectinload(Speaker.utterances),
            selectinload(Speaker.files).joinedload(SpeakerOrdering.file, innerjoin=True),
            joinedload(Speaker.dictionary),
        )
        if close:
            session.close()
        return speakers

    def files(self, session: Session = None) -> sqlalchemy.orm.Query:
        """
        Get all files in the corpus

        Parameters
        ----------
        session: sqlalchemy.orm.Session, optional
           Session to use in querying

        Returns
        -------
        sqlalchemy.orm.Query
            File query
        """
        close = False
        if session is None:
            session = self.session()
            close = True
        files = session.query(File).options(
            selectinload(File.utterances),
            selectinload(File.speakers).joinedload(SpeakerOrdering.speaker, innerjoin=True),
            joinedload(File.sound_file),
            joinedload(File.text_file),
        )
        if close:
            session.close()
        return files

    def utterances(self, session: Session = None) -> sqlalchemy.orm.Query:
        """
        Get all utterances in the corpus

        Parameters
        ----------
        session: sqlalchemy.orm.Session, optional
           Session to use in querying

        Returns
        -------
        :class:`sqlalchemy.orm.Query`
            Utterance query
        """
        close = False
        if session is None:
            session = Session(self.db_engine)
            close = True
        utterances = session.query(Utterance).options(
            joinedload(Utterance.file, innerjoin=True),
            joinedload(Utterance.speaker, innerjoin=True),
            selectinload(Utterance.phone_intervals),
            selectinload(Utterance.word_intervals),
        )
        if close:
            session.close()
        return utterances

    def initialize_jobs(self) -> None:
        """
        Initialize the corpus's Jobs
        """
        self.log_info("Initializing multiprocessing jobs...")

        with self.session() as session:
            if self.num_speakers < GLOBAL_CONFIG.num_jobs and not GLOBAL_CONFIG.single_speaker:
                self.log_warning(
                    f"Number of jobs was specified as {GLOBAL_CONFIG.num_jobs}, "
                    f"but due to only having {self.num_speakers} speakers, MFA "
                    f"will only use {self.num_speakers} jobs. Use the --single_speaker flag if you would like to split "
                    f"utterances across jobs regardless of their speaker."
                )
                GLOBAL_CONFIG.num_jobs = self.num_speakers
            session.commit()
            self.jobs = [Job(i, self.db_engine) for i in range(GLOBAL_CONFIG.num_jobs)]
            update_mappings = []
            if GLOBAL_CONFIG.single_speaker:
                utts_per_job = int(self.num_utterances / GLOBAL_CONFIG.num_jobs)
                if utts_per_job == 0:
                    utts_per_job = 1
                for j in range(min(GLOBAL_CONFIG.num_jobs, self.num_utterances)):
                    update_mappings.extend(
                        {"id": u, "job_id": j}
                        for u in range((utts_per_job * j) + 1, (utts_per_job * (j + 1)) + 1)
                    )
                for j_id in range(j + 1, GLOBAL_CONFIG.num_jobs):
                    self.jobs[j_id].has_data = False
                last_ind = update_mappings[-1]["id"] + 1
                for u in range(last_ind, self.num_utterances):
                    update_mappings.append({"id": u, "job_id": GLOBAL_CONFIG.num_jobs - 1})
                bulk_update(session, Utterance, update_mappings)
            else:
                utt_counts = {i: 0 for i in range(GLOBAL_CONFIG.num_jobs)}
                speakers = (
                    session.query(Speaker.id, sqlalchemy.func.count(Utterance.id))
                    .outerjoin(Speaker.utterances)
                    .group_by(Speaker.id)
                    .order_by(sqlalchemy.func.count(Utterance.id).desc())
                )
                for s_id, speaker_utt_count in speakers:
                    if not speaker_utt_count:
                        continue
                    job_id = min(utt_counts.keys(), key=lambda x: utt_counts[x])
                    update_mappings.append({"speaker_id": s_id, "job_id": job_id})
                    utt_counts[job_id] += speaker_utt_count
                bulk_update(session, Utterance, update_mappings, id_field="speaker_id")
            session.commit()
            for j in self.jobs:
                j.refresh_dictionaries(session)

    def _finalize_load(self, session: Session, import_data: DatabaseImportData):
        """Finalize the import of database objects after parsing"""
        with session.begin_nested():

            job_objs = [{"id": j} for j in range(GLOBAL_CONFIG.num_jobs)]
            session.execute(sqlalchemy.insert(JobTable.__table__), job_objs)
            if import_data.speaker_objects:
                session.execute(sqlalchemy.insert(Speaker.__table__), import_data.speaker_objects)
            if import_data.file_objects:
                session.execute(sqlalchemy.insert(File.__table__), import_data.file_objects)
            if import_data.text_file_objects:
                session.execute(
                    sqlalchemy.insert(TextFile.__table__), import_data.text_file_objects
                )
            if import_data.sound_file_objects:
                session.execute(
                    sqlalchemy.insert(SoundFile.__table__), import_data.sound_file_objects
                )
            if import_data.speaker_ordering_objects:
                session.execute(
                    sqlalchemy.insert(SpeakerOrdering.__table__),
                    import_data.speaker_ordering_objects,
                )
            if import_data.utterance_objects:
                session.execute(
                    sqlalchemy.insert(Utterance.__table__), import_data.utterance_objects
                )
            session.flush()
            if import_data.word_mapping:
                if isinstance(self, MultispeakerDictionaryMixin):
                    bulk_update(session, Word, list(import_data.word_mapping.values()))
                else:
                    dialect = Dialect(name="unspecified")
                    d = Dictionary(name="corpus", default=True, dialect=dialect)
                    session.add(dialect)
                    session.add(d)
                    session.flush()
                    index = 1
                    mapping = []
                    for word in sorted(import_data.word_mapping.keys()):
                        import_data.word_mapping[word]["id"] = index
                        import_data.word_mapping[word]["word"] = word
                        import_data.word_mapping[word]["mapping_id"] = index
                        import_data.word_mapping[word]["dictionary_id"] = d.id
                        index += 1
                        mapping.append(
                            {
                                "id": index,
                                "word": word,
                                "mapping_id": index,
                                "dictionary_id": d.id,
                                "word_type": WordType.speech,
                                "count": import_data.word_mapping[word]["count"],
                            }
                        )
                    session.execute(sqlalchemy.insert(Word.__table__), mapping)
                    session.flush()

        self.imported = True
        speakers = (
            session.query(Speaker.id)
            .outerjoin(Speaker.utterances)
            .group_by(Speaker.id)
            .having(sqlalchemy.func.count(Utterance.id) == 0)
        )
        self._speaker_ids = {}
        speaker_ids = [x[0] for x in speakers]
        session.query(Corpus).update(
            {
                "imported": True,
                "has_text_files": len(import_data.text_file_objects) > 0,
                "has_sound_files": len(import_data.sound_file_objects) > 0,
            }
        )
        if speaker_ids:
            session.query(Speaker).filter(Speaker.id.in_(speaker_ids)).delete()
            self._num_speakers = None
        self._num_utterances = None  # Recalculate if already cached
        self._num_files = None
        session.commit()

    def normalize_text(self) -> None:
        """Normalize the text of the corpus using a dictionary's sanitization functions and word mappings"""
        if self.text_normalized:
            return
        from montreal_forced_aligner.dictionary.multispeaker import SanitizeFunction

        sanitize_function = getattr(self, "sanitize_function", None)
        if sanitize_function is None or isinstance(sanitize_function, SanitizeFunction):
            return
        self.log_info("Normalizing text...")
        log_directory = os.path.join(self.split_directory, "log")
        words_mappings = {}
        os.makedirs(log_directory, exist_ok=True)
        update_mapping = []
        oov_words = {}
        with tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            sanitize_function = getattr(self, "sanitize_function", None)
            session.query(OovWord).delete()
            session.query(Word).update({"count": 0})
            session.commit()
            words = session.query(Word.id, Word.dictionary_id, Word.mapping_id)
            for w_id, d_id, m_id in words:
                words_mappings[(d_id, m_id)] = {"id": w_id, "count": 0}
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i in range(GLOBAL_CONFIG.num_jobs):
                    p = NormalizationWorker(
                        i, return_queue, self.read_only_db_string, sanitize_function, stopped
                    )
                    procs.append(p)
                    p.start()
                while True:
                    try:
                        result = return_queue.get(timeout=1)
                        if isinstance(result, Exception):
                            error_dict[getattr(result, "job_name", 0)] = result
                            continue
                        if stopped.stop_check():
                            continue
                    except Empty:
                        for proc in procs:
                            if not proc.finished.stop_check():
                                break
                        else:
                            break
                        continue
                    pbar.update(1)
                    result, speaker_name = result
                    if sanitize_function is not None:
                        dict_id = sanitize_function.get_dict_id_for_speaker(speaker_name)
                        for w in result["oovs"].split():
                            if (w, dict_id) not in oov_words:
                                oov_words[(w, dict_id)] = {
                                    "word": w,
                                    "count": 0,
                                    "dictionary_id": dict_id,
                                }
                            oov_words[(w, dict_id)]["count"] += 1
                        for m_id in result["normalized_text_int"].split():
                            m_id = int(m_id)
                            words_mappings[(dict_id, m_id)]["count"] += 1
                    update_mapping.append(result)
                for p in procs:
                    p.join()
                if error_dict:
                    for e in error_dict.values():
                        print(e)
                    self.dirty = True
                    sys.exit(1)
            else:
                for i in range(GLOBAL_CONFIG.num_jobs):
                    for result in normalize(i, session, sanitize_function):
                        pbar.update(1)
                        result, speaker_name = result
                        if sanitize_function is not None:
                            dict_id = sanitize_function.get_dict_id_for_speaker(speaker_name)
                            for w in result["oovs"].split():
                                if (w, dict_id) not in oov_words:
                                    oov_words[(w, dict_id)] = {
                                        "word": w,
                                        "count": 0,
                                        "dictionary_id": dict_id,
                                    }
                                oov_words[(w, dict_id)]["count"] += 1
                        for m_id in result["normalized_text_int"].split():
                            m_id = int(m_id)
                            words_mappings[(dict_id, m_id)]["count"] += 1
                        update_mapping.append(result)
            bulk_update(session, Utterance, update_mapping)
            bulk_update(session, Word, list(words_mappings.values()))
            if oov_words:
                session.execute(sqlalchemy.insert(OovWord.__table__), list(oov_words.values()))
            self.text_normalized = True
            session.query(Corpus).update({"text_normalized": True})
            session.commit()

    def add_speaker(self, name: str, session: Session = None):
        """
        Add a speaker to the corpus

        Parameters
        ----------
        name: str
            Name of the speaker
        session: sqlalchemy.orm.Session
            Database session, if not specified, will use a temporary session

        """
        if name in self._speaker_ids:
            return
        close = False
        if session is None:
            session = self.session()
            close = True

        speaker_obj = session.query(Speaker).filter_by(name=name).first()
        if not speaker_obj:
            dictionary = None
            if hasattr(self, "get_dictionary_id"):
                dictionary = session.query(Dictionary).get(self.get_dictionary_id(name))
            speaker_obj = Speaker(name=name, dictionary=dictionary)
            session.add(speaker_obj)
            session.flush()
            self._speaker_ids[name] = speaker_obj.id
        else:
            self._speaker_ids[name] = speaker_obj.id

        if close:
            session.commit()
            session.close()

    def add_file(self, file: FileData, session: Session = None):
        """
        Add a file to the corpus

        Parameters
        ----------
        file: :class:`~montreal_forced_aligner.corpus.classes.FileData`
            File to be added
        """
        close = False
        if session is None:
            session = self.session()
            close = True
        f = File(
            id=self._current_file_index,
            name=file.name,
            relative_path=file.relative_path,
            modified=False,
        )
        session.add(f)
        session.flush()
        for i, speaker in enumerate(file.speaker_ordering):
            if speaker not in self._speaker_ids:
                speaker_obj = Speaker(
                    id=self._current_speaker_index,
                    name=speaker,
                    dictionary_id=getattr(self, "_default_dictionary_id", None),
                )
                session.add(speaker_obj)
                self._speaker_ids[speaker] = self._current_speaker_index
                self._current_speaker_index += 1

            so = SpeakerOrdering(
                file_id=self._current_file_index,
                speaker_id=self._speaker_ids[speaker],
                index=i,
            )
            session.add(so)
        if file.wav_path is not None:
            sf = SoundFile(
                file_id=self._current_file_index,
                sound_file_path=file.wav_path,
                format=file.wav_info.format,
                sample_rate=file.wav_info.sample_rate,
                duration=file.wav_info.duration,
                num_channels=file.wav_info.num_channels,
                sox_string=file.wav_info.sox_string,
            )
            session.add(sf)
        if file.text_path is not None:
            text_type = file.text_type
            if isinstance(text_type, TextFileType):
                text_type = file.text_type.value

            tf = TextFile(
                file_id=self._current_file_index,
                text_file_path=file.text_path,
                file_type=text_type,
            )
            session.add(tf)
        frame_shift = getattr(self, "frame_shift", None)
        if frame_shift is not None:
            frame_shift = round(frame_shift / 1000, 4)
        for u in file.utterances:
            duration = u.end - u.begin
            num_frames = None
            if frame_shift is not None:
                num_frames = int(duration / frame_shift)
            utterance = Utterance(
                begin=u.begin,
                end=u.end,
                duration=duration,
                channel=u.channel,
                oovs=u.oovs,
                normalized_text=u.normalized_text,
                normalized_character_text=u.normalized_character_text,
                text=u.text,
                normalized_text_int=u.normalized_text_int,
                normalized_character_text_int=u.normalized_character_text_int,
                num_frames=num_frames,
                in_subset=False,
                ignored=False,
                file_id=self._current_file_index,
                speaker_id=self._speaker_ids[u.speaker_name],
            )
            session.add(utterance)

        if close:
            session.commit()
            session.close()
        self._current_file_index += 1

    def generate_import_objects(self, file: FileData) -> DatabaseImportData:
        """
        Add a file to the corpus

        Parameters
        ----------
        file: :class:`~montreal_forced_aligner.corpus.classes.FileData`
            File to be added
        """
        data = DatabaseImportData()
        data.file_objects.append(
            {
                "id": self._current_file_index,
                "name": file.name,
                "relative_path": file.relative_path,
                "modified": False,
            }
        )
        for i, speaker in enumerate(file.speaker_ordering):
            if speaker not in self._speaker_ids:
                data.speaker_objects.append(
                    {
                        "id": self._current_speaker_index,
                        "name": speaker,
                        "dictionary_id": getattr(self, "_default_dictionary_id", None),
                    }
                )
                self._speaker_ids[speaker] = self._current_speaker_index
                self._current_speaker_index += 1

            data.speaker_ordering_objects.append(
                {
                    "file_id": self._current_file_index,
                    "speaker_id": self._speaker_ids[speaker],
                    "index": i,
                }
            )
        if file.wav_path is not None:
            data.sound_file_objects.append(
                {
                    "file_id": self._current_file_index,
                    "sound_file_path": file.wav_path,
                    "format": file.wav_info.format,
                    "sample_rate": file.wav_info.sample_rate,
                    "duration": file.wav_info.duration,
                    "num_channels": file.wav_info.num_channels,
                    "sox_string": file.wav_info.sox_string,
                }
            )
        if file.text_path is not None:
            text_type = file.text_type
            if isinstance(text_type, TextFileType):
                text_type = file.text_type.value

            data.text_file_objects.append(
                {
                    "file_id": self._current_file_index,
                    "text_file_path": file.text_path,
                    "file_type": text_type,
                }
            )
        frame_shift = getattr(self, "frame_shift", None)
        if frame_shift is not None:
            frame_shift = round(frame_shift / 1000, 4)
        for u in file.utterances:
            duration = u.end - u.begin
            num_frames = None
            if frame_shift is not None:
                num_frames = int(duration / frame_shift)
            data.utterance_objects.append(
                {
                    "begin": u.begin,
                    "end": u.end,
                    "duration": duration,
                    "channel": u.channel,
                    "oovs": u.oovs,
                    "normalized_text": u.normalized_text,
                    "normalized_character_text": u.normalized_character_text,
                    "text": u.text,
                    "normalized_text_int": u.normalized_text_int,
                    "normalized_character_text_int": u.normalized_character_text_int,
                    "num_frames": num_frames,
                    "in_subset": False,
                    "ignored": False,
                    "file_id": self._current_file_index,
                    "job_id": 1,
                    "speaker_id": self._speaker_ids[u.speaker_name],
                }
            )
            c = Counter()
            if isinstance(self, MultispeakerDictionaryMixin):
                dict_id = self.get_dict_id_for_speaker(u.speaker_name)
                c.update(u.normalized_text_int.split())
                for w, count in c.items():
                    if (dict_id, int(w)) not in data.word_mapping:
                        data.word_mapping[(dict_id, int(w))] = {
                            "mapping_id": int(w),
                            "count": 0,
                            "dictionary_id": dict_id,
                        }
                    data.word_mapping[(dict_id, int(w))]["count"] += count
            else:
                c.update(u.text.split())
                for w, count in c.items():
                    if w not in data.word_mapping:
                        data.word_mapping[w] = {"word": w, "count": 0}
                    data.word_mapping[w]["count"] += count
        self._current_file_index += 1
        return data

    @property
    def data_source_identifier(self) -> str:
        """Corpus name"""
        return os.path.basename(self.corpus_directory)

    def create_subset(self, subset: int) -> None:
        """
        Create a subset of utterances to use for training

        Parameters
        ----------
        subset: int
            Number of utterances to include in subset
        """
        self.log_info(f"Creating subset directory with {subset} utterances...")
        subset_directory = os.path.join(self.corpus_output_directory, f"subset_{subset}")
        num_dictionaries = getattr(self, "num_dictionaries", 1)
        with self.session() as session:
            begin = time.time()
            session.query(Utterance).update({Utterance.in_subset: False})
            if num_dictionaries > 1:
                subsets_per_dictionary = {}
                utts_per_dictionary = {}
                subsetted = 0
                for dict_id in getattr(self, "dictionary_lookup", {}).values():
                    num_utts = (
                        session.query(Utterance)
                        .join(Utterance.speaker)
                        .filter(Speaker.dictionary_id == dict_id)
                        .count()
                    )
                    utts_per_dictionary[dict_id] = num_utts
                    if num_utts < int(subset / num_dictionaries):
                        subsets_per_dictionary[dict_id] = num_utts
                        subsetted += 1
                remaining_subset = subset - sum(subsets_per_dictionary.values())
                remaining_dicts = num_dictionaries - subsetted
                remaining_subset_per_dictionary = int(remaining_subset / remaining_dicts)
                for dict_id in getattr(self, "dictionary_lookup", {}).values():
                    num_utts = utts_per_dictionary[dict_id]
                    if dict_id in subsets_per_dictionary:
                        subset_per_dictionary = subsets_per_dictionary[dict_id]
                    else:
                        subset_per_dictionary = remaining_subset_per_dictionary
                    self.log_debug(f"For {dict_id}, total number of utterances is {num_utts}")
                    larger_subset_num = int(subset_per_dictionary * 10)
                    if num_utts > larger_subset_num:

                        larger_subset_query = (
                            session.query(Utterance.id)
                            .join(Utterance.speaker)
                            .filter(Speaker.dictionary_id == dict_id)
                            .filter(Utterance.text.like("% %"))
                            .filter(Utterance.ignored == False)  # noqa
                            .order_by(Utterance.duration)
                            .limit(larger_subset_num)
                        )

                        sq = larger_subset_query.subquery()
                        subset_utts = (
                            sqlalchemy.select(sq.c.id)
                            .order_by(sqlalchemy.func.random())
                            .limit(subset_per_dictionary)
                            .scalar_subquery()
                        )
                        query = (
                            sqlalchemy.update(Utterance)
                            .execution_options(synchronize_session="fetch")
                            .values(in_subset=True)
                            .where(Utterance.id.in_(subset_utts))
                        )
                        session.execute(query)
                        self.log_debug(f"For {dict_id}, subset is {subset_per_dictionary}")
                    elif num_utts > subset_per_dictionary:

                        larger_subset_query = (
                            session.query(Utterance.id)
                            .join(Utterance.speaker)
                            .filter(Speaker.dictionary_id == dict_id)
                            .filter(Utterance.ignored == False)  # noqa
                        )
                        sq = larger_subset_query.subquery()
                        subset_utts = (
                            sqlalchemy.select(sq.c.id)
                            .order_by(sqlalchemy.func.random())
                            .limit(subset_per_dictionary)
                            .scalar_subquery()
                        )
                        query = (
                            sqlalchemy.update(Utterance)
                            .execution_options(synchronize_session="fetch")
                            .values(in_subset=True)
                            .where(Utterance.id.in_(subset_utts))
                        )
                        session.execute(query)

                        self.log_debug(f"For {dict_id}, subset is {subset_per_dictionary}")
                    else:
                        larger_subset_query = (
                            session.query(Utterance.id)
                            .join(Utterance.speaker)
                            .filter(Speaker.dictionary_id == dict_id)
                            .filter(Utterance.ignored == False)  # noqa
                        )
                        sq = larger_subset_query.subquery()
                        subset_utts = sqlalchemy.select(sq.c.id).scalar_subquery()
                        query = (
                            sqlalchemy.update(Utterance)
                            .execution_options(synchronize_session="fetch")
                            .values(in_subset=True)
                            .where(Utterance.id.in_(subset_utts))
                        )
                        session.execute(query)

            else:
                larger_subset_num = subset * 10
                if subset < self.num_utterances:
                    # Get all shorter utterances that are not one word long
                    larger_subset_query = (
                        session.query(Utterance.id)
                        .filter(Utterance.text.like("% %"))
                        .filter(Utterance.ignored == False)  # noqa
                        .order_by(Utterance.duration)
                        .limit(larger_subset_num)
                    )
                    sq = larger_subset_query.subquery()
                    subset_utts = (
                        sqlalchemy.select(sq.c.id)
                        .order_by(sqlalchemy.func.random())
                        .limit(subset)
                        .scalar_subquery()
                    )
                    query = (
                        sqlalchemy.update(Utterance)
                        .execution_options(synchronize_session="fetch")
                        .values(in_subset=True)
                        .where(Utterance.id.in_(subset_utts))
                    )
                    session.execute(query)
                else:
                    session.query(Utterance).update({Utterance.in_subset: True})

            session.commit()

            self.log_debug(f"Setting subset flags took {time.time()-begin} seconds")
            log_dir = os.path.join(subset_directory, "log")
            os.makedirs(log_dir, exist_ok=True)
            for j in self.jobs:
                j.output_to_directory(subset_directory, session, subset=True)

    @property
    def num_files(self) -> int:
        """Number of files in the corpus"""
        if self._num_files is None:
            with self.session() as session:
                self._num_files = session.query(File).count()
        return self._num_files

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the corpus"""
        if self._num_utterances is None:
            with self.session() as session:
                self._num_utterances = session.query(Utterance).count()
        return self._num_utterances

    @property
    def num_speakers(self) -> int:
        """Number of speakers in the corpus"""
        if self._num_speakers is None:
            with self.session() as session:
                self._num_speakers = session.query(sqlalchemy.func.count(Speaker.id)).scalar()
        return self._num_speakers

    def subset_directory(self, subset: typing.Optional[int]) -> str:
        """
        Construct a subset directory for the corpus

        Parameters
        ----------
        subset: int, optional
            Number of utterances to include, if larger than the total number of utterance or not specified, the
            split_directory is returned

        Returns
        -------
        str
            Path to subset directory
        """
        if subset is None or subset >= self.num_utterances or subset <= 0:
            return self.split_directory
        directory = os.path.join(self.corpus_output_directory, f"subset_{subset}")
        if not os.path.exists(directory):
            self.create_subset(subset)
        for j in self.jobs:
            j.has_data = False
        with self.session() as session:
            for j in self.jobs:
                q = (
                    session.query(Utterance)
                    .join(Utterance.speaker)
                    .filter(Utterance.in_subset == True)  # noqa
                    .filter(Utterance.job_id == j.name)
                )
                if session.query(q.exists()).scalar():
                    j.has_data = True
        return directory

    def get_latest_workflow_run(self, workflow: WorkflowType, session: Session) -> CorpusWorkflow:
        """
        Get the latest version of a workflow type

        Parameters
        ----------
        workflow: :class:`~montreal_forced_aligner.data.WorkflowType`
            Workflow type
        session: :class:`sqlalchemy.orm.Session`
            Database session

        Returns
        -------
        :class:`~montreal_forced_aligner.db.CorpusWorkflow` or None
            Latest run of workflow type
        """
        workflow = (
            session.query(CorpusWorkflow)
            .filter(CorpusWorkflow.workflow == workflow)
            .order_by(CorpusWorkflow.time_stamp.desc())
            .first()
        )
        return workflow

    def _load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.inspect_database()

        self.log_info("Setting up corpus information...")
        if not self.imported:
            self.log_debug("Could not load from temp")
            self.log_info("Loading corpus from source files...")
            if GLOBAL_CONFIG.use_mp:
                self._load_corpus_from_source_mp()
            else:
                self._load_corpus_from_source()
        else:
            self.log_debug("Successfully loaded from temporary files")
        if not self.num_files:
            raise CorpusError(
                "There were no files found for this corpus. Please validate the corpus."
            )
        if not self.num_speakers:
            raise CorpusError(
                "There were no sound files found of the appropriate format. Please double check the corpus path "
                "and/or run the validation utility (mfa validate)."
            )
        average_utterances = self.num_utterances / self.num_speakers
        self.log_info(
            f"Found {self.num_speakers} speaker{'s' if self.num_speakers > 1 else ''} across {self.num_files} file{'s' if self.num_files > 1 else ''}, "
            f"average number of utterances per speaker: {average_utterances}"
        )

    @property
    def base_data_directory(self) -> str:
        """Corpus data directory"""
        return self.corpus_output_directory

    @property
    def data_directory(self) -> str:
        """Corpus data directory"""
        return self.split_directory

    @abstractmethod
    def _load_corpus_from_source_mp(self) -> None:
        """Abstract method for loading a corpus with multiprocessing"""
        ...

    @abstractmethod
    def _load_corpus_from_source(self) -> None:
        """Abstract method for loading a corpus without multiprocessing"""
        ...
