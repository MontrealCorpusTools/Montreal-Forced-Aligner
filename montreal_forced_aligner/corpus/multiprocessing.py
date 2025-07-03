"""
Corpus loading worker
---------------------
"""
from __future__ import annotations

import os
import threading
import typing
from pathlib import Path
from queue import Empty, Queue

import sqlalchemy
from _kalpy.util import Int32VectorVectorWriter
from kalpy.utils import generate_write_specifier
from sqlalchemy.orm import joinedload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.data import Language, MfaArguments, PhoneType, WorkflowType
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    Dictionary,
    Job,
    Phone,
    PhoneMapping,
    ReferencePhoneInterval,
    Speaker,
    Utterance,
)
from montreal_forced_aligner.exceptions import SoundFileError, TextGridParseError, TextParseError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import Counter

if typing.TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.models import G2PModel
    from montreal_forced_aligner.tokenization.simple import SimpleTokenizer

    try:
        from spacy.language import Language as SpacyLanguage
    except ImportError:
        SpacyLanguage = None
else:
    from dataclassy import dataclass

__all__ = [
    "AcousticDirectoryParser",
    "CorpusProcessWorker",
    "ExportKaldiFilesFunction",
    "ExportKaldiFilesArguments",
    "NormalizeTextFunction",
    "NormalizeTextArguments",
    "dictionary_ids_for_job",
]


def dictionary_ids_for_job(session, job_id):
    dictionary_ids = [
        x[0]
        for x in session.query(Dictionary.id)
        .join(Utterance.speaker)
        .join(Speaker.dictionary)
        .filter(Utterance.in_subset == True)  # noqa
        .filter(Utterance.job_id == job_id)
        .distinct()
    ]
    return dictionary_ids


class AcousticDirectoryParser(threading.Thread):
    """
    Worker for processing directories for acoustic sound files

    Parameters
    ----------
    corpus_directory: str
        Directory to parse
    job_queue: Queue
        Queue to add file names to
    audio_directory: str
        Directory with additional audio files
    stopped: :class:`~threading.Event`
        Check for whether to exit early
    finished_adding: :class:`~threading.Event`
        Check to set when the parser is done adding files to the queue
    file_counts: :class:`~montreal_forced_aligner.utils.Counter`
        Counter for the number of total files that the parser has found
    """

    def __init__(
        self,
        corpus_directory: str,
        job_queue: Queue,
        audio_directory: str,
        stopped: threading.Event,
        finished_adding: threading.Event,
        file_counts: Counter,
    ):
        super().__init__()
        self.corpus_directory = corpus_directory
        self.job_queue = job_queue
        self.audio_directory = audio_directory
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.file_counts = file_counts

    def run(self) -> None:
        """
        Run the corpus loading job
        """

        use_audio_directory = False
        all_sound_files = {}
        if self.audio_directory and os.path.exists(self.audio_directory):
            use_audio_directory = True
            for root, _, files in os.walk(self.audio_directory, followlinks=True):
                if root.startswith("."):  # Ignore hidden directories
                    continue
                exts = find_exts(files)
                wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                other_audio_files = {
                    k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                }
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)
        for root, _, files in os.walk(self.corpus_directory, followlinks=True):
            if self.stopped.is_set():
                break
            if root.startswith("."):  # Ignore hidden directories
                continue
            exts = find_exts(files)
            relative_path = root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")
            if not use_audio_directory:
                all_sound_files = {}
                exts.wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                exts.other_audio_files = {
                    k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                }
                all_sound_files.update(exts.other_audio_files)
                all_sound_files.update(exts.wav_files)
            for file_name in exts.identifiers:
                if self.stopped.is_set():
                    break
                wav_path = None
                transcription_path = None
                if file_name in all_sound_files:
                    wav_path = all_sound_files[file_name]
                if file_name in exts.lab_files:
                    lab_name = exts.lab_files[file_name]
                    transcription_path = os.path.join(root, lab_name)

                elif file_name in exts.textgrid_files:
                    tg_name = exts.textgrid_files[file_name]
                    transcription_path = os.path.join(root, tg_name)
                if wav_path is None:
                    continue
                self.job_queue.put((file_name, wav_path, transcription_path, relative_path))
                self.file_counts.increment()

        self.finished_adding.set()


class CorpusProcessWorker(threading.Thread):
    """
    Multiprocessing corpus loading worker

    Attributes
    ----------
    job_q: :class:`~multiprocessing.Queue`
        Job queue for files to process
    return_dict: dict
        Dictionary to catch errors
    return_q: :class:`~multiprocessing.Queue`
        Return queue for processed Files
    stopped: :class:`~threading.Event`
        Stop check for whether corpus loading should exit
    finished_adding: :class:`~threading.Event`
        Signal that the main thread has stopped adding new files to be processed
    """

    def __init__(
        self,
        name: int,
        job_q: Queue,
        return_q: Queue,
        stopped: threading.Event,
        finished_adding: threading.Event,
        speaker_characters: typing.Union[int, str],
        sample_rate: typing.Optional[int],
    ):
        super().__init__()
        self.name = str(name)
        self.job_q = job_q
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = threading.Event()
        self.speaker_characters = speaker_characters
        self.sample_rate = sample_rate

    def run(self) -> None:
        """
        Run the corpus loading job
        """
        while True:
            try:
                file_name, wav_path, text_path, relative_path = self.job_q.get(timeout=1)
            except Empty:
                if self.finished_adding.is_set():
                    break
                continue
            if self.stopped.is_set():
                continue
            try:
                file = FileData.parse_file(
                    file_name,
                    wav_path,
                    text_path,
                    relative_path,
                    self.speaker_characters,
                    self.sample_rate,
                )
                self.return_q.put(file)
            except TextParseError as e:
                self.return_q.put(("decode_error_files", e))
            except TextGridParseError as e:
                self.return_q.put(("textgrid_read_errors", e))
            except SoundFileError as e:
                self.return_q.put(("sound_file_errors", e))
            except Exception as e:
                self.stopped.set()
                self.return_q.put(("error", e))
        self.finished_processing.set()
        return


@dataclass
class NormalizeTextArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.multiprocessing.NormalizeTextFunction`

    """

    tokenizers: typing.Union[typing.Dict[int, SimpleTokenizer], Language]
    g2p_model: typing.Optional[G2PModel]
    ignore_case: bool
    use_cutoff_model: bool


@dataclass
class ExportKaldiFilesArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.multiprocessing.NormalizeTextFunction`

    Parameters
    ----------

    """

    split_directory: Path
    time_step: float


class NormalizeTextFunction(KaldiFunction):
    """
    Multiprocessing function for normalizing text.
    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.multiprocessing.NormalizeTextArguments`
        Arguments for the function
    """

    def __init__(self, args: NormalizeTextArguments):
        super().__init__(args)
        self.tokenizers = args.tokenizers
        self.g2p_model = args.g2p_model
        self.ignore_case = args.ignore_case
        self.use_cutoff_model = args.use_cutoff_model

    def _run(self):
        """Run the function"""

        from montreal_forced_aligner.tokenization.simple import SimpleTokenizer

        with self.session() as session:
            dict_count = session.query(Dictionary).join(Dictionary.words).limit(1).count()
            if dict_count > 0 or isinstance(self.tokenizers, dict):
                dictionaries = session.query(Dictionary)
                for d in dictionaries:
                    if isinstance(self.tokenizers, dict):
                        tokenizer = self.tokenizers[d.id]
                    else:
                        tokenizer = self.tokenizers
                    simple_tokenization = isinstance(tokenizer, SimpleTokenizer)
                    if isinstance(tokenizer, Language):
                        from montreal_forced_aligner.tokenization.spacy import (
                            generate_language_tokenizer,
                        )

                        tokenizer = generate_language_tokenizer(
                            tokenizer, ignore_case=self.ignore_case
                        )

                    utterances = (
                        session.query(Utterance.id, Utterance.text)
                        .join(Utterance.speaker)
                        .filter(Utterance.text != "")
                        .filter(Utterance.job_id == self.job_name)
                        .filter(Speaker.dictionary_id == d.id)
                    )
                    for u_id, u_text in utterances:
                        if simple_tokenization:
                            normalized_text, normalized_character_text, oovs = tokenizer(u_text)
                            if self.use_cutoff_model:
                                new_text = []
                                text = normalized_text.split()
                                for i, w in enumerate(text):
                                    if w == d.cutoff_word and i != len(text) - 1:
                                        next_w = text[i + 1]
                                        if tokenizer.word_table.member(
                                            next_w
                                        ) and not tokenizer.bracket_regex.match(next_w):
                                            w = f"{d.cutoff_word[:-1]}-{next_w}{d.cutoff_word[-1]}"
                                    new_text.append(w)
                                normalized_text = " ".join(new_text)
                            self.callback(
                                (
                                    {
                                        "id": u_id,
                                        "oovs": " ".join(sorted(oovs)),
                                        "normalized_text": normalized_text,
                                        "normalized_character_text": normalized_character_text,
                                    },
                                    d.id,
                                )
                            )
                        else:
                            tokenized = tokenizer(u_text)
                            if isinstance(tokenized, tuple):
                                normalized_text, pronunciation_form = tokenized
                            else:
                                if not isinstance(tokenized, str):
                                    tokenized = " ".join([x.text for x in tokenized])
                                if self.ignore_case:
                                    tokenized = tokenized.lower()
                                normalized_text, pronunciation_form = tokenized, tokenized.lower()
                            oovs = set()
                            self.callback(
                                (
                                    {
                                        "id": u_id,
                                        "oovs": " ".join(sorted(oovs)),
                                        "normalized_text": normalized_text,
                                        "normalized_character_text": pronunciation_form,
                                    },
                                    d.id,
                                )
                            )
            else:
                tokenizer = self.tokenizers
                if isinstance(tokenizer, Language):
                    from montreal_forced_aligner.tokenization.spacy import (
                        generate_language_tokenizer,
                    )

                    tokenizer = generate_language_tokenizer(
                        tokenizer, ignore_case=self.ignore_case
                    )
                utterances = (
                    session.query(Utterance.id, Utterance.text)
                    .filter(Utterance.text != "")
                    .filter(Utterance.job_id == self.job_name)
                )
                for u_id, u_text in utterances:
                    if tokenizer is None:
                        normalized_text, pronunciation_form = u_text, u_text
                    else:
                        tokenized = tokenizer(u_text)
                        if isinstance(tokenized, tuple):
                            normalized_text, pronunciation_form = tokenized[:2]
                        else:
                            if not isinstance(tokenized, str):
                                tokenized = " ".join([x.text for x in tokenized])
                            if self.ignore_case:
                                tokenized = tokenized.lower()
                            normalized_text, pronunciation_form = tokenized, tokenized.lower()
                    self.callback(
                        (
                            {
                                "id": u_id,
                                "normalized_text": normalized_text,
                                "normalized_character_text": pronunciation_form,
                            },
                            None,
                        )
                    )


class ExportKaldiFilesFunction(KaldiFunction):
    """
    Multiprocessing function for normalizing text.
    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.multiprocessing.NormalizeTextArguments`
        Arguments for the function
    """

    def __init__(self, args: ExportKaldiFilesArguments):
        super().__init__(args)
        self.split_directory = args.split_directory
        self.time_step = args.time_step

    def output_to_directory(self, session) -> None:
        """
        Output job information to a directory

        Parameters
        ----------
        split_directory: str
            Directory to output to
        """
        job = (
            session.query(Job)
            .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
            .filter(Job.id == self.job_name)
            .first()
        )
        reference_workflow = (
            session.query(CorpusWorkflow)
            .filter(CorpusWorkflow.workflow_type == WorkflowType.reference)
            .first()
        )
        base_utterance_query = (
            session.query(sqlalchemy.func.count(Utterance.id))
            .filter(Utterance.job_id == job.id)
            .filter(Utterance.ignored == False)  # noqa
        )
        if job.corpus.current_subset:
            base_utterance_query = base_utterance_query.filter(Utterance.in_subset == True)  # noqa

        if not base_utterance_query.scalar():
            return
        if not job.has_dictionaries:
            utterances = (
                session.query(
                    Utterance.id,
                    Utterance.speaker_id,
                    Utterance.features,
                    Utterance.vad_ark,
                    Utterance.ivector_ark,
                    Speaker.cmvn,
                )
                .join(Utterance.speaker)
                .filter(Utterance.job_id == job.id)
                .filter(Utterance.ignored == False)  # noqa
                .order_by(Utterance.kaldi_id)
            )
            if job.corpus.current_subset:
                utterances = utterances.filter(Utterance.in_subset == True)  # noqa
            utt2spk_path = job.construct_path(self.split_directory, "utt2spk", "scp")
            feats_path = job.construct_path(self.split_directory, "feats", "scp")
            cmvns_path = job.construct_path(self.split_directory, "cmvn", "scp")
            spk2utt_path = job.construct_path(self.split_directory, "spk2utt", "scp")
            vad_path = job.construct_path(self.split_directory, "vad", "scp")
            ivectors_path = job.construct_path(self.split_directory, "ivectors", "scp")

            spk2utt = {}
            vad = {}
            ivectors = {}
            feats = {}
            cmvns = {}
            utt2spk = {}

            for (
                u_id,
                s_id,
                features,
                vad_ark,
                ivector_ark,
                cmvn,
            ) in utterances:
                utterance = str(u_id)
                speaker = str(s_id)
                utterance = f"{speaker}-{utterance}"
                if speaker not in spk2utt:
                    spk2utt[speaker] = []
                spk2utt[speaker].append(utterance)
                utt2spk[utterance] = speaker
                feats[utterance] = features
                if vad_ark:
                    vad[utterance] = vad_ark
                if ivector_ark:
                    ivectors[utterance] = ivector_ark
                cmvns[speaker] = cmvn
                self.callback(1)

            with mfa_open(spk2utt_path, "w") as f:
                for speaker, utts in sorted(spk2utt.items()):
                    utts = " ".join(sorted(utts))
                    f.write(f"{speaker} {utts}\n")

            with mfa_open(cmvns_path, "w") as f:
                for speaker, cmvn in sorted(cmvns.items()):
                    f.write(f"{speaker} {cmvn}\n")

            with mfa_open(utt2spk_path, "w") as f:
                for utt, spk in sorted(utt2spk.items()):
                    f.write(f"{utt} {spk}\n")

            with mfa_open(feats_path, "w") as f:
                for utt, feat in sorted(feats.items()):
                    f.write(f"{utt} {feat}\n")
            if vad:
                with mfa_open(vad_path, "w") as f:
                    for utt, ark in sorted(vad.items()):
                        f.write(f"{utt} {ark}\n")
            if ivectors:
                with mfa_open(ivectors_path, "w") as f:
                    for utt, ark in sorted(ivectors.items()):
                        f.write(f"{utt} {ark}\n")

        else:
            base_utterance_query = (
                session.query(
                    Utterance.id,
                    Utterance.speaker_id,
                    Utterance.features,
                    Speaker.cmvn,
                )
                .join(Utterance.speaker)
                .filter(Utterance.job_id == job.id)
                .filter(Utterance.ignored == False)  # noqa
                .order_by(Utterance.kaldi_id)
            )
            if job.corpus.current_subset:
                base_utterance_query = base_utterance_query.filter(
                    Utterance.in_subset == True  # noqa
                )
            base_utterance_intervals_query = None
            phone_mapping = {}
            phone_id_mapping = {}
            reference_mappings = {}
            if reference_workflow is not None:
                for p in (
                    session.query(Phone)
                    .filter(~Phone.phone_type.in_([PhoneType.disambiguation]))
                    .all()
                ):
                    phone_id_mapping[p.id] = p.mapping_id
                    phone_mapping[p.phone] = p.mapping_id
                for x in session.query(PhoneMapping).all():
                    if x.reference_phone_string not in reference_mappings:
                        reference_mappings[x.reference_phone_string] = []
                    reference_mappings[x.reference_phone_string].append(x.model_phone_string)
                base_utterance_intervals_query = (
                    session.query(Utterance)
                    .join(Utterance.speaker)
                    .filter(Utterance.job_id == job.id)
                    .filter(Utterance.ignored == False)  # noqa
                    .filter(Utterance.manual_alignments == True)  # noqa
                    .order_by(Utterance.kaldi_id)
                    .options(
                        sqlalchemy.orm.subqueryload(
                            Utterance.reference_phone_intervals
                        ).joinedload(ReferencePhoneInterval.phone)
                    )
                )
                if job.corpus.current_subset:
                    base_utterance_intervals_query = base_utterance_intervals_query.filter(
                        Utterance.in_subset == True  # noqa
                    )
            utt2spk_paths = job.per_dictionary_utt2spk_scp_paths
            feats_paths = job.per_dictionary_feats_scp_paths
            ref_phone_paths = job.per_dictionary_ref_phone_scp_paths
            cmvns_paths = job.per_dictionary_cmvn_scp_paths
            spk2utt_paths = job.per_dictionary_spk2utt_scp_paths
            for d in job.dictionaries:
                spk2utt = {}
                feats = {}
                cmvns = {}
                utt2spk = {}
                ref_phones = {}
                if reference_workflow is not None:
                    time_step_tolerance = self.time_step / 2
                    utterances = base_utterance_intervals_query.filter(
                        Speaker.dictionary_id == d.id
                    )
                    for u in utterances:
                        utterance = f"{u.speaker_id}-{u.id}"
                        phones = []
                        reference_intervals = u.reference_phone_intervals
                        reference_index = 0
                        for i in range(u.num_frames):
                            time_point = round(i * self.time_step, 3)
                            interval_end = round(
                                reference_intervals[reference_index].end - u.begin, 3
                            )
                            if (
                                reference_index < len(reference_intervals) - 1
                                and time_point >= interval_end - time_step_tolerance
                            ):
                                reference_index += 1
                            frame_phones = []
                            if reference_mappings:
                                reference_phone = reference_intervals[reference_index].phone.phone
                                if reference_phone in reference_mappings:
                                    for mapped_phone in reference_mappings[reference_phone]:
                                        for split_mapped_phone in mapped_phone.split():
                                            if split_mapped_phone not in phone_mapping:
                                                continue
                                            frame_phones.append(phone_mapping[split_mapped_phone])
                                elif reference_phone in phone_mapping:
                                    frame_phones.append(phone_mapping[reference_phone])
                                else:
                                    frame_phones.append(-1)
                                if phone_mapping["spn"] not in frame_phones:
                                    frame_phones.append(phone_mapping["spn"])
                            else:
                                frame_phones.append(
                                    phone_id_mapping[reference_intervals[reference_index].phone_id]
                                )
                            phones.append(frame_phones)
                        ref_phones[utterance] = phones

                utterances = base_utterance_query.filter(Speaker.dictionary_id == d.id)
                for (
                    u_id,
                    s_id,
                    features,
                    cmvn,
                ) in utterances:
                    utterance = str(u_id)
                    speaker = str(s_id)
                    utterance = f"{speaker}-{utterance}"
                    if speaker not in spk2utt:
                        spk2utt[speaker] = []
                    spk2utt[speaker].append(utterance)
                    utt2spk[utterance] = speaker
                    feats[utterance] = features
                    cmvns[speaker] = cmvn
                    self.callback(1)
                if not spk2utt:
                    continue
                with mfa_open(spk2utt_paths[d.id], "w") as f:
                    for speaker, utts in sorted(spk2utt.items()):
                        utts = " ".join(sorted(utts))
                        f.write(f"{speaker} {utts}\n")

                with mfa_open(cmvns_paths[d.id], "w") as f:
                    for speaker, cmvn in sorted(cmvns.items()):
                        f.write(f"{speaker} {cmvn}\n")

                with mfa_open(utt2spk_paths[d.id], "w") as f:
                    for utt, spk in sorted(utt2spk.items()):
                        f.write(f"{utt} {spk}\n")

                with mfa_open(feats_paths[d.id], "w") as f:
                    for utt, feat in sorted(feats.items()):
                        f.write(f"{utt} {feat}\n")
                if ref_phones:
                    write_specifier = generate_write_specifier(ref_phone_paths[d.id], text=True)

                    writer = Int32VectorVectorWriter(write_specifier)
                    for utt, phones in sorted(ref_phones.items()):
                        if not phones:
                            continue
                        writer.Write(str(utt), phones)
                    writer.Close()

    def _run(self):
        """Run the function"""
        with self.session() as session:
            self.output_to_directory(session)
