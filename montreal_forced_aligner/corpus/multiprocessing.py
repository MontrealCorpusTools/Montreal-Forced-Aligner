"""
Corpus loading worker
---------------------
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import typing
from queue import Empty, Queue

import sqlalchemy
from sqlalchemy.orm import Session, joinedload, subqueryload

from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.corpus.helper import find_exts
from montreal_forced_aligner.data import MfaArguments, WordType
from montreal_forced_aligner.db import (
    Dictionary,
    File,
    Grapheme,
    Job,
    SoundFile,
    Speaker,
    Utterance,
    Word,
)
from montreal_forced_aligner.exceptions import SoundFileError, TextGridParseError, TextParseError
from montreal_forced_aligner.helper import make_re_character_set_safe, mfa_open
from montreal_forced_aligner.utils import Counter, KaldiFunction, Stopped

if typing.TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from dataclassy import dataclass

__all__ = [
    "AcousticDirectoryParser",
    "CorpusProcessWorker",
    "ExportKaldiFilesFunction",
    "ExportKaldiFilesArguments",
    "NormalizeTextFunction",
    "NormalizeTextArguments",
    "construct_path",
]


def construct_path(job_name, directory: str, identifier: str, extension: str) -> str:
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
    return os.path.join(directory, f"{identifier}.{job_name}.{extension}")


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


class AcousticDirectoryParser(mp.Process):
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
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Check for whether to exit early
    finished_adding: :class:`~montreal_forced_aligner.utils.Stopped`
        Check to set when the parser is done adding files to the queue
    file_counts: :class:`~montreal_forced_aligner.utils.Counter`
        Counter for the number of total files that the parser has found
    """

    def __init__(
        self,
        corpus_directory: str,
        job_queue: Queue,
        audio_directory: str,
        stopped: Stopped,
        finished_adding: Stopped,
        file_counts: Counter,
    ):
        mp.Process.__init__(self)
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
                exts = find_exts(files)
                wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                other_audio_files = {
                    k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                }
                all_sound_files.update(other_audio_files)
                all_sound_files.update(wav_files)
        for root, _, files in os.walk(self.corpus_directory, followlinks=True):
            exts = find_exts(files)
            relative_path = root.replace(str(self.corpus_directory), "").lstrip("/").lstrip("\\")

            if self.stopped.stop_check():
                break
            if not use_audio_directory:
                all_sound_files = {}
                exts.wav_files = {k: os.path.join(root, v) for k, v in exts.wav_files.items()}
                exts.other_audio_files = {
                    k: os.path.join(root, v) for k, v in exts.other_audio_files.items()
                }
                all_sound_files.update(exts.other_audio_files)
                all_sound_files.update(exts.wav_files)
            for file_name in exts.identifiers:
                if self.stopped.stop_check():
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

        self.finished_adding.stop()


class CorpusProcessWorker(mp.Process):
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
    stopped: :class:`~montreal_forced_aligner.utils.Stopped`
        Stop check for whether corpus loading should exit
    finished_adding: :class:`~montreal_forced_aligner.utils.Stopped`
        Signal that the main thread has stopped adding new files to be processed
    """

    def __init__(
        self,
        name: int,
        job_q: mp.Queue,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
        speaker_characters: typing.Union[int, str],
        sample_rate: typing.Optional[int],
    ):
        mp.Process.__init__(self)
        self.name = str(name)
        self.job_q = job_q
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding
        self.finished_processing = Stopped()
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
                if self.finished_adding.stop_check():
                    break
                continue
            if self.stopped.stop_check():
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
                self.stopped.stop()
                self.return_q.put(("error", e))
        self.finished_processing.stop()
        return


@dataclass
class NormalizeTextArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.multiprocessing.NormalizeTextFunction`

    Parameters
    ----------
    model_path: str
        Path to model file
    phone_pdf_counts_path: str
        Path to output PDF counts
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    """

    word_break_markers: typing.List[str]
    punctuation: typing.List[str]
    clitic_markers: typing.List[str]
    compound_markers: typing.List[str]
    brackets: typing.List[typing.Tuple[str, str]]
    laughter_word: str
    oov_word: str
    bracketed_word: str
    ignore_case: bool
    use_g2p: bool


@dataclass
class ExportKaldiFilesArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.multiprocessing.NormalizeTextFunction`

    Parameters
    ----------

    """

    split_directory: str
    for_features: bool


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
        self.word_break_markers = args.word_break_markers
        self.brackets = args.brackets
        self.punctuation = args.punctuation
        self.compound_markers = args.compound_markers
        self.clitic_markers = args.clitic_markers
        self.ignore_case = args.ignore_case
        self.use_g2p = args.use_g2p
        self.laughter_word = args.laughter_word
        self.oov_word = args.oov_word
        self.bracketed_word = args.bracketed_word
        self.clitic_marker = None
        self.clitic_cleanup_regex = None
        self.compound_regex = None
        self.bracket_regex = None
        self.bracket_sanitize_regex = None
        self.laughter_regex = None
        self.word_break_regex = None
        self.clitic_quote_regex = None
        self.punctuation_regex = None
        self.non_speech_regexes = {}

    def compile_regexes(self) -> None:
        """Compile regular expressions necessary for corpus parsing"""
        if len(self.clitic_markers) >= 1:
            other_clitic_markers = self.clitic_markers[1:]
            if other_clitic_markers:
                extra = ""
                if "-" in other_clitic_markers:
                    extra = "-"
                    other_clitic_markers = [x for x in other_clitic_markers if x != "-"]
                self.clitic_cleanup_regex = re.compile(
                    rf'[{extra}{"".join(other_clitic_markers)}]'
                )
            self.clitic_marker = self.clitic_markers[0]
        if self.compound_markers:
            extra = ""
            compound_markers = self.compound_markers
            if "-" in self.compound_markers:
                extra = "-"
                compound_markers = [x for x in compound_markers if x != "-"]
            self.compound_regex = re.compile(rf"(?<=\w)[{extra}{''.join(compound_markers)}](?=\w)")
        if self.brackets:
            left_brackets = [x[0] for x in self.brackets]
            right_brackets = [x[1] for x in self.brackets]
            self.bracket_regex = re.compile(
                rf"[{re.escape(''.join(left_brackets))}].*?[{re.escape(''.join(right_brackets))}]+"
            )
            self.laughter_regex = re.compile(
                rf"[{re.escape(''.join(left_brackets))}](laugh(ing|ter)?|lachen|lg)[{re.escape(''.join(right_brackets))}]+",
                flags=re.IGNORECASE,
            )
        all_punctuation = set()
        non_word_character_set = set(self.punctuation)
        non_word_character_set -= {b for x in self.brackets for b in x}

        if self.clitic_markers:
            all_punctuation.update(self.clitic_markers)
        if self.compound_markers:
            all_punctuation.update(self.compound_markers)
        self.bracket_sanitize_regex = None
        if self.brackets:
            word_break_set = (
                non_word_character_set | set(self.clitic_markers) | set(self.compound_markers)
            )
            if self.word_break_markers:
                word_break_set |= set(self.word_break_markers)
            word_break_set = make_re_character_set_safe(word_break_set, [r"\s"])
            self.bracket_sanitize_regex = re.compile(f"(?<!^){word_break_set}(?!$)")

        word_break_character_set = make_re_character_set_safe(non_word_character_set, [r"\s"])
        self.word_break_regex = re.compile(rf"{word_break_character_set}+")
        punctuation_set = make_re_character_set_safe(all_punctuation)
        if all_punctuation:
            self.punctuation_regex = re.compile(rf"^{punctuation_set}+$")
        if len(self.clitic_markers) >= 1:
            non_clitic_punctuation = all_punctuation - set(self.clitic_markers)
            non_clitic_punctuation_set = make_re_character_set_safe(non_clitic_punctuation)
            non_punctuation_set = "[^" + punctuation_set[1:]
            self.clitic_quote_regex = re.compile(
                rf"((?<=\W)|(?<=^)){non_clitic_punctuation_set}*{self.clitic_marker}{non_clitic_punctuation_set}*(?P<word>{non_punctuation_set}+){non_clitic_punctuation_set}*{self.clitic_marker}{non_clitic_punctuation_set}*((?=\W)|(?=$))"
            )

        if self.laughter_regex is not None:
            self.non_speech_regexes[self.laughter_word] = self.laughter_regex
        if self.bracket_regex is not None:
            self.non_speech_regexes[self.bracketed_word] = self.bracket_regex

    def _dictionary_sanitize(self, session):
        from montreal_forced_aligner.dictionary.mixins import SanitizeFunction, SplitWordsFunction

        dictionaries: typing.List[Dictionary] = session.query(Dictionary)
        grapheme_mapping = {}
        grapheme_query = session.query(Grapheme.grapheme, Grapheme.mapping_id)
        for w, m_id in grapheme_query:
            grapheme_mapping[w] = m_id
        for d in dictionaries:
            words_mapping = {}
            words_query = session.query(Word.word, Word.mapping_id).filter(
                Word.dictionary_id == d.id
            )
            for w, m_id in words_query:
                words_mapping[w] = m_id
            sanitize_function = SanitizeFunction(
                self.clitic_marker,
                self.clitic_cleanup_regex,
                self.clitic_quote_regex,
                self.punctuation_regex,
                self.word_break_regex,
                self.bracket_regex,
                self.bracket_sanitize_regex,
                self.ignore_case,
            )
            clitic_set = set(
                x[0]
                for x in session.query(Word.word)
                .filter(Word.word_type == WordType.clitic)
                .filter(Word.dictionary_id == d.id)
            )
            initial_clitic_regex = None
            final_clitic_regex = None
            if self.clitic_marker is not None:
                initial_clitics = sorted(x for x in clitic_set if x.endswith(self.clitic_marker))
                final_clitics = sorted(x for x in clitic_set if x.startswith(self.clitic_marker))
                if initial_clitics:
                    initial_clitic_regex = re.compile(rf"^({'|'.join(initial_clitics)})(?=\w)")
                if final_clitics:
                    final_clitic_regex = re.compile(rf"(?<=\w)({'|'.join(final_clitics)})$")

            non_speech_regexes = {}
            if self.laughter_regex is not None:
                non_speech_regexes[d.laughter_word] = self.laughter_regex
            if self.bracket_regex is not None:
                non_speech_regexes[d.bracketed_word] = self.bracket_regex
            split_function = SplitWordsFunction(
                self.clitic_marker,
                initial_clitic_regex,
                final_clitic_regex,
                self.compound_regex,
                non_speech_regexes,
                d.oov_word,
                words_mapping,
                grapheme_mapping,
            )
            utterances = (
                session.query(Utterance.id, Utterance.text)
                .join(Utterance.speaker)
                .filter(Utterance.text != "")
                .filter(Utterance.job_id == self.job_name)
                .filter(Speaker.dictionary_id == d.id)
            )
            for u_id, u_text in utterances:
                words = sanitize_function(u_text)
                normalized_text = []
                normalized_character_text = []
                oovs = set()
                text = ""
                for w in words:
                    for new_w in split_function(w):
                        if new_w not in words_mapping:
                            oovs.add(new_w)
                        normalized_text.append(split_function.to_str(new_w))
                        if normalized_character_text:
                            if not self.clitic_marker or (
                                not normalized_text[-1].endswith(self.clitic_marker)
                                and not new_w.startswith(self.clitic_marker)
                            ):
                                normalized_character_text.append("<space>")
                        for c in split_function.parse_graphemes(new_w):
                            normalized_character_text.append(c)
                    if text:
                        text += " "
                    text += w
                yield {
                    "id": u_id,
                    "oovs": " ".join(sorted(oovs)),
                    "normalized_text": " ".join(normalized_text),
                    "normalized_character_text": " ".join(normalized_character_text),
                }, d.id

    def _no_dictionary_sanitize(self, session):
        from montreal_forced_aligner.dictionary.mixins import SanitizeFunction

        sanitize_function = SanitizeFunction(
            self.clitic_marker,
            self.clitic_cleanup_regex,
            self.clitic_quote_regex,
            self.punctuation_regex,
            self.word_break_regex,
            self.bracket_regex,
            self.bracket_sanitize_regex,
            self.ignore_case,
        )
        utterances = (
            session.query(Utterance.id, Utterance.text)
            .join(Utterance.speaker)
            .filter(Utterance.text != "")
            .filter(Utterance.job_id == self.job_name)
        )
        for u_id, u_text in utterances:
            text = " ".join(sanitize_function(u_text))
            oovs = set()
            yield {
                "id": u_id,
                "oovs": " ".join(sorted(oovs)),
                "normalized_text": text,
            }, None

    def _run(self) -> typing.Generator[typing.Tuple[int, float]]:
        """Run the function"""
        self.compile_regexes()
        with Session(self.db_engine()) as session:
            dict_count = session.query(Dictionary).join(Dictionary.words).limit(1).count()
            if self.use_g2p or dict_count > 0:
                yield from self._dictionary_sanitize(session)
            else:
                yield from self._no_dictionary_sanitize(session)


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
        self.for_features = args.for_features

    def output_for_features(self, session: Session) -> None:
        """
        Output the necessary files for Kaldi to generate features

        Parameters
        ----------
        split_directory: str
            Split directory for the corpus
        """
        job = (
            session.query(Job)
            .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
            .filter(Job.id == self.job_name)
            .first()
        )
        wav_scp_path = job.wav_scp_path
        segments_scp_path = job.segments_scp_path
        if os.path.exists(segments_scp_path):
            return
        with mfa_open(wav_scp_path, "w") as wav_file:
            files = (
                session.query(File.id, SoundFile.sox_string, SoundFile.sound_file_path)
                .join(File.sound_file)
                .join(File.utterances)
                .filter(Utterance.job_id == job.id)
                .order_by(File.id.cast(sqlalchemy.String))
                .distinct(File.id.cast(sqlalchemy.String))
            )
            for f_id, sox_string, sound_file_path in files:
                if not sox_string:
                    sox_string = sound_file_path
                wav_file.write(f"{f_id} {sox_string}\n")
                yield 1

        with mfa_open(segments_scp_path, "w") as segments_file:
            utterances = (
                session.query(
                    Utterance.kaldi_id,
                    Utterance.file_id,
                    Utterance.begin,
                    Utterance.end,
                    Utterance.channel,
                )
                .filter(Utterance.job_id == job.id)
                .order_by(Utterance.kaldi_id)
            )
            for u_id, f_id, begin, end, channel in utterances:
                segments_file.write(f"{u_id} {f_id} {begin} {end} {channel}\n")
                yield 1

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
                    Utterance.normalized_text,
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
            text_ints_path = job.construct_path(self.split_directory, "text", "int.scp")
            vad_path = job.construct_path(self.split_directory, "vad", "scp")
            ivectors_path = job.construct_path(self.split_directory, "ivectors", "scp")

            spk2utt = {}
            vad = {}
            ivectors = {}
            feats = {}
            cmvns = {}
            utt2spk = {}
            text_ints = {}

            for (
                u_id,
                s_id,
                features,
                vad_ark,
                ivector_ark,
                normalized_text,
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
                text_ints[utterance] = normalized_text
                yield 1

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

            with mfa_open(text_ints_path, "w") as f:
                for utt, text in sorted(text_ints.items()):
                    f.write(f"{utt} {text}\n")
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
                    Utterance.normalized_text,
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
            utt2spk_paths = job.per_dictionary_utt2spk_scp_paths
            feats_paths = job.per_dictionary_feats_scp_paths
            cmvns_paths = job.per_dictionary_cmvn_scp_paths
            spk2utt_paths = job.per_dictionary_spk2utt_scp_paths
            text_ints_paths = job.per_dictionary_text_int_scp_paths
            for d in job.dictionaries:

                words_mapping = {}
                words_query = session.query(Word.word, Word.mapping_id).filter(
                    Word.dictionary_id == d.id
                )
                for w, m_id in words_query:
                    words_mapping[w] = m_id
                spk2utt = {}
                feats = {}
                cmvns = {}
                utt2spk = {}
                text_ints = {}
                utterances = base_utterance_query.filter(Speaker.dictionary_id == d.id)
                for (
                    u_id,
                    s_id,
                    features,
                    normalized_text,
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
                    words = normalized_text.split()
                    text_ints[utterance] = " ".join([str(words_mapping[x]) for x in words])
                    yield 1

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

                with mfa_open(text_ints_paths[d.id], "w") as f:
                    for utt, text in sorted(text_ints.items()):
                        f.write(f"{utt} {text}\n")

    def _run(self) -> typing.Generator[typing.Tuple[int, float]]:
        """Run the function"""
        with Session(self.db_engine()) as session:
            if self.for_features:
                yield from self.output_for_features(session)
            else:
                yield from self.output_to_directory(session)
