"""
Segmenter
=========

"""
from __future__ import annotations

import collections
import logging
import os
import sys
import typing
from pathlib import Path
from typing import Dict, List, Optional

import sqlalchemy
from kalpy.fstext.lexicon import Pronunciation as KalpyPronunciation
from sqlalchemy.orm import joinedload, selectinload
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import FileExporterMixin, MetaDict, TopLevelMfaWorker
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.features import VadConfigMixin
from montreal_forced_aligner.data import Language, TextFileType, WorkflowType
from montreal_forced_aligner.db import (
    CorpusWorkflow,
    File,
    Pronunciation,
    Utterance,
    Word,
    full_load_utterance,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer
from montreal_forced_aligner.transcription.transcriber import TranscriberMixin
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function
from montreal_forced_aligner.vad.multiprocessing import (
    FOUND_SPEECHBRAIN,
    VAD,
    SegmentTranscriptArguments,
    SegmentTranscriptFunction,
    SegmentVadArguments,
    SegmentVadFunction,
    segment_utterance_transcript,
)

SegmentationType = List[Dict[str, float]]

__all__ = ["VadSegmenter", "SpeechbrainSegmenterMixin", "TranscriptionSegmenter"]

logger = logging.getLogger("mfa")


class SpeechbrainSegmenterMixin:
    def __init__(
        self,
        segment_padding: float = 0.01,
        large_chunk_size: float = 30,
        small_chunk_size: float = 0.05,
        overlap_small_chunk: bool = False,
        apply_energy_vad: bool = False,
        double_check: bool = True,
        close_th: float = 0.333,
        len_th: float = 0.333,
        activation_th: float = 0.5,
        deactivation_th: float = 0.25,
        en_activation_th: float = 0.5,
        en_deactivation_th: float = 0.4,
        speech_th: float = 0.5,
        cuda: bool = False,
        speechbrain: bool = False,
        **kwargs,
    ):
        if speechbrain and not FOUND_SPEECHBRAIN:
            logger.error(
                "Could not import speechbrain, please ensure it is installed via `pip install speechbrain`"
            )
            sys.exit(1)
        super().__init__(**kwargs)
        self.large_chunk_size = large_chunk_size
        self.small_chunk_size = small_chunk_size
        self.overlap_small_chunk = overlap_small_chunk
        self.apply_energy_vad = apply_energy_vad
        self.double_check = double_check
        self.close_th = close_th
        self.len_th = len_th
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th
        self.en_activation_th = en_activation_th
        self.en_deactivation_th = en_deactivation_th
        self.speech_th = speech_th
        self.cuda = cuda
        self.speechbrain = speechbrain
        self.segment_padding = segment_padding
        self.vad_model = None
        if self.speechbrain:
            model_dir = os.path.join(config.TEMPORARY_DIRECTORY, "models", "VAD")
            os.makedirs(model_dir, exist_ok=True)
            run_opts = None
            if self.cuda:
                run_opts = {"device": "cuda"}
            self.vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty", savedir=model_dir, run_opts=run_opts
            )

    @property
    def segmentation_options(self) -> MetaDict:
        """Options for segmentation"""
        return {
            "large_chunk_size": self.large_chunk_size,
            "frame_shift": getattr(self, "export_frame_shift", 0.01),
            "small_chunk_size": self.small_chunk_size,
            "overlap_small_chunk": self.overlap_small_chunk,
            "apply_energy_VAD": self.apply_energy_vad,
            "double_check": self.double_check,
            "activation_th": self.activation_th,
            "deactivation_th": self.deactivation_th,
            "en_activation_th": self.en_activation_th,
            "en_deactivation_th": self.en_deactivation_th,
            "speech_th": self.speech_th,
            "close_th": self.close_th,
            "len_th": self.len_th,
        }


class VadSegmenter(
    VadConfigMixin,
    AcousticCorpusMixin,
    FileExporterMixin,
    SpeechbrainSegmenterMixin,
    TopLevelMfaWorker,
):
    """
    Class for performing speaker classification, parameters are passed to
    `speechbrain.pretrained.interfaces.VAD.get_speech_segments
    <https://speechbrain.readthedocs.io/en/latest/API/speechbrain.pretrained.interfaces.html#speechbrain.pretrained.interfaces.VAD.get_speech_segments>`_

    Parameters
    ----------
    segment_padding: float
        Size of padding on both ends of a segment
    large_chunk_size: float
        Size (in seconds) of the large chunks that are read sequentially
        from the input audio file.
    small_chunk_size: float
        Size (in seconds) of the small chunks extracted from the large ones.
        The audio signal is processed in parallel within the small chunks.
        Note that large_chunk_size/small_chunk_size must be an integer.
    overlap_small_chunk: bool
        If True, it creates overlapped small chunks (with 50% overal).
        The probabilities of the overlapped chunks are combined using
        hamming windows.
    apply_energy_VAD: bool
        If True, a energy-based VAD is used on the detected speech segments.
        The neural network VAD often creates longer segments and tends to
        merge close segments together. The energy VAD post-processes can be
        useful for having a fine-grained voice activity detection.
        The energy thresholds is  managed by activation_th and
        deactivation_th (see below).
    double_check: bool
        If True, double checks (using the neural VAD) that the candidate
        speech segments actually contain speech. A threshold on the mean
        posterior probabilities provided by the neural network is applied
        based on the speech_th parameter (see below).
    activation_th:  float
        Threshold of the neural posteriors above which starting a speech segment.
    deactivation_th: float
        Threshold of the neural posteriors below which ending a speech segment.
    en_activation_th: float
        A new speech segment is started it the energy is above activation_th.
        This is active only if apply_energy_VAD is True.
    en_deactivation_th: float
        The segment is considered ended when the energy is <= deactivation_th.
        This is active only if apply_energy_VAD is True.
    speech_th: float
        Threshold on the mean posterior probability within the candidate
        speech segment. Below that threshold, the segment is re-assigned to
        a non-speech region. This is active only if double_check is True.
    close_th: float
        If the distance between boundaries is smaller than close_th, the
        segments will be merged.
    len_th: float
        If the length of the segment is smaller than len_th, the segments
        will be merged.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[Path] = None,
        args: Optional[Dict[str, typing.Any]] = None,
        unknown_args: Optional[typing.Iterable[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters for segmentation from a config path or command-line arguments

        Parameters
        ----------
        config_path: :class:`~pathlib.Path`
            Config path
        args: dict[str, Any]
            Parsed arguments
        unknown_args: list[str]
            Optional list of arguments that were not parsed

        Returns
        -------
        dict[str, Any]
            Configuration parameters
        """
        global_params = {}
        if config_path and os.path.exists(config_path):
            data = load_configuration(config_path)
            for k, v in data.items():
                if k == "features":
                    if "type" in v:
                        v["feature_type"] = v["type"]
                        del v["type"]
                    global_params.update(v)
                else:
                    if v is None and k in cls.nullable_fields:
                        v = []
                    global_params[k] = v
        global_params.update(cls.parse_args(args, unknown_args))
        return global_params

    def segment_vad_arguments(self) -> List[SegmentVadArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`

        Returns
        -------
        list[SegmentVadArguments]
            Arguments for processing
        """
        return [
            SegmentVadArguments(
                j.id,
                getattr(self, "session", ""),
                self.working_log_directory.joinpath(f"segment_vad.{j.id}.log"),
                j.construct_path(self.split_directory, "vad", "scp"),
                self.segmentation_options,
            )
            for j in self.jobs
        ]

    def segment_vad_speechbrain(self) -> None:
        """
        Run segmentation based off of VAD.

        See Also
        --------
        :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`
            Multiprocessing helper function for each job
        segment_vad_arguments
            Job method for generating arguments for helper function
        """

        old_utts = set()
        new_utts = []
        kwargs = self.segmentation_options
        kwargs.pop("frame_shift")
        with tqdm(
            total=self.num_utterances, disable=config.QUIET
        ) as pbar, self.session() as session:
            utt_index = session.query(sqlalchemy.func.max(Utterance.id)).scalar()
            if not utt_index:
                utt_index = 0
            utt_index += 1
            files: List[File] = (
                session.query(File, Utterance)
                .options(joinedload(File.sound_file))
                .join(Utterance.file)
            )
            for f, u in files:
                boundaries = self.vad_model.get_speech_segments(
                    str(f.sound_file.sound_file_path), **kwargs
                ).numpy()
                for i in range(boundaries.shape[0]):
                    old_utts.add(u.id)
                    begin, end = boundaries[i, :]
                    begin -= self.segment_padding
                    end += self.segment_padding
                    begin = max(0.0, begin)
                    end = min(f.sound_file.duration, end)
                    new_utts.append(
                        {
                            "id": utt_index,
                            "begin": begin,
                            "end": end,
                            "text": "speech",
                            "speaker_id": u.speaker_id,
                            "file_id": u.file_id,
                            "oovs": "",
                            "normalized_text": "",
                            "features": "",
                            "in_subset": False,
                            "ignored": False,
                            "channel": u.channel,
                        }
                    )
                    utt_index += 1
                pbar.update(1)
            session.query(Utterance).filter(Utterance.id.in_(old_utts)).delete()
            session.bulk_insert_mappings(
                Utterance, new_utts, return_defaults=False, render_nulls=True
            )
            session.commit()

    def segment_vad_mfa(self) -> None:
        """
        Run segmentation based off of VAD.

        See Also
        --------
        :class:`~montreal_forced_aligner.segmenter.SegmentVadFunction`
            Multiprocessing helper function for each job
        segment_vad_arguments
            Job method for generating arguments for helper function
        """

        arguments = self.segment_vad_arguments()
        old_utts = set()
        new_utts = []

        with tqdm(
            total=self.num_utterances, disable=config.QUIET
        ) as pbar, self.session() as session:
            utterances = session.query(
                Utterance.id, Utterance.channel, Utterance.speaker_id, Utterance.file_id
            )
            utterance_cache = {}
            for u_id, channel, speaker_id, file_id in utterances:
                utterance_cache[u_id] = (channel, speaker_id, file_id)
            for utt, segments in run_kaldi_function(SegmentVadFunction, arguments, pbar.update):
                old_utts.add(utt)
                channel, speaker_id, file_id = utterance_cache[utt]
                for seg in segments:
                    new_utts.append(
                        {
                            "begin": seg.begin,
                            "end": seg.end,
                            "text": "speech",
                            "speaker_id": speaker_id,
                            "file_id": file_id,
                            "oovs": "",
                            "normalized_text": "",
                            "features": "",
                            "in_subset": False,
                            "ignored": False,
                            "channel": channel,
                        }
                    )
            session.query(Utterance).filter(Utterance.id.in_(old_utts)).delete()
            session.bulk_insert_mappings(
                Utterance, new_utts, return_defaults=False, render_nulls=True
            )
            session.commit()

    def setup(self) -> None:
        """Setup segmentation"""
        super().setup()
        self.create_new_current_workflow(WorkflowType.segmentation)
        log_dir = self.working_directory.joinpath("log")
        os.makedirs(log_dir, exist_ok=True)
        try:
            if self.speechbrain:
                self.initialize_database()
                self._load_corpus()
            else:
                self.load_corpus()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def segment(self) -> None:
        """
        Performs VAD and segmentation into utterances

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.setup()
        self.create_new_current_workflow(WorkflowType.segmentation)
        wf = self.current_workflow
        if wf.done:
            logger.info("Segmentation already done, skipping.")
            return
        try:
            if not self.speechbrain:
                self.compute_vad()
                self.segment_vad_mfa()
            else:
                self.segment_vad_speechbrain()
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"done": True}
                )
                session.commit()
        except Exception as e:
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"dirty": True}
                )
                session.commit()
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def export_files(self, output_directory: str, output_format: Optional[str] = None) -> None:
        """
        Export the results of segmentation as TextGrids

        Parameters
        ----------
        output_directory: str
            Directory to save segmentation TextGrids
        output_format: str, optional
            Format to force output files into
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        os.makedirs(output_directory, exist_ok=True)
        with self.session() as session:
            for f in session.query(File).options(
                selectinload(File.utterances).joinedload(Utterance.speaker, innerjoin=True),
                joinedload(File.sound_file, innerjoin=True),
                joinedload(File.text_file),
            ):
                f.save(output_directory, output_format=output_format)


class TranscriptionSegmenter(
    VadConfigMixin, TranscriberMixin, SpeechbrainSegmenterMixin, TopLevelMfaWorker
):
    def __init__(self, acoustic_model_path: Path = None, **kwargs):
        self.acoustic_model = AcousticModel(acoustic_model_path)
        kw = self.acoustic_model.parameters
        kw["apply_energy_vad"] = True
        kw.update(kwargs)
        super().__init__(**kw)

    def setup(self) -> None:
        TopLevelMfaWorker.setup(self)

        self.create_new_current_workflow(WorkflowType.segmentation)
        self.setup_acoustic_model()

        self.dictionary_setup()

        self._load_corpus()

        self.initialize_jobs()

        self.normalize_text()

        self.write_lexicon_information(write_disambiguation=False)

    def setup_acoustic_model(self):
        self.acoustic_model.validate(self)
        self.acoustic_model.export_model(self.model_directory)
        self.acoustic_model.export_model(self.working_directory)
        self.acoustic_model.log_details()

    def segment(self):
        """
        Performs VAD and segmentation into utterances

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        self.setup()
        self.create_new_current_workflow(WorkflowType.segmentation)
        wf = self.current_workflow
        if wf.done:
            logger.info("Segmentation already done, skipping.")
            return
        try:
            self.segment_transcripts()
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"done": True}
                )
                session.commit()
        except Exception as e:
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"dirty": True}
                )
                session.commit()
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def segment_transcript_arguments(self) -> List[SegmentTranscriptArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.segmenter.SegmentTranscriptFunction`

        Returns
        -------
        list[SegmentTranscriptArguments]
            Arguments for processing
        """
        decode_options = self.decode_options
        boost_silence = decode_options.pop("boost_silence", 1.0)
        if boost_silence != 1.0:
            self.acoustic_model.acoustic_model.boost_silence(
                self.acoustic_model.transition_model, self.silence_symbols, boost_silence
            )
        return [
            SegmentTranscriptArguments(
                j.id,
                getattr(self, "session", ""),
                self.working_log_directory.joinpath(f"segment_vad.{j.id}.log"),
                self.acoustic_model,
                self.vad_model,
                self.lexicon_compilers,
                self.mfcc_options,
                self.vad_options,
                self.segmentation_options,
                self.decode_options,
            )
            for j in self.jobs
        ]

    def segment_transcripts(self) -> None:

        arguments = self.segment_transcript_arguments()
        old_utts = set()
        new_utterance_mapping = []

        with tqdm(
            total=self.num_utterances, disable=config.QUIET
        ) as pbar, self.session() as session:
            utterances = session.query(Utterance.id, Utterance.speaker_id, Utterance.file_id)
            utterance_cache = {}
            for u_id, speaker_id, file_id in utterances:
                utterance_cache[u_id] = (speaker_id, file_id)
            for utt, new_utts in run_kaldi_function(
                SegmentTranscriptFunction, arguments, pbar.update
            ):
                old_utts.add(utt)
                speaker_id, file_id = utterance_cache[utt]
                for new_utt in new_utts:
                    new_utterance_mapping.append(
                        {
                            "begin": new_utt.segment.begin,
                            "end": new_utt.segment.end,
                            "speaker_id": speaker_id,
                            "file_id": file_id,
                            "oovs": "",
                            "text": new_utt.transcript,
                            "normalized_text": new_utt.transcript,
                            "features": "",
                            "in_subset": False,
                            "ignored": False,
                            "channel": new_utt.segment.channel,
                        }
                    )
            session.query(Utterance).filter(Utterance.id.in_(old_utts)).delete()
            session.bulk_insert_mappings(
                Utterance, new_utterance_mapping, return_defaults=False, render_nulls=True
            )
            session.commit()

    def segment_transcript(self, utterance_id: int):
        with self.session() as session:
            # interjection_words = [x for x, in session.query(Word.word).filter(Word.word_type == WordType.interjection)]
            interjection_words = []
            utterance = full_load_utterance(session, utterance_id)
            if self.acoustic_model.language is not Language.unknown:
                tokenizer = generate_language_tokenizer(self.acoustic_model.language)
                utterance.normalized_text, utterance.normalized_character_text = tokenizer(
                    utterance.text
                )
                session.flush()
            if self.g2p_model is None:
                lexicon_compiler = self.lexicon_compilers[utterance.speaker.dictionary_id]
            else:
                lexicon_compiler = self.acoustic_model.lexicon_compiler
                lexicon_compiler.disambiguation = bool(interjection_words)
                words = set(utterance.normalized_text.split())
                query = (
                    session.query(Word, Pronunciation)
                    .join(Pronunciation.word)
                    .filter(Word.dictionary_id == utterance.speaker.dictionary_id)
                    .filter(Word.word.in_(words))
                    .order_by(Word.word)
                )
                for w, p in query:
                    lexicon_compiler.word_table.add_symbol(w.word)
                    lexicon_compiler.pronunciations.append(
                        KalpyPronunciation(
                            w.word,
                            p.pronunciation,
                            p.probability,
                            p.silence_after_probability,
                            p.silence_before_correction,
                            p.non_silence_before_correction,
                            None,
                        )
                    )
                to_g2p = set()
                word_to_g2p_mapping = collections.defaultdict(set)
                for w, pron in zip(
                    utterance.normalized_text.split(), utterance.normalized_character_text.split()
                ):
                    if not lexicon_compiler.word_table.member(w):
                        word_to_g2p_mapping[w].add(pron)
                        to_g2p.add(pron)
                if to_g2p:
                    from montreal_forced_aligner.g2p.generator import PyniniGenerator

                    gen = PyniniGenerator(
                        g2p_model_path=self.g2p_model.source,
                        word_list=to_g2p,
                        num_pronunciations=1,
                        strict_graphemes=True,
                    )
                    g2pped = gen.generate_pronunciations()
                    for w, ps in word_to_g2p_mapping.items():
                        pronunciations = [g2pped[x][0] for x in ps if x in g2pped and g2pped[x]]
                        if not pronunciations:
                            pronunciations = [self.oov_phone]
                        lexicon_compiler.word_table.add_symbol(w)
                        print(w, ps, pronunciations)
                        for p in pronunciations:
                            lexicon_compiler.pronunciations.append(
                                KalpyPronunciation(
                                    w,
                                    p,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                            )
                lexicon_compiler.compute_disambiguation_symbols()
                lexicon_compiler.create_fsts()

            new_utterances = segment_utterance_transcript(
                self.acoustic_model,
                utterance.to_kalpy(),
                lexicon_compiler,
                self.vad_model if self.speechbrain else None,
                self.segmentation_options,
                interjection_words=interjection_words,
                mfcc_options=self.mfcc_options if not self.speechbrain else None,
                vad_options=self.vad_options if not self.speechbrain else None,
                **self.decode_options,
            )
        return new_utterances

    def export_files(self, output_directory: str, output_format: Optional[str] = None) -> None:
        """
        Export the results of segmentation as TextGrids

        Parameters
        ----------
        output_directory: str
            Directory to save segmentation TextGrids
        output_format: str, optional
            Format to force output files into
        """
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        os.makedirs(output_directory, exist_ok=True)
        with self.session() as session:
            for f in session.query(File).options(
                selectinload(File.utterances).joinedload(Utterance.speaker, innerjoin=True),
                joinedload(File.sound_file, innerjoin=True),
                joinedload(File.text_file),
            ):
                f.save(output_directory, output_format=output_format)
