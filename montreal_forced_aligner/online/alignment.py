"""Classes for calculating alignments online"""
from __future__ import annotations

import os
import subprocess
import typing
from pathlib import Path

from sqlalchemy.orm import Session

from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.corpus.classes import UtteranceData
from montreal_forced_aligner.corpus.features import online_feature_proc
from montreal_forced_aligner.data import CtmInterval, MfaArguments, WordCtmInterval, WordType
from montreal_forced_aligner.db import Dictionary, Phone, Pronunciation, Word
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import parse_ctm_output, thirdparty_binary

if typing.TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from dataclassy import dataclass


@dataclass
class OnlineAlignmentArguments(MfaArguments):
    """
    Arguments for performing alignment online on single utterances
    """

    working_directory: Path
    sox_string: str
    utterance_data: UtteranceData
    mfcc_options: MetaDict
    pitch_options: MetaDict
    feature_options: MetaDict
    lda_options: MetaDict
    align_options: MetaDict
    model_path: Path
    tree_path: Path
    dictionary_id: int


class OnlineAlignmentFunction(KaldiFunction):
    """
    Multiprocessing function to align an utterance

    See Also
    --------
    :meth:`.PretrainedAligner.align_one_utterance`
        Main function that calls this function in parallel

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.online.alignment.OnlineAlignmentArguments`
        Arguments for the function
    """

    def __init__(self, args: OnlineAlignmentArguments):
        super(OnlineAlignmentFunction, self).__init__(args)
        self.working_directory = args.working_directory
        self.sox_string = args.sox_string
        self.utterance = args.utterance_data
        self.mfcc_options = args.mfcc_options
        self.pitch_options = args.pitch_options
        self.feature_options = args.feature_options
        self.lda_options = args.lda_options
        self.align_options = args.align_options
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.dictionary_id = args.dictionary_id
        self.reversed_phone_mapping = {}
        self.reversed_word_mapping = {}
        self.pronunciation_mapping = {}
        self.phone_mapping = {}
        self.silence_words = set()

    def cleanup_intervals(
        self,
        utterance_name: int,
        intervals: typing.List[CtmInterval],
        word_pronunciations: typing.List[typing.Tuple[str, typing.List[str]]],
    ) -> typing.Tuple[typing.List[CtmInterval], typing.List[CtmInterval], typing.List[int]]:
        """
        Clean up phone intervals to remove silence

        Parameters
        ----------
        intervals: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
            Intervals to process

        Returns
        -------
        list[:class:`~montreal_forced_aligner.data.CtmInterval`]
            Cleaned up intervals
        """
        actual_phone_intervals = []
        actual_word_intervals = []
        utterance_begin = self.utterance.begin
        current_word_begin = None
        words_index = 0
        phone_word_mapping = []
        current_phones = []
        for interval in intervals:
            interval.begin += utterance_begin
            interval.end += utterance_begin
            if interval.label == self.optional_silence_phone:
                interval.label = self.phone_to_phone_id[interval.label]
                cur_word = word_pronunciations[words_index]
                actual_phone_intervals.append(interval)
                actual_word_intervals.append(
                    WordCtmInterval(
                        interval.begin,
                        interval.end,
                        word_pronunciations[words_index][0],
                        self.pronunciation_mapping[(cur_word[0], " ".join(cur_word[1]))],
                    )
                )
                phone_word_mapping.append(len(actual_word_intervals) - 1)
                current_word_begin = None
                current_phones = []
                words_index += 1
                continue
            if current_word_begin is None:
                current_word_begin = interval.begin
            current_phones.append(interval.label)
            cur_word = word_pronunciations[words_index]
            if current_phones == cur_word[1]:
                actual_word_intervals.append(
                    WordCtmInterval(
                        current_word_begin,
                        interval.end,
                        cur_word[0],
                        self.pronunciation_mapping[(cur_word[0], " ".join(cur_word[1]))],
                    )
                )
                for _ in range(len(current_phones)):
                    phone_word_mapping.append(len(actual_word_intervals) - 1)
                current_word_begin = None
                current_phones = []
                words_index += 1
            interval.label = self.phone_to_phone_id[interval.label]
            actual_phone_intervals.append(interval)

        return actual_word_intervals, actual_phone_intervals, phone_word_mapping

    def _run(self) -> typing.Tuple[typing.List[CtmInterval], typing.List[CtmInterval], float]:
        """Run the function"""
        with Session(self.db_engine()) as session:
            d: Dictionary = session.get(Dictionary, self.dictionary_id)

            self.clitic_marker = d.clitic_marker
            self.silence_word = d.silence_word
            self.oov_word = d.oov_word
            self.optional_silence_phone = d.optional_silence_phone
            lexicon_path = d.lexicon_fst_path
            align_lexicon_path = d.align_lexicon_path
            disambig_path = d.disambiguation_symbols_int_path
            silence_words = (
                session.query(Word.id)
                .filter(Word.dictionary_id == self.dictionary_id)
                .filter(Word.word_type == WordType.silence)
            )
            self.silence_words.update(x for x, in silence_words)

            words = session.query(Word.mapping_id, Word.id).filter(
                Word.dictionary_id == self.dictionary_id
            )
            for m_id, w in words:
                self.reversed_word_mapping[m_id] = w
        self.phone_to_phone_id = {}
        ds = session.query(Phone.phone, Phone.id, Phone.mapping_id).all()
        for phone, p_id, mapping_id in ds:
            self.reversed_phone_mapping[mapping_id] = phone
            self.phone_to_phone_id[phone] = p_id
            self.phone_mapping[phone] = mapping_id

        pronunciations = (
            session.query(Word.id, Pronunciation.pronunciation, Pronunciation.id)
            .join(Pronunciation.word)
            .filter(Word.dictionary_id == self.dictionary_id)
        )
        for w_id, pron, p_id in pronunciations:
            self.pronunciation_mapping[(w_id, pron)] = p_id
        wav_path = self.working_directory.joinpath("wav.scp")
        likelihood_path = self.working_directory.joinpath("likelihoods.scp")
        feat_path = self.working_directory.joinpath("feats.scp")
        segment_path = self.working_directory.joinpath("segments.scp")
        text_int_path = self.working_directory.joinpath("text.int")
        lda_mat_path = self.working_directory.joinpath("lda.mat")
        fst_path = self.working_directory.joinpath("fsts.ark")
        ali_path = self.working_directory.joinpath("ali.ark")
        if self.align_options["boost_silence"] != 1.0:
            mdl_string = f"gmm-boost-silence --boost={self.align_options['boost_silence']} {self.align_options['optional_silence_csl']} {self.model_path} - |"

        else:
            mdl_string = self.model_path
        if not os.path.exists(lda_mat_path):
            lda_mat_path = None
        with mfa_open(self.log_path, "w") as log_file:
            proc = subprocess.Popen(
                [
                    thirdparty_binary("compile-train-graphs"),
                    f"--read-disambig-syms={disambig_path}",
                    self.tree_path,
                    self.model_path,
                    lexicon_path,
                    f"ark:{text_int_path}",
                    f"ark:{fst_path}",
                ],
                stderr=log_file,
                encoding="utf8",
                env=os.environ,
            )
            proc.communicate()
            if not os.path.exists(feat_path):
                feature_proc = online_feature_proc(
                    self.working_directory,
                    wav_path,
                    segment_path,
                    self.mfcc_options,
                    self.pitch_options,
                    self.lda_options,
                    log_file,
                )

                # Features done, alignment
                align_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-align-compiled"),
                        f"--transition-scale={self.align_options['transition_scale']}",
                        f"--acoustic-scale={self.align_options['acoustic_scale']}",
                        f"--self-loop-scale={self.align_options['self_loop_scale']}",
                        f"--beam={self.align_options['beam']}",
                        f"--retry-beam={self.align_options['retry_beam']}",
                        "--careful=false",
                        mdl_string,
                        f"ark:{fst_path}",
                        "ark:-",
                        f"ark:{ali_path}",
                        f"ark,t:{likelihood_path}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    encoding="utf8",
                    stdin=feature_proc.stdout,
                    env=os.environ,
                )
            else:
                feat_string = f"ark,s,cs:splice-feats --left-context={self.lda_options['splice_left_context']} --right-context={self.lda_options['splice_right_context']} scp,s,cs:\"{feat_path}\" ark:- |"
                feat_string += f' transform-feats "{lda_mat_path}" ark:- ark:- |'
                align_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-align-compiled"),
                        f"--transition-scale={self.align_options['transition_scale']}",
                        f"--acoustic-scale={self.align_options['acoustic_scale']}",
                        f"--self-loop-scale={self.align_options['self_loop_scale']}",
                        f"--beam={self.align_options['beam']}",
                        f"--retry-beam={self.align_options['retry_beam']}",
                        "--careful=false",
                        mdl_string,
                        f"ark:{fst_path}",
                        feat_string,
                        f"ark:{ali_path}",
                        f"ark,t:{likelihood_path}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    encoding="utf8",
                    env=os.environ,
                )
            align_proc.communicate()
            ctm_proc = subprocess.Popen(
                [
                    thirdparty_binary("ali-to-phones"),
                    "--ctm-output",
                    f"--frame-shift={round(self.feature_options['frame_shift']/1000,4)}",
                    self.model_path,
                    f"ark:{ali_path}",
                    "-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )

            phones_proc = subprocess.Popen(
                [
                    thirdparty_binary("ali-to-phones"),
                    self.model_path,
                    f"ark:{ali_path}",
                    "ark,t:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            prons_proc = subprocess.Popen(
                [
                    thirdparty_binary("phones-to-prons"),
                    align_lexicon_path,
                    str(self.phone_mapping["#1"]),
                    str(self.phone_mapping["#2"]),
                    "ark:-",
                    f"ark:{text_int_path}",
                    "ark,t:-",
                ],
                stdin=phones_proc.stdout,
                stderr=log_file,
                encoding="utf8",
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            for utterance, intervals in parse_ctm_output(ctm_proc, self.reversed_phone_mapping):
                while True:
                    prons_line = prons_proc.stdout.readline().strip()
                    if prons_line:
                        break
                utt_id, prons_line = prons_line.split(maxsplit=1)
                prons = prons_line.split(";")
                word_pronunciations = []
                for pron in prons:
                    pron = pron.strip()
                    if not pron:
                        continue
                    pron = pron.split()
                    word = pron.pop(0)
                    word = self.reversed_word_mapping[int(word)]
                    pron = [self.reversed_phone_mapping[int(x)] for x in pron]
                    word_pronunciations.append((word, pron))
                word_intervals, phone_intervals, phone_word_mapping = self.cleanup_intervals(
                    utterance, intervals, word_pronunciations
                )
                log_likelihood = None
                with mfa_open(likelihood_path, "r") as f:
                    for line in f:
                        line = line.strip().split()
                        log_likelihood = float(line[-1])
                yield utterance, word_intervals, phone_intervals, phone_word_mapping, log_likelihood
            self.check_call(ctm_proc)
