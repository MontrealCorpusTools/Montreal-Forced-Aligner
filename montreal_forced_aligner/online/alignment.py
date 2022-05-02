"""Classes for calculating alignments online"""
from __future__ import annotations

import os
import subprocess
import typing

from montreal_forced_aligner.abc import MetaDict
from montreal_forced_aligner.corpus.classes import UtteranceData
from montreal_forced_aligner.data import CtmInterval, MfaArguments
from montreal_forced_aligner.helper import make_safe
from montreal_forced_aligner.textgrid import process_ctm_line
from montreal_forced_aligner.utils import KaldiFunction, thirdparty_binary


class OnlineAlignmentArguments(MfaArguments):
    """
    Arguments for performing alignment online on single utterances
    """

    working_directory: str
    sox_string: str
    utterance_data: UtteranceData
    mfcc_options: MetaDict
    pitch_options: MetaDict
    feature_options: MetaDict
    align_options: MetaDict
    model_path: str
    tree_path: str
    disambig_path: str
    lexicon_fst_path: str
    word_boundary_int_path: str
    reversed_phone_mapping: typing.Dict[int, str]
    optional_silence_phone: str
    silence_words: typing.Set[str]


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
        self.align_options = args.align_options
        self.model_path = args.model_path
        self.tree_path = args.tree_path
        self.disambig_path = args.disambig_path
        self.lexicon_fst_path = args.lexicon_fst_path
        self.word_boundary_int_path = args.word_boundary_int_path
        self.reversed_phone_mapping = args.reversed_phone_mapping
        self.optional_silence_phone = args.optional_silence_phone
        self.silence_words = args.silence_words

    def cleanup_intervals(self, utterance_name: int, intervals: typing.List[CtmInterval]):
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
        utterance_name = utterance_name
        utterance_begin = self.utterance.begin
        current_word_begin = None
        words = self.utterance.normalized_text
        words_index = 0
        for interval in intervals:
            interval.begin += utterance_begin
            interval.end += utterance_begin
            phone_label = self.reversed_phone_mapping[int(interval.label)]
            if phone_label == self.optional_silence_phone:
                if words_index < len(words) and words[words_index] in self.silence_words:
                    interval.label = phone_label
                else:
                    interval.label = phone_label
                    actual_phone_intervals.append(interval)
                    continue
            phone, position = phone_label.split("_")
            if position in {"B", "S"}:
                current_word_begin = interval.begin
            if position in {"E", "S"}:
                actual_word_intervals.append(
                    CtmInterval(
                        current_word_begin, interval.end, words[words_index], utterance_name
                    )
                )
                words_index += 1
                current_word_begin = None
            interval.label = phone
            actual_phone_intervals.append(interval)
        return actual_word_intervals, actual_phone_intervals

    def run(self):
        """Run the function"""
        wav_path = os.path.join(self.working_directory, "wav.scp")
        likelihood_path = os.path.join(self.working_directory, "likelihoods.scp")
        feat_path = os.path.join(self.working_directory, "feats.scp")
        cmvn_path = os.path.join(self.working_directory, "cmvn.scp")
        utt2spk_path = os.path.join(self.working_directory, "utt2spk.scp")
        segment_path = os.path.join(self.working_directory, "segments.scp")
        text_int_path = os.path.join(self.working_directory, "text.int")
        lda_mat_path = os.path.join(self.working_directory, "lda.mat")
        fst_path = os.path.join(self.working_directory, "fsts.ark")
        if self.align_options["boost_silence"] != 1.0:
            mdl_string = f"gmm-boost-silence --boost={self.align_options['boost_silence']} {self.align_options['optional_silence_csl']} {self.model_path} - |"

        else:
            mdl_string = self.model_path
        if not os.path.exists(lda_mat_path):
            lda_mat_path = None
        with open(self.log_path, "w") as log_file:
            proc = subprocess.Popen(
                [
                    thirdparty_binary("compile-train-graphs"),
                    f"--read-disambig-syms={self.disambig_path}",
                    self.tree_path,
                    self.model_path,
                    self.lexicon_fst_path,
                    f"ark:{text_int_path}",
                    f"ark:{fst_path}",
                ],
                stderr=log_file,
                encoding="utf8",
                env=os.environ,
            )
            proc.communicate()
            if not os.path.exists(feat_path):
                use_pitch = self.pitch_options.pop("use-pitch")
                mfcc_base_command = [thirdparty_binary("compute-mfcc-feats"), "--verbose=2"]
                for k, v in self.mfcc_options.items():
                    mfcc_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
                mfcc_base_command += ["ark:-", "ark:-"]
                seg_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-segments"),
                        f"scp:{wav_path}",
                        segment_path,
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                mfcc_proc = subprocess.Popen(
                    mfcc_base_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=seg_proc.stdout,
                    env=os.environ,
                )
                if use_pitch:
                    pitch_base_command = [
                        thirdparty_binary("compute-and-process-kaldi-pitch-feats"),
                        "--verbose=2",
                    ]
                    for k, v in self.pitch_options.items():
                        pitch_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
                        if k == "delta-pitch":
                            pitch_base_command.append(f"--delta-pitch-noise-stddev={make_safe(v)}")
                    pitch_command = " ".join(pitch_base_command)
                    if os.path.exists(segment_path):
                        segment_command = (
                            f"extract-segments scp:{wav_path} {segment_path} ark:- | "
                        )
                        pitch_input = "ark:-"
                    else:
                        segment_command = ""
                        pitch_input = f"scp:{wav_path}"
                    pitch_feat_string = (
                        f"ark,s,cs:{segment_command}{pitch_command} {pitch_input} ark:- |"
                    )
                    length_tolerance = 2
                    feature_proc = subprocess.Popen(
                        [
                            thirdparty_binary("paste-feats"),
                            f"--length-tolerance={length_tolerance}",
                            "ark:-",
                            pitch_feat_string,
                            "ark:-",
                        ],
                        stdin=mfcc_proc.stdout,
                        env=os.environ,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                    )
                else:
                    feature_proc = mfcc_proc
                cvmn_proc = subprocess.Popen(
                    [thirdparty_binary("apply-cmvn-sliding"), "--center", "ark:-", "ark:-"],
                    stdin=feature_proc.stdout,
                    env=os.environ,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                )
                if lda_mat_path is not None:
                    splice_proc = subprocess.Popen(
                        [
                            thirdparty_binary("splice-feats"),
                            f'--left-context={self.feature_options["splice_left_context"]}',
                            f'--right-context={self.feature_options["splice_right_context"]}',
                            "ark:-",
                            "ark:-",
                        ],
                        stdin=cvmn_proc.stdout,
                        env=os.environ,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                    )
                    transform_proc = subprocess.Popen(
                        [thirdparty_binary("transform-feats"), lda_mat_path, "ark:-", "ark:-"],
                        stdin=splice_proc.stdout,
                        env=os.environ,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                    )
                elif self.feature_options["uses_deltas"]:
                    transform_proc = subprocess.Popen(
                        [thirdparty_binary("add-deltas"), "ark:-", "ark:-"],
                        stdin=cvmn_proc.stdout,
                        env=os.environ,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
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
                        "ark:-",
                        f"ark,t:{likelihood_path}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    encoding="utf8",
                    stdin=transform_proc.stdout,
                    env=os.environ,
                )
            else:
                feat_string = f"ark,s,cs:apply-cmvn --utt2spk=ark:{utt2spk_path} scp:{cmvn_path} scp:{feat_path} ark:- |"
                if lda_mat_path is not None:
                    feat_string += f" splice-feats --left-context={self.feature_options['splice_left_context']} --right-context={self.feature_options['splice_right_context']} ark:- ark:- |"
                    feat_string += f" transform-feats {lda_mat_path} ark:- ark:- |"
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
                            "ark:-",
                            f"ark,t:{likelihood_path}",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        encoding="utf8",
                        env=os.environ,
                    )
            lin_proc = subprocess.Popen(
                [
                    thirdparty_binary("linear-to-nbest"),
                    "ark:-",
                    f"ark:{text_int_path}",
                    "",
                    "",
                    "ark:-",
                ],
                stdin=align_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            align_words_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-align-words"),
                    self.word_boundary_int_path,
                    self.model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=lin_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            phone_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-to-phone-lattice"),
                    self.model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stdin=align_words_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            nbest_proc = subprocess.Popen(
                [
                    thirdparty_binary("nbest-to-ctm"),
                    "--print-args=false",
                    f"--frame-shift={round(self.feature_options['frame_shift']/1000,4)}",
                    "ark:-",
                    "-",
                ],
                stdin=phone_proc.stdout,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            intervals = []
            log_likelihood = None
            for line in nbest_proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    interval = process_ctm_line(line)
                    intervals.append(interval)
                except ValueError:
                    pass
            nbest_proc.wait()
            with open(likelihood_path, "r", encoding="utf8") as f:
                try:
                    log_likelihood = float(f.read().split()[-1])
                except ValueError:
                    pass
        return *self.cleanup_intervals(0, intervals), log_likelihood
