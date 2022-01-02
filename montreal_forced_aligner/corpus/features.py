"""Classes for configuring feature generation"""
from __future__ import annotations

import os
import re
import subprocess
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Union

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.utils import thirdparty_binary

if TYPE_CHECKING:
    SpeakerCharacterType = Union[str, int]
    from montreal_forced_aligner.abc import MetaDict

__all__ = [
    "FeatureConfigMixin",
    "CalcFmllrFunction",
    "ComputeVadFunction",
    "VadArguments",
    "MfccArguments",
    "CalcFmllrArguments",
    "ExtractIvectorsFunction",
    "ExtractIvectorsArguments",
]


class VadArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`"""

    log_path: str
    feats_scp_path: str
    vad_scp_path: str
    vad_options: MetaDict


class MfccArguments(NamedTuple):
    """
    Arguments for :class:`~montreal_forced_aligner.corpus.features.MfccFunction`
    """

    log_path: str
    wav_path: str
    segment_path: str
    feats_scp_path: str
    mfcc_options: MetaDict


class CalcFmllrArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.CalcFmllrFunction`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    ali_paths: Dict[str, str]
    ali_model_path: str
    model_path: str
    spk2utt_paths: Dict[str, str]
    trans_paths: Dict[str, str]
    fmllr_options: MetaDict


def make_safe(value: Any) -> str:
    """
    Transform an arbitrary value into a string

    Parameters
    ----------
    value: Any
        Value to make safe

    Returns
    -------
    str
        Safe value
    """
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class MfccFunction(KaldiFunction):
    """
    Multiprocessing function for generating MFCC features

    See Also
    --------
    :meth:`.AcousticCorpusMixin.mfcc`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.MfccArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.* Copied (?P<num_utterances>\d+) feature matrices.")

    def __init__(self, args: MfccArguments):
        self.log_path = args.log_path
        self.wav_path = args.wav_path
        self.segment_path = args.segment_path
        self.feats_scp_path = args.feats_scp_path
        self.mfcc_options = args.mfcc_options

    def run(self):
        """Run the function"""
        with open(self.log_path, "w") as log_file:
            mfcc_base_command = [thirdparty_binary("compute-mfcc-feats"), "--verbose=2"]
            raw_ark_path = self.feats_scp_path.replace(".scp", ".ark")
            for k, v in self.mfcc_options.items():
                mfcc_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
            if os.path.exists(self.segment_path):
                mfcc_base_command += ["ark:-", "ark:-"]
                seg_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-segments"),
                        f"scp,p:{self.wav_path}",
                        self.segment_path,
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                comp_proc = subprocess.Popen(
                    mfcc_base_command,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    stdin=seg_proc.stdout,
                    env=os.environ,
                )
            else:
                mfcc_base_command += [f"scp,p:{self.wav_path}", "ark:-"]
                comp_proc = subprocess.Popen(
                    mfcc_base_command, stdout=subprocess.PIPE, stderr=log_file, env=os.environ
                )
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--verbose=2",
                    "--compress=true",
                    "ark:-",
                    f"ark,scp:{raw_ark_path},{self.feats_scp_path}",
                ],
                stdin=comp_proc.stdout,
                stderr=subprocess.PIPE,
                env=os.environ,
                encoding="utf8",
            )
            for line in copy_proc.stderr:
                m = self.progress_pattern.match(line.strip())
                if m:
                    yield int(m.group("num_utterances"))


class ComputeVadFunction(KaldiFunction):
    """
    Multiprocessing function to compute voice activity detection

    See Also
    --------
    :meth:`.AcousticCorpusMixin.compute_vad`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.compute_vad_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-vad`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.VadArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*processed (?P<done>\d+) utterances.*(?P<no_feats>\d+) had.*(?P<unvoiced>\d+) were.*"
    )

    def __init__(self, args: VadArguments):
        self.log_path = args.log_path
        self.feats_scp_path = args.feats_scp_path
        self.vad_scp_path = args.vad_scp_path
        self.vad_options = args.vad_options

    def run(self):
        """Run the function"""
        with open(self.log_path, "w") as log_file:
            feats_scp_path = self.feats_scp_path
            vad_scp_path = self.vad_scp_path
            vad_proc = subprocess.Popen(
                [
                    thirdparty_binary("compute-vad"),
                    f"--vad-energy-mean-scale={self.vad_options['energy_mean_scale']}",
                    f"--vad-energy-threshold={self.vad_options['energy_threshold']}",
                    f"scp:{feats_scp_path}",
                    f"ark,t:{vad_scp_path}",
                ],
                stderr=subprocess.PIPE,
                encoding="utf8",
                env=os.environ,
            )
            for line in vad_proc.stderr:
                log_file.write(line)
                m = self.progress_pattern.match(line.strip())
                if m:
                    yield int(m.group("done")), int(m.group("no_feats")), int(m.group("unvoiced"))


class CalcFmllrFunction(KaldiFunction):
    """
    Multiprocessing function for calculating fMLLR transforms

    See Also
    --------
    :meth:`.AcousticCorpusMixin.calc_fmllr`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.calc_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-est-fmllr`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr-gpost`
        Relevant Kaldi binary
    :kaldi_src:`gmm-post-to-gpost`
        Relevant Kaldi binary
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`compose-transforms`
        Relevant Kaldi binary
    :kaldi_src:`transform-feats`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.CalcFmllrArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.*For speaker (?P<speaker>.*),.*$")

    def __init__(self, args: CalcFmllrArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.ali_paths = args.ali_paths
        self.ali_model_path = args.ali_model_path
        self.model_path = args.model_path
        self.spk2utt_paths = args.spk2utt_paths
        self.trans_paths = args.trans_paths
        self.fmllr_options = args.fmllr_options

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                feature_string = self.feature_strings[dict_name]
                ali_path = self.ali_paths[dict_name]
                spk2utt_path = self.spk2utt_paths[dict_name]
                trans_path = self.trans_paths[dict_name]
                initial = True
                if os.path.exists(trans_path):
                    initial = False
                post_proc = subprocess.Popen(
                    [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )

                weight_proc = subprocess.Popen(
                    [
                        thirdparty_binary("weight-silence-post"),
                        "0.0",
                        self.fmllr_options["silence_csl"],
                        self.ali_model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stderr=log_file,
                    stdin=post_proc.stdout,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )

                temp_trans_path = trans_path + ".tmp"
                if self.ali_model_path != self.model_path:
                    post_gpost_proc = subprocess.Popen(
                        [
                            thirdparty_binary("gmm-post-to-gpost"),
                            self.ali_model_path,
                            feature_string,
                            "ark:-",
                            "ark:-",
                        ],
                        stderr=log_file,
                        stdin=weight_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    est_proc = subprocess.Popen(
                        [
                            thirdparty_binary("gmm-est-fmllr-gpost"),
                            "--verbose=4",
                            f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                            f"--spk2utt=ark:{spk2utt_path}",
                            self.model_path,
                            feature_string,
                            "ark,s,cs:-",
                            f"ark:{temp_trans_path}",
                        ],
                        stderr=subprocess.PIPE,
                        encoding="utf8",
                        stdin=post_gpost_proc.stdout,
                        env=os.environ,
                    )

                else:

                    if not initial:
                        temp_composed_trans_path = trans_path + ".cmp.tmp"
                        est_proc = subprocess.Popen(
                            [
                                thirdparty_binary("gmm-est-fmllr"),
                                "--verbose=4",
                                f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                                f"--spk2utt=ark:{spk2utt_path}",
                                self.model_path,
                                feature_string,
                                "ark:-",
                                f"ark:{temp_trans_path}",
                            ],
                            stderr=subprocess.PIPE,
                            encoding="utf8",
                            stdin=weight_proc.stdout,
                            stdout=subprocess.PIPE,
                            env=os.environ,
                        )
                    else:
                        est_proc = subprocess.Popen(
                            [
                                thirdparty_binary("gmm-est-fmllr"),
                                "--verbose=4",
                                f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                                f"--spk2utt=ark:{spk2utt_path}",
                                self.model_path,
                                feature_string,
                                "ark,s,cs:-",
                                f"ark:{trans_path}",
                            ],
                            stderr=subprocess.PIPE,
                            encoding="utf8",
                            stdin=weight_proc.stdout,
                            env=os.environ,
                        )

                for line in est_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("speaker")
                if not initial:
                    compose_proc = subprocess.Popen(
                        [
                            thirdparty_binary("compose-transforms"),
                            "--b-is-affine=true",
                            f"ark:{temp_trans_path}",
                            f"ark:{trans_path}",
                            f"ark:{temp_composed_trans_path}",
                        ],
                        stderr=log_file,
                        env=os.environ,
                    )
                    compose_proc.communicate()

                    os.remove(trans_path)
                    os.remove(temp_trans_path)
                    os.rename(temp_composed_trans_path, trans_path)


class FeatureConfigMixin:
    """
    Class to store configuration information about MFCC generation

    Attributes
    ----------
    feature_type : str
        Feature type, defaults to "mfcc"
    use_energy : bool
        Flag for whether first coefficient should be used, defaults to False
    frame_shift : int
        number of milliseconds between frames, defaults to 10
    snip_edges : bool
        Flag for enabling Kaldi's snip edges, should be better time precision
    pitch : bool
        Flag for including pitch in features, currently nonfunctional, defaults to False
    low_frequency : int
        Frequency floor
    high_frequency : int
        Frequency ceiling
    sample_frequency : int
        Sampling frequency
    allow_downsample : bool
        Flag for whether to allow downsampling, default is True
    allow_upsample : bool
        Flag for whether to allow upsampling, default is True
    speaker_independent : bool
        Flag for whether features are speaker independent, default is True
    uses_cmvn : bool
        Flag for whether to use CMVN, default is True
    uses_deltas : bool
        Flag for whether to use delta features, default is True
    uses_splices : bool
        Flag for whether to use splices and LDA transformations, default is False
    uses_speaker_adaptation : bool
        Flag for whether to use speaker adaptation, default is False
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to "full"
    silence_weight : float
        Weight of silence in calculating LDA or fMLLR
    splice_left_context : int or None
        Number of frames to splice on the left for calculating LDA
    splice_right_context : int or None
        Number of frames to splice on the right for calculating LDA
    """

    def __init__(
        self,
        feature_type: str = "mfcc",
        use_energy: bool = False,
        frame_shift: int = 10,
        snip_edges: bool = True,
        pitch: bool = False,
        low_frequency: int = 20,
        high_frequency: int = 7800,
        sample_frequency: int = 16000,
        allow_downsample: bool = True,
        allow_upsample: bool = True,
        speaker_independent: bool = True,
        uses_cmvn: bool = True,
        uses_deltas: bool = True,
        uses_splices: bool = False,
        uses_voiced: bool = False,
        uses_speaker_adaptation: bool = False,
        fmllr_update_type: str = "full",
        silence_weight: float = 0.0,
        splice_left_context: int = 3,
        splice_right_context: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_type = feature_type
        self.use_energy = use_energy
        self.frame_shift = frame_shift
        self.snip_edges = snip_edges
        self.pitch = pitch
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.sample_frequency = sample_frequency
        self.allow_downsample = allow_downsample
        self.allow_upsample = allow_upsample
        self.speaker_independent = speaker_independent
        self.uses_cmvn = uses_cmvn
        self.uses_deltas = uses_deltas
        self.uses_splices = uses_splices
        self.uses_voiced = uses_voiced
        self.uses_speaker_adaptation = uses_speaker_adaptation
        self.fmllr_update_type = fmllr_update_type
        self.silence_weight = silence_weight
        self.splice_left_context = splice_left_context
        self.splice_right_context = splice_right_context

    @property
    def vad_options(self) -> MetaDict:
        """Abstract method for VAD options"""
        raise NotImplementedError

    @property
    def alignment_model_path(self) -> str:  # needed for fmllr
        """Abstract method for alignment model path"""
        raise NotImplementedError

    @property
    def model_path(self) -> str:  # needed for fmllr
        """Abstract method for model path"""
        raise NotImplementedError

    @property
    @abstractmethod
    def working_directory(self) -> str:
        """Abstract method for working directory"""
        ...

    @property
    @abstractmethod
    def corpus_output_directory(self) -> str:
        """Abstract method for working directory of corpus"""
        ...

    @property
    @abstractmethod
    def data_directory(self) -> str:
        """Abstract method for corpus data directory"""
        ...

    @property
    def feature_options(self) -> MetaDict:
        """Parameters for feature generation"""
        options = {
            "type": self.feature_type,
            "use_energy": self.use_energy,
            "frame_shift": self.frame_shift,
            "snip_edges": self.snip_edges,
            "low_frequency": self.low_frequency,
            "high_frequency": self.high_frequency,
            "sample_frequency": self.sample_frequency,
            "allow_downsample": self.allow_downsample,
            "allow_upsample": self.allow_upsample,
            "pitch": self.pitch,
            "uses_cmvn": self.uses_cmvn,
            "uses_deltas": self.uses_deltas,
            "uses_voiced": self.uses_voiced,
            "uses_splices": self.uses_splices,
            "uses_speaker_adaptation": self.uses_speaker_adaptation,
        }
        if self.uses_splices:
            options.update(
                {
                    "splice_left_context": self.splice_left_context,
                    "splice_right_context": self.splice_right_context,
                }
            )
        return options

    @abstractmethod
    def calc_fmllr(self) -> None:
        """Abstract method for calculating fMLLR transforms"""
        ...

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for use in calculating fMLLR transforms"""
        return {
            "fmllr_update_type": self.fmllr_update_type,
            "silence_weight": self.silence_weight,
            "silence_csl": getattr(
                self, "silence_csl", ""
            ),  # If we have silence phones from a dictionary, use them
        }

    @property
    def mfcc_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "use-energy": self.use_energy,
            "frame-shift": self.frame_shift,
            "low-freq": self.low_frequency,
            "high-freq": self.high_frequency,
            "sample-frequency": self.sample_frequency,
            "allow-downsample": self.allow_downsample,
            "allow-upsample": self.allow_upsample,
            "snip-edges": self.snip_edges,
        }


class IvectorConfigMixin(FeatureConfigMixin):
    """
    Mixin class for ivector features

    Parameters
    ----------
    ivector_dimension: int
        Dimension of ivectors
    num_gselect: int
        Gaussian-selection using diagonal model: number of Gaussians to select
    posterior_scale: float
        Scale on the acoustic posteriors, intended to account for inter-frame correlations
    min_post : float
        Minimum posterior to use (posteriors below this are pruned out)
    max_count: int
        The use of this option (e.g. --max-count 100) can make iVectors more consistent for different lengths of
        utterance, by scaling up the prior term when the data-count exceeds this value. The data-count is after
        posterior-scaling, so assuming the posterior-scale is 0.1, --max-count 100 starts having effect after 1000
        frames, or 10 seconds of data.

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters
    """

    def __init__(
        self,
        ivector_dimension=128,
        num_gselect=20,
        posterior_scale=1.0,
        min_post=0.025,
        max_count=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ivector_dimension = ivector_dimension
        self.num_gselect = num_gselect
        self.posterior_scale = posterior_scale
        self.min_post = min_post
        self.max_count = max_count

    @abstractmethod
    def extract_ivectors(self):
        """Abstract method for extracting ivectors"""
        ...

    @property
    def ivector_options(self) -> MetaDict:
        """Options for ivector training and extracting"""
        return {
            "num_gselect": self.num_gselect,
            "posterior_scale": self.posterior_scale,
            "min_post": self.min_post,
            "silence_weight": self.silence_weight,
            "max_count": self.max_count,
            "ivector_dimension": self.ivector_dimension,
            "silence_csl": getattr(
                self, "silence_csl", ""
            ),  # If we have silence phones from a dictionary, use them,
        }


class VadConfigMixin(FeatureConfigMixin):
    """
    Abstract mixin class for performing voice activity detection

    Parameters
    ----------
    use_energy: bool
        Flag for using the first coefficient of MFCCs
    energy_threshold: float
        Energy threshold above which a frame will be counted as voiced
    energy_mean_scale: float
        Proportion of the mean energy of the file that should be added to the energy_threshold

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.features.FeatureConfigMixin`
        For feature generation parameters
    """

    def __init__(self, use_energy=True, energy_threshold=5.5, energy_mean_scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.use_energy = use_energy
        self.energy_threshold = energy_threshold
        self.energy_mean_scale = energy_mean_scale

    @property
    def vad_options(self) -> MetaDict:
        """Options for performing VAD"""
        return {
            "energy_threshold": self.energy_threshold,
            "energy_mean_scale": self.energy_mean_scale,
        }


class ExtractIvectorsArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`"""

    log_path: str
    feature_string: str
    ivector_options: MetaDict
    ie_path: str
    ivectors_path: str
    model_path: str
    dubm_path: str


class ExtractIvectorsFunction(KaldiFunction):
    """
    Multiprocessing function for extracting ivectors.

    See Also
    --------
    :meth:`.IvectorCorpusMixin.extract_ivectors`
        Main function that calls this function in parallel
    :meth:`.IvectorCorpusMixin.extract_ivectors_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`ivector-extract`
        Relevant Kaldi binary
    :kaldi_src:`gmm-global-get-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-post`
        Relevant Kaldi binary
    :kaldi_src:`post-to-weights`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(r"^LOG.*Ivector norm for speaker (?P<speaker>.+) was.*")

    def __init__(self, args: ExtractIvectorsArguments):
        self.log_path = args.log_path
        self.feature_string = args.feature_string
        self.ivector_options = args.ivector_options
        self.ie_path = args.ie_path
        self.ivectors_path = args.ivectors_path
        self.model_path = args.model_path
        self.dubm_path = args.dubm_path

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:

            gmm_global_get_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-post"),
                    f"--n={self.ivector_options['num_gselect']}",
                    f"--min-post={self.ivector_options['min_post']}",
                    self.dubm_path,
                    self.feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            extract_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extract"),
                    f"--acoustic-weight={self.ivector_options['posterior_scale']}",
                    "--compute-objf-change=true",
                    f"--max-count={self.ivector_options['max_count']}",
                    self.ie_path,
                    self.feature_string,
                    "ark,s,cs:-",
                    f"ark,t:{self.ivectors_path}",
                ],
                stderr=subprocess.PIPE,
                encoding="utf8",
                stdin=gmm_global_get_post_proc.stdout,
                env=os.environ,
            )
            for line in extract_proc.stderr:
                log_file.write(line)
                m = self.progress_pattern.match(line.strip())
                if m:
                    yield m.group("speaker")
