import os
import subprocess
from typing import NamedTuple

from ..abc import MetaDict
from ..utils import run_mp, run_non_mp, thirdparty_binary
from .acoustic_corpus import AcousticCorpusMixin
from .features import IvectorConfigMixin


class ExtractIvectorsArguments(NamedTuple):
    """Arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.extract_ivectors_func`"""

    log_path: str
    dictionaries: list[str]
    feature_strings: dict[str, str]
    ivector_options: MetaDict
    ali_paths: dict[str, str]
    ie_path: str
    ivector_paths: dict[str, str]
    weight_paths: dict[str, str]
    model_path: str
    dubm_path: str


def extract_ivectors_func(
    log_path: str,
    dictionaries: list[str],
    feature_strings: dict[str, str],
    ivector_options: MetaDict,
    ali_paths: dict[str, str],
    ie_path: str,
    ivector_paths: dict[str, str],
    weight_paths: dict[str, str],
    model_path: str,
    dubm_path: str,
) -> None:
    """
    Multiprocessing function for extracting ivectors.

    See Also
    --------
    :func:`~montreal_forced_aligner.multiprocessing.ivector.extract_ivectors`
        Main function that calls this function in parallel
    :meth:`.Job.extract_ivectors_arguments`
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
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    ivector_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for ivector extraction
    ali_paths: Dict[str, str]
        Dictionary of alignment archives per dictionary name
    ie_path: str
        Path to the ivector extractor file
    ivector_paths: Dict[str, str]
        Dictionary of ivector archives per dictionary name
    weight_paths: Dict[str, str]
        Dictionary of weighted archives per dictionary name
    model_path: str
        Path to the acoustic model file
    dubm_path: str
        Path to the DUBM file
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            ali_path = ali_paths[dict_name]
            weight_path = weight_paths[dict_name]
            ivectors_path = ivector_paths[dict_name]
            feature_string = feature_strings[dict_name]
            use_align = os.path.exists(ali_path)
            if use_align:
                ali_to_post_proc = subprocess.Popen(
                    [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                weight_silence_proc = subprocess.Popen(
                    [
                        thirdparty_binary("weight-silence-post"),
                        str(ivector_options["silence_weight"]),
                        ivector_options["sil_phones"],
                        model_path,
                        "ark:-",
                        "ark:-",
                    ],
                    stderr=log_file,
                    stdin=ali_to_post_proc.stdout,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                post_to_weight_proc = subprocess.Popen(
                    [thirdparty_binary("post-to-weights"), "ark:-", f"ark:{weight_path}"],
                    stderr=log_file,
                    stdin=weight_silence_proc.stdout,
                    env=os.environ,
                )
                post_to_weight_proc.communicate()

            gmm_global_get_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-post"),
                    f"--n={ivector_options['num_gselect']}",
                    f"--min-post={ivector_options['min_post']}",
                    dubm_path,
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            if use_align:
                weight_proc = subprocess.Popen(
                    [
                        thirdparty_binary("weight-post"),
                        "ark:-",
                        f"ark,s,cs:{weight_path}",
                        "ark:-",
                    ],
                    stdin=gmm_global_get_post_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                extract_in = weight_proc.stdout
            else:
                extract_in = gmm_global_get_post_proc.stdout
            extract_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extract"),
                    f"--acoustic-weight={ivector_options['posterior_scale']}",
                    "--compute-objf-change=true",
                    f"--max-count={ivector_options['max_count']}",
                    ie_path,
                    feature_string,
                    "ark,s,cs:-",
                    f"ark,t:{ivectors_path}",
                ],
                stderr=log_file,
                stdin=extract_in,
                env=os.environ,
            )
            extract_proc.communicate()


class IvectorCorpus(AcousticCorpusMixin, IvectorConfigMixin):
    """
    Abstract corpus mixin for corpora that extract ivectors

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ie_path(self):
        """Ivector extractor ie path"""
        raise NotImplementedError

    @property
    def dubm_path(self):
        """DUBM model path"""
        raise

    def write_corpus_information(self) -> None:
        """
        Output information to the temporary directory for later loading
        """
        super().write_corpus_information()
        self._write_utt2spk()

    def _write_utt2spk(self):
        """Write feats scp file for Kaldi"""
        with open(
            os.path.join(self.corpus_output_directory, "utt2spk.scp"), "w", encoding="utf8"
        ) as f:
            for utterance in self.utterances.values():
                f.write(f"{utterance.name} {utterance.speaker.name}\n")

    def extract_ivectors_arguments(self) -> list[ExtractIvectorsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.multiprocessing.ivector.extract_ivectors_func`

        Returns
        -------
        list[ExtractIvectorsArguments]
            Arguments for processing
        """
        feat_strings = self.construct_feature_proc_strings()
        return [
            ExtractIvectorsArguments(
                os.path.join(self.working_log_directory, f"extract_ivectors.{j.name}.log"),
                j.current_dictionary_names,
                feat_strings[j.name],
                self.ivector_options,
                j.construct_path_dictionary(self.working_directory, "ali", "ark"),
                self.ie_path,
                j.construct_path_dictionary(self.working_directory, "ivectors", "scp"),
                j.construct_path_dictionary(self.working_directory, "weights", "ark"),
                self.model_path,
                self.dubm_path,
            )
            for j in self.jobs
        ]

    def extract_ivectors(self) -> None:
        """
        Multiprocessing function that extracts job_name-vectors.

        See Also
        --------
        :func:`~montreal_forced_aligner.multiprocessing.ivector.extract_ivectors_func`
            Multiprocessing helper function for each job
        :meth:`.Job.extract_ivectors_arguments`
            Job method for generating arguments for helper function
        :kaldi_steps_sid:`extract_ivectors`
            Reference Kaldi script
        """

        log_dir = self.working_log_directory
        os.makedirs(log_dir, exist_ok=True)

        jobs = self.extract_ivectors_arguments()
        if self.use_mp:
            run_mp(extract_ivectors_func, jobs, log_dir)
        else:
            run_non_mp(extract_ivectors_func, jobs, log_dir)
