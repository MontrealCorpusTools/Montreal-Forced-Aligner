"""Class definitions for Monophone trainer"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import subprocess
import typing
from queue import Empty

import tqdm
from sqlalchemy.orm import Session, joinedload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.acoustic_modeling.base import AcousticModelTrainingMixin
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import CorpusWorkflow, Job
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import KaldiProcessWorker, Stopped, thirdparty_binary

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

__all__ = ["MonophoneTrainer", "MonoAlignEqualFunction", "MonoAlignEqualArguments"]

logger = logging.getLogger("mfa")


class MonoAlignEqualArguments(MfaArguments):
    """Arguments for :func:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualFunction`"""

    model_path: str
    feature_options: MetaDict


class MonoAlignEqualFunction(KaldiFunction):
    """
    Multiprocessing function for initializing monophone alignments

    See Also
    --------
    :meth:`.MonophoneTrainer.mono_align_equal`
        Main function that calls this function in parallel
    :meth:`.MonophoneTrainer.mono_align_equal_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`align-equal-compiled`
        Relevant Kaldi binary
    :kaldi_src:`gmm-acc-stats-ali`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.* Done (?P<utterances>\d+) files, (?P<errors>\d+) with errors.$"
    )

    def __init__(self, args: MonoAlignEqualArguments):
        super().__init__(args)
        self.model_path = args.model_path
        self.feature_options = args.feature_options

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""

        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine) as session:
            job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            workflow: CorpusWorkflow = (
                session.query(CorpusWorkflow)
                .filter(CorpusWorkflow.current == True)  # noqa
                .first()
            )
            for dict_id in job.dictionary_ids:
                feature_string = job.construct_feature_proc_string(
                    workflow.working_directory,
                    dict_id,
                    self.feature_options["uses_splices"],
                    self.feature_options["splice_left_context"],
                    self.feature_options["splice_right_context"],
                    self.feature_options["uses_speaker_adaptation"],
                )

                fst_ark_path = job.construct_path(
                    workflow.working_directory, "fsts", "ark", dict_id
                )
                ali_path = job.construct_path(workflow.working_directory, "ali", "ark", dict_id)
                acc_path = job.construct_path(workflow.working_directory, "0", "acc", dict_id)
                align_proc = subprocess.Popen(
                    [
                        thirdparty_binary("align-equal-compiled"),
                        f"ark:{fst_ark_path}",
                        feature_string,
                        f"ark:{ali_path}",
                    ],
                    stderr=log_file,
                    env=os.environ,
                )
                align_proc.communicate()
                acc_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-acc-stats-ali"),
                        "--binary=true",
                        self.model_path,
                        feature_string,
                        f"ark:{ali_path}",
                        acc_path,
                    ],
                    stdin=align_proc.stdout,
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    env=os.environ,
                )
                for line in acc_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("utterances")), int(m.group("errors"))
                self.check_call(acc_proc)


class MonophoneTrainer(AcousticModelTrainingMixin):
    """
    Configuration class for monophone training

    Attributes
    ----------
    subset : int
        Number of utterances to use, defaults to 2000
    initial_gaussians : int
        Number of gaussians to begin training, defaults to 135
    max_gaussians : int
        Total number of gaussians, defaults to 1000
    power : float
        Exponent for number of gaussians according to occurrence counts, defaults to 0.25

    See Also
    --------
    :class:`~montreal_forced_aligner.acoustic_modeling.base.AcousticModelTrainingMixin`
        For acoustic model training parsing parameters
    """

    def __init__(
        self,
        subset: int = 2000,
        initial_gaussians: int = 135,
        initial_beam: int = 6,
        max_gaussians: int = 1000,
        power: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subset = subset
        self.initial_gaussians = initial_gaussians
        self.initial_beam = initial_beam
        self.max_gaussians = max_gaussians
        self.power = power
        self.last_gaussian_increase_iteration = 0

    def mono_align_equal_arguments(self) -> typing.List[MonoAlignEqualArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualArguments`]
            Arguments for processing
        """
        return [
            MonoAlignEqualArguments(
                j.id,
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"mono_align_equal.{j.id}.log"),
                self.model_path,
                self.feature_options,
            )
            for j in self.jobs
        ]

    def compute_calculated_properties(self) -> None:
        """Generate realignment iterations and initial gaussians based on configuration"""
        self.final_gaussian_iteration = self.num_iterations - 10
        self.realignment_iterations = [0]
        for i in range(1, self.num_iterations):
            if i <= int(self.num_iterations / 4):
                self.realignment_iterations.append(i)
            elif i <= int(self.num_iterations * 2 / 4):
                if i - self.realignment_iterations[-1] > 1:
                    self.realignment_iterations.append(i)
            else:
                if i - self.realignment_iterations[-1] > 2:
                    self.realignment_iterations.append(i)

    @property
    def train_type(self) -> str:
        """Training identifier"""
        return "mono"

    @property
    def phone_type(self) -> str:
        """Phone type"""
        return "monophone"

    @property
    def align_options(self) -> MetaDict:
        """Alignment parameters"""
        options = super().align_options
        if self.iteration == 1:
            options["beam"] = self.initial_beam
        return options

    def mono_align_equal(self) -> None:
        """
        Multiprocessing function that creates equal alignments for base monophone training.

        See Also
        --------
        :func:`~montreal_forced_aligner.acoustic_modeling.monophone.MonoAlignEqualFunction`
            Multiprocessing helper function for each job
        :meth:`.MonophoneTrainer.mono_align_equal_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_steps:`train_mono`
            Reference Kaldi script
        """

        logger.info("Generating initial alignments...")
        arguments = self.mono_align_equal_arguments()
        with tqdm.tqdm(total=self.num_current_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            if GLOBAL_CONFIG.use_mp:
                error_dict = {}
                return_queue = mp.Queue()
                stopped = Stopped()
                procs = []
                for i, args in enumerate(arguments):
                    function = MonoAlignEqualFunction(args)
                    p = KaldiProcessWorker(i, return_queue, function, stopped)
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
                    num_utterances, errors = result
                    pbar.update(num_utterances + errors)
                for p in procs:
                    p.join()
                if error_dict:
                    error_logs = []
                    for e in error_dict.values():
                        if isinstance(e, KaldiProcessingError):
                            error_logs.extend(e.error_logs)
                        else:
                            raise e
                    if error_logs:
                        e = KaldiProcessingError(e.error_logs)
                        e.update_log_file()
                        raise e
            else:
                for args in arguments:
                    function = MonoAlignEqualFunction(args)
                    for num_utterances, errors in function.run():
                        pbar.update(num_utterances + errors)

        log_path = os.path.join(self.working_log_directory, "update.0.log")
        with mfa_open(log_path, "w") as log_file:
            acc_files = []
            for j in self.jobs:
                for dict_id in j.dictionary_ids:
                    acc_files.append(j.construct_path(self.working_directory, "0", "acc", dict_id))
            sum_proc = subprocess.Popen(
                [thirdparty_binary("gmm-sum-accs"), "-"] + acc_files,
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            est_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est"),
                    "--min-gaussian-occupancy=3",
                    f"--mix-up={self.current_gaussians}",
                    f"--power={self.power}",
                    self.model_path,
                    "-",
                    self.next_model_path,
                ],
                stderr=log_file,
                stdin=sum_proc.stdout,
                env=os.environ,
            )
            est_proc.communicate()
        if est_proc.returncode != 0:
            raise KaldiProcessingError([log_path])
        if not GLOBAL_CONFIG.debug:
            for f in acc_files:
                os.remove(f)

    def _trainer_initialization(self) -> None:
        """Monophone training initialization"""
        if self.initialized:
            return
        self.iteration = 0
        tree_path = os.path.join(self.working_directory, "tree")
        feat_dim = self.worker.get_feat_dim()

        feature_string = self.jobs[0].construct_feature_proc_string(
            self.working_directory,
            self.jobs[0].dictionary_ids[0],
            self.feature_options["uses_splices"],
            self.feature_options["splice_left_context"],
            self.feature_options["splice_right_context"],
            self.feature_options["uses_speaker_adaptation"],
        )
        shared_phones_path = os.path.join(self.worker.phones_dir, "sets.int")
        init_log_path = os.path.join(self.working_log_directory, "init.log")
        temp_feats_path = os.path.join(self.working_directory, "temp_feats")
        with mfa_open(init_log_path, "w") as log_file:
            subprocess.call(
                [
                    thirdparty_binary("subset-feats"),
                    "--n=10",
                    feature_string,
                    f"ark:{temp_feats_path}",
                ],
                stderr=log_file,
            )
            subprocess.call(
                [
                    thirdparty_binary("gmm-init-mono"),
                    f"--shared-phones={shared_phones_path}",
                    f"--train-feats=ark:{temp_feats_path}",
                    os.path.join(self.worker.topo_path),
                    str(feat_dim),
                    self.model_path,
                    tree_path,
                ],
                stderr=log_file,
            )
            proc = subprocess.Popen(
                [thirdparty_binary("gmm-info"), "--print-args=false", self.model_path],
                stderr=log_file,
                stdout=subprocess.PIPE,
                encoding="utf8",
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise KaldiProcessingError([init_log_path])
            matches = re.search(r"gaussians (\d+)", stdout)
            num_gauss = int(matches.groups()[0])
        os.remove(temp_feats_path)
        self.initial_gaussians = num_gauss
        self.current_gaussians = num_gauss
        if os.path.exists(self.model_path):
            os.remove(
                init_log_path
            )  # Has some errors related to subsetting that trigger larger failures
        self.compile_train_graphs()
        self.mono_align_equal()
