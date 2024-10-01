"""Class definitions for adapting acoustic models"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import List

from _kalpy.gmm import AccumAmDiagGmm, IsmoothStatsAmDiagGmmFromModel
from _kalpy.matrix import DoubleVector
from kalpy.gmm.utils import read_gmm_model, write_gmm_model
from kalpy.utils import kalpy_logger

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import AdapterMixin, MetaDict
from montreal_forced_aligner.alignment.multiprocessing import AccStatsArguments, AccStatsFunction
from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
from montreal_forced_aligner.data import WorkflowType
from montreal_forced_aligner.db import CorpusWorkflow
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.models import AcousticModel
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function

__all__ = ["AdaptingAligner"]

logger = logging.getLogger("mfa")


class AdaptingAligner(PretrainedAligner, AdapterMixin):
    """
    Adapt an acoustic model to a new dataset

    Parameters
    ----------
    mapping_tau: int
        Tau to use in mapping stats between new domain data and pretrained model

    See Also
    --------
    :class:`~montreal_forced_aligner.alignment.pretrained.PretrainedAligner`
        For dictionary, corpus, and alignment parameters
    :class:`~montreal_forced_aligner.abc.AdapterMixin`
        For adapting parameters

    Attributes
    ----------
    initialized: bool
        Flag for whether initialization is complete
    adaptation_done: bool
        Flag for whether adaptation is complete
    """

    def __init__(self, mapping_tau: int = 20, **kwargs):
        self.initialized = False
        self.adaptation_done = False
        super().__init__(**kwargs)
        self.mapping_tau = mapping_tau

    def map_acc_stats_arguments(self, alignment=False) -> List[AccStatsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsArguments`]
            Arguments for processing
        """
        if alignment:
            model_path = self.alignment_model_path
        else:
            model_path = self.model_path
        arguments = []
        for j in self.jobs:
            arguments.append(
                AccStatsArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"map_acc_stats.{j.id}.log"),
                    self.working_directory,
                    model_path,
                )
            )
        return arguments

    def acc_stats(self, alignment: bool = False) -> None:
        """
        Accumulate stats for the mapped model

        Parameters
        ----------
        alignment: bool
            Flag for whether to accumulate stats for the mapped alignment model
        """
        arguments = self.map_acc_stats_arguments(alignment)
        if alignment:
            initial_mdl_path = self.working_directory.joinpath("unadapted.alimdl")
            final_mdl_path = self.working_directory.joinpath("final.alimdl")
        else:
            initial_mdl_path = self.working_directory.joinpath("unadapted.mdl")
            final_mdl_path = self.working_directory.joinpath("final.mdl")
        logger.info("Accumulating statistics...")
        transition_model, acoustic_model = read_gmm_model(initial_mdl_path)
        transition_accs = DoubleVector()
        gmm_accs = AccumAmDiagGmm()
        transition_model.InitStats(transition_accs)
        gmm_accs.init(acoustic_model)
        for result in run_kaldi_function(
            AccStatsFunction, arguments, total_count=self.num_current_utterances
        ):
            if isinstance(result, tuple):
                job_transition_accs, job_gmm_accs = result
                transition_accs.AddVec(1.0, job_transition_accs)
                gmm_accs.Add(1.0, job_gmm_accs)
        log_path = self.working_log_directory.joinpath("map_model_est.log")
        with kalpy_logger("kalpy.train", log_path):
            IsmoothStatsAmDiagGmmFromModel(acoustic_model, self.mapping_tau, gmm_accs)
            objf_impr, count = transition_model.mle_update(transition_accs)
            logger.debug(
                f"Transition model update: Overall {objf_impr / count} "
                f"log-like improvement per frame over {count} frames."
            )
            objf_impr, count = acoustic_model.mle_update(
                gmm_accs, update_flags_str="m", remove_low_count_gaussians=False
            )
            logger.debug(
                f"GMM update: Overall {objf_impr / count} "
                f"objective function improvement per frame over {count} frames."
            )
            tot_like = gmm_accs.TotLogLike()
            tot_t = gmm_accs.TotCount()
            logger.debug(
                f"Average Likelihood per frame = {tot_like / tot_t} " f"over {tot_t} frames."
            )
            write_gmm_model(str(final_mdl_path), transition_model, acoustic_model)

    @property
    def align_directory(self) -> Path:
        """Align directory"""
        return self.output_directory.joinpath("adapted_align")

    @property
    def working_log_directory(self) -> Path:
        """Current log directory"""
        return self.working_directory.joinpath("log")

    @property
    def model_path(self) -> Path:
        """Current acoustic model path"""
        if self.current_workflow.workflow_type == WorkflowType.acoustic_model_adaptation:
            return self.working_directory.joinpath("unadapted.mdl")
        return self.working_directory.joinpath("final.mdl")

    @property
    def alignment_model_path(self) -> Path:
        """Current acoustic model path"""
        if self.current_workflow.workflow_type == WorkflowType.acoustic_model_adaptation:
            path = self.working_directory.joinpath("unadapted.alimdl")
            if path.exists() and not getattr(self, "uses_speaker_adaptation", False):
                return path
            return self.model_path
        return super().alignment_model_path

    @property
    def next_model_path(self) -> Path:
        """Mapped acoustic model path"""
        return self.working_directory.joinpath("final.mdl")

    def train_map(self) -> None:
        """
        Trains an adapted acoustic model through mapping model states and update those with
        enough data.

        See Also
        --------
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AccStatsFunction`
            Multiprocessing helper function for each job
        :meth:`.AdaptingAligner.map_acc_stats_arguments`
            Job method for generating arguments for the helper function
        :kaldi_src:`gmm-sum-accs`
            Relevant Kaldi binary
        :kaldi_src:`gmm-ismooth-stats`
            Relevant Kaldi binary
        :kaldi_src:`gmm-est`
            Relevant Kaldi binary
        :kaldi_steps:`train_map`
            Reference Kaldi script

        """
        begin = time.time()
        log_directory = self.working_log_directory
        log_directory.mkdir(parents=True, exist_ok=True)
        self.acc_stats(alignment=False)

        if self.uses_speaker_adaptation:
            self.acc_stats(alignment=True)

        logger.debug(f"Mapping models took {time.time() - begin:.3f} seconds")

    def adapt(self) -> None:
        """Run the adaptation"""
        logger.info("Generating initial alignments...")
        self.align()
        alignment_workflow = self.current_workflow
        self.create_new_current_workflow(WorkflowType.acoustic_model_adaptation)
        for f in ["final.mdl", "final.alimdl", "tree", "lda.mat"]:
            path = alignment_workflow.working_directory.joinpath(f)
            new_path = self.working_directory.joinpath(f)
            if f.startswith("final"):
                new_path = new_path.with_stem("unadapted")
            if not path.exists():
                continue
            shutil.copyfile(
                path,
                new_path,
            )
        for j in self.jobs:
            old_paths = j.construct_path_dictionary(
                alignment_workflow.working_directory, "ali", "ark"
            )
            new_paths = j.construct_path_dictionary(self.working_directory, "ali", "ark")
            for k, v in old_paths.items():
                shutil.copyfile(v, new_paths[k])
        self.align_directory.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Adapting pretrained model...")
            self.train_map()
            self.export_model(self.working_log_directory.joinpath("acoustic_model.zip"))
            for f in ["final.mdl", "final.alimdl", "tree", "lda.mat"]:
                path = self.working_directory.joinpath(f)
                if not path.exists():
                    continue
                shutil.copyfile(
                    path,
                    self.align_directory.joinpath(f),
                )
            wf = self.current_workflow
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"done": True}
                )
                session.commit()
        except Exception as e:
            wf = self.current_workflow
            with self.session() as session:
                session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
                    {"dirty": True}
                )
                session.commit()
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    @property
    def meta(self) -> MetaDict:
        """Acoustic model metadata"""
        from datetime import datetime

        from ..utils import get_mfa_version

        data = {
            "phones": sorted(self.non_silence_phones),
            "version": get_mfa_version(),
            "architecture": self.acoustic_model.meta["architecture"],
            "train_date": str(datetime.now()),
            "features": self.feature_options,
            "phone_set_type": str(self.phone_set_type),
            "dictionaries": {
                "names": sorted(self.dictionary_base_names.values()),
                "default": self.dictionary_base_names[self._default_dictionary_id],
                "silence_word": self.silence_word,
                "use_g2p": self.use_g2p,
                "oov_word": self.oov_word,
                "bracketed_word": self.bracketed_word,
                "laughter_word": self.laughter_word,
                "clitic_marker": self.clitic_marker,
                "position_dependent_phones": self.position_dependent_phones,
            },
            "oov_phone": self.oov_phone,
            "optional_silence_phone": self.optional_silence_phone,
            "silence_probability": self.silence_probability,
            "initial_silence_probability": self.initial_silence_probability,
            "final_silence_correction": self.final_silence_correction,
            "final_non_silence_correction": self.final_non_silence_correction,
        }
        return data

    def export_model(self, output_model_path: Path) -> None:
        """
        Output an acoustic model to the specified path

        Parameters
        ----------
        output_model_path : str
            Path to save adapted acoustic model
        """
        directory = output_model_path.parent

        acoustic_model = AcousticModel.empty(
            output_model_path.stem, root_directory=self.working_log_directory
        )
        acoustic_model.add_meta_file(self)
        acoustic_model.add_model(self.working_directory)
        acoustic_model.add_model(self.phones_dir)
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        acoustic_model.dump(output_model_path)
