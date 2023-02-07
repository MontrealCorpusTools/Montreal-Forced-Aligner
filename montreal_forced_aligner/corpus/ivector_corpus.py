"""Classes for corpora that use ivectors as features"""
import logging
import os
import pickle
import re
import subprocess
import time
import typing
from typing import List

import numpy as np
import tqdm

from montreal_forced_aligner.config import GLOBAL_CONFIG, IVECTOR_DIMENSION
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.features import (
    ExtractIvectorsArguments,
    ExtractIvectorsFunction,
    IvectorConfigMixin,
    PldaModel,
)
from montreal_forced_aligner.db import Corpus, Speaker, Utterance, bulk_update
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import read_feats, run_kaldi_function, thirdparty_binary

__all__ = ["IvectorCorpusMixin"]

logger = logging.getLogger("mfa")


class IvectorCorpusMixin(AcousticCorpusMixin, IvectorConfigMixin):
    """
    Abstract corpus mixin for corpora that extract ivectors

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusMixin`
        For corpus parsing parameters
    :class:`~montreal_forced_aligner.corpus.features.IvectorConfigMixin`
        For ivector extraction parameters

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plda: typing.Optional[PldaModel] = None

    @property
    def ie_path(self) -> str:
        """Ivector extractor ie path"""
        return os.path.join(self.working_directory, "final.ie")

    @property
    def dubm_path(self) -> str:
        """DUBM model path"""
        return os.path.join(self.working_directory, "final.dubm")

    def extract_ivectors_arguments(self) -> List[ExtractIvectorsArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`

        Returns
        -------
        list[ExtractIvectorsArguments]
            Arguments for processing
        """
        arguments = []
        for j in self.jobs:
            arguments.append(
                ExtractIvectorsArguments(
                    j.id,
                    getattr(self, "db_string", ""),
                    os.path.join(self.working_log_directory, f"extract_ivectors.{j.id}.log"),
                    self.ivector_options,
                    self.ie_path,
                    j.construct_path(self.split_directory, "ivectors", "scp"),
                    self.dubm_path,
                )
            )

        return arguments

    @property
    def utterance_ivector_path(self) -> str:
        """Path to scp file containing all ivectors"""
        return os.path.join(self.corpus_output_directory, "ivectors.scp")

    @property
    def adapted_plda_path(self) -> str:
        """Path to adapted PLDA model"""
        return os.path.join(self.working_directory, "plda_adapted")

    @property
    def plda_path(self) -> str:
        """Path to trained PLDA model"""
        return os.path.join(self.working_directory, "plda")

    def adapt_plda(self) -> None:
        """Adapted a trained PLDA model with new ivectors"""
        if not os.path.exists(self.utterance_ivector_path):
            self.extract_ivectors()

        log_path = os.path.join(self.working_log_directory, "adapt_plda.log")
        with mfa_open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-adapt-plda"),
                    self.plda_path,
                    f"scp:{self.utterance_ivector_path}",
                    self.adapted_plda_path,
                ],
                stderr=log_file,
            )
            proc.communicate()

    def compute_speaker_ivectors(self) -> None:
        """Calculated and save per-speaker ivectors as the mean over their utterances"""
        if not self.has_ivectors():
            self.extract_ivectors()
        speaker_ivector_ark_path = os.path.join(
            self.working_directory, "current_speaker_ivectors.ark"
        )
        self._write_spk2utt()
        spk2utt_path = os.path.join(self.corpus_output_directory, "spk2utt.scp")

        log_path = os.path.join(self.working_log_directory, "speaker_ivectors.log")
        num_utts_path = os.path.join(self.working_directory, "current_num_utts.ark")
        logger.info("Computing speaker ivectors...")
        self.stopped.reset()
        if self.stopped.stop_check():
            logger.debug("Speaker ivector computation stopped early.")
            return
        with mfa_open(log_path, "w") as log_file:

            normalize_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-normalize-length"),
                    f"scp:{self.utterance_ivector_path}",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            mean_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-mean"),
                    f"ark:{spk2utt_path}",
                    "ark:-",
                    "ark:-",
                    f"ark,t:{num_utts_path}",
                ],
                stdin=normalize_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            speaker_normalize_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-normalize-length"),
                    "ark:-",
                    f"ark:{speaker_ivector_ark_path}",
                ],
                stdin=mean_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            speaker_normalize_proc.communicate()
        self.collect_speaker_ivectors()

    def compute_plda(self) -> None:
        """Train a PLDA model"""
        if not os.path.exists(self.utterance_ivector_path):
            if not self.has_any_ivectors():
                raise Exception(
                    "Must have either ivectors or xvectors calculated to compute PLDA."
                )
        self._write_spk2utt()
        spk2utt_path = os.path.join(self.corpus_output_directory, "spk2utt.scp")

        plda_path = os.path.join(self.working_directory, "plda")
        log_path = os.path.join(self.working_log_directory, "plda.log")
        logger.info("Computing PLDA...")
        self.stopped.reset()
        if self.stopped.stop_check():
            logger.debug("PLDA computation stopped early.")
            return
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar, mfa_open(
            log_path, "w"
        ) as log_file:

            normalize_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-normalize-length"),
                    f"scp:{self.utterance_ivector_path}",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            plda_compute_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-compute-plda"),
                    f"ark:{spk2utt_path}",
                    "ark:-",
                    plda_path,
                ],
                stdin=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            for line in normalize_proc.stdout:
                if self.stopped.stop_check():
                    break
                plda_compute_proc.stdin.write(line)
                plda_compute_proc.stdin.flush()
                if re.search(rb"\d+-\d+ ", line):
                    pbar.update(1)

            plda_compute_proc.stdin.close()
            plda_compute_proc.wait()
            if self.stopped.stop_check():
                logger.debug("PLDA computation stopped early.")
                return
        assert os.path.exists(plda_path)

    def _write_ivectors(self) -> None:
        """Collect single scp file for all ivectors"""
        with self.session() as session, mfa_open(self.utterance_ivector_path, "w") as outf:
            utterances = (
                session.query(Utterance.kaldi_id, Utterance.ivector_ark)
                .join(Utterance.speaker)
                .filter(Utterance.ivector_ark != None, Speaker.name != "MFA_UNKNOWN")  # noqa,
            )
            for utt_id, ivector_ark in utterances:
                outf.write(f"{utt_id} {ivector_ark}\n")

    def extract_ivectors(self) -> None:
        """
        Multiprocessing function that extracts job_name-vectors.

        See Also
        --------
        :class:`~montreal_forced_aligner.corpus.features.ExtractIvectorsFunction`
            Multiprocessing helper function for each job
        :meth:`.IvectorCorpusMixin.extract_ivectors_arguments`
            Job method for generating arguments for helper function
        :kaldi_steps_sid:`extract_ivectors`
            Reference Kaldi script
        """
        begin = time.time()

        log_dir = self.working_log_directory
        os.makedirs(log_dir, exist_ok=True)
        with self.session() as session:
            c = session.query(Corpus).first()
            if c.ivectors_calculated:
                logger.info("Ivectors already computed, skipping!")
                return
        logger.info("Extracting ivectors...")
        arguments = self.extract_ivectors_arguments()
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(ExtractIvectorsFunction, arguments, pbar.update):
                pass
        self.collect_utterance_ivectors()
        logger.debug(f"Ivector extraction took {time.time() - begin:.3f} seconds")

    def transform_ivectors(self):
        plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
        if os.path.exists(plda_transform_path):
            with open(plda_transform_path, "rb") as f:
                self.plda = pickle.load(f)
        if self.has_ivectors() and os.path.exists(plda_transform_path):
            return
        plda_path = (
            self.adapted_plda_path if os.path.exists(self.adapted_plda_path) else self.plda_path
        )
        if not os.path.exists(plda_path):
            logger.info("Missing plda, skipping speaker ivector transformation")
            return
        self.adapt_plda()
        plda_path = (
            self.adapted_plda_path if os.path.exists(self.adapted_plda_path) else self.plda_path
        )
        self.plda = PldaModel.load(plda_path)
        with self.session() as session:
            query = session.query(Utterance.id, Utterance.ivector).filter(
                Utterance.ivector != None  # noqa
            )
            ivectors = np.empty((query.count(), IVECTOR_DIMENSION))
            utterance_ids = []
            for i, (u_id, ivector) in enumerate(query):
                utterance_ids.append(u_id)
                ivectors[i, :] = ivector
            update_mapping = []
            ivectors = self.plda.process_ivectors(ivectors)
            for i, utt_id in enumerate(utterance_ids):
                update_mapping.append({"id": utt_id, "plda_vector": ivectors[i, :]})
            bulk_update(session, Utterance, update_mapping)
            session.commit()
            with open(plda_transform_path, "wb") as f:
                pickle.dump(self.plda, f)

    def collect_utterance_ivectors(self) -> None:
        """Collect trained per-utterance ivectors"""
        logger.info("Collecting ivectors...")
        ivector_arks = {}
        for j in self.jobs:
            ivector_scp_path = j.construct_path(self.split_directory, "ivectors", "scp")
            with open(ivector_scp_path, "r") as f:
                for line in f:
                    scp_line = line.strip().split(maxsplit=1)
                    ivector_arks[int(scp_line[0].split("-")[-1])] = scp_line[-1]
        with self.session() as session, tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            update_mapping = {}
            for j in self.jobs:
                ivector_scp_path = j.construct_path(self.split_directory, "ivectors", "scp")
                norm_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ivector-normalize-length"),
                        f"scp:{ivector_scp_path}",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=os.environ,
                )
                copy_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ivector-subtract-global-mean"),
                        "ark:-",
                        "ark:-",
                    ],
                    stdin=norm_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=os.environ,
                )
                norm2_proc = subprocess.Popen(
                    [
                        thirdparty_binary("ivector-normalize-length"),
                        "ark:-",
                        "ark,t:-",
                    ],
                    stdin=copy_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=os.environ,
                )
                for utt_id, ivector in read_feats(norm2_proc):
                    update_mapping[utt_id] = {
                        "id": utt_id,
                        "ivector": ivector,
                        "ivector_ark": ivector_arks[utt_id],
                    }
                    pbar.update(1)
            bulk_update(session, Utterance, list(update_mapping.values()))
            session.query(Corpus).update({Corpus.ivectors_calculated: True})
            session.commit()
        self._write_ivectors()
        self.transform_ivectors()

    def collect_speaker_ivectors(self) -> None:
        """Collect trained per-speaker ivectors"""
        if self.plda is None:
            self.collect_utterance_ivectors()
        logger.info("Collecting speaker ivectors...")
        speaker_ivector_ark_path = os.path.join(
            self.working_directory, "current_speaker_ivectors.ark"
        )
        num_utts_path = os.path.join(self.working_directory, "current_num_utts.ark")
        if not os.path.exists(speaker_ivector_ark_path):
            self.compute_speaker_ivectors()
        with self.session() as session, tqdm.tqdm(
            total=self.num_speakers, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            utterance_counts = {}
            with open(num_utts_path) as f:
                for line in f:
                    speaker, utt_count = line.strip().split()
                    utt_count = int(utt_count)
                    utterance_counts[int(speaker)] = utt_count
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-subtract-global-mean"),
                    f"ark:{speaker_ivector_ark_path}",
                    "ark,t:-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=os.environ,
            )
            ivectors = []
            speaker_ids = []
            speaker_counts = []
            update_mapping = {}
            for speaker_id, ivector in read_feats(copy_proc, raw_id=True):
                speaker_id = int(speaker_id)
                if speaker_id not in utterance_counts:
                    continue
                speaker_ids.append(speaker_id)
                ivectors.append(ivector)
                speaker_counts.append(utterance_counts[speaker_id])
                update_mapping[speaker_id] = {"id": speaker_id, "ivector": ivector}
                pbar.update(1)
            ivectors = np.array(ivectors)
            if len(ivectors.shape) < 2:
                ivectors = ivectors[np.newaxis, :]
            speaker_counts = np.array(speaker_counts)
            ivectors = self.plda.process_ivectors(ivectors, counts=speaker_counts)
            for i, speaker_id in enumerate(speaker_ids):
                update_mapping[speaker_id]["plda_vector"] = ivectors[i, :]
            bulk_update(session, Speaker, list(update_mapping.values()))
            session.commit()
