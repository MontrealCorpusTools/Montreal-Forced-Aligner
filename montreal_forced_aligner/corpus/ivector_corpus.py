"""Classes for corpora that use ivectors as features"""
import logging
import os
import time
import typing
from pathlib import Path
from typing import List

import numpy as np
import sqlalchemy
from _kalpy.ivector import (
    Plda,
    PldaEstimationConfig,
    PldaEstimator,
    PldaStats,
    PldaUnsupervisedAdaptor,
    PldaUnsupervisedAdaptorConfig,
    ivector_normalize_length,
    ivector_subtract_mean,
)
from _kalpy.matrix import DoubleMatrix, FloatVector
from _kalpy.util import BaseFloatVectorWriter, SequentialBaseFloatVectorReader
from kalpy.data import KaldiMapping
from kalpy.ivector.data import IvectorArchive
from kalpy.utils import (
    generate_read_specifier,
    generate_write_specifier,
    kalpy_logger,
    read_kaldi_object,
    write_kaldi_object,
)
from tqdm.rich import tqdm

from montreal_forced_aligner.config import GLOBAL_CONFIG, IVECTOR_DIMENSION
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.features import (
    ExtractIvectorsArguments,
    ExtractIvectorsFunction,
    IvectorConfigMixin,
)
from montreal_forced_aligner.db import Corpus, Speaker, Utterance, bulk_update
from montreal_forced_aligner.exceptions import IvectorTrainingError
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import run_kaldi_function

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
        self.plda: typing.Optional[Plda] = None

    @property
    def ie_path(self) -> Path:
        """Ivector extractor ie path"""
        return self.working_directory.joinpath("final.ie")

    @property
    def dubm_path(self) -> Path:
        """DUBM model path"""
        return self.working_directory.joinpath("final.dubm")

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
                    getattr(self, "session", ""),
                    self.working_log_directory.joinpath(f"extract_ivectors.{j.id}.log"),
                    self.ivector_options,
                    self.ie_path,
                    j.construct_path(self.split_directory, "ivectors", "scp"),
                    self.dubm_path,
                )
            )

        return arguments

    @property
    def utterance_ivector_path(self) -> Path:
        """Path to scp file containing all ivectors"""
        return self.corpus_output_directory.joinpath("ivectors.scp")

    @property
    def adapted_plda_path(self) -> Path:
        """Path to adapted PLDA model"""
        return self.working_directory.joinpath("plda_adapted")

    @property
    def plda_path(self) -> Path:
        """Path to trained PLDA model"""
        if self.adapted_plda_path.exists():
            return self.adapted_plda_path
        return self.working_directory.joinpath("plda")

    def adapt_plda(self) -> None:
        """Adapted a trained PLDA model with new ivectors"""
        if not os.path.exists(self.utterance_ivector_path):
            self.extract_ivectors()

        config = PldaUnsupervisedAdaptorConfig()
        plda = read_kaldi_object(Plda, self.plda_path)
        adaptor = PldaUnsupervisedAdaptor()
        reader = SequentialBaseFloatVectorReader(
            generate_read_specifier(self.utterance_ivector_path)
        )
        while not reader.Done():
            adaptor.AddStats(1.0, reader.Value())
            reader.Next()
        reader.Close()
        adaptor.UpdatePlda(config, plda)
        write_kaldi_object(plda, self.adapted_plda_path)

    def compute_speaker_ivectors(self) -> None:
        """Calculated and save per-speaker ivectors as the mean over their utterances"""
        if not self.has_ivectors():
            self.extract_ivectors()
        speaker_ivector_ark_path = os.path.join(
            self.working_directory, "current_speaker_ivectors.ark"
        )
        self._write_spk2utt()
        spk2utt_path = os.path.join(self.corpus_output_directory, "spk2utt.scp")

        log_path = self.working_log_directory.joinpath("speaker_ivectors.log")
        num_utts_path = self.working_directory.joinpath("current_num_utts.ark")
        logger.info("Computing speaker ivectors...")
        if self.stopped.is_set():
            logger.debug("Speaker ivector computation stopped early.")
            return
        with (
            self.session() as session,
            tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar,
            mfa_open(num_utts_path, "w") as num_utts_archive,
            kalpy_logger("kalpy.ivector", log_path),
        ):
            speaker_mean_archive = BaseFloatVectorWriter(
                generate_write_specifier(speaker_ivector_ark_path)
            )
            spk2utt = KaldiMapping(list_mapping=True)
            spk2utt.load(spk2utt_path)
            query = session.query(Speaker.id, Utterance.ivector_ark).join(Utterance.speaker)
            current_speaker = None
            utt_count = 0
            speaker_mean = FloatVector(IVECTOR_DIMENSION)
            for speaker_id, ivector_path in query:
                if current_speaker is None:
                    current_speaker = speaker_id
                if speaker_id != current_speaker:
                    speaker_mean.Scale(1.0 / utt_count)

                    speaker_mean_archive.Write(str(speaker_id), speaker_mean)
                    num_utts_archive.write(f"{speaker_id} {utt_count}\n")
                    speaker_mean = FloatVector(IVECTOR_DIMENSION)
                    utt_count = 0
                    current_speaker = speaker_id
                    pbar.update(1)
                ivector = read_kaldi_object(FloatVector, ivector_path)
                # ivector-normalize-length
                ivector_normalize_length(ivector)
                utt_count += 1
                speaker_mean.AddVec(1.0, ivector)
            speaker_mean_archive.Close()

        self.collect_speaker_ivectors()

    def compute_plda(self) -> None:
        """Train a PLDA model"""
        if not os.path.exists(self.utterance_ivector_path):
            if not self.has_any_ivectors():
                raise Exception(
                    "Must have either ivectors or xvectors calculated to compute PLDA."
                )

        plda_path = self.working_directory.joinpath("plda")
        log_path = self.working_log_directory.joinpath("plda.log")
        logger.info("Computing PLDA...")
        self.stopped.reset()
        if self.stopped.is_set():
            logger.debug("PLDA computation stopped early.")
            return

        with (
            tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar,
            kalpy_logger("kalpy.ivector", log_path) as ivector_logger,
        ):
            plda_config = PldaEstimationConfig()
            plda_stats = PldaStats()
            num_utt_done = 0
            num_spk_done = 0
            num_spk_err = 0
            num_utt_err = 0
            for j in self.jobs:
                ivector_scp_path = j.construct_path(self.split_directory, "ivectors", "scp")
                spk2utt_path = j.construct_path(self.split_directory, "spk2utt", "scp")
                spk2utt = KaldiMapping(list_mapping=True)
                spk2utt.load(spk2utt_path)
                ivector_archive = IvectorArchive(ivector_scp_path)
                for utt_list in spk2utt.values():
                    ivector_mat = DoubleMatrix(len(utt_list), IVECTOR_DIMENSION)
                    for i, utt_id in enumerate(utt_list):
                        ivector = ivector_archive[utt_id]

                        # ivector-normalize-length
                        ivector_normalize_length(ivector)
                        ivector_mat.Row(i).CopyFromVec(ivector)
                        num_utt_done += 1
                    pbar.update(1)
                    plda_stats.AddSamples(1.0, ivector_mat)
                    num_spk_done += 1
                if num_spk_done == 0:
                    raise IvectorTrainingError("No stats accumulated, unable to estimate PLDA.")
                if num_utt_done <= plda_stats.Dim():
                    raise IvectorTrainingError(
                        "Number of training iVectors is not greater than their "
                        "dimension, unable to estimate PLDA."
                    )
                if num_spk_done == num_utt_done:
                    raise IvectorTrainingError(
                        "No speakers with multiple utterances, " "unable to estimate PLDA."
                    )
                ivector_logger.info(
                    f"Accumulated stats from {num_spk_done} speakers "
                    f"({num_spk_err}  with no utterances), consisting of {num_utt_done} utterances "
                    f"({num_utt_err} absent from input)."
                )

                plda_stats.Sort()
                plda_estimator = PldaEstimator(plda_stats)
                plda = plda_estimator.estimate(plda_config)
                write_kaldi_object(plda, plda_path)

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
        with tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            for _ in run_kaldi_function(ExtractIvectorsFunction, arguments, pbar.update):
                pass
        self.collect_utterance_ivectors()
        logger.debug(f"Ivector extraction took {time.time() - begin:.3f} seconds")

    def transform_ivectors(self):
        plda_transform_path = self.working_directory.joinpath("plda.pkl")
        if self.has_ivectors() and os.path.exists(plda_transform_path):
            return
        if not self.plda_path.exists():
            logger.info("Missing plda, skipping speaker ivector transformation")
            return
        self.plda = read_kaldi_object(Plda, self.plda_path)
        self.adapt_plda()
        self.plda = read_kaldi_object(Plda, self.plda_path)
        with self.session() as session:
            query = session.query(Utterance.id, Utterance.ivector).filter(
                Utterance.ivector != None  # noqa
            )
            ivectors = np.empty((query.count(), IVECTOR_DIMENSION))
            update_mapping = []
            for i, (u_id, ivector) in enumerate(query):
                kaldi_ivector = FloatVector()
                kaldi_ivector.from_numpy(ivector)
                update_mapping.append(
                    {
                        "id": u_id,
                        "plda_vector": self.plda.transform_ivector(kaldi_ivector, 1).numpy(),
                    }
                )
                ivectors[i, :] = ivector

            bulk_update(session, Utterance, update_mapping)
            session.commit()

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

    def collect_utterance_ivectors(self) -> None:
        """Collect trained per-utterance ivectors"""
        logger.info("Collecting ivectors...")
        with self.session() as session, tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            update_mapping = {}
            for j in self.jobs:
                ivector_scp_path = j.construct_path(self.split_directory, "ivectors", "scp")

                ivector_sum = FloatVector(IVECTOR_DIMENSION)
                count = 0
                with mfa_open(ivector_scp_path) as f:
                    ivectors = {}
                    for line in f:
                        line = line.strip()
                        utt_id, ivector_ark_path = line.split(maxsplit=1)
                        utt_id = int(utt_id.split("-")[-1])
                        ivector = read_kaldi_object(FloatVector, ivector_ark_path)
                        ivectors[utt_id] = ivector
                        update_mapping[utt_id] = {"id": utt_id, "ivector_ark": ivector_ark_path}
                        ivector_normalize_length(ivector)
                        count += 1
                        ivector_sum.AddVec(1.0, ivector)
                ivector_sum.Scale(1.0 / count)
                for utt_id, ivector in ivectors.items():
                    ivector.AddVec(-1.0, ivector_sum)

                    ivector_normalize_length(ivector)
                    update_mapping[utt_id]["ivector"] = ivector.numpy()
                    pbar.update(1)
            bulk_update(session, Utterance, list(update_mapping.values()))
            session.flush()
            session.execute(
                sqlalchemy.text(
                    "CREATE INDEX IF NOT EXISTS utterance_ivector_index ON utterance USING ivfflat (ivector vector_cosine_ops);"
                )
            )
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
        num_utts_path = self.working_directory.joinpath("current_num_utts.ark")
        if not os.path.exists(speaker_ivector_ark_path):
            self.compute_speaker_ivectors()
        with self.session() as session, tqdm(
            total=self.num_speakers, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            ivector_archive = IvectorArchive(
                speaker_ivector_ark_path, num_utterances_file_name=num_utts_path
            )
            ivectors = []
            speaker_ids = []
            num_utts = []
            for speaker_id, ivector, utts in ivector_archive:
                speaker_ids.append(speaker_id)
                num_utts.append(utts)
                ivector_normalize_length(ivector)
                ivectors.append(FloatVector(ivector))
            ivector_subtract_mean(ivectors)
            update_mapping = []
            for i in range(len(speaker_ids)):
                ivector = ivectors[i]
                ivector_normalize_length(ivector)

                update_mapping.append(
                    {
                        "id": speaker_ids[i],
                        "ivector": ivector.numpy(),
                        "plda_vector": self.plda.transform_ivector(ivector, num_utts[i]).numpy(),
                    }
                )
                pbar.update(1)
            bulk_update(session, Speaker, update_mapping)
            session.flush()
            session.execute(
                sqlalchemy.text(
                    "CREATE INDEX IF NOT EXISTS speaker_ivector_index ON speaker USING ivfflat (ivector vector_cosine_ops);"
                )
            )
            session.execute(
                sqlalchemy.text(
                    "CREATE INDEX IF NOT EXISTS speaker_plda_vector_index ON speaker USING ivfflat (plda_vector vector_cosine_ops);"
                )
            )
            session.commit()
