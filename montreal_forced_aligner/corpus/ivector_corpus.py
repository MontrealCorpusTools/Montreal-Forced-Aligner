"""Classes for corpora that use ivectors as features"""
import logging
import os
import pickle
import subprocess
import time
import typing
from typing import List

import numpy as np
import sqlalchemy
import tqdm

from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusMixin
from montreal_forced_aligner.corpus.features import (
    ExportIvectorsFunction,
    ExtractIvectorsArguments,
    ExtractIvectorsFunction,
    IvectorConfigMixin,
    PldaModel,
)
from montreal_forced_aligner.data import MfaArguments
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
        For dictionary and corpus parsing parameters
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
    def utterance_ivector_path(self):
        return os.path.join(self.corpus_output_directory, "ivectors.scp")

    @property
    def adapted_plda_path(self):
        return os.path.join(self.working_directory, "plda_adapted")

    @property
    def plda_path(self):
        return os.path.join(self.working_directory, "plda")

    def adapt_plda(self):
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

    def export_current_ivectors(self):
        logger.info("Exporting ivectors...")

        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            arguments = [
                MfaArguments(
                    j.id,
                    self.db_string,
                    j.construct_path(self.working_log_directory, "export_ivectors", "log"),
                )
                for j in self.jobs
            ]
            for _ in run_kaldi_function(ExportIvectorsFunction, arguments, pbar.update):
                pass
        self._write_ivectors()

    def compute_speaker_ivectors(self):
        if not self.has_ivectors():
            self.extract_ivectors()
        speaker_ivector_ark_path = os.path.join(self.working_directory, "speaker_ivectors.ark")
        self._write_spk2utt()
        spk2utt_path = os.path.join(self.corpus_output_directory, "spk2utt.scp")

        log_path = os.path.join(self.working_log_directory, "speaker_ivectors.log")
        num_utts_path = os.path.join(self.working_directory, "num_utts.ark")
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
        if not self.has_ivectors():
            self.extract_ivectors()
        self._write_spk2utt()
        spk2utt_path = os.path.join(self.corpus_output_directory, "spk2utt.scp")

        plda_path = os.path.join(self.working_directory, "plda")
        log_path = os.path.join(self.working_log_directory, "plda.log")
        logger.info("Computing PLDA...")
        self.stopped.reset()
        if self.stopped.stop_check():
            logger.debug("PLDA computation stopped early.")
            return
        with tqdm.tqdm(
            total=self.num_utterances + self.num_speakers, disable=GLOBAL_CONFIG.quiet
        ) as pbar, mfa_open(log_path, "w") as log_file:

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
                pbar.update(1)

            plda_compute_proc.stdin.close()
            plda_compute_proc.wait()
            if self.stopped.stop_check():
                logger.debug("PLDA computation stopped early.")
                return
        assert os.path.exists(plda_path)

    def _write_ivectors(self):
        lines = []
        with self.session() as session:
            ignored_utterances = {
                x
                for x, in session.query(Utterance.kaldi_id).filter(
                    Utterance.ignored == True  # noqa
                )
            }
        for j in self.jobs:
            with mfa_open(j.construct_path(self.split_directory, "ivectors", "scp")) as inf:
                for line in inf:
                    if line.split(maxsplit=1)[0] in ignored_utterances:
                        continue
                    lines.append(line)
        with mfa_open(self.utterance_ivector_path, "w") as outf:
            for line in sorted(lines, key=lambda x: x.split(maxsplit=1)[0]):
                outf.write(line)

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
        self._write_ivectors()
        self.collect_utterance_ivectors()
        logger.debug(f"Ivector extraction took {time.time() - begin}")

    def collect_utterance_ivectors(self):
        plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
        if os.path.exists(plda_transform_path):
            with open(plda_transform_path, "rb") as f:
                self.plda = pickle.load(f)
        if self.has_ivectors(speaker=False) and os.path.exists(plda_transform_path):
            return
        plda_path = (
            self.adapted_plda_path if os.path.exists(self.adapted_plda_path) else self.plda_path
        )
        if not os.path.exists(plda_path):
            logger.info("Missing plda, skipping speaker ivector collection")
            return
        logger.info("Collecting ivectors...")
        self.adapt_plda()
        plda_path = (
            self.adapted_plda_path if os.path.exists(self.adapted_plda_path) else self.plda_path
        )
        self.plda = PldaModel.load(plda_path)
        ivector_arks = {}
        for j in self.jobs:
            ivector_scp_path = j.construct_path(self.split_directory, "ivectors", "scp")
            with open(ivector_scp_path, "r") as f:
                for line in f:
                    scp_line = line.strip().split(maxsplit=1)
                    ivector_arks[int(scp_line[0].split("-")[-1])] = scp_line[-1]
        with self.session() as session:
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS utterance_ivector_index"))
            session.execute(sqlalchemy.text("ALTER TABLE utterance DISABLE TRIGGER all"))
            session.commit()
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-vector"),
                    f"scp:{self.utterance_ivector_path}",
                    "ark,t:-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=os.environ,
            )
            ivectors = []
            utt_ids = []
            for utt_id, ivector in read_feats(copy_proc):
                utt_ids.append(utt_id)
                ivectors.append(ivector)
            ivectors = np.array(ivectors)
            ivectors = self.plda.process_ivectors(ivectors)
            update_mapping = []
            for i, utt_id in enumerate(utt_ids):
                update_mapping.append(
                    {"id": utt_id, "ivector": ivectors[i, :], "ivector_ark": ivector_arks[utt_id]}
                )
            bulk_update(session, Utterance, update_mapping)
            session.query(Corpus).update({Corpus.ivectors_calculated: True})
            session.execute(
                sqlalchemy.text(
                    "CREATE INDEX utterance_ivector_index ON utterance "
                    "USING ivfflat (ivector vector_cosine_ops) "
                    "WITH (lists = 1000)"
                )
            )
            session.execute(sqlalchemy.text("ALTER TABLE utterance ENABLE TRIGGER all"))
            session.commit()
            with open(plda_transform_path, "wb") as f:
                pickle.dump(self.plda, f)

    def collect_speaker_ivectors(self):
        if self.has_ivectors(speaker=True):
            return
        if self.plda is None:
            self.collect_utterance_ivectors()
        logger.info("Collecting speaker ivectors...")
        speaker_ivector_ark_path = os.path.join(self.working_directory, "speaker_ivectors.ark")
        if not os.path.exists(speaker_ivector_ark_path):
            self.compute_speaker_ivectors()
        with self.session() as session:
            session.execute(sqlalchemy.text("ALTER TABLE speaker DISABLE TRIGGER all"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS speaker_ivector_index"))
            session.commit()
            copy_proc = subprocess.Popen(
                [thirdparty_binary("copy-vector"), f"ark:{speaker_ivector_ark_path}", "ark,t:-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=os.environ,
            )
            ivectors = []
            speaker_ids = []
            for speaker_id, ivector in read_feats(copy_proc, raw_id=True):
                speaker_ids.append(int(speaker_id))
                ivectors.append(ivector)
            ivectors = np.array(ivectors)
            ivectors = self.plda.process_ivectors(ivectors)
            update_mapping = []
            for i, speaker_id in enumerate(speaker_ids):
                update_mapping.append({"id": speaker_id, "ivector": ivectors[i, :]})
            bulk_update(session, Speaker, update_mapping)
            session.query(Corpus).update({Corpus.ivectors_calculated: True})
            session.execute(
                sqlalchemy.text(
                    "CREATE INDEX speaker_ivector_index ON speaker "
                    "USING ivfflat (ivector vector_cosine_ops) "
                    "WITH (lists = 1000)"
                )
            )
            session.execute(sqlalchemy.text("ALTER TABLE speaker ENABLE TRIGGER all"))
            session.commit()
