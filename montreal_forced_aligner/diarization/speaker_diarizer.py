"""
Speaker classification
======================


"""
from __future__ import annotations

import collections
import csv
import logging
import os
import pickle
import subprocess
import sys
import typing
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import librosa
import numpy as np
import sqlalchemy
import tqdm
from sklearn.metrics import pairwise_distances
from sqlalchemy.orm import joinedload, selectinload

from montreal_forced_aligner.abc import FileExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.config import GLOBAL_CONFIG
from montreal_forced_aligner.corpus.ivector_corpus import IvectorCorpusMixin
from montreal_forced_aligner.data import TextFileType, WorkflowType
from montreal_forced_aligner.db import (
    File,
    SoundFile,
    Speaker,
    SpeakerCluster,
    SpeakerOrdering,
    TextFile,
    Utterance,
    bulk_update,
)
from montreal_forced_aligner.diarization.multiprocessing import (
    PldaClassificationArguments,
    PldaClassificationFunction,
    UtteranceDistanceArguments,
    cluster_matrix,
    score_plda,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, mfa_open
from montreal_forced_aligner.models import IvectorExtractorModel
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function, thirdparty_binary

try:
    torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
    torch_logger.setLevel(logging.ERROR)
    torch_logger = logging.getLogger("speechbrain.utils.train_logger")
    torch_logger.setLevel(logging.ERROR)
    import torch
    from speechbrain.dataio.batch import PaddedBatch
    from speechbrain.pretrained import EncoderClassifier

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError) as e:
    print(e)
    FOUND_SPEECHBRAIN = False
    EncoderClassifier = None

if TYPE_CHECKING:

    from montreal_forced_aligner.abc import MetaDict
__all__ = ["SpeakerDiarizer"]

logger = logging.getLogger("mfa")


class SpeakerDiarizer(
    IvectorCorpusMixin, TopLevelMfaWorker, FileExporterMixin
):  # pragma: no cover
    """
    Class for performing speaker classification, not currently very functional, but
    is planned to be expanded in the future

    Parameters
    ----------
    ivector_extractor_path : str
        Path to ivector extractor model
    expected_num_speakers: int, optional
        Number of speakers in the corpus, if known
    cluster: bool, optional
        Flag for whether speakers should be clustered instead of classified
    """

    def __init__(
        self,
        ivector_extractor_path: str = "speechbrain",
        expected_num_speakers: int = 0,
        cluster: bool = True,
        cuda: bool = False,
        use_plda: bool = False,
        cluster_type: str = "hdbscan",
        eps: float = 0.5,
        **kwargs,
    ):
        self.ivector_extractor = None
        if ivector_extractor_path == "speechbrain":
            if not FOUND_SPEECHBRAIN:
                logger.error(
                    "Could not import speechbrain, please ensure it is installed via `pip install speechbrain`"
                )
                sys.exit(1)

        else:
            self.ivector_extractor = IvectorExtractorModel(ivector_extractor_path)
            kwargs.update(self.ivector_extractor.parameters)
        super().__init__(**kwargs)
        self.expected_num_speakers = expected_num_speakers
        self.cluster = cluster
        self.use_plda = use_plda
        self.cuda = cuda
        self.cluster_type = cluster_type
        self.eps = eps

    @classmethod
    def parse_parameters(
        cls,
        config_path: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        unknown_args: Optional[List[str]] = None,
    ) -> MetaDict:
        """
        Parse parameters for speaker classification from a config path or command-line arguments

        Parameters
        ----------
        config_path: str
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

    @property
    def ie_path(self) -> str:
        """Path for the ivector extractor model file"""
        return os.path.join(self.working_directory, "final.ie")

    @property
    def model_path(self) -> str:
        """Path for the acoustic model file"""
        return os.path.join(self.working_directory, "final.mdl")

    @property
    def dubm_path(self) -> str:
        """Path for the DUBM model"""
        return os.path.join(self.working_directory, "final.dubm")

    def setup(self) -> None:
        """
        Sets up the corpus and speaker classifier

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        super().setup()
        self.create_new_current_workflow(WorkflowType.speaker_diarization)
        wf = self.current_workflow
        if wf.done:
            logger.info("Diarization already done, skipping initialization.")
            return
        log_dir = os.path.join(self.working_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        try:
            if self.ivector_extractor is None:
                run_opts = None
                if self.cuda:
                    run_opts = {"device": "cuda"}
                self.speaker_recognition_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join(
                        GLOBAL_CONFIG.current_profile.temporary_directory,
                        "models",
                        "SpeakerRecognition",
                    ),
                    run_opts=run_opts,
                )
                self.initialize_database()
                self._load_corpus()
            else:
                self.load_corpus()
                self.compute_vad()
                self.ivector_extractor.export_model(self.working_directory)
                self.extract_ivectors()
                self.compute_speaker_ivectors()
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

    def plda_score_speakers(
        self, speaker_one: typing.Union[int, np.array], speaker_two: typing.Union[int, np.array]
    ):
        if isinstance(speaker_one, int):
            pass

    def classify_speakers(self):
        self.setup()
        if self.ivector_extractor is None:
            self.classify_speakers_speechbrain()
        else:
            self.classify_speakers_mfa()

    def plda_classification_arguments(self) -> List[PldaClassificationArguments]:
        return [
            PldaClassificationArguments(
                j.id,
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"plda_classification.{j.id}.log"),
                self.plda,
            )
            for j in self.jobs
        ]

    def classify_speakers_mfa(self):
        utt_mapping = []
        speaker_ordering_mapping = []
        arguments = self.plda_classification_arguments()
        lines = []
        with open(self.num_utts_path) as f:
            for line in f:
                lines.append(line)
        with open(self.num_utts_path, "w") as f:
            for line in sorted(lines, key=lambda x: x.split()[0]):
                f.write(line)

        input_proc = subprocess.Popen(
            [
                thirdparty_binary("copy-vector"),
                "--binary=false",
                f"ark:{self.speaker_ivector_path}",
                "ark,t:-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=os.environ,
        )
        lines = []
        for line in input_proc.stdout:
            lines.append(line)
        input_proc.wait()
        output_proc = subprocess.Popen(
            [
                thirdparty_binary("copy-vector"),
                "--binary=true",
                "ark,t:",
                f"ark:{self.speaker_ivector_path}",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=os.environ,
        )
        for line in sorted(lines, key=lambda x: x.split()[0]):
            output_proc.stdin.write(line)
            output_proc.stdin.flush()
        output_proc.stdin.close()
        output_proc.wait()

        with self.session() as session, tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:

            file_clusters = collections.defaultdict(list)
            utterance_clusters = collections.defaultdict(list)
            speaker_clusters = collections.defaultdict(collections.Counter)
            for utt_id, file_id, speaker_id, classified_speaker in run_kaldi_function(
                PldaClassificationFunction, arguments, pbar.update
            ):

                utterance_clusters[classified_speaker].append(utt_id)
                file_clusters[classified_speaker].append(file_id)
                speaker_clusters[classified_speaker][speaker_id] += 1
            utterance_mapping = []
            for cluster_id, utterance_ids in utterance_clusters.items():
                cluster_name = max(
                    speaker_clusters[cluster_id], key=lambda x: speaker_clusters[cluster_id][x]
                )
                for u_id in utterance_ids:
                    utterance_mapping.append({"id": u_id, "speaker_id": cluster_name})
                for file_id in file_clusters[cluster_id]:
                    speaker_ordering_mapping.append(
                        {"speaker_id": cluster_name, "file_id": file_id, "index": 1}
                    )

            bulk_update(session, Utterance, utt_mapping)
            if speaker_ordering_mapping:
                session.query(SpeakerOrdering).delete()
                session.execute(
                    sqlalchemy.insert(SpeakerOrdering),
                    speaker_ordering_mapping,
                )
                session.flush()
            sq = (
                session.query(
                    Speaker.id, sqlalchemy.func.count(Utterance.id).label("utterance_count")
                )
                .outerjoin(Speaker.utterances)
                .group_by(Speaker.id)
                .subquery()
            )
            sq2 = session.query(sq.c.id).filter(sq.c.utterance_count == 0)
            session.query(Speaker).filter(Speaker.id.in_(sq2)).delete(synchronize_session="fetch")
            session.commit()

    @property
    def num_utts_path(self):
        return os.path.join(self.working_directory, "num_utts.ark")

    @property
    def speaker_ivector_path(self):
        return os.path.join(self.working_directory, "speaker_ivectors.ark")

    @property
    def ivector_path(self):
        return os.path.join(self.corpus_output_directory, "ivectors.scp")

    def classify_speakers_speechbrain(self) -> None:
        """
        Classify utterance speakers

        """
        utt_mapping = []
        speaker_ordering_mapping = []
        speaker_mapping = []
        import gc

        logger.info("Classifying utterances...")
        with tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            speakers = {x.name: x.id for x in session.query(Speaker)}
            current_speaker_id = self.get_next_primary_key(Speaker)
            utterances: List[Utterance] = session.query(Utterance).options(
                joinedload(Utterance.file).joinedload(File.sound_file)
            )
            batch = []
            utt_ids = []
            file_ids = []
            speaker_ordering_set = set()
            batch_size = 25
            for u in utterances:
                y, sr = librosa.load(
                    u.file.sound_file.sound_file_path,
                    sr=16000,
                    mono=False,
                    offset=u.begin,
                    duration=u.duration,
                )
                batch.append({"id": u.id, "wav": torch.tensor(y)})
                utt_ids.append(u.id)
                file_ids.append(u.file_id)
                if len(batch) == batch_size:
                    batch = PaddedBatch(batch)
                    (
                        out_prob,
                        score,
                        index,
                        text_lab,
                    ) = self.speaker_recognition_model.classify_batch(
                        batch["wav"].data, batch["wav"].lengths
                    )
                    for i in range(batch_size):
                        utt_id = utt_ids[i]
                        speaker = text_lab[i]
                        if speaker not in speakers:
                            speakers[speaker] = current_speaker_id
                            speaker_mapping.append({"id": current_speaker_id, "name": speaker})
                            current_speaker_id += 1
                        if (speakers[speaker], file_ids[i]) not in speaker_ordering_set:
                            speaker_ordering_mapping.append(
                                {
                                    "file_id": file_ids[i],
                                    "speaker_id": speakers[speaker],
                                    "index": 10,
                                }
                            )
                            speaker_ordering_set.add((speakers[speaker], file_ids[i]))
                        utt_mapping.append({"id": utt_id, "speaker_id": speakers[speaker]})
                    del out_prob
                    del score
                    del index
                    del text_lab
                    del batch
                    gc.collect()
                    batch = []
                    utt_ids = []
                    file_ids = []
                    pbar.update(batch_size)
            if len(batch):
                batch = PaddedBatch(batch)
                out_prob, score, index, text_lab = self.speaker_recognition_model.classify_batch(
                    batch["wav"].data, batch["wav"].lengths
                )
                for i in range(len(batch)):
                    utt_id = utt_ids[i]
                    speaker = text_lab[i]
                    if speaker not in speakers:
                        speakers[speaker] = current_speaker_id
                        speaker_mapping.append({"id": current_speaker_id, "name": speaker})
                        current_speaker_id += 1
                    if (speakers[speaker], file_ids[i]) not in speaker_ordering_set:
                        speaker_ordering_mapping.append(
                            {"file_id": file_ids[i], "speaker_id": speakers[speaker], "index": 10}
                        )
                        speaker_ordering_set.add((speakers[speaker], file_ids[i]))
                    utt_mapping.append({"id": utt_id, "speaker_id": speakers[speaker]})
                del out_prob
                del score
                del index
                del text_lab
                del batch
                gc.collect()
                pbar.update(batch_size)
            if speaker_mapping:
                session.bulk_insert_mappings(
                    Speaker, speaker_mapping, return_defaults=False, render_nulls=True
                )

            bulk_update(session, Utterance, utt_mapping)
            session.execute(
                sqlalchemy.insert(SpeakerOrdering),
                speaker_ordering_mapping,
            )
            session.commit()

    def cluster_speakers_mfa(self):
        logger.info("Clustering speakers...")
        with self.session() as session:
            session.query(SpeakerCluster).delete()
            session.commit()
            speaker_counts = {}
            speakers = {}
            query = (
                session.query(Speaker, sqlalchemy.func.count(Utterance.id))
                .join(Speaker.utterances)
                .group_by(Speaker.id)
                .filter(Speaker.ivector != None)  # noqa
            )
            speaker_ids = []
            ivectors = np.empty((self.num_speakers, self.plda.dimension))
            for i, (speaker, utt_count) in enumerate(query):
                ivectors[i, :] = speaker.ivector
                speaker_ids.append(speaker.id)
                speakers[speaker.id] = speaker
                speaker_counts[speaker.id] = utt_count
            num_speakers = ivectors.shape[0]
            if self.cluster_type in [
                "spectral",
                "dbscan",
                "optics",
                "affinity",
                "hdbscan",
                "agglomerative",
            ]:
                distance_matrix = score_plda(
                    ivectors, ivectors, self.plda, normalize=True, distance=True
                )
            speaker_clusters = collections.defaultdict(list)
            with tqdm.tqdm(total=num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:

                kwargs = {}
                if self.cluster_type == "affinity":
                    logger.info("Running Affinity Propagation...")
                    to_fit = 1 - distance_matrix
                elif self.cluster_type == "agglomerative":
                    logger.info("Running Agglomerative Clustering...")
                    kwargs["n_clusters"] = self.expected_num_speakers
                    to_fit = distance_matrix
                elif self.cluster_type == "spectral":
                    logger.info("Running Spectral Clustering...")
                    kwargs["n_clusters"] = self.expected_num_speakers
                    to_fit = distance_matrix
                elif self.cluster_type == "dbscan":
                    logger.info("Running DBSCAN...")
                    to_fit = distance_matrix
                elif self.cluster_type == "hdbscan":
                    logger.info("Running HDBSCAN...")
                    to_fit = distance_matrix
                    kwargs["min_cluster_size"] = 60
                    kwargs["min_samples"] = 15
                    kwargs["cluster_selection_epsilon"] = 0.0
                elif self.cluster_type == "optics":
                    logger.info("Running OPTICS...")
                    to_fit = distance_matrix
                    kwargs["min_samples"] = 15
                    kwargs["eps"] = self.eps
                elif self.cluster_type == "kmeans":
                    logger.info("Running KMeans...")
                    to_fit = ivectors
                    kwargs["n_clusters"] = self.expected_num_speakers
                else:
                    raise NotImplementedError(
                        f"The cluster type '{self.cluster_type} is not supported."
                    )
                labels = cluster_matrix(to_fit, self.cluster_type, **kwargs)
                for i in range(num_speakers):
                    speaker_id = speaker_ids[i]
                    speaker_cluster_id = labels[i]
                    speaker_clusters[speaker_cluster_id].append(speaker_id)
                    pbar.update(1)
                cluster_mapping = []
                speaker_mapping = []
                for cluster_id, speaker_ids in speaker_clusters.items():
                    cluster_name = None
                    max_utts = None
                    for s_id in speaker_ids:
                        if cluster_name is None or speaker_counts[s_id] > max_utts:
                            max_utts = speaker_counts[s_id]
                            cluster_name = speakers[s_id].name
                        speaker_mapping.append({"id": s_id, "cluster_id": cluster_id})
                    cluster_mapping.append(
                        {
                            "id": cluster_id,
                            "name": cluster_name,
                        }
                    )
                session.bulk_insert_mappings(
                    SpeakerCluster, cluster_mapping, return_defaults=False, render_nulls=True
                )

                bulk_update(session, Speaker, speaker_mapping)
                session.commit()

    def utterance_distance_arguments(self) -> List[UtteranceDistanceArguments]:
        return [
            UtteranceDistanceArguments(
                j.id,
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"utterance_distance.{j.id}.log"),
                self.sparse_threshold,
            )
            for j in self.jobs
        ]

    def cluster_utterances_mfa(self):
        plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
        with open(plda_transform_path, "rb") as f:
            self.plda = pickle.load(f)
        with self.session() as session:
            utt2spk = {}
            query = session.query(Utterance.id, Utterance.speaker_id, Utterance.ivector).filter(
                Utterance.ivector != None  # noqa
            )
            utterance_ids = []
            ivectors = np.empty((query.count(), self.plda.dimension))
            for i, (u_id, s_id, ivector) in enumerate(query):
                utterance_ids.append(u_id)
                utt2spk[u_id] = s_id
                ivectors[i, :] = ivector
            num_utterances = ivectors.shape[0]
            kwargs = {}

            if (
                self.cluster_type
                in ["spectral", "hdbscan", "dbscan", "optics", "affinity", "agglomerative"]
                and self.use_plda
            ):
                logger.info("Generating distance matrix...")
                to_fit = score_plda(ivectors, ivectors, self.plda, normalize=True, distance=True)

                precomputed = True
            else:
                precomputed = False
                to_fit = ivectors

            if self.stopped.stop_check():
                logger.debug("Stopping clustering early.")
            logger.info("Clustering utterances...")
            with tqdm.tqdm(total=num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                if self.cluster_type == "affinity":
                    logger.info("Running Affinity Propagation...")
                elif self.cluster_type == "agglomerative":
                    logger.info("Running Agglomerative Clustering...")
                    kwargs["n_clusters"] = self.expected_num_speakers
                elif self.cluster_type == "spectral":
                    logger.info("Running Spectral Clustering...")
                    kwargs["n_clusters"] = self.expected_num_speakers
                elif self.cluster_type == "dbscan":
                    logger.info("Running DBSCAN...")
                elif self.cluster_type == "hdbscan":
                    logger.info("Running HDBSCAN...")
                    kwargs["min_cluster_size"] = 100
                    kwargs["min_samples"] = 10
                    kwargs["cluster_selection_epsilon"] = 0.0
                    kwargs["location"] = os.path.join(self.working_directory, "cache_dir")
                elif self.cluster_type == "optics":
                    logger.info("Running OPTICS...")
                    kwargs["min_samples"] = 15
                    kwargs["eps"] = self.eps
                elif self.cluster_type == "kmeans":
                    logger.info("Running KMeans...")
                    kwargs["n_clusters"] = self.expected_num_speakers
                else:
                    raise NotImplementedError(
                        f"The cluster type '{self.cluster_type} is not supported."
                    )
                labels = cluster_matrix(
                    to_fit, self.cluster_type, precomputed=precomputed, **kwargs
                )
                utterance_clusters = collections.defaultdict(list)
                speaker_clusters = collections.defaultdict(collections.Counter)
                for i in range(num_utterances):
                    u_id = utterance_ids[i]
                    speaker_cluster_id = labels[i]
                    utterance_clusters[speaker_cluster_id].append(u_id)
                    speaker_clusters[speaker_cluster_id][utt2spk[u_id]] += 1
                    pbar.update(1)
                utterance_mapping = []
                speaker_id = self.get_next_primary_key(Speaker)
                speaker_mapping = []

                for cluster_id, utterance_ids in utterance_clusters.items():
                    if cluster_id < 0:
                        speaker_name = "unknown"
                    else:
                        speaker_name = f"Cluster {cluster_id}"
                    speaker_mapping.append({"id": speaker_id, "name": speaker_name})
                    for u_id in utterance_ids:
                        utterance_mapping.append({"id": u_id, "speaker_id": speaker_id})
                    speaker_id += 1
                session.bulk_insert_mappings(Speaker, speaker_mapping)
                bulk_update(session, Utterance, utterance_mapping)

                session.commit()

    def cluster_speakers(self) -> None:
        self.setup()
        if self.ivector_extractor is None:
            raise NotImplementedError("SpeechBrain can only cluster utterances.")
        else:
            self.cluster_speakers_mfa()

    def cluster_utterances(self) -> None:
        self.setup()
        if self.ivector_extractor is None:
            self.cluster_utterances_speechbrain()
        else:
            self.cluster_utterances_mfa()

    def cluster_utterances_speechbrain(self) -> None:
        """
        Cluster utterances based on their ivectors
        """
        self.setup()
        utt_mapping = []
        import gc

        logger.info("Clustering utterances...")
        with tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            utterances = (
                session.query(Utterance)
                .filter(Utterance.ignored == False)  # noqa
                .options(
                    joinedload(Utterance.file).joinedload(File.sound_file),
                    joinedload(Utterance.speaker),
                )
            )
            num_utterances = utterances.count()
            batch = []
            utt_ids = []
            file_ids = []
            speaker_ids = []
            batch_size = GLOBAL_CONFIG.num_jobs
            embeddings = None
            for u in utterances:
                y, sr = librosa.load(
                    u.file.sound_file.sound_file_path,
                    sr=16000,
                    mono=False,
                    offset=u.begin,
                    duration=u.duration,
                )
                batch.append({"id": u.id, "wav": torch.tensor(y)})
                utt_ids.append(u.id)
                file_ids.append(u.file_id)
                speaker_ids.append(u.speaker_id)
                if len(batch) == batch_size:
                    batch = PaddedBatch(batch)
                    emb = (
                        self.speaker_recognition_model.encode_batch(
                            batch["wav"].data, batch["wav"].lengths
                        )
                        .cpu()
                        .numpy()
                        .squeeze(axis=1)
                    )
                    if embeddings is None:
                        embeddings = emb
                    else:
                        embeddings = np.append(embeddings, emb, axis=0)
                    del batch
                    gc.collect()
                    batch = []
                    pbar.update(batch_size)
            if len(batch):
                batch = PaddedBatch(batch)
                emb = (
                    self.speaker_recognition_model.encode_batch(
                        batch["wav"].data, batch["wav"].lengths
                    )
                    .cpu()
                    .numpy()
                    .squeeze(axis=1)
                )

                if embeddings is None:
                    embeddings = emb
                else:
                    embeddings = np.append(embeddings, emb, axis=0)
                pbar.update(len(batch))
                del batch
                gc.collect()

            if self.cluster_type in [
                "spectral",
                "dbscan",
                "optics",
                "affinity",
                "hdbscan",
                "agglomerative",
            ]:
                distance_matrix = pairwise_distances(
                    embeddings,
                    embeddings,
                    metric="cosine",
                    n_jobs=GLOBAL_CONFIG.current_profile.num_jobs,
                )

            kwargs = {}
            if self.cluster_type == "affinity":
                logger.info("Running Affinity Propagation...")
                to_fit = 1 - distance_matrix
            elif self.cluster_type == "agglomerative":
                logger.info("Running Agglomerative Clustering...")
                kwargs["n_clusters"] = self.expected_num_speakers
                to_fit = distance_matrix
            elif self.cluster_type == "spectral":
                logger.info("Running Spectral Clustering...")
                kwargs["n_clusters"] = self.expected_num_speakers
                to_fit = distance_matrix
            elif self.cluster_type == "dbscan":
                logger.info("Running DBSCAN...")
                to_fit = distance_matrix
            elif self.cluster_type == "hdbscan":
                logger.info("Running HDBSCAN...")
                to_fit = distance_matrix
                kwargs["min_cluster_size"] = 60
                kwargs["min_samples"] = 15
                kwargs["cluster_selection_epsilon"] = self.eps
            elif self.cluster_type == "optics":
                logger.info("Running OPTICS...")
                to_fit = distance_matrix
                kwargs["min_samples"] = 15
                kwargs["eps"] = self.eps
            elif self.cluster_type == "kmeans":
                logger.info("Running KMeans...")
                to_fit = embeddings
                kwargs["n_clusters"] = self.expected_num_speakers
            else:
                raise NotImplementedError(
                    f"The cluster type '{self.cluster_type} is not supported."
                )
            labels = cluster_matrix(to_fit, self.cluster_type, **kwargs)
            utterance_clusters = collections.defaultdict(list)
            speaker_clusters = collections.defaultdict(collections.Counter)
            for i in range(num_utterances):
                u_id = utt_ids[i]
                speaker_id = speaker_ids[i]
                speaker_cluster_id = labels[i]
                utterance_clusters[speaker_cluster_id].append(u_id)
                speaker_clusters[speaker_cluster_id][speaker_id] += 1
                pbar.update(1)
            utterance_mapping = []
            for cluster_id, utterance_ids in utterance_clusters.items():

                cluster_name = max(
                    speaker_clusters[cluster_id], key=lambda x: speaker_clusters[cluster_id][x]
                )
                for u_id in utterance_ids:
                    utterance_mapping.append({"id": u_id, "speaker_id": cluster_name})
            bulk_update(session, Utterance, utt_mapping)
            session.commit()

    def export_files(self, output_directory: str) -> None:
        """
        Export files with their new speaker labels

        Parameters
        ----------
        output_directory: str
            Output directory to save files
        """
        if not self.overwrite and os.path.exists(output_directory):
            output_directory = os.path.join(self.working_directory, "speaker_classification")
        os.makedirs(output_directory, exist_ok=True)

        with self.session() as session:
            logger.info("Writing results csv...")
            with mfa_open(
                os.path.join(output_directory, "speaker_classification_results.csv"), "w"
            ) as f, tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                writer = csv.DictWriter(f, ["file", "begin", "end", "speaker"])
                utterances = session.query(Utterance).options(
                    joinedload(Utterance.speaker, innerjoin=True),
                    joinedload(Utterance.file, innerjoin=True),
                )
                writer.writeheader()
                for u in utterances:
                    line = {
                        "file": u.file_name,
                        "begin": u.begin,
                        "end": u.end,
                        "speaker": u.speaker_name,
                    }
                    writer.writerow(line)
                    pbar.update(1)

            logger.info("Writing output files...")
            files = session.query(File).options(
                selectinload(File.utterances),
                selectinload(File.speakers),
                joinedload(File.sound_file, innerjoin=True).load_only(SoundFile.duration),
                joinedload(File.text_file, innerjoin=True).load_only(TextFile.file_type),
            )
            with tqdm.tqdm(total=self.num_files, disable=GLOBAL_CONFIG.quiet) as pbar:
                for file in files:
                    utterance_count = len(file.utterances)

                    if utterance_count == 0:
                        logger.debug(f"Could not find any utterances for {file.name}")
                    file.save(
                        output_directory, overwrite=True, output_format=TextFileType.TEXTGRID.value
                    )
                    pbar.update(1)
