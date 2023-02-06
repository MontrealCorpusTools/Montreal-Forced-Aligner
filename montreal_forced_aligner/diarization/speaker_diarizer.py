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
import random
import shutil
import subprocess
import sys
import time
import typing
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import sqlalchemy
import tqdm
import yaml
from sklearn import decomposition, metrics
from sqlalchemy.orm import joinedload, selectinload

from montreal_forced_aligner.abc import FileExporterMixin, TopLevelMfaWorker
from montreal_forced_aligner.alignment.multiprocessing import construct_output_path
from montreal_forced_aligner.config import (
    GLOBAL_CONFIG,
    IVECTOR_DIMENSION,
    MEMORY,
    PLDA_DIMENSION,
    XVECTOR_DIMENSION,
)
from montreal_forced_aligner.corpus.features import (
    ExportIvectorsArguments,
    ExportIvectorsFunction,
    PldaModel,
)
from montreal_forced_aligner.corpus.ivector_corpus import IvectorCorpusMixin
from montreal_forced_aligner.data import (
    ClusterType,
    DistanceMetric,
    ManifoldAlgorithm,
    WorkflowType,
)
from montreal_forced_aligner.db import (
    Corpus,
    File,
    SoundFile,
    Speaker,
    SpeakerOrdering,
    TextFile,
    Utterance,
    bulk_update,
)
from montreal_forced_aligner.diarization.multiprocessing import (
    ComputeEerArguments,
    ComputeEerFunction,
    PldaClassificationArguments,
    PldaClassificationFunction,
    SpeechbrainArguments,
    SpeechbrainClassificationFunction,
    SpeechbrainEmbeddingFunction,
    cluster_matrix,
    visualize_clusters,
)
from montreal_forced_aligner.exceptions import KaldiProcessingError
from montreal_forced_aligner.helper import load_configuration, mfa_open
from montreal_forced_aligner.models import IvectorExtractorModel
from montreal_forced_aligner.textgrid import export_textgrid
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function, thirdparty_binary

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        import torch
        from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
        from speechbrain.utils.metric_stats import EER

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError):
    FOUND_SPEECHBRAIN = False
    EncoderClassifier = None

if TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

__all__ = ["SpeakerDiarizer"]

logger = logging.getLogger("mfa")


class SpeakerDiarizer(IvectorCorpusMixin, TopLevelMfaWorker, FileExporterMixin):
    """
    Class for performing speaker classification, not currently very functional, but
    is planned to be expanded in the future

    Parameters
    ----------
    ivector_extractor_path : str
        Path to ivector extractor model, or "speechbrain"
    expected_num_speakers: int, optional
        Number of speakers in the corpus, if known
    cluster: bool
        Flag for whether speakers should be clustered instead of classified
    evaluation_mode: bool
        Flag for evaluating against existing speaker labels
    cuda: bool
        Flag for using CUDA for speechbrain models
    metric: str or :class:`~montreal_forced_aligner.data.DistanceMetric`
        One of "cosine", "plda", or "euclidean"
    cluster_type: str or :class:`~montreal_forced_aligner.data.ClusterType`
        Clustering algorithm
    relative_distance_threshold: float
        Threshold to use clustering based on distance
    """

    def __init__(
        self,
        ivector_extractor_path: str = "speechbrain",
        expected_num_speakers: int = 0,
        cluster: bool = True,
        evaluation_mode: bool = False,
        cuda: bool = False,
        use_pca: bool = True,
        metric: typing.Union[str, DistanceMetric] = "cosine",
        cluster_type: typing.Union[str, ClusterType] = "hdbscan",
        manifold_algorithm: typing.Union[str, ManifoldAlgorithm] = "tsne",
        distance_threshold: float = None,
        score_threshold: float = None,
        min_cluster_size: int = 60,
        max_iterations: int = 10,
        linkage: str = "average",
        **kwargs,
    ):
        self.use_xvector = False
        self.ivector_extractor = None
        self.ivector_extractor_path = ivector_extractor_path
        if ivector_extractor_path == "speechbrain":
            if not FOUND_SPEECHBRAIN:
                logger.error(
                    "Could not import speechbrain, please ensure it is installed via `pip install speechbrain`"
                )
                sys.exit(1)
            self.use_xvector = True
        else:
            self.ivector_extractor = IvectorExtractorModel(ivector_extractor_path)
            kwargs.update(self.ivector_extractor.parameters)
        super().__init__(**kwargs)
        self.expected_num_speakers = expected_num_speakers
        self.cluster = cluster
        self.metric = DistanceMetric[metric]
        self.cuda = cuda
        self.cluster_type = ClusterType[cluster_type]
        self.manifold_algorithm = ManifoldAlgorithm[manifold_algorithm]
        self.distance_threshold = distance_threshold
        self.score_threshold = score_threshold
        if self.distance_threshold is None:
            if self.use_xvector:
                self.distance_threshold = 0.25
        self.evaluation_mode = evaluation_mode
        self.min_cluster_size = min_cluster_size
        self.linkage = linkage
        self.use_pca = use_pca

        self.max_iterations = max_iterations
        self.current_labels = []
        self.classification_score = None
        self.initial_plda_score_threshold = 0
        self.plda_score_threshold = 10
        self.initial_sb_score_threshold = 0.25

        self.ground_truth_utt2spk = {}
        self.ground_truth_speakers = {}
        self.single_clusters = set()

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

    # noinspection PyTypeChecker
    def setup(self) -> None:
        """
        Sets up the corpus and speaker classifier

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.KaldiProcessingError`
            If there were any errors in running Kaldi binaries
        """
        if self.initialized:
            return
        super().setup()
        self.create_new_current_workflow(WorkflowType.speaker_diarization)
        wf = self.current_workflow
        if wf.done:
            logger.info("Diarization already done, skipping initialization.")
            return
        log_dir = os.path.join(self.working_directory, "log")
        os.makedirs(log_dir, exist_ok=True)
        try:
            if self.ivector_extractor is None:  # Download models if needed
                _ = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join(
                        GLOBAL_CONFIG.current_profile.temporary_directory,
                        "models",
                        "EncoderClassifier",
                    ),
                )
                _ = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join(
                        GLOBAL_CONFIG.current_profile.temporary_directory,
                        "models",
                        "SpeakerRecognition",
                    ),
                )
                self.initialize_database()
                self._load_corpus()
                self.initialize_jobs()
                self.load_embeddings()
                if self.cluster:
                    self.compute_speaker_embeddings()
            else:
                if not self.has_ivectors():
                    if self.ivector_extractor.meta["version"] < "2.1":
                        logger.warning(
                            "The ivector extractor was trained in an earlier version of MFA. "
                            "There may be incompatibilities in feature generation that cause errors. "
                            "Please download the latest version of the model via `mfa model download`, "
                            "use a different ivector extractor, or use version 2.0.6 of MFA."
                        )
                    self.ivector_extractor.export_model(self.working_directory)
                    self.load_corpus()
                    self.extract_ivectors()
                    self.compute_speaker_ivectors()
            if self.evaluation_mode:
                self.ground_truth_utt2spk = {}
                with self.session() as session:
                    query = session.query(Utterance.id, Utterance.speaker_id, Speaker.name).join(
                        Utterance.speaker
                    )
                    for u_id, s_id, name in query:
                        self.ground_truth_utt2spk[u_id] = s_id
                        self.ground_truth_speakers[s_id] = name
        except Exception as e:
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise
        self.initialized = True

    def plda_classification_arguments(self) -> List[PldaClassificationArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.diarization.multiprocessing.PldaClassificationFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.diarization.multiprocessing.PldaClassificationArguments`]
            Arguments for processing
        """
        return [
            PldaClassificationArguments(
                j.id,
                getattr(self, "db_string", ""),
                os.path.join(self.working_log_directory, f"plda_classification.{j.id}.log"),
                self.plda,
                self.speaker_ivector_path,
                self.num_utts_path,
                self.use_xvector,
            )
            for j in self.jobs
        ]

    def classify_speakers(self):
        """Classify speakers based on ivector or speechbrain model"""
        self.setup()
        logger.info("Classifying utterances...")

        with self.session() as session, tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, mfa_open(
            os.path.join(self.working_directory, "speaker_classification_results.csv"), "w"
        ) as f:
            writer = csv.DictWriter(f, ["utt_id", "file", "begin", "end", "speaker", "score"])

            writer.writeheader()
            file_names = {
                k: v for k, v in session.query(Utterance.id, File.name).join(Utterance.file)
            }
            utterance_times = {
                k: (b, e)
                for k, b, e in session.query(Utterance.id, Utterance.begin, Utterance.end)
            }
            utterance_mapping = []
            next_speaker_id = self.get_next_primary_key(Speaker)
            speaker_mapping = {}
            existing_speakers = {
                name: s_id for s_id, name in session.query(Speaker.id, Speaker.name)
            }
            self.classification_score = 0
            if session.query(Speaker).filter(Speaker.name == "MFA_UNKNOWN").first() is None:
                session.add(Speaker(id=next_speaker_id, name="MFA_UNKNOWN"))
                session.commit()
                next_speaker_id += 1
            unknown_speaker_id = (
                session.query(Speaker).filter(Speaker.name == "MFA_UNKNOWN").first().id
            )

            if self.use_xvector:
                arguments = [
                    SpeechbrainArguments(j.id, self.db_string, None, self.cuda, self.cluster)
                    for j in self.jobs
                ]
                func = SpeechbrainClassificationFunction
            else:
                plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
                with open(plda_transform_path, "rb") as f:
                    self.plda: PldaModel = pickle.load(f)
                arguments = self.plda_classification_arguments()
                func = PldaClassificationFunction
            for utt_id, classified_speaker, score in run_kaldi_function(
                func, arguments, pbar.update
            ):
                classified_speaker = str(classified_speaker)
                self.classification_score += score / self.num_utterances
                if self.score_threshold is not None and score < self.score_threshold:
                    speaker_id = unknown_speaker_id
                elif classified_speaker in existing_speakers:
                    speaker_id = existing_speakers[classified_speaker]
                else:
                    if classified_speaker not in speaker_mapping:
                        speaker_mapping[classified_speaker] = {
                            "id": next_speaker_id,
                            "name": classified_speaker,
                        }
                        next_speaker_id += 1
                    speaker_id = speaker_mapping[classified_speaker]["id"]
                utterance_mapping.append({"id": utt_id, "speaker_id": speaker_id})
                line = {
                    "utt_id": utt_id,
                    "file": file_names[utt_id],
                    "begin": utterance_times[utt_id][0],
                    "end": utterance_times[utt_id][1],
                    "speaker": classified_speaker,
                    "score": score,
                }
                writer.writerow(line)

            if self.stopped.stop_check():
                logger.debug("Stopping clustering early.")
                return
            if speaker_mapping:
                session.bulk_insert_mappings(Speaker, list(speaker_mapping.values()))
                session.flush()
            session.commit()
            bulk_update(session, Utterance, utterance_mapping)
            session.commit()
        if not self.evaluation_mode:
            self.clean_up_unknown_speaker()
        self.fix_speaker_ordering()
        if not self.evaluation_mode:
            self.cleanup_empty_speakers()
        self.refresh_speaker_vectors()
        if self.evaluation_mode:
            self.evaluate_classification()

    def map_speakers_to_ground_truth(self):
        with self.session() as session:

            utterances = session.query(Utterance.id, Utterance.speaker_id)
            labels = []
            utterance_ids = []
            for utt_id, s_id in utterances:
                utterance_ids.append(utt_id)
                labels.append(s_id)
            ground_truth = np.array([self.ground_truth_utt2spk[x] for x in utterance_ids])
            cluster_labels = np.unique(labels)
            ground_truth_labels = np.unique(ground_truth)
            cm = np.zeros((cluster_labels.shape[0], ground_truth_labels.shape[0]), dtype="int16")
            for y_pred, y in zip(labels, ground_truth):
                if y_pred < 0:
                    continue
                cm[np.where(cluster_labels == y_pred), np.where(ground_truth_labels == y)] += 1

            cm_argmax = cm.argmax(axis=1)
            label_to_ground_truth_mapping = {}
            for i in range(cluster_labels.shape[0]):
                label_to_ground_truth_mapping[int(cluster_labels[i])] = int(
                    ground_truth_labels[cm_argmax[i]]
                )
        return label_to_ground_truth_mapping

    def evaluate_clustering(self) -> None:
        """Compute clustering metric scores and output clustering evaluation results"""
        label_to_ground_truth_mapping = self.map_speakers_to_ground_truth()
        with self.session() as session, mfa_open(
            os.path.join(self.working_directory, "diarization_evaluation_results.csv"), "w"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "begin",
                    "end",
                    "text",
                    "predicted_speaker",
                    "ground_truth_speaker",
                ],
            )
            writer.writeheader()
            predicted_utt2spk = {}
            query = session.query(
                Utterance.id,
                File.name,
                Utterance.begin,
                Utterance.end,
                Utterance.text,
                Utterance.speaker_id,
            ).join(Utterance.file)
            for u_id, file_name, begin, end, text, s_id in query:
                s_id = label_to_ground_truth_mapping[s_id]
                predicted_utt2spk[u_id] = s_id
                writer.writerow(
                    {
                        "file": file_name,
                        "begin": begin,
                        "end": end,
                        "text": text,
                        "predicted_speaker": self.ground_truth_speakers[s_id],
                        "ground_truth_speaker": self.ground_truth_speakers[
                            self.ground_truth_utt2spk[u_id]
                        ],
                    }
                )

            ground_truth_labels = np.array([v for v in self.ground_truth_utt2spk.values()])
            predicted_labels = np.array(
                [predicted_utt2spk[k] for k in self.ground_truth_utt2spk.keys()]
            )
            rand_score = metrics.adjusted_rand_score(ground_truth_labels, predicted_labels)
            ami_score = metrics.adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
            nmi_score = metrics.normalized_mutual_info_score(ground_truth_labels, predicted_labels)
            homogeneity_score = metrics.homogeneity_score(ground_truth_labels, predicted_labels)
            completeness_score = metrics.completeness_score(ground_truth_labels, predicted_labels)
            v_measure_score = metrics.v_measure_score(ground_truth_labels, predicted_labels)
            fm_score = metrics.fowlkes_mallows_score(ground_truth_labels, predicted_labels)
            logger.info(f"Adjusted Rand index score (0-1, higher is better): {rand_score:.4f}")
            logger.info(f"Normalized Mutual Information score (perfect=1.0): {nmi_score:.4f}")
            logger.info(f"Adjusted Mutual Information score (perfect=1.0): {ami_score:.4f}")
            logger.info(f"Homogeneity score (0-1, higher is better): {homogeneity_score:.4f}")
            logger.info(f"Completeness score (0-1, higher is better): {completeness_score:.4f}")
            logger.info(f"V measure score (0-1, higher is better): {v_measure_score:.4f}")
            logger.info(f"Fowlkes-Mallows score (0-1, higher is better): {fm_score:.4f}")

    def evaluate_classification(self) -> None:
        """Evaluate and output classification accuracy"""
        label_to_ground_truth_mapping = self.map_speakers_to_ground_truth()
        with self.session() as session, mfa_open(
            os.path.join(self.working_directory, "diarization_evaluation_results.csv"), "w"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "begin",
                    "end",
                    "text",
                    "predicted_speaker",
                    "ground_truth_speaker",
                ],
            )
            writer.writeheader()
            predicted_utt2spk = {}
            query = session.query(
                Utterance.id,
                File.name,
                Utterance.begin,
                Utterance.end,
                Utterance.text,
                Utterance.speaker_id,
            ).join(Utterance.file)
            for u_id, file_name, begin, end, text, s_id in query:
                s_id = label_to_ground_truth_mapping[s_id]
                predicted_utt2spk[u_id] = s_id
                writer.writerow(
                    {
                        "file": file_name,
                        "begin": begin,
                        "end": end,
                        "text": text,
                        "predicted_speaker": self.ground_truth_speakers[s_id],
                        "ground_truth_speaker": self.ground_truth_speakers[
                            self.ground_truth_utt2spk[u_id]
                        ],
                    }
                )

            ground_truth_labels = np.array([v for v in self.ground_truth_utt2spk.values()])
            predicted_labels = np.array(
                [
                    predicted_utt2spk[k] if k in predicted_utt2spk else -1
                    for k in self.ground_truth_utt2spk.keys()
                ]
            )
            precision_score = metrics.precision_score(
                ground_truth_labels, predicted_labels, average="weighted"
            )
            recall_score = metrics.recall_score(
                ground_truth_labels, predicted_labels, average="weighted"
            )
            f1_score = metrics.f1_score(ground_truth_labels, predicted_labels, average="weighted")
            logger.info(f"Precision (0-1): {precision_score:.4f}")
            logger.info(f"Recall (0-1): {recall_score:.4f}")
            logger.info(f"F1 (0-1): {f1_score:.4f}")

    @property
    def num_utts_path(self) -> str:
        """Path to archive containing number of per training speaker"""
        return os.path.join(self.working_directory, "num_utts.ark")

    @property
    def speaker_ivector_path(self) -> str:
        """Path to archive containing training speaker ivectors"""
        return os.path.join(self.working_directory, "speaker_ivectors.ark")

    def visualize_clusters(self, ivectors, cluster_labels=None):
        import seaborn as sns
        from matplotlib import pyplot as plt

        sns.set()
        metric = self.metric
        if metric is DistanceMetric.plda:
            metric = DistanceMetric.cosine
        points = visualize_clusters(ivectors, self.manifold_algorithm, metric, 10, self.plda)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            num_unique_labels = unique_labels.shape[0]
            has_noise = 0 in set(unique_labels)
            if has_noise:
                num_unique_labels -= 1
            cm = sns.color_palette("tab20", num_unique_labels)
            for cluster in unique_labels:
                if cluster == -1:
                    color = "k"
                    name = "Noise"
                    alpha = 0.75
                else:
                    name = cluster
                    if not isinstance(name, str):
                        name = f"Cluster {name}"
                        cluster_id = cluster
                    else:
                        cluster_id = np.where(unique_labels == cluster)[0][0]
                    if has_noise:
                        color = cm[cluster_id - 1]
                    else:
                        color = cm[cluster_id]
                    alpha = 1.0
                idx = np.where(cluster_labels == cluster)
                ax.scatter(points[idx, 0], points[idx, 1], color=color, label=name, alpha=alpha)
        else:
            ax.scatter(points[:, 0], points[:, 1])
        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.3, wspace=0.33)
        plt.axis("off")
        lgd = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        plot_path = os.path.join(self.working_directory, "cluster_plot.png")
        plt.savefig(plot_path, bbox_extra_artists=(lgd,), bbox_inches="tight", transparent=True)
        if GLOBAL_CONFIG.current_profile.verbose:
            plt.show(block=False)
            plt.pause(10)
            logger.debug(f"Closing cluster plot, it has been saved to {plot_path}.")
            plt.close()

    def export_xvectors(self):
        logger.info("Exporting SpeechBrain embeddings...")
        os.makedirs(self.split_directory, exist_ok=True)
        with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
            arguments = [
                ExportIvectorsArguments(
                    j.id,
                    self.db_string,
                    j.construct_path(self.working_log_directory, "export_ivectors", "log"),
                    self.use_xvector,
                )
                for j in self.jobs
            ]
            utterance_mapping = []
            for utt_id, ark_path in run_kaldi_function(
                ExportIvectorsFunction, arguments, pbar.update
            ):
                utterance_mapping.append({"id": utt_id, "ivector_ark": ark_path})
            with self.session() as session:
                bulk_update(session, Utterance, utterance_mapping)
                session.commit()
        self._write_ivectors()

    def fix_speaker_ordering(self):
        with self.session() as session:
            query = (
                session.query(Speaker.id, File.id)
                .join(Utterance.speaker)
                .join(Utterance.file)
                .distinct()
            )
            speaker_ordering_mapping = []
            for s_id, f_id in query:
                speaker_ordering_mapping.append({"speaker_id": s_id, "file_id": f_id, "index": 10})
            session.execute(sqlalchemy.delete(SpeakerOrdering))
            session.flush()
            session.execute(
                sqlalchemy.dialects.postgresql.insert(SpeakerOrdering)
                .values(speaker_ordering_mapping)
                .on_conflict_do_nothing()
            )
            session.commit()

    def initialize_mfa_clustering(self):

        with self.session() as session:

            next_speaker_id = self.get_next_primary_key(Speaker)
            speaker_mapping = {}
            existing_speakers = {
                name: s_id for s_id, name in session.query(Speaker.id, Speaker.name)
            }
            utterance_mapping = []
            self.classification_score = 0
            unk_count = 0
            if self.use_xvector:
                arguments = [
                    SpeechbrainArguments(j.id, self.db_string, None, self.cuda, self.cluster)
                    for j in self.jobs
                ]
                func = SpeechbrainClassificationFunction
                score_threshold = self.initial_sb_score_threshold
                self.export_xvectors()
            else:
                plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
                with open(plda_transform_path, "rb") as f:
                    self.plda: PldaModel = pickle.load(f)
                arguments = self.plda_classification_arguments()
                func = PldaClassificationFunction
                score_threshold = self.initial_plda_score_threshold

            logger.info("Generating initial speaker labels...")
            utt2spk = {k: v for k, v in session.query(Utterance.id, Utterance.speaker_id)}
            with tqdm.tqdm(total=self.num_utterances, disable=GLOBAL_CONFIG.quiet) as pbar:
                for utt_id, classified_speaker, score in run_kaldi_function(
                    func, arguments, pbar.update
                ):
                    classified_speaker = str(classified_speaker)
                    self.classification_score += score / self.num_utterances
                    if score < score_threshold:
                        unk_count += 1
                        utterance_mapping.append(
                            {"id": utt_id, "speaker_id": existing_speakers["MFA_UNKNOWN"]}
                        )
                        continue
                    if classified_speaker in existing_speakers:
                        speaker_id = existing_speakers[classified_speaker]
                    else:
                        if classified_speaker not in speaker_mapping:
                            speaker_mapping[classified_speaker] = {
                                "id": next_speaker_id,
                                "name": classified_speaker,
                            }
                            next_speaker_id += 1
                        speaker_id = speaker_mapping[classified_speaker]["id"]
                    if speaker_id == utt2spk[utt_id]:
                        continue
                    utterance_mapping.append({"id": utt_id, "speaker_id": speaker_id})
            if speaker_mapping:
                session.bulk_insert_mappings(Speaker, list(speaker_mapping.values()))
                session.flush()
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS ix_utterance_speaker_id"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS utterance_position_index"))
            session.commit()
            bulk_update(session, Utterance, utterance_mapping)
            session.execute(
                sqlalchemy.text("CREATE INDEX ix_utterance_speaker_id on utterance(speaker_id)")
            )
            session.execute(
                sqlalchemy.text(
                    'CREATE INDEX utterance_position_index on utterance(file_id, speaker_id, begin, "end", channel)'
                )
            )
            session.commit()
        self.breakup_large_clusters()
        self.cleanup_empty_speakers()

    def export_speaker_ivectors(self):
        logger.info("Exporting current speaker ivectors...")

        with self.session() as session, tqdm.tqdm(
            total=self.num_speakers, disable=GLOBAL_CONFIG.quiet
        ) as pbar, mfa_open(self.num_utts_path, "w") as f:
            if self.use_xvector:
                ivector_column = Speaker.xvector
            else:
                ivector_column = Speaker.ivector

            speakers = (
                session.query(Speaker.id, ivector_column, sqlalchemy.func.count(Utterance.id))
                .join(Speaker.utterances)
                .filter(Speaker.name != "MFA_UNKNOWN")
                .group_by(Speaker.id)
                .order_by(Speaker.id)
            )
            input_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-vector"),
                    "--binary=true",
                    "ark,t:-",
                    f"ark:{self.speaker_ivector_path}",
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=os.environ,
            )
            for s_id, ivector, utterance_count in speakers:
                if ivector is None:
                    continue
                ivector = " ".join([format(x, ".12g") for x in ivector])
                in_line = f"{s_id}  [ {ivector} ]\n".encode("utf8")
                input_proc.stdin.write(in_line)
                input_proc.stdin.flush()
                pbar.update(1)
                f.write(f"{s_id} {utterance_count}\n")
            input_proc.stdin.close()
            input_proc.wait()

    def classify_iteration(self, iteration=None) -> None:
        logger.info("Classifying utterances...")

        low_count = None
        if iteration is not None and self.min_cluster_size:
            low_count = np.linspace(0, self.min_cluster_size, self.max_iterations)[iteration]
            logger.debug(f"Minimum size: {low_count}")
        score_threshold = self.plda_score_threshold
        if iteration is not None:
            score_threshold = np.linspace(
                self.initial_plda_score_threshold,
                self.plda_score_threshold,
                self.max_iterations,
            )[iteration]
        logger.debug(f"Score threshold: {score_threshold}")
        with self.session() as session, tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:

            unknown_speaker_id = (
                session.query(Speaker.id).filter(Speaker.name == "MFA_UNKNOWN").first()[0]
            )

            utterance_mapping = []
            self.classification_score = 0
            plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
            with open(plda_transform_path, "rb") as f:
                self.plda: PldaModel = pickle.load(f)
            arguments = self.plda_classification_arguments()
            func = PldaClassificationFunction
            utt2spk = {k: v for k, v in session.query(Utterance.id, Utterance.speaker_id)}

            for utt_id, classified_speaker, score in run_kaldi_function(
                func, arguments, pbar.update
            ):
                self.classification_score += score / self.num_utterances
                if score < score_threshold:
                    speaker_id = unknown_speaker_id
                else:
                    speaker_id = classified_speaker
                if speaker_id == utt2spk[utt_id]:
                    continue
                utterance_mapping.append({"id": utt_id, "speaker_id": speaker_id})

            logger.debug(f"Updating {len(utterance_mapping)} utterances with new speakers")

            session.commit()
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS ix_utterance_speaker_id"))
            session.execute(sqlalchemy.text("DROP INDEX IF EXISTS utterance_position_index"))
            session.commit()
            bulk_update(session, Utterance, utterance_mapping)
            session.execute(
                sqlalchemy.text("CREATE INDEX ix_utterance_speaker_id on utterance(speaker_id)")
            )
            session.execute(
                sqlalchemy.text(
                    'CREATE INDEX utterance_position_index on utterance(file_id, speaker_id, begin, "end", channel)'
                )
            )
            session.commit()
        if iteration is not None and iteration < self.max_iterations - 2:
            self.breakup_large_clusters()
        self.cleanup_empty_speakers(low_count)

    def breakup_large_clusters(self):
        with self.session() as session:
            unknown_speaker_id = (
                session.query(Speaker.id).filter(Speaker.name == "MFA_UNKNOWN").first()[0]
            )
            sq = (
                session.query(Speaker.id, sqlalchemy.func.count().label("utterance_count"))
                .join(Speaker.utterances)
                .filter(Speaker.id != unknown_speaker_id)
                .group_by(Speaker.id)
            )
            above_threshold_speakers = [unknown_speaker_id]
            threshold = 500
            for s_id, utterance_count in sq:
                if threshold and utterance_count > threshold and s_id not in self.single_clusters:
                    above_threshold_speakers.append(s_id)
            logger.info("Breaking up large speakers...")
            logger.debug(f"Unknown speaker is {unknown_speaker_id}")
            next_speaker_id = self.get_next_primary_key(Speaker)
            with tqdm.tqdm(
                total=len(above_threshold_speakers), disable=GLOBAL_CONFIG.quiet
            ) as pbar:
                utterance_mapping = []
                new_speakers = {}
                for s_id in above_threshold_speakers:
                    logger.debug(f"Breaking up {s_id}")
                    query = session.query(Utterance.id, Utterance.plda_vector).filter(
                        Utterance.plda_vector != None, Utterance.speaker_id == s_id  # noqa
                    )
                    pbar.update(1)
                    ivectors = np.empty((query.count(), PLDA_DIMENSION))
                    logger.debug(f"Had {ivectors.shape[0]} utterances.")
                    if ivectors.shape[0] == 0:
                        continue
                    utterance_ids = []
                    for i, (u_id, ivector) in enumerate(query):
                        if self.stopped.stop_check():
                            break
                        utterance_ids.append(u_id)
                        ivectors[i, :] = ivector
                    if ivectors.shape[0] < self.min_cluster_size:
                        continue
                    labels = cluster_matrix(
                        ivectors,
                        ClusterType.optics,
                        metric=DistanceMetric.cosine,
                        strict=False,
                        no_visuals=True,
                        working_directory=self.working_directory,
                        distance_threshold=0.25,
                    )
                    unique, counts = np.unique(labels, return_counts=True)
                    num_clusters = unique.shape[0]
                    counts = dict(zip(unique, counts))
                    logger.debug(f"{num_clusters} clusters found: {counts}")
                    if num_clusters == 1:
                        if s_id != unknown_speaker_id:
                            logger.debug(f"Deleting {s_id} due to no clusters found")
                            session.execute(
                                sqlalchemy.update(Utterance)
                                .filter(Utterance.speaker_id == s_id)
                                .values({Utterance.speaker_id: unknown_speaker_id})
                            )
                            session.flush()
                        continue
                    if num_clusters == 2:
                        if s_id != unknown_speaker_id:
                            logger.debug(
                                f"Only found one cluster for {s_id} will skip in the future"
                            )
                            self.single_clusters.add(s_id)
                        continue
                    for i, utt_id in enumerate(utterance_ids):
                        label = labels[i]
                        if label == -1:
                            speaker_id = unknown_speaker_id
                        else:
                            if s_id in self.single_clusters:
                                continue
                            if label not in new_speakers:
                                if s_id == unknown_speaker_id:
                                    label = self._unknown_speaker_break_up_count
                                    self._unknown_speaker_break_up_count += 1
                                new_speakers[label] = {
                                    "id": next_speaker_id,
                                    "name": f"{s_id}_{label}",
                                }
                                next_speaker_id += 1
                            speaker_id = new_speakers[label]["id"]
                        utterance_mapping.append({"id": utt_id, "speaker_id": speaker_id})
                if new_speakers:
                    session.bulk_insert_mappings(Speaker, list(new_speakers.values()))
                    session.commit()
                if utterance_mapping:
                    bulk_update(session, Utterance, utterance_mapping)
                session.commit()
            logger.debug(f"Broke speakers into {len(new_speakers)} new speakers.")

    def cleanup_empty_speakers(self, threshold=None):
        with self.session() as session:
            session.execute(sqlalchemy.delete(SpeakerOrdering))
            session.flush()
            unknown_speaker_id = (
                session.query(Speaker.id).filter(Speaker.name == "MFA_UNKNOWN").first()[0]
            )
            non_empty_speakers = [unknown_speaker_id]
            sq = (
                session.query(Speaker.id, sqlalchemy.func.count().label("utterance_count"))
                .join(Speaker.utterances)
                .filter(Speaker.id != unknown_speaker_id)
                .group_by(Speaker.id)
            )
            below_threshold_speakers = []
            for s_id, utterance_count in sq:
                if threshold and utterance_count < threshold:
                    below_threshold_speakers.append(s_id)
                    continue
                non_empty_speakers.append(s_id)
            session.execute(
                sqlalchemy.update(Utterance)
                .where(Utterance.speaker_id.in_(below_threshold_speakers))
                .values(speaker_id=unknown_speaker_id)
            )
            session.execute(sqlalchemy.delete(Speaker).where(~Speaker.id.in_(non_empty_speakers)))
            session.commit()
            self._num_speakers = session.query(Speaker).count()
        conn = self.db_engine.connect()
        try:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(
                sqlalchemy.text(f"ANALYZE {Speaker.__tablename__}, {Utterance.__tablename__}")
            )
        finally:
            conn.close()

    def cluster_utterances_mfa(self) -> None:
        """
        Cluster utterances with a ivector or speechbrain model
        """
        self.cluster = False
        self.setup()
        with self.session() as session:
            if session.query(Speaker).filter(Speaker.name == "MFA_UNKNOWN").first() is None:
                session.add(Speaker(id=self.get_next_primary_key(Speaker), name="MFA_UNKNOWN"))
                session.commit()
        self.initialize_mfa_clustering()
        with self.session() as session:
            uncategorized_count = (
                session.query(Utterance)
                .join(Utterance.speaker)
                .filter(Speaker.name == "MFA_UNKNOWN")
                .count()
            )
        if self.use_xvector:
            logger.info(f"Initial average cosine score {self.classification_score:.4f}")
        else:
            logger.info(f"Initial average PLDA score {self.classification_score:.4f}")
        logger.info(f"Number of speakers: {self.num_speakers}")
        logger.info(f"Unclassified utterances: {uncategorized_count}")
        self._unknown_speaker_break_up_count = 0
        for i in range(self.max_iterations):
            logger.info(f"Iteration {i}:")
            current_score = self.classification_score
            self._write_ivectors()
            self.compute_plda()
            self.refresh_plda_vectors()
            self.refresh_speaker_vectors()
            self.export_speaker_ivectors()
            self.classify_iteration(i)
            improvement = self.classification_score - current_score
            with self.session() as session:
                uncategorized_count = (
                    session.query(Utterance)
                    .join(Utterance.speaker)
                    .filter(Speaker.name == "MFA_UNKNOWN")
                    .count()
                )
            logger.info(f"Average PLDA score {self.classification_score:.4f}")
            logger.info(f"Improvement: {improvement:.4f}")
            logger.info(f"Number of speakers: {self.num_speakers}")
            logger.info(f"Unclassified utterances: {uncategorized_count}")
        logger.debug(f"Found {self.num_speakers} clusters")
        if GLOBAL_CONFIG.current_profile.debug:
            self.visualize_current_clusters()

    def visualize_current_clusters(self):
        with self.session() as session:
            query = (
                session.query(Speaker.name, Utterance.plda_vector)
                .join(Utterance.speaker)
                .filter(Utterance.plda_vector is not None)
            )
            dim = PLDA_DIMENSION
            num_utterances = query.count()
            if num_utterances == 0:
                if self.use_xvector:
                    column = Utterance.xvector
                    dim = XVECTOR_DIMENSION
                else:
                    column = Utterance.ivector
                    dim = IVECTOR_DIMENSION
                query = (
                    session.query(Speaker.name, column)
                    .join(Utterance.speaker)
                    .filter(column is not None)
                )
                num_utterances = query.count()
                if num_utterances == 0:
                    logger.warning("No ivectors/xvectors to visualize")
                    return
            ivectors = np.empty((query.count(), dim))
            labels = []
            for s_name, ivector in query:
                ivectors[len(labels), :] = ivector
                labels.append(s_name)
            self.visualize_clusters(ivectors, labels)

    def cluster_utterances(self) -> None:
        """
        Cluster utterances with a ivector or speechbrain model
        """
        if self.cluster_type is ClusterType.mfa:
            self.cluster_utterances_mfa()
            self.fix_speaker_ordering()
            if not self.evaluation_mode:
                self.cleanup_empty_speakers()
            self.refresh_speaker_vectors()
            if self.evaluation_mode:
                self.evaluate_clustering()
            return
        self.setup()

        os.environ["OMP_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.num_jobs}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.num_jobs}"
        os.environ["MKL_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.num_jobs}"
        if self.metric is DistanceMetric.plda:
            plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
            with open(plda_transform_path, "rb") as f:
                self.plda: PldaModel = pickle.load(f)
        if self.evaluation_mode and GLOBAL_CONFIG.current_profile.debug:
            self.calculate_eer()
        logger.info("Clustering utterances (this may take a while, please be patient)...")
        with self.session() as session:
            if self.use_pca:
                query = session.query(Utterance.id, Utterance.plda_vector).filter(
                    Utterance.plda_vector != None  # noqa
                )
                ivectors = np.empty((query.count(), PLDA_DIMENSION))
            elif self.use_xvector:
                query = session.query(Utterance.id, Utterance.xvector).filter(
                    Utterance.xvector != None  # noqa
                )
                ivectors = np.empty((query.count(), XVECTOR_DIMENSION))
            else:
                query = session.query(Utterance.id, Utterance.ivector).filter(
                    Utterance.ivector != None  # noqa
                )
                ivectors = np.empty((query.count(), IVECTOR_DIMENSION))
            utterance_ids = []
            for i, (u_id, ivector) in enumerate(query):
                if self.stopped.stop_check():
                    break
                utterance_ids.append(u_id)
                ivectors[i, :] = ivector
            num_utterances = ivectors.shape[0]
            kwargs = {}

            if self.stopped.stop_check():
                logger.debug("Stopping clustering early.")
                return
            kwargs["min_cluster_size"] = self.min_cluster_size
            kwargs["distance_threshold"] = self.distance_threshold
            if self.cluster_type is ClusterType.agglomerative:
                kwargs["memory"] = MEMORY
                kwargs["linkage"] = self.linkage
                kwargs["n_clusters"] = self.expected_num_speakers
                if not self.expected_num_speakers:
                    kwargs["n_clusters"] = None
            elif self.cluster_type is ClusterType.spectral:
                kwargs["n_clusters"] = self.expected_num_speakers
            elif self.cluster_type is ClusterType.hdbscan:
                kwargs["memory"] = MEMORY
            elif self.cluster_type is ClusterType.optics:
                kwargs["memory"] = MEMORY
            elif self.cluster_type is ClusterType.kmeans:
                kwargs["n_clusters"] = self.expected_num_speakers
            labels = cluster_matrix(
                ivectors,
                self.cluster_type,
                metric=self.metric,
                plda=self.plda,
                working_directory=self.working_directory,
                **kwargs,
            )
            if self.stopped.stop_check():
                logger.debug("Stopping clustering early.")
                return
            if GLOBAL_CONFIG.current_profile.debug:
                self.visualize_clusters(ivectors, labels)

            utterance_clusters = collections.defaultdict(list)
            for i in range(num_utterances):
                u_id = utterance_ids[i]
                cluster_id = int(labels[i])
                utterance_clusters[cluster_id].append(u_id)

            utterance_mapping = []
            next_speaker_id = self.get_next_primary_key(Speaker)
            speaker_mapping = []
            unknown_speaker_id = None
            for cluster_id, utterance_ids in sorted(utterance_clusters.items()):
                if cluster_id < 0:
                    if unknown_speaker_id is None:
                        speaker_name = "MFA_UNKNOWN"
                        speaker_mapping.append({"id": next_speaker_id, "name": speaker_name})
                        speaker_id = next_speaker_id
                        unknown_speaker_id = speaker_id
                        next_speaker_id += 1
                    else:
                        speaker_id = unknown_speaker_id
                else:
                    speaker_name = f"Cluster {cluster_id}"
                    speaker_mapping.append({"id": next_speaker_id, "name": speaker_name})
                    speaker_id = next_speaker_id
                    next_speaker_id += 1
                for u_id in utterance_ids:
                    utterance_mapping.append({"id": u_id, "speaker_id": speaker_id})
            if self.stopped.stop_check():
                logger.debug("Stopping clustering early.")
                return
            if speaker_mapping:
                session.bulk_insert_mappings(Speaker, speaker_mapping)
                session.flush()
            session.commit()
            bulk_update(session, Utterance, utterance_mapping)
            session.flush()
            session.commit()
        if not self.evaluation_mode:
            self.clean_up_unknown_speaker()
        self.fix_speaker_ordering()
        if not self.evaluation_mode:
            self.cleanup_empty_speakers()
        self.refresh_speaker_vectors()
        if self.evaluation_mode:
            self.evaluate_clustering()

        os.environ["OMP_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"

    def clean_up_unknown_speaker(self):
        with self.session() as session:
            unknown_speaker = session.query(Speaker).filter(Speaker.name == "MFA_UNKNOWN").first()
            next_speaker_id = self.get_next_primary_key(Speaker)
            if unknown_speaker is not None:
                speaker_mapping = {}
                utterance_mapping = []
                query = (
                    session.query(File.id, File.name)
                    .join(File.utterances)
                    .filter(Utterance.speaker_id == unknown_speaker.id)
                    .distinct()
                )
                for file_id, file_name in query:
                    speaker_mapping[file_id] = {"id": next_speaker_id, "name": file_name}
                    next_speaker_id += 1
                query = (
                    session.query(Utterance.id, Utterance.file_id)
                    .join(File.utterances)
                    .filter(Utterance.speaker_id == unknown_speaker.id)
                )
                for utterance_id, file_id in query:
                    utterance_mapping.append(
                        {"id": utterance_id, "speaker_id": speaker_mapping[file_id]["id"]}
                    )

                session.bulk_insert_mappings(Speaker, list(speaker_mapping.values()))
                session.flush()
                session.execute(
                    sqlalchemy.delete(SpeakerOrdering).where(
                        SpeakerOrdering.c.speaker_id == unknown_speaker.id
                    )
                )
                session.commit()
                bulk_update(session, Utterance, utterance_mapping)
                session.commit()

    def calculate_eer(self) -> typing.Tuple[float, float]:
        """
        Calculate Equal Error Rate (EER) and threshold for the diarization metric using the ground truth data.

        Returns
        -------
        float
            EER
        float
            Threshold of EER
        """
        if not FOUND_SPEECHBRAIN:
            logger.info("No speechbrain found, skipping EER calculation.")
            return 0.0, 0.0
        logger.info("Calculating EER using ground truth speakers...")
        limit_per_speaker = 5
        limit_within_speaker = 30
        begin = time.time()
        with tqdm.tqdm(total=self.num_speakers, disable=GLOBAL_CONFIG.quiet) as pbar:
            arguments = [
                ComputeEerArguments(
                    j.id,
                    self.db_string,
                    None,
                    self.plda,
                    self.metric,
                    self.use_xvector,
                    limit_within_speaker,
                    limit_per_speaker,
                )
                for j in self.jobs
            ]
            match_scores = []
            mismatch_scores = []
            for matches, mismatches in run_kaldi_function(
                ComputeEerFunction, arguments, pbar.update
            ):
                match_scores.extend(matches)
                mismatch_scores.extend(mismatches)
            random.shuffle(mismatches)
            mismatch_scores = mismatch_scores[: len(match_scores)]
            match_scores = np.array(match_scores)
            mismatch_scores = np.array(mismatch_scores)
            device = torch.device("cuda" if self.cuda else "cpu")
            eer, thresh = EER(
                torch.tensor(mismatch_scores, device=device),
                torch.tensor(match_scores, device=device),
            )
            logger.debug(
                f"Matching scores: {np.min(match_scores):.3f}-{np.max(match_scores):.3f} (mean = {match_scores.mean():.3f}, n = {match_scores.shape[0]})"
            )
            logger.debug(
                f"Mismatching scores: {np.min(mismatch_scores):.3f}-{np.max(mismatch_scores):.3f} (mean = {mismatch_scores.mean():.3f}, n = {mismatch_scores.shape[0]})"
            )
            logger.info(f"EER: {eer*100:.2f}%")
            logger.info(f"Threshold: {thresh:.4f}")
        logger.debug(f"Calculating EER took {time.time() - begin:.3f} seconds")
        return eer, thresh

    def load_embeddings(self) -> None:
        """Load embeddings from a speechbrain model"""
        if self.has_xvectors():
            logger.info("Embeddings already loaded.")
            return
        logger.info("Loading SpeechBrain embeddings...")
        with tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            begin = time.time()
            update_mapping = {}
            arguments = [
                SpeechbrainArguments(j.id, self.db_string, None, self.cuda, self.cluster)
                for j in self.jobs
            ]
            embeddings = []
            utterance_ids = []
            for u_id, emb in run_kaldi_function(
                SpeechbrainEmbeddingFunction, arguments, pbar.update
            ):
                utterance_ids.append(u_id)
                embeddings.append(emb)
                update_mapping[u_id] = {"id": u_id, "xvector": emb}
            embeddings = np.array(embeddings)
            if PLDA_DIMENSION != XVECTOR_DIMENSION:
                if embeddings.shape[0] < PLDA_DIMENSION:
                    logger.debug("Can't run PLDA due to too few features.")
                else:
                    pca = decomposition.PCA(PLDA_DIMENSION)
                    pca.fit(embeddings)
                    logger.debug(
                        f"PCA explained variance: {np.sum(pca.explained_variance_ratio_)*100:.2f}%"
                    )
                    transformed = pca.transform(embeddings)
                    for i, u_id in enumerate(utterance_ids):
                        update_mapping[u_id]["plda_vector"] = transformed[i, :]
            else:
                for v in update_mapping.values():
                    v["plda_vector"] = v["xvector"]
            bulk_update(session, Utterance, list(update_mapping.values()))
            session.query(Corpus).update({Corpus.xvectors_loaded: True})
            session.commit()
            logger.debug(f"Loading embeddings took {time.time() - begin:.3f} seconds")

    def refresh_plda_vectors(self):
        logger.info("Refreshing PLDA vectors...")
        self.plda = PldaModel.load(self.plda_path)
        with self.session() as session, tqdm.tqdm(
            total=self.num_utterances, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            if self.use_xvector:
                ivector_column = Utterance.xvector
            else:
                ivector_column = Utterance.ivector
            update_mapping = []
            utterance_ids = []
            ivectors = []
            utterances = session.query(Utterance.id, ivector_column).filter(
                ivector_column != None  # noqa
            )
            for utt_id, ivector in utterances:
                pbar.update(1)
                utterance_ids.append(utt_id)
                ivectors.append(ivector)
            ivectors = np.array(ivectors)
            ivectors = self.plda.process_ivectors(ivectors)
            for i, utt_id in enumerate(utterance_ids):
                update_mapping.append({"id": utt_id, "plda_vector": ivectors[i, :]})
            bulk_update(session, Utterance, update_mapping)
            session.commit()
        plda_transform_path = os.path.join(self.working_directory, "plda.pkl")
        with open(plda_transform_path, "wb") as f:
            pickle.dump(self.plda, f)

    def refresh_speaker_vectors(self) -> None:
        """Refresh speaker vectors following clustering or classification"""
        logger.info("Refreshing speaker vectors...")
        with self.session() as session, tqdm.tqdm(
            total=self.num_speakers, disable=GLOBAL_CONFIG.quiet
        ) as pbar:
            if self.use_xvector:
                ivector_column = Utterance.xvector
            else:
                ivector_column = Utterance.ivector
            update_mapping = {}
            speaker_ids = []
            ivectors = []
            speakers = session.query(Speaker.id)
            for (s_id,) in speakers:
                query = session.query(ivector_column).filter(Utterance.speaker_id == s_id)
                s_ivectors = []
                for (u_ivector,) in query:
                    s_ivectors.append(u_ivector)
                if not s_ivectors:
                    continue
                mean_ivector = np.mean(np.array(s_ivectors), axis=0)
                speaker_ids.append(s_id)
                ivectors.append(mean_ivector)
                if self.use_xvector:
                    key = "xvector"
                else:
                    key = "ivector"
                update_mapping[s_id] = {"id": s_id, key: mean_ivector}
                pbar.update(1)
            ivectors = np.array(ivectors)
            if self.plda is not None:
                ivectors = self.plda.process_ivectors(ivectors)
                for i, speaker_id in enumerate(speaker_ids):
                    update_mapping[speaker_id]["plda_vector"] = ivectors[i, :]
            bulk_update(session, Speaker, list(update_mapping.values()))
            session.commit()

    def compute_speaker_embeddings(self) -> None:
        """Generate per-speaker embeddings as the mean over their utterances"""
        if not self.has_xvectors():
            self.load_embeddings()
        logger.info("Computing SpeechBrain speaker embeddings...")
        with tqdm.tqdm(
            total=self.num_speakers, disable=GLOBAL_CONFIG.quiet
        ) as pbar, self.session() as session:
            update_mapping = []
            speakers = session.query(Speaker.id)
            for (s_id,) in speakers:
                u_query = session.query(Utterance.xvector).filter(
                    Utterance.speaker_id == s_id, Utterance.xvector != None  # noqa
                )
                embeddings = np.empty((u_query.count(), XVECTOR_DIMENSION))
                if embeddings.shape[0] == 0:
                    continue
                for i, (xvector,) in enumerate(u_query):
                    embeddings[i, :] = xvector
                speaker_xvector = np.mean(embeddings, axis=0)
                update_mapping.append({"id": s_id, "xvector": speaker_xvector})
                pbar.update(1)
            bulk_update(session, Speaker, update_mapping)
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
        diagnostic_files = [
            "diarization_evaluation_results.csv",
            "cluster_plot.png",
            "nearest_neighbors.png",
        ]
        for fname in diagnostic_files:
            path = os.path.join(self.working_directory, fname)
            if os.path.exists(path):
                shutil.copyfile(
                    path,
                    os.path.join(output_directory, fname),
                )
        with mfa_open(os.path.join(output_directory, "parameters.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "ivector_extractor_path": self.ivector_extractor_path,
                    "expected_num_speakers": self.expected_num_speakers,
                    "cluster": self.cluster,
                    "cuda": self.cuda,
                    "metric": self.metric.name,
                    "cluster_type": self.cluster_type.name,
                    "distance_threshold": self.distance_threshold,
                    "min_cluster_size": self.min_cluster_size,
                    "linkage": self.linkage,
                },
                f,
            )
        with self.session() as session:

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
                        continue
                    output_format = file.text_file.file_type
                    output_path = construct_output_path(
                        file.name,
                        file.relative_path,
                        output_directory,
                        output_format=output_format,
                    )
                    if output_format == "lab":
                        with mfa_open(output_path, "w") as f:
                            f.write(file.utterances[0].text)
                    else:
                        data = file.construct_transcription_tiers(original_text=True)
                        export_textgrid(
                            data,
                            output_path,
                            file.duration,
                            self.export_frame_shift,
                            output_format,
                        )
                    pbar.update(1)
