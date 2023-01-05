"""Multiprocessing functionality for speaker diarization"""
from __future__ import annotations

import logging
import typing

import dataclassy
import hdbscan
import joblib
import numpy as np
import sqlalchemy
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    MiniBatchKMeans,
    SpectralClustering,
)
from sqlalchemy.orm import Session, joinedload

from montreal_forced_aligner.abc import KaldiFunction
from montreal_forced_aligner.corpus.features import PldaModel
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import Job, Speaker, Utterance

try:
    torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
    torch_logger.setLevel(logging.ERROR)
    torch_logger = logging.getLogger("speechbrain.utils.train_logger")
    torch_logger.setLevel(logging.ERROR)
    from speechbrain.pretrained import EncoderClassifier

    FOUND_SPEECHBRAIN = True
except (ImportError, OSError) as e:
    print(e)
    FOUND_SPEECHBRAIN = False
    EncoderClassifier = None

if typing.TYPE_CHECKING:
    SpeakerCharacterType = typing.Union[str, int]

__all__ = ["PldaClassificationArguments", "PldaClassificationFunction"]

logger = logging.getLogger("mfa")
M_LOG_2PI = 1.8378770664093454835606594728112


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PldaClassificationArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`"""

    plda: PldaModel


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class UtteranceDistanceArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.corpus.features.ComputeVadFunction`"""

    sparse_threshold: float


def score_plda(
    train_ivectors: np.ndarray,
    test_ivectors: np.ndarray,
    plda: PldaModel,
    normalize=True,
    distance=False,
) -> np.ndarray:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function
    Parameters
    ----------
    train_ivectors : numpy.ndarray
        Ivectors to compare test ivectors against against 1 X N X D
    test_ivectors : numpy.ndarray
        Ivectors to compare against training examples 1 X M X D
    Returns
    -------
    np.ndarray
        Affinity matrix, shape is number of train ivectors by the number of test ivectors (M X N)
    """
    psi = plda.psi
    mean = (psi) / (psi + 1.0)
    mean = mean.reshape(1, -1) * train_ivectors  # N X D , X[0]- Train ivectors
    # given class computation
    variance_given = 1.0 + psi / (psi + 1.0)
    logdet_given = np.sum(np.log(variance_given))
    variance_given = 1.0 / variance_given
    # without class computation
    variance_without = 1.0 + psi
    logdet_without = np.sum(np.log(variance_without))
    variance_without = 1.0 / variance_without

    sqdiff = test_ivectors  # ---- Test x-vectors
    num_train = train_ivectors.shape[0]
    num_test = test_ivectors.shape[0]
    dim = test_ivectors.shape[1]

    loglikes = np.empty((num_test, num_train))
    for i in range(num_train):
        sqdiff_given = sqdiff - mean[i]
        sqdiff_given = sqdiff_given**2

        loglikes[:, i] = -0.5 * (
            logdet_given + M_LOG_2PI * dim + np.matmul(sqdiff_given, variance_given)
        )
    sqdiff_without = sqdiff**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * dim + np.matmul(sqdiff_without, variance_without)
    )
    loglike_without_class = loglike_without_class.reshape(-1, 1)
    # loglike_given_class - N X N, loglike_without_class - N X1
    loglikes -= loglike_without_class  # N X N
    if normalize:
        # loglike_ratio -= np.min(loglike_ratio)
        loglikes /= np.max(loglikes)
    if distance:
        loglikes = np.max(loglikes) - loglikes
    return loglikes


def score_plda_train_counts(
    train_ivectors: np.ndarray,
    test_ivectors: np.ndarray,
    plda: PldaModel,
    normalize=True,
    distance=False,
    counts=None,
) -> np.ndarray:
    """
    Adapted from https://github.com/prachiisc/PLDA_scoring/blob/master/PLDA_scoring.py#L177
    Computes plda affinity matrix using Loglikelihood function
    Parameters
    ----------
    train_ivectors : numpy.ndarray
        Ivectors to compare test ivectors against against 1 X N X D
    test_ivectors : numpy.ndarray
        Ivectors to compare against training examples 1 X M X D
    Returns
    -------
    np.ndarray
        Affinity matrix, shape is number of train ivectors by the number of test ivectors (M X N)
    """
    if counts is None:
        counts = np.ones((train_ivectors.shape[0], 1))
    psi = plda.psi[np.newaxis, :]
    mean = (counts * psi) / (counts * psi + 1.0)
    mean = mean * train_ivectors  # N X D , X[0]- Train ivectors
    # given class computation
    variance_given = 1.0 + psi / (counts * psi + 1.0)
    logdet_given = np.sum(np.log(variance_given))
    variance_given = 1.0 / variance_given
    # without class computation
    variance_without = 1.0 + plda.psi
    logdet_without = np.sum(np.log(variance_without))
    variance_without = 1.0 / variance_without

    sqdiff = test_ivectors  # ---- Test x-vectors
    dim = test_ivectors.shape[1]

    # loglikes = np.empty((num_test, num_train))
    sqdiff_given = sqdiff - mean
    sqdiff_given = sqdiff_given**2

    loglikes = -0.5 * (logdet_given + M_LOG_2PI * dim + np.matmul(sqdiff_given, variance_given.T))
    sqdiff_without = sqdiff**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * dim + np.matmul(sqdiff_without, variance_without)
    )
    loglike_without_class = loglike_without_class.reshape(-1, 1)
    # loglike_given_class - N X N, loglike_without_class - N X1
    loglikes -= loglike_without_class  # N X N
    if normalize:
        # loglike_ratio -= np.min(loglike_ratio)
        loglikes /= np.max(loglikes)
    if distance:
        loglikes = np.max(loglikes) - loglikes
    return loglikes


def cluster_matrix(to_fit, cluster_type, precomputed=False, **kwargs):
    from montreal_forced_aligner.config import GLOBAL_CONFIG

    if precomputed and "affinity" not in kwargs and "metric" not in kwargs:
        if cluster_type in ["spectral", "affinity", "agglomerative"]:
            kwargs["affinity"] = "precomputed"
        else:
            kwargs["metric"] = "precomputed"

    if cluster_type == "affinity":
        if precomputed:
            to_fit = np.max(to_fit) - to_fit
        c_labels = AffinityPropagation(**kwargs).fit_predict(to_fit)
    elif cluster_type == "agglomerative":
        c_labels = AgglomerativeClustering(**kwargs).fit_predict(to_fit)
    elif cluster_type == "spectral":
        c_labels = SpectralClustering(
            n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "dbscan":
        c_labels = DBSCAN(
            metric="precomputed", n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "hdbscan":
        memory = joblib.Memory(location=kwargs.pop("location"))
        c_labels = hdbscan.HDBSCAN(
            core_dist_n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, memory=memory, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "optics":
        c_labels = OPTICS(n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs).fit_predict(
            to_fit
        )
    elif cluster_type == "kmeans":
        c_labels = MiniBatchKMeans(**kwargs).fit_predict(to_fit)
    else:
        raise NotImplementedError(f"The cluster type '{cluster_type} is not supported.")
    return c_labels


class PldaClassificationFunction(KaldiFunction):
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

    def __init__(self, args: PldaClassificationArguments):
        super().__init__(args)
        self.plda = args.plda

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session:

            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            utterances = (
                session.query(
                    Utterance.id, Utterance.ivector, Utterance.speaker_id, Utterance.file_id
                )
                .filter(Utterance.ignored == False)  # noqa
                .filter(Utterance.job_id == job.id)
                .order_by(Utterance.kaldi_id)
            )
            speakers = session.query(Speaker.id, Speaker.ivector).filter(
                Speaker.ivector != None  # noqa
            )
            speaker_count = speakers.count()
            speaker_ids = []
            speaker_ivectors = np.empty((speaker_count, self.plda.dimension), dtype="float32")
            for i, (s_id, s_ivector) in enumerate(speakers):
                speaker_ids.append(s_id)
                speaker_ivectors[i, :] = s_ivector.astype("float32")
            speaker_ivectors = np.array(speaker_ivectors, dtype="float32")
            for u_id, u_ivector, speaker_id, file_id in utterances:
                affinity_matrix = score_plda(
                    speaker_ivectors, u_ivector.astype("float32")[np.newaxis, :], self.plda
                )
                speaker = speaker_ids[affinity_matrix.argmax(axis=1)[0]]
                yield u_id, file_id, speaker_id, speaker
        db_engine.dispose()
