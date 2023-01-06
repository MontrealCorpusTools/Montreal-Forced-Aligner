"""Multiprocessing functionality for speaker diarization"""
from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import typing

import dataclassy
import hdbscan
import librosa
import numpy as np
import sqlalchemy
from numba import njit
from sklearn import metrics
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
from montreal_forced_aligner.config import GLOBAL_CONFIG, PLDA_DIMENSION
from montreal_forced_aligner.corpus.features import PldaModel
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import File, Job, SoundFile, Speaker, Utterance
from montreal_forced_aligner.utils import Stopped

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_logger = logging.getLogger("speechbrain.utils.torch_audio_backend")
        torch_logger.setLevel(logging.ERROR)
        torch_logger = logging.getLogger("speechbrain.utils.train_logger")
        torch_logger.setLevel(logging.ERROR)
        import torch
        from speechbrain.pretrained import EncoderClassifier
    FOUND_SPEECHBRAIN = True
except (ImportError, OSError) as e:
    print(e)
    FOUND_SPEECHBRAIN = False
    EncoderClassifier = None

if typing.TYPE_CHECKING:
    SpeakerCharacterType = typing.Union[str, int]

__all__ = [
    "PldaClassificationArguments",
    "PldaClassificationFunction",
]

logger = logging.getLogger("mfa")
M_LOG_2PI = 1.8378770664093454835606594728112


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class PldaClassificationArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.diarization.multiprocessing.PldaClassificationFunction`"""

    plda: PldaModel


# noinspection PyUnresolvedReferences
@dataclassy.dataclass(slots=True)
class SpeechbrainArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.diarization.multiprocessing.SpeechbrainClassificationFunction`"""

    cuda: bool
    cuda_batch_size: int


@njit(parallel=True)
def score_plda(
    train_ivectors: np.ndarray,
    test_ivectors: np.ndarray,
    psi: np.ndarray,
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
    train_ivectors = train_ivectors.astype("float64")
    test_ivectors = test_ivectors.astype("float64")
    psi = psi.astype("float64")
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

        loglikes[:, i] = -0.5 * (logdet_given + M_LOG_2PI * dim + (sqdiff_given @ variance_given))
    sqdiff_without = sqdiff**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * dim + (sqdiff_without @ variance_without)
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

    loglikes = -0.5 * (logdet_given + M_LOG_2PI * dim + (sqdiff_given @ variance_given.T))
    sqdiff_without = sqdiff**2
    loglike_without_class = -0.5 * (
        logdet_without + M_LOG_2PI * dim + (sqdiff_without @ variance_without)
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


def cluster_matrix(to_fit, cluster_type, metric="euclidean", **kwargs):
    from montreal_forced_aligner.config import GLOBAL_CONFIG

    os.environ["OMP_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.num_jobs}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.num_jobs}"
    os.environ["MKL_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.num_jobs}"

    if cluster_type == "affinity":
        c_labels = AffinityPropagation(affinity=metric, **kwargs).fit_predict(to_fit)
    elif cluster_type == "agglomerative":
        c_labels = AgglomerativeClustering(affinity=metric, **kwargs).fit_predict(to_fit)
    elif cluster_type == "spectral":
        c_labels = SpectralClustering(
            affinity=metric, n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "dbscan":
        c_labels = DBSCAN(
            metric=metric, n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "hdbscan":
        c_labels = hdbscan.HDBSCAN(
            metric=metric, core_dist_n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "optics":
        c_labels = OPTICS(
            metric=metric, n_jobs=GLOBAL_CONFIG.current_profile.num_jobs, **kwargs
        ).fit_predict(to_fit)
    elif cluster_type == "kmeans":
        c_labels = MiniBatchKMeans(**kwargs).fit_predict(to_fit)
    else:
        raise NotImplementedError(f"The cluster type '{cluster_type}' is not supported.")
    try:
        score = metrics.silhouette_score(to_fit, c_labels, metric=metric)
        logger.debug(f"Silhouette score (-1-1): {score}")
    except ValueError:
        logger.warning(
            "Only found one cluster, please adjust cluster parameters to generate more clusters."
        )
        raise

    os.environ["OMP_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"
    os.environ["MKL_NUM_THREADS"] = f"{GLOBAL_CONFIG.current_profile.blas_num_threads}"

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
            speaker_ivectors = np.empty((speaker_count, PLDA_DIMENSION))
            for i, (s_id, s_ivector) in enumerate(speakers):
                speaker_ids.append(s_id)
                speaker_ivectors[i, :] = s_ivector
            for u_id, u_ivector, speaker_id, file_id in utterances:
                affinity_matrix = score_plda(
                    speaker_ivectors, u_ivector[np.newaxis, :], self.plda.psi
                )
                speaker = speaker_ids[affinity_matrix.argmax(axis=1)[0]]
                yield u_id, file_id, speaker_id, speaker
        db_engine.dispose()


class SpeechbrainClassificationFunction(KaldiFunction):
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

    def __init__(self, args: SpeechbrainArguments):
        super().__init__(args)
        self.cuda = args.cuda
        self.cuda_batch_size = args.cuda_batch_size

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        run_opts = None
        if self.cuda:
            run_opts = {"device": "cuda"}
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(
                GLOBAL_CONFIG.current_profile.temporary_directory,
                "models",
                "SpeakerRecognition",
            ),
            run_opts=run_opts,
        )
        with Session(db_engine) as session:

            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            utterances = (
                session.query(
                    Utterance.id,
                    Utterance.file_id,
                    Utterance.begin,
                    Utterance.duration,
                    SoundFile.sound_file_path,
                )
                .join(Utterance.file)
                .join(File.sound_file)
                .filter(Utterance.job_id == job.id)
            )
            for u_id, file_id, begin, duration, sound_file_path in utterances:
                y, sr = librosa.load(
                    sound_file_path,
                    sr=16000,
                    mono=False,
                    offset=begin,
                    duration=duration,
                )
                y = torch.tensor(y)
                y = model.audio_normalizer(y, sr)
                y = y.unsqueeze(0)
                length = torch.tensor([1.0])
                (
                    out_prob,
                    score,
                    index,
                    text_lab,
                ) = model.classify_batch(y, length)
                new_speaker = text_lab[0]
                del out_prob
                del score
                del index
                del text_lab
                del y
                del length
                if self.cuda:
                    torch.cuda.empty_cache()
                yield u_id, file_id, new_speaker
        db_engine.dispose()


class SpeechbrainEmbeddingFunction(KaldiFunction):
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

    def __init__(self, args: SpeechbrainArguments):
        super().__init__(args)
        self.cuda = args.cuda
        self.cuda_batch_size = args.cuda_batch_size

    def _run(self) -> typing.Generator[typing.Tuple[int, int, int]]:
        """Run the function"""
        db_engine = sqlalchemy.create_engine(self.db_string)
        run_opts = None
        if self.cuda:
            run_opts = {"device": "cuda"}
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(
                GLOBAL_CONFIG.current_profile.temporary_directory,
                "models",
                "SpeakerRecognition",
            ),
            run_opts=run_opts,
        )
        with Session(db_engine) as session:

            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True))
                .filter(Job.id == self.job_name)
                .first()
            )
            utterances = (
                session.query(
                    Utterance.id,
                    Utterance.file_id,
                    Utterance.speaker_id,
                    Utterance.begin,
                    Utterance.duration,
                    SoundFile.sound_file_path,
                )
                .join(Utterance.file)
                .join(File.sound_file)
                .filter(Utterance.job_id == job.id)
            )
            for u_id, file_id, speaker_id, begin, duration, sound_file_path in utterances:
                y, sr = librosa.load(
                    sound_file_path,
                    sr=16000,
                    mono=False,
                    offset=begin,
                    duration=duration,
                )
                y = torch.tensor(y)
                y = model.audio_normalizer(y, sr)
                y = y.unsqueeze(0)
                length = torch.tensor([1.0])
                emb = model.encode_batch(y, length).cpu().numpy().squeeze(axis=1)
                yield u_id, file_id, speaker_id, emb[0]

                del emb
                if self.cuda:
                    gc.collect()
                    torch.cuda.empty_cache()

        db_engine.dispose()


class UtteranceFileLoader(mp.Process):
    def __init__(
        self,
        db_string: str,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
    ):
        super().__init__()
        self.db_string = db_string
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding

    def run(self) -> None:
        """
        Run the corpus loading job
        """
        db_engine = sqlalchemy.create_engine(self.db_string)
        with Session(db_engine) as session:
            try:
                utterances = (
                    session.query(
                        Utterance.id,
                        Utterance.file_id,
                        Utterance.speaker_id,
                        Utterance.begin,
                        Utterance.duration,
                        SoundFile.sound_file_path,
                    )
                    .join(Utterance.file)
                    .join(File.sound_file)
                )
                for u_id, file_id, speaker_id, begin, duration, sound_file_path in utterances:
                    if self.stopped.stop_check():
                        break
                    y, _ = librosa.load(
                        sound_file_path,
                        sr=16000,
                        mono=False,
                        offset=begin,
                        duration=duration,
                    )
                    self.return_q.put((u_id, file_id, speaker_id, y))
            except Exception as e:
                self.return_q.put(e)
            finally:
                self.finished_adding.stop()
        db_engine.dispose()
