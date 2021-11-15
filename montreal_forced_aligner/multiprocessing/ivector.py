"""
Ivector extractor functions
---------------------------


"""
from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Dict, List, Union

from ..abc import MetaDict
from ..helper import load_scp
from ..utils import thirdparty_binary
from .helper import run_mp, run_non_mp

if TYPE_CHECKING:
    from ..abc import IvectorExtractor
    from ..corpus.classes import File, Speaker, Utterance  # noqa
    from ..segmenter import SegmentationType, Segmenter
    from ..trainers.ivector_extractor import IvectorExtractorTrainer


__all__ = [
    "gmm_gselect",
    "acc_global_stats",
    "gauss_to_post",
    "acc_ivector_stats",
    "extract_ivectors",
    "get_initial_segmentation",
    "merge_segments",
    "segment_vad",
    "segment_vad_func",
    "gmm_gselect_func",
    "gauss_to_post_func",
    "acc_global_stats_func",
    "acc_ivector_stats_func",
    "extract_ivectors_func",
]


def gmm_gselect_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    dubm_path: str,
    gselect_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for running gmm-gselect

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    ivector_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for ivector extractor training
    dubm_path: str
        Path to the DUBM file
    gselect_paths: Dict[str, str]
        Dictionary of gselect archives per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            gselect_path = gselect_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={ivector_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            gselect_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-gselect"),
                    f"--n={ivector_options['num_gselect']}",
                    dubm_path,
                    "ark:-",
                    f"ark:{gselect_path}",
                ],
                stdin=subsample_feats_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            gselect_proc.communicate()


def gmm_gselect(trainer: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that stores Gaussian selection indices on disk

    See:

    - http://kaldi-asr.org/doc/gmm-gselect_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_diag_ubm.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    trainer : :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
        Ivector Extractor Trainer
    """
    jobs = [x.gmm_gselect_arguments(trainer) for x in trainer.corpus.jobs]
    if trainer.use_mp:
        run_mp(gmm_gselect_func, jobs, trainer.working_log_directory)
    else:
        run_non_mp(gmm_gselect_func, jobs, trainer.working_log_directory)


def acc_global_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    gselect_paths: Dict[str, str],
    acc_paths: Dict[str, str],
    dubm_path: str,
) -> None:
    """
    Multiprocessing function for accumulating global model stats

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    ivector_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for ivector extractor training
    gselect_paths: Dict[str, str]
        Dictionary of gselect archives per dictionary name
    acc_paths: Dict[str, str]
        Dictionary of accumulated stats files per dictionary name
    dubm_path: str
        Path to the DUBM file
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            gselect_path = gselect_paths[dict_name]
            acc_path = acc_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={ivector_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            gmm_global_acc_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-acc-stats"),
                    f"--gselect=ark:{gselect_path}",
                    dubm_path,
                    "ark:-",
                    acc_path,
                ],
                stderr=log_file,
                stdin=subsample_feats_proc.stdout,
                env=os.environ,
            )
            gmm_global_acc_proc.communicate()


def acc_global_stats(trainer: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that accumulates global GMM stats

    See:

    - http://kaldi-asr.org/doc/gmm-global-acc-stats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_diag_ubm.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    trainer : :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
        Ivector Extractor Trainer
    """
    jobs = [x.acc_global_stats_arguments(trainer) for x in trainer.corpus.jobs]
    if trainer.use_mp:
        run_mp(acc_global_stats_func, jobs, trainer.working_log_directory)
    else:
        run_non_mp(acc_global_stats_func, jobs, trainer.working_log_directory)

    # Don't remove low-count Gaussians till the last tier,
    # or gselect info won't be valid anymore
    if trainer.iteration < trainer.ubm_num_iterations:
        opt = "--remove-low-count-gaussians=false"
    else:
        opt = f"--remove-low-count-gaussians={trainer.ubm_remove_low_count_gaussians}"
    log_path = os.path.join(trainer.working_log_directory, f"update.{trainer.iteration}.log")
    with open(log_path, "w") as log_file:
        acc_files = []
        for j in jobs:
            acc_files.extend(j.acc_paths.values())
        sum_proc = subprocess.Popen(
            [thirdparty_binary("gmm-global-sum-accs"), "-"] + acc_files,
            stderr=log_file,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        gmm_global_est_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-global-est"),
                opt,
                f"--min-gaussian-weight={trainer.ubm_min_gaussian_weight}",
                trainer.current_dubm_path,
                "-",
                trainer.next_dubm_path,
            ],
            stderr=log_file,
            stdin=sum_proc.stdout,
            env=os.environ,
        )
        gmm_global_est_proc.communicate()
        # Clean up
        if not trainer.debug:
            for p in acc_files:
                os.remove(p)


def gauss_to_post_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    post_paths: Dict[str, str],
    dubm_path: str,
):
    """
    Multiprocessing function to get posteriors during UBM training

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    ivector_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for ivector extractor training
    post_paths: Dict[str, str]
        Dictionary of posterior archives per dictionary name
    dubm_path: str
        Path to the DUBM file
    """
    modified_posterior_scale = ivector_options["posterior_scale"] * ivector_options["subsample"]
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            post_path = post_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={ivector_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            gmm_global_get_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-global-get-post"),
                    f"--n={ivector_options['num_gselect']}",
                    f"--min-post={ivector_options['min_post']}",
                    dubm_path,
                    "ark:-",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stdin=subsample_feats_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            scale_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("scale-post"),
                    "ark:-",
                    str(modified_posterior_scale),
                    f"ark:{post_path}",
                ],
                stdin=gmm_global_get_post_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            scale_post_proc.communicate()


def gauss_to_post(trainer: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that does Gaussian selection and posterior extraction

    See:

    - http://kaldi-asr.org/doc/gmm-global-get-post_8cc.html
    - http://kaldi-asr.org/doc/scale-post_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    trainer: :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
        Ivector Extractor Trainer
    """
    jobs = [x.gauss_to_post_arguments(trainer) for x in trainer.corpus.jobs]
    if trainer.use_mp:
        run_mp(gauss_to_post_func, jobs, trainer.working_log_directory)
    else:
        run_non_mp(gauss_to_post_func, jobs, trainer.working_log_directory)


def acc_ivector_stats_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    ie_path: str,
    post_paths: Dict[str, str],
    acc_init_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function that accumulates stats for ivector training

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        PronunciationDictionary of feature strings per dictionary name
    ivector_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for ivector extractor training
    ie_path: str
        Path to the ivector extractor file
    post_paths: Dict[str, str]
        PronunciationDictionary of posterior archives per dictionary name
    acc_init_paths: Dict[str, str]
        PronunciationDictionary of accumulated stats files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            post_path = post_paths[dict_name]
            acc_init_path = acc_init_paths[dict_name]
            subsample_feats_proc = subprocess.Popen(
                [
                    thirdparty_binary("subsample-feats"),
                    f"--n={ivector_options['subsample']}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            acc_stats_proc = subprocess.Popen(
                [
                    thirdparty_binary("ivector-extractor-acc-stats"),
                    "--num-threads=1",
                    ie_path,
                    "ark:-",
                    f"ark:{post_path}",
                    acc_init_path,
                ],
                stdin=subsample_feats_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            acc_stats_proc.communicate()


def acc_ivector_stats(trainer: IvectorExtractorTrainer) -> None:
    """
    Multiprocessing function that calculates job_name-vector extractor stats

    See:

    - http://kaldi-asr.org/doc/ivector-extractor-acc-stats_8cc.html
    - http://kaldi-asr.org/doc/ivector-extractor-sum-accs_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    trainer: :class:`~montreal_forced_aligner.trainers.IvectorExtractorTrainer`
        Ivector Extractor Trainer
    """

    jobs = [x.ivector_acc_stats_arguments(trainer) for x in trainer.corpus.jobs]
    if trainer.use_mp:
        run_mp(acc_ivector_stats_func, jobs, trainer.working_log_directory)
    else:
        run_non_mp(acc_ivector_stats_func, jobs, trainer.working_log_directory)

    log_path = os.path.join(trainer.working_log_directory, f"sum_acc.{trainer.iteration}.log")
    acc_path = os.path.join(trainer.working_directory, f"acc.{trainer.iteration}")
    with open(log_path, "w", encoding="utf8") as log_file:
        accinits = []
        for j in jobs:
            accinits.extend(j.acc_init_paths.values())
        sum_accs_proc = subprocess.Popen(
            [thirdparty_binary("ivector-extractor-sum-accs"), "--parallel=true"]
            + accinits
            + [acc_path],
            stderr=log_file,
            env=os.environ,
        )

        sum_accs_proc.communicate()
    # clean up
    for p in accinits:
        os.remove(p)
        # Est extractor
    log_path = os.path.join(trainer.working_log_directory, f"update.{trainer.iteration}.log")
    with open(log_path, "w") as log_file:
        extractor_est_proc = subprocess.Popen(
            [
                thirdparty_binary("ivector-extractor-est"),
                f"--num-threads={trainer.corpus.num_jobs}",
                f"--gaussian-min-count={trainer.gaussian_min_count}",
                trainer.current_ie_path,
                os.path.join(trainer.working_directory, f"acc.{trainer.iteration}"),
                trainer.next_ie_path,
            ],
            stderr=log_file,
            env=os.environ,
        )
        extractor_est_proc.communicate()


def extract_ivectors_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    ivector_options: MetaDict,
    ali_paths: Dict[str, str],
    ie_path: str,
    ivector_paths: Dict[str, str],
    weight_paths: Dict[str, str],
    model_path: str,
    dubm_path: str,
) -> None:
    """
    Multiprocessing function for extracting ivectors

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


def extract_ivectors(ivector_extractor: IvectorExtractor) -> None:
    """
    Multiprocessing function that extracts job_name-vectors.

    See:

    - http://kaldi-asr.org/doc/ivector-extract-online2_8cc.html
    - http://kaldi-asr.org/doc/copy-feats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/extract_ivectors_online.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    ivector_extractor: IvectorExtractor
        Ivector extractor
    """

    log_dir = ivector_extractor.log_directory
    os.makedirs(log_dir, exist_ok=True)

    jobs = [x.extract_ivector_arguments(ivector_extractor) for x in ivector_extractor.corpus.jobs]
    if ivector_extractor.use_mp:
        run_mp(extract_ivectors_func, jobs, log_dir)
    else:
        run_non_mp(extract_ivectors_func, jobs, log_dir)


def get_initial_segmentation(frames: List[Union[int, str]], frame_shift: int) -> SegmentationType:
    """
    Compute initial segmentation over voice activity

    Parameters
    ----------
    frames: List[Union[int, str]]
        List of frames with VAD output
    frame_shift: int
        Frame shift of features in ms

    Returns
    -------
    SegmentationType
        Initial segmentation
    """
    segs = []
    cur_seg = None
    silent_frames = 0
    non_silent_frames = 0
    for i, f in enumerate(frames):
        if int(f) > 0:
            non_silent_frames += 1
            if cur_seg is None:
                cur_seg = {"begin": i * frame_shift}
        else:
            silent_frames += 1
            if cur_seg is not None:
                cur_seg["end"] = (i - 1) * frame_shift
                segs.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        cur_seg["end"] = len(frames) * frame_shift
        segs.append(cur_seg)
    return segs


def merge_segments(
    segments: SegmentationType,
    min_pause_duration: float,
    max_segment_length: float,
    snap_boundary_threshold: float,
) -> SegmentationType:
    """
    Merge segments together

    Parameters
    ----------
    segments: SegmentationType
        Initial segments
    min_pause_duration: float
        Minimum amount of silence time to mark an utterance boundary
    max_segment_length: float
        Maximum length of segments before they're broken up
    snap_boundary_threshold:
        Boundary threshold to snap boundaries together

    Returns
    -------
    SegmentationType
        Merged segments
    """
    merged_segs = []
    for s in segments:
        if (
            not merged_segs
            or s["begin"] > merged_segs[-1]["end"] + min_pause_duration
            or s["end"] - merged_segs[-1]["begin"] > max_segment_length
        ):
            if s["end"] - s["begin"] > min_pause_duration:
                if merged_segs and snap_boundary_threshold:
                    boundary_gap = s["begin"] - merged_segs[-1]["end"]
                    if boundary_gap < snap_boundary_threshold:
                        half_boundary = boundary_gap / 2
                    else:
                        half_boundary = snap_boundary_threshold / 2
                    merged_segs[-1]["end"] += half_boundary
                    s["begin"] -= half_boundary

                merged_segs.append(s)
        else:
            merged_segs[-1]["end"] = s["end"]
    return merged_segs


def segment_vad_func(
    dictionaries: List[str],
    vad_paths: Dict[str, str],
    segmentation_options: MetaDict,
) -> Dict[str, Utterance]:
    """
    Multiprocessing function to generate segments from VAD output

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    vad_paths: Dict[str, str]
        Dictionary of VAD archives per dictionary name
    segmentation_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for segmentation
    """

    utterances = {}
    from ..corpus.classes import File, Speaker, Utterance  # noqa

    speaker = Speaker("speech")
    for dict_name in dictionaries:
        vad_path = vad_paths[dict_name]

        vad = load_scp(vad_path, data_type=int)
        for recording, frames in vad.items():
            file = File(recording)
            initial_segments = get_initial_segmentation(
                frames, segmentation_options["frame_shift"]
            )
            merged = merge_segments(
                initial_segments,
                segmentation_options["min_pause_duration"],
                segmentation_options["max_segment_length"],
                segmentation_options["snap_boundary_threshold"],
            )
            for seg in merged:
                utterances[recording] = Utterance(
                    speaker, file, begin=seg["begin"], end=seg["end"], text="speech"
                )
    return utterances


def segment_vad(segmenter: Segmenter) -> None:
    """
    Run segmentation based off of VAD

    Parameters
    ----------
    segmenter: :class:`~montreal_forced_aligner.segmenter.Segmenter`
        Segmenter
    """

    from ..corpus.classes import Speaker  # noqa

    jobs = [x.segments_vad_arguments(segmenter) for x in segmenter.corpus.jobs]
    if segmenter.segmentation_config.use_mp:
        segment_info = run_mp(
            segment_vad_func, jobs, segmenter.corpus.features_log_directory, True
        )
    else:
        segment_info = run_non_mp(
            segment_vad_func, jobs, segmenter.corpus.features_log_directory, True
        )
    for j in segmenter.corpus.jobs:
        for old_utt, utterance in segment_info[j.name].items():
            old_utt = segmenter.corpus.utterances[old_utt]
            file = old_utt.file
            if segmenter.corpus.no_speakers:
                if utterance.speaker_name not in segmenter.corpus.speakers:
                    segmenter.corpus.speakers[utterance.speaker_name] = Speaker(
                        utterance.speaker_name
                    )
                speaker = segmenter.corpus.speakers[utterance.speaker_name]
            else:
                speaker = old_utt.speaker
            utterance.file = file
            utterance.set_speaker(speaker)
            segmenter.corpus.add_utterance(utterance)
    utterance_ids = [x.name for x in segmenter.corpus.utterances.values() if x.begin is None]
    for u in utterance_ids:
        segmenter.corpus.delete_utterance(u)
