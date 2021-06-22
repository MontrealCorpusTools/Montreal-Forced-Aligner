import subprocess
import os
from .helper import run_mp, run_non_mp, thirdparty_binary
from ..helper import load_scp


def gmm_gselect_func(iteration, train_directory, config, feature_string, x):
    log_path = os.path.join(train_directory, 'log', 'gselect.{}.log'.format(x))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config['subsample']),
                                                 feature_string,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=log_file)

        gselect_proc = subprocess.Popen([thirdparty_binary('gmm-gselect'),
                                         '--n=' + str(config['num_gselect']),
                                         os.path.join(train_directory, '{}.dubm'.format(iteration)),
                                         'ark:-',
                                         'ark:' + os.path.join(train_directory, 'gselect.{}'.format(x))],
                                        stdin=subsample_feats_proc.stdout,
                                        stderr=log_file)
        gselect_proc.communicate()


def gmm_gselect(iteration, config, num_jobs):
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
    config : :class:`~aligner.config.DiagUbmConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation

    """
    directory = config.train_directory
    jobs = [(iteration, directory, config.ivector_options,
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x),
             x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(gmm_gselect_func, jobs, config.log_directory)
    else:
        run_non_mp(gmm_gselect_func, jobs, config.log_directory)


def acc_global_stats_func(train_directory, config, feature_string, x, iteration):
    log_path = os.path.join(train_directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config['subsample']),
                                                 feature_string,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=log_file)
        bin_name = 'gmm-global-acc-stats'
        mdl_path = os.path.join(train_directory, '{}.dubm'.format(iteration))
        gmm_global_acc_proc = subprocess.Popen([thirdparty_binary(bin_name),
                                                '--gselect=' + 'ark:' + os.path.join(train_directory,
                                                                                     'gselect.{}'.format(x)),
                                                mdl_path,
                                                'ark:-',
                                                os.path.join(train_directory, '{}.{}.acc'.format(iteration, x))],
                                               stderr=log_file,
                                               stdin=subsample_feats_proc.stdout)
        gmm_global_acc_proc.communicate()


def acc_global_stats(config, num_jobs, iteration):
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
    config : :class:`~aligner.config.DiagUbmConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    iteration : int
        Iteration to calculate stats for
    """
    directory = config.train_directory
    jobs = [(directory, config.ivector_options,
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x),
             x, iteration) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(acc_global_stats_func, jobs, config.log_directory)
    else:
        run_non_mp(acc_global_stats_func, jobs, config.log_directory)


def gauss_to_post_func(train_directory, config, feature_string, x):
    modified_posterior_scale = config['posterior_scale'] * config['subsample']
    log_path = os.path.join(train_directory, 'log', 'post.{}.log'.format(x))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config['subsample']),
                                                 feature_string,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=log_file)
        gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                     '--n=' + str(config['num_gselect']),
                                                     '--min-post=' + str(config['min_post']),
                                                     os.path.join(train_directory, 'final.dubm'),
                                                     'ark:-',
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stdin=subsample_feats_proc.stdout,
                                                    stderr=log_file)
        scale_post_proc = subprocess.Popen([thirdparty_binary('scale-post'),
                                            'ark:-',
                                            str(modified_posterior_scale),
                                            'ark:' + os.path.join(train_directory, 'post.{}'.format(x))],
                                           stdin=gmm_global_get_post_proc.stdout,
                                           stderr=log_file)
        scale_post_proc.communicate()


def gauss_to_post(config, num_jobs):
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
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """
    func = gauss_to_post_func
    directory = config.train_directory
    jobs = [(config.train_directory, config.ivector_options,
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x),
             x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(func, jobs, config.log_directory)
    else:
        run_non_mp(func, jobs, config.log_directory)


def acc_ivector_stats_func(train_directory, config, feature_string, x, iteration):
    log_path = os.path.join(train_directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config['subsample']),
                                                 feature_string,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=log_file)
        acc_stats_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-acc-stats'),
                                           '--num-threads=1',
                                           os.path.join(train_directory, '{}.ie'.format(iteration)),
                                           'ark:-',
                                           'ark:' + os.path.join(train_directory, 'post.{}'.format(x)),
                                           os.path.join(train_directory, 'accinit.{}.{}'.format(iteration, x))],
                                          stdin=subsample_feats_proc.stdout,
                                          stderr=log_file)
        acc_stats_proc.communicate()


def acc_ivector_stats(config, num_jobs, iteration):
    """
    Multiprocessing function that calculates i-vector extractor stats

    See:

    - http://kaldi-asr.org/doc/ivector-extractor-acc-stats_8cc.html
    - http://kaldi-asr.org/doc/ivector-extractor-sum-accs_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/train_ivector_extractor.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    iteration : int
        Iteration to calculate stats for
    """
    directory = config.train_directory
    jobs = [(config.train_directory, config.ivector_options,
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x),
             x, iteration) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(acc_ivector_stats_func, jobs, config.log_directory)
    else:
        run_non_mp(acc_ivector_stats_func, jobs, config.log_directory)

    accinits = [os.path.join(config.train_directory, 'accinit.{}.{}'.format(iteration, j)) for j in range(num_jobs)]
    log_path = os.path.join(config.train_directory, 'log', 'sum_acc.{}.log'.format(iteration))
    with open(log_path, 'w', encoding='utf8') as log_file:
        sum_accs_proc = subprocess.Popen([thirdparty_binary('ivector-extractor-sum-accs'),
                                          '--parallel=true']
                                         + accinits
                                         + [os.path.join(config.train_directory, 'acc.{}'.format(iteration))],
                                         stderr=log_file)

        sum_accs_proc.communicate()
    # clean up
    for p in accinits:
        os.remove(p)


def extract_ivectors_func(directory, split_directory, config, feature_string, sil_phones, job_id, align_directory=None):
    """
    Parameters
    ----------
    config : :class:`~aligner.trainers.IvectorExtractorTrainer`
        Configuration object for training
    job_id : int
        Job identifier
    """
    use_align = False
    ali_path = None
    if align_directory is not None:
        ali_path = os.path.join(align_directory, 'ali.{}'.format(job_id))
        use_align = os.path.exists(ali_path)

    log_dir = os.path.join(directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    ivector_mdl = os.path.join(directory, 'final.ie')
    log_path = os.path.join(directory, 'log', 'extract_ivectors.{}.log'.format(job_id))
    ivectors_path = os.path.join(directory, 'ivectors.{}'.format(job_id))
    weight_path = os.path.join(directory, 'weight.{}'.format(job_id))
    mdl_path = os.path.join(directory, 'final.mdl')
    spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_id))

    silence_weight = 0.0
    posterior_scale = 0.1
    max_count = 100
    with open(log_path, 'w', encoding='utf8') as log_file:
        if use_align:
            ali_to_post_proc = subprocess.Popen([thirdparty_binary('ali-to-post'),
                                                 'ark:' + ali_path, 'ark:-'],
                                                stderr=log_file,
                                                stdout=subprocess.PIPE)
            weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                    str(silence_weight),
                                                    sil_phones,
                                                    mdl_path,
                                                    'ark:-', 'ark:-'],
                                                   stderr=log_file,
                                                   stdin=ali_to_post_proc.stdout,
                                                   stdout=subprocess.PIPE)
            post_to_weight_proc = subprocess.Popen([thirdparty_binary('post-to-weights'),
                                                    'ark:-', 'ark:' + weight_path],
                                                   stderr=log_file,
                                                   stdin=weight_silence_proc.stdout)
            post_to_weight_proc.communicate()

        gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('gmm-global-get-post'),
                                                     '--n=' + str(config['num_gselect']),
                                                     '--min-post=' + str(config['min_post']),
                                                     os.path.join(directory, 'final.dubm'),
                                                     feature_string,
                                                     'ark:-'],
                                                    stdout=subprocess.PIPE,
                                                    stderr=log_file)
        if use_align:
            weight_proc = subprocess.Popen([thirdparty_binary('weight-post'),
                                            'ark:-', 'ark,s,cs:' + weight_path, 'ark:-'],
                                           stdin=gmm_global_get_post_proc.stdout,
                                           stdout=subprocess.PIPE, stderr=log_file)
            extract_in = weight_proc.stdout
        else:
            extract_in = gmm_global_get_post_proc.stdout
        extract_proc = subprocess.Popen([thirdparty_binary('ivector-extract'),
                                         '--acoustic-weight={}'.format(posterior_scale),
                                         '--compute-objf-change=true',
                                         '--max-count={}'.format(max_count),
                                         ivector_mdl,
                                         feature_string,
                                         'ark,s,cs:-',
                                         'ark,t:' + ivectors_path],
                                        stderr=log_file,
                                        stdin=extract_in)
        extract_proc.communicate()


def extract_ivectors(directory, split_directory, config, num_jobs, align_directory=None):
    """
    Multiprocessing function that extracts i-vectors.

    See:

    - http://kaldi-asr.org/doc/ivector-extract-online2_8cc.html
    - http://kaldi-asr.org/doc/copy-feats_8cc.html

    for more details
    on the Kaldi binary this runs.

    Also see https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/online/nnet2/extract_ivectors_online.sh
    for the original bash script that this function was based on.

    Parameters
    ----------
    config : :class:`~montreal_forced_aligner.config.iVectorExtractorConfig`
        Configuration object for training
    num_jobs : int
        The number of processes to use in calculation
    """

    log_dir = os.path.join(directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    data_directory = split_directory
    func = extract_ivectors_func
    try:
        csl = config.dictionary.silence_csl
    except AttributeError:
        csl = None
    jobs = [(directory, config.corpus.split_directory(), config.ivector_options,
         config.feature_config.construct_feature_proc_string(data_directory, directory, x),
             csl,
             x, align_directory) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(func, jobs, log_dir)
    else:
        run_non_mp(func, jobs, log_dir)


def get_initial_segmentation(frames, frame_shift):
    segs = []
    cur_seg = None
    silent_frames = 0
    non_silent_frames = 0
    for i, f in enumerate(frames):
        if int(f) > 0:
            non_silent_frames += 1
            if cur_seg is None:
                cur_seg = {'begin': i * frame_shift}
        else:
            silent_frames += 1
            if cur_seg is not None:
                cur_seg['end'] = (i - 1) * frame_shift
                segs.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        cur_seg['end'] = len(frames) * frame_shift
        segs.append(cur_seg)
    total = non_silent_frames + silent_frames
    return segs


def merge_segments(segments, min_pause_duration, max_segment_length, snap_boundary_threshold):
    merged_segs = []
    for s in segments:
        if not merged_segs or s['begin'] > merged_segs[-1]['end'] + min_pause_duration or \
                s['end'] - merged_segs[-1]['begin'] > max_segment_length:
            if s['end'] - s['begin'] > min_pause_duration:
                if merged_segs and snap_boundary_threshold:
                    boundary_gap = s['begin'] - merged_segs[-1]['end']
                    if boundary_gap < snap_boundary_threshold:
                        half_boundary = boundary_gap / 2
                    else:
                        half_boundary = snap_boundary_threshold / 2
                    merged_segs[-1]['end'] += half_boundary
                    s['begin'] -= half_boundary

                merged_segs.append(s)
        else:
            merged_segs[-1]['end'] = s['end']
    return merged_segs


def segment_vad_func(directory, job_name, config):
    vad_path = os.path.join(directory, 'vad.{}.scp'.format(job_name))
    vad_segments_path = os.path.join(directory, 'vad_segments.{}.scp'.format(job_name))

    vad = load_scp(vad_path, data_type=int)
    with open(vad_segments_path, 'w', encoding='utf8') as out_file:
        for recording, frames in vad.items():
            initial_segments = get_initial_segmentation(frames, config['frame_shift'])
            merged = merge_segments(initial_segments, config['min_pause_duration'], config['max_segment_length'], config['snap_boundary_threshold'])
            for seg in merged:
                start = seg['begin']
                end = seg['end']
                new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                    utt_id=recording, s=int(round(100 * start)),
                    e=int(round(100 * end)))
                out_file.write("{utt_id} {recording} {s:.3f} {e:.3f}\n".format(utt_id=new_utt, recording=recording,
                                                                               s=start, e=end))


def segment_vad(corpus, config):
    split_dir = corpus.split_directory()
    log_directory = os.path.join(split_dir, 'log')
    num_jobs = corpus.num_jobs
    jobs = [(split_dir, x, config.segmentation_options) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(segment_vad_func, jobs, log_directory)
    else:
        run_non_mp(segment_vad_func, jobs, log_directory)


def classify_speakers_func(directory, job_name):
    from ..helper import load_scp, save_scp
    from joblib import load
    import numpy as np
    import warnings
    from collections import defaultdict
    mdl_path = os.path.join(directory, 'speaker_classifier.mdl')
    labels_path = os.path.join(directory, 'speaker_labels.txt')
    speakers = {}
    with open(labels_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            speaker, speak_ind = line
            speakers[int(speak_ind)] = speaker
    ivectors_path = os.path.join(directory, 'ivectors.{}'.format(job_name))
    spk2utt_path = os.path.join(directory, 'spk2utt.{}'.format(job_name))
    utt2spk_path = os.path.join(directory, 'utt2spk.{}'.format(job_name))
    ivec = load_scp(ivectors_path)
    x = []
    for utt, ivector in ivec.items():
        ivector = [float(x) for x in ivector]
        x.append(ivector)
    x = np.array(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = load(mdl_path)
    y = clf.predict(x)
    speak_utt_mapping = defaultdict(list)
    utt_speak_mapping = {}
    for i, utt in enumerate(ivec.keys()):
        speak_ind = y[i]
        speaker = speakers[speak_ind]
        speak_utt_mapping[speaker].append(utt)
        utt_speak_mapping[utt] = speaker
    save_scp(([k, v] for k,v in speak_utt_mapping.items()), spk2utt_path)
    save_scp(([k, v] for k,v in utt_speak_mapping.items()), utt2spk_path)


def classify_speakers(directory, config, num_jobs):
    log_directory = os.path.join(directory, 'log')
    jobs = [(directory, x) for x in range(num_jobs)]

    if config.use_mp:
        run_mp(classify_speakers_func, jobs, log_directory)
    else:
        run_non_mp(classify_speakers_func, jobs, log_directory)