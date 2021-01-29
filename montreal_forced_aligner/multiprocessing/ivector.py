import subprocess
import os
from .helper import run_mp, run_non_mp, thirdparty_binary


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


def gmm_gselect(iteration, config, num_jobs, vad=True):
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
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x, voiced=vad),
             x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(gmm_gselect_func, jobs, config.log_directory)
    else:
        run_non_mp(gmm_gselect_func, jobs, config.log_directory)


def acc_global_stats_func(train_directory, config, feature_string, x, iteration, full=False):
    log_path = os.path.join(train_directory, 'log', 'acc.{}.{}.log'.format(iteration, x))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config['subsample']),
                                                 feature_string,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=log_file)
        if full:
            bin_name = 'fgmm-global-acc-stats'
            mdl_path = os.path.join(train_directory, '{}.ubm'.format(iteration))
        else:
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


def acc_global_stats(config, num_jobs, iteration, full=False, vad=True):
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
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x, voiced=vad),
             x, iteration, full) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(acc_global_stats_func, jobs, config.log_directory)
    else:
        run_non_mp(acc_global_stats_func, jobs, config.log_directory)


def gauss_to_post_vad_func(train_directory, config, feature_string, x):
    modified_posterior_scale = config['posterior_scale'] * config['subsample']
    log_path = os.path.join(train_directory, 'log', 'post.{}.log'.format(x))
    gselect_path = os.path.join(train_directory, 'gselect.{}'.format(x))
    with open(log_path, 'w', encoding='utf8') as log_file:
        subsample_feats_proc = subprocess.Popen([thirdparty_binary('subsample-feats'),
                                                 '--n=' + str(config['subsample']),
                                                 feature_string,
                                                 'ark:-'],
                                                stdout=subprocess.PIPE,
                                                stderr=log_file)
        gmm_global_get_post_proc = subprocess.Popen([thirdparty_binary('fgmm-global-gselect-to-post'),
                                                     '--min-post=' + str(config['min_post']),
                                                     os.path.join(train_directory, 'final.ubm'),
                                                     'ark:-',
                                                     'ark:' + gselect_path,
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


def gauss_to_post(config, num_jobs, vad=True):
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
    if vad:
        func = gauss_to_post_vad_func
    else:
        func = gauss_to_post_func
    directory = config.train_directory
    jobs = [(config.train_directory, config.ivector_options,
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x, voiced=vad),
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


def acc_ivector_stats(config, num_jobs, iteration, vad=True):
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
             config.feature_config.construct_feature_proc_string(config.data_directory, directory, x, voiced=vad),
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


def extract_ivectors_vad_func(directory, config, feature_string, job_id):
    """

    Parameters
    ----------
    config : :class:`~aligner.trainers.IvectorExtractorTrainer`
        Configuration object for training
    job_id : int
        Job identifier
    """

    log_dir = os.path.join(directory, 'log')
    ivector_mdl = os.path.join(directory, 'final.ie')
    ubm_path = os.path.join(directory, 'final.ubm')
    dubm_path = os.path.join(directory, 'final.dubm')
    log_path = os.path.join(log_dir, 'extract_ivectors.{}.log'.format(job_id))
    ivector_scp_path = os.path.join(directory, 'ivectors.{}.scp'.format(job_id))
    ivector_ark_path = os.path.join(directory, 'ivectors.{}.ark'.format(job_id))

    with open(log_path, 'w', encoding='utf8') as log_file:
        if not os.path.exists(dubm_path):
            create_dubm_proc = subprocess.Popen([thirdparty_binary('fgmm-global-to-gmm'),
                                                 ubm_path, dubm_path], stderr=log_file)
            create_dubm_proc.communicate()

        gselect_proc = subprocess.Popen([thirdparty_binary('gmm-gselect'),
                                         '--n=' + str(config['num_gselect']),
                                         dubm_path,
                                         feature_string,
                                         'ark:-'],
                                        stdout=subprocess.PIPE,
                                        stderr=log_file)

        post_proc = subprocess.Popen([thirdparty_binary('fgmm-global-gselect-to-post'),
                                      '--min-post={}'.format(config['min_post']),
                                      ubm_path, feature_string,
                                      'ark,s,cs:-', 'ark:-'],
                                     stdin=gselect_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        scale_proc = subprocess.Popen([thirdparty_binary('scale-post'),
                                       'ark:-', str(config['posterior_scale']), 'ark:-'],
                                      stdin=post_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        extract_proc = subprocess.Popen([thirdparty_binary('ivector-extract'),
                                         '--verbose=2', ivector_mdl,
                                         feature_string,
                                         'ark,s,cs:-',
                                         'ark,scp,t:{},{}'.format(ivector_ark_path, ivector_scp_path)],
                                        stdin=scale_proc.stdout, stderr=log_file)
        extract_proc.communicate()


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


def extract_ivectors(directory, split_directory, config, num_jobs, vad=True, align_directory=None):
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
    if vad:
        data_directory = os.path.join(split_directory, 'subsegments')
        func = extract_ivectors_vad_func
        jobs = [(directory, config.ivector_options,
             config.feature_config.construct_feature_proc_string(data_directory, directory, x, voiced=vad),
                 x, align_directory) for x in range(num_jobs)]
    else:
        data_directory = split_directory
        func = extract_ivectors_func
        try:
            csl = config.dictionary.silence_csl
        except AttributeError:
            csl = None
        jobs = [(directory, config.corpus.split_directory(), config.ivector_options,
             config.feature_config.construct_feature_proc_string(data_directory, directory, x, voiced=vad),
                 csl,
                 x, align_directory) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(func, jobs, log_dir)
    else:
        run_non_mp(func, jobs, log_dir)


def transform_ivectors_func(directory, features_path, job_id):
    log_dir = os.path.join(directory, 'log')
    mean_path = os.path.join(directory, 'mean.vec')
    transform_path = os.path.join(directory, 'trans.mat')
    log_path = os.path.join(log_dir, 'transform_ivectors.{}.log'.format(job_id))
    transform_ivector_scp_path = os.path.join(directory, 'ivectors_transformed.{}.scp'.format(job_id))
    transform_ivector_ark_path = os.path.join(directory, 'ivectors_transformed.{}.ark'.format(job_id))

    with open(log_path, 'w', encoding='utf8') as log_file:
        subtract_mean_proc = subprocess.Popen([thirdparty_binary('ivector-subtract-global-mean'),
                                               mean_path, 'scp:'+features_path, 'ark:-'],
                                              stdout=subprocess.PIPE, stderr=log_file)
        transform_proc = subprocess.Popen([thirdparty_binary('transform-vec'),
                                               transform_path, 'ark:-', 'ark:-'],
                                          stdin=subtract_mean_proc.stdout,
                                              stdout=subprocess.PIPE, stderr=log_file)
        normalize_proc = subprocess.Popen([thirdparty_binary('ivector-normalize-length'),
                                               'ark:-',
                                           'ark,scp:{},{}'.format(transform_ivector_ark_path, transform_ivector_scp_path)],
                                          stdin=transform_proc.stdout, stderr=log_file)
        normalize_proc.communicate()


def transform_ivectors(directory, config, num_jobs):
    log_dir = os.path.join(directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    feat_name = 'ivectors.{}.scp'
    jobs = [(directory,
             os.path.join(directory, feat_name.format(x)),
             x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(transform_ivectors_func, jobs, log_dir)
    else:
        run_non_mp(transform_ivectors_func, jobs, log_dir)


def score_plda_func(directory, split_directory, config, features_path, job_name):
    log_directory = os.path.join(directory, 'log')
    plda_path = os.path.join(directory, 'plda')
    log_path = os.path.join(log_directory, 'score_plda.{}.log'.format(job_name))
    reco2utt_path = os.path.join(split_directory, 'subsegments', 'reco2utt.{}'.format(job_name))
    scores_ark_path = os.path.join(directory, 'scores.{}.ark'.format(job_name))
    scores_scp_path = os.path.join(directory, 'scores.{}.scp'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        score_plda_proc = subprocess.Popen([thirdparty_binary('ivector-plda-scoring-dense'),
                                            '--target-energy={}'.format(config['target_energy']),
                                            plda_path, 'ark:'+reco2utt_path, 'scp:'+features_path,
                                            'ark,scp:{},{}'.format(scores_ark_path, scores_scp_path)
                                            ], stderr=log_file)
        score_plda_proc.communicate()


def score_plda(directory, split_directory, config, num_jobs):
    log_dir = os.path.join(directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    feat_name = 'ivectors_transformed.{}.scp'
    jobs = [(directory, split_directory, config.plda_options,
             os.path.join(directory, feat_name.format(x)),
             x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(score_plda_func, jobs, log_dir)
    else:
        run_non_mp(score_plda_func, jobs, log_dir)
    # Combine scores
    with open(os.path.join(directory, 'scores.scp'), 'w', encoding='utf8') as out_f:
        for i in range(num_jobs):
            with open(os.path.join(directory, 'scores.{}.scp'.format(i)), 'r', encoding='utf8') as in_f:
                for line in in_f:
                    out_f.write(line)


def cluster_func(directory, split_directory, config, job_name):
    log_directory = os.path.join(directory, 'log')
    log_path = os.path.join(log_directory, 'clustering.{}.log'.format(job_name))
    scores_path = os.path.join(directory, 'scores.{}.scp'.format(job_name))
    reco2num_speakers_path = os.path.join(split_directory, 'subsegments', 'reco2num_speaks.{}'.format(job_name))
    com = [thirdparty_binary('agglomerative-cluster'),
                                         '--threshold={}'.format(config['cluster_threshold']),
                                         '--read-costs={}'.format(str(config['read_costs']).lower()),
                                         ]
    reco2utt = os.path.join(split_directory, 'subsegments', 'reco2utt.{}'.format(job_name))
    labels_path = os.path.join(directory, 'labels.{}.scp'.format(job_name))
    if os.path.exists(reco2num_speakers_path):
        com.append('--reco2num-spk-rspecifier=ark,t:'+ reco2num_speakers_path)
    com += ['--max-spk-fraction={}'.format(config['max_speaker_fraction']),
            '--first-pass-max-utterances={}'.format(config['first_pass_max_utterances']),
            'scp:' + scores_path,
            'ark,t:' + reco2utt, 'ark,t:'+ labels_path]

    with open(log_path, 'w', encoding='utf8') as log_file:
        cluster_proc = subprocess.Popen(com, stderr=log_file)
        cluster_proc.communicate()


def cluster(directory, split_directory, config, num_jobs):
    log_dir = os.path.join(directory, 'log')
    os.makedirs(log_dir, exist_ok=True)

    jobs = [(directory, split_directory, config.cluster_options,
             x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(cluster_func, jobs, log_dir)
    else:
        run_non_mp(cluster_func, jobs, log_dir)

    # Combine scores
    with open(os.path.join(directory, 'labels.scp'), 'w', encoding='utf8') as out_f:
        for i in range(num_jobs):
            with open(os.path.join(directory, 'labels.{}.scp'.format(i)), 'r', encoding='utf8') as in_f:
                for line in in_f:
                    out_f.write(line)


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
    total = non_silent_frames + silent_frames
    return segs


def merge_segments(segments, min_pause_duration, max_segment_length):
    merged_segs = []
    for s in segments:
        if not merged_segs or s['begin'] > merged_segs[-1]['end'] + min_pause_duration or \
                s['end'] - merged_segs[-1]['begin'] > max_segment_length:
            if s['end'] - s['begin'] > min_pause_duration:
                merged_segs.append(s)
        else:
            merged_segs[-1]['end'] = s['end']
    return merged_segs


def segment_vad_func(directory, job_name):
    silence_proportion = 0  # The amount of silence at the sides of segments is
    # tuned to give this proportion of silence.
    frame_shift = 0.01  # Affects the interpretation of the options such as max_segment_length,
    # and the seconds in the "segments" file.
    max_segment_length = 30.0  # Maximum segment length while we are merging segments...
    # it will not allow merging segments to make segments longer than this.
    min_pause_duration = 0.05
    log_path = os.path.join(directory, 'log', 'segment_vad.{}.log'.format(job_name))
    vad_path = os.path.join(directory, 'vad.{}.scp'.format(job_name))
    vad_segments_path = os.path.join(directory, 'vad_segments.{}.scp'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file, open(vad_segments_path, 'w', encoding='utf8') as out_file:
        proc = subprocess.Popen([thirdparty_binary('copy-vector'),
                                 'scp:' + vad_path, 'ark,t:-'
                                 ], stdout=subprocess.PIPE, stderr=log_file, text=True)
        stdout, _ = proc.communicate()
        vad = stdout.splitlines()
        for line in vad:
            line = line.strip().replace('[', '').replace(']', '').split()
            recording = line[0]
            frames = line[1:]

            initial_segments = get_initial_segmentation(frames, frame_shift)
            merged = merge_segments(initial_segments, min_pause_duration, max_segment_length)
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
    jobs = [(split_dir, x) for x in range(num_jobs)]
    if config.use_mp:
        run_mp(segment_vad_func, jobs, log_directory)
    else:
        run_non_mp(segment_vad_func, jobs, log_directory)


def classify_speakers_func(directory, job_name):
    from ..helper import load_scp, save_scp
    from joblib import load
    import numpy as np
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