import subprocess
import os
import shutil

from .helper import run_mp, run_non_mp, thirdparty_binary


def decode_func(directory, job_name, mdl, config, feat_string, output_directory, num_threads=None):
    log_path = os.path.join(output_directory, 'log', 'decode.{}.log'.format(job_name))
    lat_path = os.path.join(output_directory, 'lat.{}'.format(job_name))
    if os.path.exists(lat_path):
        return
    word_symbol_path = os.path.join(directory, 'words.txt')
    hclg_path = os.path.join(directory, 'HCLG.fst')
    if config.fmllr and config.first_beam is not None:
        beam = config.first_beam
    else:
        beam = config.beam
    if config.fmllr and config.first_max_active is not None:
        max_active = config.first_max_active
    else:
        max_active = config.max_active
    with open(log_path, 'w', encoding='utf8') as log_file:
        if num_threads is None:
            decode_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster'),
                                            '--max-active={}'.format(max_active),
                                            '--beam={}'.format(beam),
                                            '--lattice-beam={}'.format(config.lattice_beam),
                                            '--allow-partial=true',
                                            '--word-symbol-table={}'.format(word_symbol_path),
                                            '--acoustic-scale={}'.format(config.acoustic_scale),
                                            mdl, hclg_path, feat_string,
                                            "ark:" + lat_path],
                                           stderr=log_file)
        else:
            decode_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster-parallel'),
                                            '--max-active={}'.format(max_active),
                                            '--beam={}'.format(beam),
                                            '--lattice-beam={}'.format(config.lattice_beam),
                                            '--allow-partial=true',
                                            '--word-symbol-table={}'.format(word_symbol_path),
                                            '--acoustic-scale={}'.format(config.acoustic_scale),
                                            '--num-threads={}'.format(num_threads),
                                            mdl, hclg_path, feat_string,
                                            "ark:" + lat_path],
                                           stderr=log_file)
        decode_proc.communicate()


def score_func(directory, job_name, config, output_directory, language_model_weight=None, word_insertion_penalty=None):
    lat_path = os.path.join(directory, 'lat.{}'.format(job_name))
    words_path = os.path.join(directory, 'words.txt')
    tra_path = os.path.join(output_directory, 'tra.{}'.format(job_name))
    log_path = os.path.join(output_directory, 'log', 'score.{}.log'.format(job_name))
    if language_model_weight is None:
        language_model_weight = config.language_model_weight
    if word_insertion_penalty is None:
        word_insertion_penalty = config.word_insertion_penalty
    with open(log_path, 'w', encoding='utf8') as log_file:
        scale_proc = subprocess.Popen([thirdparty_binary('lattice-scale'),
                                       '--inv-acoustic-scale={}'.format(language_model_weight),
                                       'ark:' + lat_path, 'ark:-'
                                       ], stdout=subprocess.PIPE, stderr=log_file)
        penalty_proc = subprocess.Popen([thirdparty_binary('lattice-add-penalty'),
                                         '--word-ins-penalty={}'.format(word_insertion_penalty),
                                         'ark:-', 'ark:-'],
                                        stdin=scale_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        best_path_proc = subprocess.Popen([thirdparty_binary('lattice-best-path'),
                                           '--word-symbol-table={}'.format(words_path),
                                           'ark:-', 'ark,t:' + tra_path], stdin=penalty_proc.stdout, stderr=log_file)
        best_path_proc.communicate()


def transcribe(transcriber):
    """
    """
    directory = transcriber.transcribe_directory
    output_directory = transcriber.transcribe_directory
    log_directory = os.path.join(output_directory, 'log')
    config = transcriber.transcribe_config
    mdl_path = os.path.join(directory, 'final.mdl')
    corpus = transcriber.corpus
    num_jobs = corpus.num_jobs
    speakers = corpus.speakers

    if config.use_mp and num_jobs > 1:
        jobs = [(directory, x, mdl_path, config,
                 config.feature_config.construct_feature_proc_string(corpus.split_directory(), directory, x),
                 output_directory)
                for x in range(len(speakers))]
    else:
        jobs = [(directory, x, mdl_path, config,
                 config.feature_config.construct_feature_proc_string(corpus.split_directory(), directory, x),
                 output_directory, corpus.original_num_jobs)
                for x in range(len(speakers))]

    if config.use_mp and num_jobs > 1:
        run_mp(decode_func, jobs, log_directory, num_jobs)
    else:
        run_non_mp(decode_func, jobs, log_directory)

    if transcriber.evaluation_mode:
        best_wer = 10000
        best = None
        for lmwt in range(transcriber.min_language_model_weight, transcriber.max_language_model_weight):
            for wip in transcriber.word_insertion_penalties:
                out_dir = os.path.join(output_directory, 'eval_{}_{}'.format(lmwt, wip))
                log_dir = os.path.join(out_dir, 'log')
                os.makedirs(log_dir, exist_ok=True)
                jobs = [(directory, x, config, out_dir, lmwt, wip)
                        for x in range(len(speakers))]
                if config.use_mp:
                    run_mp(score_func, jobs, log_dir, num_jobs)
                else:
                    run_non_mp(score_func, jobs, log_dir)
                ser, wer = transcriber.evaluate(out_dir, out_dir)
                if wer < best_wer:
                    best = (lmwt, wip)
        transcriber.transcribe_config.language_model_weight = best[0]
        transcriber.transcribe_config.word_insertion_penalty = best[1]
    else:
        jobs = [(directory, x, config, output_directory)
                for x in range(len(speakers))]
        if config.use_mp:
            run_mp(score_func, jobs, log_directory, num_jobs)
        else:
            run_non_mp(score_func, jobs, log_directory)


def initial_fmllr_func(directory, split_directory, sil_phones, job_name, mdl, config, feat_string, output_directory,
                       num_threads=None):

    log_path = os.path.join(output_directory, 'log', 'initial_fmllr.{}.log'.format(job_name))
    pre_trans_path = os.path.join(output_directory, 'pre_trans.{}'.format(job_name))
    lat_path = os.path.join(directory, 'lat.{}'.format(job_name))
    spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_name))

    with open(log_path, 'w', encoding='utf8') as log_file:
        latt_post_proc = subprocess.Popen([thirdparty_binary('lattice-to-post'),
                                           '--acoustic-scale={}'.format(config.acoustic_scale),
                                           'ark:' + lat_path, 'ark:-'], stdout=subprocess.PIPE,
                                          stderr=log_file)
        weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                str(config.silence_weight),
                                                sil_phones, mdl, 'ark:-', 'ark:-'],
                                               stdin=latt_post_proc.stdout, stdout=subprocess.PIPE,
                                               stderr=log_file)
        gmm_gpost_proc = subprocess.Popen([thirdparty_binary('gmm-post-to-gpost'),
                                           mdl, feat_string, 'ark:-', 'ark:-'],
                                          stdin=weight_silence_proc.stdout, stdout=subprocess.PIPE,
                                          stderr=log_file)
        fmllr_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr-gpost'),
                                       '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                       '--spk2utt=ark:' + spk2utt_path, mdl, feat_string,
                                       'ark,s,cs:-', 'ark:' + pre_trans_path],
                                      stdin=gmm_gpost_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        fmllr_proc.communicate()


def lat_gen_fmllr_func(directory, split_directory, sil_phones, job_name, mdl, config, feat_string, output_directory,
                       num_threads=None):
    log_path = os.path.join(output_directory, 'log', 'lat_gen.{}.log'.format(job_name))
    word_symbol_path = os.path.join(directory, 'words.txt')
    hclg_path = os.path.join(directory, 'HCLG.fst')
    tmp_lat_path = os.path.join(output_directory, 'lat.tmp.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        if num_threads is None:
            lat_gen_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster'),
                                             '--max-active={}'.format(config.max_active),
                                             '--beam={}'.format(config.beam),
                                             '--lattice-beam={}'.format(config.lattice_beam),
                                             '--acoustic-scale={}'.format(config.acoustic_scale),
                                             '--determinize-lattice=false',
                                             '--allow-partial=true',
                                             '--word-symbol-table={}'.format(word_symbol_path),
                                             mdl, hclg_path, feat_string, 'ark:' + tmp_lat_path
                                             ], stderr=log_file)
        else:
            lat_gen_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster-parallel'),
                                             '--max-active={}'.format(config.max_active),
                                             '--beam={}'.format(config.beam),
                                             '--lattice-beam={}'.format(config.lattice_beam),
                                             '--acoustic-scale={}'.format(config.acoustic_scale),
                                             '--determinize-lattice=false',
                                             '--allow-partial=true',
                                             '--num-threads={}'.format(num_threads),
                                             '--word-symbol-table={}'.format(word_symbol_path),
                                             mdl, hclg_path, feat_string, 'ark:' + tmp_lat_path
                                             ], stderr=log_file)
        lat_gen_proc.communicate()


def final_fmllr_est_func(directory, split_directory, sil_phones, job_name, mdl, config, feat_string, output_directory,
                         num_threads=None):
    log_path = os.path.join(output_directory, 'log', 'final_fmllr.{}.log'.format(job_name))
    pre_trans_path = os.path.join(output_directory, 'pre_trans.{}'.format(job_name))
    trans_tmp_path = os.path.join(output_directory, 'trans_tmp.{}'.format(job_name))
    trans_path = os.path.join(output_directory, 'trans.{}'.format(job_name))
    lat_path = os.path.join(directory, 'lat.{}'.format(job_name))
    spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_name))
    tmp_lat_path = os.path.join(output_directory, 'lat.tmp.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        if num_threads is None:
            determinize_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned'),
                                                 '--acoustic-scale={}'.format(config.acoustic_scale),
                                                 '--beam=4.0', 'ark:' + tmp_lat_path, 'ark:-'],
                                                stderr=log_file, stdout=subprocess.PIPE)
        else:
            determinize_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned-parallel'),
                                                 '--acoustic-scale={}'.format(config.acoustic_scale),
                                                 '--num-threads={}'.format(num_threads),
                                                 '--beam=4.0', 'ark:' + tmp_lat_path, 'ark:-'],
                                                stderr=log_file, stdout=subprocess.PIPE)
        latt_post_proc = subprocess.Popen([thirdparty_binary('lattice-to-post'),
                                           '--acoustic-scale={}'.format(config.acoustic_scale),
                                           'ark:' + lat_path, 'ark:-'],
                                          stdin=determinize_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                str(config.silence_weight),
                                                sil_phones, mdl, 'ark:-', 'ark:-'],
                                               stdin=latt_post_proc.stdout, stdout=subprocess.PIPE,
                                               stderr=log_file)
        fmllr_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                       '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                       '--spk2utt=ark:' + spk2utt_path, mdl, feat_string,
                                       'ark,s,cs:-', 'ark:' + trans_tmp_path],
                                      stdin=weight_silence_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
        fmllr_proc.communicate()

        compose_proc = subprocess.Popen([thirdparty_binary('compose-transforms'),
                                         '--b-is-affine=true', 'ark:' + trans_tmp_path,
                                         'ark:' + pre_trans_path, 'ark:' + trans_path],
                                        stderr=log_file)
        compose_proc.communicate()


def fmllr_rescore_func(directory, split_directory, sil_phones, job_name, mdl, config, feat_string, output_directory,
                       num_threads=None):
    log_path = os.path.join(output_directory, 'log', 'fmllr_rescore.{}.log'.format(job_name))
    tmp_lat_path = os.path.join(output_directory, 'lat.tmp.{}'.format(job_name))
    final_lat_path = os.path.join(output_directory, 'lat.{}'.format(job_name))
    with open(log_path, 'w', encoding='utf8') as log_file:
        rescore_proc = subprocess.Popen([thirdparty_binary('gmm-rescore-lattice'),
                                         mdl, 'ark:' + tmp_lat_path,
                                         feat_string, 'ark:-'],
                                        stdout=subprocess.PIPE, stderr=log_file)
        if num_threads is None:
            determinize_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned'),
                                                 '--acoustic-scale={}'.format(config.acoustic_scale),
                                                 '--beam={}'.format(config.lattice_beam),
                                                 'ark:-', 'ark:' + final_lat_path
                                                 ], stdin=rescore_proc.stdout, stderr=log_file)
        else:
            determinize_proc = subprocess.Popen([thirdparty_binary('lattice-determinize-pruned-parallel'),
                                                 '--acoustic-scale={}'.format(config.acoustic_scale),
                                                 '--beam={}'.format(config.lattice_beam),
                                                 '--num-threads={}'.format(num_threads),
                                                 'ark:-', 'ark:' + final_lat_path
                                                 ], stdin=rescore_proc.stdout, stderr=log_file)
        determinize_proc.communicate()


def transcribe_fmllr(transcriber):
    directory = transcriber.transcribe_directory
    output_directory = transcriber.transcribe_directory
    config = transcriber.transcribe_config
    corpus = transcriber.corpus
    num_jobs = corpus.num_jobs
    split_directory = corpus.split_directory()
    sil_phones = transcriber.dictionary.optional_silence_csl

    fmllr_directory = os.path.join(output_directory, 'fmllr')
    log_dir = os.path.join(fmllr_directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    mdl_path = os.path.join(directory, 'final.mdl')
    feat_name = config.feature_file_base_name
    feat_name += '.{}.scp'
    jobs = []
    for x in range(num_jobs):
        if num_jobs > 1:
            jobs = [(directory, split_directory, sil_phones, x, mdl_path, config,
                     config.feature_config.construct_feature_proc_string(split_directory, directory, x), fmllr_directory)
                    for x in range(num_jobs)]
        else:
            jobs = [(directory, split_directory, sil_phones, x, mdl_path, config,
                 config.feature_config.construct_feature_proc_string(split_directory, directory, x), fmllr_directory, corpus.original_num_jobs)
                for x in range(num_jobs)]

    run_non_mp(initial_fmllr_func, jobs, log_dir)

    if config.use_mp and num_jobs > 1:
        run_mp(lat_gen_fmllr_func, jobs, log_dir)
    else:
        run_non_mp(lat_gen_fmllr_func, jobs, log_dir)

    run_non_mp(final_fmllr_est_func, jobs, log_dir)

    if config.use_mp:
        run_mp(fmllr_rescore_func, jobs, log_dir)
    else:
        run_non_mp(fmllr_rescore_func, jobs, log_dir)

    if transcriber.evaluation_mode:
        best_wer = 10000
        best = None
        for lmwt in range(transcriber.min_language_model_weight, transcriber.max_language_model_weight):
            for wip in transcriber.word_insertion_penalties:
                out_dir = os.path.join(fmllr_directory, 'eval_{}_{}'.format(lmwt, wip))
                log_dir = os.path.join(out_dir, 'log')
                os.makedirs(log_dir, exist_ok=True)
                jobs = [(directory, x, config, out_dir, lmwt, wip)
                        for x in range(num_jobs)]
                if config.use_mp:
                    run_mp(score_func, jobs, log_dir)
                else:
                    run_non_mp(score_func, jobs, log_dir)
                ser, wer = transcriber.evaluate(out_dir, out_dir)
                if wer < best_wer:
                    best = (lmwt, wip)
        transcriber.transcribe_config.language_model_weight = best[0]
        transcriber.transcribe_config.word_insertion_penalty = best[1]
        out_dir = os.path.join(fmllr_directory, 'eval_{}_{}'.format(best[0], best[1]))
        for j in range(num_jobs):
            tra_path = os.path.join(out_dir, 'tra.{}'.format(j))
            saved_tra_path = os.path.join(fmllr_directory, 'tra.{}'.format(j))
            shutil.copyfile(tra_path, saved_tra_path)
    else:
        jobs = [(directory, x, config, fmllr_directory)
                for x in range(num_jobs)]
        if config.use_mp:
            run_mp(score_func, jobs, log_dir)
        else:
            run_non_mp(score_func, jobs, log_dir)