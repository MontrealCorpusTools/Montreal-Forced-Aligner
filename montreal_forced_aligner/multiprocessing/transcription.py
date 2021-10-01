import subprocess
import os
import shutil
import sys

from .helper import run_mp, run_non_mp, thirdparty_binary
from ..dictionary import MultispeakerDictionary


def decode_func(model_directory, job_name, config, feat_string, output_directory, num_threads=None,
                dictionary_names=None):
    mdl_path = os.path.join(model_directory, 'final.mdl')
    if dictionary_names is None:
        lat_path = os.path.join(output_directory, 'lat.{}'.format(job_name))
        if os.path.exists(lat_path):
            return
        word_symbol_path = os.path.join(model_directory, 'words.txt')
        hclg_path = os.path.join(model_directory, 'HCLG.fst')
        if config.fmllr and config.first_beam is not None:
            beam = config.first_beam
        else:
            beam = config.beam
        if config.fmllr and config.first_max_active is not None and not config.no_speakers:
            max_active = config.first_max_active
        else:
            max_active = config.max_active
        log_path = os.path.join(output_directory, 'log', 'decode.{}.log'.format(job_name))
        with open(log_path, 'w', encoding='utf8') as log_file:
            if num_threads is None:
                decode_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster'),
                                                '--max-active={}'.format(max_active),
                                                '--beam={}'.format(beam),
                                                '--lattice-beam={}'.format(config.lattice_beam),
                                                '--allow-partial=true',
                                                '--word-symbol-table={}'.format(word_symbol_path),
                                                '--acoustic-scale={}'.format(config.acoustic_scale),
                                                mdl_path, hclg_path, feat_string,
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
                                                mdl_path, hclg_path, feat_string,
                                                "ark:" + lat_path],
                                               stderr=log_file)
            decode_proc.communicate()
    else:
        for name in dictionary_names:
            lat_path = os.path.join(output_directory, 'lat.{}.{}'.format(job_name, name))
            if os.path.exists(lat_path):
                continue
            word_symbol_path = os.path.join(model_directory, name + '_words.txt')
            hclg_path = os.path.join(model_directory, name + '_HCLG.fst')
            if config.fmllr and config.first_beam is not None:
                beam = config.first_beam
            else:
                beam = config.beam
            if config.fmllr and config.first_max_active is not None and not config.no_speakers:
                max_active = config.first_max_active
            else:
                max_active = config.max_active
            log_path = os.path.join(output_directory, 'log', 'decode.{}.{}.log'.format(job_name, name))
            dictionary_feat_string = feat_string.replace('feats.{}.scp'.format(job_name),
                                                         'feats.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('cmvn.{}.scp'.format(job_name),
                                                                    'cmvn.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('utt2spk.{}'.format(job_name),
                                                                    'utt2spk.{}.{}'.format(job_name, name))
            with open(log_path, 'w', encoding='utf8') as log_file:
                if num_threads is None:
                    decode_proc = subprocess.Popen([thirdparty_binary('gmm-latgen-faster'),
                                                    '--max-active={}'.format(max_active),
                                                    '--beam={}'.format(beam),
                                                    '--lattice-beam={}'.format(config.lattice_beam),
                                                    '--allow-partial=true',
                                                    '--word-symbol-table={}'.format(word_symbol_path),
                                                    '--acoustic-scale={}'.format(config.acoustic_scale),
                                                    mdl_path, hclg_path, dictionary_feat_string,
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
                                                    mdl_path, hclg_path, dictionary_feat_string,
                                                    "ark:" + lat_path],
                                                   stderr=log_file)
                decode_proc.communicate()


def score_func(model_directory, transcribe_directory, job_name, config, output_directory, language_model_weight=None,
               word_insertion_penalty=None, dictionary_names=None):
    if language_model_weight is None:
        language_model_weight = config.language_model_weight
    if word_insertion_penalty is None:
        word_insertion_penalty = config.word_insertion_penalty
    if dictionary_names is None:
        lat_path = os.path.join(transcribe_directory, 'lat.{}'.format(job_name))
        rescored_lat_path = os.path.join(transcribe_directory, 'lat.{}.rescored'.format(job_name))
        carpa_rescored_lat_path = os.path.join(transcribe_directory, 'lat.{}.carparescored'.format(job_name))
        if os.path.exists(carpa_rescored_lat_path):
            lat_path = carpa_rescored_lat_path
        elif os.path.exists(rescored_lat_path):
            lat_path = rescored_lat_path
        words_path = os.path.join(model_directory, 'words.txt')
        tra_path = os.path.join(output_directory, 'tra.{}'.format(job_name))
        log_path = os.path.join(output_directory, 'log', 'score.{}.log'.format(job_name))
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
                                               'ark:-', 'ark,t:' + tra_path], stdin=penalty_proc.stdout,
                                              stderr=log_file)
            best_path_proc.communicate()
    else:
        for name in dictionary_names:
            lat_path = os.path.join(transcribe_directory, 'lat.{}.{}'.format(job_name, name))
            rescored_lat_path = os.path.join(transcribe_directory, 'lat.{}.{}.rescored'.format(job_name, name))
            carpa_rescored_lat_path = os.path.join(transcribe_directory,
                                                   'lat.{}.{}.carparescored'.format(job_name, name))
            if os.path.exists(carpa_rescored_lat_path):
                lat_path = carpa_rescored_lat_path
            elif os.path.exists(rescored_lat_path):
                lat_path = rescored_lat_path
            words_path = os.path.join(model_directory, name + '_words.txt')
            tra_path = os.path.join(output_directory, 'tra.{}.{}'.format(job_name, name))
            log_path = os.path.join(output_directory, 'log', 'score.{}.{}.log'.format(job_name, name))
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
                                                   'ark:-', 'ark,t:' + tra_path], stdin=penalty_proc.stdout,
                                                  stderr=log_file)
                best_path_proc.communicate()


def lm_rescore_func(model_directory, job_name, config, feat_string, output_directory, num_threads=None,
                    dictionary_names=None):
    if sys.platform == 'win32':
        project_type_arg = '--project_output=true'
    else:
        project_type_arg = '--project_type=output'
    if dictionary_names is None:
        rescored_lat_path = os.path.join(output_directory, 'lat.{}.lmrescored'.format(job_name))
        lat_path = os.path.join(output_directory, 'lat.{}'.format(job_name))
        old_g_path = os.path.join(model_directory, 'small_G.fst')
        new_g_path = os.path.join(model_directory, 'med_G.fst')
        log_path = os.path.join(output_directory, 'log', 'lm_rescore.{}.log'.format(job_name))
        with open(log_path, 'w', encoding='utf8') as log_file:
            lattice_scale_proc = subprocess.Popen([thirdparty_binary('lattice-lmrescore-pruned'),
                                                   '--acoustic-scale={}'.format(config.acoustic_scale),
                                                   f"fstproject {project_type_arg} {old_g_path} |",
                                                   f"fstproject {project_type_arg} {new_g_path} |",
                                                   'ark:' + lat_path, 'ark:' + rescored_lat_path], stderr=log_file)
            lattice_scale_proc.communicate()
    else:
        for name in dictionary_names:
            rescored_lat_path = os.path.join(output_directory, 'lat.{}.{}.lmrescored'.format(job_name, name))
            lat_path = os.path.join(output_directory, 'lat.{}.{}'.format(job_name, name))
            old_g_path = os.path.join(model_directory, name + '_small_G.fst')
            new_g_path = os.path.join(model_directory, name + '_med_G.fst')
            log_path = os.path.join(output_directory, 'log', 'lm_rescore.{}.{}.log'.format(job_name, name))

            with open(log_path, 'w', encoding='utf8') as log_file:
                lattice_scale_proc = subprocess.Popen([thirdparty_binary('lattice-lmrescore-pruned'),
                                                       '--acoustic-scale={}'.format(config.acoustic_scale),
                                                       f"fstproject {project_type_arg} {old_g_path} |",
                                                       f"fstproject {project_type_arg} {new_g_path} |",
                                                       'ark:' + lat_path, 'ark:' + rescored_lat_path], stderr=log_file)
                lattice_scale_proc.communicate()


def carpa_lm_rescore_func(model_directory, job_name, config, feat_string, output_directory, num_threads=None,
                          dictionary_names=None):
    if sys.platform == 'win32':
        project_type_arg = '--project_output=true'
    else:
        project_type_arg = '--project_type=output'
    if dictionary_names is None:
        lat_path = os.path.join(output_directory, 'lat.{}.lmrescored'.format(job_name))
        rescored_lat_path = os.path.join(output_directory, 'lat.{}.carparescored'.format(job_name))
        if os.path.exists(rescored_lat_path):
            return
        old_g_path = os.path.join(model_directory, 'med_G.fst')
        new_g_path = os.path.join(model_directory, 'G.carpa')
        log_path = os.path.join(output_directory, 'log', 'carpa_lm_rescore.{}.log'.format(job_name))
        with open(log_path, 'w', encoding='utf8') as log_file:
            lmrescore_proc = subprocess.Popen([thirdparty_binary('lattice-lmrescore'),
                                               '--lm-scale=-1.0', 'ark:' + lat_path,
                                               f"fstproject {project_type_arg} {old_g_path} |",
                                               'ark:-'], stdout=subprocess.PIPE, stderr=log_file)
            lmrescore_const_proc = subprocess.Popen([thirdparty_binary('lattice-lmrescore-const-arpa'),
                                                     '--lm-scale=1.0', 'ark:-',
                                                     new_g_path,
                                                     'ark:' + rescored_lat_path], stdin=lmrescore_proc.stdout,
                                                    stderr=log_file)
            lmrescore_const_proc.communicate()
    else:
        for name in dictionary_names:
            lat_path = os.path.join(output_directory, 'lat.{}.{}.lmrescored'.format(job_name, name))
            rescored_lat_path = os.path.join(output_directory, 'lat.{}.{}.carparescored'.format(job_name, name))
            if os.path.exists(rescored_lat_path):
                continue
            old_g_path = os.path.join(model_directory, name + '_med_G.fst')
            new_g_path = os.path.join(model_directory, name + '_G.carpa')
            log_path = os.path.join(output_directory, 'log', 'carpa_lm_rescore.{}.{}.log'.format(job_name, name))

            with open(log_path, 'w', encoding='utf8') as log_file:
                lmrescore_proc = subprocess.Popen([thirdparty_binary('lattice-lmrescore'),
                                                   '--lm-scale=-1.0', 'ark:' + lat_path,
                                                   f"fstproject {project_type_arg} {old_g_path} |",
                                                   'ark:-'], stdout=subprocess.PIPE, stderr=log_file)
                lmrescore_const_proc = subprocess.Popen([thirdparty_binary('lattice-lmrescore-const-arpa'),
                                                         '--lm-scale=1.0', 'ark:-',
                                                         new_g_path,
                                                         'ark:' + rescored_lat_path], stdin=lmrescore_proc.stdout,
                                                        stderr=log_file)
                lmrescore_const_proc.communicate()


def transcribe(transcriber):
    """
    """
    model_directory = transcriber.model_directory
    decode_directory = transcriber.transcribe_directory
    log_directory = os.path.join(decode_directory, 'log')
    config = transcriber.transcribe_config
    corpus = transcriber.corpus
    num_jobs = corpus.num_jobs

    jobs = [(model_directory, x, config,
             config.feature_config.construct_feature_proc_string(corpus.split_directory(), model_directory, x),
             decode_directory, corpus.original_num_jobs, transcriber.dictionaries_for_job(x))
            for x in range(num_jobs)]

    run_non_mp(decode_func, jobs, log_directory)

    if config.use_mp:
        run_mp(lm_rescore_func, jobs, log_directory)
    else:
        run_non_mp(lm_rescore_func, jobs, log_directory)

    if config.use_mp:
        run_mp(carpa_lm_rescore_func, jobs, log_directory)
    else:
        run_non_mp(carpa_lm_rescore_func, jobs, log_directory)

    if transcriber.evaluation_mode:
        best_wer = 10000
        best = None
        for lmwt in range(transcriber.min_language_model_weight, transcriber.max_language_model_weight):
            for wip in transcriber.word_insertion_penalties:
                out_dir = os.path.join(decode_directory, 'eval_{}_{}'.format(lmwt, wip))
                log_dir = os.path.join(out_dir, 'log')
                os.makedirs(log_dir, exist_ok=True)

                jobs = [(model_directory, decode_directory, x, config, out_dir, lmwt, wip,
                         transcriber.dictionaries_for_job(x))
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
    else:
        jobs = [(model_directory, decode_directory, x, config, decode_directory, None, None,
                 transcriber.dictionaries_for_job(x))
                for x in range(num_jobs)]
        if config.use_mp:
            run_mp(score_func, jobs, log_directory)
        else:
            run_non_mp(score_func, jobs, log_directory)


def initial_fmllr_func(initial_directory, split_directory, sil_phones, job_name, mdl, config, feat_string,
                       output_directory,
                       num_threads=None, dictionary_names=None):
    if dictionary_names is None:
        log_path = os.path.join(output_directory, 'log', 'initial_fmllr.{}.log'.format(job_name))
        pre_trans_path = os.path.join(output_directory, 'pre_trans.{}'.format(job_name))
        lat_path = os.path.join(initial_directory, 'lat.{}'.format(job_name))
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
    else:
        for name in dictionary_names:
            log_path = os.path.join(output_directory, 'log', 'initial_fmllr.{}.{}.log'.format(job_name, name))
            pre_trans_path = os.path.join(output_directory, 'pre_trans.{}.{}'.format(job_name, name))
            lat_path = os.path.join(initial_directory, 'lat.{}.{}'.format(job_name, name))
            spk2utt_path = os.path.join(split_directory, 'spk2utt.{}.{}'.format(job_name, name))
            dictionary_feat_string = feat_string.replace('feats.{}.scp'.format(job_name),
                                                         'feats.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('cmvn.{}.scp'.format(job_name),
                                                                    'cmvn.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('utt2spk.{}'.format(job_name),
                                                                    'utt2spk.{}.{}'.format(job_name, name))

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
                                                   mdl, dictionary_feat_string, 'ark:-', 'ark:-'],
                                                  stdin=weight_silence_proc.stdout, stdout=subprocess.PIPE,
                                                  stderr=log_file)
                fmllr_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr-gpost'),
                                               '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                               '--spk2utt=ark:' + spk2utt_path, mdl, dictionary_feat_string,
                                               'ark,s,cs:-', 'ark:' + pre_trans_path],
                                              stdin=gmm_gpost_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
                fmllr_proc.communicate()


def lat_gen_fmllr_func(model_directory, split_directory, sil_phones, job_name, mdl, config, feat_string,
                       output_directory,
                       num_threads=None, dictionary_names=None):
    if dictionary_names is None:
        log_path = os.path.join(output_directory, 'log', 'lat_gen.{}.log'.format(job_name))
        word_symbol_path = os.path.join(model_directory, 'words.txt')
        hclg_path = os.path.join(model_directory, 'HCLG.fst')
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
    else:
        for name in dictionary_names:
            log_path = os.path.join(output_directory, 'log', 'lat_gen.{}.{}.log'.format(job_name, name))
            word_symbol_path = os.path.join(model_directory, name + '_words.txt')
            hclg_path = os.path.join(model_directory, name + '_HCLG.fst')
            tmp_lat_path = os.path.join(output_directory, 'lat.tmp.{}.{}'.format(job_name, name))
            dictionary_feat_string = feat_string.replace('feats.{}.scp'.format(job_name),
                                                         'feats.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('cmvn.{}.scp'.format(job_name),
                                                                    'cmvn.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('utt2spk.{}'.format(job_name),
                                                                    'utt2spk.{}.{}'.format(job_name, name))
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
                                                     mdl, hclg_path, dictionary_feat_string, 'ark:' + tmp_lat_path
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
                                                     mdl, hclg_path, dictionary_feat_string, 'ark:' + tmp_lat_path
                                                     ], stderr=log_file)
                lat_gen_proc.communicate()


def final_fmllr_est_func(model_directory, split_directory, sil_phones, job_name, mdl, config, feat_string, si_directory,
                         fmllr_directory, num_threads=None, dictionary_names=None):
    if dictionary_names is None:
        log_path = os.path.join(fmllr_directory, 'log', 'final_fmllr.{}.log'.format(job_name))
        pre_trans_path = os.path.join(fmllr_directory, 'pre_trans.{}'.format(job_name))
        trans_tmp_path = os.path.join(fmllr_directory, 'trans_tmp.{}'.format(job_name))
        trans_path = os.path.join(fmllr_directory, 'trans.{}'.format(job_name))
        lat_path = os.path.join(si_directory, 'lat.{}'.format(job_name))
        spk2utt_path = os.path.join(split_directory, 'spk2utt.{}'.format(job_name))
        tmp_lat_path = os.path.join(fmllr_directory, 'lat.tmp.{}'.format(job_name))
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
                                               'ark:-', 'ark:-'],
                                              stdin=determinize_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)
            weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                    str(config.silence_weight),
                                                    sil_phones, mdl, 'ark:-', 'ark:-'],
                                                   stdin=latt_post_proc.stdout, stdout=subprocess.PIPE,
                                                   stderr=log_file)
            fmllr_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                           '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                           '--spk2utt=ark:' + spk2utt_path, mdl, feat_string,
                                           'ark,s,cs:-', 'ark:-'],
                                          stdin=weight_silence_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)

            compose_proc = subprocess.Popen([thirdparty_binary('compose-transforms'),
                                             '--b-is-affine=true', 'ark:-',
                                             'ark:' + pre_trans_path, 'ark:' + trans_path],
                                            stderr=log_file, stdin=fmllr_proc.stdout)
            compose_proc.communicate()
    else:
        for name in dictionary_names:
            log_path = os.path.join(fmllr_directory, 'log', 'final_fmllr.{}.{}.log'.format(job_name, name))
            pre_trans_path = os.path.join(fmllr_directory, 'pre_trans.{}.{}'.format(job_name, name))
            trans_tmp_path = os.path.join(fmllr_directory, 'trans_tmp.{}.{}'.format(job_name, name))
            trans_path = os.path.join(fmllr_directory, 'trans.{}.{}'.format(job_name, name))
            lat_path = os.path.join(si_directory, 'lat.{}.{}'.format(job_name, name))
            spk2utt_path = os.path.join(split_directory, 'spk2utt.{}.{}'.format(job_name, name))
            tmp_lat_path = os.path.join(fmllr_directory, 'lat.tmp.{}.{}'.format(job_name, name))
            dictionary_feat_string = feat_string.replace('feats.{}.scp'.format(job_name),
                                                         'feats.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('cmvn.{}.scp'.format(job_name),
                                                                    'cmvn.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('utt2spk.{}'.format(job_name),
                                                                    'utt2spk.{}.{}'.format(job_name, name))

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
                                                   'ark:-', 'ark:-'],
                                                  stdin=determinize_proc.stdout, stdout=subprocess.PIPE,
                                                  stderr=log_file)
                weight_silence_proc = subprocess.Popen([thirdparty_binary('weight-silence-post'),
                                                        str(config.silence_weight),
                                                        sil_phones, mdl, 'ark:-', 'ark:-'],
                                                       stdin=latt_post_proc.stdout, stdout=subprocess.PIPE,
                                                       stderr=log_file)
                fmllr_proc = subprocess.Popen([thirdparty_binary('gmm-est-fmllr'),
                                               '--fmllr-update-type={}'.format(config.fmllr_update_type),
                                               '--spk2utt=ark:' + spk2utt_path, mdl, dictionary_feat_string,
                                               'ark,s,cs:-', 'ark:-'],
                                              stdin=weight_silence_proc.stdout, stdout=subprocess.PIPE, stderr=log_file)

                compose_proc = subprocess.Popen([thirdparty_binary('compose-transforms'),
                                                 '--b-is-affine=true', 'ark:-',
                                                 'ark:' + pre_trans_path, 'ark:' + trans_path],
                                                stderr=log_file, stdin=fmllr_proc.stdout)
                compose_proc.communicate()


def fmllr_rescore_func(directory, split_directory, sil_phones, job_name, mdl, config, feat_string, output_directory,
                       num_threads=None, dictionary_names=None):
    if dictionary_names is None:
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
    else:
        for name in dictionary_names:
            log_path = os.path.join(output_directory, 'log', 'fmllr_rescore.{}.{}.log'.format(job_name, name))
            tmp_lat_path = os.path.join(output_directory, 'lat.tmp.{}.{}'.format(job_name, name))
            final_lat_path = os.path.join(output_directory, 'lat.{}.{}'.format(job_name, name))
            dictionary_feat_string = feat_string.replace('feats.{}.scp'.format(job_name),
                                                         'feats.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('cmvn.{}.scp'.format(job_name),
                                                                    'cmvn.{}.{}.scp'.format(job_name, name))
            dictionary_feat_string = dictionary_feat_string.replace('utt2spk.{}'.format(job_name),
                                                                    'utt2spk.{}.{}'.format(job_name, name))
            with open(log_path, 'w', encoding='utf8') as log_file:
                rescore_proc = subprocess.Popen([thirdparty_binary('gmm-rescore-lattice'),
                                                 mdl, 'ark:' + tmp_lat_path,
                                                 dictionary_feat_string, 'ark:-'],
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
    model_directory = transcriber.model_directory
    output_directory = transcriber.transcribe_directory
    config = transcriber.transcribe_config
    corpus = transcriber.corpus
    num_jobs = corpus.num_jobs
    split_directory = corpus.split_directory()
    sil_phones = transcriber.dictionary.optional_silence_csl

    fmllr_directory = os.path.join(output_directory, 'fmllr')
    log_dir = os.path.join(fmllr_directory, 'log')
    os.makedirs(log_dir, exist_ok=True)
    mdl_path = os.path.join(model_directory, 'final.mdl')

    if num_jobs > 1:
        num_threads = None
    else:
        num_threads = corpus.original_num_jobs

    jobs = [(output_directory, split_directory, sil_phones, x, mdl_path, config,
             config.feature_config.construct_feature_proc_string(split_directory, model_directory, x),
             fmllr_directory, num_threads, transcriber.dictionaries_for_job(x))
            for x in range(num_jobs)]

    run_non_mp(initial_fmllr_func, jobs, log_dir)

    jobs = [(model_directory, split_directory, sil_phones, x, mdl_path, config,
             config.feature_config.construct_feature_proc_string(split_directory, model_directory, x),
             fmllr_directory, corpus.original_num_jobs, transcriber.dictionaries_for_job(x))
            for x in range(num_jobs)]

    run_non_mp(lat_gen_fmllr_func, jobs, log_dir)

    jobs = [(model_directory, split_directory, sil_phones, x, mdl_path, config,
             config.feature_config.construct_feature_proc_string(split_directory, model_directory, x),
             output_directory, fmllr_directory, num_threads, transcriber.dictionaries_for_job(x))
            for x in range(num_jobs)]

    run_non_mp(final_fmllr_est_func, jobs, log_dir)

    jobs = [(model_directory, split_directory, sil_phones, x, mdl_path, config,
             config.feature_config.construct_feature_proc_string(split_directory, model_directory, x),
             fmllr_directory, num_threads, transcriber.dictionaries_for_job(x))
            for x in range(num_jobs)]

    if config.use_mp:
        run_mp(fmllr_rescore_func, jobs, log_dir)
    else:
        run_non_mp(fmllr_rescore_func, jobs, log_dir)

    jobs = [(model_directory, x, config,
             config.feature_config.construct_feature_proc_string(corpus.split_directory(), model_directory, x),
             fmllr_directory, num_threads, transcriber.dictionaries_for_job(x))
            for x in range(num_jobs)]

    if config.use_mp:
        run_mp(lm_rescore_func, jobs, log_dir)
    else:
        run_non_mp(lm_rescore_func, jobs, log_dir)

    if config.use_mp:
        run_mp(carpa_lm_rescore_func, jobs, log_dir)
    else:
        run_non_mp(carpa_lm_rescore_func, jobs, log_dir)

    if transcriber.evaluation_mode:
        best_wer = 10000
        best = None
        for lmwt in range(transcriber.min_language_model_weight, transcriber.max_language_model_weight):
            for wip in transcriber.word_insertion_penalties:
                out_dir = os.path.join(fmllr_directory, 'eval_{}_{}'.format(lmwt, wip))
                log_dir = os.path.join(out_dir, 'log')
                os.makedirs(log_dir, exist_ok=True)
                jobs = [(model_directory, fmllr_directory, x, config, out_dir, lmwt, wip,
                         transcriber.dictionaries_for_job(x))
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
        for filename in os.listdir(out_dir):
            if not filename.startswith('tra'):
                continue
            tra_path = os.path.join(out_dir, filename)
            saved_tra_path = os.path.join(fmllr_directory, filename)
            shutil.copyfile(tra_path, saved_tra_path)
    else:
        jobs = [(model_directory, fmllr_directory, x, config, fmllr_directory, None, None,
                 transcriber.dictionaries_for_job(x))
                for x in range(num_jobs)]
        if config.use_mp:
            run_mp(score_func, jobs, log_dir)
        else:
            run_non_mp(score_func, jobs, log_dir)
