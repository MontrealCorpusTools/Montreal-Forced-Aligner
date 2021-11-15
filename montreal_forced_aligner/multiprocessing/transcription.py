"""
Transcription functions
-----------------------

"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, TextIO, Union

from ..abc import MetaDict
from ..exceptions import KaldiProcessingError
from ..utils import thirdparty_binary
from .helper import run_mp, run_non_mp

if TYPE_CHECKING:
    from ..dictionary import MappingType
    from ..transcriber import Transcriber

    DictionaryNames = Optional[Union[List[str], str]]

__all__ = [
    "compose_g",
    "compose_lg",
    "compose_clg",
    "compose_hclg",
    "compose_g_carpa",
    "create_hclgs",
    "transcribe",
    "transcribe_fmllr",
    "score_transcriptions",
    "fmllr_rescore_func",
    "final_fmllr_est_func",
    "initial_fmllr_func",
    "lat_gen_fmllr_func",
    "score_func",
    "lm_rescore_func",
    "carpa_lm_rescore_func",
    "decode_func",
    "create_hclg_func",
]


def compose_lg(dictionary_path: str, small_g_path: str, lg_path: str, log_file: TextIO) -> None:
    """
    Compose an LG.fst

    Parameters
    ----------
    dictionary_path: str
        Path to a lexicon fst file
    small_g_path: str
        Path to the small language model's G.fst
    lg_path: str
        Output path to LG.fst
    log_file: TextIO
        Log file handler to output logging info to
    """
    if os.path.exists(lg_path):
        return
    temp_compose_path = lg_path + ".temp"
    compose_proc = subprocess.Popen(
        [thirdparty_binary("fsttablecompose"), dictionary_path, small_g_path, temp_compose_path],
        stderr=log_file,
        env=os.environ,
    )
    compose_proc.communicate()

    temp2_compose_path = lg_path + ".temp2"
    determinize_proc = subprocess.Popen(
        [
            thirdparty_binary("fstdeterminizestar"),
            "--use-log=true",
            temp_compose_path,
            temp2_compose_path,
        ],
        stderr=log_file,
        env=os.environ,
    )
    determinize_proc.communicate()
    os.remove(temp_compose_path)

    minimize_proc = subprocess.Popen(
        [thirdparty_binary("fstminimizeencoded"), temp2_compose_path, temp_compose_path],
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    minimize_proc.communicate()
    os.remove(temp2_compose_path)
    push_proc = subprocess.Popen(
        [thirdparty_binary("fstpushspecial"), temp_compose_path, lg_path],
        stderr=log_file,
        env=os.environ,
    )
    push_proc.communicate()
    os.remove(temp_compose_path)


def compose_clg(
    in_disambig: str,
    out_disambig: str,
    context_width: int,
    central_pos: int,
    ilabels_temp: str,
    lg_path: str,
    clg_path: str,
    log_file: TextIO,
) -> None:
    """
    Compose a CLG.fst

    Parameters
    ----------
    in_disambig: str
        Path to read disambiguation symbols file
    out_disambig: str
        Path to write disambiguation symbols file
    context_width: int
        Context width of the acoustic model
    central_pos: int
        Central position of the acoustic model
    ilabels_temp:
        Temporary file for ilabels
    lg_path: str
        Path to a LG.fst file
    clg_path:
        Path to save CLG.fst file
    log_file: TextIO
        Log file handler to output logging info to
    """
    compose_proc = subprocess.Popen(
        [
            thirdparty_binary("fstcomposecontext"),
            f"--context-size={context_width}",
            f"--central-position={central_pos}",
            f"--read-disambig-syms={in_disambig}",
            f"--write-disambig-syms={out_disambig}",
            ilabels_temp,
            lg_path,
        ],
        stdout=subprocess.PIPE,
        stderr=log_file,
    )
    sort_proc = subprocess.Popen(
        [thirdparty_binary("fstarcsort"), "--sort_type=ilabel", "-", clg_path],
        stdin=compose_proc.stdout,
        stderr=log_file,
        env=os.environ,
    )
    sort_proc.communicate()


def compose_hclg(
    model_directory: str,
    ilabels_temp: str,
    transition_scale: float,
    clg_path: str,
    hclga_path: str,
    log_file: TextIO,
) -> None:
    """
    Compost HCLG.fst for a dictionary

    Parameters
    ----------
    model_directory: str
        Model working directory with acoustic model information
    ilabels_temp: str
        Path to temporary ilabels file
    transition_scale: float
        Transition scale for the fst
    clg_path: str
        Path to CLG.fst file
    hclga_path: str
        Path to save HCLGa.fst file
    log_file: TextIO
        Log file handler to output logging info to
    """
    model_path = os.path.join(model_directory, "final.mdl")
    tree_path = os.path.join(model_directory, "tree")
    ha_path = hclga_path.replace("HCLGa", "Ha")
    ha_out_disambig = os.path.join(model_directory, "disambig_tid.int")
    make_h_proc = subprocess.Popen(
        [
            thirdparty_binary("make-h-transducer"),
            f"--disambig-syms-out={ha_out_disambig}",
            f"--transition-scale={transition_scale}",
            ilabels_temp,
            tree_path,
            model_path,
            ha_path,
        ],
        stderr=log_file,
        stdout=log_file,
        env=os.environ,
    )
    make_h_proc.communicate()

    temp_compose_path = hclga_path + ".temp"
    compose_proc = subprocess.Popen(
        [thirdparty_binary("fsttablecompose"), ha_path, clg_path, temp_compose_path],
        stderr=log_file,
        env=os.environ,
    )
    compose_proc.communicate()

    determinize_proc = subprocess.Popen(
        [thirdparty_binary("fstdeterminizestar"), "--use-log=true", temp_compose_path],
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    rmsymbols_proc = subprocess.Popen(
        [thirdparty_binary("fstrmsymbols"), ha_out_disambig],
        stdin=determinize_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    rmeps_proc = subprocess.Popen(
        [thirdparty_binary("fstrmepslocal")],
        stdin=rmsymbols_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    minimize_proc = subprocess.Popen(
        [thirdparty_binary("fstminimizeencoded"), "-", hclga_path],
        stdin=rmeps_proc.stdout,
        stderr=log_file,
        env=os.environ,
    )
    minimize_proc.communicate()
    os.remove(temp_compose_path)


def compose_g(arpa_path: str, words_path: str, g_path: str, log_file: TextIO) -> None:
    """
    Create G.fst from an ARPA formatted language model

    Parameters
    ----------
    arpa_path: str
        Path to ARPA file
    words_path: str
        Path to words symbols file
    g_path: str
        Path to output G.fst file
    log_file: TextIO
        Log file handler to output logging info to
    """
    arpafst_proc = subprocess.Popen(
        [
            thirdparty_binary("arpa2fst"),
            "--disambig-symbol=#0",
            f"--read-symbol-table={words_path}",
            arpa_path,
            g_path,
        ],
        stderr=log_file,
        stdout=log_file,
    )
    arpafst_proc.communicate()


def compose_g_carpa(
    in_carpa_path: str,
    temp_carpa_path: str,
    words_mapping: MappingType,
    carpa_path: str,
    log_file: TextIO,
):
    """
    Compose a large ARPA model into a G.carpa file

    Parameters
    ----------
    in_carpa_path: str
        Input ARPA model path
    temp_carpa_path: str
        Temporary CARPA model path
    words_mapping: Dict[str, int]
        Words symbols mapping
    carpa_path: str
        Path to save output G.carpa
    log_file: TextIO
        Log file handler to output logging info to
    """
    bos_symbol = words_mapping["<s>"]
    eos_symbol = words_mapping["</s>"]
    unk_symbol = words_mapping["<unk>"]
    with open(in_carpa_path, "r", encoding="utf8") as f, open(
        temp_carpa_path, "w", encoding="utf8"
    ) as outf:
        current_order = -1
        num_oov_lines = 0
        for line in f:
            line = line.strip()
            col = line.split()
            if current_order == -1 and not re.match(r"^\\data\\$", line):
                continue
            if re.match(r"^\\data\\$", line):
                log_file.write(r"Processing data...\n")
                current_order = 0
                outf.write(line + "\n")
            elif re.match(r"^\\[0-9]*-grams:$", line):
                current_order = int(re.sub(r"\\([0-9]*)-grams:$", r"\1", line))
                log_file.write(f"Processing {current_order} grams...\n")
                outf.write(line + "\n")
            elif re.match(r"^\\end\\$", line):
                outf.write(line + "\n")
            elif not line:
                if current_order >= 1:
                    outf.write("\n")
            else:
                if current_order == 0:
                    outf.write(line + "\n")
                else:
                    if len(col) > 2 + current_order or len(col) < 1 + current_order:
                        raise Exception(f'Bad line in arpa lm "{line}"')
                    prob = col.pop(0)
                    is_oov = False
                    for i in range(current_order):
                        try:
                            col[i] = str(words_mapping[col[i]])
                        except KeyError:
                            is_oov = True
                            num_oov_lines += 1
                            break
                    if not is_oov:
                        rest_of_line = " ".join(col)
                        outf.write(f"{prob}\t{rest_of_line}\n")
    carpa_proc = subprocess.Popen(
        [
            thirdparty_binary("arpa-to-const-arpa"),
            f"--bos-symbol={bos_symbol}",
            f"--eos-symbol={eos_symbol}",
            f"--unk-symbol={unk_symbol}",
            temp_carpa_path,
            carpa_path,
        ],
        stdin=subprocess.PIPE,
        stderr=log_file,
        stdout=log_file,
        env=os.environ,
    )
    carpa_proc.communicate()
    os.remove(temp_carpa_path)


def create_hclg_func(
    log_path: str,
    working_directory: str,
    path_template: str,
    words_path: str,
    carpa_path: str,
    small_arpa_path: str,
    medium_arpa_path: str,
    big_arpa_path: str,
    model_path: str,
    disambig_L_path: str,
    disambig_int_path: str,
    hclg_options: MetaDict,
    words_mapping: MappingType,
):
    """
    Create HCLG.fst file

    Parameters
    ----------
    log_path: str
        Path to log all stderr
    working_directory: str
        Current working directory
    path_template: str
        Path template to construct FST names from
    words_path: str
        Path to word symbols file
    carpa_path: str
        Path to G.carpa file
    small_arpa_path: str
        Path to small ARPA file
    medium_arpa_path:
        Path to medium ARPA file
    big_arpa_path:
        Path to big ARPA file
    model_path: str
        Path to acoustic model file
    disambig_L_path:
        Path to L_disambig.fst file
    disambig_int_path:
        Path to dictionary's disambiguation symbols file
    hclg_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Configuration options for composing HCLG.fst
    words_mapping: Dict[str, int]
        Word labels to integer ID mapping
    """
    hclg_path = path_template.format(file_name="HCLG")
    small_g_path = path_template.format(file_name="G.small")
    medium_g_path = path_template.format(file_name="G.med")
    lg_path = path_template.format(file_name="LG")
    hclga_path = path_template.format(file_name="HCLGa")
    if os.path.exists(hclg_path):
        return
    with open(log_path, "w") as log_file:
        context_width = hclg_options["context_width"]
        central_pos = hclg_options["central_pos"]

        clg_path = path_template.format(file_name=f"CLG_{context_width}_{central_pos}")
        ilabels_temp = path_template.format(
            file_name=f"ilabels_{context_width}_{central_pos}"
        ).replace(".fst", "")
        out_disambig = path_template.format(
            file_name=f"disambig_ilabels_{context_width}_{central_pos}"
        ).replace(".fst", ".int")

        log_file.write("Generating decoding graph...\n")
        if not os.path.exists(small_g_path):
            log_file.write("Generating small_G.fst...")
            compose_g(small_arpa_path, words_path, small_g_path, log_file)
        if not os.path.exists(medium_g_path):
            log_file.write("Generating med_G.fst...")
            compose_g(medium_arpa_path, words_path, medium_g_path, log_file)
        if not os.path.exists(carpa_path):
            log_file.write("Generating G.carpa...")
            temp_carpa_path = carpa_path + ".temp"
            compose_g_carpa(big_arpa_path, temp_carpa_path, words_mapping, carpa_path, log_file)
        if not os.path.exists(lg_path):
            log_file.write("Generating LG.fst...")
            compose_lg(disambig_L_path, small_g_path, lg_path, log_file)
        if not os.path.exists(clg_path):
            log_file.write("Generating CLG.fst...")
            compose_clg(
                disambig_int_path,
                out_disambig,
                context_width,
                central_pos,
                ilabels_temp,
                lg_path,
                clg_path,
                log_file,
            )
        if not os.path.exists(hclga_path):
            log_file.write("Generating HCLGa.fst...")
            compose_hclg(
                working_directory,
                ilabels_temp,
                hclg_options["transition_scale"],
                clg_path,
                hclga_path,
                log_file,
            )
        log_file.write("Generating HCLG.fst...")
        self_loop_proc = subprocess.Popen(
            [
                thirdparty_binary("add-self-loops"),
                f"--self-loop-scale={hclg_options['self_loop_scale']}",
                "--reorder=true",
                model_path,
                hclga_path,
            ],
            stderr=log_file,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        convert_proc = subprocess.Popen(
            [
                thirdparty_binary("fstconvert"),
                "--v=100",
                "--fst_type=const",
                "-",
                hclg_path,
            ],
            stdin=self_loop_proc.stdout,
            stderr=log_file,
            env=os.environ,
        )
        convert_proc.communicate()
        if os.path.exists(hclg_path):
            log_file.write(f"Done generating {hclg_path}!")
        else:
            log_file.write(f"There was an error in generating {hclg_path}")


def create_hclgs(transcriber: Transcriber):
    """
    Create HCLG.fst files for every dictionary being used by a :class:`~montreal_forced_aligner.transcriber.Transcriber`

    Parameters
    ----------
    transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
        Transcriber in use
    """
    dict_arguments = {}

    for j in transcriber.corpus.jobs:
        dict_arguments.update(j.create_hclgs_arguments(transcriber))
    dict_arguments = list(dict_arguments.values())
    if transcriber.transcribe_config.use_mp:
        run_mp(create_hclg_func, dict_arguments, transcriber.working_log_directory)
    else:
        run_non_mp(create_hclg_func, dict_arguments, transcriber.working_log_directory)
    error_logs = []
    for arg in dict_arguments:
        if not os.path.exists(arg.hclg_path):
            error_logs.append(arg.log_path)
        else:
            with open(arg.log_path, "r", encoding="utf8") as f:
                for line in f:
                    transcriber.logger.warning(line)
    if error_logs:
        raise KaldiProcessingError(error_logs)


def decode_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    decode_options: MetaDict,
    model_path: str,
    lat_paths: Dict[str, str],
    word_symbol_paths: Dict[str, str],
    hclg_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for performing decoding

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    decode_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for decoding
    model_path: str
        Path to acoustic model file
    lat_paths: Dict[str, str]
        Dictionary of lattice archive paths per dictionary name
    word_symbol_paths: Dict[str, str]
        Dictionary of word symbol paths per dictionary name
    hclg_paths: Dict[str, str]
        Dictionary of HCLG.fst paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            lat_path = lat_paths[dict_name]
            word_symbol_path = word_symbol_paths[dict_name]
            hclg_path = hclg_paths[dict_name]
            if os.path.exists(lat_path):
                continue
            if decode_options["fmllr"] and decode_options["first_beam"] is not None:
                beam = decode_options["first_beam"]
            else:
                beam = decode_options["beam"]
            if (
                decode_options["fmllr"]
                and decode_options["first_max_active"] is not None
                and not decode_options["ignore_speakers"]
            ):
                max_active = decode_options["first_max_active"]
            else:
                max_active = decode_options["max_active"]
            decode_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-latgen-faster"),
                    f"--max-active={max_active}",
                    f"--beam={beam}",
                    f"--lattice-beam={decode_options['lattice_beam']}",
                    "--allow-partial=true",
                    f"--word-symbol-table={word_symbol_path}",
                    f"--acoustic-scale={decode_options['acoustic_scale']}",
                    model_path,
                    hclg_path,
                    feature_string,
                    f"ark:{lat_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            decode_proc.communicate()


def score_func(
    log_path: str,
    dictionaries: List[str],
    score_options: MetaDict,
    lat_paths: Dict[str, str],
    rescored_lat_paths: Dict[str, str],
    carpa_rescored_lat_paths: Dict[str, str],
    words_paths: Dict[str, str],
    tra_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for scoring lattices

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    score_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for scoring
    lat_paths: Dict[str, str]
        Dictionary of lattice archive paths per dictionary name
    rescored_lat_paths: Dict[str, str]
        Dictionary of medium G.fst rescored lattice archive paths per dictionary name
    carpa_rescored_lat_paths: Dict[str, str]
        Dictionary of carpa-rescored lattice archive paths per dictionary name
    words_paths: Dict[str, str]
        Dictionary of word symbol paths per dictionary name
    tra_paths: Dict[str, str]
        Dictionary of transcription archive paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            language_model_weight = score_options["language_model_weight"]
            word_insertion_penalty = score_options["word_insertion_penalty"]
            carpa_rescored_lat_path = carpa_rescored_lat_paths[dict_name]
            rescored_lat_path = rescored_lat_paths[dict_name]
            lat_path = lat_paths[dict_name]
            words_path = words_paths[dict_name]
            tra_path = tra_paths[dict_name]
            if os.path.exists(carpa_rescored_lat_path):
                lat_path = carpa_rescored_lat_path
            elif os.path.exists(rescored_lat_path):
                lat_path = rescored_lat_path
            scale_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-scale"),
                    f"--inv-acoustic-scale={language_model_weight}",
                    f"ark:{lat_path}",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            penalty_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-add-penalty"),
                    f"--word-ins-penalty={word_insertion_penalty}",
                    "ark:-",
                    "ark:-",
                ],
                stdin=scale_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            best_path_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-best-path"),
                    f"--word-symbol-table={words_path}",
                    "ark:-",
                    f"ark,t:{tra_path}",
                ],
                stdin=penalty_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            best_path_proc.communicate()


def lm_rescore_func(
    log_path: str,
    dictionaries: List[str],
    lm_rescore_options: MetaDict,
    lat_paths: Dict[str, str],
    rescored_lat_paths: Dict[str, str],
    old_g_paths: Dict[str, str],
    new_g_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function rescore lattices by replacing the small G.fst with the medium G.fst

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    lm_rescore_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for rescoring
    lat_paths: Dict[str, str]
        Dictionary of lattice archive paths per dictionary name
    rescored_lat_paths: Dict[str, str]
        Dictionary of rescored lattice archive paths per dictionary name
    old_g_paths: Dict[str, str]
        Dictionary of small G.fst paths per dictionary name
    new_g_paths: Dict[str, str]
        Dictionary of medium G.fst paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            lat_path = lat_paths[dict_name]
            rescored_lat_path = rescored_lat_paths[dict_name]
            old_g_path = old_g_paths[dict_name]
            new_g_path = new_g_paths[dict_name]
            if sys.platform == "win32":
                project_type_arg = "--project_output=true"
            else:
                project_type_arg = "--project_type=output"
            if os.path.exists(rescored_lat_path):
                continue

            lattice_scale_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-lmrescore-pruned"),
                    f"--acoustic-scale={lm_rescore_options['acoustic_scale']}",
                    f"fstproject {project_type_arg} {old_g_path} |",
                    f"fstproject {project_type_arg} {new_g_path} |",
                    f"ark:{lat_path}",
                    f"ark:{rescored_lat_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            lattice_scale_proc.communicate()


def carpa_lm_rescore_func(
    log_path: str,
    dictionaries: List[str],
    lat_paths: Dict[str, str],
    rescored_lat_paths: Dict[str, str],
    old_g_paths: Dict[str, str],
    new_g_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function to rescore lattices by replacing medium G.fst with large G.carpa

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    lat_paths: Dict[str, str]
        PronunciationDictionary of lattice archive paths per dictionary name
    rescored_lat_paths: Dict[str, str]
        PronunciationDictionary of rescored lattice archive paths per dictionary name
    old_g_paths: Dict[str, str]
        PronunciationDictionary of medium G.fst paths per dictionary name
    new_g_paths: Dict[str, str]
        PronunciationDictionary of large G.carpa paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            if sys.platform == "win32":
                project_type_arg = "--project_output=true"
            else:
                project_type_arg = "--project_type=output"
            lat_path = lat_paths[dict_name]
            rescored_lat_path = rescored_lat_paths[dict_name]
            old_g_path = old_g_paths[dict_name]
            new_g_path = new_g_paths[dict_name]
            if os.path.exists(rescored_lat_path):
                continue
            project_proc = subprocess.Popen(
                [thirdparty_binary("fstproject"), project_type_arg, old_g_path],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            lmrescore_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-lmrescore"),
                    "--lm-scale=-1.0",
                    f"ark:{lat_path}",
                    "-",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stdin=project_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            lmrescore_const_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-lmrescore-const-arpa"),
                    "--lm-scale=1.0",
                    "ark:-",
                    new_g_path,
                    f"ark:{rescored_lat_path}",
                ],
                stdin=lmrescore_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            lmrescore_const_proc.communicate()


def transcribe(transcriber: Transcriber) -> None:
    """
    Transcribe a corpus using a Transcriber

    Parameters
    ----------
    transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
        Transcriber
    """
    config = transcriber.transcribe_config

    jobs = [x.decode_arguments(transcriber) for x in transcriber.corpus.jobs]

    if config.use_mp:
        run_mp(decode_func, jobs, transcriber.working_log_directory)
    else:
        run_non_mp(decode_func, jobs, transcriber.working_log_directory)

    jobs = [x.lm_rescore_arguments(transcriber) for x in transcriber.corpus.jobs]

    if config.use_mp:
        run_mp(lm_rescore_func, jobs, transcriber.working_log_directory)
    else:
        run_non_mp(lm_rescore_func, jobs, transcriber.working_log_directory)

    jobs = [x.carpa_lm_rescore_arguments(transcriber) for x in transcriber.corpus.jobs]

    if config.use_mp:
        run_mp(carpa_lm_rescore_func, jobs, transcriber.working_log_directory)
    else:
        run_non_mp(carpa_lm_rescore_func, jobs, transcriber.working_log_directory)


def score_transcriptions(transcriber: Transcriber):
    """
    Score transcriptions if reference text is available in the corpus

    Parameters
    ----------
    transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
        Transcriber
    """
    if transcriber.evaluation_mode:
        best_wer = 10000
        best = None
        for lmwt in range(
            transcriber.min_language_model_weight, transcriber.max_language_model_weight
        ):
            for wip in transcriber.word_insertion_penalties:
                transcriber.transcribe_config.language_model_weight = lmwt
                transcriber.transcribe_config.word_insertion_penalty = wip
                os.makedirs(transcriber.evaluation_log_directory, exist_ok=True)

                jobs = [x.score_arguments(transcriber) for x in transcriber.corpus.jobs]
                if transcriber.transcribe_config.use_mp:
                    run_mp(score_func, jobs, transcriber.evaluation_log_directory)
                else:
                    run_non_mp(score_func, jobs, transcriber.evaluation_log_directory)
                ser, wer = transcriber.evaluate()
                if wer < best_wer:
                    best = (lmwt, wip)
        transcriber.transcribe_config.language_model_weight = best[0]
        transcriber.transcribe_config.word_insertion_penalty = best[1]
        for j in transcriber.corpus.jobs:
            score_args = j.score_arguments(transcriber)
            for p in score_args.tra_paths.values():
                shutil.copyfile(
                    p,
                    p.replace(transcriber.evaluation_directory, transcriber.transcribe_directory),
                )
    else:
        jobs = [x.score_arguments(transcriber) for x in transcriber.corpus.jobs]
        if transcriber.transcribe_config.use_mp:
            run_mp(score_func, jobs, transcriber.working_log_directory)
        else:
            run_non_mp(score_func, jobs, transcriber.working_log_directory)


def initial_fmllr_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    model_path: str,
    fmllr_options: MetaDict,
    trans_paths: Dict[str, str],
    lat_paths: Dict[str, str],
    spk2utt_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for running initial fMLLR calculation

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    model_path: str
        Path to acoustic model file
    fmllr_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for calculating fMLLR transforms
    trans_paths: Dict[str, str]
        Dictionary of transform archives per dictionary name
    lat_paths: Dict[str, str]
        Dictionary of lattice archive paths per dictionary name
    spk2utt_paths: Dict[str, str]
        Dictionary of spk2utt scp files per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            lat_path = lat_paths[dict_name]
            feature_string = feature_strings[dict_name]
            spk2utt_path = spk2utt_paths[dict_name]
            trans_path = trans_paths[dict_name]

            latt_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-to-post"),
                    f"--acoustic-scale={fmllr_options['acoustic_scale']}",
                    f"ark:{lat_path}",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            weight_silence_proc = subprocess.Popen(
                [
                    thirdparty_binary("weight-silence-post"),
                    f"{fmllr_options['silence_weight']}",
                    fmllr_options["sil_phones"],
                    model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=latt_post_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            gmm_gpost_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-post-to-gpost"),
                    model_path,
                    feature_string,
                    "ark:-",
                    "ark:-",
                ],
                stdin=weight_silence_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            fmllr_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est-fmllr-gpost"),
                    f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                    f"--spk2utt=ark:{spk2utt_path}",
                    model_path,
                    feature_string,
                    "ark,s,cs:-",
                    f"ark:{trans_path}",
                ],
                stdin=gmm_gpost_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            fmllr_proc.communicate()


def lat_gen_fmllr_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    model_path: str,
    decode_options: MetaDict,
    word_symbol_paths: Dict[str, str],
    hclg_paths: Dict[str, str],
    tmp_lat_paths: Dict[str, str],
) -> None:
    """
    Regenerate lattices using initial fMLLR transforms

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    model_path: str
        Path to acoustic model file
    decode_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for decoding
    word_symbol_paths: Dict[str, str]
        Dictionary of word symbol paths per dictionary name
    hclg_paths: Dict[str, str]
        Dictionary of HCLG.fst paths per dictionary name
    tmp_lat_paths: Dict[str, str]
        Dictionary of temporary lattice archive paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            words_path = word_symbol_paths[dict_name]
            hclg_path = hclg_paths[dict_name]
            tmp_lat_path = tmp_lat_paths[dict_name]
            lat_gen_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-latgen-faster"),
                    f"--max-active={decode_options['max_active']}",
                    f"--beam={decode_options['beam']}",
                    f"--lattice-beam={decode_options['lattice_beam']}",
                    f"--acoustic-scale={decode_options['acoustic_scale']}",
                    "--determinize-lattice=false",
                    "--allow-partial=true",
                    f"--word-symbol-table={words_path}",
                    model_path,
                    hclg_path,
                    feature_string,
                    f"ark:{tmp_lat_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )

            lat_gen_proc.communicate()


def final_fmllr_est_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    model_path: str,
    fmllr_options: MetaDict,
    trans_paths: Dict[str, str],
    spk2utt_paths: Dict[str, str],
    tmp_lat_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function for running final fMLLR estimation

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    model_path: str
        Path to acoustic model file
    fmllr_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for calculating fMLLR transforms
    trans_paths: Dict[str, str]
        Dictionary of transform archives per dictionary name
    spk2utt_paths: Dict[str, str]
        Dictionary of spk2utt scp files per dictionary name
    tmp_lat_paths: Dict[str, str]
        Dictionary of temporary lattice archive paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            trans_path = trans_paths[dict_name]
            temp_trans_path = trans_path + ".temp"
            spk2utt_path = spk2utt_paths[dict_name]
            tmp_lat_path = tmp_lat_paths[dict_name]
            determinize_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-determinize-pruned"),
                    f"--acoustic-scale={fmllr_options['acoustic_scale']}",
                    "--beam=4.0",
                    f"ark:{tmp_lat_path}",
                    "ark:-",
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            latt_post_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-to-post"),
                    f"--acoustic-scale={fmllr_options['acoustic_scale']}",
                    "ark:-",
                    "ark:-",
                ],
                stdin=determinize_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            weight_silence_proc = subprocess.Popen(
                [
                    thirdparty_binary("weight-silence-post"),
                    f"{fmllr_options['silence_weight']}",
                    fmllr_options["sil_phones"],
                    model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=latt_post_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            fmllr_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-est-fmllr"),
                    f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                    f"--spk2utt=ark:{spk2utt_path}",
                    model_path,
                    feature_string,
                    "ark,s,cs:-",
                    "ark:-",
                ],
                stdin=weight_silence_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            compose_proc = subprocess.Popen(
                [
                    thirdparty_binary("compose-transforms"),
                    "--b-is-affine=true",
                    "ark:-",
                    f"ark:{trans_path}",
                    f"ark:{temp_trans_path}",
                ],
                stderr=log_file,
                stdin=fmllr_proc.stdout,
                env=os.environ,
            )
            compose_proc.communicate()
            os.remove(trans_path)
            os.rename(temp_trans_path, trans_path)


def fmllr_rescore_func(
    log_path: str,
    dictionaries: List[str],
    feature_strings: Dict[str, str],
    model_path: str,
    fmllr_options: MetaDict,
    tmp_lat_paths: Dict[str, str],
    final_lat_paths: Dict[str, str],
) -> None:
    """
    Multiprocessing function to rescore lattices following fMLLR estimation

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        Dictionary of feature strings per dictionary name
    model_path: str
        Path to acoustic model file
    fmllr_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for calculating fMLLR transforms
    tmp_lat_paths: Dict[str, str]
        Dictionary of temporary lattice archive paths per dictionary name
    final_lat_paths: Dict[str, str]
        Dictionary of lattice archive paths per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            tmp_lat_path = tmp_lat_paths[dict_name]
            final_lat_path = final_lat_paths[dict_name]
            rescore_proc = subprocess.Popen(
                [
                    thirdparty_binary("gmm-rescore-lattice"),
                    model_path,
                    f"ark:{tmp_lat_path}",
                    feature_string,
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            determinize_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-determinize-pruned"),
                    f"--acoustic-scale={fmllr_options['acoustic_scale']}",
                    f"--beam={fmllr_options['lattice_beam']}",
                    "ark:-",
                    f"ark:{final_lat_path}",
                ],
                stdin=rescore_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )

            determinize_proc.communicate()


def transcribe_fmllr(transcriber: Transcriber) -> None:
    """
    Run fMLLR estimation over initial decoding lattices and rescore

    Parameters
    ----------
    transcriber: :class:`~montreal_forced_aligner.transcriber.Transcriber`
        Transcriber
    """
    jobs = [x.initial_fmllr_arguments(transcriber) for x in transcriber.corpus.jobs]

    run_non_mp(initial_fmllr_func, jobs, transcriber.working_log_directory)
    transcriber.speaker_independent = False

    jobs = [x.lat_gen_fmllr_arguments(transcriber) for x in transcriber.corpus.jobs]

    run_non_mp(lat_gen_fmllr_func, jobs, transcriber.working_log_directory)

    jobs = [x.final_fmllr_arguments(transcriber) for x in transcriber.corpus.jobs]

    run_non_mp(final_fmllr_est_func, jobs, transcriber.working_log_directory)

    jobs = [x.fmllr_rescore_arguments(transcriber) for x in transcriber.corpus.jobs]

    if transcriber.transcribe_config.use_mp:
        run_mp(fmllr_rescore_func, jobs, transcriber.working_log_directory)
    else:
        run_non_mp(fmllr_rescore_func, jobs, transcriber.working_log_directory)

    jobs = [x.lm_rescore_arguments(transcriber) for x in transcriber.corpus.jobs]

    if transcriber.transcribe_config.use_mp:
        run_mp(lm_rescore_func, jobs, transcriber.working_log_directory)
    else:
        run_non_mp(lm_rescore_func, jobs, transcriber.working_log_directory)

    jobs = [x.carpa_lm_rescore_arguments(transcriber) for x in transcriber.corpus.jobs]

    if transcriber.transcribe_config.use_mp:
        run_mp(carpa_lm_rescore_func, jobs, transcriber.working_log_directory)
    else:
        run_non_mp(carpa_lm_rescore_func, jobs, transcriber.working_log_directory)
