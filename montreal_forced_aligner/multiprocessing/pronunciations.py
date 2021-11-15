"""
Pronunciation probability functions
-----------------------------------

"""
from __future__ import annotations

import os
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from ..abc import Aligner
from ..utils import thirdparty_binary
from .helper import run_mp, run_non_mp

__all__ = ["generate_pronunciations", "generate_pronunciations_func"]


def generate_pronunciations_func(
    log_path: str,
    dictionaries: List[str],
    text_int_paths: Dict[str, str],
    word_boundary_paths: Dict[str, str],
    ali_paths: Dict[str, str],
    model_path: str,
    pron_paths: Dict[str, str],
):
    """
    Multiprocessing function for generating pronunciations

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    text_int_paths: Dict[str, str]
        Dictionary of text int files per dictionary name
    word_boundary_paths: Dict[str, str]
        Dictionary of word boundary files per dictionary name
    ali_paths: Dict[str, str]
        Dictionary of alignment archives per dictionary name
    model_path: str
        Path to acoustic model file
    pron_paths: Dict[str, str]
        Dictionary of pronunciation archives per dictionary name
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        for dict_name in dictionaries:
            text_int_path = text_int_paths[dict_name]
            word_boundary_path = word_boundary_paths[dict_name]
            ali_path = ali_paths[dict_name]
            pron_path = pron_paths[dict_name]

            lin_proc = subprocess.Popen(
                [
                    thirdparty_binary("linear-to-nbest"),
                    f"ark:{ali_path}",
                    f"ark:{text_int_path}",
                    "",
                    "",
                    "ark:-",
                ],
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            align_proc = subprocess.Popen(
                [
                    thirdparty_binary("lattice-align-words"),
                    word_boundary_path,
                    model_path,
                    "ark:-",
                    "ark:-",
                ],
                stdin=lin_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )

            prons_proc = subprocess.Popen(
                [thirdparty_binary("nbest-to-prons"), model_path, "ark:-", pron_path],
                stdin=align_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            prons_proc.communicate()


def generate_pronunciations(
    aligner: Aligner,
) -> Tuple[Dict[str, defaultdict[Counter]], Dict[str, Dict[str, List[str, ...]]]]:
    """
    Generates pronunciations based on alignments for a corpus and calculates pronunciation probabilities

    Parameters
    ----------
    aligner: :class:`~montreal_forced_aligner.aligner.pretrained.PretrainedAligner`
        Aligner
    """
    jobs = [x.generate_pronunciations_arguments(aligner) for x in aligner.corpus.jobs]
    if aligner.align_config.use_mp:
        run_mp(generate_pronunciations_func, jobs, aligner.working_log_directory)
    else:
        run_non_mp(generate_pronunciations_func, jobs, aligner.working_log_directory)
    pron_counts = {}
    utt_mapping = {}
    for j in aligner.corpus.jobs:
        args = j.generate_pronunciations_arguments(aligner)
        dict_data = j.dictionary_data()
        for dict_name, pron_path in args.pron_paths.items():
            if dict_name not in pron_counts:
                pron_counts[dict_name] = defaultdict(Counter)
                utt_mapping[dict_name] = {}
            word_lookup = dict_data[dict_name].reversed_words_mapping
            phone_lookup = dict_data[dict_name].reversed_phone_mapping
            with open(pron_path, "r", encoding="utf8") as f:
                last_utt = None
                for line in f:
                    line = line.split()
                    utt = line[0]
                    if utt not in utt_mapping[dict_name]:
                        if last_utt is not None:
                            utt_mapping[dict_name][last_utt].append("</s>")
                        utt_mapping[dict_name][utt] = ["<s>"]
                        last_utt = utt

                    word = word_lookup[int(line[3])]
                    if word == "<eps>":
                        utt_mapping[dict_name][utt].append(word)
                    else:
                        pron = tuple(phone_lookup[int(x)].split("_")[0] for x in line[4:])
                        pron_string = " ".join(pron)
                        utt_mapping[dict_name][utt].append(word + " " + pron_string)
                        pron_counts[dict_name][word][pron] += 1
    return pron_counts, utt_mapping
