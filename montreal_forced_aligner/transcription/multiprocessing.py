"""
Transcription functions
-----------------------

"""
from __future__ import annotations

import os
import re
import subprocess
import typing
from typing import TYPE_CHECKING, Dict, List, TextIO

import pynini
from sqlalchemy.orm import Session, joinedload, subqueryload

from montreal_forced_aligner.abc import KaldiFunction, MetaDict
from montreal_forced_aligner.data import MfaArguments
from montreal_forced_aligner.db import Job, Phone, Utterance
from montreal_forced_aligner.helper import mfa_open
from montreal_forced_aligner.utils import thirdparty_binary

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from dataclassy import dataclass


__all__ = [
    "compose_g",
    "compose_lg",
    "compose_clg",
    "compose_hclg",
    "compose_g_carpa",
    "FmllrRescoreFunction",
    "FinalFmllrFunction",
    "InitialFmllrFunction",
    "LatGenFmllrFunction",
    "CarpaLmRescoreFunction",
    "DecodeFunction",
    "LmRescoreFunction",
    "CreateHclgFunction",
]


@dataclass
class CreateHclgArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    working_directory: str
        Current working directory
    path_template: str
        Path template for intermediate files
    words_path: str
        Path to words symbol table
    carpa_path: str
        Path to .carpa file
    small_arpa_path: str
        Path to small ARPA file
    medium_arpa_path: str
        Path to medium ARPA file
    big_arpa_path: str
        Path to big ARPA file
    model_path: str
        Acoustic model path
    disambig_L_path: str
        Path to disambiguated lexicon file
    disambig_int_path: str
        Path to disambiguation symbol integer file
    hclg_options: dict[str, Any]
        HCLG options
    words_mapping: dict[str, int]
        Words mapping
    """

    working_directory: str
    path_template: str
    words_path: str
    carpa_path: str
    small_arpa_path: str
    medium_arpa_path: str
    big_arpa_path: str
    model_path: str
    disambig_L_path: str
    disambig_int_path: str
    hclg_options: MetaDict
    words_mapping: Dict[str, int]

    @property
    def hclg_path(self) -> str:
        """Path to HCLG FST file"""
        return self.path_template.format(file_name="HCLG")


@dataclass
class DecodeArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    decode_options: dict[str, Any]
        Decoding options
    model_path: str
        Path to model file
    lat_paths: dict[int, str]
        Per dictionary lattice paths
    word_symbol_paths: dict[int, str]
        Per dictionary word symbol table paths
    hclg_paths: dict[int, str]
        Per dictionary HCLG.fst paths
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    decode_options: MetaDict
    model_path: str
    lat_paths: Dict[int, str]
    word_symbol_paths: Dict[int, str]
    hclg_paths: Dict[int, str]


@dataclass
class DecodePhoneArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.validation.corpus_validator.DecodePhoneFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    decode_options: dict[str, Any]
        Decoding options
    model_path: str
        Path to model file
    lat_paths: dict[int, str]
        Per dictionary lattice paths
    phone_symbol_path: str
        Phone symbol table paths
    hclg_path: str
        HCLG.fst paths
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    decode_options: MetaDict
    model_path: str
    lat_paths: Dict[int, str]
    phone_symbol_path: str
    hclg_path: str


@dataclass
class LmRescoreArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    lm_rescore_options: dict[str, Any]
        Rescoring options
    lat_paths: dict[int, str]
        Per dictionary lattice paths
    rescored_lat_paths: dict[int, str]
        Per dictionary rescored lattice paths
    old_g_paths: dict[int, str]
        Mapping of dictionaries to small G.fst paths
    new_g_paths: dict[int, str]
        Mapping of dictionaries to medium G.fst paths
    """

    dictionaries: List[int]
    lm_rescore_options: MetaDict
    lat_paths: Dict[int, str]
    rescored_lat_paths: Dict[int, str]
    old_g_paths: Dict[int, str]
    new_g_paths: Dict[int, str]


@dataclass
class CarpaLmRescoreArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    lat_paths: dict[int, str]
        Per dictionary lattice paths
    rescored_lat_paths: dict[int, str]
        Per dictionary rescored lattice paths
    old_g_paths: dict[int, str]
        Mapping of dictionaries to medium G.fst paths
    new_g_paths: dict[int, str]
        Mapping of dictionaries to G.carpa paths
    """

    dictionaries: List[int]
    lat_paths: Dict[int, str]
    rescored_lat_paths: Dict[int, str]
    old_g_paths: Dict[int, str]
    new_g_paths: Dict[int, str]


@dataclass
class InitialFmllrArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: str
        Path to model file
    fmllr_options: dict[str, Any]
        fMLLR options
    pre_trans_paths: dict[int, str]
        Per dictionary pre-fMLLR lattice paths
    lat_paths: dict[int, str]
        Per dictionary lattice paths
    spk2utt_paths: dict[int, str]
        Per dictionary speaker to utterance mapping paths
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    model_path: str
    fmllr_options: MetaDict
    pre_trans_paths: Dict[int, str]
    lat_paths: Dict[int, str]
    spk2utt_paths: Dict[int, str]


@dataclass
class LatGenFmllrArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: str
        Path to model file
    decode_options: dict[str, Any]
        Decoding options
    hclg_paths: dict[int, str]
        Per dictionary HCLG.fst paths
    tmp_lat_paths: dict[int, str]
        Per dictionary temporary lattice paths
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    model_path: str
    decode_options: MetaDict
    word_symbol_paths: Dict[int, str]
    hclg_paths: typing.Union[Dict[int, str], str]
    tmp_lat_paths: Dict[int, str]


@dataclass
class FinalFmllrArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: str
        Path to model file
    fmllr_options: dict[str, Any]
        fMLLR options
    trans_paths: dict[int, str]
        Per dictionary transform paths
    spk2utt_paths: dict[int, str]
        Per dictionary speaker to utterance mapping paths
    tmp_lat_paths: dict[int, str]
        Per dictionary temporary lattice paths
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    model_path: str
    fmllr_options: MetaDict
    trans_paths: Dict[int, str]
    spk2utt_paths: Dict[int, str]
    tmp_lat_paths: Dict[int, str]


@dataclass
class FmllrRescoreArguments(MfaArguments):
    """
    Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`

    Parameters
    ----------
    job_name: int
        Integer ID of the job
    db_string: str
        String for database connections
    log_path: str
        Path to save logging information during the run
    dictionaries: list[int]
        List of dictionary ids
    feature_strings: dict[int, str]
        Mapping of dictionaries to feature generation strings
    model_path: str
        Path to model file
    fmllr_options: dict[str, Any]
        fMLLR options
    tmp_lat_paths: dict[int, str]
        Per dictionary temporary lattice paths
    final_lat_paths: dict[int, str]
        Per dictionary lattice paths
    """

    dictionaries: List[int]
    feature_strings: Dict[int, str]
    model_path: str
    fmllr_options: MetaDict
    tmp_lat_paths: Dict[int, str]
    final_lat_paths: Dict[int, str]


def compose_lg(dictionary_path: str, small_g_path: str, lg_path: str, log_file: TextIO) -> None:
    """
    Compose an LG.fst

    See Also
    --------
    :kaldi_src:`fsttablecompose`
        Relevant Kaldi binary
    :kaldi_src:`fstdeterminizestar`
        Relevant Kaldi binary
    :kaldi_src:`fstminimizeencoded`
        Relevant Kaldi binary
    :kaldi_src:`fstpushspecial`
        Relevant Kaldi binary


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
    compose_proc = subprocess.Popen(
        [thirdparty_binary("fsttablecompose"), dictionary_path, small_g_path],
        stderr=log_file,
        stdout=subprocess.PIPE,
        env=os.environ,
    )

    determinize_proc = subprocess.Popen(
        [
            thirdparty_binary("fstdeterminizestar"),
            "--use-log=true",
        ],
        stdin=compose_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )

    minimize_proc = subprocess.Popen(
        [thirdparty_binary("fstminimizeencoded")],
        stdin=determinize_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )

    push_proc = subprocess.Popen(
        [thirdparty_binary("fstpushspecial"), "-", lg_path],
        stdin=minimize_proc.stdout,
        stderr=log_file,
        env=os.environ,
    )
    push_proc.communicate()


def compose_clg(
    in_disambig: typing.Optional[str],
    out_disambig: typing.Optional[str],
    context_width: int,
    central_pos: int,
    ilabels_temp: str,
    lg_path: str,
    clg_path: str,
    log_file: TextIO,
) -> None:
    """
    Compose a CLG.fst

    See Also
    --------
    :kaldi_src:`fstcomposecontext`
        Relevant Kaldi binary
    :openfst_src:`fstarcsort`
        Relevant OpenFst binary

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
    com = [
        thirdparty_binary("fstcomposecontext"),
        f"--context-size={context_width}",
        f"--central-position={central_pos}",
    ]
    if in_disambig:
        com.append(f"--read-disambig-syms={in_disambig}")
    if out_disambig:
        com.append(f"--write-disambig-syms={out_disambig}")
    com.extend([ilabels_temp, lg_path])
    compose_proc = subprocess.Popen(
        com,
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
    model_path: str,
    ilabels_temp: str,
    transition_scale: float,
    clg_path: str,
    hclga_path: str,
    log_file: TextIO,
) -> None:
    """
    Compost HCLG.fst for a dictionary

    See Also
    --------
    :kaldi_src:`make-h-transducer`
        Relevant Kaldi binary
    :kaldi_src:`fsttablecompose`
        Relevant Kaldi binary
    :kaldi_src:`fstdeterminizestar`
        Relevant Kaldi binary
    :kaldi_src:`fstrmsymbols`
        Relevant Kaldi binary
    :kaldi_src:`fstrmepslocal`
        Relevant Kaldi binary
    :kaldi_src:`fstminimizeencoded`
        Relevant Kaldi binary
    :openfst_src:`fstarcsort`
        Relevant OpenFst binary

    Parameters
    ----------
    model_path: str
        Path to acoustic model
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
    tree_path = model_path.replace("final.mdl", "tree")
    ha_path = hclga_path.replace("HCLGa", "Ha")
    ha_out_disambig = hclga_path.replace("HCLGa", "disambig_tid")
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

    compose_proc = subprocess.Popen(
        [thirdparty_binary("fsttablecompose"), ha_path, clg_path],
        stderr=log_file,
        stdout=subprocess.PIPE,
        env=os.environ,
    )

    determinize_proc = subprocess.Popen(
        [thirdparty_binary("fstdeterminizestar"), "--use-log=true"],
        stdin=compose_proc.stdout,
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


def compose_g(arpa_path: str, words_path: str, g_path: str, log_file: TextIO) -> None:
    """
    Create G.fst from an ARPA formatted language model

    See Also
    --------
    :kaldi_src:`arpa2fst`
        Relevant Kaldi binary

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
    words_mapping: Dict[str, int],
    carpa_path: str,
    log_file: TextIO,
):
    """
    Compose a large ARPA model into a G.carpa file

    See Also
    --------
    :kaldi_src:`arpa-to-const-arpa`
        Relevant Kaldi binary

    Parameters
    ----------
    in_carpa_path: str
        Input ARPA model path
    temp_carpa_path: str
        Temporary CARPA model path
    words_mapping: dict[str, int]
        Words symbols mapping
    carpa_path: str
        Path to save output G.carpa
    log_file: TextIO
        Log file handler to output logging info to
    """
    bos_symbol = words_mapping["<s>"]
    eos_symbol = words_mapping["</s>"]
    unk_symbol = words_mapping["<unk>"]
    with mfa_open(in_carpa_path, "r") as f, mfa_open(temp_carpa_path, "w") as outf:
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


class CreateHclgFunction(KaldiFunction):
    """
    Create HCLG.fst file

    See Also
    --------
    :meth:`.Transcriber.create_hclgs`
        Main function that calls this function in parallel
    :meth:`.Transcriber.create_hclgs_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`add-self-loops`
        Relevant Kaldi binary
    :openfst_src:`fstconvert`
        Relevant OpenFst binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgArguments`
        Arguments for the function
    """

    def __init__(self, args: CreateHclgArguments):
        super().__init__(args)
        self.working_directory = args.working_directory
        self.path_template = args.path_template
        self.words_path = args.words_path
        self.carpa_path = args.carpa_path
        self.small_arpa_path = args.small_arpa_path
        self.medium_arpa_path = args.medium_arpa_path
        self.big_arpa_path = args.big_arpa_path
        self.model_path = args.model_path
        self.disambig_L_path = args.disambig_L_path
        self.disambig_int_path = args.disambig_int_path
        self.hclg_options = args.hclg_options
        self.words_mapping = args.words_mapping

    def _run(self) -> typing.Generator[typing.Tuple[bool, str]]:
        """Run the function"""
        hclg_path = self.path_template.format(file_name="HCLG")
        small_g_path = self.path_template.format(file_name="G.small")
        medium_g_path = self.path_template.format(file_name="G.med")
        lg_path = self.path_template.format(file_name="LG")
        hclga_path = self.path_template.format(file_name="HCLGa")
        if os.path.exists(hclg_path):
            return
        with mfa_open(self.log_path, "w") as log_file:
            context_width = self.hclg_options["context_width"]
            central_pos = self.hclg_options["central_pos"]

            clg_path = self.path_template.format(file_name=f"CLG_{context_width}_{central_pos}")
            ilabels_temp = self.path_template.format(
                file_name=f"ilabels_{context_width}_{central_pos}"
            ).replace(".fst", "")
            out_disambig = self.path_template.format(
                file_name=f"disambig_ilabels_{context_width}_{central_pos}"
            ).replace(".fst", ".int")

            log_file.write("Generating decoding graph...\n")
            if not os.path.exists(small_g_path):
                log_file.write("Generating small_G.fst...")
                compose_g(self.small_arpa_path, self.words_path, small_g_path, log_file)
                yield 1
            if not os.path.exists(medium_g_path):
                log_file.write("Generating med_G.fst...")
                compose_g(self.medium_arpa_path, self.words_path, medium_g_path, log_file)
                yield 1
            if not os.path.exists(self.carpa_path):
                log_file.write("Generating G.carpa...")
                temp_carpa_path = self.carpa_path + ".temp"
                compose_g_carpa(
                    self.big_arpa_path,
                    temp_carpa_path,
                    self.words_mapping,
                    self.carpa_path,
                    log_file,
                )
                yield 1
            if not os.path.exists(lg_path):
                log_file.write("Generating LG.fst...")
                compose_lg(self.disambig_L_path, small_g_path, lg_path, log_file)
                yield 1
            if not os.path.exists(clg_path):
                log_file.write("Generating CLG.fst...")
                compose_clg(
                    self.disambig_int_path,
                    out_disambig,
                    context_width,
                    central_pos,
                    ilabels_temp,
                    lg_path,
                    clg_path,
                    log_file,
                )
                yield 1
            if not os.path.exists(hclga_path):
                log_file.write("Generating HCLGa.fst...")
                compose_hclg(
                    self.model_path,
                    ilabels_temp,
                    self.hclg_options["transition_scale"],
                    clg_path,
                    hclga_path,
                    log_file,
                )
                yield 1
            log_file.write("Generating HCLG.fst...")
            self_loop_proc = subprocess.Popen(
                [
                    thirdparty_binary("add-self-loops"),
                    f"--self-loop-scale={self.hclg_options['self_loop_scale']}",
                    "--reorder=true",
                    self.model_path,
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
            self.check_call(convert_proc)
            if os.path.exists(hclg_path):
                yield True, hclg_path
            else:
                yield False, hclg_path


class DecodeFunction(KaldiFunction):
    """
    Multiprocessing function for performing decoding

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.decode_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Log-like per frame for utterance (?P<utterance>.*) is (?P<loglike>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: DecodeArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.lat_paths = args.lat_paths
        self.word_symbol_paths = args.word_symbol_paths
        self.hclg_paths = args.hclg_paths
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def _run(self) -> typing.Generator[typing.Tuple[str, float, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                lat_path = self.lat_paths[dict_id]
                word_symbol_path = self.word_symbol_paths[dict_id]
                hclg_path = self.hclg_paths[dict_id]
                if os.path.exists(lat_path):
                    continue
                if (
                    self.decode_options["uses_speaker_adaptation"]
                    and self.decode_options["first_beam"] is not None
                ):
                    beam = self.decode_options["first_beam"]
                else:
                    beam = self.decode_options["beam"]
                if (
                    self.decode_options["uses_speaker_adaptation"]
                    and self.decode_options["first_max_active"] is not None
                ):
                    max_active = self.decode_options["first_max_active"]
                else:
                    max_active = self.decode_options["max_active"]
                decode_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-latgen-faster"),
                        f"--max-active={max_active}",
                        f"--beam={beam}",
                        f"--lattice-beam={self.decode_options['lattice_beam']}",
                        "--allow-partial=true",
                        f"--word-symbol-table={word_symbol_path}",
                        f"--acoustic-scale={self.decode_options['acoustic_scale']}",
                        self.model_path,
                        hclg_path,
                        feature_string,
                        f"ark:{lat_path}",
                    ],
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in decode_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("utterance"), float(m.group("loglike")), int(
                            m.group("num_frames")
                        )
                self.check_call(decode_proc)


class LmRescoreFunction(KaldiFunction):
    """
    Multiprocessing function rescore lattices by replacing the small G.fst with the medium G.fst

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.lm_rescore_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-lmrescore-pruned`
        Relevant Kaldi binary
    :openfst_src:`fstproject`
        Relevant OpenFst binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG .* Overall, succeeded for (?P<succeeded>\d+) lattices, failed for (?P<failed>\d+)"
    )

    def __init__(self, args: LmRescoreArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.lat_paths = args.lat_paths
        self.rescored_lat_paths = args.rescored_lat_paths
        self.old_g_paths = args.old_g_paths
        self.new_g_paths = args.new_g_paths
        self.lm_rescore_options = args.lm_rescore_options

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                lat_path = self.lat_paths[dict_id]
                rescored_lat_path = self.rescored_lat_paths[dict_id]
                old_g_path = self.old_g_paths[dict_id]
                new_g_path = self.new_g_paths[dict_id]
                if " " in new_g_path:
                    new_g_path = f'"{new_g_path}"'
                project_type_arg = "--project_type=output"
                if os.path.exists(rescored_lat_path):
                    continue

                project_proc = subprocess.Popen(
                    [thirdparty_binary("fstproject"), project_type_arg, old_g_path],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                lattice_scale_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-lmrescore-pruned"),
                        f"--acoustic-scale={self.lm_rescore_options['acoustic_scale']}",
                        "-",
                        f"fstproject {project_type_arg} {new_g_path} |",
                        f"ark,s,cs:{lat_path}",
                        f"ark:{rescored_lat_path}",
                    ],
                    stdin=project_proc.stdout,
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in lattice_scale_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("succeeded")), int(m.group("failed"))
            self.check_call(lattice_scale_proc)


class CarpaLmRescoreFunction(KaldiFunction):
    """
    Multiprocessing function to rescore lattices by replacing medium G.fst with large G.carpa

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.carpa_lm_rescore_arguments`
        Job method for generating arguments for this function
    :openfst_src:`fstproject`
        Relevant OpenFst binary
    :kaldi_src:`lattice-lmrescore`
        Relevant Kaldi binary
    :kaldi_src:`lattice-lmrescore-const-arpa`
        Relevant Kaldi binary

    Parameters
    ----------
    args: CarpaLmRescoreArguments
        Arguments
    """

    progress_pattern = re.compile(
        r"^LOG .* Overall, succeeded for (?P<succeeded>\d+) lattices, failed for (?P<failed>\d+)"
    )

    def __init__(self, args: CarpaLmRescoreArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.lat_paths = args.lat_paths
        self.rescored_lat_paths = args.rescored_lat_paths
        self.old_g_paths = args.old_g_paths
        self.new_g_paths = args.new_g_paths

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "a") as log_file:
            for dict_id in self.dictionaries:
                project_type_arg = "--project_type=output"
                lat_path = self.lat_paths[dict_id]
                rescored_lat_path = self.rescored_lat_paths[dict_id]
                old_g_path = self.old_g_paths[dict_id]
                new_g_path = self.new_g_paths[dict_id]
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
                        f"ark,s,cs:{lat_path}",
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
                        "ark,s,cs:-",
                        new_g_path,
                        f"ark:{rescored_lat_path}",
                    ],
                    stdin=lmrescore_proc.stdout,
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in lmrescore_const_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("succeeded")), int(m.group("failed"))
            self.check_call(lmrescore_const_proc)


class InitialFmllrFunction(KaldiFunction):
    """
    Multiprocessing function for running initial fMLLR calculation

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.initial_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-post-to-gpost`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr-gpost`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*For speaker \w+, auxf-impr from fMLLR is [\d.]+, over [\d.]+ frames."
    )

    def __init__(self, args: InitialFmllrArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options
        self.pre_trans_paths = args.pre_trans_paths
        self.lat_paths = args.lat_paths
        self.spk2utt_paths = args.spk2utt_paths

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                lat_path = self.lat_paths[dict_id]
                feature_string = self.feature_strings[dict_id]
                spk2utt_path = self.spk2utt_paths[dict_id]
                trans_path = self.pre_trans_paths[dict_id]

                latt_post_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-to-post"),
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
                        f"ark,s,cs:{lat_path}",
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                weight_silence_proc = subprocess.Popen(
                    [
                        thirdparty_binary("weight-silence-post"),
                        f"{self.fmllr_options['silence_weight']}",
                        self.fmllr_options["sil_phones"],
                        self.model_path,
                        "ark,s,cs:-",
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
                        self.model_path,
                        feature_string,
                        "ark,s,cs:-",
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
                        f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                        f"--spk2utt=ark,s,cs:{spk2utt_path}",
                        self.model_path,
                        feature_string,
                        "ark,s,cs:-",
                        f"ark:{trans_path}",
                    ],
                    stdin=gmm_gpost_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in fmllr_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield 1
            self.check_call(fmllr_proc)


class LatGenFmllrFunction(KaldiFunction):
    """
    Regenerate lattices using initial fMLLR transforms

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.lat_gen_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Log-like per frame for utterance (?P<utterance>.*) is (?P<loglike>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: LatGenFmllrArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.tmp_lat_paths = args.tmp_lat_paths
        self.word_symbol_paths = args.word_symbol_paths
        self.hclg_paths = args.hclg_paths
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def _run(self) -> typing.Generator[typing.Tuple[str, float, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                if isinstance(self.hclg_paths, dict):
                    words_path = self.word_symbol_paths[dict_id]
                    hclg_path = self.hclg_paths[dict_id]
                else:
                    words_path = self.word_symbol_paths
                    hclg_path = self.hclg_paths
                tmp_lat_path = self.tmp_lat_paths[dict_id]
                lat_gen_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-latgen-faster"),
                        f"--max-active={self.decode_options['max_active']}",
                        f"--beam={self.decode_options['beam']}",
                        f"--lattice-beam={self.decode_options['lattice_beam']}",
                        f"--acoustic-scale={self.decode_options['acoustic_scale']}",
                        "--determinize-lattice=false",
                        "--allow-partial=true",
                        f"--word-symbol-table={words_path}",
                        self.model_path,
                        hclg_path,
                        feature_string,
                        f"ark:{tmp_lat_path}",
                    ],
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in lat_gen_proc.stderr:
                    log_file.write(line)
                    log_file.flush()
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("utterance"), float(m.group("loglike")), int(
                            m.group("num_frames")
                        )
            self.check_call(lat_gen_proc)


class FinalFmllrFunction(KaldiFunction):

    """
    Multiprocessing function for running final fMLLR estimation

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.final_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-determinize-pruned`
        Relevant Kaldi binary
    :kaldi_src:`lattice-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr`
        Relevant Kaldi binary
    :kaldi_src:`compose-transforms`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*For speaker \w+, auxf-impr from fMLLR is [\d.]+, over [\d.]+ frames."
    )

    def __init__(self, args: FinalFmllrArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options
        self.trans_paths = args.trans_paths
        self.tmp_lat_paths = args.tmp_lat_paths
        self.spk2utt_paths = args.spk2utt_paths

    def _run(self) -> typing.Generator[int]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                trans_path = self.trans_paths[dict_id]
                temp_trans_path = trans_path + ".temp"
                temp_composed_trans_path = trans_path + ".temp_composed"
                spk2utt_path = self.spk2utt_paths[dict_id]
                tmp_lat_path = self.tmp_lat_paths[dict_id]
                determinize_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-determinize-pruned"),
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
                        "--beam=4.0",
                        f"ark,s,cs:{tmp_lat_path}",
                        "ark:-",
                    ],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )

                latt_post_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-to-post"),
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
                        "ark,s,cs:-",
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
                        f"{self.fmllr_options['silence_weight']}",
                        self.fmllr_options["sil_phones"],
                        self.model_path,
                        "ark,s,cs:-",
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
                        f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                        f"--spk2utt=ark,s,cs:{spk2utt_path}",
                        self.model_path,
                        feature_string,
                        "ark,s,cs:-",
                        f"ark:{temp_trans_path}",
                    ],
                    stdin=weight_silence_proc.stdout,
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in fmllr_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield 1
                self.check_call(fmllr_proc)

                compose_proc = subprocess.Popen(
                    [
                        thirdparty_binary("compose-transforms"),
                        "--b-is-affine=true",
                        f"ark:{temp_trans_path}",
                        f"ark:{trans_path}",
                        f"ark:{temp_composed_trans_path}",
                    ],
                    stderr=log_file,
                    stdin=fmllr_proc.stdout,
                    env=os.environ,
                )
                compose_proc.communicate()
                self.check_call(compose_proc)
                os.remove(trans_path)
                os.remove(temp_trans_path)
                os.rename(temp_composed_trans_path, trans_path)


class FmllrRescoreFunction(KaldiFunction):
    """
    Multiprocessing function to rescore lattices following fMLLR estimation

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.fmllr_rescore_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-rescore-lattice`
        Relevant Kaldi binary
    :kaldi_src:`lattice-determinize-pruned`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Done (?P<done>\d+) lattices, determinization finished earlier than specified by the beam (or output was empty) on (?P<errors>\d+) of these."
    )

    def __init__(self, args: FmllrRescoreArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options
        self.tmp_lat_paths = args.tmp_lat_paths
        self.final_lat_paths = args.final_lat_paths

    def _run(self) -> typing.Generator[typing.Tuple[int, int]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file:
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                tmp_lat_path = self.tmp_lat_paths[dict_id]
                final_lat_path = self.final_lat_paths[dict_id]
                rescore_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-rescore-lattice"),
                        self.model_path,
                        f"ark,s,cs:{tmp_lat_path}",
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
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
                        f"--beam={self.fmllr_options['lattice_beam']}",
                        "ark,s,cs:-",
                        f"ark:{final_lat_path}",
                    ],
                    stdin=rescore_proc.stdout,
                    stderr=subprocess.PIPE,
                    encoding="utf8",
                    env=os.environ,
                )
                for line in determinize_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("done")), int(m.group("errors"))
                self.check_call(determinize_proc)


@dataclass
class PerSpeakerDecodeArguments(MfaArguments):
    """Arguments for :class:`~montreal_forced_aligner.validation.corpus_validator.PerSpeakerDecodeFunction`"""

    model_directory: str
    feature_strings: Dict[int, str]
    lat_paths: Dict[int, str]
    model_path: str
    disambiguation_symbols_int_path: str
    decode_options: MetaDict
    tree_path: str
    order: int
    method: str


class PerSpeakerDecodeFunction(KaldiFunction):
    """
    Multiprocessing function to test utterance transcriptions with utterance and speaker ngram models

    See Also
    --------
    :kaldi_src:`compile-train-graphs-fsts`
        Relevant Kaldi binary
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary
    :kaldi_src:`lattice-oracle`
        Relevant Kaldi binary
    :openfst_src:`farcompilestrings`
        Relevant OpenFst binary
    :ngram_src:`ngramcount`
        Relevant OpenGrm-Ngram binary
    :ngram_src:`ngrammake`
        Relevant OpenGrm-Ngram binary
    :ngram_src:`ngramshrink`
        Relevant OpenGrm-Ngram binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.validation.corpus_validator.PerSpeakerDecodeArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Log-like per frame for utterance (?P<utterance>.*) is (?P<loglike>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: PerSpeakerDecodeArguments):
        super().__init__(args)
        self.feature_strings = args.feature_strings
        self.disambiguation_symbols_int_path = args.disambiguation_symbols_int_path
        self.model_directory = args.model_directory
        self.model_path = args.model_path
        self.decode_options = args.decode_options
        self.lat_paths = args.lat_paths
        self.tree_path = args.tree_path
        self.order = args.order
        self.method = args.method
        self.word_symbols_paths = {}

    def _run(self) -> typing.Generator[typing.Tuple[int, str]]:
        """Run the function"""
        with mfa_open(self.log_path, "w") as log_file, Session(self.db_engine()) as session:

            job: Job = (
                session.query(Job)
                .options(joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries))
                .filter(Job.id == self.job_name)
                .first()
            )
            for d in job.dictionaries:

                self.oov_word = d.oov_word
                self.word_symbols_paths[d.id] = d.words_symbol_path
                feature_string = self.feature_strings[d.id]
                lat_path = self.lat_paths[d.id]
                latgen_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-latgen-faster"),
                        f"--acoustic-scale={self.decode_options['acoustic_scale']}",
                        f"--beam={self.decode_options['beam']}",
                        f"--max-active={self.decode_options['max_active']}",
                        f"--lattice-beam={self.decode_options['lattice_beam']}",
                        f"--word-symbol-table={d.words_symbol_path}",
                        self.model_path,
                        "ark,s,cs:-",
                        feature_string,
                        f"ark:{lat_path}",
                    ],
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    env=os.environ,
                )
                current_speaker = None
                for utt_id, speaker_id in (
                    session.query(Utterance.kaldi_id, Utterance.speaker_id)
                    .filter(Utterance.job_id == job.id)
                    .order_by(Utterance.kaldi_id)
                ):
                    if speaker_id != current_speaker:
                        lm_path = os.path.join(d.temp_directory, f"{speaker_id}.fst")
                        fst = pynini.Fst.read(lm_path)
                        fst_string = fst.write_to_string()
                        del fst

                    latgen_proc.stdin.write(utt_id.encode("utf8") + b" " + fst_string)
                    latgen_proc.stdin.flush()

                    while True:
                        line = latgen_proc.stderr.readline().decode("utf8")
                        line = line.strip()
                        if not line:
                            break
                        log_file.write(line + "\n")
                        log_file.flush()
                        m = self.progress_pattern.match(line.strip())
                        if m:
                            yield m.group("utterance"), float(m.group("loglike")), int(
                                m.group("num_frames")
                            )
                            break
                latgen_proc.stdin.close()
                self.check_call(latgen_proc)


class DecodePhoneFunction(KaldiFunction):
    """
    Multiprocessing function for performing decoding

    See Also
    --------
    :meth:`.TranscriberMixin.transcribe_utterances`
        Main function that calls this function in parallel
    :meth:`.TranscriberMixin.decode_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-latgen-faster`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG.*Log-like per frame for utterance (?P<utterance>.*) is (?P<loglike>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: DecodePhoneArguments):
        super().__init__(args)
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.lat_paths = args.lat_paths
        self.phone_symbol_path = args.phone_symbol_path
        self.hclg_path = args.hclg_path
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def _run(self) -> typing.Generator[typing.Tuple[str, float, int]]:
        """Run the function"""
        with Session(self.db_engine()) as session, mfa_open(self.log_path, "w") as log_file:
            phones = session.query(Phone.mapping_id, Phone.phone)
            reversed_phone_mapping = {}
            for p_id, phone in phones:
                reversed_phone_mapping[p_id] = phone
            for dict_id in self.dictionaries:
                feature_string = self.feature_strings[dict_id]
                lat_path = self.lat_paths[dict_id]
                if os.path.exists(lat_path):
                    continue
                if (
                    self.decode_options["uses_speaker_adaptation"]
                    and self.decode_options["first_beam"] is not None
                ):
                    beam = self.decode_options["first_beam"]
                else:
                    beam = self.decode_options["beam"]
                if (
                    self.decode_options["uses_speaker_adaptation"]
                    and self.decode_options["first_max_active"] is not None
                ):
                    max_active = self.decode_options["first_max_active"]
                else:
                    max_active = self.decode_options["max_active"]
                decode_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-latgen-faster"),
                        f"--max-active={max_active}",
                        f"--beam={beam}",
                        f"--lattice-beam={self.decode_options['lattice_beam']}",
                        "--allow-partial=true",
                        f"--word-symbol-table={self.phone_symbol_path}",
                        f"--acoustic-scale={self.decode_options['acoustic_scale']}",
                        self.model_path,
                        self.hclg_path,
                        feature_string,
                        f"ark:{lat_path}",
                    ],
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in decode_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("utterance"), float(m.group("loglike")), int(
                            m.group("num_frames")
                        )
            self.check_call(decode_proc)
