"""
Transcription functions
-----------------------

"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Dict, List, NamedTuple, TextIO

from ..abc import KaldiFunction, MetaDict
from ..utils import thirdparty_binary

if TYPE_CHECKING:
    from ..abc import MappingType


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
    "ScoreFunction",
    "CarpaLmRescoreFunction",
    "DecodeFunction",
    "LmRescoreFunction",
    "CreateHclgFunction",
]


class CreateHclgArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CreateHclgFunction`"""

    log_path: str
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
    words_mapping: MappingType

    @property
    def hclg_path(self) -> str:
        """Path to HCLG FST file"""
        return self.path_template.format(file_name="HCLG")


class DecodeArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.DecodeFunction`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    decode_options: MetaDict
    model_path: str
    lat_paths: Dict[str, str]
    word_symbol_paths: Dict[str, str]
    hclg_paths: Dict[str, str]


class ScoreArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreFunction`"""

    log_path: str
    dictionaries: List[str]
    score_options: MetaDict
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    carpa_rescored_lat_paths: Dict[str, str]
    words_paths: Dict[str, str]
    tra_paths: Dict[str, str]


class LmRescoreArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LmRescoreFunction`"""

    log_path: str
    dictionaries: List[str]
    lm_rescore_options: MetaDict
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    old_g_paths: Dict[str, str]
    new_g_paths: Dict[str, str]


class CarpaLmRescoreArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.CarpaLmRescoreFunction`"""

    log_path: str
    dictionaries: List[str]
    lat_paths: Dict[str, str]
    rescored_lat_paths: Dict[str, str]
    old_g_paths: Dict[str, str]
    new_g_paths: Dict[str, str]


class InitialFmllrArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.InitialFmllrFunction`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: MetaDict
    pre_trans_paths: Dict[str, str]
    lat_paths: Dict[str, str]
    spk2utt_paths: Dict[str, str]


class LatGenFmllrArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.LatGenFmllrFunction`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    decode_options: MetaDict
    word_symbol_paths: Dict[str, str]
    hclg_paths: Dict[str, str]
    tmp_lat_paths: Dict[str, str]


class FinalFmllrArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FinalFmllrFunction`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: MetaDict
    trans_paths: Dict[str, str]
    spk2utt_paths: Dict[str, str]
    tmp_lat_paths: Dict[str, str]


class FmllrRescoreArguments(NamedTuple):
    """Arguments for :class:`~montreal_forced_aligner.transcription.multiprocessing.FmllrRescoreFunction`"""

    log_path: str
    dictionaries: List[str]
    feature_strings: Dict[str, str]
    model_path: str
    fmllr_options: MetaDict
    tmp_lat_paths: Dict[str, str]
    final_lat_paths: Dict[str, str]


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
    words_mapping: MappingType,
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

    progress_pattern = re.compile(
        r"^LOG.*Log-like per frame for utterance (?P<utterance>.*) is (?P<loglike>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: CreateHclgArguments):
        self.log_path = args.log_path
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

    def run(self):
        """Run the function"""
        hclg_path = self.path_template.format(file_name="HCLG")
        small_g_path = self.path_template.format(file_name="G.small")
        medium_g_path = self.path_template.format(file_name="G.med")
        lg_path = self.path_template.format(file_name="LG")
        hclga_path = self.path_template.format(file_name="HCLGa")
        if os.path.exists(hclg_path):
            return
        with open(self.log_path, "w") as log_file:
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
            if not os.path.exists(medium_g_path):
                log_file.write("Generating med_G.fst...")
                compose_g(self.medium_arpa_path, self.words_path, medium_g_path, log_file)
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
            if not os.path.exists(lg_path):
                log_file.write("Generating LG.fst...")
                compose_lg(self.disambig_L_path, small_g_path, lg_path, log_file)
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
            if not os.path.exists(hclga_path):
                log_file.write("Generating HCLGa.fst...")
                compose_hclg(
                    self.working_directory,
                    ilabels_temp,
                    self.hclg_options["transition_scale"],
                    clg_path,
                    hclga_path,
                    log_file,
                )
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
            if os.path.exists(hclg_path):
                yield True, hclg_path
            else:
                yield False, hclg_path


class DecodeFunction(KaldiFunction):
    """
    Multiprocessing function for performing decoding

    See Also
    --------
    :meth:`.Transcriber.transcribe`
        Main function that calls this function in parallel
    :meth:`.Transcriber.decode_arguments`
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.lat_paths = args.lat_paths
        self.word_symbol_paths = args.word_symbol_paths
        self.hclg_paths = args.hclg_paths
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                feature_string = self.feature_strings[dict_name]
                lat_path = self.lat_paths[dict_name]
                word_symbol_path = self.word_symbol_paths[dict_name]
                hclg_path = self.hclg_paths[dict_name]
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
                        yield m.group("utterance"), m.group("loglike"), m.group("num_frames")


class ScoreFunction(KaldiFunction):
    """
    Multiprocessing function for scoring lattices

    See Also
    --------
    :meth:`~montreal_forced_aligner.transcription.Transcriber.score_transcriptions`
        Main function that calls this function in parallel
    :meth:`.Transcriber.score_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`lattice-scale`
        Relevant Kaldi binary
    :kaldi_src:`lattice-add-penalty`
        Relevant Kaldi binary
    :kaldi_src:`lattice-best-path`
        Relevant Kaldi binary

    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.transcription.multiprocessing.ScoreArguments`
        Arguments for the function
    """

    progress_pattern = re.compile(
        r"^LOG .* For utterance (?P<utterance>.*), best cost (?P<graph_cost>[-\d.]+) \+ (?P<acoustic_cost>[-\d.]+) = (?P<total_cost>[-\d.]+) over (?P<num_frames>\d+) frames."
    )

    def __init__(self, args: ScoreArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.score_options = args.score_options
        self.lat_paths = args.lat_paths
        self.rescored_lat_paths = args.rescored_lat_paths
        self.carpa_rescored_lat_paths = args.carpa_rescored_lat_paths
        self.words_paths = args.words_paths
        self.tra_paths = args.tra_paths

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                language_model_weight = self.score_options["language_model_weight"]
                word_insertion_penalty = self.score_options["word_insertion_penalty"]
                carpa_rescored_lat_path = self.carpa_rescored_lat_paths[dict_name]
                rescored_lat_path = self.rescored_lat_paths[dict_name]
                lat_path = self.lat_paths[dict_name]
                words_path = self.words_paths[dict_name]
                tra_path = self.tra_paths[dict_name]
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
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in best_path_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("utterance"), float(m.group("graph_cost")), float(
                            m.group("acoustic_cost")
                        ), float(m.group("total_cost")), int(m.group("num_frames"))


class LmRescoreFunction(KaldiFunction):
    """
    Multiprocessing function rescore lattices by replacing the small G.fst with the medium G.fst

    See Also
    --------
    :func:`~montreal_forced_aligner.transcription.Transcriber.transcribe`
        Main function that calls this function in parallel
    :meth:`.Transcriber.lm_rescore_arguments`
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.lat_paths = args.lat_paths
        self.rescored_lat_paths = args.rescored_lat_paths
        self.old_g_paths = args.old_g_paths
        self.new_g_paths = args.new_g_paths
        self.lm_rescore_options = args.lm_rescore_options

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                lat_path = self.lat_paths[dict_name]
                rescored_lat_path = self.rescored_lat_paths[dict_name]
                old_g_path = self.old_g_paths[dict_name]
                new_g_path = self.new_g_paths[dict_name]
                if sys.platform == "win32":
                    project_type_arg = "--project_output=true"
                else:
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
                        f"ark:{lat_path}",
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


class CarpaLmRescoreFunction(KaldiFunction):
    """
    Multiprocessing function to rescore lattices by replacing medium G.fst with large G.carpa

    See Also
    --------
    :func:`~montreal_forced_aligner.transcription.Transcriber.transcribe`
        Main function that calls this function in parallel
    :meth:`.Transcriber.carpa_lm_rescore_arguments`
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.lat_paths = args.lat_paths
        self.rescored_lat_paths = args.rescored_lat_paths
        self.old_g_paths = args.old_g_paths
        self.new_g_paths = args.new_g_paths

    def run(self):
        """Run the function"""
        with open(self.log_path, "a", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                if sys.platform == "win32":
                    project_type_arg = "--project_output=true"
                else:
                    project_type_arg = "--project_type=output"
                lat_path = self.lat_paths[dict_name]
                rescored_lat_path = self.rescored_lat_paths[dict_name]
                old_g_path = self.old_g_paths[dict_name]
                new_g_path = self.new_g_paths[dict_name]
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
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
                for line in lmrescore_const_proc.stderr:
                    log_file.write(line)
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield int(m.group("succeeded")), int(m.group("failed"))


class InitialFmllrFunction(KaldiFunction):
    """
    Multiprocessing function for running initial fMLLR calculation

    See Also
    --------
    :func:`~montreal_forced_aligner.transcription.Transcriber.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.Transcriber.initial_fmllr_arguments`
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
        r"^LOG.*Done (?P<done>\d+) files, (?P<no_gpost>\d+) with no g?posts, (?P<other_errors>\d+) with other errors."
    )

    def __init__(self, args: InitialFmllrArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options
        self.pre_trans_paths = args.pre_trans_paths
        self.lat_paths = args.lat_paths
        self.spk2utt_paths = args.spk2utt_paths

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                lat_path = self.lat_paths[dict_name]
                feature_string = self.feature_strings[dict_name]
                spk2utt_path = self.spk2utt_paths[dict_name]
                trans_path = self.pre_trans_paths[dict_name]

                latt_post_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-to-post"),
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
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
                        f"{self.fmllr_options['silence_weight']}",
                        self.fmllr_options["sil_phones"],
                        self.model_path,
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
                        self.model_path,
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
                        f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                        f"--spk2utt=ark:{spk2utt_path}",
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
                        yield int(m.group("done")), int(m.group("no_gpost")), int(
                            m.group("other_errors")
                        )


class LatGenFmllrFunction(KaldiFunction):
    """
    Regenerate lattices using initial fMLLR transforms

    See Also
    --------
    :func:`~montreal_forced_aligner.transcription.Transcriber.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.Transcriber.lat_gen_fmllr_arguments`
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.tmp_lat_paths = args.tmp_lat_paths
        self.word_symbol_paths = args.word_symbol_paths
        self.hclg_paths = args.hclg_paths
        self.decode_options = args.decode_options
        self.model_path = args.model_path

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                feature_string = self.feature_strings[dict_name]
                words_path = self.word_symbol_paths[dict_name]
                hclg_path = self.hclg_paths[dict_name]
                tmp_lat_path = self.tmp_lat_paths[dict_name]
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
                    m = self.progress_pattern.match(line.strip())
                    if m:
                        yield m.group("utterance"), m.group("loglike"), m.group("num_frames")


class FinalFmllrFunction(KaldiFunction):

    """
    Multiprocessing function for running final fMLLR estimation

    See Also
    --------
    :func:`~montreal_forced_aligner.transcription.Transcriber.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.Transcriber.final_fmllr_arguments`
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
        r"^LOG.*Done (?P<done>\d+) files, (?P<no_gpost>\d+) with no g?posts, (?P<other_errors>\d+) with other errors."
    )

    def __init__(self, args: FinalFmllrArguments):
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options
        self.trans_paths = args.trans_paths
        self.tmp_lat_paths = args.tmp_lat_paths
        self.spk2utt_paths = args.spk2utt_paths

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                feature_string = self.feature_strings[dict_name]
                trans_path = self.trans_paths[dict_name]
                temp_trans_path = trans_path + ".temp"
                temp_composed_trans_path = trans_path + ".temp_composed"
                spk2utt_path = self.spk2utt_paths[dict_name]
                tmp_lat_path = self.tmp_lat_paths[dict_name]
                determinize_proc = subprocess.Popen(
                    [
                        thirdparty_binary("lattice-determinize-pruned"),
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
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
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
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
                        f"{self.fmllr_options['silence_weight']}",
                        self.fmllr_options["sil_phones"],
                        self.model_path,
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
                        f"--fmllr-update-type={self.fmllr_options['fmllr_update_type']}",
                        f"--spk2utt=ark:{spk2utt_path}",
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
                        yield int(m.group("done")), int(m.group("no_gpost")), int(
                            m.group("other_errors")
                        )

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
                os.remove(trans_path)
                os.remove(temp_trans_path)
                os.rename(temp_composed_trans_path, trans_path)


class FmllrRescoreFunction(KaldiFunction):
    """
    Multiprocessing function to rescore lattices following fMLLR estimation

    See Also
    --------
    :func:`~montreal_forced_aligner.transcription.Transcriber.transcribe_fmllr`
        Main function that calls this function in parallel
    :meth:`.Transcriber.fmllr_rescore_arguments`
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
        self.log_path = args.log_path
        self.dictionaries = args.dictionaries
        self.feature_strings = args.feature_strings
        self.model_path = args.model_path
        self.fmllr_options = args.fmllr_options
        self.tmp_lat_paths = args.tmp_lat_paths
        self.final_lat_paths = args.final_lat_paths

    def run(self):
        """Run the function"""
        with open(self.log_path, "w", encoding="utf8") as log_file:
            for dict_name in self.dictionaries:
                feature_string = self.feature_strings[dict_name]
                tmp_lat_path = self.tmp_lat_paths[dict_name]
                final_lat_path = self.final_lat_paths[dict_name]
                rescore_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-rescore-lattice"),
                        self.model_path,
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
                        f"--acoustic-scale={self.fmllr_options['acoustic_scale']}",
                        f"--beam={self.fmllr_options['lattice_beam']}",
                        "ark:-",
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
