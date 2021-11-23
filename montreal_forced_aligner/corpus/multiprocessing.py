"""
Corpus loading worker
---------------------


"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from queue import Empty
from typing import TYPE_CHECKING, Collection, Optional, Union

from ..exceptions import TextGridParseError, TextParseError
from ..helper import make_safe, output_mapping
from ..utils import thirdparty_binary

if TYPE_CHECKING:

    from ..abc import MetaDict, OneToManyMappingType, OneToOneMappingType
    from ..corpus.helper import SoundFileInfoDict

    FileInfoDict = dict[
        str, Union[str, SoundFileInfoDict, OneToOneMappingType, OneToManyMappingType]
    ]
    from ..abc import MappingType, ReversedMappingType, WordsType
    from ..corpus.classes import File, Speaker, Utterance
    from ..dictionary import DictionaryData, PronunciationDictionaryMixin
    from ..utils import Stopped


__all__ = ["mfcc_func", "compute_vad_func", "CorpusProcessWorker", "Job"]


class CorpusProcessWorker(mp.Process):
    """
    Multiprocessing corpus loading worker

    Attributes
    ----------
    job_q: :class:`~multiprocessing.Queue`
        Job queue for files to process
    return_dict: Dict
        Dictionary to catch errors
    return_q: :class:`~multiprocessing.Queue`
        Return queue for processed Files
    stopped: :func:`~montreal_forced_aligner.multiprocessing.helper.Stopped`
        Stop check for whether corpus loading should exit
    finished_adding: :class:`~montreal_forced_aligner.multiprocessing.helper.Stopped`
        Signal that the main thread has stopped adding new files to be processed
    """

    def __init__(
        self,
        job_q: mp.Queue,
        return_dict: dict,
        return_q: mp.Queue,
        stopped: Stopped,
        finished_adding: Stopped,
    ):
        mp.Process.__init__(self)
        self.job_q = job_q
        self.return_dict = return_dict
        self.return_q = return_q
        self.stopped = stopped
        self.finished_adding = finished_adding

    def run(self) -> None:
        """
        Run the corpus loading job
        """
        from ..corpus.classes import parse_file

        while True:
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty:
                if self.finished_adding.stop_check():
                    break
                continue
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                file = parse_file(*arguments, stop_check=self.stopped.stop_check)
                self.return_q.put(file)
            except TextParseError as e:
                self.return_dict["decode_error_files"].append(e)
            except TextGridParseError as e:
                self.return_dict["textgrid_read_errors"][e.file_name] = e
            except Exception:
                self.stopped.stop()
                self.return_dict["error"] = arguments, Exception(
                    traceback.format_exception(*sys.exc_info())
                )
        return


def mfcc_func(
    log_path: str,
    dictionaries: list[str],
    feats_scp_paths: dict[str, str],
    lengths_paths: dict[str, str],
    segment_paths: dict[str, str],
    wav_paths: dict[str, str],
    mfcc_options: MetaDict,
) -> None:
    """
    Multiprocessing function for generating MFCC features

    See Also
    --------
    :func:`~montreal_forced_aligner.multiprocessing.features.mfcc`
        Main function that calls this function in parallel
    :meth:`.Job.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feats_scp_paths: Dict[str, str]
        Dictionary of feature scp files per dictionary name
    lengths_paths: Dict[str, str]
        Dictionary of feature lengths files per dictionary name
    segment_paths: Dict[str, str]
        Dictionary of segment scp files per dictionary name
    wav_paths: Dict[str, str]
        Dictionary of sound file scp files per dictionary name
    mfcc_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for MFCC generation
    """
    with open(log_path, "w") as log_file:
        for dict_name in dictionaries:
            mfcc_base_command = [thirdparty_binary("compute-mfcc-feats"), "--verbose=2"]
            raw_ark_path = feats_scp_paths[dict_name].replace(".scp", ".ark")
            for k, v in mfcc_options.items():
                mfcc_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
            if os.path.exists(segment_paths[dict_name]):
                mfcc_base_command += ["ark:-", "ark:-"]
                seg_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-segments"),
                        f"scp,p:{wav_paths[dict_name]}",
                        segment_paths[dict_name],
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ,
                )
                comp_proc = subprocess.Popen(
                    mfcc_base_command,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    stdin=seg_proc.stdout,
                    env=os.environ,
                )
            else:
                mfcc_base_command += [f"scp,p:{wav_paths[dict_name]}", "ark:-"]
                comp_proc = subprocess.Popen(
                    mfcc_base_command, stdout=subprocess.PIPE, stderr=log_file, env=os.environ
                )
            copy_proc = subprocess.Popen(
                [
                    thirdparty_binary("copy-feats"),
                    "--compress=true",
                    "ark:-",
                    f"ark,scp:{raw_ark_path},{feats_scp_paths[dict_name]}",
                ],
                stdin=comp_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            copy_proc.communicate()

            utt_lengths_proc = subprocess.Popen(
                [
                    thirdparty_binary("feat-to-len"),
                    f"scp:{feats_scp_paths[dict_name]}",
                    f"ark,t:{lengths_paths[dict_name]}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            utt_lengths_proc.communicate()


def compute_vad_func(
    log_path: str,
    dictionaries: list[str],
    feats_scp_paths: dict[str, str],
    vad_scp_paths: dict[str, str],
    vad_options: MetaDict,
) -> None:
    """
    Multiprocessing function to compute voice activity detection

    See Also
    --------
    :func:`~montreal_forced_aligner.multiprocessing.features.compute_vad`
        Main function that calls this function in parallel
    :meth:`.Job.compute_vad_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-vad`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feats_scp_paths: Dict[str, str]
        PronunciationDictionary of feature scp files per dictionary name
    vad_scp_paths: Dict[str, str]
        PronunciationDictionary of vad scp files per dictionary name
    vad_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for VAD
    """
    with open(log_path, "w") as log_file:
        for dict_name in dictionaries:
            feats_scp_path = feats_scp_paths[dict_name]
            vad_scp_path = vad_scp_paths[dict_name]
            vad_proc = subprocess.Popen(
                [
                    thirdparty_binary("compute-vad"),
                    f"--vad-energy-mean-scale={vad_options['energy_mean_scale']}",
                    f"--vad-energy-threshold={vad_options['energy_threshold']}",
                    f"scp:{feats_scp_path}",
                    f"ark,t:{vad_scp_path}",
                ],
                stderr=log_file,
                env=os.environ,
            )
            vad_proc.communicate()


def calc_fmllr_func(
    log_path: str,
    dictionaries: list[str],
    feature_strings: dict[str, str],
    ali_paths: dict[str, str],
    ali_model_path: str,
    model_path: str,
    spk2utt_paths: dict[str, str],
    trans_paths: dict[str, str],
    fmllr_options: MetaDict,
) -> None:
    """
    Multiprocessing function for calculating fMLLR transforms

    See Also
    --------
    :func:`~montreal_forced_aligner.multiprocessing.alignment.calc_fmllr`
        Main function that calls this function in parallel
    :meth:`.Job.calc_fmllr_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`gmm-est-fmllr`
        Relevant Kaldi binary
    :kaldi_src:`gmm-est-fmllr-gpost`
        Relevant Kaldi binary
    :kaldi_src:`gmm-post-to-gpost`
        Relevant Kaldi binary
    :kaldi_src:`ali-to-post`
        Relevant Kaldi binary
    :kaldi_src:`weight-silence-post`
        Relevant Kaldi binary
    :kaldi_src:`compose-transforms`
        Relevant Kaldi binary
    :kaldi_src:`transform-feats`
        Relevant Kaldi binary

    Parameters
    ----------
    log_path: str
        Path to save log output
    dictionaries: List[str]
        List of dictionary names
    feature_strings: Dict[str, str]
        PronunciationDictionary of feature strings per dictionary name
    ali_paths: Dict[str, str]
        PronunciationDictionary of alignment archives per dictionary name
    ali_model_path: str
        Path to the alignment acoustic model file
    model_path: str
        Path to the acoustic model file
    spk2utt_paths: Dict[str, str]
        PronunciationDictionary of spk2utt scps per dictionary name
    trans_paths: Dict[str, str]
        PronunciationDictionary of fMLLR transform archives per dictionary name
    fmllr_options: :class:`~montreal_forced_aligner.abc.MetaDict`
        Options for fMLLR estimation
    """
    with open(log_path, "w", encoding="utf8") as log_file:
        log_file.writelines(f"{k}: {v}\n" for k, v in os.environ.items())
        for dict_name in dictionaries:
            feature_string = feature_strings[dict_name]
            ali_path = ali_paths[dict_name]
            spk2utt_path = spk2utt_paths[dict_name]
            trans_path = trans_paths[dict_name]
            post_proc = subprocess.Popen(
                [thirdparty_binary("ali-to-post"), f"ark:{ali_path}", "ark:-"],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            weight_proc = subprocess.Popen(
                [
                    thirdparty_binary("weight-silence-post"),
                    "0.0",
                    fmllr_options["silence_csl"],
                    ali_model_path,
                    "ark:-",
                    "ark:-",
                ],
                stderr=log_file,
                stdin=post_proc.stdout,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            if ali_model_path != model_path:
                post_gpost_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-post-to-gpost"),
                        ali_model_path,
                        feature_string,
                        "ark:-",
                        "ark:-",
                    ],
                    stderr=log_file,
                    stdin=weight_proc.stdout,
                    stdout=subprocess.PIPE,
                    env=os.environ,
                )
                est_proc = subprocess.Popen(
                    [
                        thirdparty_binary("gmm-est-fmllr-gpost"),
                        "--verbose=4",
                        f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                        f"--spk2utt=ark:{spk2utt_path}",
                        model_path,
                        feature_string,
                        "ark,s,cs:-",
                        f"ark:{trans_path}",
                    ],
                    stderr=log_file,
                    stdin=post_gpost_proc.stdout,
                    env=os.environ,
                )
                est_proc.communicate()

            else:

                if os.path.exists(trans_path):
                    cmp_trans_path = trans_paths[dict_name] + ".tmp"
                    est_proc = subprocess.Popen(
                        [
                            thirdparty_binary("gmm-est-fmllr"),
                            "--verbose=4",
                            f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                            f"--spk2utt=ark:{spk2utt_path}",
                            model_path,
                            feature_string,
                            "ark:-",
                            "ark:-",
                        ],
                        stderr=log_file,
                        stdin=weight_proc.stdout,
                        stdout=subprocess.PIPE,
                        env=os.environ,
                    )
                    comp_proc = subprocess.Popen(
                        [
                            thirdparty_binary("compose-transforms"),
                            "--b-is-affine=true",
                            "ark:-",
                            f"ark:{trans_path}",
                            f"ark:{cmp_trans_path}",
                        ],
                        stderr=log_file,
                        stdin=est_proc.stdout,
                        env=os.environ,
                    )
                    comp_proc.communicate()

                    os.remove(trans_path)
                    os.rename(cmp_trans_path, trans_path)
                else:
                    est_proc = subprocess.Popen(
                        [
                            thirdparty_binary("gmm-est-fmllr"),
                            "--verbose=4",
                            f"--fmllr-update-type={fmllr_options['fmllr_update_type']}",
                            f"--spk2utt=ark:{spk2utt_path}",
                            model_path,
                            feature_string,
                            "ark,s,cs:-",
                            f"ark:{trans_path}",
                        ],
                        stderr=log_file,
                        stdin=weight_proc.stdout,
                        env=os.environ,
                    )
                    est_proc.communicate()


class Job:
    """
    Class representing information about corpus jobs that will be run in parallel.
    Jobs have a set of speakers that they will process, along with all files and utterances associated with that speaker.
    As such, Jobs also have a set of dictionaries that the speakers use, and argument outputs are largely dependent on
    the pronunciation dictionaries in use.

    Parameters
    ----------
    name: int
        Job number is the job's identifier

    Attributes
    ----------
    speakers: List[:class:`~montreal_forced_aligner.corpus.Speaker`]
        List of speakers associated with this job
    dictionaries: Set[:class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`]
        Set of dictionaries that the job's speakers use
    subset_utts: Set[:class:`~montreal_forced_aligner.corpus.Utterance`]
        When trainers are just using a subset of the corpus, the subset of utterances on each job will be set and used to
        filter the job's utterances
    subset_speakers: Set[:class:`~montreal_forced_aligner.corpus.Speaker`]
        When subset_utts is set, this property will be calculated as the subset of speakers that the utterances correspond to
    subset_dictionaries: Set[:class:`~montreal_forced_aligner.dictionary.PronunciationDictionary`]
        Subset of dictionaries that the subset of speakers use

    """

    name: int
    speakers: list[Speaker]
    subset_utts: set[Utterance]
    subset_speakers: set[Speaker]
    dictionaries: set[PronunciationDictionaryMixin]
    subset_dictionaries: set[PronunciationDictionaryMixin]

    def __init__(self, name: int):
        self.name = name
        self.speakers = []
        self.dictionaries = set()

        self.subset_utts = set()
        self.subset_speakers = set()
        self.subset_dictionaries = set()

    def add_speaker(self, speaker: Speaker) -> None:
        """
        Add a speaker to a job

        Parameters
        ----------
        speaker: :class:`~montreal_forced_aligner.corpus.Speaker`
            Speaker to add
        """
        self.speakers.append(speaker)
        self.dictionaries.add(speaker.dictionary)

    def set_subset(self, subset_utts: Optional[Collection[Utterance]]) -> None:
        """
        Set the current subset for the trainer

        Parameters
        ----------
        subset_utts: Collection[:class:`~montreal_forced_aligner.corpus.Utterance`], optional
            Subset of utterances for this job to use
        """
        if subset_utts is None:
            self.subset_utts = set()
            self.subset_speakers = set()
            self.subset_dictionaries = set()
        else:
            self.subset_utts = set(subset_utts)
            self.subset_speakers = {u.speaker for u in subset_utts if u.speaker in self.speakers}
            self.subset_dictionaries = {s.dictionary for s in self.subset_speakers}

    def text_scp_data(self) -> dict[str, dict[str, list[str]]]:
        """
        Generate the job's data for Kaldi's text scp files

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            Text for each utterance, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if not u.text:
                    continue
                data[key][u.name] = u.text_for_scp()
        return data

    def text_int_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's text int scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Text converted to integer IDs for each utterance, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                continue
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if u.ignored:
                    continue
                if not u.text:
                    continue
                data[key][u.name] = " ".join(map(str, u.text_int_for_scp()))
        return data

    def wav_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's wav scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Wav scp strings for each file, per dictionary name
        """
        data = {}
        done = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
                done[key] = set()
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if not u.is_segment:
                    data[key][u.name] = u.file.for_wav_scp()
                elif u.file.name not in done:
                    data[key][u.file.name] = u.file.for_wav_scp()
                    done[key].add(u.file.name)
        return data

    def utt2spk_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's utt2spk scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Utterance to speaker mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                data[key][u.name] = s.name
        return data

    def feat_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's feature scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Utterance to feature archive ID mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if u.features:
                    data[key][u.name] = u.features
        return data

    def spk2utt_scp_data(self) -> dict[str, dict[str, list[str]]]:
        """
        Generate the job's data for Kaldi's spk2utt scp files

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            Speaker to utterance mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            data[key][s.name] = sorted(
                [
                    u.name
                    for u in s.utterances.values()
                    if not u.ignored and not (self.subset_utts and u not in self.subset_utts)
                ]
            )
        return data

    def cmvn_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's CMVN scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Speaker to CMVN mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            if s.cmvn:
                data[key][s.name] = s.cmvn
        return data

    def segments_scp_data(self) -> dict[str, dict[str, str]]:
        """
        Generate the job's data for Kaldi's segments scp files

        Returns
        -------
        Dict[str, Dict[str, str]]
            Utterance to segment mapping, per dictionary name
        """
        data = {}
        for s in self.speakers:
            if s.dictionary is None:
                key = None
            else:
                key = s.dictionary.name
            if key not in data:
                data[key] = {}
            if self.subset_speakers and s not in self.subset_speakers:
                continue
            for u in s.utterances.values():
                if u.ignored:
                    continue
                if self.subset_utts and u not in self.subset_utts:
                    continue
                if not u.is_segment:
                    continue
                data[key][u.name] = u.segment_for_scp()
        return data

    def construct_path_dictionary(
        self, directory: str, identifier: str, extension: str
    ) -> dict[str, str]:
        """
        Helper function for constructing dictionary-dependent paths for the Job

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        Dict[str, str]
            Path for each dictionary
        """
        output = {}
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(
                directory, f"{identifier}.{dict_name}.{self.name}.{extension}"
            )
        return output

    def construct_dictionary_dependent_paths(
        self, directory: str, identifier: str, extension: str
    ) -> dict[str, str]:
        """
        Helper function for constructing paths that depend only on the dictionaries of the job, and not the job name itself.
        These paths should be merged with all other jobs to get a full set of dictionary paths.

        Parameters
        ----------
        directory: str
            Directory to use as the root
        identifier: str
            Identifier for the path name, like ali or acc
        extension: str
            Extension of the path, like .scp or .ark

        Returns
        -------
        Dict[str, str]
            Path for each dictionary
        """
        output = {}
        for dict_name in self.current_dictionary_names:
            output[dict_name] = os.path.join(directory, f"{identifier}.{dict_name}.{extension}")
        return output

    @property
    def dictionary_count(self):
        """Number of dictionaries currently used"""
        if self.subset_dictionaries:
            return len(self.subset_dictionaries)
        return len(self.dictionaries)

    @property
    def current_dictionaries(self) -> Collection[PronunciationDictionaryMixin]:
        """Current dictionaries depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return self.subset_dictionaries
        return self.dictionaries

    @property
    def current_dictionary_names(self) -> list[Optional[str]]:
        """Current dictionary names depending on whether a subset is being used"""
        if self.subset_dictionaries:
            return sorted(x.name for x in self.subset_dictionaries)
        if self.dictionaries == {None}:
            return [None]
        return sorted(x.name for x in self.dictionaries)

    def word_boundary_int_files(self) -> dict[str, str]:
        """
        Generate mapping for dictionaries to word boundary int files

        Returns
        -------
        Dict[str, str]
            Per dictionary word boundary int files
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = os.path.join(dictionary.phones_dir, "word_boundary.int")
        return data

    def reversed_phone_mappings(self) -> dict[str, ReversedMappingType]:
        """
        Generate mapping for dictionaries to reversed phone mapping

        Returns
        -------
        Dict[str, ReversedMappingType]
            Per dictionary reversed phone mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_phone_mapping
        return data

    def reversed_word_mappings(self) -> dict[str, ReversedMappingType]:
        """
        Generate mapping for dictionaries to reversed word mapping

        Returns
        -------
        Dict[str, ReversedMappingType]
            Per dictionary reversed word mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.reversed_word_mapping
        return data

    def words_mappings(self) -> dict[str, MappingType]:
        """
        Generate mapping for dictionaries to word mapping

        Returns
        -------
        Dict[str, MappingType]
            Per dictionary word mapping
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words_mapping
        return data

    def words(self) -> dict[str, WordsType]:
        """
        Generate mapping for dictionaries to words

        Returns
        -------
        Dict[str, WordsType]
            Per dictionary words
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.words
        return data

    def punctuation(self):
        """
        Generate mapping for dictionaries to punctuation

        Returns
        -------
        Dict[str, str]
            Per dictionary punctuation
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.punctuation
        return data

    def clitic_set(self) -> dict[str, set[str]]:
        """
        Generate mapping for dictionaries to clitic sets

        Returns
        -------
        Dict[str, str]
            Per dictionary clitic sets
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_set
        return data

    def clitic_markers(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to clitic markers

        Returns
        -------
        Dict[str, str]
            Per dictionary clitic markers
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.clitic_markers
        return data

    def compound_markers(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to compound markers

        Returns
        -------
        Dict[str, str]
            Per dictionary compound markers
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.compound_markers
        return data

    def strip_diacritics(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to diacritics to strip

        Returns
        -------
        Dict[str, List[str]]
            Per dictionary strip diacritics
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.strip_diacritics
        return data

    def oov_codes(self) -> dict[str, str]:
        """
        Generate mapping for dictionaries to oov symbols

        Returns
        -------
        Dict[str, str]
            Per dictionary oov symbols
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_word
        return data

    def oov_ints(self) -> dict[str, int]:
        """
        Generate mapping for dictionaries to oov ints

        Returns
        -------
        Dict[str, int]
            Per dictionary oov ints
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.oov_int
        return data

    def positions(self) -> dict[str, list[str]]:
        """
        Generate mapping for dictionaries to positions

        Returns
        -------
        Dict[str, List[str]]
            Per dictionary positions
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.positions
        return data

    def silences(self) -> dict[str, set[str]]:
        """
        Generate mapping for dictionaries to silence symbols

        Returns
        -------
        Dict[str, Set[str]]
            Per dictionary silence symbols
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.silences
        return data

    def multilingual_ipa(self) -> dict[str, bool]:
        """
        Generate mapping for dictionaries to multilingual IPA flags

        Returns
        -------
        Dict[str, bool]
            Per dictionary multilingual IPA flags
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.multilingual_ipa
        return data

    def job_utts(self) -> dict[str, dict[str, Utterance]]:
        """
        Generate utterances by dictionary name for the Job

        Returns
        -------
        Dict[str, Dict[str, :class:`~montreal_forced_aligner.corpus.Utterance`]]
            Mapping of dictionary name to Utterance mappings
        """
        data = {}
        speakers = self.subset_speakers
        if not speakers:
            speakers = self.speakers
        for s in speakers:
            if s.dictionary.name not in data:
                data[s.dictionary.name] = {}
            data[s.dictionary.name].update(s.utterances)
        return data

    def job_files(self) -> dict[str, File]:
        """
        Generate files for the Job

        Returns
        -------
        Dict[str, :class:`~montreal_forced_aligner.corpus.File`]
            Mapping of file name to File objects
        """
        data = {}
        speakers = self.subset_speakers
        if not speakers:
            speakers = self.speakers
        for s in speakers:
            for f in s.files:
                for sf in f.speaker_ordering:
                    if sf.name == s.name:
                        sf.dictionary_data = s.dictionary_data
                data[f.name] = f
        return data

    def dictionary_data(self) -> dict[str, DictionaryData]:
        """
        Generate dictionary data for the job

        Returns
        -------
        Dict[str, DictionaryData]
            Mapping of dictionary name to dictionary data
        """
        data = {}
        for dictionary in self.current_dictionaries:
            data[dictionary.name] = dictionary.data()
        return data

    def output_to_directory(self, split_directory: str) -> None:
        """
        Output job information to a directory

        Parameters
        ----------
        split_directory: str
            Directory to output to
        """
        wav = self.wav_scp_data()
        for dict_name, scp in wav.items():
            wav_scp_path = os.path.join(split_directory, f"wav.{dict_name}.{self.name}.scp")
            output_mapping(scp, wav_scp_path, skip_safe=True)

        spk2utt = self.spk2utt_scp_data()
        for dict_name, scp in spk2utt.items():
            spk2utt_scp_path = os.path.join(
                split_directory, f"spk2utt.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, spk2utt_scp_path)

        feats = self.feat_scp_data()
        for dict_name, scp in feats.items():
            feats_scp_path = os.path.join(split_directory, f"feats.{dict_name}.{self.name}.scp")
            output_mapping(scp, feats_scp_path)

        cmvn = self.cmvn_scp_data()
        for dict_name, scp in cmvn.items():
            cmvn_scp_path = os.path.join(split_directory, f"cmvn.{dict_name}.{self.name}.scp")
            output_mapping(scp, cmvn_scp_path)

        utt2spk = self.utt2spk_scp_data()
        for dict_name, scp in utt2spk.items():
            utt2spk_scp_path = os.path.join(
                split_directory, f"utt2spk.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, utt2spk_scp_path)

        segments = self.segments_scp_data()
        for dict_name, scp in segments.items():
            segments_scp_path = os.path.join(
                split_directory, f"segments.{dict_name}.{self.name}.scp"
            )
            output_mapping(scp, segments_scp_path)

        text_scp = self.text_scp_data()
        for dict_name, scp in text_scp.items():
            if not scp:
                continue
            text_scp_path = os.path.join(split_directory, f"text.{dict_name}.{self.name}.scp")
            output_mapping(scp, text_scp_path)

        text_int = self.text_int_scp_data()
        for dict_name, scp in text_int.items():
            if dict_name is None:
                continue
            if not scp:
                continue
            text_int_scp_path = os.path.join(
                split_directory, f"text.{dict_name}.{self.name}.int.scp"
            )
            output_mapping(scp, text_int_scp_path, skip_safe=True)
