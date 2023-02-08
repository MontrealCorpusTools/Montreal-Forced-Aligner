"""
Model classes
=============

"""
from __future__ import annotations

import json
import logging
import os
import shutil
import typing
from shutil import copy, copyfile, make_archive, move, rmtree, unpack_archive
from typing import TYPE_CHECKING, Collection, Dict, List, Optional, Tuple, Union

import requests
import yaml

from montreal_forced_aligner.abc import MfaModel, ModelExporterMixin
from montreal_forced_aligner.data import PhoneSetType
from montreal_forced_aligner.exceptions import (
    LanguageModelNotFoundError,
    ModelLoadError,
    ModelsConnectionError,
    PronunciationAcousticMismatchError,
    RemoteModelNotFoundError,
)
from montreal_forced_aligner.helper import EnhancedJSONEncoder, TerminalPrinter, mfa_open

if TYPE_CHECKING:
    from dataclasses import dataclass

    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.dictionary.mixins import DictionaryMixin
    from montreal_forced_aligner.g2p.trainer import G2PTrainer
else:
    from dataclassy import dataclass

# default format for output
FORMAT = "zip"

logger = logging.getLogger("mfa")

__all__ = [
    "Archive",
    "LanguageModel",
    "AcousticModel",
    "IvectorExtractorModel",
    "DictionaryModel",
    "G2PModel",
    "ModelManager",
    "ModelRelease",
    "MODEL_TYPES",
    "guess_model_type",
]


def guess_model_type(path: str) -> List[str]:
    """
    Guess a model type given a path

    Parameters
    ----------
    path: str
        Model archive to guess

    Returns
    -------
    list[str]
        Possible model types that use that extension
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        return []
    possible = []
    for m, mc in MODEL_TYPES.items():
        if ext in mc.extensions:
            possible.append(m)
    return possible


class Archive(MfaModel):
    """
    Class representing data in a directory or archive file (zip, tar,
    tar.gz/tgz)

    Based on the prosodylab-aligner
    (https://github.com/prosodylab/Prosodylab-Aligner) archive class.

    Parameters
    ----------
    source: str
        Source path
    root_directory: str
        Root directory to unpack and store temporary files
    """

    extensions = [".zip"]

    model_type = None

    def __init__(self, source: str, root_directory: Optional[str] = None):
        from .config import get_temporary_directory

        if root_directory is None:
            root_directory = os.path.join(
                get_temporary_directory(), "extracted_models", self.model_type
            )
        self.root_directory = root_directory
        self.source = source
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        else:
            self.dirname = os.path.join(root_directory, f"{self.name}_{self.model_type}")
            if os.path.exists(self.dirname):
                shutil.rmtree(self.dirname, ignore_errors=True)

            os.makedirs(root_directory, exist_ok=True)
            unpack_archive(source, self.dirname)
            files = os.listdir(self.dirname)
            old_dir_path = os.path.join(self.dirname, files[0])
            if len(files) == 1 and os.path.isdir(old_dir_path):  # Backwards compatibility
                for f in os.listdir(old_dir_path):
                    move(os.path.join(old_dir_path, f), os.path.join(self.dirname, f))
                os.rmdir(old_dir_path)

    def parse_old_features(self) -> None:
        """
        Parse MFA model's features and ensure that they are up-to-date with current functionality
        """
        if "features" not in self._meta:
            return
        feature_key_remapping = {
            "type": "feature_type",
            "deltas": "uses_deltas",
            "fmllr": "uses_speaker_adaptation",
        }

        for key, new_key in feature_key_remapping.items():
            if key in self._meta["features"]:
                self._meta["features"][new_key] = self._meta["features"][key]
                del self._meta["features"][key]
        if "uses_splices" not in self._meta["features"]:  # Backwards compatibility
            self._meta["features"]["uses_splices"] = os.path.exists(
                os.path.join(self.dirname, "lda.mat")
            )
        if "uses_speaker_adaptation" not in self._meta["features"]:
            self._meta["features"]["uses_speaker_adaptation"] = os.path.exists(
                os.path.join(self.dirname, "final.alimdl")
            )

    def get_subclass_object(
        self,
    ) -> Union[AcousticModel, G2PModel, LanguageModel, IvectorExtractorModel]:
        """
        Instantiate subclass models based on files contained in the archive

        Returns
        -------
        :class:`~montreal_forced_aligner.models.AcousticModel`, :class:`~montreal_forced_aligner.models.G2PModel`, :class:`~montreal_forced_aligner.models.LanguageModel`, or :class:`~montreal_forced_aligner.models.IvectorExtractorModel`
            Subclass model that was auto detected

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.ModelLoadError`
            If the model type cannot be determined
        """
        for f in os.listdir(self.dirname):
            if f == "tree":
                return AcousticModel(self.dirname, self.root_directory)
            if f in {"phones.sym", "phones.txt"}:
                return G2PModel(self.dirname, self.root_directory)
            if f.endswith(".arpa"):
                return LanguageModel(self.dirname, self.root_directory)
            if f == "final.ie":
                return IvectorExtractorModel(self.dirname, self.root_directory)
        raise ModelLoadError(self.source)

    @classmethod
    def valid_extension(cls, filename: str) -> bool:
        """
        Check whether a file has a valid extension for the given model archive

        Parameters
        ----------
        filename: str
            File name to check

        Returns
        -------
        bool
            True if the extension matches the models allowed extensions
        """
        if os.path.splitext(filename)[1] in cls.extensions:
            return True
        return False

    @classmethod
    def generate_path(cls, root: str, name: str, enforce_existence: bool = True) -> Optional[str]:
        """
        Generate a path for a given model from the root directory and the name of the model

        Parameters
        ----------
        root: str
            Root directory for the full path
        name: str
            Name of the model
        enforce_existence: bool
            Flag to return None if the path doesn't exist, defaults to True

        Returns
        -------
        str
           Full path in the root directory for the model
        """
        for ext in cls.extensions:
            path = os.path.join(root, name + ext)
            if os.path.exists(path) or not enforce_existence:
                return path
        return None

    def pretty_print(self) -> None:
        """
        Pretty print the archive's meta data using TerminalPrinter
        """
        printer = TerminalPrinter()
        configuration_data = {"Archive": {"name": (self.name, "green"), "data": self.meta}}
        printer.print_config(configuration_data)

    @property
    def meta(self) -> dict:
        """
        Get the meta data associated with the model
        """
        if not self._meta:
            meta_path = os.path.join(self.dirname, "meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = os.path.join(self.dirname, "meta.yaml")
                format = "yaml"
            with mfa_open(meta_path, "r") as f:
                if format == "yaml":
                    self._meta = yaml.safe_load(f)
                else:
                    self._meta = json.load(f)
        self.parse_old_features()
        return self._meta

    def add_meta_file(self, trainer: ModelExporterMixin) -> None:
        """
        Add a metadata file from a given trainer to the model

        Parameters
        ----------
        trainer: :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
            The trainer to construct the metadata from
        """
        with mfa_open(os.path.join(self.dirname, "meta.json"), "w") as f:
            json.dump(trainer.meta, f, ensure_ascii=False)

    @classmethod
    def empty(
        cls, head: str, root_directory: Optional[str] = None
    ) -> Union[Archive, IvectorExtractorModel, AcousticModel, G2PModel, LanguageModel]:
        """
        Initialize an archive using an empty directory

        Parameters
        ----------
        head: str
            Directory name to create
        root_directory: str, optional
            Root directory to create temporary data, defaults to the MFA temporary directory

        Returns
        -------
        :class:`~montreal_forced_aligner.models.Archive`, :class:`~montreal_forced_aligner.models.AcousticModel`, :class:`~montreal_forced_aligner.models.G2PModel`, :class:`~montreal_forced_aligner.models.LanguageModel`, or :class:`~montreal_forced_aligner.models.IvectorExtractorModel`
            Model constructed from the empty directory
        """
        from .config import get_temporary_directory

        if root_directory is None:
            root_directory = os.path.join(get_temporary_directory(), "temp_models", cls.model_type)

        os.makedirs(root_directory, exist_ok=True)
        source = os.path.join(root_directory, head)
        os.makedirs(source, exist_ok=True)
        return cls(source)

    def add(self, source: str):
        """
        Add file into archive

        Parameters
        ----------
        source: str
            Path to file to copy into the directory
        """
        copy(source, self.dirname)

    def __repr__(self) -> str:
        """Representation string of a model"""
        return f"{self.__class__.__name__}(dirname={self.dirname!r})"

    def clean_up(self) -> None:
        """Remove temporary directory"""
        rmtree(self.dirname)

    def dump(self, path: str, archive_fmt: str = FORMAT) -> str:
        """
        Write archive to disk, and return the name of final archive

        Parameters
        ----------
        path: str
            Path to write to
        archive_fmt: str, optional
            Archive extension to use, defaults to ".zip"

        Returns
        -------
        str
            Path of constructed archive
        """
        return make_archive(os.path.splitext(path)[0], archive_fmt, *os.path.split(self.dirname))


class AcousticModel(Archive):
    """
    Class for storing acoustic models in MFA, exported as zip files containing the necessary Kaldi files
    to be reused

    """

    files = [
        "final.mdl",
        "final.alimdl",
        "lda.mat",
        "phone_pdf.counts",
        # "rules.yaml",
        "phone_lm.fst",
        "tree",
        "phones.txt",
        "graphemes.txt",
    ]
    extensions = [".zip", ".am"]

    model_type = "acoustic"

    def __init__(self, source: str, root_directory: Optional[str] = None):
        if source in AcousticModel.get_available_models():
            source = AcousticModel.get_pretrained_path(source)

        super().__init__(source, root_directory)

    def add_meta_file(self, trainer: ModelExporterMixin) -> None:
        """
        Add metadata file from a model trainer

        Parameters
        ----------
        trainer: :class:`~montreal_forced_aligner.abc.ModelExporterMixin`
            Trainer to supply metadata information about the acoustic model
        """
        with mfa_open(os.path.join(self.dirname, "meta.json"), "w") as f:
            json.dump(trainer.meta, f, ensure_ascii=False)

    @property
    def parameters(self) -> MetaDict:
        """Parameters to pass to top-level workers"""
        params = {**self.meta["features"]}
        params["non_silence_phones"] = {x for x in self.meta["phones"]}
        params["oov_phone"] = self.meta["oov_phone"]
        params["optional_silence_phone"] = self.meta["optional_silence_phone"]
        params["phone_set_type"] = self.meta["phone_set_type"]
        params["silence_probability"] = self.meta.get("silence_probability", 0.5)
        params["initial_silence_probability"] = self.meta.get("initial_silence_probability", 0.5)
        params["final_non_silence_correction"] = self.meta.get(
            "final_non_silence_correction", None
        )
        params["final_silence_correction"] = self.meta.get("final_silence_correction", None)
        if "other_noise_phone" in self.meta:
            params["other_noise_phone"] = self.meta["other_noise_phone"]
        # rules_path = os.path.join(self.dirname, "rules.yaml")
        # if os.path.exists(rules_path):
        #    params["rules_path"] = rules_path
        if (
            "dictionaries" in self.meta
            and "position_dependent_phones" in self.meta["dictionaries"]
        ):
            params["position_dependent_phones"] = self.meta["dictionaries"][
                "position_dependent_phones"
            ]
        else:
            params["position_dependent_phones"] = self.meta.get("position_dependent_phones", True)
        return params

    @property
    def meta(self) -> MetaDict:
        """
        Metadata information for the acoustic model
        """
        default_features = {
            "feature_type": "mfcc",
            "use_energy": False,
            "frame_shift": 10,
            "snip_edges": True,
            "low_frequency": 20,
            "high_frequency": 7800,
            "sample_frequency": 16000,
            "allow_downsample": True,
            "allow_upsample": True,
            "use_pitch": False,
            "use_voicing": False,
            "uses_cmvn": True,
            "uses_deltas": True,
            "uses_splices": False,
            "uses_voiced": False,
            "uses_speaker_adaptation": False,
            "silence_weight": 0.0,
            "fmllr_update_type": "full",
            "splice_left_context": 3,
            "splice_right_context": 3,
        }
        if not self._meta:
            meta_path = os.path.join(self.dirname, "meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = os.path.join(self.dirname, "meta.yaml")
                format = "yaml"
            if not os.path.exists(meta_path):
                self._meta = {
                    "version": "0.9.0",
                    "architecture": "gmm-hmm",
                    "features": default_features,
                }
            else:
                with mfa_open(meta_path, "r") as f:
                    if format == "yaml":
                        self._meta = yaml.safe_load(f)
                    else:
                        self._meta = json.load(f)
                if self._meta["features"] == "mfcc+deltas":
                    self._meta["features"] = default_features
                    if "pitch" in self._meta["features"]:
                        self._meta["features"]["use_pitch"] = self._meta["features"].pop("pitch")
                if (
                    self._meta["features"].get("use_pitch", False)
                    and self._meta["version"] < "2.0.6"
                ):
                    self._meta["features"]["use_delta_pitch"] = True
            if "phone_type" not in self._meta:
                self._meta["phone_type"] = "triphone"
            if "optional_silence_phone" not in self._meta:
                self._meta["optional_silence_phone"] = "sil"
            if "oov_phone" not in self._meta:
                self._meta["oov_phone"] = "spn"
            if format == "yaml":
                self._meta["other_noise_phone"] = "sp"
            if "phone_set_type" not in self._meta:
                self._meta["phone_set_type"] = "UNKNOWN"
            self._meta["phones"] = set(self._meta.get("phones", []))
            if (
                "uses_speaker_adaptation" not in self._meta["features"]
                or not self._meta["features"]["uses_speaker_adaptation"]
            ):
                self._meta["features"]["uses_speaker_adaptation"] = os.path.exists(
                    os.path.join(self.dirname, "final.alimdl")
                )
            if self._meta["version"] in {"0.9.0", "1.0.0"}:
                self._meta["features"]["uses_speaker_adaptation"] = True
            if (
                "uses_splices" not in self._meta["features"]
                or not self._meta["features"]["uses_splices"]
            ):
                self._meta["features"]["uses_splices"] = os.path.exists(
                    os.path.join(self.dirname, "lda.mat")
                )
                if self._meta["features"]["uses_splices"]:
                    self._meta["features"]["uses_deltas"] = False
            if (
                self._meta["features"].get("use_pitch", False)
                and "use_voicing" not in self._meta["features"]
            ):
                self._meta["features"]["use_voicing"] = True
            if (
                "dictionaries" in self._meta
                and "position_dependent_phones" not in self._meta["dictionaries"]
            ):
                if self._meta["version"] < "2.0":
                    default_value = True
                else:
                    default_value = False
                self._meta["dictionaries"]["position_dependent_phones"] = self._meta.get(
                    "position_dependent_phones", default_value
                )
        self.parse_old_features()
        return self._meta

    def pretty_print(self) -> None:
        """
        Prints the metadata information to the terminal
        """
        from .utils import get_mfa_version

        printer = TerminalPrinter()
        configuration_data = {"Acoustic model": {"name": (self.name, "green"), "data": {}}}
        version_color = "green"
        if self.meta["version"] != get_mfa_version():
            version_color = "red"
        configuration_data["Acoustic model"]["data"]["Version"] = (
            self.meta["version"],
            version_color,
        )

        if "citation" in self.meta:
            configuration_data["Acoustic model"]["data"]["Citation"] = self.meta["citation"]
        if "train_date" in self.meta:
            configuration_data["Acoustic model"]["data"]["Train date"] = self.meta["train_date"]
        configuration_data["Acoustic model"]["data"]["Architecture"] = self.meta["architecture"]
        configuration_data["Acoustic model"]["data"]["Phone type"] = self.meta["phone_type"]
        configuration_data["Acoustic model"]["data"]["Features"] = {
            "Feature type": self.meta["features"]["feature_type"],
            "Frame shift": self.meta["features"]["frame_shift"],
            "Performs speaker adaptation": self.meta["features"]["uses_speaker_adaptation"],
            "Performs LDA on features": self.meta["features"]["uses_splices"],
        }
        if self.meta["phones"]:
            configuration_data["Acoustic model"]["data"]["Phones"] = self.meta["phones"]
        else:
            configuration_data["Acoustic model"]["data"]["Phones"] = ("None found!", "red")

        printer.print_config(configuration_data)

    def add_model(self, source: str) -> None:
        """
        Add file into archive

        Parameters
        ----------
        source: str
            File to add
        """
        for f in self.files:
            if os.path.exists(os.path.join(source, f)):
                copyfile(os.path.join(source, f), os.path.join(self.dirname, f))

    def add_pronunciation_models(
        self, source: str, dictionary_base_names: Collection[str]
    ) -> None:
        """
        Add file into archive

        Parameters
        ----------
        source: str
            File to add
        dictionary_base_names: list[str]
            Base names of dictionaries to add pronunciation models
        """
        for base_name in dictionary_base_names:
            for f in [f"{base_name}.fst", f"{base_name}_align.fst"]:
                if os.path.exists(os.path.join(source, f)):
                    copyfile(os.path.join(source, f), os.path.join(self.dirname, f))

    def export_model(self, destination: str) -> None:
        """
        Extract the model files to a new directory

        Parameters
        ----------
        destination: str
            Destination directory to extract files to
        """
        os.makedirs(destination, exist_ok=True)
        for f in self.files:
            if os.path.exists(os.path.join(self.dirname, f)):
                copyfile(os.path.join(self.dirname, f), os.path.join(destination, f))

    def log_details(self) -> None:
        """
        Log metadata information to a logger
        """
        logger.debug("")
        logger.debug("====ACOUSTIC MODEL INFO====")
        logger.debug("Acoustic model root directory: " + self.root_directory)
        logger.debug("Acoustic model dirname: " + self.dirname)
        meta_path = os.path.join(self.dirname, "meta.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(self.dirname, "meta.yaml")
        logger.debug("Acoustic model meta path: " + meta_path)
        if not os.path.exists(meta_path):
            logger.debug("META.YAML DOES NOT EXIST, this may cause issues in validating the model")
        logger.debug("Acoustic model meta information:")
        stream = yaml.dump(self.meta)
        logger.debug(stream)
        logger.debug("")

    def validate(self, dictionary: DictionaryMixin) -> None:
        """
        Validate this acoustic model against a pronunciation dictionary to ensure their
        phone sets are compatible

        Parameters
        ----------
        dictionary: :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
            DictionaryMixin  to compare phone sets with

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.PronunciationAcousticMismatchError`
            If there are phones missing from the acoustic model
        """
        if isinstance(dictionary, G2PModel):
            missing_phones = dictionary.meta["phones"] - set(self.meta["phones"])
        else:
            missing_phones = dictionary.non_silence_phones - set(self.meta["phones"])
        missing_phones -= {"sp", "<eps>"}
        if missing_phones:  # Compatibility
            raise (PronunciationAcousticMismatchError(missing_phones))


class IvectorExtractorModel(Archive):
    """
    Model class for IvectorExtractor
    """

    model_type = "ivector"

    model_files = [
        "final.ie",
        "final.ubm",
        "final.dubm",
        "ivector_lda.mat",
        "plda",
        "num_utts.ark",
        "speaker_ivectors.ark",
    ]
    extensions = [
        ".ivector",
        ".zip",
    ]

    def __init__(self, source: str, root_directory: Optional[str] = None):
        if source in IvectorExtractorModel.get_available_models():
            source = IvectorExtractorModel.get_pretrained_path(source)

        super().__init__(source, root_directory)

    @property
    def parameters(self) -> MetaDict:
        """Parameters to pass to top-level workers"""
        params = {**self.meta["features"]}
        for key in ["ivector_dimension", "num_gselect", "min_post", "posterior_scale"]:
            params[key] = self.meta[key]
        return params

    def add_model(self, source: str) -> None:
        """
        Add file into archive

        Parameters
        ----------
        source: str
            File to add
        """
        for filename in self.model_files:
            if os.path.exists(os.path.join(source, filename)):
                copyfile(os.path.join(source, filename), os.path.join(self.dirname, filename))

    def export_model(self, destination: str) -> None:
        """
        Extract the model files to a new directory

        Parameters
        ----------
        destination: str
            Destination directory to extract files to
        """
        os.makedirs(destination, exist_ok=True)
        for filename in self.model_files:
            if os.path.exists(os.path.join(self.dirname, filename)):
                copyfile(os.path.join(self.dirname, filename), os.path.join(destination, filename))


class G2PModel(Archive):
    """
    Class for G2P models

    Parameters
    ----------
    source: str
        Path to source archive
    root_directory: str
        Path to save exported model
    """

    extensions = [".zip", ".g2p"]

    model_type = "g2p"

    def __init__(self, source: str, root_directory: Optional[str] = None):
        if source in G2PModel.get_available_models():
            source = G2PModel.get_pretrained_path(source)

        super().__init__(source, root_directory)

    def add_meta_file(self, g2p_trainer: G2PTrainer) -> None:
        """
        Construct metadata information for the G2P model from the dictionary it was trained from

        Parameters
        ----------
        g2p_trainer: :class:`~montreal_forced_aligner.g2p.trainer.G2PTrainer`
            Trainer for the G2P model
        """

        with mfa_open(os.path.join(self.dirname, "meta.json"), "w") as f:
            json.dump(g2p_trainer.meta, f, cls=EnhancedJSONEncoder)

    @property
    def meta(self) -> dict:
        """Metadata for the G2P model"""
        if not self._meta:
            meta_path = os.path.join(self.dirname, "meta.json")
            format = "json"
            if not os.path.exists(meta_path):
                meta_path = os.path.join(self.dirname, "meta.yaml")
                format = "yaml"
            if not os.path.exists(meta_path):
                self._meta = {"version": "0.9.0", "architecture": "phonetisaurus"}
            else:
                with mfa_open(meta_path, "r") as f:
                    if format == "json":
                        self._meta = json.load(f)
                    else:
                        self._meta = yaml.safe_load(f)
            self._meta["phones"] = set(self._meta.get("phones", []))
            self._meta["graphemes"] = set(self._meta.get("graphemes", []))
            self._meta["evaluation"] = self._meta.get("evaluation", [])
            self._meta["training"] = self._meta.get("training", [])
        return self._meta

    @property
    def fst_path(self) -> str:
        """G2P model's FST path"""
        return os.path.join(self.dirname, "model.fst")

    @property
    def sym_path(self) -> str:
        """G2P model's symbols path"""
        path = os.path.join(self.dirname, "phones.txt")
        if os.path.exists(path):
            return path
        return os.path.join(self.dirname, "phones.sym")

    @property
    def grapheme_sym_path(self) -> str:
        """G2P model's grapheme symbols path"""
        path = os.path.join(self.dirname, "graphemes.txt")
        if os.path.exists(path):
            return path
        return os.path.join(self.dirname, "graphemes.sym")

    def add_sym_path(self, source_directory: str) -> None:
        """
        Add symbols file into archive

        Parameters
        ----------
        source_directory: str
            Source directory path
        """
        if not os.path.exists(self.sym_path):
            copyfile(os.path.join(source_directory, "phones.txt"), self.sym_path)
        if not os.path.exists(self.grapheme_sym_path) and os.path.exists(
            os.path.join(source_directory, "graphemes.txt")
        ):
            copyfile(os.path.join(source_directory, "graphemes.txt"), self.grapheme_sym_path)

    def add_fst_model(self, source_directory: str) -> None:
        """
        Add FST file into archive

        Parameters
        ----------
        source_directory: str
            Source directory path
        """
        if not os.path.exists(self.fst_path):
            copyfile(os.path.join(source_directory, "model.fst"), self.fst_path)

    def export_fst_model(self, destination: str) -> None:
        """
        Extract FST model path to destination

        Parameters
        ----------
        destination: str
            Destination directory
        """
        os.makedirs(destination, exist_ok=True)
        copy(self.fst_path, destination)

    def validate(self, word_list: Collection[str]) -> bool:
        """
        Validate the G2P model against a word list to ensure that all graphemes are known

        Parameters
        ----------
        word_list: Collection[str]
            Word list to validate against

        Returns
        -------
        bool
            False if missing graphemes were found
        """
        graphemes = set()
        for w in word_list:
            graphemes.update(w)
        missing_graphemes = graphemes - self.meta["graphemes"]
        if missing_graphemes:
            return False
        else:
            return True


class LanguageModel(Archive):
    """
    Class for MFA language models

    Parameters
    ----------
    source: str
        Path to source archive
    root_directory: str
        Path to save exported model
    """

    model_type = "language_model"

    arpa_extension = ".arpa"
    extensions = [f".{FORMAT}", arpa_extension, ".lm"]

    def __init__(self, source: str, root_directory: Optional[str] = None):
        if source in LanguageModel.get_available_models():
            source = LanguageModel.get_pretrained_path(source)
        from .config import get_temporary_directory

        if root_directory is None:
            root_directory = os.path.join(
                get_temporary_directory(), "extracted_models", self.model_type
            )

        if source.endswith(self.arpa_extension):
            self.root_directory = root_directory
            self._meta = {}
            self.name, _ = os.path.splitext(os.path.basename(source))
            self.dirname = os.path.join(root_directory, f"{self.name}_{self.model_type}")
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname, exist_ok=True)
            copy(source, self.large_arpa_path)
        else:
            super().__init__(source, root_directory)

    @property
    def decode_arpa_path(self) -> str:
        """
        Uses the smallest language model for decoding
        """
        for path in [self.small_arpa_path, self.medium_arpa_path, self.large_arpa_path]:
            if os.path.exists(path):
                return path
        raise LanguageModelNotFoundError(self.small_arpa_path)

    @property
    def carpa_path(self) -> str:
        """
        Uses the largest language model for rescoring
        """
        for path in [self.large_arpa_path, self.medium_arpa_path, self.small_arpa_path]:
            if os.path.exists(path):
                return path
        raise LanguageModelNotFoundError(self.large_arpa_path)

    @property
    def small_arpa_path(self) -> str:
        """Small arpa path"""
        for file in os.listdir(self.dirname):
            if file.endswith("_small" + self.arpa_extension):
                return os.path.join(self.dirname, file)
        return os.path.join(self.dirname, f"{self.name}_small{self.arpa_extension}")

    @property
    def medium_arpa_path(self) -> str:
        """Medium arpa path"""
        for file in os.listdir(self.dirname):
            if file.endswith("_med" + self.arpa_extension):
                return os.path.join(self.dirname, file)
        return os.path.join(self.dirname, f"{self.name}_med{self.arpa_extension}")

    @property
    def large_arpa_path(self) -> str:
        """Large arpa path"""
        for file in os.listdir(self.dirname):
            if file.endswith(self.arpa_extension) and "_small" not in file and "_med" not in file:
                return os.path.join(self.dirname, file)
        return os.path.join(self.dirname, self.name + self.arpa_extension)

    def add_arpa_file(self, arpa_path: str) -> None:
        """
        Adds an ARPA file to the model

        Parameters
        ----------
        arpa_path: str
            Path to ARPA file
        """
        output_name = self.large_arpa_path
        if arpa_path.endswith("_small.arpa"):
            output_name = self.small_arpa_path
        elif arpa_path.endswith("_medium.arpa"):
            output_name = self.medium_arpa_path
        copyfile(arpa_path, output_name)


class DictionaryModel(MfaModel):
    """
    Class for representing MFA pronunciation dictionaries

    Parameters
    ----------
    path: str
        Path to the dictionary file
    root_directory: str, optional
        Path to working directory (currently not needed, but present to maintain consistency with other MFA Models
    """

    model_type = "dictionary"

    extensions = [".dict", ".txt", ".yaml", ".yml"]

    def __init__(
        self,
        path: str,
        root_directory: Optional[str] = None,
        phone_set_type: typing.Union[str, PhoneSetType] = "UNKNOWN",
    ):
        if path in DictionaryModel.get_available_models():
            path = DictionaryModel.get_pretrained_path(path)

        if root_directory is None:
            from montreal_forced_aligner.config import get_temporary_directory

            root_directory = os.path.join(
                get_temporary_directory(), "extracted_models", self.model_type
            )
        self.path = path
        self.dirname = os.path.join(root_directory, f"{self.name}_{self.model_type}")
        self.pronunciation_probabilities = True
        self.silence_probabilities = True
        self.oov_probabilities = True
        if not isinstance(phone_set_type, PhoneSetType):
            phone_set_type = PhoneSetType[phone_set_type]
        self.phone_set_type = phone_set_type
        detect_phone_set = False
        if self.phone_set_type == PhoneSetType.AUTO:
            detect_phone_set = True

        patterns = {
            PhoneSetType.ARPA: PhoneSetType.ARPA.regex_detect,
            PhoneSetType.IPA: PhoneSetType.IPA.regex_detect,
            PhoneSetType.PINYIN: PhoneSetType.PINYIN.regex_detect,
        }
        counts = {
            PhoneSetType.UNKNOWN: 0,
            PhoneSetType.ARPA: 0,
            PhoneSetType.IPA: 0,
            PhoneSetType.PINYIN: 0,
        }

        count = 0
        with mfa_open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if detect_phone_set:
                    for phone_set, pattern in patterns.items():
                        if pattern.search(line):
                            counts[phone_set] += 1

                            break
                    else:
                        counts[PhoneSetType.UNKNOWN] += 1
                        continue
                    if counts[phone_set] > 100:
                        other_sets_max = max(counts[x] for x in counts if x is not phone_set)
                        if counts[phone_set] - other_sets_max >= 100:
                            break
                else:
                    count += 1
                    if count > 15:
                        break
                _, line = line.split(maxsplit=1)  # word
                try:
                    next_item, line = line.split(maxsplit=1)
                except ValueError:
                    next_item = line
                    line = ""
                if self.pronunciation_probabilities:
                    try:
                        prob = float(next_item)
                        if prob > 1 or prob < 0:
                            raise ValueError
                    except ValueError:
                        self.pronunciation_probabilities = False
                try:
                    next_item, line = line.split(maxsplit=1)
                except ValueError:
                    self.silence_probabilities = False
                    continue
                if self.silence_probabilities:
                    try:
                        prob = float(next_item)
                        if prob > 1 or prob < 0:
                            raise ValueError
                    except ValueError:
                        self.silence_probabilities = False
                self.oov_probabilities = False
        if detect_phone_set:
            self.phone_set_type = max(counts.keys(), key=lambda x: counts[x])

    @property
    def meta(self) -> MetaDict:
        """Metadata for the dictionary"""
        return {
            "phone_set_type": self.phone_set_type,
            "pronunciation_probabilities": self.pronunciation_probabilities,
            "silence_probabilities": self.silence_probabilities,
        }

    def add_meta_file(self, trainer: ModelExporterMixin) -> None:
        """Not implemented method"""
        raise NotImplementedError

    def pretty_print(self) -> None:
        """
        Pretty print the dictionary's metadata using TerminalPrinter
        """
        from montreal_forced_aligner.dictionary.multispeaker import MultispeakerDictionary

        printer = TerminalPrinter()
        configuration_data = {"Dictionary": {"name": (self.name, "green"), "data": self.meta}}
        temp_directory = os.path.join(self.dirname, "temp")
        if os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)
        dictionary = MultispeakerDictionary(self.path, phone_set_type=self.phone_set_type)
        graphemes, phone_counts = dictionary.dictionary_setup()
        configuration_data["Dictionary"]["data"]["phones"] = sorted(dictionary.non_silence_phones)
        configuration_data["Dictionary"]["data"]["detailed_phone_info"] = {}
        if self.phone_set_type.has_base_phone_regex:
            for k, v in sorted(dictionary.base_phones.items()):
                if k not in configuration_data["Dictionary"]["data"]["detailed_phone_info"]:
                    configuration_data["Dictionary"]["data"]["detailed_phone_info"][k] = []
                for p2 in sorted(v, key=lambda x: -phone_counts[x]):
                    detail_string = f"{p2} ({phone_counts[p2]})"
                    configuration_data["Dictionary"]["data"]["detailed_phone_info"][k].append(
                        detail_string
                    )

        else:
            configuration_data["Dictionary"]["data"]["detailed_phone_info"] = {}
            for phone in sorted(dictionary.non_silence_phones):
                configuration_data["Dictionary"]["data"]["detailed_phone_info"][
                    phone
                ] = phone_counts[phone]
        if len(graphemes) < 50:
            configuration_data["Dictionary"]["data"]["graphemes"] = sorted(graphemes)
        else:
            configuration_data["Dictionary"]["data"]["graphemes"] = f"{len(graphemes)} graphemes"
        printer.print_config(configuration_data)

    @classmethod
    def valid_extension(cls, filename: str) -> bool:
        """
        Check whether a file has a valid extension for the given model archive

        Parameters
        ----------
        filename: str
            File name to check

        Returns
        -------
        bool
            True if the extension matches the models allowed extensions
        """
        if os.path.splitext(filename)[1] in cls.extensions:
            return True
        return False

    @classmethod
    def generate_path(cls, root: str, name: str, enforce_existence: bool = True) -> Optional[str]:
        """
        Generate a path for a given model from the root directory and the name of the model

        Parameters
        ----------
        root: str
            Root directory for the full path
        name: str
            Name of the model
        enforce_existence: bool
            Flag to return None if the path doesn't exist, defaults to True

        Returns
        -------
        str
           Full path in the root directory for the model
        """
        for ext in cls.extensions:
            path = os.path.join(root, name + ext)
            if os.path.exists(path) or not enforce_existence:
                return path
        return None

    @property
    def is_multiple(self) -> bool:
        """Flag for whether the dictionary contains multiple lexicons"""
        return os.path.splitext(self.path)[1] in [".yaml", ".yml"]

    @property
    def name(self) -> str:
        """Name of the dictionary"""
        return os.path.splitext(os.path.basename(self.path))[0]

    def load_dictionary_paths(self) -> Dict[str, Tuple[DictionaryModel, List[str]]]:
        """
        Load the pronunciation dictionaries

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.models.DictionaryModel`]
            Mapping of component pronunciation dictionaries
        """
        mapping = {}
        if self.is_multiple:
            with mfa_open(self.path, "r") as f:
                data = yaml.safe_load(f)
                for speaker, path in data.items():
                    if path not in mapping:
                        mapping[path] = (DictionaryModel(path), set())
                    mapping[path][1].add(speaker)
        else:
            mapping[self.path] = (self, {"default"})
        return mapping


MODEL_TYPES = {
    "acoustic": AcousticModel,
    "g2p": G2PModel,
    "dictionary": DictionaryModel,
    "language_model": LanguageModel,
    "ivector": IvectorExtractorModel,
}


@dataclass(slots=True)
class ModelRelease:
    """
    Dataclas for model releases

    Parameters
    ----------
    model_name: str
        Name of the model
    tag_name: str
        Tag on GitHub
    version: str
        Version of the model
    download_link: str
        Link to download the model
    download_file_name: str
        File name to save as
    release_id: int
        Release ID on GitHub
    """

    model_name: str
    tag_name: str
    version: str
    download_link: str
    download_file_name: str
    release_id: int = None

    @property
    def release_link(self) -> Optional[str]:
        """Generate link pointing to the release on GitHub"""
        if not self.release_id:
            return None
        return ModelManager.base_url + f"/{self.release_id}"


class ModelManager:
    """
    Class for managing the currently available models on the local system and the models available to be downloaded

    Parameters
    ----------
        token: str, optional
            GitHub authentication token to use to increase release limits
    """

    base_url = "https://api.github.com/repos/MontrealCorpusTools/mfa-models/releases"

    def __init__(self, token=None):
        from montreal_forced_aligner.config import get_temporary_directory

        pretrained_dir = os.path.join(get_temporary_directory(), "pretrained_models")
        os.makedirs(pretrained_dir, exist_ok=True)
        self.local_models = {k: [] for k in MODEL_TYPES.keys()}
        self.remote_models: Dict[str, Dict[str, ModelRelease]] = {
            k: {} for k in MODEL_TYPES.keys()
        }
        self.token = token
        environment_token = os.environ.get("GITHUB_TOKEN", None)
        if self.token is not None:
            self.token = environment_token
        self.synced_remote = False
        self.printer = TerminalPrinter()
        self._cache_info = {}
        self.refresh_local()

    @property
    def cache_path(self) -> str:
        """Path to json file with cached etags and download links"""
        from montreal_forced_aligner.config import get_temporary_directory

        pretrained_dir = os.path.join(get_temporary_directory(), "pretrained_models")
        return os.path.join(pretrained_dir, "cache.json")

    def reset_local(self) -> None:
        """Reset cached models"""
        from montreal_forced_aligner.config import get_temporary_directory

        pretrained_dir = os.path.join(get_temporary_directory(), "pretrained_models")
        shutil.rmtree(pretrained_dir, ignore_errors=True)

    def refresh_local(self) -> None:
        """Refresh cached information with the latest list of local model"""
        if os.path.exists(self.cache_path):
            with mfa_open(self.cache_path, "r") as f:
                self._cache_info = json.load(f)
                if "list_etags" in self._cache_info:
                    self._cache_info["list_etags"] = {
                        int(k): v for k, v in self._cache_info["list_etags"].items()
                    }
        self.local_models = {
            model_type: model_class.get_available_models()
            for model_type, model_class in MODEL_TYPES.items()
        }

    def refresh_remote(self) -> None:
        """Refresh cached information with the latest list of downloadable models"""
        self.remote_models = {k: {} for k in MODEL_TYPES.keys()}
        data_count = 100
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        else:
            logger.debug("No Github Token supplied")
        page = 1
        etags = {}
        if "list_etags" in self._cache_info:
            etags = self._cache_info["list_etags"]
        else:
            self._cache_info["list_etags"] = {}
        while data_count == 100:
            if page in etags:
                headers["If-None-Match"] = etags[page]
            r = requests.get(
                self.base_url, params={"per_page": 100, "page": page}, headers=headers
            )
            if r.status_code >= 400:
                raise ModelsConnectionError(r.status_code, r.json(), r.headers)
            if r.status_code >= 300:  # Using cached releases
                for model_type, model_releases in self._cache_info.items():
                    if model_type not in MODEL_TYPES:
                        continue
                    for model_name, data in model_releases.items():
                        self.remote_models[model_type][model_name] = ModelRelease(*data)
                return
            self._cache_info["list_etags"][page] = r.headers["etag"]
            data = r.json()
            data_count = len(data)
            for d in data:
                tag = d["tag_name"]
                model_type, model_name, version = tag.split(
                    "-"
                )  # tag format "{model_type}-{model_name}-v{version}"
                if model_type not in self.remote_models:  # Other releases, archived, etc
                    continue
                if (
                    model_name in self.remote_models[model_type]
                ):  # Older version than currently tracked
                    continue
                if not tag.startswith(model_type):
                    continue
                if "archive" in tag:
                    continue
                download_url = d["assets"][0]["url"]
                file_name = d["assets"][0]["name"]
                self.remote_models[model_type][model_name] = ModelRelease(
                    model_name, tag, version, download_url, file_name, d["id"]
                )
                if model_type not in self._cache_info:
                    self._cache_info[model_type] = {}
                self._cache_info[model_type][model_name] = [
                    model_name,
                    tag,
                    version,
                    download_url,
                    file_name,
                    d["id"],
                ]
            page += 1
        with mfa_open(self.cache_path, "w") as f:
            json.dump(self._cache_info, f, ensure_ascii=False)
        self.synced_remote = True

    def has_local_model(self, model_type: str, model_name: str) -> bool:
        """Check for local model"""
        return model_name in self.local_models[model_type]

    def print_local_models(self, model_type: typing.Optional[str] = None) -> None:
        """
        List all local pretrained models

        Parameters
        ----------
        model_type: str, optional
            Model type, will list models of all model types if None
        """
        self.refresh_local()
        if model_type is None:
            self.printer.print_information_line("Available local models", "", level=0)
            for model_type, model_class in MODEL_TYPES.items():
                names = model_class.get_available_models()
                if names:
                    self.printer.print_information_line(model_type, names, value_color="green")
                else:
                    self.printer.print_information_line(
                        model_type, "No models found", value_color="yellow"
                    )
        else:
            self.printer.print_information_line(
                f"Available local {model_type} models", "", level=0
            )
            model_class = MODEL_TYPES[model_type]
            names = model_class.get_available_models()
            if names:
                for name in names:
                    self.printer.print_information_line("", name, value_color="green", level=1)
            else:
                self.printer.print_information_line(
                    "", "No models found", value_color="yellow", level=1
                )

    def print_remote_models(self, model_type: typing.Optional[str] = None) -> None:
        """
        Print of models available for download

        Parameters
        ----------
        model_type: str
            Model type to look up
        """
        if not self.synced_remote:
            self.refresh_remote()
        if model_type is None:
            self.printer.print_information_line("Available models for download", "", level=0)
            for model_type, release_data in self.remote_models.items():
                names = sorted(release_data.keys())
                if names:
                    self.printer.print_information_line(model_type, names, value_color="green")
                else:
                    self.printer.print_information_line(
                        model_type, "No models found", value_color="red"
                    )
        else:
            self.printer.print_information_line(
                f"Available {model_type} models for download", "", level=0
            )
            names = sorted(self.remote_models[model_type].keys())
            if names:
                for name in names:
                    self.printer.print_information_line("", name, value_color="green", level=1)
            else:
                self.printer.print_information_line(
                    "", "No models found", value_color="yellow", level=1
                )

    def download_model(
        self, model_type: str, model_name=typing.Optional[str], ignore_cache=False
    ) -> None:
        """
        Download a model to MFA's temporary directory

        Parameters
        ----------
        model_type: str
            Model type
        model_name: str
            Name of model
        ignore_cache: bool
            Flag to ignore previously downloaded files
        """
        if not model_name:
            return self.print_remote_models(model_type)
        if not self.synced_remote:
            self.refresh_remote()
        if model_name not in self.remote_models[model_type]:
            raise RemoteModelNotFoundError(
                model_name, model_type, sorted(self.remote_models[model_type].keys())
            )
        release = self.remote_models[model_type][model_name]
        local_path = os.path.join(
            MODEL_TYPES[model_type].pretrained_directory(), release.download_file_name
        )
        if os.path.exists(local_path) and not ignore_cache:
            return
        headers = {"Accept": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        if release.download_link in self._cache_info:
            headers["If-None-Match"] = self._cache_info[release.download_link]
        r = requests.get(release.download_link, headers=headers)
        if r.status_code >= 400:
            raise ModelsConnectionError(r.status_code, r.json(), r.headers)
        self._cache_info[release.download_link] = r.headers["etag"]
        with mfa_open(local_path, "wb") as f:
            f.write(r.content)
        self.refresh_local()
        logger.info(
            f"Saved model to {local_path}, you can now use {model_name} in place of {model_type} paths in mfa commands."
        )
