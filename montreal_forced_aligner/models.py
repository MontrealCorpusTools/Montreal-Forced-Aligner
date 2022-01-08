"""
Model classes
=============

"""
from __future__ import annotations

import json
import os
import shutil
import typing
from shutil import copy, copyfile, make_archive, move, rmtree, unpack_archive
from typing import TYPE_CHECKING, Collection, Dict, Optional, Union

import yaml

from montreal_forced_aligner.abc import MfaModel, ModelExporterMixin
from montreal_forced_aligner.data import PhoneSetType
from montreal_forced_aligner.exceptions import (
    LanguageModelNotFoundError,
    ModelLoadError,
    PronunciationAcousticMismatchError,
)
from montreal_forced_aligner.helper import EnhancedJSONEncoder, TerminalPrinter

if TYPE_CHECKING:
    from logging import Logger

    from montreal_forced_aligner.abc import MetaDict
    from montreal_forced_aligner.dictionary.pronunciation import DictionaryMixin
    from montreal_forced_aligner.g2p.trainer import G2PTrainer


# default format for output
FORMAT = "zip"

__all__ = [
    "Archive",
    "LanguageModel",
    "AcousticModel",
    "IvectorExtractorModel",
    "DictionaryModel",
    "G2PModel",
    "MODEL_TYPES",
]


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

    def __init__(self, source: str, root_directory: Optional[str] = None):
        from .config import get_temporary_directory

        if root_directory is None:
            root_directory = os.path.join(get_temporary_directory(), "extracted_models")
        self.root_directory = root_directory
        self.source = source
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        else:
            self.dirname = os.path.join(root_directory, self.name)
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
            if f == "phones.sym":
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
            with open(meta_path, "r", encoding="utf8") as f:
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
        with open(os.path.join(self.dirname, "meta.json"), "w", encoding="utf8") as f:
            json.dump(trainer.meta, f)

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
            root_directory = get_temporary_directory()

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

    files = ["final.mdl", "final.alimdl", "final.occs", "lda.mat", "tree"]
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
        with open(os.path.join(self.dirname, "meta.json"), "w", encoding="utf8") as f:
            json.dump(trainer.meta, f)

    @property
    def parameters(self) -> MetaDict:
        """Parameters to pass to top-level workers"""
        params = {**self.meta["features"]}
        params["non_silence_phones"] = {x for x in self.meta["phones"]}
        params["oov_phone"] = self.meta["oov_phone"]
        params["optional_silence_phone"] = self.meta["optional_silence_phone"]
        params["other_noise_phone"] = self.meta["other_noise_phone"]
        params["phone_set_type"] = self.meta["phone_set_type"]
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
            "pitch": False,
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
                with open(meta_path, "r", encoding="utf8") as f:
                    if format == "yaml":
                        self._meta = yaml.safe_load(f)
                    else:
                        self._meta = json.load(f)
                if self._meta["features"] == "mfcc+deltas":
                    self._meta["features"] = default_features
            if "phone_type" not in self._meta:
                self._meta["phone_type"] = "triphone"
            if "optional_silence_phone" not in self._meta:
                self._meta["optional_silence_phone"] = "sil"
            if "oov_phone" not in self._meta:
                self._meta["oov_phone"] = "spn"
            if "other_noise_phone" not in self._meta:
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
            if (
                "uses_splices" not in self._meta["features"]
                or not self._meta["features"]["uses_splices"]
            ):
                self._meta["features"]["uses_splices"] = os.path.exists(
                    os.path.join(self.dirname, "lda.mat")
                )
                if self._meta["features"]["uses_splices"]:
                    self._meta["features"]["uses_deltas"] = False
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

    def log_details(self, logger: Logger) -> None:
        """
        Log metadata information to a logger

        Parameters
        ----------
        logger: :class:`~logging.Logger`
            Logger to send debug information to
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
        if missing_phones:
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
        "plda",
        "mean.vec",
        "trans.mat",
    ]
    extensions = [".zip", ".ivector"]

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

        with open(os.path.join(self.dirname, "meta.json"), "w", encoding="utf8") as f:
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
                with open(meta_path, "r", encoding="utf8") as f:
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
        """G2p model's symbols path"""
        return os.path.join(self.dirname, "phones.sym")

    def add_sym_path(self, source_directory: str) -> None:
        """
        Add symbols file into archive

        Parameters
        ----------
        source_directory: str
            Source directory path
        """
        if not os.path.exists(self.sym_path):
            copyfile(os.path.join(source_directory, "phones.sym"), self.sym_path)

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
    """

    model_type = "language_model"

    arpa_extension = ".arpa"
    extensions = [f".{FORMAT}", arpa_extension, ".lm"]

    def __init__(self, source: str, root_directory: Optional[str] = None):
        if source in LanguageModel.get_available_models():
            source = LanguageModel.get_pretrained_path(source)
        from .config import get_temporary_directory

        if root_directory is None:
            root_directory = get_temporary_directory()

        if source.endswith(self.arpa_extension):
            self.root_directory = root_directory
            self._meta = {}
            self.name, _ = os.path.splitext(os.path.basename(source))
            self.dirname = os.path.join(root_directory, self.name)
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
        raise LanguageModelNotFoundError()

    @property
    def carpa_path(self) -> str:
        """
        Uses the largest language model for rescoring
        """
        for path in [self.large_arpa_path, self.medium_arpa_path, self.small_arpa_path]:
            if os.path.exists(path):
                return path
        raise LanguageModelNotFoundError()

    @property
    def small_arpa_path(self) -> str:
        """Small arpa path"""
        return os.path.join(self.dirname, f"{self.name}_small{self.arpa_extension}")

    @property
    def medium_arpa_path(self) -> str:
        """Medium arpa path"""
        return os.path.join(self.dirname, f"{self.name}_med{self.arpa_extension}")

    @property
    def large_arpa_path(self) -> str:
        """Large arpa path"""
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

            root_directory = get_temporary_directory()
        self.path = path
        self.dirname = os.path.join(root_directory, self.name)
        self.pronunciation_probabilities = True
        self.silence_probabilities = True
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
        with open(self.path, "r", encoding="utf8") as f:
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

    def pretty_print(self):
        """
        Pretty print the dictionary's metadata using TerminalPrinter
        """
        from montreal_forced_aligner.dictionary.pronunciation import PronunciationDictionary

        printer = TerminalPrinter()
        configuration_data = {"Dictionary": {"name": (self.name, "green"), "data": self.meta}}
        dictionary = PronunciationDictionary(
            self.path, temporary_directory=self.dirname, phone_set_type=self.phone_set_type
        )
        configuration_data["Dictionary"]["data"]["phones"] = sorted(dictionary.non_silence_phones)
        if self.phone_set_type.has_base_phone_regex:
            configuration_data["Dictionary"]["data"]["base_phones"] = {
                k: sorted(v) for k, v in sorted(dictionary.base_phones.items())
            }
        else:
            configuration_data["Dictionary"]["data"]["base_phones"] = "None"
        if len(dictionary.graphemes) < 50:
            configuration_data["Dictionary"]["data"]["graphemes"] = sorted(dictionary.graphemes)
        else:
            configuration_data["Dictionary"]["data"][
                "graphemes"
            ] = f"{len(dictionary.graphemes)} graphemes"
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

    def load_dictionary_paths(self) -> Dict[str, DictionaryModel]:
        """
        Load the pronunciation dictionaries

        Returns
        -------
        dict[str, :class:`~montreal_forced_aligner.models.DictionaryModel`]
            Mapping of component pronunciation dictionaries
        """
        mapping = {}
        if self.is_multiple:
            with open(self.path, "r", encoding="utf8") as f:
                data = yaml.safe_load(f)
                for speaker, path in data.items():
                    mapping[speaker] = DictionaryModel(path)
        else:
            mapping["default"] = self
        return mapping


MODEL_TYPES = {
    "acoustic": AcousticModel,
    "g2p": G2PModel,
    "dictionary": DictionaryModel,
    "language_model": LanguageModel,
    "ivector": IvectorExtractorModel,
}
