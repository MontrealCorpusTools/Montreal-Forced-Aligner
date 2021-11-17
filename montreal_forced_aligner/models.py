"""
Model classes
=============

"""
from __future__ import annotations

import os
from shutil import copy, copyfile, make_archive, move, rmtree, unpack_archive
from typing import TYPE_CHECKING, Collection, Dict, Optional, Tuple, Union

import yaml

from .abc import Dictionary, MetaDict, MfaModel, Trainer
from .exceptions import (
    LanguageModelNotFoundError,
    ModelLoadError,
    PronunciationAcousticMismatchError,
)
from .helper import TerminalPrinter

if TYPE_CHECKING:
    from logging import Logger

    from .config import FeatureConfig
    from .config.dictionary_config import DictionaryConfig
    from .config.train_config import TrainingConfig
    from .dictionary import PronunciationDictionary


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
        from .config import TEMP_DIR

        if root_directory is None:
            root_directory = TEMP_DIR
        self.root_directory = root_directory
        self.source = source
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        else:
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(root_directory, exist_ok=True)
                unpack_archive(source, self.dirname)
                files = os.listdir(self.dirname)
                old_dir_path = os.path.join(self.dirname, files[0])
                if len(files) == 1 and os.path.isdir(old_dir_path):  # Backwards compatibility
                    for f in os.listdir(old_dir_path):
                        move(os.path.join(old_dir_path, f), os.path.join(self.dirname, f))
                    os.rmdir(old_dir_path)

    def get_subclass_object(
        self,
    ) -> Union[AcousticModel, G2PModel, LanguageModel, IvectorExtractorModel]:
        """
        Instantiate subclass models based on files contained in the archive

        Returns
        -------
        Union[AcousticModel, G2PModel, LanguageModel, IvectorExtractor]
            Subclass model that was auto detected
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

    def pretty_print(self):
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
            meta_path = os.path.join(self.dirname, "meta.yaml")
            with open(meta_path, "r", encoding="utf8") as f:
                self._meta = yaml.safe_load(f)
        return self._meta

    def add_meta_file(self, trainer: Trainer) -> None:
        """
        Add a metadata file from a given trainer to the model

        Parameters
        ----------
        trainer: Trainer
            The trainer to construct the metadata from
        """
        with open(os.path.join(self.dirname, "meta.yaml"), "w", encoding="utf8") as f:
            yaml.dump(trainer.meta, f)

    @classmethod
    def empty(cls, head: str, root_directory: Optional[str] = None) -> Archive:
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
        Archive
            Model constructed from the empty directory
        """
        from .config import TEMP_DIR

        if root_directory is None:
            root_directory = TEMP_DIR

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

    def add_meta_file(self, trainer: Trainer) -> None:
        """
        Add metadata file from a model trainer

        Parameters
        ----------
        trainer: :class:`~montreal_forced_aligner.abc.Trainer`
            Trainer to supply metadata information about the acoustic model
        """
        with open(os.path.join(self.dirname, "meta.yaml"), "w", encoding="utf8") as f:
            yaml.dump(trainer.meta, f)

    @property
    def feature_config(self) -> FeatureConfig:
        """
        Return the FeatureConfig used in training the model
        """
        from .config.feature_config import FeatureConfig

        fc = FeatureConfig()
        fc.update(self.meta["features"])
        return fc

    def adaptation_config(self) -> Tuple[TrainingConfig, DictionaryConfig]:
        """
        Generate an adaptation configuration

        Returns
        -------
        TrainingConfig
            Configuration to be used in adapting the acoustic model to new data
        """
        from .config.train_config import load_no_sat_adapt, load_sat_adapt

        if self.meta["features"]["fmllr"]:
            train, align, dictionary = load_sat_adapt()
        else:
            train, align, dictionary = load_no_sat_adapt()
        return train, dictionary

    @property
    def meta(self) -> MetaDict:
        """
        Metadata information for the acoustic model
        """
        default_features = {
            "type": "mfcc",
            "use_energy": False,
            "frame_shift": 10,
            "pitch": False,
            "fmllr": True,
        }
        if not self._meta:
            meta_path = os.path.join(self.dirname, "meta.yaml")
            if not os.path.exists(meta_path):
                self._meta = {
                    "version": "0.9.0",
                    "architecture": "gmm-hmm",
                    "multilingual_ipa": False,
                    "features": default_features,
                }
            else:
                with open(meta_path, "r", encoding="utf8") as f:
                    self._meta = yaml.safe_load(f)
                if self._meta["features"] == "mfcc+deltas":
                    self._meta["features"] = default_features
            if "uses_lda" not in self._meta:  # Backwards compatibility
                self._meta["uses_lda"] = os.path.exists(os.path.join(self.dirname, "lda.mat"))
            if "multilingual_ipa" not in self._meta:
                self._meta["multilingual_ipa"] = False
            if "uses_sat" not in self._meta:
                self._meta["uses_sat"] = False
            if "phone_type" not in self._meta:
                self._meta["phone_type"] = "triphone"
            self._meta["phones"] = set(self._meta.get("phones", []))
            self._meta["has_speaker_independent_model"] = os.path.exists(
                os.path.join(self.dirname, "final.alimdl")
            )
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
            "Type": self.meta["features"]["type"],
            "Frame shift": self.meta["features"]["frame_shift"],
        }
        if self.meta["phones"]:
            configuration_data["Acoustic model"]["data"]["Phones"] = self.meta["phones"]
        else:
            configuration_data["Acoustic model"]["data"]["Phones"] = ("None found!", "red")

        configuration_data["Acoustic model"]["data"]["Configuration options"] = {
            "Multilingual IPA": self.meta["multilingual_ipa"],
            "Performs speaker adaptation": self.meta["uses_sat"],
            "Has speaker-independent model": self.meta["has_speaker_independent_model"],
            "Performs LDA on features": self.meta["uses_lda"],
        }
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
        meta_path = os.path.join(self.dirname, "meta.yaml")
        logger.debug("Acoustic model meta path: " + meta_path)
        if not os.path.exists(meta_path):
            logger.debug("META.YAML DOES NOT EXIST, this may cause issues in validating the model")
        logger.debug("Acoustic model meta information:")
        stream = yaml.dump(self.meta)
        logger.debug(stream)
        logger.debug("")

    def validate(self, dictionary: Union[Dictionary, G2PModel]) -> None:
        """
        Validate this acoustic model against a pronunciation dictionary or G2P model to ensure their
        phone sets are compatible

        Parameters
        ----------
        dictionary: Union[DictionaryConfig, G2PModel]
            PronunciationDictionary or G2P model to compare phone sets with

        Raises
        ------
        PronunciationAcousticMismatchError
            If there are phones missing from the acoustic model
        """
        if isinstance(dictionary, G2PModel):
            missing_phones = dictionary.meta["phones"] - set(self.meta["phones"])
        else:
            missing_phones = dictionary.config.non_silence_phones - set(self.meta["phones"])
        if missing_phones:
            raise (PronunciationAcousticMismatchError(missing_phones))


class IvectorExtractorModel(Archive):
    """
    Model class for IvectorExtractor
    """

    model_files = [
        "final.ie",
        "final.ubm",
        "final.dubm",
        "plda",
        "mean.vec",
        "trans.mat",
    ]
    extensions = [".zip", ".ivector"]

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

    @property
    def feature_config(self) -> FeatureConfig:
        """
        Return the FeatureConfig used in training the model
        """
        from .config.feature_config import FeatureConfig

        fc = FeatureConfig()
        fc.update(self.meta["features"])
        return fc


class G2PModel(Archive):
    extensions = [".zip", ".g2p"]

    def add_meta_file(
        self, dictionary: PronunciationDictionary, architecture: Optional[str] = None
    ) -> None:
        """
        Construct meta data information for the G2P model from the dictionary it was trained from

        Parameters
        ----------
        dictionary: PronunciationDictionary
            PronunciationDictionary that was the training data for the G2P model
        architecture: str, optional
            Architecture of the G2P model, defaults to "pynini"
        """
        from .utils import get_mfa_version

        if architecture is None:
            architecture = "pynini"
        with open(os.path.join(self.dirname, "meta.yaml"), "w", encoding="utf8") as f:
            meta = {
                "phones": sorted(dictionary.config.non_silence_phones),
                "graphemes": sorted(dictionary.graphemes),
                "architecture": architecture,
                "version": get_mfa_version(),
            }
            yaml.dump(meta, f)

    @property
    def meta(self) -> dict:
        """Metadata for the G2P model"""
        if not self._meta:
            meta_path = os.path.join(self.dirname, "meta.yaml")
            if not os.path.exists(meta_path):
                self._meta = {"version": "0.9.0", "architecture": "phonetisaurus"}
            else:
                with open(meta_path, "r", encoding="utf8") as f:
                    self._meta = yaml.safe_load(f)
            self._meta["phones"] = set(self._meta.get("phones", []))
            self._meta["graphemes"] = set(self._meta.get("graphemes", []))
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
        source: str
            Source directory path
        """
        if not os.path.exists(self.sym_path):
            copyfile(os.path.join(source_directory, "phones.sym"), self.sym_path)

    def add_fst_model(self, source_directory: str) -> None:
        """
        Add FST file into archive

        Parameters
        ----------
        source: str
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
            print(
                "WARNING! The following graphemes were not found in the specified G2P model: "
                f"{' '.join(sorted(missing_graphemes))}"
            )
            return False
        else:
            return True


class LanguageModel(Archive):
    """
    Class for MFA language models
    """

    arpa_extension = ".arpa"
    extensions = [f".{FORMAT}", arpa_extension, ".lm"]

    def __init__(self, source: str, root_directory: Optional[str] = None):
        from .config import TEMP_DIR

        if root_directory is None:
            root_directory = TEMP_DIR
        self.root_directory = root_directory
        self._meta = {}
        self.name, _ = os.path.splitext(os.path.basename(source))
        if os.path.isdir(source):
            self.dirname = os.path.abspath(source)
        elif source.endswith(self.arpa_extension):
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname, exist_ok=True)
            copy(source, self.large_arpa_path)
        elif any(source.endswith(x) for x in self.extensions):
            base = root_directory
            self.dirname = os.path.join(root_directory, self.name)
            if not os.path.exists(self.dirname):
                os.makedirs(root_directory, exist_ok=True)
                unpack_archive(source, base)

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
        return os.path.join(self.dirname, self.name + "_small" + self.arpa_extension)

    @property
    def medium_arpa_path(self) -> str:
        """Medium arpa path"""
        return os.path.join(self.dirname, self.name + "_med" + self.arpa_extension)

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
        name = os.path.basename(arpa_path)
        copyfile(arpa_path, os.path.join(self.dirname, name))


class DictionaryModel(MfaModel):
    """
    Class for representing MFA pronunciation dictionaries
    """

    extensions = [".dict", ".txt", ".yaml", ".yml"]

    def __init__(self, path: str):
        self.path = path
        count = 0
        self.pronunciation_probabilities = True
        self.silence_probabilities = True
        with open(self.path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                _ = line.pop(0)  # word
                next_item = line.pop(0)
                if self.pronunciation_probabilities:
                    try:
                        prob = float(next_item)
                        if prob > 1 or prob < 0:
                            raise ValueError
                    except ValueError:
                        self.pronunciation_probabilities = False
                try:
                    next_item = line.pop(0)
                except IndexError:
                    self.silence_probabilities = False
                if self.silence_probabilities:
                    try:
                        prob = float(next_item)
                        if prob > 1 or prob < 0:
                            raise ValueError
                    except ValueError:
                        self.silence_probabilities = False
                count += 1
                if count > 10:
                    break

    @property
    def meta(self) -> MetaDict:
        return {
            "pronunciation_probabilities": self.pronunciation_probabilities,
            "silence_probabilities": self.silence_probabilities,
        }

    def add_meta_file(self, trainer: Trainer) -> None:
        raise NotImplementedError

    def pretty_print(self):
        """
        Pretty print the dictionary's meta data using TerminalPrinter
        """
        printer = TerminalPrinter()
        configuration_data = {"Dictionary": {"name": (self.name, "green"), "data": self.meta}}
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
    def is_multiple(self):
        return os.path.splitext(self.path)[1] in [".yaml", ".yml"]

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.path))[0]

    def load_dictionary_paths(self) -> Dict[str, DictionaryModel]:
        from .utils import get_available_dictionaries, get_dictionary_path

        mapping = {}
        if self.is_multiple:
            available_langs = get_available_dictionaries()
            with open(self.path, "r", encoding="utf8") as f:
                data = yaml.safe_load(f)
                for speaker, path in data.items():
                    if path in available_langs:
                        path = get_dictionary_path(path)
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
