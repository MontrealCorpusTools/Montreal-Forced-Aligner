from abc import ABCMeta, abstractmethod
from typing import Dict, List

from montreal_forced_aligner.abc import MfaWorker
from montreal_forced_aligner.dictionary.mixins import DictionaryMixin


class G2PMixin(metaclass=ABCMeta):
    """
    Abstract mixin class for G2P functionality

    Parameters
    ----------
    include_bracketed: bool
        Flag for whether to generate pronunciations for fully bracketed words, defaults to False
    num_pronunciations: int
        Number of pronunciations to generate, defaults to 0
    g2p_threshold: float
        Weight threshold for generating pronunciations between 0 and 1,
        1 returns the optimal path only, 0 returns all pronunciations, defaults to 0.99 (only used if num_pronunciations is 0)
    """

    def __init__(
        self,
        include_bracketed: bool = False,
        num_pronunciations: int = 0,
        g2p_threshold: float = 0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_pronunciations = num_pronunciations
        self.g2p_threshold = g2p_threshold
        self.include_bracketed = include_bracketed

    @abstractmethod
    def generate_pronunciations(self) -> Dict[str, List[str]]:
        """
        Generate pronunciations

        Returns
        -------
        dict[str, list[str]]
            Mappings of keys to their generated pronunciations
        """
        ...

    @property
    @abstractmethod
    def words_to_g2p(self):
        """Words to produce pronunciations"""
        ...


class G2PTopLevelMixin(MfaWorker, DictionaryMixin, G2PMixin):
    """
    Abstract mixin class for top-level G2P functionality

    See Also
    --------
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For base MFA parameters
    :class:`~montreal_forced_aligner.dictionary.mixins.DictionaryMixin`
        For dictionary parsing parameters
    :class:`~montreal_forced_aligner.g2p.mixins.G2PMixin`
        For base G2P parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def workflow_identifier(self):
        """G2P workflow identifier"""
        return "g2p"

    def generate_pronunciations(self) -> Dict[str, List[str]]:
        """
        Generate pronunciations

        Returns
        -------
        dict[str, list[str]]
            Mappings of keys to their generated pronunciations
        """
        raise NotImplementedError

    def export_pronunciations(self, output_file_path: str) -> None:
        """
        Output pronunciations to text file

        Parameters
        ----------
        output_file_path: str
            Path to save
        """
        results = self.generate_pronunciations()
        with open(output_file_path, "w", encoding="utf8") as f:
            for (orthography, word) in results.items():
                if not word.pronunciations:
                    continue
                for p in word.pronunciations:
                    if not p:
                        continue
                    f.write(f"{orthography}\t{p}\n")
