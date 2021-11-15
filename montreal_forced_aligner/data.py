"""
Data classes
============

"""
from dataclasses import dataclass
from typing import List

from praatio.utilities.constants import Interval

__all__ = ["CtmInterval"]


@dataclass
class CtmInterval:
    """
    Data class for intervals derived from CTM files

    Attributes
    ----------
    begin: float
        Start time of interval
    end: float
        End time of interval
    label: str
        Text of interval
    utterance: str
        Utterance ID that the interval belongs to
    """

    begin: float
    end: float
    label: str
    utterance: str

    def shift_times(self, offset: float):
        """
        Shift times of the interval based on some offset (i.e., segments in Kaldi)

        Parameters
        ----------
        offset: float
            Offset to add to the interval's begin and end

        """
        self.begin += offset
        self.end += offset

    def to_tg_interval(self) -> Interval:
        """
        Converts the CTMInterval to `PraatIO's Interval class <http://timmahrt.github.io/praatIO/praatio/utilities/constants.html#Interval>`_

        Returns
        -------
        :class:`~praatio.utilities.constants.Interval`
            Derived PraatIO Interval
        """
        return Interval(self.begin, self.end, self.label)


CtmType = List[CtmInterval]
