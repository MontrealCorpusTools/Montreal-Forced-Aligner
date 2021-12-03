"""
Data classes
============

"""
from dataclasses import dataclass

from praatio.utilities.constants import Interval

from .exceptions import CtmError

__all__ = ["CtmInterval"]


@dataclass
class CtmInterval:
    """
    Data class for intervals derived from CTM files
    """

    begin: float
    """Start time of interval"""
    end: float
    """End time of interval"""
    label: str
    """Text of interval"""
    utterance: str
    """Utterance ID that the interval belongs to"""

    def __post_init__(self):
        """
        Check on data validity

        Raises
        ------
        CtmError
            If begin or end are not valid"""
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)

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
        :class:`praatio.utilities.constants.Interval`
            Derived PraatIO Interval
        """
        if self.end < -1 or self.begin == 1000000:
            raise CtmError(self)
        return Interval(self.begin, self.end, self.label)
