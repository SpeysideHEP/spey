"""Module specific exceptions"""

import logging
import warnings
from functools import wraps

log = logging.getLogger("Spey")

# pylint: disable=logging-fstring-interpolation

__all__ = [
    "FrozenInstanceError",
    "AnalysisQueryError",
    "NegativeExpectedYields",
    "UnknownCrossSection",
    "UnknownTestStatistics",
    "InvalidInput",
    "PluginError",
    "MethodNotAvailable",
]


def warning_tracker(func: callable) -> callable:
    """Warning tracker decorator"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            result = func(*args, **kwargs)
            for warning in w:
                log.debug(
                    f"{warning.message} (file: {warning.filename}::L{warning.lineno})"
                )
        return result

    return wrapper


class FrozenInstanceError(Exception):
    """Frozen instance exception"""

    def __init__(self, message="This class has been frozen."):
        super().__init__(message)


class AnalysisQueryError(Exception):
    """Analysis query exception"""

    def __init__(self, message="This analysis has not been found."):
        super().__init__(message)


class NegativeExpectedYields(Exception):
    """Negative expected yields exception"""

    def __init__(self, message="Negative expected yields has been found."):
        super().__init__(message)


class UnknownCrossSection(Exception):
    """Unknown cross-section exception"""

    def __init__(self, message="Please initialise cross section value."):
        super().__init__(message)


class UnknownTestStatistics(Exception):
    """Unknown test statistics exception"""

    def __init__(self, message="Unknown test statistics."):
        super().__init__(message)


class InvalidInput(Exception):
    """Invalid input exception"""

    def __init__(self, message="Unknown input type."):
        super().__init__(message)


class PluginError(Exception):
    """Invalid plugin exception"""


class MethodNotAvailable(Exception):
    """If the method is not available for a given backend"""


class CanNotFindRoots(Exception):
    """Unable to find roots of the function"""


class UnknownComputer(Exception):
    """Unknown computation base"""


class CalculatorNotAvailable(Exception):
    """Unavailable calculator Exception"""


class CombinerNotAvailable(Exception):
    """Unavailable combination routine exception"""


class DistributionError(Exception):
    """Unknown Distribution"""


class AsimovTestStatZero(Exception):
    """Asimov Test Statistic is zero"""

    def __init__(
        self,
        message="Asimov test statistic is zero. "
        "Note: Asimov test statistic of zero indicates a "
        "lack of evidence for a signal or deviation from a null hypothesis.",
    ):
        super().__init__(message)
