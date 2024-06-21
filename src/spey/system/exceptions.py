"""Module specific exceptions"""

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
