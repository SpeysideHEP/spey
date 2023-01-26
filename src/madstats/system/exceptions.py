class FrozenInstanceError(Exception):
    """Frozen instance exception"""

    def __init__(self, message="This class has been frozen."):
        super(FrozenInstanceError, self).__init__(message)


class AnalysisQueryError(Exception):
    """Analysis query exception"""

    def __init__(self, message="This analysis has not been found."):
        super(AnalysisQueryError, self).__init__(message)


class NegativeExpectedYields(Exception):
    """Negative expected yields exception"""

    def __init__(self, message="Negative expected yields has been found."):
        super(NegativeExpectedYields, self).__init__(message)


class UnknownCrossSection(Exception):
    """Unknown cross-section exception"""

    def __init__(self, message="Please initialise cross section value."):
        super(UnknownCrossSection, self).__init__(message)