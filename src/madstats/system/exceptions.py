class FrozenInstanceError(Exception):
    """Frozen instance exception"""

    def __init__(self, message="This class has been frozen."):
        super(FrozenInstanceError, self).__init__(message)

class AnalysisQueryError(Exception):
    """Frozen instance exception"""

    def __init__(self, message="This analysis has not been found."):
        super(AnalysisQueryError, self).__init__(message)
