class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""
    pass


class ObjectiveNotFoundError(Exception):
    """Exception to raise when specified objective does not exist."""
    pass


class IllFormattedClassNameError(Exception):
    """Exception to raise when a class name does not comply with EvalML standards"""
    pass

class NegativeTargetNotAcceptedError(Exception):
    """Exception to raise when an objective does not accept negative target values"""
    pass