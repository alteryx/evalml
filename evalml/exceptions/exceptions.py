class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""
    pass


class ObjectiveNotFoundError(Exception):
    """Exception to raise when specified objective does not exist."""
    pass


class IllFormattedClassNameError(Exception):
    """Exception to raise when a class name does not comply with EvalML standards"""
    pass


class DimensionMismatchError(Exception):
    """Exception to raise when two input dimensions are mismatched and cannot be compared"""
    pass
