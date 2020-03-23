class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""
    pass


class IllFormattedClassNameError(Exception):
    """Exception to raise when a class name does not comply with EvalML standards"""
    pass
