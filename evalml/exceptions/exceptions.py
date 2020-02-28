class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""
    pass


class ObjectiveNotFoundError(Exception):
    """Exception to raise when specified objective does not exist."""
    pass
