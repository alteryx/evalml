class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""
    pass


class PipelineNotFoundError(Exception):
    """An exception raised when a particular pipeline is not found in automl search results"""
    pass


class ObjectiveNotFoundError(Exception):
    """Exception to raise when specified objective does not exist."""
    pass


class IllFormattedClassNameError(Exception):
    """Exception to raise when a class name does not comply with EvalML standards"""
    pass


class MissingComponentError(Exception):
    """An exception raised when a component is not found in all_components()"""
    pass
