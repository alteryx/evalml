"""Exception thrown by tuner classes."""


class NoParamsException(Exception):
    """Raised when a tuner exhausts its search space and runs out of parameters to propose."""

    pass


class ParameterError(Exception):
    """Raised when a tuner encounters an error with the parameters being used with it."""

    pass
