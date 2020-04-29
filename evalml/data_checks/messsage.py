class Message:
    """Base class for all Messages."""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DataCheckError(Message):
    """Message subclass for errors returned by data checks."""

    def __init__(self, message):
        super().__init__(message=message)


class DataCheckWarning(Message):
    """Message subclass for warnings returned by data checks."""

    def __init__(self, message):
        super().__init__(message=message)
