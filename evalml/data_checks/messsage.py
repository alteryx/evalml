class Message:
    """Base class for all Messages."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DataCheckError(Message):
    """Message class for errors returned by data checks."""
    def __init__(self, message):
        super().__init__(message=message)


class DataCheckWarning(Message):
    """Message class for warnings returned by data checks."""
    def __init__(self, message):
        super().__init__(message=message)
