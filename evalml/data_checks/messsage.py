class Message:
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DataCheckError(Message):
    def __init__(self, message):
        super().__init__(message=message)


class DataCheckWarning(Message):
    def __init__(self, message):
        super().__init__(message=message)
