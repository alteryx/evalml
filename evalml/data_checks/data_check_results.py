

class DataCheckResults:

    def __init__(self, errors=None, warnings=None):
        self.errors = errors or []
        self.warnings = warnings or []

    def __eq__(self, other):
        return isinstance(other, DataCheckResults) and self.errors == other.errors and self.warnings == other.warnings

    def to_json(self):
        messages = []
        for error in self.errors:
            messages.append(error.to_json())
        for warning in self.warnings:
            messages.append(warning.to_json())
        return messages
