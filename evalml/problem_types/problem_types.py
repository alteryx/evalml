from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class ProblemTypes(AutoName):
    BINARY = auto()
    MULTICLASS = auto()
    REGRESSION = auto()
