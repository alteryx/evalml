"""The supported types of machine learning problems."""
from evalml.problem_types.problem_types import ProblemTypes
from evalml.problem_types.utils import (
    handle_problem_types,
    detect_problem_type,
    is_regression,
    is_binary,
    is_multiclass,
    is_classification,
    is_time_series,
)
