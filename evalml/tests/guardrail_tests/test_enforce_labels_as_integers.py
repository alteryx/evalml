import pandas as pd

from evalml.guardrails import enforce_labels_as_integers
from evalml.problem_types import ProblemTypes


def test_enforce_labels_as_integers():
    y = pd.Series([1, 0, 1, 1], dtype=bool)
    labels, unique_label_strings = enforce_labels_as_integers(y, [ProblemTypes.BINARY])
    pd.testing.assert_series_equal(labels, pd.Series([1, 0, 1, 1], dtype=bool))
    assert unique_label_strings == None

    y = pd.Series([1, 0, 1, 1], dtype=int)
    labels, unique_label_strings = enforce_labels_as_integers(y, [ProblemTypes.BINARY])
    pd.testing.assert_series_equal(labels, pd.Series([1, 0, 1, 1], dtype=int))
    assert unique_label_strings == None

    y = pd.Series([1.5, 0, 1, 1], dtype=float)
    labels, unique_label_strings = enforce_labels_as_integers(y, [ProblemTypes.BINARY])
    pd.testing.assert_series_equal(labels, pd.Series([1.5, 0, 1, 1], dtype=float))
    assert unique_label_strings == None

