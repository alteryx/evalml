import numpy as np
import pandas as pd

import pytest

from evalml.utils.gen_utils import (
    classproperty,
    convert_to_seconds,
    import_or_raise,
    normalize_confusion_matrix
)


def test_import_or_raise_errors():
    with pytest.raises(ImportError, match="Failed to import _evalml"):
        import_or_raise("_evalml")
    with pytest.raises(ImportError, match="error message"):
        import_or_raise("_evalml", "error message")


def test_import_or_raise_imports():
    math = import_or_raise("math", "error message")
    assert math.ceil(0.1) == 1


def test_convert_to_seconds():
    assert convert_to_seconds("10 s") == 10
    assert convert_to_seconds("10 sec") == 10
    assert convert_to_seconds("10 second") == 10
    assert convert_to_seconds("10 seconds") == 10

    assert convert_to_seconds("10 m") == 600
    assert convert_to_seconds("10 min") == 600
    assert convert_to_seconds("10 minute") == 600
    assert convert_to_seconds("10 minutes") == 600

    assert convert_to_seconds("10 h") == 36000
    assert convert_to_seconds("10 hr") == 36000
    assert convert_to_seconds("10 hour") == 36000
    assert convert_to_seconds("10 hours") == 36000


def test_normalize_confusion_matrix():
    conf_mat = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum() == 1.0

    # testing with pd.DataFrames
    conf_mat_df = pd.DataFrame()
    conf_mat_df["col_1"] = [0, 1, 2]
    conf_mat_df["col_2"] = [0, 0, 3]
    conf_mat_df["col_3"] = [2, 0, 0]
    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert list(conf_mat_normalized.columns) == ['col_1', 'col_2', 'col_3']

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0


def test_class_property():
    class MockClass:
        name = "MockClass"

        @classproperty
        def caps_name(cls):
            return cls.name.upper()

    assert MockClass.caps_name == "MOCKCLASS"
