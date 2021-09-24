from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal


from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import LabelEncoder


def test_label_encoder_init():
    encoder = LabelEncoder()
    assert encoder.parameters == {}
    assert encoder.random_seed == 0


def test_fit_transform_y_is_None():
    X = pd.DataFrame({})
    encoder = LabelEncoder()
    encoder.fit(X)
    # should raise error
    # X_t, y_t = encoder.transform(X, y)


def test_fit_transform():
    X = pd.DataFrame({})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X_expected, X_t)
