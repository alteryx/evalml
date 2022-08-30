from cmath import exp

import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import Boolean, Categorical, Double, Integer

from evalml.pipelines.components import StandardScaler


def test_standard_scaler_applies_to_numeric_columns_only():
    X = pd.DataFrame(
        {
            "bool": [True, False, True],
            "boolean_nullable": [True, None, True],
            "categorical": ["a", "b", "c"],
            "double": [1.2, 3.5, 6.5],
            "integer": [1, 2, 3],
            "integer_nullable": [1, None, 3],
        },
    )
    X.ww.init(
        logical_types={
            "bool": "Boolean",
            "boolean_nullable": "BooleanNullable",
            "categorical": "Categorical",
            "double": "Double",
            "integer": "Integer",
            "integer_nullable": "IntegerNullable",
        },
    )
    expected = pd.DataFrame(
        {
            "bool": [True, False, True],
            "boolean_nullable": [True, None, True],
            "categorical": ["a", "b", "c"],
            "double": [-1.167436, -0.107527, 1.274963],
            "integer": [-1.224745, 0.0, 1.224745],
            "integer_nullable": [-1, None, 1],
        },
    )
    expected.ww.init(
        logical_types={
            "bool": "Boolean",
            "boolean_nullable": "BooleanNullable",
            "categorical": "Categorical",
            "double": "Double",
            "integer": "Double",
            "integer_nullable": "Double",
        },
    )
    std_scaler = StandardScaler()
    std_scaler.fit(X)
    X_t = std_scaler.transform(X)
    assert_frame_equal(X_t, expected)
