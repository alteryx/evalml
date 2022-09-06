import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.pipelines.components import StandardScaler
from evalml.problem_types import ProblemTypes


def test_standard_scaler_applies_to_numeric_columns_only(
    get_test_data_from_configuration,
):
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

    X, y = get_test_data_from_configuration(
        "ww",
        ProblemTypes.BINARY,
        column_names=[
            "categorical",
            "url",
            "email",
            "numerical",
            "int_null",
            "age_null",
            "bool_null",
            "dates",
        ],
    )

    std_scaler = StandardScaler()
    std_scaler.fit(X)
    X_t = std_scaler.transform(X)

    expected = X.copy()
    expected.ww.init(schema=X.ww.schema, logical_types=X.ww.logical_types)
    expected["numerical"] = [
        -1.647509,
        -1.474087,
        -1.300665,
        -1.127243,
        -0.953821,
        -0.780399,
        -0.606977,
        -0.433555,
        -0.260133,
        -0.086711,
        0.086711,
        0.260133,
        0.433555,
        0.606977,
        0.780399,
        0.953821,
        1.127243,
        1.300665,
        1.474087,
        1.647509,
    ]
    expected["int_null"] = [
        -1.463697,
        -1.147222,
        -0.830747,
        np.NaN,
        -0.197797,
        np.NaN,
        0.435153,
        0.751628,
        1.068103,
        1.384579,
        -1.463697,
        -1.147222,
        -0.830747,
        np.NaN,
        -0.197797,
        np.NaN,
        0.435153,
        0.751628,
        1.068103,
        1.384579,
    ]
    expected["age_null"] = [
        -1.463697,
        -1.147222,
        -0.830747,
        np.NaN,
        -0.197797,
        np.NaN,
        0.435153,
        0.751628,
        1.068103,
        1.384579,
        -1.463697,
        -1.147222,
        -0.830747,
        np.NaN,
        -0.197797,
        np.NaN,
        0.435153,
        0.751628,
        1.068103,
        1.384579,
    ]
    numeric_columns = ["numerical", "int_null", "age_null"]
    for col in numeric_columns:
        assert_series_equal(X_t[col], expected[col])
        assert X_t.ww.logical_types[col].type_string == "double"

    # assert all other columns remain the same
    expected.drop(numeric_columns + ["dates"], inplace=True, axis=1)
    X_t.drop(numeric_columns, inplace=True, axis=1)
    assert_frame_equal(X_t, expected)
