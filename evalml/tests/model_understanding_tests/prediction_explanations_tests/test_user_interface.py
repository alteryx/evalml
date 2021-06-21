import copy
import json
from itertools import product

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding.prediction_explanations._user_interface import (
    _BinarySHAPTable,
    _make_json_serializable,
    _make_rows,
    _make_text_table,
    _MultiClassSHAPTable,
    _RegressionSHAPTable,
)

make_rows_test_cases = [
    ({"a": [0.2], "b": [0.1]}, 3, [["a", "1.20", "++"], ["b", "1.10", "+"]]),
    (
        {
            "a": [0.3],
            "b": [-0.9],
            "c": [0.5],
            "d": [0.33],
            "e": [-0.67],
            "f": [-0.2],
            "g": [0.71],
        },
        4,
        [
            ["g", "1.71", "++++"],
            ["c", "1.50", "+++"],
            ["e", "0.33", "----"],
            ["b", "0.10", "-----"],
        ],
    ),
    (
        {"a": [1.0], "f": [-1.0], "e": [0.0]},
        5,
        [["a", "2.00", "+++++"], ["e", "1.00", "+"], ["f", "0.00", "-----"]],
    ),
]


@pytest.mark.parametrize(
    "test_case,include_shap_values,include_string_features",
    product(make_rows_test_cases, [True, False], [True, False]),
)
def test_make_rows_and_make_table(
    test_case, include_shap_values, include_string_features
):
    values, top_k, answer = test_case

    pipeline_features = pd.DataFrame(
        {name: value[0] + 1 for name, value in values.items()}, index=[5]
    )

    if include_string_features:
        pipeline_features["a"] = ["foo-feature"]
        pipeline_features["b"] = [np.datetime64("2020-08-14")]

    if include_shap_values:
        new_answer = copy.deepcopy(answer)
        for row in new_answer:
            row.append("{:.2f}".format(values[row[0]][0]))
    else:
        new_answer = copy.deepcopy(answer)

    if include_string_features:
        filtered_answer = []
        for row in new_answer:
            filtered_answer.append(row)
            val = row[1]
            if row[0] == "a":
                val = "foo-feature"
            elif row[0] == "b":
                val = "2020-08-14 00:00:00"
            filtered_answer[-1][1] = val
        new_answer = filtered_answer

    assert (
        _make_rows(
            values,
            values,
            pipeline_features,
            pipeline_features,
            top_k,
            include_shap_values,
        )
        == new_answer
    )

    table = _make_text_table(
        values, values, pipeline_features, pipeline_features, top_k, include_shap_values
    ).splitlines()
    if include_shap_values:
        assert "SHAP Value" in table[0]
    # Subtracting two because a header and a line under the header are included in the table.
    assert len(table) - 2 == len(new_answer)


@pytest.mark.parametrize(
    "value,answer",
    [
        (np.int64(3), 3),
        (np.float32(3.2), 3.2),
        (np.str_("foo"), "foo"),
        (np.bool_(True), True),
    ],
)
def test_make_json_serializable(value, answer):
    value = _make_json_serializable(value)
    if answer != "foo":
        np.testing.assert_almost_equal(value, answer)
    else:
        assert value == answer
    json.dumps(value)


regression = {
    "a": [6.500],
    "b": [1.770],
    "c": [0.570],
    "d": [-0.090],
    "e": [-0.290],
    "f": [-1.910],
    "foo": [0.01],
    "bar": [-0.02],
}

regression_normalized = {
    "a": [0.6214],
    "b": [0.1692],
    "bar": [-0.0019],
    "c": [0.0544],
    "d": [-0.0086],
    "e": [-0.0277],
    "f": [-0.8],
    "foo": [0.0001],
}

regression_pipeline_features = pd.DataFrame(
    {
        "a": 7.5,
        "b": 2.77,
        "c": 1.57,
        "d": 0.91,
        "e": 0.71,
        "f": -0.21,
        "foo": -20,
        "bar": -30,
    },
    index=[31],
)
regression_original_features = pd.DataFrame(
    {
        "a": 0.75,
        "b": 0.277,
        "c": 0.57,
        "d": 1.91,
        "e": 1.71,
        "f": -1.21,
        "foo": -20,
        "bar": -40,
    },
    index=[31],
)

regression_table = """Feature Name  Feature Value Contribution to Prediction
                      =========================================================
                      a 7.50 ++++
                      b 2.77 +
                      f -0.21 -----""".splitlines()

regression_table_shap = """Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         a 7.50 ++++ 6.50
                         b 2.77 + 1.77
                         f -0.21 ----- -1.91""".splitlines()

regression_dict = {
    "explanations": [
        {
            "feature_names": ["a", "b", "f"],
            "feature_values": [7.5, 2.77, -0.21],
            "qualitative_explanation": ["++++", "+", "-----"],
            "quantitative_explanation": [None, None, None],
            "drill_down": {},
            "class_name": None,
            "expected_value": [0],
        }
    ]
}

regression_dict_shap = {
    "explanations": [
        {
            "feature_names": ["a", "b", "f"],
            "feature_values": [7.5, 2.77, -0.21],
            "qualitative_explanation": ["++++", "+", "-----"],
            "quantitative_explanation": [6.50, 1.77, -1.91],
            "drill_down": {},
            "class_name": None,
            "expected_value": [0],
        }
    ]
}

binary = [
    {"a": [0], "b": [0], "c": [0], "d": [0], "e": [0], "f": [0], "foo": [-1]},
    {
        "a": [1.180],
        "b": [0.0],
        "c": [1.120],
        "d": [-0.560],
        "e": [-2.600],
        "f": [-0.900],
        "foo": [-1],
    },
]

binary_normalized = [
    {
        "a": [0.0],
        "b": [0.0],
        "c": [0.0],
        "d": [0.0],
        "e": [0.0],
        "f": [0.0],
        "foo": [-1.0],
    },
    {
        "a": [0.16],
        "b": [0.0],
        "c": [0.15],
        "d": [-0.08],
        "e": [-0.35],
        "f": [-0.12],
        "foo": [-0.14],
    },
]

binary_pipeline_features = pd.DataFrame(
    {"a": 2.18, "b": 2.12, "c": 1.0, "d": -1.56, "e": -1.8, "f": -1.9, "foo": -20},
    index=[23],
)
binary_original_features = pd.DataFrame(
    {"a": 1.18, "b": 1.12, "c": 2.0, "d": -2.56, "e": -2.8, "f": -2.9, "foo": -30},
    index=[23],
)

binary_table = """Feature Name Feature Value Contribution to Prediction
                =========================================================
                a 2.18 +
                c 1.00 +
                e -1.80 --""".splitlines()

binary_table_shap = """Feature Name Feature Value Contribution to Prediction SHAP Value
                    ======================================================================
                     a 2.18 + 1.18
                     c 1.00 + 1.12
                     e -1.80 -- -2.60""".splitlines()

binary_dict = {
    "explanations": [
        {
            "feature_names": ["a", "c", "e"],
            "feature_values": [2.180, 1.0, -1.80],
            "qualitative_explanation": ["+", "+", "--"],
            "quantitative_explanation": [None, None, None],
            "drill_down": {},
            "class_name": "1",
            "expected_value": [0],
        }
    ]
}

binary_dict_shap = {
    "explanations": [
        {
            "feature_names": ["a", "c", "e"],
            "feature_values": [2.180, 1.0, -1.80],
            "qualitative_explanation": ["+", "+", "--"],
            "quantitative_explanation": [1.180, 1.120, -2.60],
            "drill_down": {},
            "class_name": "1",
            "expected_value": [0],
        }
    ]
}

multiclass = [
    {"a": [0], "b": [0], "c": [0], "d": [0.11], "e": [0.18], "f": [0], "foo": [-1]},
    {
        "a": [1.180],
        "b": [1.120],
        "c": [0.000],
        "d": [-2.560],
        "e": [-2.800],
        "f": [-2.900],
        "foo": [-1],
    },
    {
        "a": [0.680],
        "b": [0.000],
        "c": [0.000],
        "d": [-2.040],
        "e": [-1.840],
        "f": [-2.680],
        "foo": [-1],
    },
]

multiclass_normalized = [
    {
        "a": [0.0],
        "b": [0.0],
        "c": [0.0],
        "d": [0.07],
        "e": [0.08],
        "f": [0.0],
        "foo": [-1.0],
    },
    {
        "a": [0.102],
        "b": [0.097],
        "c": [0.0],
        "d": [-0.221],
        "e": [-0.242],
        "f": [-0.251],
        "foo": [-0.0865],
    },
    {
        "a": [0.08],
        "b": [0.0],
        "c": [0.0],
        "d": [-0.25],
        "e": [-0.22],
        "f": [-0.33],
        "foo": [-0.12],
    },
]
multiclass_pipeline_features = pd.DataFrame(
    {"a": 2.18, "b": 2.12, "c": 1.0, "d": -1.56, "e": -1.8, "f": -1.9, "foo": 30},
    index=[10],
)
multiclass_original_features = pd.DataFrame(
    {"a": 1.18, "b": 1.12, "c": 2.0, "d": -2.56, "e": -4.8, "f": -5.9, "foo": 40},
    index=[10],
)

multiclass_table = """Class: 0

                    Feature Name Feature Value Contribution to Prediction
                    =========================================================
                    e -1.80 +
                    d -1.56 +
                    foo 30.00 -----


                    Class: 1

                    Feature Name Feature Value Contribution to Prediction
                    =========================================================
                    d -1.56 --
                    e -1.80 --
                    f -1.90 --


                    Class: 2

                    Feature Name Feature Value Contribution to Prediction
                    =========================================================
                    e -1.80 --
                    d -1.56 --
                    f -1.90 --""".splitlines()

multiclass_table_shap = """Class: 0

                         Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         e -1.80 + 0.18
                         d -1.56 + 0.11
                         foo 30.00 ----- -1.00


                         Class: 1

                         Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         d -1.56 -- -2.56
                         e -1.80 -- -2.80
                         f -1.90 -- -2.90


                         Class: 2

                         Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         e -1.80 -- -1.84
                         d -1.56 -- -2.04
                         f -1.90 -- -2.68""".splitlines()

multiclass_dict = {
    "explanations": [
        {
            "feature_names": ["e", "d", "foo"],
            "feature_values": [-1.8, -1.56, 30],
            "qualitative_explanation": ["+", "+", "-----"],
            "quantitative_explanation": [None, None, None],
            "drill_down": {},
            "class_name": "0",
            "expected_value": 0,
        },
        {
            "feature_names": ["d", "e", "f"],
            "feature_values": [-1.56, -1.8, -1.9],
            "qualitative_explanation": ["--", "--", "--"],
            "quantitative_explanation": [None, None, None],
            "drill_down": {},
            "class_name": "1",
            "expected_value": 1,
        },
        {
            "feature_names": ["e", "d", "f"],
            "feature_values": [-1.8, -1.56, -1.9],
            "qualitative_explanation": ["--", "--", "--"],
            "quantitative_explanation": [None, None, None],
            "drill_down": {},
            "class_name": "2",
            "expected_value": 2,
        },
    ]
}

multiclass_dict_shap = {
    "explanations": [
        {
            "feature_names": ["e", "d", "foo"],
            "feature_values": [-1.8, -1.56, 30],
            "qualitative_explanation": ["+", "+", "-----"],
            "quantitative_explanation": [0.18, 0.11, -1],
            "drill_down": {},
            "class_name": "0",
            "expected_value": 0,
        },
        {
            "feature_names": ["d", "e", "f"],
            "feature_values": [-1.56, -1.8, -1.9],
            "qualitative_explanation": ["--", "--", "--"],
            "quantitative_explanation": [-2.56, -2.8, -2.9],
            "drill_down": {},
            "class_name": "1",
            "expected_value": 1,
        },
        {
            "feature_names": ["e", "d", "f"],
            "feature_values": [-1.8, -1.56, -1.9],
            "qualitative_explanation": ["--", "--", "--"],
            "quantitative_explanation": [-1.84, -2.04, -2.68],
            "drill_down": {},
            "class_name": "2",
            "expected_value": 2,
        },
    ]
}


@pytest.mark.parametrize(
    "values,normalized_values,pipeline_features,original_features,include_shap,expected_values, output_format,answer",
    [
        (
            regression,
            regression_normalized,
            regression_pipeline_features,
            regression_original_features,
            False,
            [0],
            "text",
            regression_table,
        ),
        (
            regression,
            regression_normalized,
            regression_pipeline_features,
            regression_original_features,
            True,
            [0],
            "text",
            regression_table_shap,
        ),
        (
            regression,
            regression_normalized,
            regression_pipeline_features,
            regression_original_features,
            False,
            [0],
            "dict",
            regression_dict,
        ),
        (
            regression,
            regression_normalized,
            regression_pipeline_features,
            regression_original_features,
            True,
            [0],
            "dict",
            regression_dict_shap,
        ),
        (
            binary,
            binary_normalized,
            binary_pipeline_features,
            binary_original_features,
            False,
            [0],
            "text",
            binary_table,
        ),
        (
            binary,
            binary_normalized,
            binary_pipeline_features,
            binary_original_features,
            True,
            [0],
            "text",
            binary_table_shap,
        ),
        (
            binary,
            binary_normalized,
            binary_pipeline_features,
            binary_original_features,
            False,
            [0],
            "dict",
            binary_dict,
        ),
        (
            binary,
            binary_normalized,
            binary_pipeline_features,
            binary_original_features,
            True,
            [0],
            "dict",
            binary_dict_shap,
        ),
        (
            multiclass,
            multiclass_normalized,
            multiclass_pipeline_features,
            multiclass_original_features,
            False,
            [0, 1, 2],
            "text",
            multiclass_table,
        ),
        (
            multiclass,
            multiclass_normalized,
            multiclass_pipeline_features,
            multiclass_original_features,
            True,
            [0, 1, 2],
            "text",
            multiclass_table_shap,
        ),
        (
            multiclass,
            multiclass_normalized,
            multiclass_pipeline_features,
            multiclass_original_features,
            False,
            [0, 1, 2],
            "dict",
            multiclass_dict,
        ),
        (
            multiclass,
            multiclass_normalized,
            multiclass_pipeline_features,
            multiclass_original_features,
            True,
            [0, 1, 2],
            "dict",
            multiclass_dict_shap,
        ),
    ],
)
def test_make_single_prediction_table(
    values,
    normalized_values,
    pipeline_features,
    original_features,
    include_shap,
    expected_values,
    output_format,
    answer,
):

    class_names = ["0", "1", "2"]

    if isinstance(values, list):
        if len(values) > 2:
            table_maker = _MultiClassSHAPTable(
                top_k=3,
                include_shap_values=include_shap,
                include_expected_value=False,
                class_names=class_names,
                provenance={},
            )
        else:
            table_maker = _BinarySHAPTable(
                class_names=class_names,
                top_k=3,
                include_shap_values=include_shap,
                include_expected_value=False,
                provenance={},
            )
    else:
        table_maker = _RegressionSHAPTable(
            top_k=3,
            include_shap_values=include_shap,
            include_expected_value=False,
            provenance={},
        )

    table_maker = (
        table_maker.make_text if output_format == "text" else table_maker.make_dict
    )

    table = table_maker(
        aggregated_shap_values=values,
        aggregated_normalized_values=normalized_values,
        shap_values=values,
        normalized_values=normalized_values,
        pipeline_features=pipeline_features,
        original_features=pipeline_features,
        expected_value=expected_values,
    )

    # Making sure the content is the same, regardless of formatting.
    if output_format == "text":
        for index, (row_table, row_answer) in enumerate(
            zip(table.splitlines(), answer)
        ):
            assert row_table.strip().split() == row_answer.strip().split()
    else:
        assert table == answer
