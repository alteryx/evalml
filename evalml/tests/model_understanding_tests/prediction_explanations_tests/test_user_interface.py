import copy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding.prediction_explanations._user_interface import (
    _DictBinarySHAPTable,
    _DictMultiClassSHAPTable,
    _DictRegressionSHAPTable,
    _make_rows,
    _make_text_table,
    _TextBinarySHAPTable,
    _TextMultiClassSHAPTable,
    _TextRegressionSHAPTable
)

make_rows_test_cases = [({"a": [0.2], "b": [0.1]}, 3, [["a", "1.20", "++"], ["b", "1.10", "+"]]),
                        ({"a": [0.3], "b": [-0.9], "c": [0.5],
                          "d": [0.33], "e": [-0.67], "f": [-0.2],
                          "g": [0.71]}, 3,
                         [["g", "1.71", "++++"], ["c", "1.50", "+++"], ["d", "1.33", "++"],
                          ["f", "0.80", "--"], ["e", "0.33", "----"], ["b", "0.10", "-----"]]),
                        ({"a": [1.0], "f": [-1.0], "e": [0.0]}, 5,
                         [["a", "2.00", "+++++"], ["e", "1.00", "+"], ["f", "0.00", "-----"]])]


@pytest.mark.parametrize("test_case,include_shap_values,include_string_features",
                         product(make_rows_test_cases, [True, False], [True, False]))
def test_make_rows_and_make_table(test_case, include_shap_values, include_string_features):
    values, top_k, answer = test_case

    pipeline_features = pd.DataFrame({name: value[0] + 1 for name, value in values.items()}, index=[5])

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

    assert _make_rows(values, values, pipeline_features, top_k, include_shap_values) == new_answer

    table = _make_text_table(values, values, pipeline_features, top_k, include_shap_values).splitlines()
    if include_shap_values:
        assert "SHAP Value" in table[0]
    # Subtracting two because a header and a line under the header are included in the table.
    assert len(table) - 2 == len(new_answer)


regression = {"a": [6.500], "b": [1.770], "c": [0.570],
              "d": [-0.090], "e": [-0.290], "f": [-1.210],
              "foo": [0.01], "bar": [-0.02]}

regression_normalized = {'a': [0.6214], 'b': [0.1692], 'bar': [-0.0019],
                         'c': [0.0544], 'd': [-0.0086], 'e': [-0.0277],
                         'f': [-0.1156], 'foo': [0.0001]}

regression_pipeline_features = pd.DataFrame({"a": 7.5, "b": 2.77, "c": 1.57, "d": 0.91, "e": 0.71, "f": -0.21,
                                             "foo": -20, "bar": -30}, index=[31])


regression_table = """Feature Name  Feature Value Contribution to Prediction
                      =========================================================
                      a 7.50 ++++
                      b 2.77 +
                      c 1.57 +
                      d 0.91 -
                      e 0.71 -
                      f -0.21 -""".splitlines()

regression_table_shap = """Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         a 7.50 ++++ 6.50
                         b 2.77 + 1.77
                         c 1.57 + 0.57
                         d 0.91 - -0.09
                         e 0.71 - -0.29
                         f -0.21 - -1.21""".splitlines()

regression_dict = {
    "explanation": [{
        "feature_names": ["a", "b", "c", "d", "e", "f"],
        "feature_values": [7.5, 2.77, 1.57, 0.91, 0.71, -0.21],
        "qualitative_explanation": ["++++", "+", "+", "-", "-", "-"],
        "quantitative_explanation": [None, None, None, None, None, None],
        "class_name": None
    }]
}

regression_dict_shap = {
    "explanation": [{
        "feature_names": ["a", "b", "c", "d", "e", "f"],
        "feature_values": [7.5, 2.77, 1.57, 0.91, 0.71, -0.21],
        "qualitative_explanation": ["++++", "+", "+", "-", "-", "-"],
        "quantitative_explanation": [6.50, 1.77, 0.57, -0.09, -0.29, -1.21],
        "class_name": None
    }]
}

binary = [{"a": [0], "b": [0], "c": [0],
           "d": [0], "e": [0], "f": [0], "foo": [-1]},
          {"a": [1.180], "b": [1.120], "c": [0.000],
           "d": [-2.560], "e": [-2.800], "f": [-2.900], "foo": [-1]}]

binary_normalized = [{'a': [0.0], 'b': [0.0], 'c': [0.0], 'd': [0.0], 'e': [0.0], 'f': [0.0], 'foo': [-1.0]},
                     {'a': [0.102], 'b': [0.097], 'c': [0.0], 'd': [-0.225],
                      'e': [-0.2422], 'f': [-0.251], 'foo': [-0.087]}]
binary_pipeline_features = pd.DataFrame({"a": 2.18, "b": 2.12, "c": 1.0, "d": -1.56, "e": -1.8, "f": -1.9,
                                         "foo": -20}, index=[23])

binary_table = """Feature Name Feature Value Contribution to Prediction
                =========================================================
                a 2.18 +
                b 2.12 +
                c 1.00 +
                d -1.56 --
                e -1.80 --
                f -1.90 --""".splitlines()

binary_table_shap = """Feature Name Feature Value Contribution to Prediction SHAP Value
                    ======================================================================
                     a 2.18 + 1.18
                     b 2.12 + 1.12
                     c 1.00 + 0.00
                     d -1.56 -- -2.56
                     e -1.80 -- -2.80
                     f -1.90 -- -2.90""".splitlines()

binary_dict = {
    "explanation": [{
        "feature_names": ["a", "b", "c", "d", "e", "f"],
        "feature_values": [2.180, 2.120, 1.000, -1.560, -1.800, -1.900],
        "qualitative_explanation": ["+", "+", "+", "--", "--", "--"],
        "quantitative_explanation": [None, None, None, None, None, None],
        "class_name": "1"
    }]
}

binary_dict_shap = {
    "explanation": [{
        "feature_names": ["a", "b", "c", "d", "e", "f"],
        "feature_values": [2.180, 2.120, 1.000, -1.560, -1.800, -1.900],
        "qualitative_explanation": ["+", "+", "+", "--", "--", "--"],
        "quantitative_explanation": [1.180, 1.120, 0.000, -2.560, -2.800, -2.900],
        "class_name": "1"
    }]
}

multiclass = [{"a": [0], "b": [0], "c": [0],
               "d": [0], "e": [0], "f": [0], "foo": [-1]},
              {"a": [1.180], "b": [1.120], "c": [0.000], "d": [-2.560],
               "e": [-2.800], "f": [-2.900], "foo": [-1]},
              {"a": [0.680], "b": [0.000], "c": [0.000],
               "d": [-1.840], "e": [-2.040], "f": [-2.680], "foo": [-1]}]

multiclass_normalized = [{'a': [0.0], 'b': [0.0], 'c': [0.0], 'd': [0.0], 'e': [0.0], 'f': [0.0], 'foo': [-1.0]},
                         {'a': [0.102], 'b': [0.097], 'c': [0.0], 'd': [-0.221], 'e': [-0.242], 'f': [-0.251], 'foo': [-0.0865]},
                         {'a': [0.0825], 'b': [0.0], 'c': [0.0], 'd': [-0.223], 'e': [-0.247], 'f': [-0.325], 'foo': [-0.121]}]

multiclass_pipeline_features = pd.DataFrame({"a": 2.18, "b": 2.12, "c": 1.0, "d": -1.56, "e": -1.8, "f": -1.9,
                                             "foo": 30}, index=[10])

multiclass_table = """Class: 0

                    Feature Name Feature Value Contribution to Prediction
                    =========================================================
                    f -1.90 +
                    e -1.80 +
                    d -1.56 +
                    b 2.12 +
                    a 2.18 +
                    foo 30.00 -----


                    Class: 1

                    Feature Name Feature Value Contribution to Prediction
                    =========================================================
                    a 2.18 +
                    b 2.12 +
                    c 1.00 +
                    d -1.56 --
                    e -1.80 --
                    f -1.90 --


                    Class: 2

                    Feature Name Feature Value Contribution to Prediction
                    =========================================================
                    a 2.18 +
                    c 1.00 +
                    b 2.12 +
                    d -1.56 --
                    e -1.80 --
                    f -1.90 --""".splitlines()

multiclass_table_shap = """Class: 0

                         Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         f -1.90 + 0.00
                         e -1.80 + 0.00
                         d -1.56 + 0.00
                         b 2.12 + 0.00
                         a 2.18 + 0.00
                         foo 30.00 ----- -1.00


                         Class: 1

                         Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         a 2.18 + 1.18
                         b 2.12 + 1.12
                         c 1.00 + 0.00
                         d -1.56 -- -2.56
                         e -1.80 -- -2.80
                         f -1.90 -- -2.90


                         Class: 2

                         Feature Name Feature Value Contribution to Prediction SHAP Value
                         ======================================================================
                         a 2.18 + 0.68
                         c 1.00 + 0.00
                         b 2.12 + 0.00
                         d -1.56 -- -1.84
                         e -1.80 -- -2.04
                         f -1.90 -- -2.68""".splitlines()

multiclass_dict = {
    "explanation": [
        {"feature_names": ["f", "e", "d", "b", "a", "foo"],
         "feature_values": [-1.9, -1.8, -1.56, 2.12, 2.18, 30],
         "qualitative_explanation": ["+", "+", "+", "+", "+", "-----"],
         "quantitative_explanation": [None, None, None, None, None, None],
         "class_name": "0"},
        {"feature_names": ["a", "b", "c", "d", "e", "f"],
         "feature_values": [2.18, 2.12, 1.0, -1.56, -1.8, -1.9],
         "qualitative_explanation": ["+", "+", "+", "--", "--", "--"],
         "quantitative_explanation": [None, None, None, None, None, None],
         "class_name": "1"},
        {"feature_names": ["a", "c", "b", "d", "e", "f"],
         "feature_values": [2.18, 1.0, 2.12, -1.56, -1.8, -1.9],
         "qualitative_explanation": ["+", "+", "+", "--", "--", "--"],
         "quantitative_explanation": [None, None, None, None, None, None],
         "class_name": "2"}
    ]
}

multiclass_dict_shap = {
    "explanation": [
        {"feature_names": ["f", "e", "d", "b", "a", "foo"],
         "feature_values": [-1.9, -1.8, -1.56, 2.12, 2.18, 30],
         "qualitative_explanation": ["+", "+", "+", "+", "+", "-----"],
         "quantitative_explanation": [0, 0, 0, 0, 0, -1],
         "class_name": "0"},
        {"feature_names": ["a", "b", "c", "d", "e", "f"],
         "feature_values": [2.18, 2.12, 1.0, -1.56, -1.8, -1.9],
         "qualitative_explanation": ["+", "+", "+", "--", "--", "--"],
         "quantitative_explanation": [1.180, 1.120, 0.000, -2.560, -2.800, -2.900],
         "class_name": "1"},
        {"feature_names": ["a", "c", "b", "d", "e", "f"],
         "feature_values": [2.18, 1.0, 2.12, -1.56, -1.8, -1.9],
         "qualitative_explanation": ["+", "+", "+", "--", "--", "--"],
         "quantitative_explanation": [0.680, 0.000, 0.000, -1.840, -2.040, -2.680],
         "class_name": "2"}
    ]
}


@pytest.mark.parametrize("values,normalized_values,pipeline_features,include_shap,output_format,answer",
                         [(regression, regression_normalized, regression_pipeline_features, False, "text", regression_table),
                          (regression, regression_normalized, regression_pipeline_features, True, "text", regression_table_shap),
                          (regression, regression_normalized, regression_pipeline_features, False, "dict", regression_dict),
                          (regression, regression_normalized, regression_pipeline_features, True, "dict", regression_dict_shap),
                          (binary, binary_normalized, binary_pipeline_features, False, "text", binary_table),
                          (binary, binary_normalized, binary_pipeline_features, True, "text", binary_table_shap),
                          (binary, binary_normalized, binary_pipeline_features, False, "dict", binary_dict),
                          (binary, binary_normalized, binary_pipeline_features, True, "dict", binary_dict_shap),
                          (multiclass, multiclass_normalized, multiclass_pipeline_features, False, "text", multiclass_table),
                          (multiclass, multiclass_normalized, multiclass_pipeline_features, True, "text", multiclass_table_shap),
                          (multiclass, multiclass_normalized, multiclass_pipeline_features, False, "dict", multiclass_dict),
                          (multiclass, multiclass_normalized, multiclass_pipeline_features, True, "dict", multiclass_dict_shap)
                          ])
def test_make_single_prediction_table(values, normalized_values, pipeline_features, include_shap, output_format, answer):

    class_names = ["0", "1", "2"]
    makers = {"text": (_TextMultiClassSHAPTable(class_names), _TextBinarySHAPTable(), _TextRegressionSHAPTable()),
              "dict": (_DictMultiClassSHAPTable(class_names), _DictBinarySHAPTable(class_names), _DictRegressionSHAPTable())}

    multiclass, binary, regression = makers[output_format]

    if isinstance(values, list):
        if len(values) > 2:
            table_maker = multiclass
        else:
            table_maker = binary
    else:
        table_maker = regression

    table = table_maker(values, normalized_values, pipeline_features, top_k=3, include_shap_values=include_shap)

    # Making sure the content is the same, regardless of formatting.
    if output_format == "text":
        for index, (row_table, row_answer) in enumerate(zip(table.splitlines(), answer)):
            assert row_table.strip().split() == row_answer.strip().split()
    else:
        assert table == answer
