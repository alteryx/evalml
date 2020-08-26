import copy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding.prediction_explanations._user_interface import (
    _make_rows,
    _make_table,
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

    table = _make_table(values, values, pipeline_features, top_k, include_shap_values).splitlines()
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


@pytest.mark.parametrize("values,normalized_values,pipeline_features,include_shap,answer",
                         [(regression, regression_normalized, regression_pipeline_features, False, regression_table),
                          (regression, regression_normalized, regression_pipeline_features, True, regression_table_shap),
                          (binary, binary_normalized, binary_pipeline_features, False, binary_table),
                          (binary, binary_normalized, binary_pipeline_features, True, binary_table_shap),
                          (multiclass, multiclass_normalized, multiclass_pipeline_features, False, multiclass_table),
                          (multiclass, multiclass_normalized, multiclass_pipeline_features, True, multiclass_table_shap)])
def test_make_single_prediction_table(values, normalized_values, pipeline_features, include_shap, answer):

    if isinstance(values, list):
        if len(values) > 2:
            table_maker = _TextMultiClassSHAPTable(class_names=["0", "1", "2"])
        else:
            table_maker = _TextBinarySHAPTable()
    else:
        table_maker = _TextRegressionSHAPTable()
    table = table_maker(values, normalized_values, pipeline_features, top_k=3, include_shap_values=include_shap)

    # Making sure the content is the same, regardless of formatting.
    for index, (row_table, row_answer) in enumerate(zip(table.splitlines(), answer)):
        assert row_table.strip().split() == row_answer.strip().split()
