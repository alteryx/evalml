import copy
from itertools import product

import pytest

from evalml.pipelines.prediction_explanations._user_interface import (
    _make_rows,
    _make_single_prediction_table,
    _make_table
)

make_rows_test_cases = [({"a": [0.2], "b": [0.1]}, 3, [["a", "++"], ["b", "+"]]),
                        ({"a": [0.3], "b": [-0.9], "c": [0.5],
                          "d": [0.33], "e": [-0.67], "f": [-0.2],
                          "g": [0.71]}, 3,
                         [["g", "++++"], ["c", "+++"], ["d", "++"],
                          ["f", "--"], ["e", "----"], ["b", "-----"]]),
                        ({"a": [1.0], "f": [-1.0], "e": [0.0]}, 5,
                         [["a", "+++++"], ["e", "+"], ["f", "-----"]])]


@pytest.mark.parametrize("test_case,include_shap_values", product(make_rows_test_cases, [True, False]))
def test_make_rows_and_make_table(test_case, include_shap_values):
    values, top_k, answer = test_case

    if include_shap_values:
        new_answer = copy.deepcopy(answer)
        for row in new_answer:
            row.append(values[row[0]][0])
    else:
        new_answer = answer

    assert _make_rows(values, values, top_k, include_shap_values) == new_answer

    dtypes = ["t", "t"]
    alignment = ["c", "c"]
    if include_shap_values:
        dtypes.append("f")
        alignment.append("c")

    table = _make_table(dtypes, alignment, values, values, top_k, include_shap_values).splitlines()
    if include_shap_values:
        assert "SHAP Value" in table[0]
    # Subtracting two because a header and a line under the header are included in the table.
    assert len(table) - 2 == len(answer)


regression = {"a": [6.500], "b": [1.770], "c": [0.570],
              "d": [-0.090], "e": [-0.290], "f": [-1.210],
              "foo": [0.01], "bar": [-0.02]}

regression_normalized = {'a': [0.6214], 'b': [0.1692], 'bar': [-0.0019],
                         'c': [0.0544], 'd': [-0.0086], 'e': [-0.0277],
                         'f': [-0.1156], 'foo': [0.0001]}

regression_table = """Feature Name  Contribution to Prediction
                      =========================================
                      a ++++
                      b +
                      c +
                      d -
                      e -
                      f -""".splitlines()

regression_table_shap = """Feature Name Contribution to Prediction SHAP Value
                         ======================================================
                         a ++++ 6.500
                         b + 1.770
                         c + 0.570
                         d - -0.090
                         e - -0.290
                         f - -1.210""".splitlines()

binary = [{"a": [0], "b": [0], "c": [0],
           "d": [0], "e": [0], "f": [0], "foo": [-1]},
          {"a": [1.180], "b": [1.120], "c": [0.000],
           "d": [-2.560], "e": [-2.800], "f": [-2.900], "foo": [-1]}]

binary_normalized = [{'a': [0.0], 'b': [0.0], 'c': [0.0], 'd': [0.0], 'e': [0.0], 'f': [0.0], 'foo': [-1.0]},
                     {'a': [0.102], 'b': [0.097], 'c': [0.0], 'd': [-0.225],
                      'e': [-0.2422], 'f': [-0.251], 'foo': [-0.087]}]

binary_table = """Feature Name Contribution to Prediction
                =========================================
                a +
                b +
                c +
                d --
                e --
                f --""".splitlines()

binary_table_shap = """Feature Name Contribution to Prediction SHAP Value
                     ======================================================
                     a + 1.180
                     b + 1.120
                     c + 0.000
                     d -- -2.560
                     e -- -2.800
                     f -- -2.900""".splitlines()

multiclass = [{"a": [0], "b": [0], "c": [0],
               "d": [0], "e": [0], "f": [0], "foo": [-1]},
              {"a": [1.180], "b": [1.120], "c": [0.000], "d": [-2.560],
               "e": [-2.800], "f": [-2.900], "foo": [-1]},
              {"a": [0.680], "b": [0.000], "c": [0.000],
               "d": [-1.840], "e": [-2.040], "f": [-2.680], "foo": [-1]}]

multiclass_normalized = [{'a': [0.0], 'b': [0.0], 'c': [0.0], 'd': [0.0], 'e': [0.0], 'f': [0.0], 'foo': [-1.0]},
                         {'a': [0.102], 'b': [0.097], 'c': [0.0], 'd': [-0.221], 'e': [-0.242], 'f': [-0.251], 'foo': [-0.0865]},
                         {'a': [0.0825], 'b': [0.0], 'c': [0.0], 'd': [-0.223], 'e': [-0.247], 'f': [-0.325], 'foo': [-0.121]}]

multiclass_table = """Class: 0

                    Feature Name Contribution to Prediction
                    =========================================
                    f +
                    e +
                    d +
                    b +
                    a +
                    foo -----


                    Class: 1

                    Feature Name Contribution to Prediction
                    =========================================
                    a +
                    b +
                    c +
                    d --
                    e --
                    f --


                    Class: 2

                    Feature Name Contribution to Prediction
                    =========================================
                    a +
                    c +
                    b +
                    d --
                    e --
                    f --""".splitlines()

multiclass_table_shap = """Class: 0

                         Feature Name Contribution to Prediction SHAP Value
                         ======================================================
                         f + 0.000
                         e + 0.000
                         d + 0.000
                         b + 0.000
                         a + 0.000
                         foo ----- -1.000


                         Class: 1

                         Feature Name Contribution to Prediction SHAP Value
                         ======================================================
                         a + 1.180
                         b + 1.120
                         c + 0.000
                         d -- -2.560
                         e -- -2.800
                         f -- -2.900


                         Class: 2

                         Feature Name Contribution to Prediction SHAP Value
                         ======================================================
                         a + 0.680
                         c + 0.000
                         b + 0.000
                         d -- -1.840
                         e -- -2.040
                         f -- -2.680""".splitlines()


@pytest.mark.parametrize("values,normalized_values,include_shap,answer",
                         [(regression, regression_normalized, False, regression_table),
                          (regression, regression_normalized, True, regression_table_shap),
                          (binary, binary_normalized, False, binary_table),
                          (binary, binary_normalized, True, binary_table_shap),
                          (multiclass, multiclass_normalized, False, multiclass_table),
                          (multiclass, multiclass_normalized, True, multiclass_table_shap)])
def test_make_single_prediction_table(values, normalized_values, include_shap, answer):
    table = _make_single_prediction_table(values, normalized_values, include_shap_values=include_shap,
                                          class_names=["0", "1", "2"])

    # Making sure the content is the same, regardless of formatting.
    for row_table, row_answer in zip(table.splitlines(), answer):
        assert row_table.strip().split() == row_answer.strip().split()
