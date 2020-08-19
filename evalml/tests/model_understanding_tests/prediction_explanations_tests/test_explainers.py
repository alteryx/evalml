from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import PipelineScoreError
from evalml.model_understanding.prediction_explanations.explainers import (
    abs_error,
    cross_entropy,
    explain_prediction,
    explain_predictions,
    explain_predictions_best_worst
)
from evalml.problem_types import ProblemTypes


def compare_two_tables(table_1, table_2):
    assert len(table_1) == len(table_2)
    for row, row_answer in zip(table_1, table_2):
        assert row.strip().split() == row_answer.strip().split()


test_features = [5, [1], np.ones((1, 15)), pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).iloc[0],
                 pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pd.DataFrame()]


@pytest.mark.parametrize("test_features", test_features)
def test_explain_prediction_value_error(test_features):
    with pytest.raises(ValueError, match="features must be stored in a dataframe of one row."):
        explain_prediction(None, input_features=test_features, training_data=None)


explain_prediction_answer = """Feature Name Feature Value Contribution to Prediction
                               =========================================================
                                 d           40.00          ++++
                                 a           10.00          +++
                                 c           30.00          --
                                 b           20.00          ----""".splitlines()


explain_prediction_multiclass_answer = """Class: class_0

        Feature Name Feature Value Contribution to Prediction
       =========================================================
            a           10.00                +
            b           20.00                +
            c           30.00                -
            d           40.00                -


        Class: class_1

        Feature Name Feature Value Contribution to Prediction
       =========================================================
            a           10.00               +++
            b           20.00               ++
            c           30.00               -
            d           40.00               --


        Class: class_2

        Feature Name Feature Value Contribution to Prediction
        =========================================================
            a          10.00            +
            b          20.00            +
            c          30.00           ---
            d          40.00           ---
            """.splitlines()


@pytest.mark.parametrize("problem_type,shap_values,normalized_shap_values,answer",
                         [(ProblemTypes.REGRESSION,
                           {"a": [1], "b": [-2], "c": [-0.25], "d": [2]},
                           {"a": [0.5], "b": [-0.75], "c": [-0.25], "d": [0.75]},
                           explain_prediction_answer),
                          (ProblemTypes.BINARY,
                           [{}, {"a": [1], "b": [-2], "c": [-0.25], "d": [2]}],
                           [{}, {"a": [0.5], "b": [-0.75], "c": [-0.25], "d": [0.75]}],
                           explain_prediction_answer),
                          (ProblemTypes.MULTICLASS,
                           [{}, {}, {}],
                           [{"a": [0.1], "b": [0.09], "c": [-0.04], "d": [-0.06]},
                            {"a": [0.53], "b": [0.24], "c": [-0.15], "d": [-0.22]},
                            {"a": [0.03], "b": [0.02], "c": [-0.42], "d": [-0.47]}],
                           explain_prediction_multiclass_answer)
                          ])
@patch("evalml.model_understanding.prediction_explanations._user_interface._compute_shap_values")
@patch("evalml.model_understanding.prediction_explanations._user_interface._normalize_shap_values")
def test_explain_prediction(mock_normalize_shap_values,
                            mock_compute_shap_values,
                            problem_type, shap_values, normalized_shap_values, answer):
    mock_compute_shap_values.return_value = shap_values
    mock_normalize_shap_values.return_value = normalized_shap_values
    pipeline = MagicMock()
    pipeline.problem_type = problem_type
    pipeline._classes = ["class_0", "class_1", "class_2"]
    pipeline._transform.return_value = pd.DataFrame({"a": [10], "b": [20], "c": [30], "d": [40]})
    features = pd.DataFrame({"a": [1], "b": [2]})
    table = explain_prediction(pipeline, features, top_k=2).splitlines()

    compare_two_tables(table, answer)


def test_error_metrics():

    pd.testing.assert_series_equal(abs_error(pd.Series([1, 2, 3]), pd.Series([4, 1, 0])), pd.Series([3, 1, 3]))
    pd.testing.assert_series_equal(cross_entropy(pd.Series([1, 0]),
                                                 pd.DataFrame({"a": [0.1, 0.2], "b": [0.9, 0.8]})),
                                   pd.Series([-np.log(0.9), -np.log(0.2)]))


input_features_and_y_true = [([1], None, "^Input features must be a dataframe with more than 10 rows!"),
                             (pd.DataFrame({"a": [1]}), None, "^Input features must be a dataframe with more than 10 rows!"),
                             (pd.DataFrame({"a": range(15)}), [1], "^Parameter y_true must be a pd.Series."),
                             (pd.DataFrame({"a": range(15)}), pd.Series(range(12)), "^Parameters y_true and input_features must have the same number of data points.")
                             ]


@pytest.mark.parametrize("input_features,y_true,error_message", input_features_and_y_true)
def test_explain_predictions_best_worst_value_errors(input_features, y_true, error_message):
    with pytest.raises(ValueError, match=error_message):
        explain_predictions_best_worst(None, input_features, y_true)


def test_explain_predictions_raises_pipeline_score_error():
    with pytest.raises(PipelineScoreError, match="Division by zero!"):

        def raise_zero_division(input_features):
            raise ZeroDivisionError("Division by zero!")

        pipeline = MagicMock()
        pipeline.problem_type = ProblemTypes.BINARY
        pipeline.predict_proba.side_effect = raise_zero_division
        explain_predictions_best_worst(pipeline, pd.DataFrame({"a": range(15)}), pd.Series(range(15)))


@pytest.mark.parametrize("input_features", [1, [1], "foo", pd.DataFrame()])
def test_explain_predictions_value_errors(input_features):
    with pytest.raises(ValueError, match="Parameter input_features must be a non-empty dataframe."):
        explain_predictions(None, input_features)


regression_best_worst_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Value: 1
                Target Value: 2
                Absolute Difference: 1

                table goes here


            Worst 1 of 1

                Predicted Value: 2
                Target Value: 3
                Absolute Difference: 4

                table goes here


"""

no_best_worst_answer = """Test Pipeline Name

        Parameters go here

            1 of 2

                table goes here


            2 of 2

                table goes here


"""


binary_best_worst_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Probabilities: [benign: 0.05, malignant: 0.95]
                Predicted Value: malignant
                Target Value: malignant
                Cross Entropy: 0.2

                table goes here


            Worst 1 of 1

                Predicted Probabilities: [benign: 0.1, malignant: 0.9]
                Predicted Value: malignant
                Target Value: benign
                Cross Entropy: 0.78

                table goes here


"""

multiclass_table = """Class: setosa

        table goes here


        Class: versicolor

        table goes here


        Class: virginica

        table goes here"""

multiclass_best_worst_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Probabilities: [setosa: 0.8, versicolor: 0.1, virginica: 0.1]
                Predicted Value: setosa
                Target Value: setosa
                Cross Entropy: 0.15

                {multiclass_table}


            Worst 1 of 1

                Predicted Probabilities: [setosa: 0.2, versicolor: 0.75, virginica: 0.05]
                Predicted Value: versicolor
                Target Value: versicolor
                Cross Entropy: 0.34

                {multiclass_table}


""".format(multiclass_table=multiclass_table)

multiclass_no_best_worst_answer = """Test Pipeline Name

    Parameters go here

        1 of 2

            {multiclass_table}


        2 of 2

            {multiclass_table}


""".format(multiclass_table=multiclass_table)


@pytest.mark.parametrize("problem_type,answer,explain_predictions_answer",
                         [(ProblemTypes.REGRESSION, regression_best_worst_answer, no_best_worst_answer),
                          (ProblemTypes.BINARY, binary_best_worst_answer, no_best_worst_answer),
                          (ProblemTypes.MULTICLASS, multiclass_best_worst_answer, multiclass_no_best_worst_answer)])
@patch("evalml.model_understanding.prediction_explanations.explainers.DEFAULT_METRICS")
@patch("evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_shap_table")
def test_explain_predictions_best_worst_and_explain_predictions(mock_make_table, mock_default_metrics,
                                                                problem_type, answer, explain_predictions_answer):

    mock_make_table.return_value = "table goes here"
    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    input_features = pd.DataFrame({"a": [3, 4]})
    pipeline.problem_type = problem_type
    pipeline.name = "Test Pipeline Name"

    if problem_type == ProblemTypes.REGRESSION:
        abs_error_mock = MagicMock(__name__="abs_error")
        abs_error_mock.return_value = pd.Series([4, 1], dtype="int")
        mock_default_metrics.__getitem__.return_value = abs_error_mock
        pipeline.predict.return_value = pd.Series([2, 1])
        y_true = pd.Series([3, 2])
    elif problem_type == ProblemTypes.BINARY:
        pipeline._classes.return_value = ["benign", "malignant"]
        cross_entropy_mock = MagicMock(__name__="cross_entropy")
        mock_default_metrics.__getitem__.return_value = cross_entropy_mock
        cross_entropy_mock.return_value = pd.Series([0.2, 0.78])
        pipeline.predict_proba.return_value = pd.DataFrame({"benign": [0.05, 0.1], "malignant": [0.95, 0.9]})
        pipeline.predict.return_value = pd.Series(["malignant"] * 2)
        y_true = pd.Series(["malignant", "benign"])
    else:
        mock_make_table.return_value = multiclass_table
        pipeline._classes.return_value = ["setosa", "versicolor", "virginica"]
        cross_entropy_mock = MagicMock(__name__="cross_entropy")
        mock_default_metrics.__getitem__.return_value = cross_entropy_mock
        cross_entropy_mock.return_value = pd.Series([0.15, 0.34])
        pipeline.predict_proba.return_value = pd.DataFrame({"setosa": [0.8, 0.2], "versicolor": [0.1, 0.75],
                                                            "virginica": [0.1, 0.05]})
        pipeline.predict.return_value = ["setosa", "versicolor"]
        y_true = pd.Series(["setosa", "versicolor"])

    best_worst_report = explain_predictions_best_worst(pipeline, input_features, y_true=y_true,
                                                       num_to_explain=1)

    compare_two_tables(best_worst_report.splitlines(), answer.splitlines())

    report = explain_predictions(pipeline, input_features)
    compare_two_tables(report.splitlines(), explain_predictions_answer.splitlines())


@pytest.mark.parametrize("problem_type,answer",
                         [(ProblemTypes.REGRESSION, no_best_worst_answer),
                          (ProblemTypes.BINARY, no_best_worst_answer),
                          (ProblemTypes.MULTICLASS, multiclass_no_best_worst_answer)])
@patch("evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_shap_table")
def test_explain_predictions_custom_index(mock_make_table, problem_type, answer):

    mock_make_table.return_value = "table goes here"
    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    input_features = pd.DataFrame({"a": [3, 4]}, index=["first", "second"])
    pipeline.problem_type = problem_type
    pipeline.name = "Test Pipeline Name"

    if problem_type == ProblemTypes.REGRESSION:
        pipeline.predict.return_value = pd.Series([2, 1])
    elif problem_type == ProblemTypes.BINARY:
        pipeline._classes.return_value = ["benign", "malignant"]
        pipeline.predict.return_value = pd.Series(["malignant"] * 2)
        pipeline.predict_proba.return_value = pd.DataFrame({"benign": [0.05, 0.1], "malignant": [0.95, 0.9]})
    else:
        mock_make_table.return_value = multiclass_table
        pipeline._classes.return_value = ["setosa", "versicolor", "virginica"]
        pipeline.predict.return_value = pd.Series(["setosa", "versicolor"])
        pipeline.predict_proba.return_value = pd.DataFrame({"setosa": [0.8, 0.2], "versicolor": [0.1, 0.75],
                                                            "virginica": [0.1, 0.05]})

    report = explain_predictions(pipeline, input_features, training_data=input_features)

    compare_two_tables(report.splitlines(), answer.splitlines())


regression_custom_metric_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Value: 1
                Target Value: 2
                sum: 3

                table goes here


            Worst 1 of 1

                Predicted Value: 2
                Target Value: 3
                sum: 5

                table goes here


"""


@patch("evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_shap_table")
def test_explain_predictions_best_worst_custom_metric(mock_make_table):

    mock_make_table.return_value = "table goes here"
    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    input_features = pd.DataFrame({"a": [5, 6]})
    pipeline.problem_type = ProblemTypes.REGRESSION
    pipeline.name = "Test Pipeline Name"

    pipeline.predict.return_value = pd.Series([2, 1])
    y_true = pd.Series([3, 2])

    def sum(y_true, y_pred):
        return y_pred + y_true

    best_worst_report = explain_predictions_best_worst(pipeline, input_features, y_true=y_true,
                                                       num_to_explain=1, metric=sum)

    compare_two_tables(best_worst_report.splitlines(), regression_custom_metric_answer.splitlines())
