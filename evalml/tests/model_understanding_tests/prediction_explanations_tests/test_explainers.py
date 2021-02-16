import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.exceptions import PipelineScoreError
from evalml.model_understanding.prediction_explanations.explainers import (
    abs_error,
    cross_entropy,
    explain_prediction,
    explain_predictions,
    explain_predictions_best_worst
)
from evalml.pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesRegressionPipeline
)
from evalml.problem_types import ProblemTypes, is_binary, is_regression


def compare_two_tables(table_1, table_2):
    assert len(table_1) == len(table_2)
    for row, row_answer in zip(table_1, table_2):
        assert row.strip().split() == row_answer.strip().split()


test_features = [[1], np.ones((15, 1)), pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).iloc[0],
                 pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pd.DataFrame()]


explain_prediction_answer = """Feature Name Feature Value Contribution to Prediction
                               =========================================================
                                 d           40.00          +++++
                                 b           20.00          -----""".splitlines()

explain_prediction_regression_dict_answer = {
    "explanations": [{
        "feature_names": ["d", "b"],
        "feature_values": [40, 20],
        "qualitative_explanation": ["+++++", "-----"],
        "quantitative_explanation": [None, None],
        "class_name": None
    }]
}

explain_predictions_regression_df_answer = pd.DataFrame({'feature_names': ['d', 'b'],
                                                         'feature_values': [40, 20],
                                                         'qualitative_explanation': ['+++++', '-----'],
                                                         "quantitative_explanation": [None, None]})

explain_prediction_binary_dict_answer = {
    "explanations": [{
        "feature_names": ["d", "b"],
        "feature_values": [40, 20],
        "qualitative_explanation": ["+++++", "-----"],
        "quantitative_explanation": [None, None],
        "class_name": "class_1"
    }]
}

explain_prediction_binary_df_answer = pd.DataFrame({
    "feature_names": ["d", "b"],
    "feature_values": [40, 20],
    "qualitative_explanation": ["+++++", "-----"],
    "quantitative_explanation": [None, None],
    "class_name": ["class_1", "class_1"]
})

explain_prediction_multiclass_answer = """Class: class_0

        Feature Name Feature Value Contribution to Prediction
       =========================================================
            a           10.00               +++++
            c           30.00                ---


        Class: class_1

        Feature Name Feature Value Contribution to Prediction
       =========================================================
            a           10.00               +++
            b           20.00               ++


        Class: class_2

        Feature Name Feature Value Contribution to Prediction
        =========================================================
            c          30.00           ---
            d          40.00           ---
            """.splitlines()

explain_prediction_multiclass_dict_answer = {
    "explanations": [
        {"feature_names": ["a", "c"],
         "feature_values": [10, 30],
         "qualitative_explanation": ["+++++", "---"],
         "quantitative_explanation": [None, None],
         "class_name": "class_0"},
        {"feature_names": ["a", "b"],
         "feature_values": [10, 20],
         "qualitative_explanation": ["+++", "++"],
         "quantitative_explanation": [None, None],
         "class_name": "class_1"},
        {"feature_names": ["c", "d"],
         "feature_values": [30, 40],
         "qualitative_explanation": ["---", "---"],
         "quantitative_explanation": [None, None],
         "class_name": "class_2"},
    ]
}

explain_prediction_multiclass_df_answer = pd.DataFrame({
    "feature_names": ["a", "c", "a", "b", "c", "d"],
    "feature_values": [10, 30, 10, 20, 30, 40],
    "qualitative_explanation": ["+++++", "---", "+++", "++", "---", "---"],
    "quantitative_explanation": [None, None, None, None, None, None],
    "class_name": ['class_0', 'class_0', 'class_1', 'class_1', 'class_2', 'class_2']
})


@pytest.mark.parametrize("problem_type, output_format, shap_values, normalized_shap_values, answer",
                         [(ProblemTypes.REGRESSION,
                           "text",
                           {"a": [1], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           {"a": [0.5], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           explain_prediction_answer),
                          (ProblemTypes.REGRESSION,
                           "dict",
                           {"a": [1], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           {"a": [0.5], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           explain_prediction_regression_dict_answer
                           ),
                          (ProblemTypes.REGRESSION,
                           "dataframe",
                           {"a": [1], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           {"a": [0.5], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           explain_predictions_regression_df_answer
                           ),
                          (ProblemTypes.TIME_SERIES_REGRESSION,
                           "text",
                           {"a": [1], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           {"a": [0.5], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           explain_prediction_answer),
                          (ProblemTypes.TIME_SERIES_REGRESSION,
                           "dict",
                           {"a": [1], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           {"a": [0.5], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           explain_prediction_regression_dict_answer
                           ),
                          (ProblemTypes.TIME_SERIES_REGRESSION,
                           "dataframe",
                           {"a": [1], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           {"a": [0.5], "b": [-2.1], "c": [-0.25], "d": [2.3]},
                           explain_predictions_regression_df_answer
                           ),
                          (ProblemTypes.BINARY,
                           "text",
                           [{}, {"a": [0.5], "b": [-0.89], "c": [0.33], "d": [0.89]}],
                           [{}, {"a": [0.5], "b": [-0.89], "c": [-0.25], "d": [0.89]}],
                           explain_prediction_answer),
                          (ProblemTypes.BINARY,
                           "dict",
                           [{}, {"a": [0.5], "b": [-0.89], "c": [0.33], "d": [0.89]}],
                           [{}, {"a": [0.5], "b": [-0.89], "c": [-0.25], "d": [0.89]}],
                           explain_prediction_binary_dict_answer),
                          (ProblemTypes.BINARY,
                           "dataframe",
                           [{}, {"a": [0.5], "b": [-0.89], "c": [0.33], "d": [0.89]}],
                           [{}, {"a": [0.5], "b": [-0.89], "c": [-0.25], "d": [0.89]}],
                           explain_prediction_binary_df_answer),
                          (ProblemTypes.MULTICLASS,
                           "text",
                           [{}, {}, {}],
                           [{"a": [1.1], "b": [0.09], "c": [-0.53], "d": [-0.06]},
                            {"a": [0.53], "b": [0.24], "c": [-0.15], "d": [-0.22]},
                            {"a": [0.03], "b": [0.02], "c": [-0.42], "d": [-0.47]}],
                           explain_prediction_multiclass_answer),
                          (ProblemTypes.MULTICLASS,
                           "dict",
                           [{}, {}, {}],
                           [{"a": [1.1], "b": [0.09], "c": [-0.53], "d": [-0.06]},
                            {"a": [0.53], "b": [0.24], "c": [-0.15], "d": [-0.22]},
                            {"a": [0.03], "b": [0.02], "c": [-0.42], "d": [-0.47]}],
                           explain_prediction_multiclass_dict_answer),
                          (ProblemTypes.MULTICLASS,
                           "dataframe",
                           [{}, {}, {}],
                           [{"a": [1.1], "b": [0.09], "c": [-0.53], "d": [-0.06]},
                            {"a": [0.53], "b": [0.24], "c": [-0.15], "d": [-0.22]},
                            {"a": [0.03], "b": [0.02], "c": [-0.42], "d": [-0.47]}],
                           explain_prediction_multiclass_df_answer)
                          ])
@pytest.mark.parametrize("input_type", ["pd", "ww"])
@patch("evalml.model_understanding.prediction_explanations._user_interface._compute_shap_values")
@patch("evalml.model_understanding.prediction_explanations._user_interface._normalize_shap_values")
def test_explain_prediction(mock_normalize_shap_values,
                            mock_compute_shap_values,
                            problem_type, output_format, shap_values, normalized_shap_values, answer,
                            input_type):
    mock_compute_shap_values.return_value = shap_values
    mock_normalize_shap_values.return_value = normalized_shap_values
    pipeline = MagicMock()
    pipeline.problem_type = problem_type
    pipeline.classes_ = ["class_0", "class_1", "class_2"]

    # By the time we call transform, we are looking at only one row of the input data.
    pipeline.compute_estimator_features.return_value = ww.DataTable(pd.DataFrame({"a": [10], "b": [20], "c": [30], "d": [40]}))
    features = pd.DataFrame({"a": [1], "b": [2]})
    if input_type == "ww":
        features = ww.DataTable(features)
    table = explain_prediction(pipeline, features, y=None, output_format=output_format, index_to_explain=0, top_k_features=2)
    if isinstance(table, str):
        compare_two_tables(table.splitlines(), answer)
    elif isinstance(table, pd.DataFrame):
        pd.testing.assert_frame_equal(table, answer)
    else:
        assert table == answer


def test_explain_prediction_errors():
    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_prediction(MagicMock(), pd.DataFrame({"a": [0, 1, 2, 3, 4]}), y=None, index_to_explain=5)

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_prediction(MagicMock(), pd.DataFrame({"a": [0, 1, 2, 3, 4]}), y=None, index_to_explain=-1)


def test_error_metrics():

    np.testing.assert_array_equal(abs_error(pd.Series([1, 2, 3]), pd.Series([4, 1, 0])), np.array([3, 1, 3]))
    np.testing.assert_allclose(cross_entropy(pd.Series([1, 0]),
                                             pd.DataFrame({"a": [0.1, 0.2], "b": [0.9, 0.8]})),
                               np.array([-np.log(0.9), -np.log(0.2)]))


input_features_and_y_true = [([[1]], pd.Series([1]), "^Input features must be a dataframe with more than 10 rows!"),
                             (pd.DataFrame({"a": [1]}), pd.Series([1]), "^Input features must be a dataframe with more than 10 rows!"),
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


def test_explain_predictions_value_errors():
    with pytest.raises(ValueError, match="Parameter input_features must be a non-empty dataframe."):
        explain_predictions(MagicMock(), pd.DataFrame(), y=None, indices_to_explain=[0])

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_predictions(MagicMock(), pd.DataFrame({"a": [0, 1, 2, 3, 4]}), y=None, indices_to_explain=[5])

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_predictions(MagicMock(), pd.DataFrame({"a": [0, 1, 2, 3, 4]}), y=None, indices_to_explain=[1, 5])

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_predictions(MagicMock(), pd.DataFrame({"a": [0, 1, 2, 3, 4]}), y=None, indices_to_explain=[-1])


def test_output_format_checked():
    input_features, y_true = pd.DataFrame(data=[range(15)]), pd.Series(range(15))
    with pytest.raises(ValueError, match="Parameter output_format must be either text, dict, or dataframe. Received bar"):
        explain_predictions(pipeline=MagicMock(), input_features=input_features, y=None, indices_to_explain=0, output_format="bar")
    with pytest.raises(ValueError, match="Parameter output_format must be either text, dict, or dataframe. Received xml"):
        explain_prediction(pipeline=MagicMock(), input_features=input_features, y=None, index_to_explain=0, output_format="xml")

    input_features, y_true = pd.DataFrame(data=range(15)), pd.Series(range(15))
    with pytest.raises(ValueError, match="Parameter output_format must be either text, dict, or dataframe. Received foo"):
        explain_predictions_best_worst(pipeline=MagicMock, input_features=input_features, y_true=y_true, output_format="foo")


regression_best_worst_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Value: 1
                Target Value: 2
                Absolute Difference: 1.0
                Index ID: {index_0}

                table goes here


            Worst 1 of 1

                Predicted Value: 2
                Target Value: 3
                Absolute Difference: 4.0
                Index ID: {index_1}

                table goes here


"""

regression_best_worst_answer_dict = {
    "explanations": [
        {"rank": {"prefix": "best", "index": 1},
         "predicted_values": {"probabilities": None, "predicted_value": 1, "target_value": 2,
                              "error_name": "Absolute Difference", "error_value": 1.},
         "explanations": ["explanation_dictionary_goes_here"]},
        {"rank": {"prefix": "worst", "index": 1},
         "predicted_values": {"probabilities": None, "predicted_value": 2, "target_value": 3,
                              "error_name": "Absolute Difference", "error_value": 4.},
         "explanations": ["explanation_dictionary_goes_here"]}
    ]
}

regression_best_worst_answer_df = pd.DataFrame({
    "feature_names": [0, 0],
    "feature_values": [0, 0],
    "qualitative_explanation": [0, 0],
    "quantitative_explanation": [0, 0],
    "rank": [1, 1],
    "predicted_value": [1, 2],
    "target_value": [2, 3],
    "error_name": ["Absolute Difference"] * 2,
    "error_value": [1., 4.],
    "prefix": ["best", "worst"],
})

no_best_worst_answer = """Test Pipeline Name

        Parameters go here

            1 of 2

                table goes here


            2 of 2

                table goes here


"""

no_best_worst_answer_dict = {
    "explanations": [
        {"explanations": ["explanation_dictionary_goes_here"]},
        {"explanations": ["explanation_dictionary_goes_here"]}
    ]
}

no_best_worst_answer_df = pd.DataFrame({
    "feature_names": [0, 0],
    "feature_values": [0, 0],
    "qualitative_explanation": [0, 0],
    "quantitative_explanation": [0, 0],
    "prediction_number": [0, 1]
})

binary_best_worst_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Probabilities: [benign: 0.05, malignant: 0.95]
                Predicted Value: malignant
                Target Value: malignant
                Cross Entropy: 0.2
                Index ID: {index_0}

                table goes here


            Worst 1 of 1

                Predicted Probabilities: [benign: 0.1, malignant: 0.9]
                Predicted Value: malignant
                Target Value: benign
                Cross Entropy: 0.78
                Index ID: {index_1}

                table goes here


"""

binary_best_worst_answer_dict = {
    "explanations": [
        {"rank": {"prefix": "best", "index": 1},
         "predicted_values": {"probabilities": {"benign": 0.05, "malignant": 0.95},
                              "predicted_value": "malignant", "target_value": "malignant",
                              "error_name": "Cross Entropy", "error_value": 0.2},
         "explanations": ["explanation_dictionary_goes_here"]},
        {"rank": {"prefix": "worst", "index": 1},
         "predicted_values": {"probabilities": {"benign": 0.1, "malignant": 0.9},
                              "predicted_value": "malignant", "target_value": "benign",
                              "error_name": "Cross Entropy", "error_value": 0.78},
         "explanations": ["explanation_dictionary_goes_here"]}
    ]
}

binary_best_worst_answer_df = pd.DataFrame({
    "feature_names": [0, 0],
    "feature_values": [0, 0],
    "qualitative_explanation": [0, 0],
    "quantitative_explanation": [0, 0],
    "rank": [1, 1],
    "prefix": ["best", "worst"],
    "label_benign_probability": [0.05, 0.1],
    "label_malignant_probability": [0.95, 0.9],
    "predicted_value": ["malignant", "malignant"],
    "target_value": ["malignant", "benign"],
    "error_name": ["Cross Entropy"] * 2,
    "error_value": [0.2, 0.78]
})

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
                Index ID: {{index_0}}

                {multiclass_table}


            Worst 1 of 1

                Predicted Probabilities: [setosa: 0.2, versicolor: 0.75, virginica: 0.05]
                Predicted Value: versicolor
                Target Value: versicolor
                Cross Entropy: 0.34
                Index ID: {{index_1}}

                {multiclass_table}


""".format(multiclass_table=multiclass_table)

multiclass_best_worst_answer_dict = {
    "explanations": [
        {"rank": {"prefix": "best", "index": 1},
         "predicted_values": {"probabilities": {"setosa": 0.8, "versicolor": 0.1, "virginica": 0.1},
                              "predicted_value": "setosa", "target_value": "setosa",
                              "error_name": "Cross Entropy", "error_value": 0.15},
         "explanations": ["explanation_dictionary_goes_here"]},
        {"rank": {"prefix": "worst", "index": 1},
         "predicted_values": {"probabilities": {"setosa": 0.2, "versicolor": 0.75, "virginica": 0.05},
                              "predicted_value": "versicolor", "target_value": "versicolor",
                              "error_name": "Cross Entropy", "error_value": 0.34},
         "explanations": ["explanation_dictionary_goes_here"]}
    ]
}

multiclass_best_worst_answer_df = pd.DataFrame({
    "feature_names": [0, 0],
    "feature_values": [0, 0],
    "qualitative_explanation": [0, 0],
    "quantitative_explanation": [0, 0],
    "rank": [1, 1],
    "prefix": ["best", "worst"],
    "label_setosa_probability": [0.8, 0.2],
    "label_versicolor_probability": [0.1, 0.75],
    "label_virginica_probability": [0.1, 0.05],
    "predicted_value": ["setosa", "versicolor"],
    "target_value": ["setosa", "versicolor"],
    "error_name": ["Cross Entropy"] * 2,
    "error_value": [0.15, 0.34]
})

multiclass_no_best_worst_answer = """Test Pipeline Name

    Parameters go here

        1 of 2

            {multiclass_table}


        2 of 2

            {multiclass_table}


""".format(multiclass_table=multiclass_table)


@pytest.mark.parametrize("problem_type,output_format,answer,explain_predictions_answer,custom_index",
                         [(ProblemTypes.REGRESSION, "text", regression_best_worst_answer, no_best_worst_answer, [0, 1]),
                          (ProblemTypes.REGRESSION, "dict", regression_best_worst_answer_dict, no_best_worst_answer_dict, [0, 1]),
                          (ProblemTypes.REGRESSION, "dataframe", regression_best_worst_answer_df, no_best_worst_answer_df, [0, 1]),
                          (ProblemTypes.REGRESSION, "text", regression_best_worst_answer, no_best_worst_answer, [4, 23]),
                          (ProblemTypes.REGRESSION, "dict", regression_best_worst_answer_dict, no_best_worst_answer_dict, [4, 10]),
                          (ProblemTypes.REGRESSION, "dataframe", regression_best_worst_answer_df, no_best_worst_answer_df, [4, 10]),
                          (ProblemTypes.REGRESSION, "text", regression_best_worst_answer, no_best_worst_answer, ["foo", "bar"]),
                          (ProblemTypes.REGRESSION, "dict", regression_best_worst_answer_dict, no_best_worst_answer_dict, ["foo", "bar"]),
                          (ProblemTypes.REGRESSION, "dataframe", regression_best_worst_answer_df, no_best_worst_answer_df, ["foo", "bar"]),
                          (ProblemTypes.BINARY, "text", binary_best_worst_answer, no_best_worst_answer, [0, 1]),
                          (ProblemTypes.BINARY, "dict", binary_best_worst_answer_dict, no_best_worst_answer_dict, [0, 1]),
                          (ProblemTypes.BINARY, "dataframe", binary_best_worst_answer_df, no_best_worst_answer_df, [0, 1]),
                          (ProblemTypes.BINARY, "text", binary_best_worst_answer, no_best_worst_answer, [7, 11]),
                          (ProblemTypes.BINARY, "dict", binary_best_worst_answer_dict, no_best_worst_answer_dict, [7, 11]),
                          (ProblemTypes.BINARY, "dataframe", binary_best_worst_answer_df, no_best_worst_answer_df, [7, 11]),
                          (ProblemTypes.BINARY, "text", binary_best_worst_answer, no_best_worst_answer, ["first", "second"]),
                          (ProblemTypes.BINARY, "dict", binary_best_worst_answer_dict, no_best_worst_answer_dict, ["first", "second"]),
                          (ProblemTypes.BINARY, "dataframe", binary_best_worst_answer_df, no_best_worst_answer_df, ["first", "second"]),
                          (ProblemTypes.MULTICLASS, "text", multiclass_best_worst_answer, multiclass_no_best_worst_answer, [0, 1]),
                          (ProblemTypes.MULTICLASS, "dict", multiclass_best_worst_answer_dict, no_best_worst_answer_dict, [0, 1]),
                          (ProblemTypes.MULTICLASS, "dataframe", multiclass_best_worst_answer_df, no_best_worst_answer_df, [0, 1]),
                          (ProblemTypes.MULTICLASS, "text", multiclass_best_worst_answer, multiclass_no_best_worst_answer, [19, 103]),
                          (ProblemTypes.MULTICLASS, "dict", multiclass_best_worst_answer_dict, no_best_worst_answer_dict, [17, 235]),
                          (ProblemTypes.MULTICLASS, "dataframe", multiclass_best_worst_answer_df, no_best_worst_answer_df, [17, 235]),
                          (ProblemTypes.MULTICLASS, "text", multiclass_best_worst_answer, multiclass_no_best_worst_answer, ["2020-10", "2020-11"]),
                          (ProblemTypes.MULTICLASS, "dict", multiclass_best_worst_answer_dict, no_best_worst_answer_dict, ["2020-15", "2020-15"]),
                          (ProblemTypes.MULTICLASS, "dataframe", multiclass_best_worst_answer_df, no_best_worst_answer_df, ["2020-15", "2020-15"]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "text", regression_best_worst_answer, no_best_worst_answer, [0, 1]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "dict", regression_best_worst_answer_dict, no_best_worst_answer_dict, [0, 1]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "dataframe", regression_best_worst_answer_df, no_best_worst_answer_df, [0, 1]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "text", regression_best_worst_answer, no_best_worst_answer, [4, 23]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "dict", regression_best_worst_answer_dict, no_best_worst_answer_dict, [4, 10]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "dataframe", regression_best_worst_answer_df, no_best_worst_answer_df, [4, 10]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "text", regression_best_worst_answer, no_best_worst_answer, ["foo", "bar"]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "dict", regression_best_worst_answer_dict, no_best_worst_answer_dict, ["foo", "bar"]),
                          (ProblemTypes.TIME_SERIES_REGRESSION, "dataframe", regression_best_worst_answer_df, no_best_worst_answer_df, ["foo", "bar"]),
                          (ProblemTypes.TIME_SERIES_BINARY, "text", binary_best_worst_answer, no_best_worst_answer, [0, 1]),
                          (ProblemTypes.TIME_SERIES_BINARY, "dict", binary_best_worst_answer_dict, no_best_worst_answer_dict, [0, 1]),
                          (ProblemTypes.TIME_SERIES_BINARY, "dataframe", binary_best_worst_answer_df, no_best_worst_answer_df, [0, 1]),
                          (ProblemTypes.TIME_SERIES_BINARY, "text", binary_best_worst_answer, no_best_worst_answer, [7, 11]),
                          (ProblemTypes.TIME_SERIES_BINARY, "dict", binary_best_worst_answer_dict, no_best_worst_answer_dict, [7, 11]),
                          (ProblemTypes.TIME_SERIES_BINARY, "dataframe", binary_best_worst_answer_df, no_best_worst_answer_df, [7, 11]),
                          (ProblemTypes.TIME_SERIES_BINARY, "text", binary_best_worst_answer, no_best_worst_answer, ["first", "second"]),
                          (ProblemTypes.TIME_SERIES_BINARY, "dict", binary_best_worst_answer_dict, no_best_worst_answer_dict, ["first", "second"]),
                          (ProblemTypes.TIME_SERIES_BINARY, "dataframe", binary_best_worst_answer_df, no_best_worst_answer_df, ["first", "second"]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "text", multiclass_best_worst_answer, multiclass_no_best_worst_answer, [0, 1]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "dict", multiclass_best_worst_answer_dict, no_best_worst_answer_dict, [0, 1]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "dataframe", multiclass_best_worst_answer_df, no_best_worst_answer_df, [0, 1]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "text", multiclass_best_worst_answer, multiclass_no_best_worst_answer, [19, 103]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "dict", multiclass_best_worst_answer_dict, no_best_worst_answer_dict, [17, 235]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "dataframe", multiclass_best_worst_answer_df, no_best_worst_answer_df, [17, 235]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "text", multiclass_best_worst_answer, multiclass_no_best_worst_answer, ["2020-10", "2020-11"]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "dict", multiclass_best_worst_answer_dict, no_best_worst_answer_dict, ["2020-15", "2020-15"]),
                          (ProblemTypes.TIME_SERIES_MULTICLASS, "dataframe", multiclass_best_worst_answer_df, no_best_worst_answer_df, ["2020-15", "2020-15"]),
                          ])
@patch("evalml.model_understanding.prediction_explanations.explainers.DEFAULT_METRICS")
@patch("evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_shap_table")
def test_explain_predictions_best_worst_and_explain_predictions(mock_make_table, mock_default_metrics,
                                                                problem_type, output_format, answer,
                                                                explain_predictions_answer, custom_index):
    if output_format == "text":
        mock_make_table.return_value = "table goes here"
    elif output_format == "dataframe":
        shap_table = pd.DataFrame({
            "feature_names": [0],
            "feature_values": [0],
            "qualitative_explanation": [0],
            "quantitative_explanation": [0],
        })
        # Use side effect so that we always get a new copy of the dataframe
        mock_make_table.side_effect = lambda *args, **kwargs: shap_table.copy()
    else:
        mock_make_table.return_value = {"explanations": ["explanation_dictionary_goes_here"]}

    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    input_features = pd.DataFrame({"a": [3, 4]}, index=custom_index)
    pipeline.problem_type = problem_type
    pipeline.name = "Test Pipeline Name"
    pipeline.compute_estimator_features.return_value = ww.DataTable(input_features)

    def _add_custom_index(answer, index_best, index_worst, output_format):

        if output_format == "text":
            answer = answer.format(index_0=index_best, index_1=index_worst)
        elif output_format == "dataframe":
            col_name = "prefix" if "prefix" in answer.columns else "rank"
            n_repeats = answer[col_name].value_counts().tolist()[0]
            answer['index_id'] = [index_best] * n_repeats + [index_worst] * n_repeats
        else:
            answer["explanations"][0]["predicted_values"]["index_id"] = index_best
            answer["explanations"][1]["predicted_values"]["index_id"] = index_worst
        return answer

    if is_regression(problem_type):
        abs_error_mock = MagicMock(__name__="abs_error")
        abs_error_mock.return_value = pd.Series([4., 1.], dtype="float64")
        mock_default_metrics.__getitem__.return_value = abs_error_mock
        pipeline.predict.return_value = ww.DataColumn(pd.Series([2, 1]))
        y_true = pd.Series([3, 2], index=custom_index)
        answer = _add_custom_index(answer, index_best=custom_index[1],
                                   index_worst=custom_index[0], output_format=output_format)
    elif is_binary(problem_type):
        pipeline.classes_.return_value = ["benign", "malignant"]
        cross_entropy_mock = MagicMock(__name__="cross_entropy")
        mock_default_metrics.__getitem__.return_value = cross_entropy_mock
        cross_entropy_mock.return_value = pd.Series([0.2, 0.78])
        pipeline.predict_proba.return_value = ww.DataTable(pd.DataFrame({"benign": [0.05, 0.1], "malignant": [0.95, 0.9]}))
        pipeline.predict.return_value = ww.DataColumn(pd.Series(["malignant"] * 2))
        y_true = pd.Series(["malignant", "benign"], index=custom_index)
        answer = _add_custom_index(answer, index_best=custom_index[0],
                                   index_worst=custom_index[1], output_format=output_format)
    else:
        # Multiclass text output is formatted slightly different so need to account for that
        if output_format == "text":
            mock_make_table.return_value = multiclass_table
        pipeline.classes_.return_value = ["setosa", "versicolor", "virginica"]
        cross_entropy_mock = MagicMock(__name__="cross_entropy")
        mock_default_metrics.__getitem__.return_value = cross_entropy_mock
        cross_entropy_mock.return_value = pd.Series([0.15, 0.34])
        pipeline.predict_proba.return_value = ww.DataTable(pd.DataFrame({"setosa": [0.8, 0.2], "versicolor": [0.1, 0.75],
                                                                         "virginica": [0.1, 0.05]}))
        pipeline.predict.return_value = ww.DataColumn(pd.Series(["setosa", "versicolor"]))
        y_true = pd.Series(["setosa", "versicolor"], index=custom_index)
        answer = _add_custom_index(answer, index_best=custom_index[0],
                                   index_worst=custom_index[1], output_format=output_format)

    report = explain_predictions(pipeline, input_features, y=y_true, indices_to_explain=[0, 1], output_format=output_format)
    if output_format == "text":
        compare_two_tables(report.splitlines(), explain_predictions_answer.splitlines())
    elif output_format == "dataframe":
        assert report.columns.tolist() == explain_predictions_answer.columns.tolist()
        pd.testing.assert_frame_equal(report, explain_predictions_answer[report.columns])
    else:
        assert report == explain_predictions_answer

    best_worst_report = explain_predictions_best_worst(pipeline, input_features, y_true=y_true,
                                                       num_to_explain=1, output_format=output_format)
    if output_format == "text":
        compare_two_tables(best_worst_report.splitlines(), answer.splitlines())
    elif output_format == "dataframe":
        # Check dataframes equal without caring about column order
        assert sorted(best_worst_report.columns.tolist()) == sorted(answer.columns.tolist())
        pd.testing.assert_frame_equal(best_worst_report, answer[best_worst_report.columns])
    else:
        assert best_worst_report == answer


regression_custom_metric_answer = """Test Pipeline Name

        Parameters go here

            Best 1 of 1

                Predicted Value: 1
                Target Value: 2
                sum: 3
                Index ID: 1

                table goes here


            Worst 1 of 1

                Predicted Value: 2
                Target Value: 3
                sum: 5
                Index ID: 0

                table goes here


"""

regression_custom_metric_answer_dict = {
    "explanations": [
        {"rank": {"prefix": "best", "index": 1},
         "predicted_values": {"probabilities": None, "predicted_value": 1, "target_value": 2,
                              "error_name": "sum", "error_value": 3,
                              "index_id": 1},
         "explanations": ["explanation_dictionary_goes_here"]},
        {"rank": {"prefix": "worst", "index": 1},
         "predicted_values": {"probabilities": None, "predicted_value": 2, "target_value": 3,
                              "error_name": "sum", "error_value": 5,
                              "index_id": 0},
         "explanations": ["explanation_dictionary_goes_here"]}
    ]
}


@pytest.mark.parametrize("output_format,answer",
                         [("text", regression_custom_metric_answer),
                          ("dict", regression_custom_metric_answer_dict)])
@patch("evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_shap_table")
def test_explain_predictions_best_worst_custom_metric(mock_make_table, output_format, answer):

    mock_make_table.return_value = "table goes here" if output_format == "text" else {"explanations": ["explanation_dictionary_goes_here"]}
    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    input_features = pd.DataFrame({"a": [5, 6]})
    pipeline.problem_type = ProblemTypes.REGRESSION
    pipeline.name = "Test Pipeline Name"
    pipeline.compute_estimator_features.return_value = ww.DataTable(input_features)

    pipeline.predict.return_value = ww.DataColumn(pd.Series([2, 1]))
    y_true = pd.Series([3, 2])

    def sum(y_true, y_pred):
        return y_pred + y_true

    best_worst_report = explain_predictions_best_worst(pipeline, input_features, y_true=y_true,
                                                       num_to_explain=1, metric=sum, output_format=output_format)

    if output_format == "text":
        compare_two_tables(best_worst_report.splitlines(), regression_custom_metric_answer.splitlines())
    else:
        assert best_worst_report == answer


def test_explain_predictions_time_series(ts_data):
    X, y = ts_data

    class TSPipeline(TimeSeriesRegressionPipeline):
        component_graph = ["Delayed Feature Transformer", "Random Forest Regressor"]
        name = "time series pipeline"

    tspipeline = TSPipeline({"pipeline": {"gap": 1, "max_delay": 2},
                             "Random Forest Regressor": {"n_jobs": 1}})

    tspipeline.fit(X, y)

    exp = explain_predictions(pipeline=tspipeline, input_features=X, y=y,
                              indices_to_explain=[5, 11], output_format="dict")

    # Check that the computed features to be explained aren't NaN.
    for exp_idx in range(len(exp["explanations"])):
        assert not np.isnan(np.array(exp["explanations"][exp_idx]["explanations"][0]["feature_values"])).any()

    with pytest.raises(ValueError, match="Requested index"):
        explain_predictions(pipeline=tspipeline, input_features=X, y=y,
                            indices_to_explain=[1, 11], output_format="text")


@pytest.mark.parametrize("pipeline_class, estimator", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"), (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier")])
def test_explain_predictions_best_worst_time_series(pipeline_class, estimator, ts_data):
    X, y = ts_data

    if is_binary(pipeline_class.problem_type):
        y = y % 2

    class TSPipeline(pipeline_class):
        component_graph = ["Delayed Feature Transformer", estimator]
        name = "time series pipeline"

    tspipeline = TSPipeline({"pipeline": {"gap": 1, "max_delay": 2}})

    tspipeline.fit(X, y)

    exp = explain_predictions_best_worst(pipeline=tspipeline, input_features=X, y_true=y,
                                         output_format="dict")

    # Check that the computed features to be explained aren't NaN.
    for exp_idx in range(len(exp["explanations"])):
        assert not np.isnan(np.array(exp["explanations"][exp_idx]["explanations"][0]["feature_values"])).any()


@pytest.mark.parametrize("problem_type", [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_json_serialization(problem_type, X_y_regression, linear_regression_pipeline_class,
                            X_y_binary, logistic_regression_binary_pipeline_class,
                            X_y_multi, logistic_regression_multiclass_pipeline_class):

    if problem_type == problem_type.REGRESSION:
        X, y = X_y_regression
        y = pd.Series(y)
        pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}})
    elif problem_type == problem_type.BINARY:
        X, y = X_y_binary
        y = pd.Series(y).astype("str")
        pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    else:
        X, y = X_y_multi
        y = pd.Series(y).astype("str")
        pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})

    pipeline.fit(X, y)

    best_worst = explain_predictions_best_worst(pipeline, pd.DataFrame(X), y,
                                                num_to_explain=1, output_format="dict")
    assert json.loads(json.dumps(best_worst)) == best_worst

    report = explain_predictions(pipeline, pd.DataFrame(X), y=y, output_format="dict", indices_to_explain=[0])
    assert json.loads(json.dumps(report)) == report
