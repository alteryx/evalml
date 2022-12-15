import json
from itertools import product
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.exceptions import PipelineScoreError
from evalml.model_understanding.prediction_explanations.explainers import (
    ExplainPredictionsStage,
    abs_error,
    cross_entropy,
    explain_predictions,
    explain_predictions_best_worst,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.pipelines.components.utils import _all_estimators
from evalml.problem_types import ProblemTypes, is_binary, is_multiclass


def compare_two_tables(table_1, table_2):
    assert len(table_1) == len(table_2)
    for row, row_answer in zip(table_1, table_2):
        assert row.strip().split() == row_answer.strip().split()


def test_error_metrics():

    np.testing.assert_array_equal(
        abs_error(pd.Series([1, 2, 3]), pd.Series([4, 1, 0])),
        np.array([3, 1, 3]),
    )
    np.testing.assert_allclose(
        cross_entropy(
            pd.Series([1, 0]),
            pd.DataFrame({"a": [0.1, 0.2], "b": [0.9, 0.8]}),
        ),
        np.array([-np.log(0.9), -np.log(0.2)]),
    )


input_features_and_y_true = [
    (
        [[1]],
        pd.Series([1]),
        "^Input features must be a dataframe with more than 10 rows!",
    ),
    (
        pd.DataFrame({"a": [1]}),
        pd.Series([1]),
        "^Input features must be a dataframe with more than 10 rows!",
    ),
    (
        pd.DataFrame({"a": range(15)}),
        pd.Series(range(12)),
        "^Parameters y_true and input_features must have the same number of data points.",
    ),
]


@pytest.mark.parametrize(
    "input_features,y_true,error_message",
    input_features_and_y_true,
)
def test_explain_predictions_best_worst_value_errors(
    input_features,
    y_true,
    error_message,
):
    with pytest.raises(ValueError, match=error_message):
        explain_predictions_best_worst(None, input_features, y_true)


def test_explain_predictions_raises_pipeline_score_error():
    with pytest.raises(PipelineScoreError, match="Division by zero!"):

        def raise_zero_division(input_features):
            raise ZeroDivisionError("Division by zero!")

        pipeline = MagicMock()
        pipeline.problem_type = ProblemTypes.BINARY
        pipeline.predict_proba.side_effect = raise_zero_division
        explain_predictions_best_worst(
            pipeline,
            pd.DataFrame({"a": range(15)}),
            pd.Series(range(15)),
        )


def test_explain_predictions_value_errors():
    with pytest.raises(
        ValueError,
        match="Parameter input_features must be a non-empty dataframe.",
    ):
        explain_predictions(MagicMock(), pd.DataFrame(), y=None, indices_to_explain=[0])

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_predictions(
            MagicMock(),
            pd.DataFrame({"a": [0, 1, 2, 3, 4]}),
            y=None,
            indices_to_explain=[5],
        )

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_predictions(
            MagicMock(),
            pd.DataFrame({"a": [0, 1, 2, 3, 4]}),
            y=None,
            indices_to_explain=[1, 5],
        )

    with pytest.raises(ValueError, match="Explained indices should be between"):
        explain_predictions(
            MagicMock(),
            pd.DataFrame({"a": [0, 1, 2, 3, 4]}),
            y=None,
            indices_to_explain=[-1],
        )


@pytest.mark.parametrize("training_target", [None, pd.Series([1, 2, 3])])
@pytest.mark.parametrize("training_data", [None, pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ],
)
def test_time_series_training_target_and_training_data_are_not_None(
    training_target,
    training_data,
    problem_type,
):
    mock_ts_pipeline = MagicMock(problem_type=problem_type)

    if training_data is not None and training_target is not None:
        pytest.xfail("No exception raised in this case")

    with pytest.raises(
        ValueError,
        match="training_target and training_data are not None",
    ):
        explain_predictions(
            mock_ts_pipeline,
            pd.DataFrame({"a": [0, 1, 2, 3, 4]}),
            y=pd.Series([1, 2, 3, 4, 5]),
            indices_to_explain=[2],
            training_data=training_data,
            training_target=training_target,
        )

    with pytest.raises(
        ValueError,
        match="training_target and training_data are not None",
    ):
        explain_predictions_best_worst(
            mock_ts_pipeline,
            pd.DataFrame({"a": [0, 1, 2, 3, 4]}),
            y_true=pd.Series([1, 2, 3, 4, 5]),
            num_to_explain=1,
            training_data=training_data,
            training_target=training_target,
        )


def test_output_format_checked():
    input_features, y_true = pd.DataFrame(data=[range(15)]), pd.Series(range(15))
    with pytest.raises(
        ValueError,
        match="Parameter output_format must be either text, dict, or dataframe. Received bar",
    ):
        explain_predictions(
            pipeline=MagicMock(),
            input_features=input_features,
            y=None,
            indices_to_explain=0,
            output_format="bar",
        )

    input_features, y_true = pd.DataFrame(data=range(15)), pd.Series(range(15))
    with pytest.raises(
        ValueError,
        match="Parameter output_format must be either text, dict, or dataframe. Received foo",
    ):
        explain_predictions_best_worst(
            pipeline=MagicMock(),
            input_features=input_features,
            y_true=y_true,
            output_format="foo",
        )


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
        {
            "rank": {"prefix": "best", "index": 1},
            "predicted_values": {
                "probabilities": None,
                "predicted_value": 1,
                "target_value": 2,
                "error_name": "Absolute Difference",
                "error_value": 1.0,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
        {
            "rank": {"prefix": "worst", "index": 1},
            "predicted_values": {
                "probabilities": None,
                "predicted_value": 2,
                "target_value": 3,
                "error_name": "Absolute Difference",
                "error_value": 4.0,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
    ],
}

regression_best_worst_answer_df = pd.DataFrame(
    {
        "feature_names": [0, 0],
        "feature_values": [0, 0],
        "qualitative_explanation": [0, 0],
        "quantitative_explanation": [0, 0],
        "rank": [1, 1],
        "predicted_value": [1, 2],
        "target_value": [2, 3],
        "error_name": ["Absolute Difference"] * 2,
        "error_value": [1.0, 4.0],
        "prefix": ["best", "worst"],
    },
)

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
        {"explanations": ["explanation_dictionary_goes_here"]},
    ],
}

no_best_worst_answer_df = pd.DataFrame(
    {
        "feature_names": [0, 0],
        "feature_values": [0, 0],
        "qualitative_explanation": [0, 0],
        "quantitative_explanation": [0, 0],
        "prediction_number": [0, 1],
    },
)

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
        {
            "rank": {"prefix": "best", "index": 1},
            "predicted_values": {
                "probabilities": {"benign": 0.05, "malignant": 0.95},
                "predicted_value": "malignant",
                "target_value": "malignant",
                "error_name": "Cross Entropy",
                "error_value": 0.2,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
        {
            "rank": {"prefix": "worst", "index": 1},
            "predicted_values": {
                "probabilities": {"benign": 0.1, "malignant": 0.9},
                "predicted_value": "malignant",
                "target_value": "benign",
                "error_name": "Cross Entropy",
                "error_value": 0.78,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
    ],
}

binary_best_worst_answer_df = pd.DataFrame(
    {
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
        "error_value": [0.2, 0.78],
    },
)

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


""".format(
    multiclass_table=multiclass_table,
)

multiclass_best_worst_answer_dict = {
    "explanations": [
        {
            "rank": {"prefix": "best", "index": 1},
            "predicted_values": {
                "probabilities": {"setosa": 0.8, "versicolor": 0.1, "virginica": 0.1},
                "predicted_value": "setosa",
                "target_value": "setosa",
                "error_name": "Cross Entropy",
                "error_value": 0.15,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
        {
            "rank": {"prefix": "worst", "index": 1},
            "predicted_values": {
                "probabilities": {"setosa": 0.2, "versicolor": 0.75, "virginica": 0.05},
                "predicted_value": "versicolor",
                "target_value": "versicolor",
                "error_name": "Cross Entropy",
                "error_value": 0.34,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
    ],
}

multiclass_best_worst_answer_df = pd.DataFrame(
    {
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
        "error_value": [0.15, 0.34],
    },
)

multiclass_no_best_worst_answer = """Test Pipeline Name

    Parameters go here

        1 of 2

            {multiclass_table}


        2 of 2

            {multiclass_table}


""".format(
    multiclass_table=multiclass_table,
)


def _add_custom_index(answer, index_best, index_worst, output_format):

    if output_format == "text":
        answer = answer.format(index_0=index_best, index_1=index_worst)
    elif output_format == "dataframe":
        col_name = "prefix" if "prefix" in answer.columns else "rank"
        n_repeats = answer[col_name].value_counts().tolist()[0]
        answer["index_id"] = [index_best] * n_repeats + [index_worst] * n_repeats
    else:
        answer["explanations"][0]["predicted_values"]["index_id"] = index_best
        answer["explanations"][1]["predicted_values"]["index_id"] = index_worst
    return answer


def _prep_pipeline_mock(problem_type, input_features):
    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    pipeline.problem_type = problem_type
    pipeline.name = "Test Pipeline Name"
    pipeline.compute_estimator_features.return_value = input_features

    return pipeline


def _compare_reports(report, predicted_report, output_format):
    if output_format == "text":
        compare_two_tables(report.splitlines(), predicted_report.splitlines())
    elif output_format == "dataframe":
        assert sorted(report.columns.tolist()) == sorted(
            predicted_report.columns.tolist(),
        )
        pd.testing.assert_frame_equal(report, predicted_report[report.columns])
    else:
        assert report == predicted_report


output_formats = ["text", "dict", "dataframe"]
algorithms = ["shap", "lime"]
regression_problem_types = [
    ProblemTypes.REGRESSION,
    ProblemTypes.TIME_SERIES_REGRESSION,
]
regression_custom_indices = [[0, 1], [4, 10], ["foo", "bar"]]


@pytest.mark.parametrize(
    "problem_type,output_format,custom_index,algorithm",
    product(
        regression_problem_types,
        output_formats,
        regression_custom_indices,
        algorithms,
    ),
)
@patch("evalml.model_understanding.prediction_explanations.explainers.DEFAULT_METRICS")
@patch(
    "evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_explanation_table",
)
def test_explain_predictions_best_worst_and_explain_predictions_regression(
    mock_make_table,
    mock_default_metrics,
    problem_type,
    output_format,
    custom_index,
    algorithm,
):
    if output_format == "text":
        mock_make_table.return_value = "table goes here"
        answer = regression_best_worst_answer
        explain_predictions_answer = no_best_worst_answer
    elif output_format == "dataframe":
        explanation_table = pd.DataFrame(
            {
                "feature_names": [0],
                "feature_values": [0],
                "qualitative_explanation": [0],
                "quantitative_explanation": [0],
            },
        )
        # Use side effect so that we always get a new copy of the dataframe
        mock_make_table.side_effect = lambda *args, **kwargs: explanation_table.copy()
        answer = regression_best_worst_answer_df
        explain_predictions_answer = no_best_worst_answer_df
    else:
        mock_make_table.return_value = {
            "explanations": ["explanation_dictionary_goes_here"],
        }
        answer = regression_best_worst_answer_dict
        explain_predictions_answer = no_best_worst_answer_dict

    input_features = pd.DataFrame({"a": [3, 4]}, index=custom_index)
    input_features.ww.init()
    pipeline = _prep_pipeline_mock(problem_type, input_features)
    pipeline.predict.return_value = ww.init_series(pd.Series([2, 1]))
    pipeline.predict_in_sample.return_value = ww.init_series(pd.Series([2, 1]))
    pipeline.transform_all_but_final.return_value = input_features

    abs_error_mock = MagicMock(__name__="abs_error")
    abs_error_mock.return_value = pd.Series([4.0, 1.0], dtype="float64")
    mock_default_metrics.__getitem__.return_value = abs_error_mock
    y_true = pd.Series([3, 2], index=custom_index)
    answer = _add_custom_index(
        answer,
        index_best=custom_index[1],
        index_worst=custom_index[0],
        output_format=output_format,
    )

    report = explain_predictions(
        pipeline,
        input_features,
        y=y_true,
        indices_to_explain=[0, 1],
        output_format=output_format,
        training_data=input_features,
        training_target=y_true,
        algorithm=algorithm,
    )
    _compare_reports(report, explain_predictions_answer, output_format)
    mock_make_table.assert_called_with(
        ANY,
        ANY,
        ANY,
        index_to_explain=ANY,
        top_k=ANY,
        include_explainer_values=ANY,
        output_format=output_format,
        algorithm=algorithm,
    )

    best_worst_report = explain_predictions_best_worst(
        pipeline,
        input_features,
        y_true=y_true,
        num_to_explain=1,
        output_format=output_format,
        training_data=input_features,
        training_target=y_true,
        algorithm=algorithm,
    )
    _compare_reports(best_worst_report, answer, output_format)
    mock_make_table.assert_called_with(
        ANY,
        ANY,
        ANY,
        index_to_explain=ANY,
        top_k=ANY,
        include_explainer_values=ANY,
        output_format=output_format,
        algorithm=algorithm,
    )


binary_problem_types = [ProblemTypes.BINARY, ProblemTypes.TIME_SERIES_BINARY]
binary_custom_indices = [[0, 1], [7, 11], ["first", "second"]]


@pytest.mark.parametrize(
    "problem_type,output_format,custom_index,algorithm",
    product(binary_problem_types, output_formats, binary_custom_indices, algorithms),
)
@patch("evalml.model_understanding.prediction_explanations.explainers.DEFAULT_METRICS")
@patch(
    "evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_explanation_table",
)
def test_explain_predictions_best_worst_and_explain_predictions_binary(
    mock_make_table,
    mock_default_metrics,
    problem_type,
    output_format,
    custom_index,
    algorithm,
):
    if output_format == "text":
        mock_make_table.return_value = "table goes here"
        answer = binary_best_worst_answer
        explain_predictions_answer = no_best_worst_answer
    elif output_format == "dataframe":
        explanation_table = pd.DataFrame(
            {
                "feature_names": [0],
                "feature_values": [0],
                "qualitative_explanation": [0],
                "quantitative_explanation": [0],
            },
        )
        # Use side effect so that we always get a new copy of the dataframe
        mock_make_table.side_effect = lambda *args, **kwargs: explanation_table.copy()
        answer = binary_best_worst_answer_df
        explain_predictions_answer = no_best_worst_answer_df
    else:
        mock_make_table.return_value = {
            "explanations": ["explanation_dictionary_goes_here"],
        }
        answer = binary_best_worst_answer_dict
        explain_predictions_answer = no_best_worst_answer_dict

    input_features = pd.DataFrame({"a": [3, 4]}, index=custom_index)
    input_features.ww.init()
    pipeline = _prep_pipeline_mock(problem_type, input_features)

    pipeline.classes_.return_value = ["benign", "malignant"]
    cross_entropy_mock = MagicMock(__name__="cross_entropy")
    mock_default_metrics.__getitem__.return_value = cross_entropy_mock
    cross_entropy_mock.return_value = pd.Series([0.2, 0.78])
    proba = pd.DataFrame({"benign": [0.05, 0.1], "malignant": [0.95, 0.9]})
    proba.ww.init()
    pipeline.predict_proba.return_value = proba
    pipeline.predict_proba_in_sample.return_value = proba
    pipeline.predict.return_value = ww.init_series(pd.Series(["malignant"] * 2))
    pipeline.predict_in_sample.return_value = ww.init_series(
        pd.Series(["malignant"] * 2),
    )
    pipeline.transform_all_but_final.return_value = input_features
    y_true = pd.Series(["malignant", "benign"], index=custom_index)
    answer = _add_custom_index(
        answer,
        index_best=custom_index[0],
        index_worst=custom_index[1],
        output_format=output_format,
    )

    report = explain_predictions(
        pipeline,
        input_features,
        y=y_true,
        indices_to_explain=[0, 1],
        output_format=output_format,
        training_data=input_features,
        training_target=y_true,
        algorithm=algorithm,
    )
    _compare_reports(report, explain_predictions_answer, output_format)
    mock_make_table.assert_called_with(
        ANY,
        ANY,
        ANY,
        index_to_explain=ANY,
        top_k=ANY,
        include_explainer_values=ANY,
        output_format=output_format,
        algorithm=algorithm,
    )

    best_worst_report = explain_predictions_best_worst(
        pipeline,
        input_features,
        y_true=y_true,
        num_to_explain=1,
        output_format=output_format,
        training_data=input_features,
        training_target=y_true,
        algorithm=algorithm,
    )
    _compare_reports(best_worst_report, answer, output_format)
    mock_make_table.assert_called_with(
        ANY,
        ANY,
        ANY,
        index_to_explain=ANY,
        top_k=ANY,
        include_explainer_values=ANY,
        output_format=output_format,
        algorithm=algorithm,
    )


multiclass_problem_types = [
    ProblemTypes.MULTICLASS,
    ProblemTypes.TIME_SERIES_MULTICLASS,
]
multiclass_custom_indices = [[0, 1], [17, 235], ["2020-15", "2020-15"]]


@pytest.mark.parametrize(
    "problem_type,output_format,custom_index,algorithm",
    product(
        multiclass_problem_types,
        output_formats,
        multiclass_custom_indices,
        algorithms,
    ),
)
@patch("evalml.model_understanding.prediction_explanations.explainers.DEFAULT_METRICS")
@patch(
    "evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_explanation_table",
)
def test_explain_predictions_best_worst_and_explain_predictions_multiclass(
    mock_make_table,
    mock_default_metrics,
    problem_type,
    output_format,
    custom_index,
    algorithm,
):
    if output_format == "text":
        mock_make_table.return_value = "table goes here"
        answer = multiclass_best_worst_answer
        explain_predictions_answer = multiclass_no_best_worst_answer
    elif output_format == "dataframe":
        explanation_table = pd.DataFrame(
            {
                "feature_names": [0],
                "feature_values": [0],
                "qualitative_explanation": [0],
                "quantitative_explanation": [0],
            },
        )
        # Use side effect so that we always get a new copy of the dataframe
        mock_make_table.side_effect = lambda *args, **kwargs: explanation_table.copy()
        answer = multiclass_best_worst_answer_df
        explain_predictions_answer = no_best_worst_answer_df
    else:
        mock_make_table.return_value = {
            "explanations": ["explanation_dictionary_goes_here"],
        }
        answer = multiclass_best_worst_answer_dict
        explain_predictions_answer = no_best_worst_answer_dict

    input_features = pd.DataFrame({"a": [3, 4]}, index=custom_index)
    input_features.ww.init()
    pipeline = _prep_pipeline_mock(problem_type, input_features)

    # Multiclass text output is formatted slightly different so need to account for that
    if output_format == "text":
        mock_make_table.return_value = multiclass_table
    pipeline.classes_.return_value = ["setosa", "versicolor", "virginica"]
    cross_entropy_mock = MagicMock(__name__="cross_entropy")
    mock_default_metrics.__getitem__.return_value = cross_entropy_mock
    cross_entropy_mock.return_value = pd.Series([0.15, 0.34])
    proba = pd.DataFrame(
        {"setosa": [0.8, 0.2], "versicolor": [0.1, 0.75], "virginica": [0.1, 0.05]},
    )
    proba.ww.init()
    pipeline.predict_proba.return_value = proba
    pipeline.predict_proba_in_sample.return_value = proba
    pipeline.predict.return_value = ww.init_series(pd.Series(["setosa", "versicolor"]))
    pipeline.predict_in_sample.return_value = ww.init_series(
        pd.Series(["setosa", "versicolor"]),
    )
    pipeline.transform_all_but_final.return_value = input_features
    y_true = pd.Series(["setosa", "versicolor"], index=custom_index)
    answer = _add_custom_index(
        answer,
        index_best=custom_index[0],
        index_worst=custom_index[1],
        output_format=output_format,
    )

    report = explain_predictions(
        pipeline,
        input_features,
        y=y_true,
        indices_to_explain=[0, 1],
        output_format=output_format,
        training_data=input_features,
        training_target=y_true,
        algorithm=algorithm,
    )
    _compare_reports(report, explain_predictions_answer, output_format)
    mock_make_table.assert_called_with(
        ANY,
        ANY,
        ANY,
        index_to_explain=ANY,
        top_k=ANY,
        include_explainer_values=ANY,
        output_format=output_format,
        algorithm=algorithm,
    )

    best_worst_report = explain_predictions_best_worst(
        pipeline,
        input_features,
        y_true=y_true,
        num_to_explain=1,
        output_format=output_format,
        training_data=input_features,
        training_target=y_true,
        algorithm=algorithm,
    )
    _compare_reports(best_worst_report, answer, output_format)
    mock_make_table.assert_called_with(
        ANY,
        ANY,
        ANY,
        index_to_explain=ANY,
        top_k=ANY,
        include_explainer_values=ANY,
        output_format=output_format,
        algorithm=algorithm,
    )


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
        {
            "rank": {"prefix": "best", "index": 1},
            "predicted_values": {
                "probabilities": None,
                "predicted_value": 1,
                "target_value": 2,
                "error_name": "sum",
                "error_value": 3,
                "index_id": 1,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
        {
            "rank": {"prefix": "worst", "index": 1},
            "predicted_values": {
                "probabilities": None,
                "predicted_value": 2,
                "target_value": 3,
                "error_name": "sum",
                "error_value": 5,
                "index_id": 0,
            },
            "explanations": ["explanation_dictionary_goes_here"],
        },
    ],
}


@pytest.mark.parametrize(
    "output_format,answer",
    [
        ("text", regression_custom_metric_answer),
        ("dict", regression_custom_metric_answer_dict),
    ],
)
@patch(
    "evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_explanation_table",
)
def test_explain_predictions_best_worst_custom_metric(
    mock_make_table,
    output_format,
    answer,
):

    mock_make_table.return_value = (
        "table goes here"
        if output_format == "text"
        else {"explanations": ["explanation_dictionary_goes_here"]}
    )
    pipeline = MagicMock()
    pipeline.parameters = "Parameters go here"
    input_features = pd.DataFrame({"a": [5, 6]})
    pipeline.problem_type = ProblemTypes.REGRESSION
    pipeline.name = "Test Pipeline Name"
    input_features.ww.init()
    pipeline.transform_all_but_final.return_value = input_features

    pipeline.predict.return_value = ww.init_series(pd.Series([2, 1]))
    y_true = pd.Series([3, 2])

    def sum(y_true, y_pred):
        return y_pred + y_true

    best_worst_report = explain_predictions_best_worst(
        pipeline,
        input_features,
        y_true=y_true,
        num_to_explain=1,
        metric=sum,
        output_format=output_format,
    )

    if output_format == "text":
        compare_two_tables(
            best_worst_report.splitlines(),
            regression_custom_metric_answer.splitlines(),
        )
    else:
        assert best_worst_report == answer


def test_explain_predictions_time_series(ts_data):
    X, _, y = ts_data()

    ts_pipeline = TimeSeriesRegressionPipeline(
        component_graph={
            "Time Series Featurizer": ["Time Series Featurizer", "X", "y"],
            "DateTime Featurizer": [
                "DateTime Featurizer",
                "Time Series Featurizer.x",
                "y",
            ],
            "Drop NaN Rows Transformer": [
                "Drop NaN Rows Transformer",
                "DateTime Featurizer.x",
                "y",
            ],
            "Estimator": [
                "Random Forest Regressor",
                "Drop NaN Rows Transformer.x",
                "Drop NaN Rows Transformer.y",
            ],
        },
        parameters={
            "pipeline": {
                "time_index": "date",
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
            },
            "Time Series Featurizer": {
                "time_index": "date",
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
            },
            "Random Forest Regressor": {"n_jobs": 1},
        },
    )
    X_train, y_train = X[:25], y[:25]
    X_validation, y_validation = X[25:], y[25:]
    ts_pipeline.fit(X_train, y_train)

    exp = explain_predictions(
        pipeline=ts_pipeline,
        input_features=X_validation,
        y=y_validation,
        indices_to_explain=[5, 11],
        output_format="dict",
        training_data=X_train,
        training_target=y_train,
    )
    # Check that the computed features to be explained aren't NaN.
    for exp_idx in range(len(exp["explanations"])):
        assert not pd.isnull(
            np.array(exp["explanations"][exp_idx]["explanations"][0]["feature_values"]),
        ).any()


@pytest.mark.parametrize("output_format", ["text", "dict", "dataframe"])
@pytest.mark.parametrize(
    "pipeline_class, estimator",
    [
        (TimeSeriesRegressionPipeline, "Random Forest Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
    ],
)
def test_explain_predictions_best_worst_time_series(
    output_format,
    pipeline_class,
    estimator,
    ts_data,
):
    X, _, y = ts_data(problem_type=pipeline_class.problem_type)

    ts_pipeline = pipeline_class(
        component_graph={
            "Time Series Featurizer": ["Time Series Featurizer", "X", "y"],
            "DateTime Featurizer": [
                "DateTime Featurizer",
                "Time Series Featurizer.x",
                "y",
            ],
            "Drop NaN Rows Transformer": [
                "Drop NaN Rows Transformer",
                "DateTime Featurizer.x",
                "y",
            ],
            "Estimator": [
                estimator,
                "Drop NaN Rows Transformer.x",
                "Drop NaN Rows Transformer.y",
            ],
        },
        parameters={
            "pipeline": {
                "time_index": "date",
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
            },
            "Time Series Featurizer": {
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
                "time_index": "date",
            },
        },
    )
    X_train, y_train = X[:25], y[:25]
    X_validation, y_validation = X[25:], y[25:]
    ts_pipeline.fit(X_train, y_train)

    exp = explain_predictions_best_worst(
        pipeline=ts_pipeline,
        input_features=X_validation,
        y_true=y_validation,
        output_format=output_format,
        training_data=X_train,
        training_target=y_train,
    )

    if output_format == "dict":
        # Check that the computed features to be explained aren't NaN.
        for exp_idx in range(len(exp["explanations"])):
            assert not pd.isnull(
                pd.Series(
                    exp["explanations"][exp_idx]["explanations"][0]["feature_values"],
                ),
            ).any()


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS],
)
def test_json_serialization(
    problem_type,
    X_y_regression,
    linear_regression_pipeline,
    X_y_binary,
    logistic_regression_binary_pipeline,
    X_y_multi,
    logistic_regression_multiclass_pipeline,
):

    if problem_type == problem_type.REGRESSION:
        X, y = X_y_regression
        y = pd.Series(y)
        pipeline = linear_regression_pipeline
    elif problem_type == problem_type.BINARY:
        X, y = X_y_binary
        y = pd.Series(y).astype("str")
        pipeline = logistic_regression_binary_pipeline
    else:
        X, y = X_y_multi
        y = pd.Series(y).astype("str")
        pipeline = logistic_regression_multiclass_pipeline

    pipeline.fit(X, y)

    best_worst = explain_predictions_best_worst(
        pipeline,
        pd.DataFrame(X),
        y,
        num_to_explain=1,
        output_format="dict",
    )
    assert json.loads(json.dumps(best_worst)) == best_worst

    report = explain_predictions(
        pipeline,
        pd.DataFrame(X),
        y=y,
        output_format="dict",
        indices_to_explain=[0],
    )
    assert json.loads(json.dumps(report)) == report


def transform_y_for_problem_type(problem_type, y):

    if problem_type == ProblemTypes.REGRESSION:
        y = y.astype("int")
    elif problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series(y).astype("str")
        y[:20] = "2"
    return y


EXPECTED_DATETIME_FEATURES = {
    "datetime_hour",
    "datetime_year",
    "datetime_month",
    "datetime_day_of_week",
}

EXPECTED_DATETIME_FEATURES_OHE = {
    "datetime_hour",
    "datetime_year",
    "datetime_month_3",
    "datetime_day_of_week_0",
    "datetime_day_of_week_1",
    "datetime_day_of_week_2",
    "datetime_day_of_week_3",
    "datetime_day_of_week_4",
    "datetime_day_of_week_5",
    "datetime_day_of_week_6",
    "datetime_month_0",
    "datetime_month_1",
    "datetime_month_2",
    "datetime_month_4",
    "datetime_month_5",
    "datetime_month_6",
    "datetime_month_7",
}

EXPECTED_CURRENCY_FEATURES = {
    "currency_XDR",
    "currency_HTG",
    "currency_PAB",
    "currency_CNY",
    "currency_TZS",
    "currency_LAK",
    "currency_NAD",
    "currency_IMP",
    "currency_QAR",
    "currency_EGP",
}

EXPECTED_PROVIDER_FEATURES_OHE = {
    "provider_JCB 16 digit",
    "provider_Discover",
    "provider_American Express",
    "provider_JCB 15 digit",
    "provider_Maestro",
    "provider_VISA 19 digit",
    "provider_VISA 13 digit",
    "provider_Mastercard",
    "provider_VISA 16 digit",
    "provider_Diners Club / Carte Blanche",
}

EXPECTED_PROVIDER_FEATURES_TEXT = {
    "DIVERSITY_SCORE(provider)",
    "LSA(provider)[0]",
    "LSA(provider)[1]",
    "MEAN_CHARACTERS_PER_WORD(provider)",
    "NUM_CHARACTERS(provider)",
    "NUM_WORDS(provider)",
    "POLARITY_SCORE(provider)",
}

pipeline_test_cases = [
    (BinaryClassificationPipeline, "Random Forest Classifier"),
    (RegressionPipeline, "Random Forest Regressor"),
    (MulticlassClassificationPipeline, "Random Forest Classifier"),
]


@pytest.mark.parametrize(
    "pipeline_class_and_estimator,algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_categories_aggregated_linear_pipeline(
    pipeline_class_and_estimator,
    algorithm,
    fraud_100,
):
    X, y = fraud_100
    pipeline_class, estimator = pipeline_class_and_estimator

    pipeline = pipeline_class(
        component_graph=[
            "Select Columns Transformer",
            "One Hot Encoder",
            "DateTime Featurizer",
            estimator,
        ],
        parameters={
            "Select Columns Transformer": {
                "columns": ["amount", "provider", "currency"],
            },
            estimator: {"n_jobs": 1},
        },
    )

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        algorithm=algorithm,
    )
    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == {"amount", "provider", "currency"}
        assert set(explanation["feature_values"]) == {"CUC", "Mastercard", 24900}
        assert explanation["drill_down"].keys() == {"currency", "provider"}
        assert (
            set(explanation["drill_down"]["currency"]["feature_names"])
            == EXPECTED_CURRENCY_FEATURES
        )
        assert (
            set(explanation["drill_down"]["provider"]["feature_names"])
            == EXPECTED_PROVIDER_FEATURES_OHE
        )


@pytest.mark.parametrize(
    "pipeline_class_and_estimator,algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_categories_aggregated_text(pipeline_class_and_estimator, algorithm, fraud_100):
    X, y = fraud_100
    pipeline_class, estimator = pipeline_class_and_estimator

    X.ww.set_types(
        logical_types={
            "provider": "NaturalLanguage",
        },
    )
    component_graph = [
        "Select Columns Transformer",
        "One Hot Encoder",
        "Natural Language Featurizer",
        "DateTime Featurizer",
        estimator,
    ]

    pipeline = pipeline_class(
        component_graph,
        parameters={
            "Select Columns Transformer": {
                "columns": ["amount", "provider", "currency", "datetime"],
            },
            estimator: {"n_jobs": 1},
        },
    )

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        top_k_features=4,
        output_format="dict",
        algorithm=algorithm,
    )
    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == {
            "amount",
            "provider",
            "currency",
            "datetime",
        }
        assert set(explanation["feature_values"]) == {
            "CUC",
            "Mastercard",
            24900,
            str(pd.Timestamp("2019-01-01 00:12:26")),
        }
        assert explanation["drill_down"].keys() == {"currency", "provider", "datetime"}
        assert (
            set(explanation["drill_down"]["currency"]["feature_names"])
            == EXPECTED_CURRENCY_FEATURES
        )
        assert (
            set(explanation["drill_down"]["provider"]["feature_names"])
            == EXPECTED_PROVIDER_FEATURES_TEXT
        )
        assert (
            set(explanation["drill_down"]["datetime"]["feature_names"])
            == EXPECTED_DATETIME_FEATURES
        )


@pytest.mark.parametrize(
    "pipeline_class_and_estimator,algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_categories_aggregated_date_ohe(
    pipeline_class_and_estimator,
    algorithm,
    fraud_100,
):
    X, y = fraud_100
    pipeline_class, estimator = pipeline_class_and_estimator

    pipeline = pipeline_class(
        component_graph=[
            "Select Columns Transformer",
            "DateTime Featurizer",
            "One Hot Encoder",
            estimator,
        ],
        parameters={
            "Select Columns Transformer": {
                "columns": ["datetime", "amount", "provider", "currency"],
            },
            "DateTime Featurizer": {"encode_as_categories": True},
            estimator: {"n_jobs": 1},
        },
    )
    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)
    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        top_k_features=7,
        algorithm=algorithm,
    )

    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == {
            "amount",
            "provider",
            "currency",
            "datetime",
        }
        assert set(explanation["feature_values"]) == {
            str(pd.Timestamp("2019-01-01 00:12:26")),
            "Mastercard",
            "CUC",
            24900,
        }
        assert explanation["drill_down"].keys() == {"currency", "provider", "datetime"}
        assert (
            set(explanation["drill_down"]["datetime"]["feature_names"])
            == EXPECTED_DATETIME_FEATURES_OHE
        )
        assert (
            set(explanation["drill_down"]["currency"]["feature_names"])
            == EXPECTED_CURRENCY_FEATURES
        )
        assert (
            set(explanation["drill_down"]["provider"]["feature_names"])
            == EXPECTED_PROVIDER_FEATURES_OHE
        )


@pytest.mark.parametrize(
    "pipeline_class_and_estimator,algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_categories_aggregated_pca_dag(
    pipeline_class_and_estimator,
    algorithm,
    fraud_100,
):
    X, y = fraud_100
    pipeline_class, estimator = pipeline_class_and_estimator

    component_graph = {
        "SelectNumeric": ["Select Columns Transformer", "X", "y"],
        "SelectCategorical": ["Select Columns Transformer", "X", "y"],
        "SelectDate": ["Select Columns Transformer", "X", "y"],
        "OHE": ["One Hot Encoder", "SelectCategorical.x", "y"],
        "DT": ["DateTime Featurizer", "SelectDate.x", "y"],
        "PCA": ["PCA Transformer", "SelectNumeric.x", "y"],
        "Estimator": [estimator, "PCA.x", "DT.x", "OHE.x", "y"],
    }
    parameters = {
        "SelectNumeric": {"columns": ["card_id", "store_id", "amount", "lat", "lng"]},
        "SelectCategorical": {"columns": ["currency", "provider"]},
        "SelectDate": {"columns": ["datetime"]},
        "PCA": {"n_components": 2},
        "Estimator": {"n_jobs": 1},
    }
    pipeline = pipeline_class(component_graph=component_graph, parameters=parameters)
    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)
    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        top_k_features=7,
        algorithm=algorithm,
    )

    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == {
            "component_0",
            "component_1",
            "provider",
            "currency",
            "datetime",
        }
        assert all(
            [
                f in explanation["feature_values"]
                for f in [
                    str(pd.Timestamp("2019-01-01 00:12:26")),
                    "Mastercard",
                    "CUC",
                ]
            ],
        )
        assert explanation["drill_down"].keys() == {"currency", "provider", "datetime"}
        assert (
            set(explanation["drill_down"]["currency"]["feature_names"])
            == EXPECTED_CURRENCY_FEATURES
        )
        assert (
            set(explanation["drill_down"]["provider"]["feature_names"])
            == EXPECTED_PROVIDER_FEATURES_OHE
        )
        assert (
            set(explanation["drill_down"]["datetime"]["feature_names"])
            == EXPECTED_DATETIME_FEATURES
        )


@pytest.mark.parametrize(
    "pipeline_class_and_estimator,algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_categories_aggregated_but_not_those_that_are_dropped(
    pipeline_class_and_estimator,
    algorithm,
    fraud_100,
):
    X, y = fraud_100
    pipeline_class, estimator = pipeline_class_and_estimator

    component_graph = [
        "Select Columns Transformer",
        "One Hot Encoder",
        "DateTime Featurizer",
        "Drop Columns Transformer",
        estimator,
    ]
    parameters = {
        "Select Columns Transformer": {
            "columns": ["amount", "provider", "currency", "datetime"],
        },
        "Drop Columns Transformer": {"columns": list(EXPECTED_DATETIME_FEATURES)},
        estimator: {"n_jobs": 1},
    }
    pipeline = pipeline_class(component_graph=component_graph, parameters=parameters)

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        algorithm=algorithm,
    )
    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == {"amount", "provider", "currency"}
        assert set(explanation["feature_values"]) == {"CUC", "Mastercard", 24900}
        assert explanation["drill_down"].keys() == {"currency", "provider"}
        assert (
            set(explanation["drill_down"]["currency"]["feature_names"])
            == EXPECTED_CURRENCY_FEATURES
        )
        assert (
            set(explanation["drill_down"]["provider"]["feature_names"])
            == EXPECTED_PROVIDER_FEATURES_OHE
        )


@pytest.mark.parametrize(
    "pipeline_class_and_estimator,algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_categories_aggregated_when_some_are_dropped(
    pipeline_class_and_estimator,
    algorithm,
    fraud_100,
):
    X, y = fraud_100
    pipeline_class, estimator = pipeline_class_and_estimator

    component_graph = [
        "Select Columns Transformer",
        "One Hot Encoder",
        "DateTime Featurizer",
        "Drop Columns Transformer",
        estimator,
    ]
    parameters = {
        "Select Columns Transformer": {
            "columns": ["amount", "provider", "currency", "datetime"],
        },
        "Drop Columns Transformer": {"columns": ["datetime_month", "datetime_hour"]},
        estimator: {"n_jobs": 1},
    }
    pipeline = pipeline_class(component_graph=component_graph, parameters=parameters)

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        top_k_features=4,
        algorithm=algorithm,
    )
    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == {
            "amount",
            "provider",
            "currency",
            "datetime",
        }
        assert set(explanation["feature_values"]) == {
            "CUC",
            "Mastercard",
            24900,
            str(pd.Timestamp("2019-01-01 00:12:26")),
        }
        assert explanation["drill_down"].keys() == {"currency", "provider", "datetime"}
        assert (
            set(explanation["drill_down"]["currency"]["feature_names"])
            == EXPECTED_CURRENCY_FEATURES
        )
        assert (
            set(explanation["drill_down"]["provider"]["feature_names"])
            == EXPECTED_PROVIDER_FEATURES_OHE
        )
        assert set(explanation["drill_down"]["datetime"]["feature_names"]) == {
            "datetime_year",
            "datetime_day_of_week",
        }


@pytest.mark.parametrize(
    "algorithm",
    ["shap", "lime"],
)
@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_explain_predictions_stacked_ensemble(
    algorithm,
    problem_type,
    fraud_100,
    X_y_multi,
    X_y_regression,
):
    if is_binary(problem_type):
        X, y = fraud_100
        pipeline = BinaryClassificationPipeline(
            {
                "DT": ["DateTime Featurizer", "X", "y"],
                "Imputer": ["Imputer", "DT.x", "y"],
                "One Hot Encoder": ["One Hot Encoder", "Imputer.x", "y"],
                "Drop Columns Transformer": [
                    "Drop Columns Transformer",
                    "One Hot Encoder.x",
                    "y",
                ],
                "Regression": [
                    "Logistic Regression Classifier",
                    "Drop Columns Transformer.x",
                    "y",
                ],
                "RF": ["Random Forest Classifier", "One Hot Encoder.x", "y"],
                "Stacked Ensembler": [
                    "Stacked Ensemble Classifier",
                    "Regression.x",
                    "RF.x",
                    "y",
                ],
            },
        )
        exp_feature_names = {"Col 1 RF.x", "Col 1 Regression.x"}
    elif is_multiclass(problem_type):
        X, y = X_y_multi
        pipeline = MulticlassClassificationPipeline(
            {
                "Imputer": ["Imputer", "X", "y"],
                "Regression": ["Logistic Regression Classifier", "Imputer.x", "y"],
                "RF": ["Random Forest Classifier", "X", "y"],
                "Stacked Ensembler": [
                    "Stacked Ensemble Classifier",
                    "Regression.x",
                    "RF.x",
                    "y",
                ],
            },
        )
        exp_feature_names = {
            "Col 0 RF.x",
            "Col 1 RF.x",
            "Col 2 RF.x",
            "Col 0 Regression.x",
            "Col 1 Regression.x",
            "Col 2 Regression.x",
        }
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            {
                "Imputer": ["Imputer", "X", "y"],
                "Regression": [
                    "Linear Regressor",
                    "Imputer.x",
                    "y",
                ],
                "RF": ["Random Forest Regressor", "X", "y"],
                "Stacked Ensembler": [
                    "Stacked Ensemble Regressor",
                    "Regression.x",
                    "RF.x",
                    "y",
                ],
            },
        )
        exp_feature_names = {"RF.x", "Regression.x"}
    pipeline.fit(X, y)

    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        top_k_features=10,
        algorithm=algorithm,
    )
    explanations_data = report["explanations"][0]["explanations"][0]
    assert set(explanations_data["feature_names"]) == exp_feature_names
    assert (
        explanations_data["quantitative_explanation"]
        == [None, None, None, None, None, None]
        if problem_type == ProblemTypes.MULTICLASS
        else [None, None]
    )

    report = explain_predictions_best_worst(
        pipeline,
        X,
        y,
        top_k_features=10,
        output_format="dict",
        algorithm=algorithm,
    )
    explanations_data = report["explanations"]
    for entry in explanations_data:
        assert set(entry["explanations"][0]["feature_names"]) == exp_feature_names


@pytest.mark.parametrize(
    "estimator,algorithm",
    product(
        [
            e
            for e in _all_estimators()
            if (
                "Classifier" in e.name
                and not any(
                    s in e.name
                    for s in ["Baseline", "Cat", "Elastic", "KN", "Ensemble", "Vowpal"]
                )
            )
        ],
        algorithms,
    ),
)
def test_explain_predictions_oversampler(estimator, algorithm, fraud_100):
    X, y = fraud_100
    pipeline = BinaryClassificationPipeline(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "One Hot Encoder": ["One Hot Encoder", "Imputer.x", "y"],
            "DateTime Featurizer": [
                "DateTime Featurizer",
                "One Hot Encoder.x",
                "y",
            ],
            "Oversampler": [
                "Oversampler",
                "DateTime Featurizer.x",
                "y",
            ],
            estimator: [estimator, "Oversampler.x", "Oversampler.y"],
        },
    )

    pipeline.fit(X, y)
    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dataframe",
        top_k_features=4,
        algorithm=algorithm,
    )
    assert report["feature_names"].isnull().sum() == 0
    assert report["feature_values"].isnull().sum() == 0


@pytest.mark.parametrize("problem_type", [ProblemTypes.MULTICLASS, ProblemTypes.BINARY])
def test_explain_predictions_class_name_matches_class_name_in_y(
    problem_type,
    fraud_100,
):
    X, y = fraud_100
    if problem_type == ProblemTypes.BINARY:
        pipeline_class = BinaryClassificationPipeline
    else:
        y = np.arange(X.shape[0]) % 3
        pipeline_class = MulticlassClassificationPipeline
    pipeline = pipeline_class(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "One Hot Encoder": ["One Hot Encoder", "Imputer.x", "y"],
            "DateTime Featurizer": [
                "DateTime Featurizer",
                "One Hot Encoder.x",
                "y",
            ],
            "Oversampler": [
                "Oversampler",
                "DateTime Featurizer.x",
                "y",
            ],
            "Random Forest Classifier": [
                "Random Forest Classifier",
                "Oversampler.x",
                "Oversampler.y",
            ],
        },
    )

    pipeline.fit(X, y)
    exp = explain_predictions_best_worst(
        pipeline,
        X,
        y,
        num_to_explain=1,
        output_format="dict",
        top_k_features=2,
        algorithm="shap",
    )
    if problem_type == ProblemTypes.BINARY:
        assert exp["explanations"][0]["explanations"][0]["class_name"]
        assert isinstance(exp["explanations"][0]["explanations"][0]["class_name"], bool)
    else:
        for i in range(3):
            assert (
                exp["explanations"][0]["explanations"][i]["class_name"]
                == pipeline.classes_[i]
            )


@patch(
    "evalml.model_understanding.prediction_explanations._user_interface._make_single_prediction_explanation_table",
)
def test_explain_predictions_best_worst_callback(mock_make_table):
    pipeline = MagicMock()
    pipeline.parameters = "Mock parameters"
    input_features = pd.DataFrame({"a": [5, 6]})
    pipeline.problem_type = ProblemTypes.REGRESSION
    pipeline.name = "Test Pipeline Name"
    input_features.ww.init()
    pipeline.transform_all_but_final.return_value = input_features
    pipeline.predict.return_value = ww.init_series(pd.Series([2, 1]))
    y_true = pd.Series([3, 2])

    class MockCallback:
        def __init__(self):
            self.progress_stages = []
            self.total_elapsed_time = 0

        def __call__(self, progress_stage, time_elapsed):
            self.progress_stages.append(progress_stage)
            self.total_elapsed_time = time_elapsed

    mock_callback = MockCallback()
    explain_predictions_best_worst(
        pipeline,
        input_features,
        y_true,
        num_to_explain=1,
        callback=mock_callback,
    )
    assert mock_callback.progress_stages == [e for e in ExplainPredictionsStage]
    assert mock_callback.total_elapsed_time > 0


@pytest.mark.parametrize("indices", [0, 1])
def test_explain_predictions_unknown(indices, X_y_binary):
    X, y = X_y_binary
    X.ww.set_types({0: "unknown"})
    pl = BinaryClassificationPipeline(["Random Forest Classifier"])
    pl.fit(X, y)

    report = explain_predictions(
        pl,
        X,
        y,
        indices_to_explain=[indices],
        output_format="dataframe",
        top_k_features=4,
    )
    assert report["feature_names"].isnull().sum() == 0
    assert report["feature_values"].isnull().sum() == 0
    if indices == 0:
        # make sure we only run this part once
        exp = explain_predictions_best_worst(
            pipeline=pl,
            input_features=X,
            y_true=y,
            output_format="dataframe",
        )
        assert exp["feature_names"].isnull().sum() == 0
        assert exp["feature_values"].isnull().sum() == 0


@pytest.mark.parametrize("algorithm", algorithms)
def test_explain_predictions_url_email(df_with_url_and_email, algorithm):
    X = df_with_url_and_email.ww.select(["url", "EmailAddress"])
    y = pd.Series([0, 1, 1, 0, 1])

    pl = BinaryClassificationPipeline(
        [
            "URL Featurizer",
            "Email Featurizer",
            "One Hot Encoder",
            "Random Forest Classifier",
        ],
    )
    pl.fit(X, y)
    explanations = explain_predictions_best_worst(
        pl,
        X,
        y,
        output_format="dict",
        num_to_explain=1,
        top_k_features=2,
        algorithm=algorithm,
    )
    assert (
        "email" in explanations["explanations"][0]["explanations"][0]["feature_names"]
    )
    assert "url" in explanations["explanations"][0]["explanations"][0]["feature_names"]
    assert (
        "email" in explanations["explanations"][1]["explanations"][0]["feature_names"]
    )
    assert "url" in explanations["explanations"][1]["explanations"][0]["feature_names"]
    assert (
        not pd.Series(
            explanations["explanations"][0]["explanations"][0][
                "qualitative_explanation"
            ],
        )
        .isnull()
        .any()
    )
    assert (
        not pd.Series(
            explanations["explanations"][1]["explanations"][0][
                "qualitative_explanation"
            ],
        )
        .isnull()
        .any()
    )


@pytest.mark.parametrize("algorithm", algorithms)
def test_explain_predictions_postalcodes(
    algorithm,
    fraud_100,
    logistic_regression_binary_pipeline,
):
    X, y = fraud_100
    X.ww.set_types(
        logical_types={
            "store_id": "PostalCode",
            "country": "CountryCode",
            "region": "SubRegionCode",
        },
    )

    pipeline = logistic_regression_binary_pipeline
    pipeline.fit(X, y)
    explanations = explain_predictions_best_worst(
        pipeline,
        X,
        y,
        output_format="dict",
        num_to_explain=1,
        top_k_features=X.shape[0],
        algorithm=algorithm,
    )
    assert (
        "store_id"
        in explanations["explanations"][0]["explanations"][0]["feature_names"]
    )
    assert (
        "country" in explanations["explanations"][0]["explanations"][0]["feature_names"]
    )
    assert (
        "region" in explanations["explanations"][0]["explanations"][0]["feature_names"]
    )
    assert (
        "store_id"
        in explanations["explanations"][1]["explanations"][0]["feature_names"]
    )
    assert (
        "country" in explanations["explanations"][1]["explanations"][0]["feature_names"]
    )
    assert (
        "region" in explanations["explanations"][1]["explanations"][0]["feature_names"]
    )
    assert (
        not pd.Series(
            explanations["explanations"][0]["explanations"][0][
                "qualitative_explanation"
            ],
        )
        .isnull()
        .any()
    )
    assert (
        not pd.Series(
            explanations["explanations"][1]["explanations"][0][
                "qualitative_explanation"
            ],
        )
        .isnull()
        .any()
    )

    explanations = explain_predictions(
        pipeline,
        X,
        y,
        output_format="dataframe",
        indices_to_explain=[0],
        top_k_features=X.shape[0],
        algorithm=algorithm,
    )
    assert "store_id" in list(explanations["feature_names"])
    assert "country" in list(explanations["feature_names"])
    assert "region" in list(explanations["feature_names"])
    assert explanations["feature_names"].isnull().sum() == 0
    assert explanations["feature_values"].isnull().sum() == 0


@pytest.mark.parametrize(
    "pipeline_class_and_estimator, algorithm",
    product(pipeline_test_cases, algorithms),
)
def test_explain_predictions_report_shows_original_value_if_possible(
    pipeline_class_and_estimator,
    algorithm,
    fraud_100,
):
    pipeline_class, estimator = pipeline_class_and_estimator
    X, y = fraud_100
    X.ww.set_types({"country": "NaturalLanguage"})
    component_graph = [
        "Imputer",
        "DateTime Featurizer",
        "Natural Language Featurizer",
        "One Hot Encoder",
        "Standard Scaler",
        estimator,
    ]
    parameters = {
        estimator: {"n_jobs": 1},
    }
    pipeline = pipeline_class(component_graph=component_graph, parameters=parameters)

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    report = explain_predictions(
        pipeline,
        X,
        y,
        indices_to_explain=[0],
        output_format="dict",
        top_k_features=20,
        algorithm=algorithm,
    )
    X_dt = X.copy()
    X_dt.ww.init()
    X_dt["datetime"] = X_dt["datetime"].astype(str)
    expected_feature_values = set(X_dt.ww.iloc[0, :].tolist())
    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == set(X.columns)
        assert set(explanation["feature_values"]) == expected_feature_values

    X_null = X.ww.copy()
    X_null.loc[0, "lat"] = None
    X_null.ww.init(schema=X.ww.schema)

    report = explain_predictions(
        pipeline,
        X_null,
        y,
        indices_to_explain=[0],
        output_format="dict",
        top_k_features=20,
        algorithm=algorithm,
    )
    for explanation in report["explanations"][0]["explanations"]:
        assert set(explanation["feature_names"]) == set(X.columns)
        for feature_name, feature_value in zip(
            explanation["feature_names"],
            explanation["feature_values"],
        ):
            if feature_name == "lat":
                assert np.isnan(feature_value)


@pytest.mark.parametrize("algorithm", algorithms)
def test_explain_predictions_best_worst_report_shows_original_value_if_possible(
    algorithm,
    fraud_100,
):
    X, y = fraud_100
    X.ww.set_types({"country": "NaturalLanguage"})
    component_graph = [
        "Imputer",
        "DateTime Featurizer",
        "Natural Language Featurizer",
        "One Hot Encoder",
        "Standard Scaler",
        "Random Forest Classifier",
    ]
    parameters = {
        "Random Forest Classifier": {"n_jobs": 1},
    }
    pipeline = BinaryClassificationPipeline(
        component_graph=component_graph,
        parameters=parameters,
    )

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)
    report = explain_predictions_best_worst(
        pipeline,
        X,
        y,
        num_to_explain=1,
        output_format="dict",
        top_k_features=20,
        algorithm=algorithm,
    )

    X_dt = X.copy()
    X_dt.ww.init()
    X_dt["datetime"] = X_dt["datetime"].astype(str)
    for index, explanation in enumerate(report["explanations"]):
        for exp in explanation["explanations"]:
            assert set(exp["feature_names"]) == set(X.columns)
            assert set(exp["feature_values"]) == set(
                X_dt.ww.iloc[explanation["predicted_values"]["index_id"], :],
            )

    X_null = X.ww.copy()
    X_null.loc[0:2, "lat"] = None
    X_null.ww.init(schema=X.ww.schema)

    report = explain_predictions_best_worst(
        pipeline,
        X_null.ww.iloc[:2],
        y.ww.iloc[:2],
        num_to_explain=1,
        output_format="dict",
        top_k_features=20,
        algorithm=algorithm,
    )
    for explanation in report["explanations"]:
        for exp in explanation["explanations"]:
            assert set(exp["feature_names"]) == set(X.columns)
            for feature_name, feature_value in zip(
                exp["feature_names"],
                exp["feature_values"],
            ):
                if feature_name == "lat":
                    assert np.isnan(feature_value)


@pytest.mark.parametrize("algorithm", algorithms)
def test_explain_predictions_best_worst_json(algorithm, fraud_100):
    pipeline = BinaryClassificationPipeline(
        [
            "Natural Language Featurizer",
            "DateTime Featurizer",
            "One Hot Encoder",
            "Logistic Regression Classifier",
        ],
    )
    X, y = fraud_100
    pipeline.fit(X, y)

    report = explain_predictions_best_worst(
        pipeline,
        X,
        y,
        algorithm=algorithm,
        output_format="dict",
    )
    json_output = json.dumps(report)
    assert isinstance(json_output, str)


def test_explain_predictions_invalid_algorithm():
    pipeline = MagicMock()
    input_features = pd.DataFrame({"a": [5, 6, 1, 2, 3, 4, 5, 6, 7, 4]})
    pipeline.problem_type = ProblemTypes.REGRESSION
    y = ww.init_series(pd.Series([2, 1, 1, 2, 3, 2, 1, 2, 1, 3]))

    with pytest.raises(ValueError, match="Unknown algorithm"):
        explain_predictions(pipeline, input_features, y, [0], algorithm="fAkE")

    with pytest.raises(ValueError, match="Unknown algorithm"):
        explain_predictions_best_worst(
            pipeline,
            input_features,
            y,
            top_k_features=1,
            algorithm="lIMe",
        )


def test_explain_predictions_lime_catboost(X_y_binary):
    pl = BinaryClassificationPipeline(
        {
            "Label Encoder": ["Label Encoder", "X", "y"],
            "CatBoost": ["CatBoost Classifier", "X", "Label Encoder.y"],
        },
    )
    X, y = X_y_binary
    error_msg = "CatBoost models are not supported by LIME at this time"

    with pytest.raises(ValueError, match=error_msg):
        explain_predictions(pl, X, y, [0], algorithm="lime")

    with pytest.raises(ValueError, match=error_msg):
        explain_predictions_best_worst(pl, X, y, algorithm="lime")
