import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.exceptions import PipelineNotYetFittedError
from evalml.objectives import FraudCost, get_objective
from evalml.pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.pipelines.utils import _get_pipeline_base_class
from evalml.preprocessing.utils import is_classification
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types, pad_with_nans


@pytest.mark.parametrize(
    "pipeline_class,estimator",
    [
        (TimeSeriesRegressionPipeline, "Linear Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier"),
    ],
)
@pytest.mark.parametrize(
    "components",
    [["One Hot Encoder"], ["Delayed Feature Transformer", "One Hot Encoder"]],
)
def test_time_series_pipeline_init(pipeline_class, estimator, components):
    component_graph = components + [estimator]
    if "Delayed Feature Transformer" not in components:
        pl = pipeline_class(
            component_graph=component_graph,
            parameters={"pipeline": {"date_index": None, "gap": 0, "max_delay": 5, "forecast_horizon": 3}},
        )
        assert "Delayed Feature Transformer" not in pl.parameters
        assert pl.parameters["pipeline"] == {
            "forecast_horizon": 3,
            "gap": 0,
            "max_delay": 5,
            "date_index": None,
        }
    else:
        parameters = {
            "Delayed Feature Transformer": {
                "date_index": None,
                "gap": 0,
                "max_delay": 5,
                "forecast_horizon": 3,
            },
            "pipeline": {"date_index": None, "gap": 0, "max_delay": 5, "forecast_horizon": 3},
        }
        pl = pipeline_class(component_graph=component_graph, parameters=parameters)
        assert pl.parameters["Delayed Feature Transformer"] == {
            "date_index": None,
            "gap": 0,
            "forecast_horizon": 3,
            "max_delay": 5,
            "delay_features": True,
            "delay_target": True,
        }
        assert pl.parameters["pipeline"] == {
            "gap": 0,
            "forecast_horizon": 3,
            "max_delay": 5,
            "date_index": None,
        }

    assert (
        pipeline_class(component_graph=component_graph, parameters=pl.parameters) == pl
    )

    with pytest.raises(
        ValueError,
        match="date_index, gap, and max_delay parameters cannot be omitted from the parameters dict",
    ):
        pipeline_class(component_graph, {})


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize(
    "forecast_horizon,gap,max_delay", [(1, 0, 1), (1, 1, 2), (2, 0, 2), (3, 1, 2), (1, 2, 2), (2, 7, 3), (3, 2, 4)]
)
@pytest.mark.parametrize(
    "pipeline_class,estimator_name",
    [
        (TimeSeriesRegressionPipeline, "Random Forest Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier"),
    ],
)
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch(
    "evalml.pipelines.TimeSeriesClassificationPipeline._encode_targets",
    side_effect=lambda y: y,
)
def test_fit_drop_nans_before_estimator(
    mock_encode_targets,
    mock_classifier_fit,
    mock_regressor_fit,
    pipeline_class,
    estimator_name,
    forecast_horizon,
    gap,
    max_delay,
    include_delayed_features,
    only_use_y,
    ts_data,
):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    if include_delayed_features:
        train_index = pd.date_range(f"2020-10-{1 + forecast_horizon + gap + max_delay}",
                                    "2020-10-31")
        expected_target = np.arange(1 + gap + max_delay + forecast_horizon, 32)
    else:
        train_index = pd.date_range(f"2020-10-01", f"2020-10-31")
        expected_target = np.arange(1, 32)

    pl = pipeline_class(
        component_graph=["Delayed Feature Transformer", estimator_name],
        parameters={
            "Delayed Feature Transformer": {
                "date_index": None,
                "gap": gap,
                "forecast_horizon": forecast_horizon,
                "max_delay": max_delay,
                "delay_features": include_delayed_features,
                "delay_target": include_delayed_features,
            },
            "pipeline": {"date_index": None, "gap": gap, "max_delay": max_delay,
                         "forecast_horizon": forecast_horizon},
        },
    )

    if only_use_y:
        pl.fit(None, y)
    else:
        pl.fit(X, y)

    if isinstance(pl, TimeSeriesRegressionPipeline):
        (
            df_passed_to_estimator,
            target_passed_to_estimator,
        ) = mock_regressor_fit.call_args[0]
    else:
        (
            df_passed_to_estimator,
            target_passed_to_estimator,
        ) = mock_classifier_fit.call_args[0]

    # NaNs introduced by shifting are dropped
    assert not df_passed_to_estimator.isna().any(axis=1).any()
    assert not target_passed_to_estimator.isna().any()

    # Check the estimator was trained on the expected dates
    pd.testing.assert_index_equal(df_passed_to_estimator.index, train_index)
    np.testing.assert_equal(target_passed_to_estimator.values, expected_target)


@pytest.mark.parametrize("only_use_y", [False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize(
    "forecast_horizon,gap,max_delay,date_index",
    [
        (1, 0, 1, None),
        (1, 0, 2, None),
        (1, 1, 1, None),
        (2, 1, 2, None),
        (2, 2, 2, None),
        (3, 7, 3, None),
        (2, 2, 4, None),
    ],
)
@pytest.mark.parametrize(
    "pipeline_class,estimator_name",
    [
        (TimeSeriesRegressionPipeline, "Random Forest Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier"),
    ],
)
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch(
    "evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets",
    side_effect=lambda y: y,
)
def test_predict_and_predict_in_sample(
    mock_decode_targets,
    mock_regressor_predict,
    mock_regressor_fit,
    mock_classifier_predict,
    mock_classifier_fit,
    pipeline_class,
    estimator_name,
    forecast_horizon,
    gap,
    max_delay,
    date_index,
    include_delayed_features,
    only_use_y,
    ts_data,
):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    pl = pipeline_class(
        component_graph=["Delayed Feature Transformer", estimator_name],
        parameters={
            "Delayed Feature Transformer": {
                "date_index": None,
                "gap": gap,
                "max_delay": max_delay,
                "forecast_horizon": forecast_horizon,
                "delay_features": include_delayed_features,
                "delay_target": include_delayed_features,
            },
            "pipeline": {"date_index": None, "gap": gap, "max_delay": max_delay, "forecast_horizon": forecast_horizon},
        },
    )

    def mock_predict(df, y=None):
        return pd.Series(range(200, 200 + df.shape[0]))

    if isinstance(pl, TimeSeriesRegressionPipeline):
        mock_regressor_predict.side_effect = mock_predict
    else:
        mock_classifier_predict.side_effect = mock_predict

    if only_use_y:
        pl.fit(None, y.iloc[:20])
        preds_in_sample = pl.predict_in_sample(X=None, y=y.iloc[20:],
                                               X_train=None, y_train=y.iloc[:20])
        preds = pl.predict(X.iloc[20:20+forecast_horizon], None, X_train=None, y_train=y.iloc[:20])
    else:
        pl.fit(X.iloc[:20], y.iloc[:20])
        preds_in_sample = pl.predict_in_sample(X.iloc[20:], y.iloc[20:], X.iloc[:20], y.iloc[:20])
        preds = pl.predict(X.iloc[20:20+forecast_horizon], None, X_train=X.iloc[:20], y_train=y.iloc[:20])

    assert len(preds) == forecast_horizon
    assert (preds.index == y.iloc[20:20 + forecast_horizon].index).all()
    assert len(preds_in_sample) == len(y.iloc[20:])
    assert (preds_in_sample.index == y.iloc[20:].index).all()


@pytest.mark.parametrize("only_use_y", [False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize(
    "forecast_horizon,gap,max_delay,date_index",
    [
        (1, 0, 1, None),
        (3, 1, 1, None),
        (2, 0, 2, None),
        (1, 1, 1, None),
        (1, 1, 2, None),
        (2, 2, 2, None),
        (3, 7, 3, None),
        (1, 2, 4, None),
    ],
)
@pytest.mark.parametrize(
    "pipeline_class,estimator_name",
    [
        (TimeSeriesRegressionPipeline, "Random Forest Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier"),
    ],
)
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.predict")
@patch(
    "evalml.pipelines.TimeSeriesClassificationPipeline._encode_targets",
    side_effect=lambda y: y,
)
@patch(
    "evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets",
    side_effect=lambda y: y,
)
@patch("evalml.pipelines.PipelineBase._score_all_objectives")
@patch("evalml.pipelines.TimeSeriesBinaryClassificationPipeline._score_all_objectives")
def test_score(
    mock_binary_score,
    mock_score,
    mock_encode_targets,
    mock_decode_targets,
    mock_classifier_predict,
    mock_classifier_fit,
    mock_regressor_predict,
    mock_regressor_fit,
    pipeline_class,
    estimator_name,
    forecast_horizon,
    gap,
    max_delay,
    date_index,
    include_delayed_features,
    only_use_y,
    ts_data,
):
    if pipeline_class == TimeSeriesBinaryClassificationPipeline:
        mock_score = mock_binary_score
    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data
    last_train_date = X.shape[0] - forecast_horizon - gap
    X_train, y_train = X.iloc[:last_train_date], y.iloc[:last_train_date]
    X_holdout, y_holdout = X.iloc[last_train_date:], y.iloc[last_train_date:]

    expected_target = np.arange(last_train_date + 1, 32)
    target_index = pd.date_range(f"2020-10-{last_train_date + 1}", f"2020-10-31")

    pl = pipeline_class(
        component_graph=["Delayed Feature Transformer", estimator_name],
        parameters={
            "Delayed Feature Transformer": {
                "date_index": None,
                "gap": gap,
                "max_delay": max_delay,
                "delay_features": include_delayed_features,
                "delay_target": include_delayed_features,
                "forecast_horizon": forecast_horizon
            },
            "pipeline": {"date_index": None, "gap": gap, "max_delay": max_delay, "forecast_horizon": forecast_horizon},
        },
    )

    def mock_predict(X, y=None):
        return pd.Series(range(200, 200 + X.shape[0]))

    if isinstance(pl, TimeSeriesRegressionPipeline):
        mock_regressor_predict.side_effect = mock_predict
    else:
        mock_classifier_predict.side_effect = mock_predict

    pl.fit(X_train, y_train)
    pl.score(X_holdout, y_holdout, objectives=["MCC Binary"], X_train=X_train, y_train=y_train)

    # Verify that NaNs are dropped before passed to objectives
    _, target, preds = mock_score.call_args[0]
    assert not target.isna().any()
    assert not preds.isna().any()

    # Target used for scoring matches expected dates
    pd.testing.assert_index_equal(target.index, target_index)
    np.testing.assert_equal(target.values, expected_target)


@pytest.mark.parametrize(
    "pipeline_class",
    [
        TimeSeriesBinaryClassificationPipeline,
        TimeSeriesMulticlassClassificationPipeline,
    ],
)
@patch("evalml.pipelines.LogisticRegressionClassifier.fit")
@patch("evalml.pipelines.LogisticRegressionClassifier.predict_proba")
@patch("evalml.pipelines.LogisticRegressionClassifier.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._score_all_objectives")
@patch("evalml.pipelines.ClassificationPipeline._decode_targets")
@patch("evalml.pipelines.ClassificationPipeline._encode_targets")
def test_classification_pipeline_encodes_targets(
    mock_encode,
    mock_decode,
    mock_score,
    mock_predict,
    mock_predict_proba,
    mock_fit,
    pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary
    y_series = pd.Series(y)
    df = pd.DataFrame({"negative": y_series, "positive": y_series})
    df.ww.init()
    mock_predict.side_effect = lambda data, y: ww.init_series(y_series.iloc[:len(data)])
    mock_predict_proba.side_effect = lambda data, y: df.ww.iloc[:len(data)]

    X = pd.DataFrame({"feature": range(len(y))})
    y_encoded = y_series.map(lambda label: "positive" if label == 1 else "negative").astype('category')
    X_train, y_encoded_train = X.iloc[:29], y_encoded.iloc[:29]
    X_holdout, y_encoded_holdout = X.iloc[29:], y_encoded.iloc[29:]

    mock_encode.side_effect = lambda series: series
    mock_decode.side_effect = lambda series: series

    pl = pipeline_class(
        component_graph=[
            "Delayed Feature Transformer",
            "Logistic Regression Classifier",
        ],
        parameters={
            "Delayed Feature Transformer": {
                "date_index": None,
                "gap": 0,
                "max_delay": 1,
                "forecast_horizon": 1
            },
            "pipeline": {"date_index": None, "gap": 0, "max_delay": 1, "forecast_horizon": 1},
        },
    )

    # Check fit encodes target
    pl.fit(X_train, y_encoded_train)
    _, target_passed_to_estimator = mock_fit.call_args[0]

    # Check that target is converted to ints. Use .iloc[1:] because the first feature row has NaNs
    assert_series_equal(target_passed_to_estimator, pl._encode_targets(y_encoded_train.iloc[2:]))

    # Check predict encodes target
    mock_encode.reset_mock()
    pl.predict(X_holdout.iloc[:1], None, X_train, y_encoded_train)
    assert mock_encode.call_count == 2

    mock_encode.reset_mock()
    pl.predict_in_sample(X_holdout, y_encoded_holdout, X_train, y_encoded_train, objective=None)
    assert mock_encode.call_count == 2

    # Check predict proba encodes target
    mock_encode.reset_mock()
    pl.predict_proba(X_holdout.iloc[:1], X_train, y_encoded_train)
    assert mock_encode.call_count == 2

    mock_encode.reset_mock()
    pl.predict_proba_in_sample(X_holdout, y_encoded_holdout, X_train, y_encoded_train)
    assert mock_encode.call_count == 2

    # Check score encodes target
    mock_encode.reset_mock()
    pl.score(X_holdout, y_encoded_holdout, X_train=X_train, y_train=y_encoded_train, objectives=["MCC Binary"])
    assert mock_encode.call_count == 4


@pytest.mark.parametrize(
    "pipeline_class,objectives",
    [
        (TimeSeriesBinaryClassificationPipeline, ["MCC Binary"]),
        (TimeSeriesBinaryClassificationPipeline, ["Log Loss Binary"]),
        (TimeSeriesBinaryClassificationPipeline, ["MCC Binary", "Log Loss Binary"]),
        (TimeSeriesMulticlassClassificationPipeline, ["MCC Multiclass"]),
        (TimeSeriesMulticlassClassificationPipeline, ["Log Loss Multiclass"]),
        (
            TimeSeriesMulticlassClassificationPipeline,
            ["MCC Multiclass", "Log Loss Multiclass"],
        ),
        (TimeSeriesRegressionPipeline, ["R2"]),
        (TimeSeriesRegressionPipeline, ["R2", "Mean Absolute Percentage Error"]),
    ],
)
@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_score_works(
    pipeline_class,
    objectives,
    data_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    make_data_type,
):

    preprocessing = ["Delayed Feature Transformer"]
    if pipeline_class == TimeSeriesRegressionPipeline:
        components = preprocessing + ["Random Forest Regressor"]
    else:
        components = preprocessing + ["Logistic Regression Classifier"]

    pl = pipeline_class(
        component_graph=components,
        parameters={
            "pipeline": {
                "date_index": None,
                "gap": 1,
                "max_delay": 3,
                "delay_features": False,
                "forecast_horizon": 10,
            },
            components[-1]: {"n_jobs": 1},
        },
    )
    if pl.problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X, y = X_y_binary
        y = pd.Series(y).map(lambda label: "good" if label == 1 else "bad")
        expected_unique_values = {"good", "bad"}
    elif pl.problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = X_y_multi
        label_map = {0: "good", 1: "bad", 2: "best"}
        y = pd.Series(y).map(lambda label: label_map[label])
        expected_unique_values = {"good", "bad", "best"}
    else:
        X, y = X_y_regression
        y = pd.Series(y)
        expected_unique_values = None

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_valid, y_valid = X.iloc[80:], y.iloc[80:]

    pl.fit(X_train, y_train)
    if expected_unique_values:
        # NaNs are expected because of padding due to max_delay
        assert set(pl.predict(X_valid.iloc[:10], objective=None, X_train=X_train, y_train=y_train).unique()) == expected_unique_values
    pl.score(X_valid, y_valid, objectives, X_train, y_train)


@patch("evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets")
@patch("evalml.objectives.BinaryClassificationObjective.decision_function")
@patch(
    "evalml.pipelines.components.Estimator.predict_proba",
    return_value=pd.DataFrame({0: [1.0]}),
)
@patch("evalml.pipelines.components.Estimator.predict", return_value=pd.Series([1.0]))
def test_binary_classification_predictions_thresholded_properly(
    mock_predict,
    mock_predict_proba,
    mock_obj_decision,
    mock_decode,
    X_y_binary,
    dummy_ts_binary_pipeline_class,
):
    mock_objs = [mock_decode, mock_predict]
    mock_decode.return_value = pd.Series([0, 1])
    X, y = X_y_binary
    binary_pipeline = dummy_ts_binary_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
        }
    )
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    binary_pipeline.fit(X, y)
    binary_pipeline.predict(X, y)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    binary_pipeline.predict(X, y, "precision")
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()

    mock_objs = [mock_decode, mock_predict_proba]
    # test custom threshold set but no objective passed
    mock_predict_proba.return_value = pd.DataFrame([[0.1, 0.2], [0.1, 0.2]])
    binary_pipeline.threshold = 0.6
    binary_pipeline._encoder.classes_ = [0, 1]
    binary_pipeline.predict(X, y)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_obj_decision.assert_not_called()
    mock_predict.assert_not_called()

    # test custom threshold set but no objective passed
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X, y)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_obj_decision.assert_not_called()
    mock_predict.assert_not_called()

    # test custom threshold set and objective passed
    binary_pipeline.threshold = 0.6
    mock_obj_decision.return_value = pd.Series([1.0])
    binary_pipeline.predict(X, y, "precision")
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_predict.assert_not_called()
    mock_obj_decision.assert_called()


def test_binary_predict_pipeline_objective_mismatch(X_y_binary, dummy_ts_binary_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = dummy_ts_binary_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {"gap": 0, "max_delay": 0, "date_index": None, "forecast_horizon": 2},
        }
    )
    binary_pipeline.fit(X[:30], y[:30])
    with pytest.raises(
        ValueError,
        match="Objective Precision Micro is not defined for time series binary classification.",
    ):
        binary_pipeline.predict(X[30:32], "precision micro", X[:30], y[:30])


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
def test_time_series_pipeline_not_fitted_error(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    time_series_binary_classification_pipeline_class,
    time_series_multiclass_classification_pipeline_class,
    time_series_regression_pipeline_class,
):
    if problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X, y = X_y_binary
        clf = time_series_binary_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None, "forecast_horizon": 20},
            }
        )

    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = X_y_multi
        clf = time_series_multiclass_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None, "forecast_horizon": 20},
            }
        )
    else:
        X, y = X_y_regression
        clf = time_series_regression_pipeline_class(
            parameters={
                "Random Forest Regressor": {"n_jobs": 1},
                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None, "forecast_horizon": 20},
            }
        )

    X_train, y_train = X[:80], y[:80]
    X_holdout, y_holdout = X[80:], y[80:]

    with pytest.raises(PipelineNotYetFittedError):
        clf.predict(X_holdout, None, X_train, y_train)
    with pytest.raises(PipelineNotYetFittedError):
        clf.feature_importance

    if is_classification(problem_type):
        with pytest.raises(PipelineNotYetFittedError):
            clf.predict_proba(X_holdout, None, X_train, y_train)

    clf.fit(X_train, y_train)

    if is_classification(problem_type):
        to_patch = "evalml.pipelines.TimeSeriesClassificationPipeline.predict_in_sample"
        if problem_type == ProblemTypes.TIME_SERIES_BINARY:
            to_patch = (
                "evalml.pipelines.TimeSeriesBinaryClassificationPipeline.predict_in_sample"
            )
        with patch(to_patch) as mock_predict:
            clf.predict(X_holdout, None, X_train, y_train)
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is None

            mock_predict.reset_mock()
            clf.predict(X_holdout, "Log Loss Binary", X_train, y_train)
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is not None

            mock_predict.reset_mock()
            clf.predict(X_holdout, objective="Log Loss Binary", X_train=X_train, y_train=y_train)
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is not None

            clf.predict_proba(X_holdout, X_train=X_train, y_train=y_train)
    else:
        clf.predict(X_holdout, None, X_train, y_train)
    clf.feature_importance


def test_ts_binary_pipeline_target_thresholding(
    make_data_type, time_series_binary_classification_pipeline_class, X_y_binary
):
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)
    objective = get_objective("F1", return_instance=True)

    binary_pipeline = time_series_binary_classification_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {"gap": 0, "max_delay": 0, "date_index": None, "forecast_horizon": 10},
        }
    )
    X_train, y_train = X.ww.iloc[:90], y.ww.iloc[:90]
    X_holdout, y_holdout = X.ww.iloc[90:], y.ww.iloc[90:]
    binary_pipeline.fit(X_train, y_train)
    assert binary_pipeline.threshold is None
    pred_proba = binary_pipeline.predict_proba(X_holdout, X_train, y_train).iloc[:, 1]
    binary_pipeline.optimize_threshold(X_holdout, y_holdout, pred_proba, objective)
    assert binary_pipeline.threshold is not None


@patch("evalml.objectives.FraudCost.decision_function")
def test_binary_predict_pipeline_use_objective(
    mock_decision_function, X_y_binary, time_series_binary_classification_pipeline_class
):
    X, y = X_y_binary
    binary_pipeline = time_series_binary_classification_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
        }
    )
    mock_decision_function.return_value = pd.Series([0] * 98)
    binary_pipeline.threshold = 0.7
    binary_pipeline.fit(X, y)
    fraud_cost = FraudCost(amount_col=0)
    binary_pipeline.score(X, y, ["precision", "auc", fraud_cost])
    mock_decision_function.assert_called()


def test_time_series_pipeline_with_detrender(ts_data):
    pytest.importorskip(
        "sktime",
        reason="Skipping polynomial detrending tests because sktime not installed",
    )
    X, y = ts_data
    component_graph = {
        "Polynomial Detrender": ["Polynomial Detrender", "X", "y"],
        "DelayedFeatures": ["Delayed Feature Transformer", "X", "y"],
        "Regressor": [
            "Linear Regressor",
            "DelayedFeatures.x",
            "Polynomial Detrender.y",
        ],
    }
    pipeline = TimeSeriesRegressionPipeline(
        component_graph=component_graph,
        parameters={
            "pipeline": {"gap": 1, "max_delay": 2, "date_index": None},
            "DelayedFeatures": {"max_delay": 2},
        },
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X, y)
    features = pipeline.compute_estimator_features(X, y)
    detrender = pipeline.component_graph.get_component("Polynomial Detrender")
    preds = pipeline.estimator.predict(features.iloc[2:])
    preds.index = y.index[2:]
    expected = detrender.inverse_transform(preds)
    expected = infer_feature_types(pad_with_nans(expected, 2))
    pd.testing.assert_series_equal(predictions, expected)


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
def test_ts_pipeline_predict_without_final_estimator(
    problem_type, make_data_type, X_y_binary
):
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)
    pipeline_class = _get_pipeline_base_class(problem_type)
    pipeline = pipeline_class(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        },
        parameters={
            "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
        },
    )
    pipeline.fit(X, y)
    if is_classification(problem_type):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot call predict_proba() on a component graph because the final component is not an Estimator."
            ),
        ):
            pipeline.predict_proba(X, y)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call predict() on a component graph because the final component is not an Estimator."
        ),
    ):
        pipeline.predict(X, y)


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
def test_ts_pipeline_transform(
    mock_ohe_transform,
    mock_imputer_transform,
    problem_type,
    make_data_type,
    X_y_binary,
):
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)
    mock_imputer_transform.return_value = X
    mock_ohe_transform.return_value = X
    pipeline_class = _get_pipeline_base_class(problem_type)
    pipeline = pipeline_class(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        },
        parameters={
            "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
        },
    )

    pipeline.fit(X, y)
    transformed_X = pipeline.transform(X, y)
    assert_frame_equal(X, transformed_X)


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
def test_ts_pipeline_transform_with_final_estimator(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    time_series_binary_classification_pipeline_class,
    time_series_multiclass_classification_pipeline_class,
    time_series_regression_pipeline_class,
):
    X, y = X_y_binary
    if problem_type == ProblemTypes.TIME_SERIES_BINARY:
        pipeline = time_series_binary_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
            }
        )

    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = X_y_multi
        pipeline = time_series_multiclass_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
            }
        )
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        X, y = X_y_regression
        pipeline = time_series_regression_pipeline_class(
            parameters={
                "Random Forest Regressor": {"n_jobs": 1},
                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None},
            }
        )

    pipeline.fit(X, y)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call transform() on a component graph because the final component is not a Transformer."
        ),
    ):
        pipeline.transform(X, y)
