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
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    TimeSeriesFeaturizer,
    Transformer,
)
from evalml.pipelines.utils import _get_pipeline_base_class
from evalml.preprocessing.utils import is_classification
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


@pytest.mark.parametrize(
    "pipeline_class,estimator",
    [
        (TimeSeriesRegressionPipeline, "Linear Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier"),
    ],
)
@pytest.mark.parametrize("gap", [0, 1, 5])
@pytest.mark.parametrize("forecast_horizon", [1, 5, 10])
@patch("evalml.pipelines.components.LinearRegressor.fit")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
def test_time_series_pipeline_validates_holdout_data(
    mock_fit_lr,
    mock_fit_linear,
    forecast_horizon,
    gap,
    pipeline_class,
    estimator,
    ts_data,
):
    pl = pipeline_class(
        component_graph=[estimator],
        parameters={
            "pipeline": {
                "date_index": "date",
                "gap": gap,
                "max_delay": 2,
                "forecast_horizon": forecast_horizon,
            }
        },
    )
    X, y = ts_data
    TRAIN_LENGTH = 15
    X_train, y_train = X.iloc[:TRAIN_LENGTH], y.iloc[:TRAIN_LENGTH]
    X = X.iloc[TRAIN_LENGTH + gap : TRAIN_LENGTH + gap + forecast_horizon + 2]

    pl.fit(X_train, y_train)

    with pytest.raises(
        ValueError, match=f"Holdout data X must have {forecast_horizon}"
    ):
        pl.predict(X, None, X_train, y_train)

    if hasattr(pl, "predict_proba"):
        with pytest.raises(
            ValueError, match=f"Holdout data X must have {forecast_horizon}"
        ):
            pl.predict_proba(X, X_train, y_train)


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
    [["One Hot Encoder"], ["Time Series Featurizer", "One Hot Encoder"]],
)
def test_time_series_pipeline_init(pipeline_class, estimator, components):
    component_graph = components + [estimator]
    if "Time Series Featurizer" not in components:
        pl = pipeline_class(
            component_graph=component_graph,
            parameters={
                "pipeline": {
                    "date_index": "date",
                    "gap": 0,
                    "max_delay": 5,
                    "forecast_horizon": 3,
                }
            },
        )
        assert "Time Series Featurizer" not in pl.parameters
        assert pl.parameters["pipeline"] == {
            "forecast_horizon": 3,
            "gap": 0,
            "max_delay": 5,
            "date_index": "date",
        }
    else:
        parameters = {
            "Time Series Featurizer": {
                "date_index": "date",
                "gap": 0,
                "max_delay": 5,
                "forecast_horizon": 3,
            },
            "pipeline": {
                "date_index": "date",
                "gap": 0,
                "max_delay": 5,
                "forecast_horizon": 3,
            },
        }
        pl = pipeline_class(component_graph=component_graph, parameters=parameters)
        assert pl.parameters["Time Series Featurizer"] == {
            "date_index": "date",
            "gap": 0,
            "forecast_horizon": 3,
            "max_delay": 5,
            "delay_features": True,
            "delay_target": True,
            "conf_level": 0.05,
            "rolling_window_size": 0.25,
        }
        assert pl.parameters["pipeline"] == {
            "gap": 0,
            "forecast_horizon": 3,
            "max_delay": 5,
            "date_index": "date",
        }

    assert (
        pipeline_class(component_graph=component_graph, parameters=pl.parameters) == pl
    )

    with pytest.raises(
        ValueError,
        match="date_index, gap, and max_delay parameters cannot be omitted from the parameters dict",
    ):
        pipeline_class(component_graph, {})


@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize(
    "forecast_horizon,gap,max_delay",
    [(1, 0, 1), (1, 1, 2), (2, 0, 2), (3, 1, 2), (1, 2, 2), (2, 7, 3), (3, 2, 4)],
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
    ts_data,
):

    X, y = ts_data

    if include_delayed_features:
        train_index = pd.date_range(
            f"2020-10-{1 + forecast_horizon + gap + max_delay}", "2020-10-31"
        )
        expected_target = np.arange(1 + gap + max_delay + forecast_horizon, 32)
        component_graph = ["Time Series Featurizer", estimator_name]
    else:
        train_index = pd.date_range(f"2020-10-01", f"2020-10-31")
        expected_target = np.arange(1, 32)
        component_graph = [estimator_name]

    pl = pipeline_class(
        component_graph=component_graph,
        parameters={
            "Time Series Featurizer": {
                "date_index": "date",
                "gap": gap,
                "forecast_horizon": forecast_horizon,
                "max_delay": max_delay,
                "delay_features": include_delayed_features,
                "delay_target": include_delayed_features,
                "conf_level": 1.0,
                "rolling_window_size": 1.0,
            },
            "pipeline": {
                "date_index": "date",
                "gap": gap,
                "max_delay": max_delay,
                "forecast_horizon": forecast_horizon,
            },
        },
    )

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


@pytest.mark.parametrize(
    "forecast_horizon,gap,max_delay",
    [
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 1),
        (2, 1, 2),
        (2, 2, 2),
        (3, 7, 3),
        (2, 2, 4),
    ],
)
def test_transform_all_but_final_for_time_series(
    forecast_horizon, gap, max_delay, ts_data
):
    X, y = ts_data
    pipeline = TimeSeriesRegressionPipeline(
        [
            "Time Series Featurizer",
            "DateTime Featurization Component",
            "Random Forest Regressor",
        ],
        parameters={
            "pipeline": {
                "forecast_horizon": forecast_horizon,
                "gap": gap,
                "max_delay": max_delay,
                "date_index": "date",
            },
            "Random Forest Regressor": {"n_jobs": 1},
            "Time Series Featurizer": {
                "max_delay": max_delay,
                "gap": gap,
                "forecast_horizon": forecast_horizon,
                "conf_level": 1.0,
                "rolling_window_size": 1.0,
                "date_index": "date",
            },
        },
    )
    X_train, y_train = X[:15], y[:15]
    X_validation, y_validation = X[15:], y[15:]
    pipeline.fit(X_train, y_train)
    features = pipeline.transform_all_but_final(X_validation, y_validation)
    delayer = TimeSeriesFeaturizer(
        max_delay=max_delay,
        gap=gap,
        forecast_horizon=forecast_horizon,
        conf_level=1.0,
        rolling_window_size=1.0,
        date_index="date",
    )
    date_featurizer = DateTimeFeaturizer()
    expected_features = date_featurizer.fit_transform(
        delayer.fit_transform(X_validation, y_validation)
    )
    assert_frame_equal(features, expected_features)
    features_with_training = pipeline.transform_all_but_final(
        X_validation, y_validation, X_train, y_train
    )
    delayed = date_featurizer.fit_transform(delayer.fit_transform(X, y)).iloc[15:]
    assert_frame_equal(features_with_training, delayed)


@pytest.mark.parametrize("include_feature_not_known_in_advance", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize(
    "forecast_horizon,gap,max_delay,n_to_pred,date_index",
    [
        (1, 0, 1, 1, None),
        (1, 0, 2, 1, None),
        (1, 1, 1, 1, None),
        (2, 1, 2, 1, None),
        (2, 1, 2, 2, None),
        (2, 2, 2, 1, None),
        (2, 2, 2, 2, None),
        (3, 7, 3, 1, None),
        (3, 7, 3, 2, None),
        (3, 7, 3, 3, None),
        (2, 2, 4, 1, None),
        (2, 2, 4, 2, None),
    ],
)
@pytest.mark.parametrize("reset_index", [True, False])
@pytest.mark.parametrize(
    "pipeline_class,estimator_name",
    [
        (TimeSeriesRegressionPipeline, "Random Forest Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier"),
    ],
)
@patch("evalml.pipelines.components.RandomForestClassifier.predict_proba")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
def test_predict_and_predict_in_sample(
    mock_regressor_predict,
    mock_classifier_predict,
    mock_classifier_predict_proba,
    pipeline_class,
    estimator_name,
    reset_index,
    forecast_horizon,
    gap,
    max_delay,
    n_to_pred,
    date_index,
    include_delayed_features,
    include_feature_not_known_in_advance,
    ts_data,
):

    X, target = ts_data
    if include_feature_not_known_in_advance:
        X["not_known_in_advance_1"] = pd.Series(range(X.shape[0]), index=X.index) + 200
        X["not_known_in_advance_2"] = pd.Series(range(X.shape[0]), index=X.index) + 100

    mock_to_check = mock_classifier_predict
    if pipeline_class == TimeSeriesBinaryClassificationPipeline:
        target = target % 2
    elif pipeline_class == TimeSeriesMulticlassClassificationPipeline:
        target = target % 3
    else:
        mock_to_check = mock_regressor_predict
    mock_to_check.side_effect = lambda x: x.iloc[: x.shape[0], 0]

    component_graph = ["DateTime Featurization Component", estimator_name]

    def predict_proba(X):
        X2 = X.iloc[: X.shape[0]]
        X2.ww.init()
        return X2

    mock_classifier_predict_proba.side_effect = predict_proba

    parameters = {
        "pipeline": {
            "date_index": "date",
            "gap": gap,
            "max_delay": max_delay,
            "forecast_horizon": forecast_horizon,
        },
        estimator_name: {"n_jobs": 1},
    }
    expected_features = DateTimeFeaturizer().fit_transform(X)
    expected_features_in_sample = expected_features.ww.iloc[20:]
    expected_features_pred = expected_features[20 + gap : 20 + gap + n_to_pred]

    X_predict_in_sample, target_predict_in_sample = X.iloc[20:], target.iloc[20:]
    X_predict = X.iloc[20 + gap : 20 + gap + n_to_pred]

    if include_feature_not_known_in_advance:
        X_predict.drop(columns=["not_known_in_advance_1", "not_known_in_advance_2"])

    if reset_index:
        X_predict = X_predict.reset_index(drop=True)

    if include_delayed_features:
        component_graph = ["Time Series Featurizer"] + component_graph
        delayer_params = {
            "date_index": "date",
            "gap": gap,
            "max_delay": max_delay,
            "forecast_horizon": forecast_horizon,
            "delay_features": True,
            "delay_target": True,
            "conf_level": 1.0,
            "rolling_window_size": 1.0,
        }
        parameters.update({"Time Series Featurizer": delayer_params})
        expected_features = TimeSeriesFeaturizer(**delayer_params).fit_transform(
            X, target
        )
        expected_features = DateTimeFeaturizer().fit_transform(
            expected_features, target
        )
        expected_features_in_sample = expected_features.ww.iloc[20:]
        expected_features_pred = expected_features[20 + gap : 20 + gap + n_to_pred]

    pl = pipeline_class(component_graph=component_graph, parameters=parameters)
    pl.fit(X.iloc[:20], target.iloc[:20])

    preds_in_sample = pl.predict_in_sample(
        X_predict_in_sample, target_predict_in_sample, X.iloc[:20], target.iloc[:20]
    )
    assert_frame_equal(mock_to_check.call_args[0][0], expected_features_in_sample)
    mock_to_check.reset_mock()
    preds = pl.predict(
        X_predict,
        None,
        X_train=X.iloc[:20],
        y_train=target.iloc[:20],
    )
    assert_frame_equal(mock_to_check.call_args[0][0], expected_features_pred)
    if is_classification(pl.problem_type):
        pred_proba = pl.predict_proba(
            X_predict,
            X_train=X.iloc[:20],
            y_train=target.iloc[:20],
        )
        assert_frame_equal(
            mock_classifier_predict_proba.call_args[0][0], expected_features_pred
        )
        assert len(pred_proba) == n_to_pred

    assert len(preds) == n_to_pred
    assert (preds.index == target.iloc[20 + gap : 20 + n_to_pred + gap].index).all()
    assert len(preds_in_sample) == len(target.iloc[20:])
    assert (preds_in_sample.index == target.iloc[20:].index).all()


@pytest.mark.parametrize(
    "pipeline_class,estimator_name",
    [
        (TimeSeriesRegressionPipeline, "Random Forest Regressor"),
        (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
        (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier"),
    ],
)
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
def test_predict_and_predict_in_sample_with_date_index(
    mock_regressor_predict,
    mock_classifier_predict,
    pipeline_class,
    estimator_name,
    ts_data,
):

    X, target = ts_data
    X["date"] = X.index
    mock_to_check = mock_classifier_predict
    if pipeline_class == TimeSeriesBinaryClassificationPipeline:
        target = target % 2
    elif pipeline_class == TimeSeriesMulticlassClassificationPipeline:
        target = target % 3
    else:
        mock_to_check = mock_regressor_predict
    mock_to_check.side_effect = lambda x: x.iloc[: x.shape[0], 0]

    component_graph = [
        "Time Series Featurizer",
        "DateTime Featurization Component",
        estimator_name,
    ]
    delayer_params = {
        "date_index": "date",
        "gap": 1,
        "max_delay": 3,
        "forecast_horizon": 1,
        "delay_features": True,
        "delay_target": True,
        "conf_level": 1.0,
        "rolling_window_size": 1.0,
    }
    parameters = {
        "pipeline": {
            "date_index": "date",
            "gap": 1,
            "max_delay": 3,
            "forecast_horizon": 1,
        },
        "Time Series Featurizer": delayer_params,
        estimator_name: {"n_jobs": 1},
    }

    feature_pipeline = pipeline_class(
        ["Time Series Featurizer", "DateTime Featurization Component"],
        parameters=parameters,
    )
    feature_pipeline.fit(X, target)
    expected_features = feature_pipeline.transform(X, target)

    expected_features_in_sample = expected_features.ww.iloc[20:]
    expected_features_pred = expected_features[20 + 1 : 20 + 1 + 1]

    pl = pipeline_class(component_graph=component_graph, parameters=parameters)

    pl.fit(X.iloc[:20], target.iloc[:20])
    preds_in_sample = pl.predict_in_sample(
        X.iloc[20:], target.iloc[20:], X.iloc[:20], target.iloc[:20]
    )
    assert_frame_equal(mock_to_check.call_args[0][0], expected_features_in_sample)
    mock_to_check.reset_mock()
    preds = pl.predict(
        X.iloc[20 + 1 : 20 + 1 + 1],
        None,
        X_train=X.iloc[:20],
        y_train=target.iloc[:20],
    )
    assert_frame_equal(mock_to_check.call_args[0][0], expected_features_pred)

    assert len(preds) == 1
    assert (preds.index == target.iloc[20 + 1 : 20 + 1 + 1].index).all()
    assert len(preds_in_sample) == len(target.iloc[20:])
    assert (preds_in_sample.index == target.iloc[20:].index).all()


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
@patch("evalml.pipelines.PipelineBase._score_all_objectives")
@patch("evalml.pipelines.TimeSeriesBinaryClassificationPipeline._score_all_objectives")
def test_ts_score(
    mock_binary_score,
    mock_score,
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
        component_graph=["Time Series Featurizer", estimator_name],
        parameters={
            "Time Series Featurizer": {
                "date_index": "date",
                "gap": gap,
                "max_delay": max_delay,
                "delay_features": include_delayed_features,
                "delay_target": include_delayed_features,
                "forecast_horizon": forecast_horizon,
            },
            "pipeline": {
                "date_index": "date",
                "gap": gap,
                "max_delay": max_delay,
                "forecast_horizon": forecast_horizon,
            },
        },
    )

    def mock_predict(X, y=None):
        return pd.Series(range(200, 200 + X.shape[0]))

    if isinstance(pl, TimeSeriesRegressionPipeline):
        mock_regressor_predict.side_effect = mock_predict
    else:
        mock_classifier_predict.side_effect = mock_predict

    pl.fit(X_train, y_train)
    pl.score(
        X_holdout,
        y_holdout,
        objectives=["MCC Binary"],
        X_train=X_train,
        y_train=y_train,
    )

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
def test_classification_pipeline_encodes_targets(
    mock_score,
    mock_predict,
    mock_predict_proba,
    mock_fit,
    pipeline_class,
    ts_data_binary,
):
    X, y = ts_data_binary
    y_series = pd.Series(y)
    df = pd.DataFrame({"negative": y_series, "positive": y_series})
    df.ww.init()
    mock_predict.side_effect = lambda data: ww.init_series(y_series[: data.shape[0]])
    mock_predict_proba.side_effect = lambda data: df.ww.iloc[: len(data)]
    y_encoded = y_series.map(
        lambda label: "positive" if label == 1 else "negative"
    ).astype("category")
    X_train, y_encoded_train = X.iloc[:29], y_encoded.iloc[:29]
    X_holdout, y_encoded_holdout = X.iloc[29:], y_encoded.iloc[29:]
    pl = pipeline_class(
        component_graph={
            "Label Encoder": ["Label Encoder", "X", "y"],
            "Time Series Featurizer": [
                "Time Series Featurizer",
                "Label Encoder.x",
                "Label Encoder.y",
            ],
            "DT": [
                "DateTime Featurization Component",
                "Time Series Featurizer.x",
                "Label Encoder.y",
            ],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "DT.x",
                "Label Encoder.y",
            ],
        },
        parameters={
            "Time Series Featurizer": {
                "date_index": "date",
                "gap": 0,
                "max_delay": 1,
                "forecast_horizon": 1,
                "conf_level": 1.0,
                "rolling_window_size": 1.0,
            },
            "pipeline": {
                "date_index": "date",
                "gap": 0,
                "max_delay": 1,
                "forecast_horizon": 1,
            },
        },
    )

    # Check fit encodes target
    pl.fit(X_train, y_encoded_train)
    _, target_passed_to_estimator = mock_fit.call_args[0]
    # Check that target is converted to ints. Use .iloc[1:] because the first feature row has NaNs
    assert_series_equal(
        target_passed_to_estimator, pl._encode_targets(y_encoded_train.iloc[2:])
    )

    # Check predict encodes target
    predictions = pl.predict(X_holdout.iloc[:1], None, X_train, y_encoded_train)
    assert set(predictions.unique()).issubset({"positive", "negative"})

    predictions_in_sample = pl.predict_in_sample(
        X_holdout, y_encoded_holdout, X_train, y_encoded_train, objective=None
    )
    assert set(predictions_in_sample.unique()).issubset({"positive", "negative"})

    # Check predict proba column names are correct
    predict_proba = pl.predict_proba(X_holdout.iloc[:1], X_train, y_encoded_train)
    assert set(predict_proba.columns.unique()).issubset({"positive", "negative"})

    predict_proba_in_sample = pl.predict_proba_in_sample(
        X_holdout, y_encoded_holdout, X_train, y_encoded_train
    )
    assert set(predict_proba_in_sample.columns.unique()).issubset(
        {"positive", "negative"}
    )


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
def test_ts_score_works(
    pipeline_class,
    objectives,
    data_type,
    ts_data_binary,
    ts_data_multi,
    ts_data,
    make_data_type,
    time_series_regression_pipeline_class,
    time_series_binary_classification_pipeline_class,
    time_series_multiclass_classification_pipeline_class,
):
    pipeline = None
    estimator = None
    if pipeline_class == TimeSeriesRegressionPipeline:
        pipeline = time_series_regression_pipeline_class
        estimator = "Random Forest Regressor"
    elif pipeline_class == TimeSeriesBinaryClassificationPipeline:
        pipeline = time_series_binary_classification_pipeline_class
        estimator = "Logistic Regression Classifier"
    else:
        pipeline = time_series_multiclass_classification_pipeline_class
        estimator = "Logistic Regression Classifier"

    pl = pipeline(
        parameters={
            "pipeline": {
                "date_index": "date",
                "gap": 1,
                "max_delay": 3,
                "delay_features": False,
                "forecast_horizon": 10,
            },
            "Time Series Featurizer": {
                "date_index": "date",
                "gap": 1,
                "max_delay": 3,
                "delay_features": False,
                "forecast_horizon": 10,
            },
            estimator: {"n_jobs": 1},
        },
    )
    if pl.problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X, y = ts_data_binary
        y = pd.Series(y).map(lambda label: "good" if label == 1 else "bad")
    elif pl.problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = ts_data_multi
        label_map = {0: "good", 1: "bad", 2: "best"}
        y = pd.Series(y).map(lambda label: label_map[label])
    else:
        X, y = ts_data
        y = pd.Series(y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    X_train, y_train = X.iloc[:20], y.iloc[:20]
    X_valid, y_valid = X.iloc[21:], y.iloc[21:]

    pl.fit(X_train, y_train)
    pl.score(X_valid, y_valid, objectives, X_train, y_train)


@patch("evalml.objectives.BinaryClassificationObjective.decision_function")
@patch("evalml.pipelines.components.Estimator.predict_proba")
@patch(
    "evalml.pipelines.components.Estimator.predict",
    return_value=ww.init_series(pd.Series([1.0, 0.0, 1.0])),
)
def test_binary_classification_predictions_thresholded_properly(
    mock_predict,
    mock_predict_proba,
    mock_obj_decision,
    X_y_binary,
    dummy_ts_binary_pipeline_class,
):
    proba = pd.DataFrame({0: [1.0, 1.0, 0.0]})
    proba.ww.init()
    mock_predict_proba.return_value = proba
    X, y = X_y_binary
    X, y = pd.DataFrame(X), pd.Series(y)
    X["date"] = pd.Series(pd.date_range("2010-01-01", periods=X.shape[0]))
    X_train, y_train = X.iloc[:60], y.iloc[:60]
    X_validation = X.iloc[60:63]
    binary_pipeline = dummy_ts_binary_pipeline_class(
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 3,
                "date_index": "date",
                "forecast_horizon": 3,
            },
        }
    )
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    binary_pipeline.fit(X_train, y_train)
    binary_pipeline.predict(X_validation, None, X_train, y_train)
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    binary_pipeline.predict(X_validation, "precision", X_train, y_train)
    mock_predict.assert_called()
    mock_predict.reset_mock()

    # test custom threshold set but no objective passed
    proba = pd.DataFrame([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    proba.ww.init()
    mock_predict_proba.return_value = proba
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X_validation, None, X_train, y_train)
    mock_predict_proba.assert_called()
    mock_predict_proba.reset_mock()
    mock_obj_decision.assert_not_called()
    mock_predict.assert_not_called()

    # test custom threshold set but no objective passed
    binary_pipeline.threshold = 0.6
    binary_pipeline.predict(X_validation, None, X_train, y_train)
    mock_predict_proba.assert_called()
    mock_predict_proba.reset_mock()
    mock_obj_decision.assert_not_called()
    mock_predict.assert_not_called()

    # test custom threshold set and objective passed
    binary_pipeline.threshold = 0.6
    mock_obj_decision.return_value = pd.Series([1.0, 0.0, 1.0])
    binary_pipeline.predict(X_validation, "precision", X_train, y_train)
    mock_predict_proba.assert_called()
    mock_predict_proba.reset_mock()
    mock_predict.assert_not_called()
    mock_obj_decision.assert_called()


def test_binary_predict_pipeline_objective_mismatch(
    X_y_binary, dummy_ts_binary_pipeline_class
):
    X, y = X_y_binary
    X, y = pd.DataFrame(X), pd.Series(y)
    X["date"] = pd.Series(pd.date_range("2010-01-01", periods=X.shape[0]))
    binary_pipeline = dummy_ts_binary_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {
                "gap": 0,
                "max_delay": 0,
                "date_index": "date",
                "forecast_horizon": 2,
            },
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
    ts_data_binary,
    ts_data_multi,
    ts_data,
    time_series_binary_classification_pipeline_class,
    time_series_multiclass_classification_pipeline_class,
    time_series_regression_pipeline_class,
):
    if problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X, y = ts_data_binary
        clf = time_series_binary_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 10,
                },
                "Time Series Featurizer": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 10,
                },
            }
        )

    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = ts_data_multi
        clf = time_series_multiclass_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 10,
                },
                "Time Series Featurizer": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 10,
                },
            }
        )
    else:
        X, y = ts_data
        clf = time_series_regression_pipeline_class(
            parameters={
                "Random Forest Regressor": {"n_jobs": 1},
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 10,
                },
                "Time Series Featurizer": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 10,
                },
            }
        )

    X, y = pd.DataFrame(X), pd.Series(y)
    X_train, y_train = X.iloc[:21], y.iloc[:21]
    X_holdout = X.iloc[21:]

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
            to_patch = "evalml.pipelines.TimeSeriesBinaryClassificationPipeline.predict_in_sample"
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
            clf.predict(
                X_holdout, objective="Log Loss Binary", X_train=X_train, y_train=y_train
            )
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is not None

            clf.predict_proba(X_holdout, X_train=X_train, y_train=y_train)
    else:
        clf.predict(X_holdout, None, X_train, y_train)
    clf.feature_importance


def test_ts_binary_pipeline_target_thresholding(
    make_data_type, time_series_binary_classification_pipeline_class, ts_data_binary
):
    X, y = ts_data_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)
    objective = get_objective("F1", return_instance=True)

    binary_pipeline = time_series_binary_classification_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {
                "gap": 0,
                "max_delay": 0,
                "date_index": "date",
                "forecast_horizon": 10,
            },
            "Time Series Featurizer": {
                "date_index": "date",
                "gap": 0,
                "max_delay": 0,
                "forecast_horizon": 10,
            },
        }
    )
    X_train, y_train = X.ww.iloc[:21], y.ww.iloc[:21]
    X_holdout, y_holdout = X.ww.iloc[21:], y.ww.iloc[21:]
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
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X["date"] = pd.Series(pd.date_range("2010-01-01", periods=X.shape[0]))
    binary_pipeline = time_series_binary_classification_pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {
                "gap": 3,
                "max_delay": 0,
                "date_index": "date",
                "forecast_horizon": 5,
            },
            "Time Series Featurizer": {
                "gap": 3,
                "max_delay": 0,
                "date_index": "date",
                "forecast_horizon": 5,
            },
        }
    )
    X_train, y_train = X[:50], y[:50]
    X_validation, y_validation = X[53:58], y[53:58]
    mock_decision_function.return_value = pd.Series([0] * 5)
    binary_pipeline.threshold = 0.7
    binary_pipeline.fit(X_train, y_train)
    fraud_cost = FraudCost(amount_col=0)
    binary_pipeline.score(
        X_validation, y_validation, ["precision", "auc", fraud_cost], X_train, y_train
    )
    mock_decision_function.assert_called()


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
@patch("evalml.pipelines.LogisticRegressionClassifier.fit")
@patch("evalml.pipelines.components.ElasticNetRegressor.fit")
def test_time_series_pipeline_fit_with_transformed_target(
    mock_en_fit, mock_lr_fit, problem_type, ts_data
):
    class AddTwo(Transformer):
        """Add Two to target for testing."""

        modifies_target = True
        modifies_features = False

        name = "AddTwo"
        hyperparameter_ranges = {}

        def __init__(self, drop_old_columns=True, random_seed=0):
            super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

        def fit(self, X, y):
            return self

        def transform(self, X, y):
            return infer_feature_types(X), infer_feature_types(y) + 2

    X, y = ts_data
    y = y % 2

    if is_classification(problem_type):
        estimator = "Logistic Regression Classifier"
        mock_to_check = mock_lr_fit
    else:
        estimator = "Elastic Net Regressor"
        mock_to_check = mock_en_fit

    pipeline_class = _get_pipeline_base_class(problem_type)
    pipeline = pipeline_class(
        component_graph={
            "AddTwo": [AddTwo, "X", "y"],
            "Estimator": [estimator, "X", "AddTwo.y"],
        },
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 2,
                "date_index": "date",
                "forecast_horizon": 3,
            },
        },
    )
    pipeline.fit(X, y)
    pd.testing.assert_series_equal(mock_to_check.call_args[0][1], y + 2)


@pytest.mark.skip_if_39
@pytest.mark.noncore_dependency
def test_time_series_pipeline_with_detrender(ts_data):
    X, y = ts_data
    component_graph = {
        "Polynomial Detrender": ["Polynomial Detrender", "X", "y"],
        "Time Series Featurizer": ["Time Series Featurizer", "X", "y"],
        "Dt": ["DateTime Featurization Component", "Time Series Featurizer.x", "y"],
        "Regressor": [
            "Linear Regressor",
            "Dt.x",
            "Polynomial Detrender.y",
        ],
    }
    pipeline = TimeSeriesRegressionPipeline(
        component_graph=component_graph,
        parameters={
            "pipeline": {
                "gap": 1,
                "max_delay": 10,
                "date_index": "date",
                "forecast_horizon": 7,
            },
            "Time Series Featurizer": {
                "max_delay": 2,
                "gap": 1,
                "forecast_horizon": 10,
                "date_index": "date",
            },
        },
    )
    X_train, y_train = X[:23], y[:23]
    X_validation, y_validation = X[24:], y[24:]
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation, None, X_train, y_train)
    features = pipeline.transform_all_but_final(
        X_validation, y_validation, X_train, y_train
    )
    detrender = pipeline.component_graph.get_component("Polynomial Detrender")
    preds = pipeline.estimator.predict(features)
    preds.index = y_validation.index
    expected = detrender.inverse_transform(preds)
    expected = infer_feature_types(expected)
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
    X.ww["date"] = pd.Series(pd.date_range("2010-01-01", periods=X.shape[0]))
    X_train, y_train = X.ww.iloc[:70], y.ww.iloc[:70]
    X_validation = X.ww.iloc[70:73]

    pipeline_class = _get_pipeline_base_class(problem_type)
    pipeline = pipeline_class(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        },
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 2,
                "date_index": "date",
                "forecast_horizon": 3,
            },
        },
    )
    pipeline.fit(X_train, y_train)
    msg = "Cannot call {method} on a component graph because the final component is not an Estimator."
    if is_classification(problem_type):
        with pytest.raises(
            ValueError, match=re.escape(msg.format(method="predict_proba()"))
        ):
            pipeline.predict_proba(X_validation, X_train, y_train)
        with pytest.raises(
            ValueError, match=re.escape(msg.format(method="predict_proba_in_sample()"))
        ):
            pipeline.predict_proba_in_sample(X_validation, None, X_train, y_train)

    with pytest.raises(ValueError, match=re.escape(msg.format(method="predict()"))):
        pipeline.predict(X_validation, None, X_train, y_train)

    with pytest.raises(
        ValueError, match=re.escape(msg.format(method="predict_in_sample()"))
    ):
        pipeline.predict_in_sample(X_validation, None, X_train, y_train)


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
    X.ww["date"] = pd.Series(pd.date_range("2010-01-01", periods=X.shape[0]))
    y = make_data_type("ww", y)
    X_train, y_train = X.ww.iloc[:70], y.ww.iloc[:70]
    X_validation, y_validation = X.ww.iloc[70:73], y.ww.iloc[70:73]
    mock_imputer_transform.side_effect = lambda x, y: x
    mock_ohe_transform.side_effect = lambda x, y: x
    pipeline_class = _get_pipeline_base_class(problem_type)
    pipeline = pipeline_class(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        },
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 0,
                "date_index": "date",
                "forecast_horizon": 3,
            },
        },
    )

    pipeline.fit(X_train, y_train)
    transformed_X = pipeline.transform(X_validation, y_validation)
    assert_frame_equal(X_validation, transformed_X)


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
    ts_data_binary,
    ts_data_multi,
    ts_data,
    time_series_binary_classification_pipeline_class,
    time_series_multiclass_classification_pipeline_class,
    time_series_regression_pipeline_class,
    make_data_type,
):
    X, y = ts_data_binary

    def make_data(X, y):
        X = make_data_type("ww", X)
        y = make_data_type("ww", y)
        X_train, y_train = X.ww.iloc[:15], y.ww.iloc[:15]
        X_validation, y_validation = X.ww.iloc[15:20], y.ww.iloc[15:20]
        return X_train, y_train, X_validation, y_validation

    if problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X_train, y_train, X_validation, y_validation = make_data(X, y)
        pipeline = time_series_binary_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 5,
                },
                "Time Series Featurizer": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 5,
                },
            }
        )

    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = ts_data_multi
        X_train, y_train, X_validation, y_validation = make_data(X, y)
        pipeline = time_series_multiclass_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 5,
                },
                "Time Series Featurizer": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 5,
                },
            }
        )
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        X, y = ts_data
        X_train, y_train, X_validation, y_validation = make_data(X, y)
        pipeline = time_series_regression_pipeline_class(
            parameters={
                "Random Forest Regressor": {"n_jobs": 1},
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 5,
                },
                "Time Series Featurizer": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": "date",
                    "forecast_horizon": 5,
                },
            }
        )

    pipeline.fit(X_train, y_train)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call transform() on a component graph because the final component is not a Transformer."
        ),
    ):
        pipeline.transform(X_validation, y_validation)


def test_date_index_cannot_be_none(time_series_regression_pipeline_class):

    with pytest.raises(ValueError, match="date_index cannot be None!"):
        time_series_regression_pipeline_class(
            {
                "pipeline": {
                    "gap": 1,
                    "max_delay": 2,
                    "forecast_horizon": 1,
                    "date_index": None,
                }
            }
        )
