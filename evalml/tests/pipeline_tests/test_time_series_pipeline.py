from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline
)
from evalml.problem_types import ProblemTypes


@pytest.mark.parametrize("pipeline_class,estimator", [(TimeSeriesRegressionPipeline, "Linear Regressor"),
                                                      (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
                                                      (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier")])
@pytest.mark.parametrize("components", [["One Hot Encoder"],
                                        ["Delayed Feature Transformer", "One Hot Encoder"]])
def test_time_series_pipeline_init(pipeline_class, estimator, components):

    class Pipeline(pipeline_class):
        component_graph = components + [estimator]

    if "Delayed Feature Transformer" not in components:
        pl = Pipeline({'pipeline': {"gap": 3, "max_delay": 5}})
        assert "Delayed Feature Transformer" not in pl.parameters
        assert pl.parameters['pipeline'] == {"gap": 3, "max_delay": 5}
    else:
        parameters = {"Delayed Feature Transformer": {"gap": 3, "max_delay": 5},
                      "pipeline": {"gap": 3, "max_delay": 5}}
        pl = Pipeline(parameters)
        assert pl.parameters['Delayed Feature Transformer'] == {"gap": 3, "max_delay": 5,
                                                                "delay_features": True, "delay_target": True}
        assert pl.parameters['pipeline'] == {"gap": 3, "max_delay": 5}

    assert Pipeline(parameters=pl.parameters) == pl

    with pytest.raises(ValueError, match="gap and max_delay parameters cannot be omitted from the parameters dict"):
        Pipeline({})


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(0, 0), (1, 0), (0, 2), (1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"),
                                                           (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
                                                           (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._encode_targets", side_effect=lambda y: y)
def test_fit_drop_nans_before_estimator(mock_encode_targets, mock_classifier_fit, mock_regressor_fit, pipeline_class,
                                        estimator_name, gap, max_delay, include_delayed_features, only_use_y, ts_data):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    if include_delayed_features:
        train_index = pd.date_range(f"2020-10-{1 + max_delay}", f"2020-10-{31-gap}")
        expected_target = np.arange(1 + gap + max_delay, 32)
    else:
        train_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")
        expected_target = np.arange(1 + gap, 32)

    class Pipeline(pipeline_class):
        component_graph = ["Delayed Feature Transformer", estimator_name]

    pl = Pipeline({"Delayed Feature Transformer": {"gap": gap, "max_delay": max_delay,
                                                   "delay_features": include_delayed_features,
                                                   "delay_target": include_delayed_features},
                   "pipeline": {"gap": gap, "max_delay": max_delay}})

    if only_use_y:
        pl.fit(None, y)
    else:
        pl.fit(X, y)

    if isinstance(pl, TimeSeriesRegressionPipeline):
        df_passed_to_estimator, target_passed_to_estimator = mock_regressor_fit.call_args[0]
    else:
        df_passed_to_estimator, target_passed_to_estimator = mock_classifier_fit.call_args[0]

    # NaNs introduced by shifting are dropped
    assert not df_passed_to_estimator.isna().any(axis=1).any()
    assert not target_passed_to_estimator.isna().any()

    # Check the estimator was trained on the expected dates
    pd.testing.assert_index_equal(df_passed_to_estimator.index, train_index)
    np.testing.assert_equal(target_passed_to_estimator.values, expected_target)


@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"),
                                                           (TimeSeriesBinaryClassificationPipeline, "Extra Trees Classifier"),
                                                           (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier")])
def test_pipeline_fit_runtime_error(pipeline_class, estimator_name, ts_data):

    X, y = ts_data

    class Pipeline(pipeline_class):
        component_graph = ["Delayed Feature Transformer", estimator_name]

    pl = Pipeline({"Delayed Feature Transformer": {"gap": 0, "max_delay": 0},
                   "pipeline": {"gap": 0, "max_delay": 0}})
    with pytest.raises(RuntimeError, match="Pipeline computed empty features during call to .fit."):
        pl.fit(None, y)

    class Pipeline2(pipeline_class):
        component_graph = [estimator_name]

    pl = Pipeline2({"pipeline": {"gap": 5, "max_delay": 7}})
    with pytest.raises(RuntimeError, match="Pipeline computed empty features during call to .fit."):
        pl.fit(None, y)


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(0, 0), (1, 0), (0, 2), (1, 1), (1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"),
                                                           (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
                                                           (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier")])
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._encode_targets", side_effect=lambda y: y)
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets", side_effect=lambda y: y)
def test_predict_pad_nans(mock_decode_targets, mock_encode_targets,
                          mock_regressor_predict, mock_regressor_fit, mock_classifier_predict, mock_classifier_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, include_delayed_features, only_use_y, ts_data):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    class Pipeline(pipeline_class):
        component_graph = ["Delayed Feature Transformer", estimator_name]

    pl = Pipeline({"Delayed Feature Transformer": {"gap": gap, "max_delay": max_delay,
                                                   "delay_features": include_delayed_features,
                                                   "delay_target": include_delayed_features},
                   "pipeline": {"gap": gap, "max_delay": max_delay}})

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    if isinstance(pl, TimeSeriesRegressionPipeline):
        mock_regressor_predict.side_effect = mock_predict
    else:
        mock_classifier_predict.side_effect = mock_predict

    if only_use_y:
        pl.fit(None, y)
        preds = pl.predict(None, y)
    else:
        pl.fit(X, y)
        preds = pl.predict(X, y)

    # Check that the predictions have NaNs for the first n_delay dates
    if include_delayed_features:
        assert np.isnan(preds.values[:max_delay]).all()
    else:
        assert not np.isnan(preds.values).any()


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(0, 0), (1, 0), (0, 2), (1, 1), (1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"),
                                                           (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
                                                           (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._encode_targets", side_effect=lambda y: y)
@patch("evalml.pipelines.PipelineBase._score_all_objectives")
def test_score_drops_nans(mock_score, mock_encode_targets,
                          mock_classifier_predict, mock_classifier_fit,
                          mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, include_delayed_features, only_use_y, ts_data):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    if include_delayed_features:
        expected_target = np.arange(1 + gap + max_delay, 32)
        target_index = pd.date_range(f"2020-10-{1 + max_delay}", f"2020-10-{31 - gap}")
    else:
        expected_target = np.arange(1 + gap, 32)
        target_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")

    class Pipeline(pipeline_class):
        component_graph = ["Delayed Feature Transformer", estimator_name]

    pl = Pipeline({"Delayed Feature Transformer": {"gap": gap, "max_delay": max_delay,
                                                   "delay_features": include_delayed_features,
                                                   "delay_target": include_delayed_features},
                   "pipeline": {"gap": gap, "max_delay": max_delay}})

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    if isinstance(pl, TimeSeriesRegressionPipeline):
        mock_regressor_predict.side_effect = mock_predict
    else:
        mock_classifier_predict.side_effect = mock_predict

    if only_use_y:
        pl.fit(None, y)
        pl.score(X=None, y=y, objectives=['MCC Binary'])
    else:
        pl.fit(X, y)
        pl.score(X, y, objectives=["MCC Binary"])

    # Verify that NaNs are dropped before passed to objectives
    _, target, preds = mock_score.call_args[0]
    assert not target.isna().any()
    assert not preds.isna().any()

    # Target used for scoring matches expected dates
    pd.testing.assert_index_equal(target.index, target_index)
    np.testing.assert_equal(target.values, expected_target)


@pytest.mark.parametrize("pipeline_class", [TimeSeriesBinaryClassificationPipeline, TimeSeriesMulticlassClassificationPipeline])
@patch("evalml.pipelines.LogisticRegressionClassifier.fit")
@patch("evalml.pipelines.LogisticRegressionClassifier.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._score_all_objectives")
def test_classification_pipeline_encodes_targets(mock_score, mock_predict, mock_fit, pipeline_class, X_y_binary):
    X, y = X_y_binary
    y_series = pd.Series(y)
    mock_predict.return_value = y_series
    X = pd.DataFrame({"feature": range(len(y))})
    y_encoded = y_series.map(lambda label: "positive" if label == 1 else "negative")

    class MyTsPipeline(pipeline_class):
        component_graph = ['Delayed Feature Transformer', 'Logistic Regression Classifier']

    pl = MyTsPipeline({"Delayed Feature Transformer": {"gap": 0, "max_delay": 1},
                       "pipeline": {"gap": 0, "max_delay": 1}})

    pl.fit(X, y_encoded)

    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "target_delay_1": y_series.shift(1)}).dropna(axis=0, how='any')

    df_passed_to_estimator, target_passed_to_estimator = mock_fit.call_args[0]

    # Check the features have target values encoded as ints.
    pd.testing.assert_frame_equal(df_passed_to_estimator, answer)

    # Check that target is converted to ints. Use .iloc[1:] because the first feature row has NaNs
    pd.testing.assert_series_equal(target_passed_to_estimator, y_series.iloc[1:])

    pl.predict(X, y_encoded)
    # Best way to get the argument since the api changes between 3.6/3.7 and 3.8
    df_passed_to_predict = mock_predict.call_args[0][0]
    pd.testing.assert_frame_equal(df_passed_to_predict, answer)

    mock_predict.reset_mock()
    # Since we mock score_all_objectives, the objective doesn't matter
    pl.score(X, y_encoded, objectives=['MCC Binary'])
    df_passed_to_predict = mock_predict.call_args[0][0]
    pd.testing.assert_frame_equal(df_passed_to_predict, answer)


@pytest.mark.parametrize("pipeline_class,objectives", [(TimeSeriesBinaryClassificationPipeline, ["MCC Binary"]),
                                                       (TimeSeriesBinaryClassificationPipeline, ["Log Loss Binary"]),
                                                       (TimeSeriesBinaryClassificationPipeline, ["MCC Binary", "Log Loss Binary"]),
                                                       (TimeSeriesMulticlassClassificationPipeline, ["MCC Multiclass"]),
                                                       (TimeSeriesMulticlassClassificationPipeline, ["Log Loss Multiclass"]),
                                                       (TimeSeriesMulticlassClassificationPipeline, ["MCC Multiclass", "Log Loss Multiclass"]),
                                                       (TimeSeriesRegressionPipeline, ['R2']),
                                                       (TimeSeriesRegressionPipeline, ['R2', "Mean Absolute Percentage Error"])])
@pytest.mark.parametrize("use_ww", [True, False])
def test_score_works(pipeline_class, objectives, use_ww, X_y_binary, X_y_multi, X_y_regression):

    preprocessing = ['Delayed Feature Transformer']
    if pipeline_class == TimeSeriesRegressionPipeline:
        components = preprocessing + ['Random Forest Regressor']
    else:
        components = preprocessing + ["Logistic Regression Classifier"]

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({"pipeline": {"gap": 1, "max_delay": 2, "delay_features": False},
                   components[-1]: {'n_jobs': 1}})
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
    if use_ww:
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    pl.fit(X, y)
    if expected_unique_values:
        # NaNs are expected because of padding due to max_delay
        assert set(pl.predict(X, y).dropna().unique()) == expected_unique_values
    pl.score(X, y, objectives)


@patch('evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets')
@patch('evalml.objectives.BinaryClassificationObjective.decision_function')
@patch('evalml.pipelines.components.Estimator.predict_proba', return_value=pd.DataFrame({0: [1.]}))
@patch('evalml.pipelines.components.Estimator.predict', return_value=pd.Series([1.]))
def test_binary_classification_predictions_thresholded_properly(mock_predict, mock_predict_proba,
                                                                mock_obj_decision, mock_decode,
                                                                X_y_binary, dummy_ts_binary_pipeline_class):
    mock_objs = [mock_decode, mock_predict]
    mock_decode.return_value = pd.Series([0, 1])
    X, y = X_y_binary
    binary_pipeline = dummy_ts_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                 "pipeline": {"gap": 0, "max_delay": 0}})
    # test no objective passed and no custom threshold uses underlying estimator's predict method
    binary_pipeline.fit(X, y)
    binary_pipeline.predict(X, y)
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()

    # test objective passed but no custom threshold uses underlying estimator's predict method
    binary_pipeline.predict(X, y, 'precision')
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
    binary_pipeline.predict(X, y, 'precision')
    for mock_obj in mock_objs:
        mock_obj.assert_called()
        mock_obj.reset_mock()
    mock_predict.assert_not_called()
    mock_obj_decision.assert_called()


@patch('evalml.pipelines.PipelineBase.compute_estimator_features')
def test_binary_predict_pipeline_objective_mismatch(mock_transform, X_y_binary, dummy_ts_binary_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = dummy_ts_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                 "pipeline": {"gap": 0, "max_delay": 0}})
    binary_pipeline.fit(X, y)
    with pytest.raises(ValueError, match="Objective Precision Micro is not defined for time series binary classification."):
        binary_pipeline.predict(X, y, "precision micro")
    mock_transform.assert_called()
