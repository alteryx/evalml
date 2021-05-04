from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_series_equal

from evalml.exceptions import PipelineNotYetFittedError
from evalml.objectives import FraudCost, get_objective
from evalml.pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline
)
from evalml.preprocessing.utils import is_classification
from evalml.problem_types import ProblemTypes


@pytest.mark.parametrize("pipeline_class,estimator", [(TimeSeriesRegressionPipeline, "Linear Regressor"),
                                                      (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
                                                      (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier")])
@pytest.mark.parametrize("components", [["One Hot Encoder"],
                                        ["Delayed Feature Transformer", "One Hot Encoder"]])
def test_time_series_pipeline_init(pipeline_class, estimator, components):
    component_graph = components + [estimator]
    if "Delayed Feature Transformer" not in components:
        pl = pipeline_class(component_graph=component_graph,
                            parameters={'pipeline': {"date_index": None, "gap": 3, "max_delay": 5}})
        assert "Delayed Feature Transformer" not in pl.parameters
        assert pl.parameters['pipeline'] == {"gap": 3, "max_delay": 5, "date_index": None}
    else:
        parameters = {"Delayed Feature Transformer": {"date_index": None, "gap": 3, "max_delay": 5},
                      "pipeline": {"date_index": None, "gap": 3, "max_delay": 5}}
        pl = pipeline_class(component_graph=component_graph, parameters=parameters)
        assert pl.parameters['Delayed Feature Transformer'] == {"date_index": None, "gap": 3, "max_delay": 5,
                                                                "delay_features": True, "delay_target": True}
        assert pl.parameters['pipeline'] == {"gap": 3, "max_delay": 5, "date_index": None}

    assert pipeline_class(component_graph=component_graph, parameters=pl.parameters) == pl

    with pytest.raises(ValueError, match="date_index, gap, and max_delay parameters cannot be omitted from the parameters dict"):
        pipeline_class(component_graph, {})


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

    pl = pipeline_class(component_graph=["Delayed Feature Transformer", estimator_name],
                        parameters={"Delayed Feature Transformer": {"date_index": None, "gap": gap, "max_delay": max_delay,
                                                                    "delay_features": include_delayed_features,
                                                                    "delay_target": include_delayed_features},
                                    "pipeline": {"date_index": None, "gap": gap, "max_delay": max_delay}})

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


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize("gap,max_delay,date_index", [(0, 0, None), (1, 0, None), (0, 2, None), (1, 1, None),
                                                      (1, 2, None), (2, 2, None), (7, 3, None), (2, 4, None)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"),
                                                           (TimeSeriesBinaryClassificationPipeline, "Random Forest Classifier"),
                                                           (TimeSeriesMulticlassClassificationPipeline, "Random Forest Classifier")])
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets", side_effect=lambda y: y)
def test_predict_pad_nans(mock_decode_targets,
                          mock_regressor_predict, mock_regressor_fit, mock_classifier_predict, mock_classifier_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, date_index, include_delayed_features, only_use_y, ts_data):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    pl = pipeline_class(component_graph=["Delayed Feature Transformer", estimator_name],
                        parameters={"Delayed Feature Transformer": {"date_index": None, "gap": gap, "max_delay": max_delay,
                                                                    "delay_features": include_delayed_features,
                                                                    "delay_target": include_delayed_features},
                                    "pipeline": {"date_index": None, "gap": gap, "max_delay": max_delay}})

    def mock_predict(df, y=None):
        return ww.DataColumn(pd.Series(range(200, 200 + df.shape[0])))

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
        assert np.isnan(preds.to_series().values[:max_delay]).all()
    else:
        assert not np.isnan(preds.to_series().values).any()


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_delayed_features", [True, False])
@pytest.mark.parametrize("gap,max_delay,date_index", [(0, 0, None), (1, 0, None), (0, 2, None), (1, 1, None), (1, 2, None),
                                                      (2, 2, None), (7, 3, None), (2, 4, None)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor"),
                                                           (TimeSeriesBinaryClassificationPipeline, "Logistic Regression Classifier"),
                                                           (TimeSeriesMulticlassClassificationPipeline, "Logistic Regression Classifier")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
@patch("evalml.pipelines.components.LogisticRegressionClassifier.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._encode_targets", side_effect=lambda y: y)
@patch("evalml.pipelines.PipelineBase._score_all_objectives")
@patch("evalml.pipelines.TimeSeriesBinaryClassificationPipeline._score_all_objectives")
def test_score_drops_nans(mock_binary_score, mock_score, mock_encode_targets,
                          mock_classifier_predict, mock_classifier_fit,
                          mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, date_index, include_delayed_features, only_use_y, ts_data):
    if pipeline_class == TimeSeriesBinaryClassificationPipeline:
        mock_score = mock_binary_score
    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    if include_delayed_features:
        expected_target = np.arange(1 + gap + max_delay, 32)
        target_index = pd.date_range(f"2020-10-{1 + max_delay}", f"2020-10-{31 - gap}")
    else:
        expected_target = np.arange(1 + gap, 32)
        target_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")

    pl = pipeline_class(component_graph=["Delayed Feature Transformer", estimator_name],
                        parameters={"Delayed Feature Transformer": {"date_index": None, "gap": gap, "max_delay": max_delay,
                                                                    "delay_features": include_delayed_features,
                                                                    "delay_target": include_delayed_features},
                                    "pipeline": {"date_index": None, "gap": gap, "max_delay": max_delay}})

    def mock_predict(X, y=None):
        return ww.DataColumn(pd.Series(range(200, 200 + X.shape[0])))

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
@patch("evalml.pipelines.LogisticRegressionClassifier.predict_proba")
@patch("evalml.pipelines.LogisticRegressionClassifier.predict")
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._score_all_objectives")
@patch("evalml.pipelines.ClassificationPipeline._decode_targets")
@patch("evalml.pipelines.ClassificationPipeline._encode_targets")
def test_classification_pipeline_encodes_targets(mock_encode, mock_decode,
                                                 mock_score, mock_predict, mock_predict_proba,
                                                 mock_fit, pipeline_class, X_y_binary):
    X, y = X_y_binary
    y_series = pd.Series(y)
    mock_predict.return_value = ww.DataColumn(y_series)
    mock_predict_proba.return_value = ww.DataTable(pd.DataFrame({"negative": y_series,
                                                                 "positive": y_series}))
    X = pd.DataFrame({"feature": range(len(y))})
    y_encoded = y_series.map(lambda label: "positive" if label == 1 else "negative")

    mock_encode.return_value = y_series
    mock_decode.return_value = y_encoded

    pl = pipeline_class(component_graph=['Delayed Feature Transformer', 'Logistic Regression Classifier'],
                        parameters={"Delayed Feature Transformer": {"date_index": None, "gap": 0, "max_delay": 1},
                                    "pipeline": {"date_index": None, "gap": 0, "max_delay": 1}})

    # Check fit encodes target
    pl.fit(X, y_encoded)
    _, target_passed_to_estimator = mock_fit.call_args[0]

    # Check that target is converted to ints. Use .iloc[1:] because the first feature row has NaNs
    assert_series_equal(target_passed_to_estimator, y_series.iloc[1:])

    # Check predict encodes target
    mock_encode.reset_mock()
    pl.predict(X, y_encoded)
    mock_encode.assert_called_once()

    # Check predict proba encodes target
    mock_encode.reset_mock()
    pl.predict_proba(X, y_encoded)
    mock_encode.assert_called_once()

    # Check score encodes target
    mock_encode.reset_mock()
    pl.score(X, y_encoded, objectives=['MCC Binary'])
    mock_encode.assert_called_once()


@pytest.mark.parametrize("pipeline_class,objectives", [(TimeSeriesBinaryClassificationPipeline, ["MCC Binary"]),
                                                       (TimeSeriesBinaryClassificationPipeline, ["Log Loss Binary"]),
                                                       (TimeSeriesBinaryClassificationPipeline, ["MCC Binary", "Log Loss Binary"]),
                                                       (TimeSeriesMulticlassClassificationPipeline, ["MCC Multiclass"]),
                                                       (TimeSeriesMulticlassClassificationPipeline, ["Log Loss Multiclass"]),
                                                       (TimeSeriesMulticlassClassificationPipeline, ["MCC Multiclass", "Log Loss Multiclass"]),
                                                       (TimeSeriesRegressionPipeline, ['R2']),
                                                       (TimeSeriesRegressionPipeline, ['R2', "Mean Absolute Percentage Error"])])
@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_score_works(pipeline_class, objectives, data_type, X_y_binary, X_y_multi, X_y_regression, make_data_type):

    preprocessing = ['Delayed Feature Transformer']
    if pipeline_class == TimeSeriesRegressionPipeline:
        components = preprocessing + ['Random Forest Regressor']
    else:
        components = preprocessing + ["Logistic Regression Classifier"]

    pl = pipeline_class(component_graph=components,
                        parameters={"pipeline": {"date_index": None, "gap": 1, "max_delay": 2, "delay_features": False},
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

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    pl.fit(X, y)
    if expected_unique_values:
        # NaNs are expected because of padding due to max_delay
        assert set(pl.predict(X, y).to_series().dropna().unique()) == expected_unique_values
    pl.score(X, y, objectives)


@patch('evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets')
@patch('evalml.objectives.BinaryClassificationObjective.decision_function')
@patch('evalml.pipelines.components.Estimator.predict_proba', return_value=ww.DataTable(pd.DataFrame({0: [1.]})))
@patch('evalml.pipelines.components.Estimator.predict', return_value=ww.DataColumn(pd.Series([1.])))
def test_binary_classification_predictions_thresholded_properly(mock_predict, mock_predict_proba,
                                                                mock_obj_decision, mock_decode,
                                                                X_y_binary, dummy_ts_binary_pipeline_class):
    mock_objs = [mock_decode, mock_predict]
    mock_decode.return_value = pd.Series([0, 1])
    X, y = X_y_binary
    binary_pipeline = dummy_ts_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                 "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})
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
    mock_predict_proba.return_value = ww.DataTable(pd.DataFrame([[0.1, 0.2], [0.1, 0.2]]))
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
    mock_obj_decision.return_value = pd.Series([1.])
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
                                                                 "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})
    binary_pipeline.fit(X, y)
    with pytest.raises(ValueError, match="Objective Precision Micro is not defined for time series binary classification."):
        binary_pipeline.predict(X, y, "precision micro")
    mock_transform.assert_called()


@pytest.mark.parametrize("problem_type", [ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS, ProblemTypes.TIME_SERIES_REGRESSION])
def test_time_series_pipeline_not_fitted_error(problem_type, X_y_binary, X_y_multi, X_y_regression,
                                               time_series_binary_classification_pipeline_class,
                                               time_series_multiclass_classification_pipeline_class,
                                               time_series_regression_pipeline_class):
    if problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X, y = X_y_binary
        clf = time_series_binary_classification_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                           "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})

    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = X_y_multi
        clf = time_series_multiclass_classification_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                               "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        X, y = X_y_regression
        clf = time_series_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1},
                                                                "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})

    with pytest.raises(PipelineNotYetFittedError):
        clf.predict(X)
    with pytest.raises(PipelineNotYetFittedError):
        clf.feature_importance

    if is_classification(problem_type):
        with pytest.raises(PipelineNotYetFittedError):
            clf.predict_proba(X)

    clf.fit(X, y)

    if is_classification(problem_type):
        to_patch = 'evalml.pipelines.TimeSeriesClassificationPipeline._predict'
        if problem_type == ProblemTypes.TIME_SERIES_BINARY:
            to_patch = 'evalml.pipelines.TimeSeriesBinaryClassificationPipeline._predict'
        with patch(to_patch) as mock_predict:
            clf.predict(X, y)
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs['objective'] is None

            mock_predict.reset_mock()
            clf.predict(X, y, 'Log Loss Binary')
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs['objective'] is not None

            mock_predict.reset_mock()
            clf.predict(X, y, objective='Log Loss Binary')
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs['objective'] is not None

            clf.predict_proba(X, y)
    else:
        clf.predict(X, y)
    clf.feature_importance


def test_ts_binary_pipeline_target_thresholding(make_data_type, time_series_binary_classification_pipeline_class, X_y_binary):
    X, y = X_y_binary
    X = make_data_type('ww', X)
    y = make_data_type('ww', y)
    objective = get_objective("F1", return_instance=True)

    binary_pipeline = time_series_binary_classification_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                                   "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})
    binary_pipeline.fit(X, y)
    assert binary_pipeline.threshold is None
    pred_proba = binary_pipeline.predict_proba(X, y).iloc[:, 1]
    binary_pipeline.optimize_threshold(X, y, pred_proba, objective)
    assert binary_pipeline.threshold is not None


@patch('evalml.objectives.FraudCost.decision_function')
def test_binary_predict_pipeline_use_objective(mock_decision_function, X_y_binary, time_series_binary_classification_pipeline_class):
    X, y = X_y_binary
    binary_pipeline = time_series_binary_classification_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1},
                                                                                   "pipeline": {"gap": 0, "max_delay": 0, "date_index": None}})
    mock_decision_function.return_value = pd.Series([0] * 98)
    binary_pipeline.threshold = 0.7
    binary_pipeline.fit(X, y)
    fraud_cost = FraudCost(amount_col=0)
    binary_pipeline.score(X, y, ['precision', 'auc', fraud_cost])
    mock_decision_function.assert_called()
