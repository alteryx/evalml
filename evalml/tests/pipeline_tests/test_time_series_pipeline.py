from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines import TimeSeriesRegressionPipeline


@pytest.fixture
def ts_data():
    X, y = pd.DataFrame({"features": range(101, 132)}), pd.Series(range(1, 32))
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")
    return X, y


@pytest.mark.parametrize("pipeline_class", [TimeSeriesRegressionPipeline])
@pytest.mark.parametrize("components", [["One Hot Encoder"],
                                        ["Delayed Feature Transformer", "One Hot Encoder"]])
def test_time_series_pipeline_init(pipeline_class, components):

    class Pipeline(pipeline_class):
        component_graph = components + ["Random Forest Regressor"]

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
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
def test_fit_drop_nans_before_estimator(mock_regressor_fit, pipeline_class,
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

    df_passed_to_estimator, target_passed_to_estimator = mock_regressor_fit.call_args[0]

    # NaNs introduced by shifting are dropped
    assert not df_passed_to_estimator.isna().any(axis=1).any()
    assert not target_passed_to_estimator.isna().any()

    # Check the estimator was trained on the expected dates
    pd.testing.assert_index_equal(df_passed_to_estimator.index, train_index)
    np.testing.assert_equal(target_passed_to_estimator.values, expected_target)


@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
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
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
def test_predict_pad_nans(mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, include_delayed_features, only_use_y, ts_data):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    mock_regressor_predict.side_effect = mock_predict

    class Pipeline(pipeline_class):
        component_graph = ["Delayed Feature Transformer", estimator_name]

    pl = Pipeline({"Delayed Feature Transformer": {"gap": gap, "max_delay": max_delay,
                                                   "delay_features": include_delayed_features,
                                                   "delay_target": include_delayed_features},
                   "pipeline": {"gap": gap, "max_delay": max_delay}})

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
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.RegressionPipeline._score_all_objectives")
def test_score_drops_nans(mock_score, mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, include_delayed_features, only_use_y, ts_data):

    if only_use_y and (not include_delayed_features or (max_delay == 0 and gap == 0)):
        pytest.skip("This would result in an empty feature dataframe.")

    X, y = ts_data

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    mock_regressor_predict.side_effect = mock_predict

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

    if only_use_y:
        pl.fit(None, y)
        pl.score(X=None, y=y, objectives=[])
    else:
        pl.fit(X, y)
        pl.score(X, y, objectives=[])

    # Verify that NaNs are dropped before passed to objectives
    _, target, preds = mock_score.call_args[0]
    assert not target.isna().any()
    assert not preds.isna().any()

    # Target used for scoring matches expected dates
    pd.testing.assert_index_equal(target.index, target_index)
    np.testing.assert_equal(target.values, expected_target)
