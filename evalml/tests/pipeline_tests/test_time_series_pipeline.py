from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines import TimeSeriesRegressionPipeline
from evalml.pipelines.components.transformers.transformer import Transformer


@pytest.fixture
def ts_data():
    X, y = pd.DataFrame({"features": range(101, 132)}), pd.Series(range(1, 32))
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")
    return X, y


class MockDelayedFeatures(Transformer):
    name = "Delayed Features Transformer"
    needs_fitting = False

    def __init__(self, max_delay=2, random_state=0, **kwargs):

        parameters = {"max_delay": max_delay}
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_state=random_state)
        self.max_delay = max_delay

    def fit(self, X, y=None):
        """Fits the LaggedFeatureExtractor."""

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        original_columns = X.columns
        X = X.assign(col_with_nans=X.iloc[:, 0].shift(self.max_delay))

        X.drop(columns=original_columns, inplace=True)

        return X


@pytest.mark.parametrize("pipeline_class", [TimeSeriesRegressionPipeline])
@pytest.mark.parametrize("components", [["One Hot Encoder"],
                                        [MockDelayedFeatures, "One Hot Encoder"]])
def test_time_series_pipeline_init(pipeline_class, components):

    class Pipeline(pipeline_class):
        component_graph = components + ["Random Forest Regressor"]

    if MockDelayedFeatures not in components:
        pl = Pipeline({}, gap=3, max_delay=5)
        assert "Delayed Features Transformer" not in pl.parameters
    else:
        parameters = {"Delayed Features Transformer": {"gap": 3, "max_delay": 5}}
        pl = Pipeline(parameters, gap=3, max_delay=5)
        assert pl.parameters['Delayed Features Transformer'] == {"gap": 3, "max_delay": 5}


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_lagged_features", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
def test_fit_drop_nans_before_estimator(mock_regressor_fit, pipeline_class,
                                        estimator_name, gap, max_delay, include_lagged_features, only_use_y, ts_data):

    X, y = ts_data

    if include_lagged_features:
        components = [MockDelayedFeatures, estimator_name]
        train_index = pd.date_range(f"2020-10-{1 + max_delay}", f"2020-10-{31-gap}")
        expected_target = np.arange(1 + gap + max_delay, 32)
    else:
        components = [estimator_name]
        train_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")
        expected_target = np.arange(1 + gap, 32)

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({"Delayed Features Transformer": {"gap": gap, "max_delay": max_delay}}, gap=gap, max_delay=max_delay)

    if only_use_y:
        pl.fit(y)
    else:
        pl.fit(X, y)

    df_passed_to_estimator, target_passed_to_estimator = mock_regressor_fit.call_args[0]

    # NaNs introduced by shifting are dropped
    assert not df_passed_to_estimator.isna().any(axis=1).any()
    assert not target_passed_to_estimator.isna().any()

    # Check the estimator was trained on the expected dates
    pd.testing.assert_index_equal(df_passed_to_estimator.index, train_index)
    np.testing.assert_equal(target_passed_to_estimator.values, expected_target)


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_lagged_features", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
def test_predict_pad_nans(mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, include_lagged_features, only_use_y, ts_data):

    X, y = ts_data

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    mock_regressor_predict.side_effect = mock_predict

    if include_lagged_features:
        components = [MockDelayedFeatures, estimator_name]
    else:
        components = [estimator_name]

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({"Delayed Features Transformer": {"gap": gap, "max_delay": max_delay}}, gap=gap, max_delay=max_delay)

    if only_use_y:
        pl.fit(y)
        preds = pl.predict(y)
    else:
        pl.fit(X, y)
        preds = pl.predict(X, y)

    # Check that the predictions have NaNs for the first n_delay dates
    if include_lagged_features:
        assert np.isnan(preds.values[:max_delay]).all()
    else:
        assert not np.isnan(preds.values).any()


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("include_lagged_features", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(1, 2), (2, 2), (7, 3), (2, 4)])
@pytest.mark.parametrize("pipeline_class,estimator_name", [(TimeSeriesRegressionPipeline, "Random Forest Regressor")])
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.predict")
@patch("evalml.pipelines.RegressionPipeline._score_all_objectives")
def test_score_drops_nans(mock_score, mock_regressor_predict, mock_regressor_fit,
                          pipeline_class,
                          estimator_name, gap, max_delay, include_lagged_features, only_use_y, ts_data):

    X, y = ts_data

    def mock_predict(df):
        return pd.Series(range(200, 200 + df.shape[0]))

    mock_regressor_predict.side_effect = mock_predict

    if include_lagged_features:
        components = [MockDelayedFeatures, estimator_name]
        expected_target = np.arange(1 + gap + max_delay, 32)
        target_index = pd.date_range(f"2020-10-{1 + max_delay}", f"2020-10-{31 - gap}")
    else:
        components = [estimator_name]
        expected_target = np.arange(1 + gap, 32)
        target_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")

    class Pipeline(pipeline_class):
        component_graph = components

    pl = Pipeline({"Delayed Features Transformer": {"gap": gap, "max_delay": max_delay}}, gap=gap, max_delay=max_delay)

    if only_use_y:
        pl.fit(y)
        pl.score(y, y=None, objectives=[])
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
