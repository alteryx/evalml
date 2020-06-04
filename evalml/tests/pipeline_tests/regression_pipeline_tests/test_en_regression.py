from unittest.mock import patch

import numpy as np
import pytest

from evalml.pipelines import ENRegressionPipeline


@pytest.fixture
def dummy_en_regression_pipeline_class(dummy_regressor_estimator_class):
    MockRegressor = dummy_regressor_estimator_class

    class MockENRegressionPipeline(ENRegressionPipeline):
        estimator = MockRegressor
        component_graph = [MockRegressor()]

    return MockENRegressionPipeline


@patch('evalml.pipelines.components.Estimator.predict')
def test_en_regression_pipeline_predict(mock_predict, X_y, dummy_en_regression_pipeline_class):
    X, y = X_y
    multi_pipeline = dummy_en_regression_pipeline_class(parameters={})
    multi_pipeline.predict(X)
    mock_predict.assert_called()
    mock_predict.reset_mock()


def test_en_init(X_y_reg):
    X, y = X_y_reg

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10, 'categories': 'auto'},
        'Elastic Net Regressor': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
        }
    }
    clf = ENRegressionPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'categories': 'auto',
            'drop': None,
            'handle_unknown': 'ignore',
            'handle_missing': 'ignore'},
        'Elastic Net Regressor': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Elastic Net Regressor w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert ENRegressionPipeline.summary == 'Elastic Net Regressor w/ One Hot Encoder + Simple Imputer'
