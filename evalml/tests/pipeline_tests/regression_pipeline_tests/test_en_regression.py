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
        'One Hot Encoder': {'top_n': 10},
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
        'One Hot Encoder': {'top_n': 10},
        'Elastic Net Regressor': {
            "alpha": 0.5,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "normalize": False
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Elastic Net Regressor w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert ENRegressionPipeline.summary == 'Elastic Net Regressor w/ One Hot Encoder + Simple Imputer'


def test_clone(X_y_reg):
    X, y = X_y_reg
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Elastic Net Regressor': {
            "alpha": 0.6,
            "l1_ratio": 0.5,
        }
    }
    clf = ENRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone()
    assert isinstance(clf_clone, ENRegressionPipeline)
    assert clf_clone.estimator.parameters['alpha'] == 0.6
    assert clf_clone.component_graph[1].parameters['impute_strategy'] == "mean"
    with pytest.raises(RuntimeError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)

    # Test learned clone
    clf_clone = clf.clone(deep=True)
    assert isinstance(clf_clone, ENRegressionPipeline)
    assert clf_clone.estimator.parameters['alpha'] == 0.6
    assert clf_clone.component_graph[1].parameters['impute_strategy'] == "mean"
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)
