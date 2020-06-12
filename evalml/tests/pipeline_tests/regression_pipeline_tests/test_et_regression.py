from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines import ETRegressionPipeline


def make_mock_et_regression_pipeline():
    class MockETRegressionPipeline(ETRegressionPipeline):
        component_graph = ['Extra Trees Regressor']

    return MockETRegressionPipeline({})


def test_et_init(X_y_reg):
    X, y = X_y_reg

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Extra Trees Regressor': {
            "n_estimators": 20,
            "max_features": "auto",
            "max_depth": 6
        }
    }
    clf = ETRegressionPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Extra Trees Regressor': {
            'max_features': "auto",
            'n_estimators': 20,
            'max_depth': 6,
            "min_samples_split": 2,
            "min_weight_fraction_leaf": 0.0,
            "n_jobs": -1
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Extra Trees Regressor w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert ETRegressionPipeline.summary == 'Extra Trees Regressor w/ One Hot Encoder + Simple Imputer'


@patch('evalml.pipelines.regression.ETRegressionPipeline.fit')
@patch('evalml.pipelines.regression.ETRegressionPipeline.predict')
def test_et_score(mock_predict, mock_fit, X_y):
    X, y = X_y

    mock_predict.return_value = y
    clf = make_mock_et_regression_pipeline()
    clf.fit(X, y)

    objective_names = ['r2']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()

    assert scores == {'R2': 1.0}


def test_et_input_feature_names(X_y_reg):
    X, y = X_y_reg
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Extra Trees Regressor': {
            "n_estimators": 20,
            "max_features": "auto",
        }
    }
    clf = ETRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name


def test_clone(X_y_reg):
    X, y = X_y_reg
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Extra Trees Regressor': {
            "n_estimators": 15,
            "max_features": "auto",
            "max_depth": 6
        }
    }
    clf = ETRegressionPipeline(parameters=parameters, random_state=4)
    clf.fit(X, y)
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone(random_state=4)
    assert isinstance(clf_clone, ETRegressionPipeline)
    assert clf_clone.estimator.parameters['n_estimators'] == 15
    assert clf_clone.component_graph[1].parameters['impute_strategy'] == "mean"
    with pytest.raises(RuntimeError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)

    # Test learned clone
    clf_clone = clf.clone(deep=True, random_state=4)
    assert isinstance(clf_clone, ETRegressionPipeline)
    assert clf_clone.estimator.parameters['n_estimators'] == 15
    assert clf_clone.component_graph[1].parameters['impute_strategy'] == "mean"
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)
