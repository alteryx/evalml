from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.objectives import PrecisionMicro
from evalml.pipelines import (
    ETBinaryClassificationPipeline,
    ETMulticlassClassificationPipeline
)


def make_mock_et_binary_pipeline():
    class MockETBinaryPipeline(ETBinaryClassificationPipeline):
        component_graph = ['Extra Trees Classifier']

    return MockETBinaryPipeline({})


def make_mock_et_multiclass_pipeline():
    class MockETMulticlassPipeline(ETMulticlassClassificationPipeline):
        component_graph = ['Extra Trees Classifier']

    return MockETMulticlassPipeline({})


def test_et_init(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Extra Trees Classifier': {
            "n_estimators": 20,
            "max_features": "auto",
        }
    }
    clf = ETBinaryClassificationPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Extra Trees Classifier': {
            'max_depth': 6,
            'max_features': "auto",
            'n_estimators': 20,
            "min_samples_split": 2,
            "min_weight_fraction_leaf": 0.0,
            "n_jobs": -1
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Extra Trees Classifier w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert ETBinaryClassificationPipeline.summary == 'Extra Trees Classifier w/ One Hot Encoder + Simple Imputer'


def test_et_objective_type(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Extra Trees Classifier': {
            "n_estimators": 20,
            "max_features": "log2"
        }
    }
    clf = ETBinaryClassificationPipeline(parameters=parameters)
    clf.fit(X, y)

    objective = PrecisionMicro()
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        clf.predict(X, objective)


@patch('evalml.pipelines.classification.ETBinaryClassificationPipeline.fit')
@patch('evalml.pipelines.classification.ETBinaryClassificationPipeline.predict')
def test_et_binary_score(mock_predict, mock_fit, X_y):
    X, y = X_y

    mock_predict.return_value = y
    clf = make_mock_et_binary_pipeline()
    clf.fit(X, y)

    objective_names = ['F1']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()

    assert scores == {'F1': 1.0}


@patch('evalml.pipelines.classification.ETMulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.classification.ETMulticlassClassificationPipeline.predict')
def test_et_multiclass_score(mock_predict, mock_fit, X_y_multi):
    X, y = X_y_multi

    mock_predict.return_value = y
    clf = make_mock_et_multiclass_pipeline()
    clf.fit(X, y)

    objective_names = ['f1_micro']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()

    assert scores == {'F1 Micro': 1.0}


def test_et_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Extra Trees Classifier': {
            "n_estimators": 20,
            "max_features": "auto",
        }
    }

    clf = ETBinaryClassificationPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name


def test_clone_binary(X_y):
    X, y = X_y
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Extra Trees Classifier': {
            "n_estimators": 10,
            "max_features": "log2"
        }
    }
    clf = ETBinaryClassificationPipeline(parameters=parameters, random_state=9)
    clf.fit(X, y)
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone(random_state=9)
    assert clf_clone.estimator.parameters['n_estimators'] == 10
    with pytest.raises(RuntimeError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)

    # Test learned clone
    clf_clone = clf.clone(deep=True)
    assert clf_clone.estimator.parameters['n_estimators'] == 10
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)


def test_clone_multiclass(X_y_multi):
    X, y = X_y_multi
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Extra Trees Classifier': {
            "n_estimators": 12,
            "max_features": "log2"
        }
    }
    clf = ETMulticlassClassificationPipeline(parameters=parameters, random_state=43)
    clf.fit(X, y)
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone(random_state=43)
    assert clf_clone.estimator.parameters['n_estimators'] == 12
    with pytest.raises(RuntimeError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)

    # Test learned clone
    clf = ETMulticlassClassificationPipeline(parameters=parameters, random_state=43)
    clf_clone = clf.clone(deep=True, random_state=43)
    assert clf_clone.estimator.parameters['n_estimators'] == 12
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)
