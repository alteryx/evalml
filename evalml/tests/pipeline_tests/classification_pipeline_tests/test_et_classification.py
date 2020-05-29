from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.objectives import Precision, PrecisionMicro
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
            'n_estimators': 20
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Extra Trees Classifier w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert ETBinaryClassificationPipeline.summary == 'Extra Trees Classifier w/ One Hot Encoder + Simple Imputer'


def test_et_objective_tuning(X_y):
    X, y = X_y

    # The classifier predicts accurately with perfect confidence given the original data,
    # so some is removed for the setting threshold test to have significance
    X[0] = np.zeros(len(X[0]))
    X[1] = np.zeros(len(X[1]))
    X[2] = np.zeros(len(X[0]))
    X[3] = np.zeros(len(X[1]))

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
    y_pred = clf.predict(X)

    objective = PrecisionMicro()
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        y_pred_with_objective = clf.predict(X, objective)

    # testing objective parameter passed in does not change results
    objective = Precision()
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)

    # testing objective parameter passed and set threshold does change results
    with pytest.raises(AssertionError):
        clf.threshold = 0.01
        y_pred_with_objective = clf.predict(X, objective)
        np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


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
