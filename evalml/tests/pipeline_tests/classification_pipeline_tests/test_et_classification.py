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


def test_et_init(X_y_binary):
    X, y = X_y_binary

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
        'One Hot Encoder': {
            'top_n': 10,
            'categories': None,
            'drop': None,
            'handle_unknown': 'ignore',
            'handle_missing': 'error'},
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


def test_et_objective_type(X_y_binary):
    X, y = X_y_binary

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
    X, y = X_y_binary

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


def test_et_input_feature_names(X_y_binary):
    X, y = X_y_binary
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
    assert len(clf.feature_importance) == len(X.columns)
    assert not clf.feature_importance.isnull().all().all()
    for col_name in clf.feature_importance["feature"]:
        assert "col_" in col_name
