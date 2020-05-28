from unittest.mock import patch

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
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
        'RF Regressor Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20,
            "max_depth": 5
        },
        'Extra Trees Regressor': {
            "n_estimators": 20,
            "max_features": "auto",
        }
    }
    clf = ETRegressionPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'RF Regressor Select From Model': {
            'percent_features': 1.0,
            'threshold': -np.inf
        },
        'Extra Trees Regressor': {
            'max_features': "auto",
            'n_estimators': 20
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Extra Trees Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model'


def test_summary():
    assert ETRegressionPipeline.summary == 'Extra Trees Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model'


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


def test_etr_input_feature_names(X_y_reg):
    X, y = X_y_reg
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": X.shape[1],
            "n_estimators": 20
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
