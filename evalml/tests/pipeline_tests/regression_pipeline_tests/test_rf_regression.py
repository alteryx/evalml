import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import RFRegressionPipeline


def test_rf_init(X_y_reg):
    X, y = X_y_reg

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters, random_state=2)
    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Random Forest Regressor': {
            'max_depth': 5,
            'n_estimators': 20,
            'n_jobs': -1
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Random Forest Regressor w/ One Hot Encoder + Simple Imputer'


def test_summary():
    assert RFRegressionPipeline.summary == 'Random Forest Regressor w/ One Hot Encoder + Simple Imputer'


def test_rf_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    imputer = SimpleImputer(strategy='most_frequent')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = RandomForestRegressor(random_state=0,
                                      n_estimators=10,
                                      max_depth=3,
                                      n_jobs=-1)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'Random Forest Regressor': {
            "n_estimators": 10,
            "max_depth": 3,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)
    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)

    # testing objective parameter passed in does not change results
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


def test_rfr_input_feature_names(X_y_reg):
    X, y = X_y_reg
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name


def test_clone(X_y_reg):
    X, y = X_y_reg
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Random Forest Regressor': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFRegressionPipeline(parameters=parameters, random_state=75)
    clf.fit(X, y)
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone(random_state=75)
    assert isinstance(clf_clone, RFRegressionPipeline)
    assert clf_clone.component_graph[-1].parameters['n_estimators'] == 20
    with pytest.raises(RuntimeError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)

    # Test learned clone
    clf_clone = clf.clone(deep=True)
    assert isinstance(clf_clone, RFRegressionPipeline)
    print(clf_clone.estimator.parameters.keys())
    assert clf_clone.estimator.parameters['n_estimators'] == 20
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)
