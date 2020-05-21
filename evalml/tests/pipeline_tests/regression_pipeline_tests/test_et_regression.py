import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, 
                              RandomForestRegressor as SKRandomForestRegressor)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import ETRegressionPipeline


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
            "max_depth": 5,
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
            'max_depth': 5,
            'n_estimators': 20
        }
    }

    assert clf.parameters == expected_parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'Extra Trees Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model'


def test_summary():
    assert ETRegressionPipeline.summary == 'Extra Trees Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model'


def test_et_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    imputer = SimpleImputer(strategy='most_frequent')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = ExtraTreesRegressor(random_state=0,
                                    n_estimators=10,
                                    max_depth=3,
                                    n_jobs=-1)
    rf_estimator = SKRandomForestRegressor(random_state=0, n_estimators=10, max_depth=3)
    feature_selection = SelectFromModel(estimator=rf_estimator,
                                        max_features=max(1, int(1 * X.shape[1])),
                                        threshold=-np.inf)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("feature_selection", feature_selection),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'RF Regressor Select From Model': {
            "percent_features": 1.0,
            "number_features": X.shape[1],
            "n_estimators": 10,
            "max_depth": 3,
        },
        'Extra Trees Regressor': {
            "n_estimators": 10,
            "max_depth": 3,
        }
    }
    clf = ETRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)
    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)

    # testing objective parameter passed in does not change results
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


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
            "max_depth": 5,
        }
    }
    clf = ETRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
