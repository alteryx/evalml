import numpy as np
from catboost import CatBoostRegressor as CBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import CatBoostRegressionPipeline


def test_catboost_init():
    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent',
            'fill_value': None
        },
        'CatBoost Regressor': {
            "n_estimators": 1000,
            "bootstrap_type": 'Bayesian',
            "eta": 0.03,
            "max_depth": 6,
        }
    }
    clf = CatBoostRegressionPipeline(objective=objective, parameters=parameters)
    assert clf.parameters == parameters


def test_catboost_regression(X_y_reg):
    X, y = X_y_reg

    imputer = SimpleImputer(strategy='mean')
    estimator = CBRegressor(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', allow_writing_files=False, random_state=0)
    sk_pipeline = Pipeline([("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'CatBoost Regressor': {
            "n_estimators": 1000,
            "bootstrap_type": 'Bayesian',
            "eta": 0.03,
            "max_depth": 6,
        }
    }
    clf = CatBoostRegressionPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_score[0], decimal=5)


def test_cbr_input_feature_names(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    objective = R2()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'CatBoost Regressor': {
            "n_estimators": 1000,
            "bootstrap_type": 'Bayesian',
            "eta": 0.03,
            "max_depth": 6,
        }
    }
    clf = CatBoostRegressionPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
