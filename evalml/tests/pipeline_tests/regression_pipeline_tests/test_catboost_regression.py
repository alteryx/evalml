import numpy as np
from pytest import importorskip
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import CatBoostRegressionPipeline

importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_init():
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
    clf = CatBoostRegressionPipeline(parameters=parameters)
    assert clf.parameters == parameters


def test_catboost_regression(X_y_reg):
    from catboost import CatBoostRegressor as CBRegressor
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
    clf = CatBoostRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)

    # testing objective parameter passed in does not change results
    clf.fit(X, y)
    y_pred_with_objective = clf.predict(X)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


def test_cbr_input_feature_names(X_y_categorical_regression):
    X, y = X_y_categorical_regression
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
    clf = CatBoostRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
