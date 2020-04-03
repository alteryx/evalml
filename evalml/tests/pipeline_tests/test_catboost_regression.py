import numpy as np
from pytest import importorskip
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import CatBoostRegressionPipeline
from evalml.utils import SEED_BOUNDS, get_random_seed, get_random_state

importorskip('catboost', reason='Skipping test because catboost not installed')


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
    clf = CatBoostRegressionPipeline(objective=objective, parameters=parameters, random_state=2)
    assert clf.parameters == parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])


def test_catboost_regression(X_y_reg):
    from catboost import CatBoostRegressor as CBRegressor
    X, y = X_y_reg

    random_seed = 42
    catboost_random_seed = get_random_seed(get_random_state(random_seed), min_bound=0, max_bound=SEED_BOUNDS.max_bound)
    imputer = SimpleImputer(strategy='mean')
    estimator = CBRegressor(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', allow_writing_files=False, random_seed=catboost_random_seed)
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
    clf = CatBoostRegressionPipeline(objective=objective, parameters=parameters, random_state=get_random_state(random_seed))
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
