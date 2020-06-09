import numpy as np
from pytest import importorskip, raises
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError

from evalml.objectives import R2
from evalml.pipelines import CatBoostRegressionPipeline
from evalml.pipelines.components import CatBoostRegressor
from evalml.utils import get_random_seed, get_random_state

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
    clf = CatBoostRegressionPipeline(parameters=parameters, random_state=2)
    assert clf.parameters == parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])
    assert clf.summary == 'CatBoost Regressor w/ Simple Imputer'


def test_summary():
    assert CatBoostRegressionPipeline.summary == 'CatBoost Regressor w/ Simple Imputer'


def test_catboost_regression(X_y_reg):
    from catboost import CatBoostRegressor as CBRegressor
    X, y = X_y_reg

    random_seed = 42
    catboost_random_seed = get_random_seed(get_random_state(random_seed), min_bound=CatBoostRegressor.SEED_MIN, max_bound=CatBoostRegressor.SEED_MAX)
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
    clf = CatBoostRegressionPipeline(parameters=parameters, random_state=get_random_state(random_seed))
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


def test_clone(X_y_reg):
    X, y = X_y_reg
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
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone(learned=False)
    assert isinstance(clf_clone, CatBoostRegressionPipeline)
    assert clf.random_state == clf_clone.random_state
    assert clf_clone.component_graph[-1].parameters['eta'] == 0.03
    with raises(NotFittedError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)

    # Test learned clone
    clf_clone = clf.clone()
    assert isinstance(clf_clone, CatBoostRegressionPipeline)
    assert clf_clone.component_graph[-1].parameters['max_depth'] == 6
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t, X_t_clone)
