import numpy as np
import pandas as pd
from pytest import importorskip
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import PrecisionMicro
from evalml.pipelines import CatBoostClassificationPipeline
from evalml.pipelines.components import CatBoostClassifier
from evalml.utils import get_random_seed, get_random_state

importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_init():
    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent',
            'fill_value': None
        },
        'CatBoost Classifier': {
            "n_estimators": 500,
            "bootstrap_type": 'Bernoulli',
            "eta": 0.1,
            "max_depth": 3,
        }
    }
    clf = CatBoostClassificationPipeline(objective=objective, parameters=parameters, random_state=2)

    assert clf.parameters == parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])


def test_catboost_multi(X_y_multi):
    from catboost import CatBoostClassifier as CBClassifier
    X, y = X_y_multi

    random_seed = 42
    catboost_random_seed = get_random_seed(get_random_state(random_seed), min_bound=CatBoostClassifier.SEED_MIN, max_bound=CatBoostClassifier.SEED_MAX)
    imputer = SimpleImputer(strategy='mean')
    estimator = CBClassifier(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', allow_writing_files=False, random_seed=catboost_random_seed)
    sk_pipeline = Pipeline([("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'CatBoost Classifier': {
            "n_estimators": 500,
            "bootstrap_type": 'Bernoulli',
            "eta": 0.1,
            "max_depth": 3,
        }
    }

    clf = CatBoostClassificationPipeline(objective=objective, parameters=parameters, random_state=get_random_state(random_seed))
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)

    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_score[0])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()


def test_catboost_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'CatBoost Classifier': {
            "n_estimators": 1000,
            "bootstrap_type": 'Bernoulli',
            "eta": 0.03,
            "max_depth": 6,
        }
    }
    clf = CatBoostClassificationPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name


def test_catboost_categorical(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'CatBoost Classifier': {
            "n_estimators": 500,
            "bootstrap_type": 'Bernoulli',
            "eta": 0.1,
            "max_depth": 3,
        }
    }
    clf = CatBoostClassificationPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
