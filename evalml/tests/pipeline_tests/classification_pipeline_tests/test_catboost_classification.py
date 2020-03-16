import numpy as np
import pandas as pd
from catboost import CatBoostClassifier as CBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import PrecisionMicro
from evalml.pipelines import (
    CatBoostBinaryClassificationPipeline,
    CatBoostMulticlassClassificationPipeline
)


def test_catboost_init():
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
    clf = CatBoostBinaryClassificationPipeline(parameters=parameters)

    assert clf.parameters == parameters


def test_catboost_multi(X_y_multi):
    X, y = X_y_multi

    imputer = SimpleImputer(strategy='mean')
    estimator = CBClassifier(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', allow_writing_files=False, random_state=0)
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

    clf = CatBoostMulticlassClassificationPipeline(parameters=parameters)
    clf.fit(X, y, objective)
    clf_score = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_score[objective.name])
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
    clf = CatBoostBinaryClassificationPipeline(parameters=parameters)
    clf.fit(X, y, objective)
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
    clf = CatBoostBinaryClassificationPipeline(parameters=parameters)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
