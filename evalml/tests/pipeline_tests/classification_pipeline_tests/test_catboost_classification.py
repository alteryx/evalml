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
    clf = CatBoostBinaryClassificationPipeline(impute_strategy='most_frequent', n_estimators=500,
                                               bootstrap_type='Bernoulli', eta=0.1, number_features=0, max_depth=3, random_state=2)
    expected_parameters = {'impute_strategy': 'most_frequent', 'eta': 0.1, 'n_estimators': 500, 'max_depth': 3, 'bootstrap_type': 'Bernoulli'}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 2


def test_catboost_multi(X_y_multi):
    X, y = X_y_multi

    imputer = SimpleImputer(strategy='mean')
    estimator = CBClassifier(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', allow_writing_files=False, random_state=0)
    sk_pipeline = Pipeline([("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    clf = CatBoostMulticlassClassificationPipeline(impute_strategy='mean', n_estimators=1000, bootstrap_type='Bayesian',
                                                   number_features=X.shape[1], eta=0.03, max_depth=6, random_state=0)
    clf.fit(X, y)
    clf_score = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_score[0])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()

    # testing objective parameter passed in does not change results
    clf.fit(X, y, objective)
    y_pred_with_objective = clf.predict(X)
    assert((y_pred == y_pred_with_objective).all())


def test_catboost_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = PrecisionMicro()
    clf = CatBoostBinaryClassificationPipeline(impute_strategy='mean', n_estimators=1000, eta=0.03,
                                               bootstrap_type='Bayesian', number_features=len(X.columns), max_depth=6, random_state=0)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name


def test_catboost_categorical(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    objective = PrecisionMicro()
    clf = CatBoostBinaryClassificationPipeline(impute_strategy='most_frequent',
                                               number_features=len(X.columns), n_estimators=1000, eta=0.03, max_depth=6, random_state=0)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
