import numpy as np
import pandas as pd
from catboost import CatBoostClassifier as CBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import PrecisionMicro
from evalml.pipelines import CatBoostClassificationPipeline


def test_catboost_init():
    objective = PrecisionMicro()
    clf = CatBoostClassificationPipeline(objective=objective, impute_strategy='mean', n_estimators=1000,
                                         bootstrap_type='Bayesian', eta=0.03, number_features=0, max_depth=6)
    expected_parameters = {'impute_strategy': 'mean', 'eta': 0.03, 'n_estimators': 1000, 'max_depth': 6, 'bootstrap_type': 'Bayesian'}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 0


def test_catboost_multi(X_y_multi):
    X, y = X_y_multi

    imputer = SimpleImputer(strategy='mean')
    estimator = CBClassifier(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', random_state=0)
    sk_pipeline = Pipeline([("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    clf = CatBoostClassificationPipeline(objective=objective, impute_strategy='mean', n_estimators=1000, bootstrap_type='Bayesian',
                                         number_features=X.shape[1], eta=0.03, max_depth=6, random_state=0)
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
    clf = CatBoostClassificationPipeline(objective=objective, impute_strategy='mean', n_estimators=1000, eta=0.03,
                                         bootstrap_type='Bayesian', number_features=len(X.columns), max_depth=6)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name


def test_catboost_categorical(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    objective = PrecisionMicro()
    clf = CatBoostClassificationPipeline(objective=objective, impute_strategy='most_frequent',
                                         number_features=len(X.columns), bootstrap_type='Bayesian',
                                         n_estimators=1000, eta=0.03, max_depth=6)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
