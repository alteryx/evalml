import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.objectives import Precision, PrecisionMicro
from evalml.pipelines import (
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline
)


def test_lor_init(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 0.5,
        }
    }
    clf = LogisticRegressionBinaryPipeline(parameters=parameters, random_state=1)
    assert clf.parameters == parameters
    assert (clf.random_state.get_state()[0] == np.random.RandomState(1).get_state()[0])
    assert clf.summary == 'Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler'


def test_lor_objective_tuning(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 0.5,
        }
    }
    clf = LogisticRegressionBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    objective = PrecisionMicro()
    with pytest.raises(ValueError, match="You can only use a binary classification objective to make predictions for a binary classification pipeline."):
        y_pred_with_objective = clf.predict(X, objective)

    # testing objective parameter passed in does not change results
    objective = Precision()
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)

    # testing objective parameter passed and set threshold does change results
    with pytest.raises(AssertionError):
        clf.threshold = 0.01
        y_pred_with_objective = clf.predict(X, objective)
        np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


def test_lor_multi(X_y_multi):
    X, y = X_y_multi
    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    scaler = SkScaler()
    estimator = LogisticRegression(random_state=0,
                                   penalty='l2',
                                   C=1.0,
                                   multi_class='auto',
                                   solver="lbfgs",
                                   n_jobs=-1)
    sk_pipeline = SKPipeline([("encoder", enc),
                              ("imputer", imputer),
                              ("scaler", scaler),
                              ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }
    clf = LogisticRegressionMulticlassPipeline(parameters=parameters, random_state=1)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)
    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_scores[objective.name])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()

    # testing objective parameter passed in does not change results
    clf.fit(X, y)
    y_pred_with_objective = clf.predict(X)
    assert((y_pred == y_pred_with_objective).all())


def test_lor_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }
    clf = LogisticRegressionBinaryPipeline(parameters=parameters, random_state=1)
    clf.fit(X, y)

    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
