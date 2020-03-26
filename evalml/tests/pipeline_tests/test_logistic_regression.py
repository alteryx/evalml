import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.objectives import PrecisionMicro
from evalml.pipelines import LogisticRegressionPipeline


def test_lor_init(X_y):
    X, y = X_y

    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 0.5,
        }
    }
    clf = LogisticRegressionPipeline(objective=objective, parameters=parameters)
    assert clf.parameters == parameters


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
            'random_state': 1
        }
    }
    clf = LogisticRegressionPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)
    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_score[0])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()


def test_lor_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)

    objective = PrecisionMicro()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 1
        }
    }

    clf = LogisticRegressionPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)

    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
