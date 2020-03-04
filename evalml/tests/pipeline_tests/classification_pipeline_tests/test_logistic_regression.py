import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.objectives import PrecisionMicro
from evalml.pipelines import (
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline
)


def test_lor_init(X_y):
    X, y = X_y

    clf = LogisticRegressionBinaryPipeline(penalty='l2', C=0.5, impute_strategy='mean', number_features=len(X[0]), random_state=1)
    expected_parameters = {'impute_strategy': 'mean', 'penalty': 'l2', 'C': 0.5}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 1


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
    clf = LogisticRegressionMulticlassPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)
    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_scores[objective.name])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()

    # testing objective parameter passed in does not change results
    clf.fit(X, y, objective)
    y_pred_with_objective = clf.predict(X)
    assert((y_pred == y_pred_with_objective).all())


def test_lor_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = PrecisionMicro()
    clf = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X.columns), random_state=0)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
