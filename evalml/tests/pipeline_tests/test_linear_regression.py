import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evalml.objectives import R2
from evalml.pipelines import LinearRegressionPipeline


def test_lr_init(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    objective = R2()
    clf = LinearRegressionPipeline(objective=objective, number_features=len(X.columns), random_state=2, impute_strategy='mean', normalize=True, fit_intercept=True, n_jobs=-1)
    expected_parameters = {'impute_strategy': 'mean', 'fit_intercept': True, 'normalize': True}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 2


def test_linear_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    estimator = LinearRegression(normalize=False, fit_intercept=True, n_jobs=-1)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("scaler", scaler),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    clf = LinearRegressionPipeline(objective=objective,
                                   number_features=len(X.columns),
                                   impute_strategy='mean',
                                   normalize=False,
                                   fit_intercept=True,
                                   random_state=0,
                                   n_jobs=-1)
    clf.fit(X, y, objective)
    clf_score = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_score[0], decimal=5)
    assert not clf.feature_importances.isnull().all().all()


def test_lr_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = R2()
    clf = LinearRegressionPipeline(objective=objective, number_features=len(X.columns), random_state=0, impute_strategy='mean', normalize=False, fit_intercept=True, n_jobs=-1)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
