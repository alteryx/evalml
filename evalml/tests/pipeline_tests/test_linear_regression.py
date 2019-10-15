import category_encoders as ce
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evalml.objectives import R2
from evalml.pipelines import LinearRegressionPipeline


def test_linear_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    scaler = StandardScaler()
    estimator = LinearRegression(normalize=False, fit_intercept=True, n_jobs=-1)

    sk_pipeline = Pipeline(
        [("encoder", enc),
         ("imputer", imputer),
         ("scaler", scaler),
         ("estimator", estimator)])

    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    clf = LinearRegressionPipeline(objective=objective, number_features=len(X.columns), random_state=0, impute_strategy='mean', normalize=False, fit_intercept=True, n_jobs=-1)
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_score[0], decimal=5)
    assert not clf.feature_importances.isnull().all().all()
