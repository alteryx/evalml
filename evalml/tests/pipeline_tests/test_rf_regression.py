import category_encoders as ce
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import RFRegressionPipeline


def test_rf_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression

    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = RandomForestRegressor(random_state=0,
                                      n_estimators=10,
                                      max_depth=3,
                                      n_jobs=-1)

    feature_selection = SelectFromModel(
        estimator=estimator,
        max_features=max(1, int(1 * X.shape[1])),
        threshold=-np.inf
    )

    sk_pipeline = Pipeline(
        [("encoder", enc),
         ("imputer", imputer),
         ("feature_selection", feature_selection),
         ("estimator", estimator)])

    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    clf = RFRegressionPipeline(objective=objective, n_estimators=10, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=X.shape[1])
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_score[0], decimal=5)
