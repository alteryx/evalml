import numpy as np
from catboost import CatBoostRegressor as CBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import R2
from evalml.pipelines import CatBoostRegressionPipeline


def test_catboost_init():
    clf = CatBoostRegressionPipeline(impute_strategy='mean', n_estimators=1000, number_features=0,
                                     bootstrap_type='Bayesian', eta=0.03, max_depth=6, random_state=2)
    expected_parameters = {'impute_strategy': 'mean', 'eta': 0.03, 'n_estimators': 1000, 'max_depth': 6, 'bootstrap_type': 'Bayesian'}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 2


def test_catboost_regression(X_y_reg):
    X, y = X_y_reg

    imputer = SimpleImputer(strategy='mean')
    estimator = CBRegressor(n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type='Bayesian', allow_writing_files=False, random_state=0)
    sk_pipeline = Pipeline([("imputer", imputer),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = R2()
    clf = CatBoostRegressionPipeline(n_estimators=1000, eta=0.03, number_features=X.shape[1],
                                     bootstrap_type='Bayesian', max_depth=6, impute_strategy='mean', random_state=0)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, sk_pipeline.predict(X), decimal=5)
    np.testing.assert_almost_equal(sk_score, clf_scores[objective.name], decimal=5)

    # testing objective parameter passed in does not change results
    clf.fit(X, y, objective)
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)

    # testing objective parameter passed in does not change results
    clf.fit(X, y, objective)
    y_pred_with_objective = clf.predict(X, objective)
    np.testing.assert_almost_equal(y_pred, y_pred_with_objective, decimal=5)


def test_cbr_input_feature_names(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    objective = R2()
    clf = CatBoostRegressionPipeline(impute_strategy='most_frequent', n_estimators=1000,
                                     number_features=len(X.columns), bootstrap_type='Bayesian',
                                     eta=0.03, max_depth=6, random_state=0)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
