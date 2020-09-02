
from evalml.model_family import ModelFamily
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert StackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_parameters(all_regression_estimators):
    estimators = [estimator_class() for estimator_class in all_regression_estimators if estimator_class.model_family != ModelFamily.ENSEMBLE]
    clf = StackedEnsembleRegressor(estimators, final_estimator=None, random_state=2)
    expected_parameters = {
        "estimators": estimators,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters


def test_problem_types():
    assert ProblemTypes.REGRESSION in StackedEnsembleRegressor.supported_problem_types
    assert len(StackedEnsembleRegressor.supported_problem_types) == 1


# def test_fit_predict_binary(X_y_binary):
#     X, y = X_y_binary

#     sk_clf = SKElasticNetClassifier(loss="log",
#                                     penalty="elasticnet",
#                                     alpha=0.5,
#                                     l1_ratio=0.5,
#                                     n_jobs=-1,
#                                     random_state=0)
#     sk_clf.fit(X, y)
#     y_pred_sk = sk_clf.predict(X)
#     y_pred_proba_sk = sk_clf.predict_proba(X)

#     clf = ElasticNetClassifier()
#     clf.fit(X, y)
#     y_pred = clf.predict(X)
#     y_pred_proba = clf.predict_proba(X)

#     np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
#     np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


# def test_fit_predict_multi(X_y_multi):
#     X, y = X_y_multi

#     sk_clf = SKElasticNetClassifier(loss="log",
#                                     penalty="elasticnet",
#                                     alpha=0.5,
#                                     l1_ratio=0.5,
#                                     n_jobs=-1,
#                                     random_state=0)
#     sk_clf.fit(X, y)
#     y_pred_sk = sk_clf.predict(X)
#     y_pred_proba_sk = sk_clf.predict_proba(X)

#     clf = ElasticNetClassifier()
#     clf.fit(X, y)
#     y_pred = clf.predict(X)
#     y_pred_proba = clf.predict_proba(X)

#     np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
#     np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


# def test_feature_importance(X_y_binary):
#     X, y = X_y_binary

#     sk_clf = SKElasticNetClassifier(loss="log",
#                                     penalty="elasticnet",
#                                     alpha=0.5,
#                                     l1_ratio=0.5,
#                                     n_jobs=-1,
#                                     random_state=0)
#     sk_clf.fit(X, y)

#     clf = ElasticNetClassifier()
#     clf.fit(X, y)

#     np.testing.assert_almost_equal(sk_clf.coef_.flatten(), clf.feature_importance, decimal=5)


# def test_feature_importance_multi(X_y_multi):
#     X, y = X_y_multi

#     sk_clf = SKElasticNetClassifier(loss="log",
#                                     penalty="elasticnet",
#                                     alpha=0.5,
#                                     l1_ratio=0.5,
#                                     n_jobs=-1,
#                                     random_state=0)
#     sk_clf.fit(X, y)

#     clf = ElasticNetClassifier()
#     clf.fit(X, y)

#     sk_features = np.linalg.norm(sk_clf.coef_, axis=0, ord=2)

#     np.testing.assert_almost_equal(sk_features, clf.feature_importance, decimal=5)
