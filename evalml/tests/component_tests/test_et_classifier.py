import numpy as np
from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines import ExtraTreesClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ExtraTreesClassifier.model_family == ModelFamily.EXTRA_TREES


def test_problem_types():
    assert set(ExtraTreesClassifier.supported_problem_types) == {ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                                                 ProblemTypes.TIME_SERIES_BINARY,
                                                                 ProblemTypes.TIME_SERIES_MULTICLASS}


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    sk_clf = SKExtraTreesClassifier(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred.to_series().values, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba.to_dataframe().values, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = SKExtraTreesClassifier(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ExtraTreesClassifier()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, ExtraTreesClassifier)

    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred.to_series().values, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba.to_dataframe().values, decimal=5)


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    clf = ExtraTreesClassifier(n_jobs=1)
    sk_clf = SKExtraTreesClassifier(max_depth=6, random_state=0, n_jobs=1)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
