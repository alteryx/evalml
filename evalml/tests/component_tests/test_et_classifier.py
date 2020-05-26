from unittest.mock import patch

import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import ExtraTreesClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ExtraTreesClassifier.model_family == ModelFamily.EXTRA_TREES


def test_problem_types():
    assert ProblemTypes.BINARY in ExtraTreesClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in ExtraTreesClassifier.supported_problem_types
    assert len(ExtraTreesClassifier.supported_problem_types) == 2


def test_fit_predict_binary(X_y):
    X, y = X_y
    
    sk_clf = SKExtraTreesClassifier()
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi
    
    sk_clf = SKExtraTreesClassifier()
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_feature_importances(X_y):
    X, y = X_y

    sk_clf = SKExtraTreesClassifier(random_state=0)
    sk_clf.fit(X, y)

    clf = ExtraTreesClassifier()
    clf.fit(X, y)

    np.testing.assert_almost_equal(sk_clf.feature_importances_, clf.feature_importances, decimal=5)