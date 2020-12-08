import numpy as np
import pandas as pd
import pytest

from evalml.objectives.standard_metrics import AUC, F1


def test_optimize_threshold():
    ypred_proba = np.array([0.2, 0.4])
    y_true = np.array([0, 1])
    obj = F1()
    np.random.seed(42)  # unfortunately scipy.optimize.minimize_scalar has no ability to accept seed as input
    threshold = obj.optimize_threshold(ypred_proba, y_true)
    assert 0.2 < threshold and threshold < 0.4


def test_optimize_threshold_neg():
    ypred_proba = np.array([0.2, 0.4])
    y_true = np.array([0, 1])
    obj = AUC()
    np.random.seed(0)
    with pytest.raises(RuntimeError, match="Trying to optimize objective that can't be optimized!"):
        obj.optimize_threshold(ypred_proba, y_true)


def test_can_optimize_threshold():
    assert F1().can_optimize_threshold
    assert not AUC().can_optimize_threshold


def test_decision_function():
    ypred_proba = np.arange(6) / 5.0
    obj = F1()
    pd.testing.assert_series_equal(obj.decision_function(ypred_proba),
                                   pd.Series(np.array([0] * 3 + [1] * 3), dtype=bool))
    pd.testing.assert_series_equal(obj.decision_function(ypred_proba, threshold=0.5),
                                   pd.Series(np.array([0] * 3 + [1] * 3), dtype=bool))
    pd.testing.assert_series_equal(obj.decision_function(ypred_proba, threshold=0.0),
                                   pd.Series(np.array([0] + [1] * 5, dtype=int), dtype=bool))
    pd.testing.assert_series_equal(obj.decision_function(ypred_proba, threshold=1.0),
                                   pd.Series(np.array([0] * 6, dtype=int), dtype=bool))


def test_decision_function_neg():
    ypred_proba = np.arange(6) / 5.0
    y_true = pd.Series(np.array([0] * 3 + [1] * 3), dtype=bool)
    obj = F1()
    pd.testing.assert_series_equal(obj.decision_function(ypred_proba), y_true)
    pd.testing.assert_series_equal(obj.decision_function(pd.Series(ypred_proba, dtype=float)), y_true)
