import numpy as np
import pytest

from evalml.exceptions import DimensionMismatchError
from evalml.objectives import AccuracyBinary, BalancedAccuracy


def test_accuracy():
    acc = AccuracyBinary()
    assert acc.score(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == pytest.approx(0.0, 1e-5)
    assert acc.score(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(0.5, 1e-5)
    assert acc.score(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == pytest.approx(1.0, 1e-5)

    with pytest.raises(ValueError, match="Length of inputs is 0"):
        acc.score(y_predicted=[], y_true=[1])
    with pytest.raises(ValueError, match="Length of inputs is 0"):
        acc.score(y_predicted=[1], y_true=[])
    with pytest.raises(DimensionMismatchError):
        acc.score(y_predicted=[0], y_true=[1, 0])
    with pytest.raises(DimensionMismatchError):
        acc.score(y_predicted=np.array([0]), y_true=np.array([1, 0]))


def test_binary_accuracy():
    bacc = BalancedAccuracy()
    assert bacc.score(np.array([0, 1, 0, 0, 1, 0]),
                      np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.625, 1e-5)

    assert bacc.score(np.array([0, 1, 0, 0, 1, 0]),
                      np.array([0, 1, 0, 0, 1, 0])) == 1.000

    assert bacc.score(np.array([0, 1, 0, 0, 1, 0]),
                      np.array([1, 0, 1, 1, 0, 1])) == 0.000
