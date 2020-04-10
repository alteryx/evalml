import numpy as np
import pytest

from evalml.exceptions import DimensionMismatchError
from evalml.objectives import (
    F1,
    AccuracyBinary,
    BalancedAccuracyBinary,
    BalancedAccuracyMulticlass,
    Precision
)


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


def test_binary_accuracy_binary():
    baccb = BalancedAccuracyBinary()
    assert baccb.score(np.array([0, 1, 0, 0, 1, 0]),
                       np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.625, 1e-5)

    assert baccb.score(np.array([0, 1, 0, 0, 1, 0]),
                       np.array([0, 1, 0, 0, 1, 0])) == 1.000

    assert baccb.score(np.array([0, 1, 0, 0, 1, 0]),
                       np.array([1, 0, 1, 1, 0, 1])) == 0.000


def test_binary_accuracy_multi():
    baccm = BalancedAccuracyMulticlass()
    assert baccm.score(np.array([0, 0, 2, 0, 0, 2, 3]),
                       np.array([0, 1, 2, 0, 1, 2, 3])) == pytest.approx(0.75, 1e-5)

    assert baccm.score(np.array([0, 1, 2, 0, 1, 2, 3]),
                       np.array([0, 1, 2, 0, 1, 2, 3])) == 1.000

    assert baccm.score(np.array([1, 0, 3, 1, 2, 1, 0]),
                       np.array([0, 1, 2, 0, 1, 2, 3])) == 0.000


def test_f1():
    f1 = F1()
    assert f1.score(np.array([0, 1, 0, 0, 1, 0]),
                    np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.5, 1e-5)

    assert f1.score(np.array([0, 1, 0, 0, 1, 1]),
                    np.array([0, 1, 0, 0, 1, 1])) == 1.000

    assert f1.score(np.array([0, 0, 0, 0, 1, 0]),
                    np.array([0, 1, 0, 0, 0, 1])) == 0.000

    assert f1.score(np.array([0, 0]),
                    np.array([0, 0])) == 0.0


def test_precision():
    precision = Precision()
    assert precision.score(np.array([0, 1, 0, 0, 1, 0]),
                           np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.5, 1e-5)

    assert precision.score(np.array([0, 1, 0, 0, 1, 1]),
                           np.array([0, 1, 0, 0, 1, 1])) == 1.000

    assert precision.score(np.array([0, 0, 0, 0, 1, 0]),
                           np.array([0, 1, 0, 0, 0, 1])) == 0.000

    assert precision.score(np.array([0, 0]),
                           np.array([1, 1])) == 0.0
