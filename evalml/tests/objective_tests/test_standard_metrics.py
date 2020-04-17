import numpy as np
import pytest

from evalml.exceptions import DimensionMismatchError
from evalml.objectives import (
    F1,
    F1Micro,
    F1Macro,
    F1Weighted,
    Precision,
    PrecisionMicro,
    PrecisionMacro,
    PrecisionWeighted,
    Recall,
    RecallMicro,
    RecallMacro,
    RecallWeighted,
    AccuracyBinary,
    BalancedAccuracyBinary,
    BalancedAccuracyMulticlass,
)


EPS = 1e-5


def test_accuracy_binary():
    obj = AccuracyBinary()
    assert obj.score(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == pytest.approx(0.0, EPS)
    assert obj.score(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(0.5, EPS)
    assert obj.score(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == pytest.approx(1.0, EPS)

    with pytest.raises(ValueError, match="Length of inputs is 0"):
        obj.score(y_predicted=[], y_true=[1])
    with pytest.raises(ValueError, match="Length of inputs is 0"):
        obj.score(y_predicted=[1], y_true=[])
    with pytest.raises(DimensionMismatchError):
        obj.score(y_predicted=[0], y_true=[1, 0])
    with pytest.raises(DimensionMismatchError):
        obj.score(y_predicted=np.array([0]), y_true=np.array([1, 0]))


def test_balanced_accuracy_binary():
    obj = BalancedAccuracyBinary()
    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.625, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 1, 0])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([1, 0, 1, 1, 0, 1])) == pytest.approx(0.0, EPS)


def test_balanced_accuracy_multi():
    obj = BalancedAccuracyMulticlass()
    assert obj.score(np.array([0, 0, 2, 0, 0, 2, 3]),
                     np.array([0, 1, 2, 0, 1, 2, 3])) == pytest.approx(0.75, EPS)

    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 3]),
                     np.array([0, 1, 2, 0, 1, 2, 3])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([1, 0, 3, 1, 2, 1, 0]),
                     np.array([0, 1, 2, 0, 1, 2, 3])) == pytest.approx(0.0, EPS)


def test_f1():
    obj = F1()
    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.5, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 1]),
                     np.array([0, 1, 0, 0, 1, 1])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([0, 0])) == pytest.approx(0.0, EPS)


def test_precision():
    obj = Precision()
    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.5, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 1]),
                     np.array([1, 1, 1, 1, 1, 1])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 1])) == pytest.approx(0.0, EPS)


def test_precision_micro_multi():
    obj = PrecisionMicro()
    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1/3.0, EPS)

    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                     np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                     np.array([2, 2, 1, 1, 0, 0, 2, 2, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)
