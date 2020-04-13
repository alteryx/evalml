from evalml.objectives import Accuracy
import numpy as np
import pytest
from evalml.exceptions import DimensionMismatchError
def test_accuracy():
    res = None

    acc = Accuracy()
    res = acc.score(y_predicted=[1, 0], y_true=[1, 0])
    assert res == 1.0

    with pytest.raises(DimensionMismatchError):
        acc.score(y_predicted=[0], y_true=[1, 0])

    with pytest.raises(ValueError, match="Length of inputs is 0"):
        acc.score(y_predicted=[], y_true=[1])

    with pytest.raises(ValueError, match="Length of inputs is 0"):
        acc.score(y_predicted=[], y_true=[0])