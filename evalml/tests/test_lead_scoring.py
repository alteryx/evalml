import pandas as pd
import pytest

from evalml import LeadScoring


@pytest.fixture
def y_prob():
    values = pd.DataFrame([
        [0.5, 0.5],
        [0.68, 0.32],
        [0.54, 0.46],
        [0.64, 0.36],
        [0.68, 0.32],
    ], columns=['A', 'B'])
    return values


@pytest.fixture
def y():
    values = pd.Series([0, 0, 0, 0, 0])
    return values


def test_fit(y_prob, y):
    ls = LeadScoring(label="B")
    ls.fit(y_prob, y)
    assert ls.threshold == 0.6180384491178528
