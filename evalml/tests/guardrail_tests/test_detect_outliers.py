import numpy as np
import pandas as pd

from evalml.guardrails import detect_outliers


def test_outliers():
    data = np.random.randn(100, 100)
    X = pd.DataFrame(data=data)
    X.iloc[3, :] = pd.Series(np.random.randn(100) * 10)
    X.iloc[25, :] = pd.Series(np.random.randn(100) * 20)
    X.iloc[55, :] = pd.Series(np.random.randn(100) * 100)
    X.iloc[72, :] = pd.Series(np.random.randn(100) * 100)

    expected = {3, 55, 25, 72}
    result = detect_outliers(X)
    assert expected.issubset(set(result))
