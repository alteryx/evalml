import numpy as np
import pandas as pd

from evalml.guardrails import detect_highly_null


def test_detect_highly_null():
    df = pd.DataFrame(np.random.random((100, 5)), columns=list("ABCDE"))
    df.loc[:11, 'A'] = np.nan
    df.loc[:9, 'B'] = np.nan
    df.loc[:30, 'C'] = np.nan
    df.loc[:, 'D'] = np.nan
    df.loc[:89, 'E'] = np.nan

    expected = {'D': 1.0, 'E': 0.9}
    highly_null_set = detect_highly_null(df, percent_threshold=.90)
    assert expected == highly_null_set

    # testing np input
    nan_arr = np.full((10, 5), np.nan)
    expected = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    highly_null_set = detect_highly_null(nan_arr, percent_threshold=1.0)
    assert expected == highly_null_set
