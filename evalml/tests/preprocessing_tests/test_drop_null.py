import numpy as np
import pandas as pd

from evalml.preprocessing import detect_highly_null


def test_detect_highly_null():
    df = pd.DataFrame(np.random.random((100, 5)), columns=list("ABCDE"))
    df.loc[:11, 'A'] = np.nan
    df.loc[:10, 'B'] = np.nan
    df.loc[:30, 'C'] = np.nan
    df.loc[:, 'D'] = np.nan
    df.loc[:9, 'E'] = np.nan

    expected = {'A', 'B', 'C', 'D'}
    nan_dropped_df = detect_highly_null(df, percent_threshold=.9)
    assert expected == nan_dropped_df
