import numpy as np
import pandas as pd

from evalml.preprocessing import drop_null


def test_drop_null():
    df = pd.DataFrame(np.random.random((100, 5)), columns=list("ABCDE"))
    df.loc[:11, 'A'] = np.nan
    df.loc[:10, 'B'] = np.nan
    df.loc[:30, 'C'] = np.nan
    df.loc[:, 'D'] = np.nan
    df.loc[:9, 'E'] = np.nan

    expected = df.drop(list('ABCD'), axis=1)
    nan_dropped_df = drop_null(df, percent_threshold=.9)
    assert (nan_dropped_df.equals(expected))
