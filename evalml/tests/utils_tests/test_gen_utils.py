import numpy as np
import pandas as pd

from evalml.utils import summarize_col, summarize_table


def test_summarize_table(capsys):
    X = pd.DataFrame({'ints': [1, 2, 3, 4, 6, 7],
                      'floats': [0.5, 1.0, 2.0, 3.0, 0.0, 1.0],
                      'has_nans': [0.5, 1.0, np.nan, 3.0, 0.0, np.nan],
                      'cat': ['blue', 'red', 'blue', 'red', 'green', 'yellow']})
    X['cat'] = X['cat'].astype('category')
    summarize_table(X)
    out, err = capsys.readouterr()
    assert "Summary for table:" in out
    assert "Number of rows: 6" in out
    assert "Number of columns: 4" in out


def test_summarize_col(capsys):
    X = pd.DataFrame({'ints': [1, 2, 3, 4, 6, 7],
                      'floats': [0.5, 1.0, 2.0, 3.0, 0.0, 1.0],
                      'has_nans': [0.5, 1.0, np.nan, 3.0, 0.0, np.nan],
                      'cat': ['blue', 'red', 'blue', 'red', 'green', 'yellow']})
    # todo: add NAN
    X['cat'] = X['cat'].astype('category')
    summarize_col(X, 0)
    out, err = capsys.readouterr()
    assert "Summary for column 0" in out
    assert "Datatype of col: {}".format(X['ints'].dtype) in out
    assert "Number of non-NaN elements in col 0: 6" in out
    assert "Statistics for numerical column:" in out

    # summarize_col(X, 1)
    # out, err = capsys.readouterr()
    # assert "nonono" in out

    # summarize_col(X, 2)
    # out, err = capsys.readouterr()
    # assert "nonono" in out
