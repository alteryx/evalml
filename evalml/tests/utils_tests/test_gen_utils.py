import pandas as pd
from evalml.utils import summarize_col, summarize_table


def test_summarize_table(X_y):
    X = pd.DataFrame({'ints': [1, 2, 3, 4, 6, 7],
                      'floats': [0.5, 1.0, 2.0, 3.0, 0.0, 1.0],
                      'cat': ['blue', 'red', 'blue', 'red', 'green', 'yellow']})
    X['cat'] = X['cat'].astype('category')
    summarize_table(X)


def test_summarize_col(X_y):
    X = pd.DataFrame({'ints': [1, 2, 3, 4, 6, 7],
                      'floats': [0.5, 1.0, 2.0, 3.0, 0.0, 1.0],
                      'cat': ['blue', 'red', 'blue', 'red', 'green', 'yellow']})
    #todo: add NAN
    X['cat'] = X['cat'].astype('category')
    summarize_col(X, 0)
    summarize_col(X, 1)
    summarize_col(X, 2)

