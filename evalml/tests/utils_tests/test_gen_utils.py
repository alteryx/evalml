from evalml.utils import summarize_col, summarize_row, summarize_table


def test_summarize_table(X_y):
    X, y = X_y
    summarize_table(X)


def test_summarize_row(X_y):
    X, y = X_y
    summarize_row(X, 2)


def test_summarize_col(X_y):
    X, y = X_y
    summarize_col(X, 2)
