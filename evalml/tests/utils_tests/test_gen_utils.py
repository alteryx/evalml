from evalml.utils import summarize_row, summarize_table


def test_summarize_table(X_y):
    X, y = X_y
    summarize_table(X)


def test_summarize_row(X_y):
    X, y = X_y
    summarize_row(X, 2)
