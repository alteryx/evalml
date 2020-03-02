import os

from evalml.preprocessing import load_data


def load_fraud(n_rows=None):
    """Load credit card fraud dataset.
    The fraud dataset can be used for binary classification problems.

    Args:
        n_rows (int) : number of rows from the dataset to return

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    data_path = os.path.join(os.path.dirname(__file__), "data/fraud_transactions.csv.tar.gz")
    X, y = load_data(path=data_path,
                     index="id",
                     label="fraud",
                     n_rows=n_rows)

    return X, y
