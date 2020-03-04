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
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    fraud_data_path = os.path.join(data_folder_path, "fraud_transactions.csv.tar.gz")

    X, y = load_data(path=fraud_data_path,
                     index="id",
                     label="fraud",
                     n_rows=n_rows)

    return X, y
