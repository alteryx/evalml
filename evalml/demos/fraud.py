import os

from evalml.preprocessing import load_data
import pkg_resources


def load_fraud(n_rows=None):
    """Load credit card fraud dataset.
    The fraud dataset can be used for binary classification problems.

    Args:
        n_rows (int) : number of rows from the dataset to return

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(dir_path, "data")
    fraud_path = os.path.join(dir_path, "fraud_transactions.csv.tar.gz")
    
    X, y = load_data(path=pkg_resources.resource_filename("evalml", "fraud_transactions.csv.tar.gz"),
                     index="id",
                     label="fraud",
                     n_rows=n_rows)

    return X, y
