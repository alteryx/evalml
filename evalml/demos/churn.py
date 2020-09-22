import os

from evalml.preprocessing import load_data


def load_churn(n_rows=None, verbose=True):
    """Load credit card fraud dataset.
        The fraud dataset can be used for binary classification problems.

    Arguments:
        n_rows (int): Number of rows from the dataset to return
        verbose (bool): Whether to print information about features and labels

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    churn_data_path = os.path.join(data_folder_path, "churn.csv")

    X, y = load_data(path=churn_data_path,
                     index="customerID",
                     target="Churn",
                     n_rows=n_rows,
                     verbose=verbose)

    return X, y
