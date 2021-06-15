import os

import evalml
from evalml.preprocessing import load_data


def load_fraud(n_rows=None, verbose=True, use_local=False):
    """Load credit card fraud dataset.
        The fraud dataset can be used for binary classification problems.

    Arguments:
        n_rows (int): Number of rows from the dataset to return
        verbose (bool): Whether to print information about features and labels

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    if use_local:
        currdir_path = os.path.dirname(os.path.abspath(__file__))
        data_folder_path = os.path.join(currdir_path, "data")
        fraud_data_path = os.path.join(data_folder_path, "fraud_transactions.csv.gz")
    else:
        fraud_data_path = (
            "https://api.featurelabs.com/datasets/fraud_transactions.csv.gz?library=evalml&version="
            + evalml.__version__
        )

    return load_data(
        path=fraud_data_path,
        index="id",
        target="fraud",
        compression="gzip",
        n_rows=n_rows,
        verbose=verbose,
    )
