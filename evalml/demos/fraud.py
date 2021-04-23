import os

from evalml.preprocessing import load_data


def load_fraud(n_rows=None, verbose=True, return_pandas=False):
    """Load credit card fraud dataset.
        The fraud dataset can be used for binary classification problems.

    Arguments:
        n_rows (int): Number of rows from the dataset to return
        verbose (bool): Whether to print information about features and labels

    Returns:
        Union[(ww.DataTable, ww.DataColumn), (pd.Dataframe, pd.Series)]: X and y
    """
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    fraud_data_path = os.path.join(data_folder_path, "fraud_transactions.csv.gz")

    X, y = load_data(path=fraud_data_path,
                     index="id",
                     target="fraud",
                     n_rows=n_rows,
                     verbose=verbose)
    X = X.set_types({"provider": "Categorical", "region": "Categorical"})

    if return_pandas:
        return X.to_dataframe(), y.to_series()
    return X, y
