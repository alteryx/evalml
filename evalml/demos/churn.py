import evalml
from evalml.preprocessing import load_data


def load_churn(n_rows=None, verbose=True):
    """Load credit card fraud dataset.
        The fraud dataset can be used for binary classification problems.

    Arguments:
        n_rows (int): Number of rows from the dataset to return
        verbose (bool): Whether to print information about features and labels

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    churn_data_path = (
        "https://api.featurelabs.com/datasets/churn.csv?library=evalml&version="
        + evalml.__version__
    )

    return load_data(
        path=churn_data_path,
        index="customerID",
        target="Churn",
        n_rows=n_rows,
        verbose=verbose,
    )
