"""Load the credit card fraud dataset, which can be used for binary classification problems."""
import evalml
from evalml.preprocessing import load_data


def load_fraud(n_rows=None, verbose=True):
    """Load credit card fraud dataset.

    The fraud dataset can be used for binary classification problems.

    Args:
        n_rows (int): Number of rows from the dataset to return
        verbose (bool): Whether to print information about features and labels

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    fraud_data_path = (
        "https://api.featurelabs.com/datasets/fraud_transactions.csv.gz?library=evalml&version="
        + evalml.__version__
    )

    X, y = load_data(
        path=fraud_data_path,
        index="id",
        target="fraud",
        compression="gzip",
        n_rows=n_rows,
        verbose=verbose,
    )
    X.ww.set_types(logical_types={"provider": "Categorical", "region": "Categorical"})
    return X, y
