from evalml.preprocessing import load_data


def load_fraud(n_rows=None):
    """Load credit card fraud dataset. Binary classification problem

    Args:
        n_rows (int) : number of rows to return

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    X, y = load_data(
        path="s3://featuretools-static/evalml/fraud_transactions.csv.tar.gz",
        index="id",
        label="fraud",
        n_rows=n_rows
    )

    return X, y
