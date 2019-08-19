from evalml.preprocessing import load_data


def load_fraud():
    """Load credit card fraud dataset. Binary classification problem"""
    X, y = load_data(
        path="s3://featuretools-static/evalml/fraud_transactions.csv.tar.gz",
        index="id",
        label="fraud"
    )

    return X, y
