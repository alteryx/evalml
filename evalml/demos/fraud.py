from evalml.preprocessing import load_data


def load_fraud():
    X, y = load_data(
        path="s3://featuretools-static/evalml/fraud_transactions.csv.tar.gz",
        index="id",
        label="fraud"
    )

    return X, y
