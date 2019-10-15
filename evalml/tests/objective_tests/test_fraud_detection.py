import numpy as np
import pandas as pd

from evalml import AutoClassifier
from evalml.objectives import FraudCost


def test_function(X_y):
    X, y = X_y

    objective = FraudCost(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    clf = AutoClassifier(objective=objective, max_pipelines=1)
    clf.fit(X, y)

    pipeline = clf.best_pipeline
    pipeline.predict(X)
    pipeline.predict_proba(X)
    pipeline.score(X, y)

    fraud_cost = FraudCost(amount_col="value")

    probabilities = pd.Series([.1, .5, .5])
    extra_columns = pd.DataFrame({"value": [100, 5, 25]})
    out = fraud_cost.decision_function(probabilities, extra_columns, 5)
    assert out.tolist() == [True, False, True]

    # testing with other inputs
    probabilities_array = np.array([.1, .5, .5])
    extra_columns = pd.DataFrame({"value": [100, 5, 25]})
    out = fraud_cost.decision_function(probabilities_array, extra_columns, 5)
    assert out.tolist() == [True, False, True]
