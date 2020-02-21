import numpy as np
import pandas as pd

from evalml import AutoClassificationSearch
from evalml.objectives import FraudCost


def test_fraud_objective(X_y):
    X, y = X_y

    objective = FraudCost(retry_percentage=.5,
                          interchange_fee=.02,
                          fraud_payout_percentage=.75,
                          amount_col=10)

    automl = AutoClassificationSearch(objective=objective, max_pipelines=1)
    automl.search(X, y, raise_errors=True)

    pipeline = automl.best_pipeline
    pipeline.predict(X)
    pipeline.predict_proba(X)
    pipeline.score(X, y, [objective])

    fraud_cost = FraudCost(amount_col="value")

    y_predicted = pd.Series([.1, .5, .5])
    y_true = [True, False, True]
    extra_columns = pd.DataFrame({"value": [100, 5, 25]})

    out = fraud_cost.decision_function(y_predicted, extra_columns, 5)
    assert out.tolist() == y_true
    score = fraud_cost.score(out, y_true, extra_columns)
    assert (score == 0.0)

    # testing with other types of inputs
    y_predicted = np.array([.1, .5, .5])
    extra_columns = {"value": [100, 5, 25]}
    out = fraud_cost.decision_function(y_predicted, extra_columns, 5)
    assert out.tolist() == y_true
    score = fraud_cost.score(out, y_true, extra_columns)
    assert (score == 0.0)
