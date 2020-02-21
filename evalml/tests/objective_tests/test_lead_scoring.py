import numpy as np
import pandas as pd

from evalml import AutoClassificationSearch
from evalml.objectives import LeadScoring


def test_lead_scoring_objective(X_y):
    X, y = X_y

    objective = LeadScoring(true_positives=1,
                            false_positives=-1)

    automl = AutoClassificationSearch(objective=objective, max_pipelines=1, random_state=0)
    automl.search(X, y, raise_errors=True)
    pipeline = automl.best_pipeline
    pipeline.predict(X, objective=objective)  # TODO
    pipeline.predict_proba(X)
    pipeline.score(X, y, [objective])

    predicted = pd.Series([1, 10, .5, 5])
    out = objective.decision_function(predicted, 1)
    y_true = [False, True, False, True]
    assert out.tolist() == [False, True, False, True]

    predicted = np.array([1, 10, .5, 5])
    out = objective.decision_function(predicted, 1)
    assert out.tolist() == y_true

    score = objective.score(out, y_true)
    assert (score == 0.5)
