import numpy as np
import pandas as pd

from evalml import AutoClassifier
from evalml.objectives import LeadScoring


def test_function(X_y):
    X, y = X_y

    objective = LeadScoring(
        true_positives=1,
        false_positives=-1
    )

    clf = AutoClassifier(objective=objective, max_pipelines=1, random_state=0)
    clf.fit(X, y)
    pipeline = clf.best_pipeline
    pipeline.predict(X)
    pipeline.predict_proba(X)
    pipeline.score(X, y)

    predicted = pd.Series([1, 10, .5])
    out = objective.decision_function(predicted, 1)
    assert out.tolist() == [False, True, False]

    predicted = np.array([1, 10, .5])
    out = objective.decision_function(predicted, 1)
    assert out.tolist() == [False, True, False]
