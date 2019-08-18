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

    X = pd.DataFrame(X)
    clf.fit(X, y)

    pipeline = clf.best_pipeline
    pipeline.predict(X)
    pipeline.predict_proba(X)
    pipeline.score(X, y)
