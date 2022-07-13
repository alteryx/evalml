import numpy as np
import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.objectives import LeadScoring


def test_lead_scoring_works_during_automl_search(X_y_binary):

    X, y = X_y_binary

    objective = LeadScoring(true_positives=1, false_positives=-1)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=objective,
        max_iterations=1,
        random_seed=0,
    )
    automl.search()
    pipeline = automl.best_pipeline
    pipeline.fit(X, y)
    pipeline.predict(X)
    pipeline.predict_proba(X)
    pipeline.score(X, y, [objective])


def test_lead_scoring_objective():

    objective = LeadScoring(true_positives=1, false_positives=-1)

    predicted = pd.Series([1, 10, 0.5, 5])
    out = objective.decision_function(predicted, 1)
    y_true = pd.Series([False, True, False, True])
    assert out.tolist() == [False, True, False, True]

    predicted = np.array([1, 10, 0.5, 5])
    out = objective.decision_function(predicted, 1)
    assert out.tolist() == y_true.to_list()

    score = objective.score(out, y_true)
    assert score == 0.5


def test_input_contains_nan(X_y_binary):
    objective = LeadScoring(true_positives=1, false_positives=-1)
    y_predicted = np.array([np.nan, 0, 0])
    y_true = np.array([1, 2, 1])
    with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
        objective.score(y_true, y_predicted)

    y_true = np.array([np.nan, 0, 0])
    y_predicted = np.array([1, 2, 0])
    with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
        objective.score(y_true, y_predicted)


def test_input_contains_inf(capsys):
    objective = LeadScoring(true_positives=1, false_positives=-1)
    y_predicted = np.array([np.inf, 0, 0])
    y_true = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
        objective.score(y_true, y_predicted)

    y_true = np.array([np.inf, 0, 0])
    y_predicted = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
        objective.score(y_true, y_predicted)


def test_different_input_lengths():
    objective = LeadScoring(true_positives=1, false_positives=-1)
    y_predicted = np.array([0, 0])
    y_true = np.array([1])
    with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
        objective.score(y_true, y_predicted)

    y_true = np.array([0, 0])
    y_predicted = np.array([1, 2, 0])
    with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
        objective.score(y_true, y_predicted)


def test_zero_input_lengths():
    objective = LeadScoring(true_positives=1, false_positives=-1)
    y_predicted = np.array([])
    y_true = np.array([])
    with pytest.raises(ValueError, match="Length of inputs is 0"):
        objective.score(y_true, y_predicted)


def test_binary_more_than_two_unique_values():
    objective = LeadScoring(true_positives=1, false_positives=-1)
    y_predicted = np.array([0, 1, 2])
    y_true = np.array([1, 0, 1])
    with pytest.raises(
        ValueError,
        match="y_predicted contains more than two unique values",
    ):
        objective.score(y_true, y_predicted)

    y_true = np.array([0, 1, 2])
    y_predicted = np.array([1, 0, 1])
    with pytest.raises(ValueError, match="y_true contains more than two unique values"):
        objective.score(y_true, y_predicted)


@pytest.mark.parametrize(
    "predicted,score",
    [([0, 1, 1, 1], 0.25), ([0, 0, 0, 0], 0), ([1, 1, 1, 1], 0)],
)
def test_lead_scoring_objective_penalty(predicted, score):
    objective = LeadScoring(true_positives=1, false_positives=-1)
    predicted = pd.Series(predicted)
    y_true = pd.Series([False, True, False, True])
    assert objective.objective_function(y_true, predicted) == score
