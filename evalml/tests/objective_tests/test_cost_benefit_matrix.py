import numpy as np
import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.objectives import CostBenefitMatrix


@pytest.mark.parametrize("optimize_thresholds", [True, False])
def test_cost_benefit_matrix_objective(optimize_thresholds, X_y_binary):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    automl = AutoMLSearch(problem_type='binary', objective=cbm, max_pipelines=2, optimize_thresholds=optimize_thresholds)
    automl.search(X, y)

    pipeline = automl.best_pipeline
    pipeline.fit(X, y)
    assert not np.isnan(pipeline.predict(X, cbm)).values.any()
    assert not np.isnan(pipeline.predict_proba(X)).values.any()
    assert not np.isnan(pipeline.score(X, y, [cbm])['Cost Benefit Matrix'])
    assert not np.isnan(list(pipeline.predict(X, cbm))).any()


def test_cbm_objective_function():
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_predicted = pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    assert cbm.objective_function(y_true, y_predicted) == (3 * 10) + (-1 * 2) + (1 * -7) + (4 * -2)


def test_cmb_input_contains_nan(X_y_binary):
    y_predicted = pd.Series([np.nan, 0, 0])
    y_true = pd.Series([1, 2, 1])
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([np.nan, 0, 0])
    y_predicted = pd.Series([1, 2, 0])
    with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
        cbm.score(y_true, y_predicted)


def test_input_contains_inf(capsys):
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    y_predicted = np.array([np.inf, 0, 0])
    y_true = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([np.inf, 0, 0])
    y_predicted = pd.Series([1, 0, 0])
    with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
        cbm.score(y_true, y_predicted)


def test_different_input_lengths():
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    y_predicted = pd.Series([0, 0])
    y_true = pd.Series([1])
    with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([0, 0])
    y_predicted = pd.Series([1, 2, 0])
    with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
        cbm.score(y_true, y_predicted)


def test_zero_input_lengths():
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    y_predicted = pd.Series([])
    y_true = pd.Series([])
    with pytest.raises(ValueError, match="Length of inputs is 0"):
        cbm.score(y_true, y_predicted)


def test_binary_more_than_two_unique_values():
    cbm = CostBenefitMatrix(true_positive=10, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    y_predicted = pd.Series([0, 1, 2])
    y_true = pd.Series([1, 0, 1])
    with pytest.raises(ValueError, match="y_predicted contains more than two unique values"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([0, 1, 2])
    y_predicted = pd.Series([1, 0, 1])
    with pytest.raises(ValueError, match="y_true contains more than two unique values"):
        cbm.score(y_true, y_predicted)
